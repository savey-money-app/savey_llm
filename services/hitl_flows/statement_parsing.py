"""
Bank Statement Parsing HITL Flow

Flow:
1. User uploads bank statement (PDF or image)
2. Agent parses it and extracts transactions
3. Present list of transactions to user for confirmation
4. User can confirm, modify, or cancel
5. If modifications requested, update and present again (loop until confirmed)
6. Once confirmed, create all transactions via bulk API
"""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from schemas.bank_statement import ParsedStatement
from schemas.hitl import (
    HITLFlowType,
    StatementParsingConfirmation,
    StatementParsingFlowData,
    StatementParsingPresentationList,
)
from schemas.api_tools import TransactionCreateShort
from services.hitl_manager import HITLManager
from services.api_client import APIClient

logger = logging.getLogger(__name__)


class StatementParsingFlow:
    """Handles bank statement parsing confirmation flow"""

    def __init__(self, hitl_manager: HITLManager, api_client: APIClient):
        self.hitl_manager = hitl_manager
        self.api_client = api_client

    def _format_transaction_list(self, transactions: List[TransactionCreateShort]) -> str:
        """
        Format transactions for user presentation

        Args:
            transactions: List of parsed transactions

        Returns:
            Formatted string
        """
        lines = []
        total_income = 0.0
        total_expenses = 0.0

        for i, t in enumerate(transactions, 1):
            amount_str = f"${abs(t.amount):.2f}"
            if t.amount > 0:
                amount_str = f"+{amount_str} (income)"
                total_income += t.amount
            else:
                amount_str = f"-{amount_str} (expense)"
                total_expenses += abs(t.amount)

            date_str = t.date.strftime("%Y-%m-%d")
            lines.append(f"{i}. {date_str} | {amount_str} | {t.category} | {t.description}")

        return "\n".join(lines), total_income, total_expenses

    async def initiate_parsing_flow(
        self, user_id: UUID, message_id: str, parsed_statement: ParsedStatement
    ) -> StatementParsingPresentationList:
        """
        Initiate statement parsing HITL flow

        Args:
            user_id: User ID
            message_id: Original message ID
            parsed_statement: Parsed statement data

        Returns:
            StatementParsingPresentationList for user confirmation
        """
        logger.info(
            f"📄 Initiating statement parsing flow for user {user_id} - {len(parsed_statement.transactions)} transactions"
        )

        if not parsed_statement.transactions:
            return StatementParsingPresentationList(
                transactions=[],
                total_income=0.0,
                total_expenses=0.0,
                net_change=0.0,
                transaction_count=0,
                message="❌ No transactions found in the statement. Please check the file and try again.",
                flow_id="",
            )

        # Create HITL flow
        flow_data = StatementParsingFlowData(
            transactions=parsed_statement.transactions,
            statement_date=parsed_statement.statement_date,
            iteration=1,
            user_remarks=None,
        )

        flow_request = await self.hitl_manager.create_flow(
            user_id=user_id,
            message_id=message_id,
            flow_type=HITLFlowType.STATEMENT_PARSING,
            data=flow_data.model_dump(mode="json"),
        )

        # Format transaction list
        formatted_list, total_income, total_expenses = self._format_transaction_list(
            parsed_statement.transactions
        )

        net_change = total_income - total_expenses

        message = f"""📄 **Bank Statement Parsed**

I found **{len(parsed_statement.transactions)} transactions** in your statement:

```
{formatted_list}
```

**Summary:**
- Total Income: +${total_income:.2f}
- Total Expenses: -${total_expenses:.2f}
- Net Change: ${net_change:+.2f}

**Please review the transactions above.**

- Reply **'confirm'** to create these transactions
- Reply **'cancel'** to abort
- Provide **modifications or remarks** if you need to change anything

Flow ID: `{flow_request.flow_id}`"""

        return StatementParsingPresentationList(
            transactions=parsed_statement.transactions,
            total_income=total_income,
            total_expenses=total_expenses,
            net_change=net_change,
            transaction_count=len(parsed_statement.transactions),
            message=message,
            flow_id=flow_request.flow_id,
        )

    async def handle_modification_iteration(
        self,
        flow_id: str,
        user_id: UUID,
        modified_transactions: List[TransactionCreateShort],
        remarks: Optional[str] = None,
    ) -> StatementParsingPresentationList:
        """
        Handle modification iteration in the HITL loop

        Args:
            flow_id: HITL flow ID
            user_id: User ID
            modified_transactions: Updated transaction list
            remarks: User's remarks

        Returns:
            Updated StatementParsingPresentationList
        """
        flow = await self.hitl_manager.get_flow(flow_id)
        if not flow:
            return StatementParsingPresentationList(
                transactions=[],
                total_income=0.0,
                total_expenses=0.0,
                net_change=0.0,
                transaction_count=0,
                message="❌ Flow not found or expired. Please start over.",
                flow_id="",
            )

        iteration = int(flow.get("iteration", 1))

        # Update flow data
        flow_data = StatementParsingFlowData(
            transactions=modified_transactions,
            statement_date=flow["data"].get("statement_date"),
            iteration=iteration,
            user_remarks=remarks,
        )

        from schemas.hitl import HITLFlowState
        await self.hitl_manager.update_flow_state(
            flow_id, HITLFlowState.IN_PROGRESS, data=flow_data.model_dump(mode="json")
        )

        # Format updated transaction list
        formatted_list, total_income, total_expenses = self._format_transaction_list(
            modified_transactions
        )

        net_change = total_income - total_expenses

        message = f"""📄 **Updated Transaction List** (Iteration {iteration})

I've updated the transactions based on your feedback:

```
{formatted_list}
```

**Summary:**
- Total Income: +${total_income:.2f}
- Total Expenses: -${total_expenses:.2f}
- Net Change: ${net_change:+.2f}

**Please review the updated list.**

- Reply **'confirm'** to create these transactions
- Reply **'cancel'** to abort
- Provide **more modifications** if needed

Flow ID: `{flow_id}`"""

        return StatementParsingPresentationList(
            transactions=modified_transactions,
            total_income=total_income,
            total_expenses=total_expenses,
            net_change=net_change,
            transaction_count=len(modified_transactions),
            message=message,
            flow_id=flow_id,
        )

    async def execute_bulk_creation(
        self, flow_id: str, user_id: UUID
    ) -> str:
        """
        Execute bulk transaction creation after user confirmation

        Args:
            flow_id: HITL flow ID
            user_id: User ID

        Returns:
            Success message
        """
        # Get flow data
        flow = await self.hitl_manager.get_flow(flow_id)
        if not flow:
            return "❌ Flow not found or expired. Please start over."

        flow_data = StatementParsingFlowData(**flow["data"])

        if not flow_data.transactions:
            return "❌ No transactions to create."

        # Create transactions via bulk API
        try:
            result, balance = await self.api_client.create_transactions_from_statement(
                user_id=user_id,
                transactions=flow_data.transactions,
                statement_date=flow_data.statement_date,
            )

            created_count = result.get("created_count", 0)
            statement_id = result.get("statement_id")

            message = f"""✅ **Bank Statement Processed Successfully!**

Created **{created_count} transactions** from your statement.

Statement ID: `{statement_id}`"""

            if balance:
                message += f"""

**Updated Balance:**
- Balance: ${balance.balance:.2f}
- Monthly spending: ${balance.monthly_spending:.2f}
- Daily spending: ${balance.daily_spending:.2f}"""
            else:
                message += "\n\nYour balance and transaction history have been updated."

            # Clean up flow
            await self.hitl_manager.delete_flow(flow_id)

            return message

        except Exception as e:
            logger.error(f"❌ Failed to create transactions: {e}")
            return f"❌ Failed to create transactions: {e}"
