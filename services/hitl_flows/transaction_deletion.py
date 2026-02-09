"""
Transaction Deletion HITL Flow

Flow:
1. User requests to delete transaction with some description/details
2. LLM calls get_user_transactions to search for matching transactions
3. Present matches to user asking them to specify which one to delete
4. User selects a transaction
5. Delete the selected transaction
"""

import logging
from typing import List
from uuid import UUID

from schemas.hitl import (
    HITLFlowType,
    TransactionDeletionFlowData,
    TransactionDeletionResponse,
)
from schemas.api_tools import TransactionRead
from services.hitl_manager import HITLManager
from services.api_client import APIClient

logger = logging.getLogger(__name__)


class TransactionDeletionFlow:
    """Handles transaction deletion confirmation flow"""

    def __init__(self, hitl_manager: HITLManager, api_client: APIClient):
        self.hitl_manager = hitl_manager
        self.api_client = api_client

    async def initiate_deletion_flow(
        self,
        user_id: UUID,
        message_id: str,
        search_query: str,
        matched_transactions: List[TransactionRead],
        user_currency: str = "USD",
    ) -> TransactionDeletionResponse:
        """
        Initiate transaction deletion HITL flow

        Args:
            user_id: User ID
            message_id: Original message ID
            search_query: User's search query
            matched_transactions: List of matching transactions

        Returns:
            TransactionDeletionResponse asking user to select transaction
        """
        logger.info(
            f"🔍 Initiating transaction deletion flow for user {user_id} - found {len(matched_transactions)} matches"
        )

        if not matched_transactions:
            return TransactionDeletionResponse(
                matches_found=0,
                transactions=[],
                message="❌ No transactions found matching your description. Please try again with different details.",
                flow_id="",
            )

        if len(matched_transactions) == 1:
            # Only one match - can delete directly, but still ask for confirmation
            transaction = matched_transactions[0]
            flow_data = TransactionDeletionFlowData(
                search_query=search_query,
                matched_transactions=matched_transactions,
                selected_transaction_id=transaction.id,
                user_currency=user_currency,
            )

            flow_request = await self.hitl_manager.create_flow(
                user_id=user_id,
                message_id=message_id,
                flow_type=HITLFlowType.TRANSACTION_DELETION,
                data=flow_data.model_dump(mode="json"),
            )

            message = f"""✅ Found 1 matching transaction:

**Transaction Details:**
- Amount: {abs(transaction.amount):.2f} {user_currency} ({transaction.transaction_type})
- Category: {transaction.category.title}
- Description: {transaction.description}
- Date: {transaction.date.strftime('%Y-%m-%d')}

Do you want to delete this transaction? Reply with 'confirm' to delete or 'cancel' to abort.

Flow ID: `{flow_request.flow_id}`"""

            return TransactionDeletionResponse(
                matches_found=1,
                transactions=matched_transactions,
                message=message,
                flow_id=flow_request.flow_id,
            )

        # Multiple matches - user needs to select one
        flow_data = TransactionDeletionFlowData(
            search_query=search_query,
            matched_transactions=matched_transactions,
            selected_transaction_id=None,
            user_currency=user_currency,
        )

        flow_request = await self.hitl_manager.create_flow(
            user_id=user_id,
            message_id=message_id,
            flow_type=HITLFlowType.TRANSACTION_DELETION,
            data=flow_data.model_dump(mode="json"),
        )

        # Format transaction list for user
        transaction_list = []
        for i, t in enumerate(matched_transactions[:20], 1):  # Limit to 20
            transaction_list.append(
                f"{i}. **{abs(t.amount):.2f} {user_currency}** - {t.category.title} - {t.description} ({t.date.strftime('%Y-%m-%d')})"
            )

        message = f"""🔍 Found {len(matched_transactions)} matching transactions:

{chr(10).join(transaction_list)}

Please specify which transaction you want to delete by number (1-{min(len(matched_transactions), 20)}), or reply 'cancel' to abort.

Flow ID: `{flow_request.flow_id}`"""

        return TransactionDeletionResponse(
            matches_found=len(matched_transactions),
            transactions=matched_transactions,
            message=message,
            flow_id=flow_request.flow_id,
        )

    async def execute_deletion(self, flow_id: str, user_id: UUID) -> str:
        """
        Execute transaction deletion after user confirmation

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

        flow_data = TransactionDeletionFlowData(**flow["data"])
        currency = flow_data.user_currency

        if not flow_data.selected_transaction_id:
            return "❌ No transaction selected. Please start over."

        # Delete the transaction
        try:
            balance = await self.api_client.delete_transaction(user_id, flow_data.selected_transaction_id)

            # Find the deleted transaction to show details
            deleted = next(
                (t for t in flow_data.matched_transactions if t.id == flow_data.selected_transaction_id),
                None,
            )

            if deleted:
                message = f"""✅ Transaction deleted successfully!

**Deleted:**
- Amount: {abs(deleted.amount):.2f} {currency} ({deleted.transaction_type})
- Category: {deleted.category.title}
- Description: {deleted.description}
- Date: {deleted.date.strftime('%Y-%m-%d')}"""
            else:
                message = "✅ Transaction deleted successfully!"

            if balance:
                message += f"\n\n**Updated Balance:** {balance.balance:.2f} {currency}"

            # Clean up flow
            await self.hitl_manager.delete_flow(flow_id)

            return message

        except Exception as e:
            logger.error(f"❌ Failed to delete transaction: {e}")
            return f"❌ Failed to delete transaction: {e}"
