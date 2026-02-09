"""Transaction-related tools for LLM function calling"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from tools.base import BaseTool

logger = logging.getLogger(__name__)


# ============================================================================
# Parameter Schemas
# ============================================================================


class SaveTransactionParams(BaseModel):
    """Parameters for saving a single transaction"""

    amount: float = Field(..., description="Transaction amount (positive for income, negative for expense)")
    category: str = Field(..., description="Transaction category (e.g., 'Food', 'Transport', 'Salary')")
    description: str = Field(..., description="Description of the transaction")
    transaction_type: str = Field(..., description="Type of transaction - either 'income' or 'expense'")
    date: Optional[str] = Field(None, description="Optional transaction date in ISO format (defaults to now)")


class GetUserTransactionsParams(BaseModel):
    """Parameters for retrieving user transactions"""

    limit: int = Field(default=20, description="Maximum number of transactions to return (default: 20)")
    transaction_type: Optional[str] = Field(None, description="Filter by 'income' or 'expense'")
    start_date: Optional[str] = Field(None, description="Filter transactions after this date (ISO format)")
    end_date: Optional[str] = Field(None, description="Filter transactions before this date (ISO format)")
    category: Optional[str] = Field(None, description="Filter by category name")


class DeleteTransactionParams(BaseModel):
    """Parameters for deleting a specific transaction"""

    transaction_id: str = Field(..., description="UUID of the transaction to delete")


class DeleteLastTransactionParams(BaseModel):
    """Parameters for deleting the most recent transaction (no arguments needed)"""

    pass


class DeleteLastStatementTransactionsParams(BaseModel):
    """Parameters for deleting all transactions from the last bank statement (no arguments needed)"""

    pass


# ============================================================================
# Tool Implementations
# ============================================================================


class SaveTransactionTool(BaseTool):
    """Save a new transaction and return updated user balance"""

    name = "save_transaction"
    description = (
        "Save a new transaction and return updated user balance with spending limits. "
        "Returns UserBalance with balance, monthly_spending, monthly_limit, daily_spending, daily_limit."
    )
    args_schema = SaveTransactionParams

    async def execute(self, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        date = None
        if arguments.get("date"):
            date = datetime.fromisoformat(arguments["date"])

        transaction, balance = await self.api_client.save_transaction(
            user_id=user_id,
            amount=arguments["amount"],
            category=arguments["category"],
            description=arguments["description"],
            transaction_type=arguments["transaction_type"],
            date=date,
        )
        result: Dict[str, Any] = {"transaction": transaction, "success": True}
        if balance:
            result["balance"] = balance.model_dump()
        return result


class GetUserTransactionsTool(BaseTool):
    """Get user transactions with optional filters"""

    name = "get_user_transactions"
    description = (
        "Get user's recent transactions with optional filters. "
        "Returns list of transactions with id, amount, category, description, date, etc."
    )
    args_schema = GetUserTransactionsParams

    async def execute(self, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        start_date = None
        end_date = None
        if arguments.get("start_date"):
            start_date = datetime.fromisoformat(arguments["start_date"])
        if arguments.get("end_date"):
            end_date = datetime.fromisoformat(arguments["end_date"])

        transactions = await self.api_client.get_user_transactions(
            user_id=user_id,
            limit=arguments.get("limit", 20),
            transaction_type=arguments.get("transaction_type"),
            start_date=start_date,
            end_date=end_date,
            category=arguments.get("category"),
        )
        return {"transactions": [t.model_dump(mode="json") for t in transactions]}


class DeleteTransactionTool(BaseTool):
    """Delete a specific transaction by its ID"""

    name = "delete_transaction"
    description = (
        "Delete a specific transaction by its ID. "
        "IMPORTANT: This should only be called after the user has confirmed which "
        "transaction to delete via HITL flow. Do not call this directly based on "
        "user description - use get_user_transactions first to find matches."
    )
    args_schema = DeleteTransactionParams

    async def execute(self, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        transaction_id = UUID(arguments["transaction_id"])
        balance = await self.api_client.delete_transaction(user_id, transaction_id)
        result: Dict[str, Any] = {"success": True, "message": "Transaction deleted successfully"}
        if balance:
            result["balance"] = balance.model_dump()
        return result


class DeleteLastTransactionTool(BaseTool):
    """Delete the most recently created transaction"""

    name = "delete_last_transaction"
    description = (
        "Delete the most recently created transaction for the user. "
        "Use this when user says something like 'delete my last transaction' "
        "or 'undo my last entry'."
    )
    args_schema = DeleteLastTransactionParams

    async def execute(self, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        await self.api_client.delete_last_transaction(user_id)
        return {"success": True, "message": "Last transaction deleted successfully"}


class DeleteLastStatementTransactionsTool(BaseTool):
    """Delete all transactions from the last bank statement upload"""

    name = "delete_last_statement_transactions"
    description = (
        "Delete all transactions created from the last bank statement upload. "
        "Use this when user wants to undo a bank statement import."
    )
    args_schema = DeleteLastStatementTransactionsParams

    async def execute(self, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        deleted_count = await self.api_client.delete_last_statement_transactions(user_id)
        return {"success": True, "deleted_count": deleted_count}
