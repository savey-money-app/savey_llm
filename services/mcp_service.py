"""
MCP (Model Context Protocol) Service

Service for executing MCP function calls to the savey_api backend.
All 7 MCP functions are implemented here with proper error handling and retries.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from core.config import settings
from schemas.mcp_tools import (
    TransactionCreateShort,
    TransactionRead,
    UserBalance,
)

logger = logging.getLogger(__name__)


class MCPService:
    """Service for executing MCP function calls to savey_api"""

    def __init__(self):
        self.base_url = settings.SAVEY_API_URL
        self.timeout = settings.MCP_TIMEOUT
        self.max_retries = settings.MCP_MAX_RETRIES

    async def _make_request(
        self, method: str, endpoint: str, user_id: UUID, **kwargs
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to savey_api with retry logic

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            user_id: User ID for authentication
            **kwargs: Additional arguments for httpx request

        Returns:
            Response JSON data

        Raises:
            Exception: If request fails after max retries
        """
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers["X-User-ID"] = str(user_id)  # Simple user identification header

        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.request(method, url, headers=headers, **kwargs)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    f"MCP request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    break
                # Simple exponential backoff
                import asyncio

                await asyncio.sleep(2**attempt)

        error_msg = f"MCP request to {endpoint} failed after {self.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    # ============================================================================
    # Transaction Management Functions
    # ============================================================================

    async def save_transaction(
        self,
        user_id: UUID,
        amount: float,
        category: str,
        description: str,
        transaction_type: str,
        date: Optional[datetime] = None,
    ) -> UserBalance:
        """
        Save a new transaction and return updated user balance

        Args:
            user_id: User ID
            amount: Transaction amount
            category: Transaction category
            description: Transaction description
            transaction_type: 'income' or 'expense'
            date: Transaction date (defaults to now)

        Returns:
            UserBalance with monthly/daily spending and limits
        """
        logger.info(f"💰 Saving transaction for user {user_id}: {description} ({amount})")

        payload = {
            "amount": amount,
            "category": category,
            "description": description,
            "transaction_type": transaction_type,
            "date": (date or datetime.utcnow()).isoformat(),
        }

        result = await self._make_request("POST", "/api/v1/transactions", user_id, json=payload)

        return UserBalance(**result["balance"])

    async def delete_transaction(self, user_id: UUID, transaction_id: UUID) -> None:
        """
        Delete a specific transaction by ID

        Args:
            user_id: User ID
            transaction_id: Transaction ID to delete
        """
        logger.info(f"🗑️ Deleting transaction {transaction_id} for user {user_id}")

        await self._make_request("DELETE", f"/api/v1/transactions/{transaction_id}", user_id)

        logger.info(f"✅ Transaction {transaction_id} deleted successfully")

    async def get_user_transactions(
        self,
        user_id: UUID,
        limit: int = 20,
        transaction_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        category: Optional[str] = None,
    ) -> List[TransactionRead]:
        """
        Get user transactions with optional filters

        Args:
            user_id: User ID
            limit: Maximum number of transactions to return (default: 20)
            transaction_type: Filter by 'income' or 'expense'
            start_date: Filter transactions after this date
            end_date: Filter transactions before this date
            category: Filter by category

        Returns:
            List of transactions
        """
        logger.info(f"📋 Fetching transactions for user {user_id} (limit: {limit})")

        # Build query parameters
        params = {"limit": limit}
        if transaction_type:
            params["transaction_type"] = transaction_type
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if category:
            params["category"] = category

        result = await self._make_request("GET", "/api/v1/transactions", user_id, params=params)

        transactions = [TransactionRead(**t) for t in result["transactions"]]
        logger.info(f"✅ Found {len(transactions)} transactions")
        return transactions

    async def delete_last_transaction(self, user_id: UUID) -> None:
        """
        Delete the most recently created transaction for the user

        Args:
            user_id: User ID
        """
        logger.info(f"🗑️ Deleting last transaction for user {user_id}")

        await self._make_request("DELETE", "/api/v1/transactions/last", user_id)

        logger.info("✅ Last transaction deleted successfully")

    async def delete_last_statement_transactions(self, user_id: UUID) -> int:
        """
        Delete all transactions from the last bank statement upload

        Args:
            user_id: User ID

        Returns:
            Number of transactions deleted
        """
        logger.info(f"🗑️ Deleting transactions from last bank statement for user {user_id}")

        result = await self._make_request("DELETE", "/api/v1/transactions/last-statement", user_id)

        deleted_count = result.get("deleted_count", 0)
        logger.info(f"✅ Deleted {deleted_count} transactions from last statement")
        return deleted_count

    # ============================================================================
    # Bank Statement Functions
    # ============================================================================

    async def create_transactions_from_statement(
        self,
        user_id: UUID,
        transactions: List[TransactionCreateShort],
        statement_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create multiple transactions from a parsed bank statement (bulk operation)

        Args:
            user_id: User ID
            transactions: List of transactions to create
            statement_date: Date of the bank statement

        Returns:
            Dict with created transaction count and bank statement ID
        """
        logger.info(
            f"📄 Creating {len(transactions)} transactions from bank statement for user {user_id}"
        )

        payload = {
            "transactions": [t.model_dump() for t in transactions],
            "statement_date": (statement_date or datetime.utcnow()).isoformat(),
        }

        result = await self._make_request(
            "POST", "/api/v1/transactions/bulk", user_id, json=payload
        )

        logger.info(
            f"✅ Created {result['created_count']} transactions from statement {result['statement_id']}"
        )
        return result

    # ============================================================================
    # Utility Functions
    # ============================================================================

    async def mcc_lookup(self, mcc_code: str) -> Dict[str, Any]:
        """
        Lookup information about a Merchant Category Code (MCC)

        Args:
            mcc_code: 4-digit MCC code

        Returns:
            Dict with MCC information (code, description, category)
        """
        logger.info(f"🔍 Looking up MCC code: {mcc_code}")

        result = await self._make_request(
            "GET", f"/api/v1/mcc/{mcc_code}", UUID("00000000-0000-0000-0000-000000000000")
        )

        logger.info(f"✅ MCC {mcc_code}: {result.get('description', 'Unknown')}")
        return result

    async def get_categories(self, user_id: UUID) -> List[str]:
        """
        Get all transaction categories for the user

        Args:
            user_id: User ID

        Returns:
            List of category names
        """
        logger.info(f"📂 Fetching categories for user {user_id}")

        result = await self._make_request("GET", "/api/v1/categories", user_id)

        categories = result.get("categories", [])
        logger.info(f"✅ Found {len(categories)} categories")
        return categories

    async def get_balance(self, user_id: UUID) -> UserBalance:
        """
        Get user balance summary with spending limits

        Args:
            user_id: User ID

        Returns:
            UserBalance with balance and spending data
        """
        logger.info(f"💳 Fetching balance for user {user_id}")

        result = await self._make_request("GET", "/api/v1/balance", user_id)

        return UserBalance(**result)
