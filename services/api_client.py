"""
API Client for Savey API

Simple HTTP client for calling savey_api backend endpoints.
Used by agents to execute transaction and financial operations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from core.config import settings
from schemas.api_tools import (
    TransactionCreateShort,
    TransactionRead,
    UserBalance,
)

logger = logging.getLogger(__name__)


class APIClient:
    """Client for calling savey_api backend"""

    def __init__(self):
        self.base_url = settings.SAVEY_API_URL
        self.timeout = 30

    async def _make_request(
        self, method: str, endpoint: str, user_id: UUID, **kwargs
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to savey_api

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            user_id: User ID for authentication
            **kwargs: Additional arguments for httpx request

        Returns:
            Response JSON data

        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers["X-User-ID"] = str(user_id)
        headers["X-Internal-Token"] = settings.INTERNAL_API_TOKEN

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                if response.status_code == 204 or not response.content:
                    return {}
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"API request failed to {endpoint}: {e}")
            raise Exception(f"API request to {endpoint} failed: {e}")

    # ============================================================================
    # Transaction Management
    # ============================================================================

    async def _resolve_category_id(self, user_id: UUID, category_name: str) -> str:
        """Resolve a category name to its UUID, creating the category if it doesn't exist."""
        categories = await self._make_request("GET", "/api/v1/categories", user_id)
        for cat in categories:
            if cat.get("title", "").lower() == category_name.lower():
                return cat["id"]

        # Category not found — create it
        try:
            new_cat = await self._make_request(
                "POST", "/api/v1/categories", user_id, json={"title": category_name}
            )
            return new_cat["id"]
        except Exception:
            # May have been created concurrently — try fetching again
            categories = await self._make_request("GET", "/api/v1/categories", user_id)
            for cat in categories:
                if cat.get("title", "").lower() == category_name.lower():
                    return cat["id"]
            raise

    async def save_transaction(
        self,
        user_id: UUID,
        amount: float,
        category: str,
        description: str,
        transaction_type: str,
        date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Save a new transaction and return the created transaction"""
        logger.info(f"💰 Saving transaction for user {user_id}")

        category_id = await self._resolve_category_id(user_id, category)

        payload = {
            "amount": abs(amount),  # API expects positive amount
            "category_id": category_id,
            "description": description,
            "transaction_type": transaction_type,
            "date": (date or datetime.utcnow()).date().isoformat(),
        }

        result = await self._make_request("POST", "/api/v1/transactions", user_id, json=payload)
        return result

    async def delete_transaction(self, user_id: UUID, transaction_id: UUID) -> None:
        """Delete a specific transaction by ID"""
        logger.info(f"🗑️ Deleting transaction {transaction_id}")
        await self._make_request("DELETE", f"/api/v1/transactions/{transaction_id}", user_id)

    async def get_user_transactions(
        self,
        user_id: UUID,
        limit: int = 20,
        transaction_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        category: Optional[str] = None,
    ) -> List[TransactionRead]:
        """Get user transactions with optional filters"""
        logger.info(f"📋 Fetching transactions for user {user_id}")

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
        if isinstance(result, list):
            transactions = result
        elif isinstance(result, dict):
            transactions = result.get("transactions") or result.get("items") or result.get("data") or []
        else:
            raise TypeError(f"Unexpected transactions response type: {type(result)}")

        return [TransactionRead(**t) for t in transactions]

    async def delete_last_transaction(self, user_id: UUID) -> None:
        """Delete the most recently created transaction"""
        logger.info(f"🗑️ Deleting last transaction for user {user_id}")
        await self._make_request("DELETE", "/api/v1/transactions/last", user_id)

    async def delete_last_statement_transactions(self, user_id: UUID) -> int:
        """Delete all transactions from the last bank statement upload"""
        logger.info(f"🗑️ Deleting last statement transactions for user {user_id}")
        result = await self._make_request("DELETE", "/api/v1/transactions/last-statement", user_id)
        return result.get("deleted_count", 0)

    # ============================================================================
    # Bank Statement
    # ============================================================================

    async def create_transactions_from_statement(
        self,
        user_id: UUID,
        transactions: List[TransactionCreateShort],
        statement_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Create multiple transactions from a parsed bank statement (bulk operation)"""
        logger.info(f"📄 Creating {len(transactions)} transactions from statement")

        payload = {
            "transactions": [t.model_dump(mode="json") for t in transactions],
            "statement_date": (statement_date or datetime.utcnow()).isoformat(),
        }

        result = await self._make_request("POST", "/api/v1/transactions/bulk", user_id, json=payload)
        return result

    # ============================================================================
    # Utility
    # ============================================================================

    async def get_categories(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get all transaction categories as [{id, title, ...}]"""
        logger.info(f"📂 Fetching categories for user {user_id}")
        result = await self._make_request("GET", "/api/v1/categories", user_id)
        if isinstance(result, list):
            return result
        return result.get("categories", [])

    async def get_balance(self, user_id: UUID) -> UserBalance:
        """Get user balance summary with spending limits"""
        logger.info(f"💳 Fetching balance for user {user_id}")
        result = await self._make_request("GET", "/api/v1/balance", user_id)
        return UserBalance(**result)
