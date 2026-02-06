"""
API Tool Schemas

These schemas define the parameters for all API functions that the LLM can call.
Functions are hosted on the main backend server (savey_api).
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Transaction Management Tools
# ============================================================================


class TransactionCreateShort(BaseModel):
    """Short transaction creation schema for batch operations"""

    amount: float = Field(..., description="Transaction amount (positive for income, negative for expense)")
    category: str = Field(..., description="Transaction category")
    description: str = Field(..., description="Transaction description")
    date: datetime = Field(..., description="Transaction date")
    mcc: Optional[str] = Field(None, description="Merchant Category Code")


class UserBalance(BaseModel):
    """User balance summary with spending limits"""

    balance: float = Field(..., description="Current balance")
    monthly_spending: float = Field(..., description="Total spending this month")
    monthly_limit: float = Field(..., description="Monthly spending limit")
    daily_spending: float = Field(..., description="Total spending today")
    daily_limit: float = Field(..., description="Daily spending limit")


class SaveTransactionTool(BaseModel):
    """Tool for saving a single transaction"""

    amount: float = Field(..., description="Transaction amount (positive for income, negative for expense)")
    category: str = Field(..., description="Transaction category")
    description: str = Field(..., description="Transaction description")
    transaction_type: str = Field(..., description="Transaction type: 'income' or 'expense'")
    date: Optional[datetime] = Field(None, description="Transaction date (defaults to now)")

    class Config:
        json_schema_extra = {
            "description": "Save a new transaction and return updated user balance with spending limits"
        }


class DeleteTransactionTool(BaseModel):
    """Tool for deleting a specific transaction by ID"""

    transaction_id: UUID = Field(..., description="UUID of the transaction to delete")

    class Config:
        json_schema_extra = {"description": "Delete a specific transaction by its ID"}


class GetUserTransactionsTool(BaseModel):
    """Tool for retrieving user transactions with optional filters"""

    limit: int = Field(default=20, description="Maximum number of transactions to return")
    transaction_type: Optional[str] = Field(None, description="Filter by type: 'income' or 'expense'")
    start_date: Optional[datetime] = Field(None, description="Filter transactions after this date")
    end_date: Optional[datetime] = Field(None, description="Filter transactions before this date")
    category: Optional[str] = Field(None, description="Filter by category")

    class Config:
        json_schema_extra = {
            "description": "Get user's recent transactions with optional filters (default limit: 20)"
        }


class DeleteLastTransactionTool(BaseModel):
    """Tool for deleting the most recently created transaction"""

    class Config:
        json_schema_extra = {"description": "Delete the last created transaction for the user"}


class DeleteLastStatementTransactionsTool(BaseModel):
    """Tool for deleting all transactions from the last bank statement"""

    class Config:
        json_schema_extra = {
            "description": "Delete all transactions created from the last bank statement upload"
        }


# ============================================================================
# Bank Statement Tools
# ============================================================================


class CreateTransactionsFromStatementTool(BaseModel):
    """Tool for creating multiple transactions from a parsed bank statement"""

    transactions: List[TransactionCreateShort] = Field(
        ..., description="List of transactions to create from bank statement"
    )
    statement_date: Optional[datetime] = Field(None, description="Date of the bank statement")

    class Config:
        json_schema_extra = {
            "description": "Create multiple transactions from a parsed bank statement (bulk operation)"
        }


# ============================================================================
# Utility Tools
# ============================================================================


class MCCLookupTool(BaseModel):
    """Tool for looking up Merchant Category Code information"""

    mcc_code: str = Field(..., description="4-digit Merchant Category Code to lookup")

    class Config:
        json_schema_extra = {"description": "Lookup information about a Merchant Category Code (MCC)"}


# ============================================================================
# Transaction Read Schema (for responses)
# ============================================================================


class TransactionRead(BaseModel):
    """Transaction read schema for API responses"""

    id: UUID
    user_id: UUID
    amount: float
    category: str
    description: str
    transaction_type: str
    date: datetime
    bank_statement_id: Optional[UUID] = None
    mcc: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
