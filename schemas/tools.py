"""Tool/Function calling schemas"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from decimal import Decimal
from datetime import date


class GetTransactionsTool(BaseModel):
    """Get user transactions with optional filters"""
    transaction_type: Optional[Literal["income", "expense"]] = Field(
        None,
        description="Filter by transaction type"
    )
    start_date: Optional[date] = Field(
        None,
        description="Filter transactions from this date"
    )
    end_date: Optional[date] = Field(
        None,
        description="Filter transactions until this date"
    )
    category: Optional[str] = Field(
        None,
        description="Filter by category name"
    )


class CreateTransactionTool(BaseModel):
    """Create a new transaction"""
    amount: Decimal = Field(
        ...,
        gt=0,
        description="Transaction amount (must be positive)"
    )
    category: str = Field(
        ...,
        description="Category name"
    )
    description: Optional[str] = Field(
        None,
        description="Optional description"
    )
    transaction_type: Literal["income", "expense"] = Field(
        ...,
        description="Type of transaction"
    )
    date: date = Field(
        ...,
        description="Date of the transaction"
    )


class GetCategoriesTool(BaseModel):
    """Get all categories for the user"""
    pass


class GetBalanceTool(BaseModel):
    """Get current balance summary"""
    pass
