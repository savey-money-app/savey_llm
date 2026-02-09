"""
API Response Schemas

Pydantic models for data returned by the savey_api backend.
Tool *parameter* schemas now live alongside their tool implementations
in the ``tools/`` package.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Shared Data Schemas
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


# ============================================================================
# Transaction Read / Response Schemas
# ============================================================================


class CategoryResponse(BaseModel):
    """Schema for category response"""

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TransactionRead(BaseModel):
    """Transaction read schema for API responses"""

    id: UUID
    user_id: UUID
    category_id: UUID
    category: CategoryResponse
    amount: Decimal
    description: Optional[str] = None
    transaction_type: str
    date: date
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
