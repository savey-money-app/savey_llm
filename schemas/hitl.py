"""
HITL (Human-in-the-Loop) Schemas

Schemas for managing human confirmation flows with Redis state management.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .api_tools import TransactionCreateShort, TransactionRead


class HITLFlowType(str, Enum):
    """Types of HITL confirmation flows"""

    TRANSACTION_DELETION = "transaction_deletion"
    STATEMENT_PARSING = "statement_parsing"


class HITLFlowState(str, Enum):
    """States of a HITL flow"""

    PENDING = "pending"  # Waiting for user response
    IN_PROGRESS = "in_progress"  # User is reviewing/modifying
    CONFIRMED = "confirmed"  # User confirmed
    CANCELLED = "cancelled"  # User cancelled
    EXPIRED = "expired"  # Flow timed out


class HITLRequest(BaseModel):
    """Request to initiate a HITL flow"""

    flow_id: str = Field(..., description="Unique identifier for this HITL flow")
    user_id: UUID = Field(..., description="User ID for this flow")
    message_id: str = Field(..., description="Original message ID that triggered the flow")
    flow_type: HITLFlowType = Field(..., description="Type of HITL flow")
    data: Dict[str, Any] = Field(..., description="Flow-specific data payload")
    expires_at: datetime = Field(..., description="When this flow expires")

    class Config:
        json_schema_extra = {"description": "Request to initiate a human-in-the-loop confirmation flow"}


class HITLResponse(BaseModel):
    """Response containing HITL flow status"""

    flow_id: str = Field(..., description="Unique identifier for this HITL flow")
    flow_type: HITLFlowType = Field(..., description="Type of HITL flow")
    state: HITLFlowState = Field(..., description="Current state of the flow")
    message: str = Field(..., description="Message to display to the user")
    data: Dict[str, Any] = Field(default_factory=dict, description="Flow-specific response data")
    requires_user_action: bool = Field(default=True, description="Whether user action is required")

    class Config:
        json_schema_extra = {"description": "Response from a HITL flow containing current state and message"}


# ============================================================================
# Transaction Deletion HITL
# ============================================================================


class TransactionDeletionFlowData(BaseModel):
    """Data for transaction deletion HITL flow"""

    search_query: str = Field(..., description="User's search query for transactions to delete")
    matched_transactions: List[TransactionRead] = Field(..., description="Transactions matching the search")
    selected_transaction_id: Optional[UUID] = Field(None, description="Transaction ID selected by user")

    class Config:
        json_schema_extra = {
            "description": "Data payload for transaction deletion HITL flow with search results"
        }


class TransactionDeletionResponse(BaseModel):
    """Response for transaction deletion flow"""

    matches_found: int = Field(..., description="Number of matching transactions")
    transactions: List[TransactionRead] = Field(..., description="List of matching transactions")
    message: str = Field(..., description="Message asking user to select which transaction to delete")
    flow_id: str = Field(..., description="Flow ID for user to respond with selection")

    class Config:
        json_schema_extra = {"description": "Response asking user to select which transaction to delete"}


# ============================================================================
# Statement Parsing HITL
# ============================================================================


class StatementParsingFlowData(BaseModel):
    """Data for bank statement parsing HITL flow"""

    transactions: List[TransactionCreateShort] = Field(..., description="Parsed transactions from statement")
    statement_date: Optional[datetime] = Field(None, description="Date of the statement")
    iteration: int = Field(default=1, description="Current iteration of the confirmation loop")
    user_remarks: Optional[str] = Field(None, description="User's remarks from previous iteration")

    class Config:
        json_schema_extra = {
            "description": "Data payload for statement parsing HITL flow with parsed transactions"
        }


class StatementParsingConfirmation(BaseModel):
    """User's confirmation or modification of parsed statement"""

    flow_id: str = Field(..., description="Flow ID this confirmation is for")
    confirmed: bool = Field(..., description="Whether user confirms the transactions")
    modified_transactions: Optional[List[TransactionCreateShort]] = Field(
        None, description="Modified transaction list if user made changes"
    )
    remarks: Optional[str] = Field(None, description="User's remarks or requested modifications")
    cancel: bool = Field(default=False, description="Whether user wants to cancel the flow")

    class Config:
        json_schema_extra = {"description": "User's confirmation or modification of parsed bank statement"}


class StatementParsingPresentationList(BaseModel):
    """Formatted list of transactions for user review"""

    transactions: List[TransactionCreateShort] = Field(..., description="Transactions to review")
    total_income: float = Field(..., description="Sum of all income transactions")
    total_expenses: float = Field(..., description="Sum of all expense transactions")
    net_change: float = Field(..., description="Net change (income - expenses)")
    transaction_count: int = Field(..., description="Total number of transactions")
    message: str = Field(..., description="Formatted message presenting the transactions")
    flow_id: str = Field(..., description="Flow ID for user response")

    class Config:
        json_schema_extra = {"description": "Presentation of parsed transactions for user confirmation"}
