"""
Bank Statement Schemas

Schemas for bank statement parsing, storage, and HITL confirmation flows.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .api_tools import TransactionCreateShort


class BankStatement(BaseModel):
    """Bank statement entity for tracking statement uploads"""

    id: UUID = Field(..., description="Unique identifier for the bank statement")
    user_id: UUID = Field(..., description="User who uploaded the statement")
    created_at: datetime = Field(..., description="When the statement was uploaded")
    statement_date: Optional[datetime] = Field(None, description="Date of the bank statement")
    transaction_count: int = Field(default=0, description="Number of transactions in this statement")

    model_config = ConfigDict(from_attributes=True)


class ParsedStatement(BaseModel):
    """Parsed bank statement data before confirmation"""

    transactions: List[TransactionCreateShort] = Field(
        ..., description="List of transactions parsed from the bank statement"
    )
    statement_date: Optional[datetime] = Field(None, description="Date of the bank statement")
    total_income: float = Field(default=0.0, description="Total income in the statement")
    total_expenses: float = Field(default=0.0, description="Total expenses in the statement")
    raw_text: Optional[str] = Field(None, description="Raw OCR text extracted from the statement")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score for the parsing (0.0-1.0)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "description": "Parsed bank statement data awaiting user confirmation in HITL flow"
        }
    )


class StatementParsingRequest(BaseModel):
    """Request to parse a bank statement from attachment"""

    user_id: UUID = Field(..., description="User ID requesting the parsing")
    message_id: str = Field(..., description="Message ID for tracking")
    attachment_data: str = Field(..., description="Base64-encoded image or PDF data")
    attachment_type: str = Field(..., description="MIME type of the attachment (e.g., 'image/png', 'application/pdf')")
    statement_date: Optional[datetime] = Field(None, description="Optional statement date provided by user")

    model_config = ConfigDict(
        json_schema_extra={"description": "Request to parse bank statement using OCR"}
    )


class StatementParsingResponse(BaseModel):
    """Response from bank statement parsing"""

    success: bool = Field(..., description="Whether parsing was successful")
    parsed_statement: Optional[ParsedStatement] = Field(None, description="Parsed statement data if successful")
    error: Optional[str] = Field(None, description="Error message if parsing failed")
    requires_confirmation: bool = Field(
        default=True, description="Whether this requires HITL confirmation before creating transactions"
    )

    model_config = ConfigDict(
        json_schema_extra={"description": "Result of bank statement parsing operation"}
    )
