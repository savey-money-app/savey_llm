"""Input message schemas"""
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from schemas.api_tools import UserBalance


class MessageAttachment(BaseModel):
    """Attachment data for messages (bank statements, receipts, etc.)"""

    data: str = Field(..., description="Base64-encoded attachment data")
    mime_type: str = Field(..., description="MIME type (e.g., 'image/png', 'application/pdf')")
    filename: Optional[str] = Field(None, description="Original filename if available")
    size: Optional[int] = Field(None, description="File size in bytes")


class MessageInput(BaseModel):
    """Input message from Redis PubSub"""

    user_id: UUID = Field(..., description="User ID (UUID format)")
    message_id: str = Field(..., description="Unique message identifier")
    content: str = Field(..., description="Message text content")
    timestamp: datetime = Field(..., description="Message timestamp")
    user_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context (conversation history, etc.)")
    attachments: Optional[List[MessageAttachment]] = Field(
        None, description="Optional file attachments (bank statements, receipts)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "message_id": "msg_123",
                "content": "Show me my transactions for this month",
                "timestamp": "2024-01-15T10:30:00Z",
                "context": {"conversation_history": []},
                "attachments": None,
            }
        }
    )

class MessageOutput(BaseModel):
    """Output message to publish to Redis"""

    content: Optional[str] = Field(None, description="LLM's text response")
    done: bool = Field(default=False, description="Whether this is the final message in the stream")
    hitl_data: Optional[Dict[str, Any]] = Field(None, description="HITL flow-specific data")
    balance: Optional[UserBalance] = Field(None, description="User balance if transaction was created/deleted")
    error: Optional[str] = Field(None, description="Error message if processing failed")
