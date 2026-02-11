"""LLM response schemas"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from schemas.api_tools import UserBalance


class ToolCall(BaseModel):
    """Function call made by LLM"""

    name: str = Field(..., description="Name of the tool/function called")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool")
    result: Optional[Dict[str, Any]] = Field(None, description="Result returned from the tool execution")


class LLMResponse(BaseModel):
    """LLM response to publish back to Redis"""

    message_id: str = Field(..., description="Original message ID")
    user_id: UUID = Field(..., description="User ID (UUID format)")
    content: str = Field(..., description="LLM's text response")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of tool calls made")
    model: str = Field(..., description="Model used for generation")
    timestamp: datetime = Field(..., description="Response timestamp")
    tokens_used: Optional[int] = Field(None, description="Total tokens used")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    # HITL fields
    hitl_required: bool = Field(
        default=False, description="Whether this response requires human confirmation (HITL)"
    )
    hitl_data: Optional[Dict[str, Any]] = Field(None, description="HITL flow-specific data")
    # Agent metadata
    agent_type: Optional[str] = Field(None, description="Type of agent that generated this response")
    # Balance data
    balance: Optional[UserBalance] = Field(None, description="User balance if transaction was created/deleted")

    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg_123",
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "Here are your transactions for this month...",
                "tool_calls": [],
                "model": "gpt-4o-mini",
                "timestamp": "2024-01-15T10:30:05Z",
                "tokens_used": 150,
                "error": None,
                "hitl_required": False,
                "hitl_data": None,
                "agent_type": "main",
            }
        }
