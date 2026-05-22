"""
Base Agent Class

Abstract base class for all LLM agents in the multi-agent architecture.
Defines common interface and functionality.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from schemas.message import MessageInput
from schemas.response import LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all LLM agents"""

    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 2000):
        """
        Initialize base agent

        Args:
            model_name: LLM model name (resolved via model factory)
            temperature: Model temperature
            max_tokens: Maximum tokens per response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """Get the name/type of this agent"""
        pass

    def build_response(
        self,
        message: MessageInput,
        content: str,
        tool_calls: Optional[List[ToolCall]] = None,
        tokens_used: Optional[int] = None,
        error: Optional[str] = None,
        hitl_data: Optional[Dict[str, Any]] = None,
        balance: Optional[Any] = None,
    ) -> LLMResponse:
        """Build LLM response object"""
        return LLMResponse(
            message_id=message.message_id,
            user_id=message.user_id,
            content=content,
            tool_calls=tool_calls or [],
            model=self.model_name,
            timestamp=datetime.utcnow(),
            tokens_used=tokens_used,
            error=error,
            hitl_data=hitl_data,
            balance=balance,
        )

    @abstractmethod
    async def process_message(self, message: MessageInput) -> LLMResponse:
        """Process a message and return response"""
        pass

    async def handle_error(self, message: MessageInput, error: Exception) -> LLMResponse:
        """Handle errors gracefully"""
        logger.error(f"Error in {self.get_agent_name()}: {error}")

        error_message = (
            "I apologize, but I encountered an error while processing your request. "
            "Please try again or rephrase your question."
        )

        return self.build_response(
            message=message, content=error_message, error=str(error), tokens_used=0
        )
