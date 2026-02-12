"""
Base Agent Class

Abstract base class for all LLM agents in the multi-agent architecture.
Defines common interface and functionality.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from schemas.message import MessageInput
from schemas.response import LLMResponse, ToolCall
from services.prompt_manager import prompt_manager

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
        self.model: Optional[BaseChatModel] = None

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Get the name/type of this agent

        Returns:
            Agent name string
        """
        pass

    @staticmethod
    def extract_content(raw_content: Any) -> str:
        """
        Extract plain text from a Gemini response content field.

        Gemini may return a list of parts: [{'type': 'text', 'text': '...'}, ...]
        or a plain string.
        """
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        return str(raw_content) if raw_content is not None else ""

    def initialize_model(self, tools: Optional[List[Any]] = None) -> BaseChatModel:
        """
        Initialize the chat model via the provider-agnostic factory.

        Args:
            tools: Optional list of tools to bind to the model

        Returns:
            Initialized BaseChatModel instance
        """
        from services.model_factory import create_chat_model

        self.model = create_chat_model(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tools,
        )
        return self.model

    def build_messages(
        self, message: MessageInput, additional_context: Optional[str] = None
    ) -> List[BaseMessage]:
        """
        Build message list for LLM invocation

        Args:
            message: Input message
            additional_context: Optional additional context to inject

        Returns:
            List of LangChain messages
        """
        messages = [SystemMessage(content=self.get_system_prompt())]

        # Add conversation history if available
        if message.user_metadata and "conversation_history" in message.user_metadata:
            for msg in message.user_metadata["conversation_history"]:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                # Add more message types as needed

        # Add additional context if provided
        if additional_context:
            messages.append(SystemMessage(content=f"Additional Context:\n{additional_context}"))

        # Add current user message
        messages.append(HumanMessage(content=message.content))

        return messages

    def extract_tool_calls(self, ai_message: Any) -> List[ToolCall]:
        """
        Extract tool calls from AI message

        Args:
            ai_message: AI message from model invocation

        Returns:
            List of ToolCall objects
        """
        tool_calls = []

        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            for tc in ai_message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.get("name", ""),
                        arguments=tc.get("args", {}),
                        result=None,  # Will be filled after execution
                    )
                )

        return tool_calls

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
        """
        Build LLM response object

        Args:
            message: Original input message
            content: Response content
            tool_calls: List of tool calls made
            tokens_used: Number of tokens used
            error: Error message if any
            hitl_data: HITL flow data
            balance: User balance if transaction was created/deleted

        Returns:
            LLMResponse object
        """
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
        """
        Process a message and return response

        Args:
            message: Input message

        Returns:
            LLM response
        """
        pass

    async def handle_error(self, message: MessageInput, error: Exception) -> LLMResponse:
        """
        Handle errors gracefully

        Args:
            message: Original message
            error: Exception that occurred

        Returns:
            Error response
        """
        logger.error(f"❌ Error in {self.get_agent_name()}: {error}")

        error_message = (
            f"I apologize, but I encountered an error while processing your request. "
            f"Please try again or rephrase your question."
        )

        return self.build_response(
            message=message, content=error_message, error=str(error), tokens_used=0
        )
