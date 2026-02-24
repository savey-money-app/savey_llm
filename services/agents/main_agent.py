"""
Main Conversational Agent

Handles general user interactions with access to all tools:
1. save_transaction - Save new transaction and return UserBalance
2. delete_transaction - Delete specific transaction by ID
3. get_user_transactions - Get user transactions with filters
4. delete_last_transaction - Delete most recent transaction
5. delete_last_statement_transactions - Delete last statement transactions
6. mcc_lookup - Lookup Merchant Category Code

Also initiates HITL flows when needed for transaction deletion.
"""

import asyncio
import logging
from collections.abc import Callable, Awaitable
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.toolsets.function import FunctionToolset

from core.config import settings
from schemas.message import MessageInput
from schemas.response import LLMResponse
from services.agents.base_agent import BaseAgent
from services.hitl_flows.transaction_deletion import TransactionDeletionFlow
from services.hitl_manager import HITLManager
from services.api_client import APIClient
from services.model_factory import create_model, create_openai_model, get_model_name
from services.prompt_manager import prompt_manager
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class HITLInterrupt(Exception):
    """Raised by a tool when a HITL flow is triggered, to short-circuit the agent run."""

    def __init__(self, response: LLMResponse):
        self.response = response
        super().__init__("HITL flow triggered")


class MainAgent(BaseAgent):
    """Main conversational agent with all tools"""

    def __init__(self):
        super().__init__(
            model_name=get_model_name("main"),
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
        self.api_client = APIClient()
        self.hitl_manager = HITLManager()
        self.deletion_flow = TransactionDeletionFlow(self.hitl_manager, self.api_client)
        self.tool_registry = ToolRegistry(self.api_client)

    def get_agent_name(self) -> str:
        return "main"

    def get_system_prompt(self) -> str:
        return prompt_manager.get("main_agent")

    def _create_toolsets(
        self,
        user_id: UUID,
        user_currency: str,
        message_id: str,
        message_content: str,
    ) -> list:
        """
        Create toolsets for the PydanticAI Agent.

        Returns a list of two FunctionToolsets:
        1. All registry tools except get_user_transactions
        2. HITL-aware get_user_transactions override

        All tools receive user_id via closure (no RunContext needed).
        """
        # 1. Registry toolset for all tools except get_user_transactions
        registry_toolset = FunctionToolset()
        for tool in self.tool_registry._tools.values():
            if tool.name == "get_user_transactions":
                continue
            wrapper = self.tool_registry._make_tool_wrapper(tool, user_id)
            registry_toolset.tool(wrapper, name=tool.name, description=tool.description)

        # 2. HITL-aware get_user_transactions
        hitl_toolset = FunctionToolset()
        deletion_flow = self.deletion_flow
        tool_registry = self.tool_registry
        model_name = self.model_name

        @hitl_toolset.tool(
            name="get_user_transactions",
            description=self.tool_registry.get_tool("get_user_transactions").description,
        )
        async def get_user_transactions_with_hitl(
            limit: int = 20,
            transaction_type: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            category: Optional[str] = None,
            for_deletion: bool = False,
        ) -> dict:
            result = await tool_registry.execute(
                "get_user_transactions",
                user_id,
                {
                    "limit": limit,
                    "transaction_type": transaction_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "category": category,
                    "for_deletion": for_deletion,
                },
            )

            if for_deletion:
                transactions_data = result.get("transactions", [])
                if transactions_data:
                    from schemas.api_tools import TransactionRead

                    transactions = [TransactionRead(**t) for t in transactions_data]
                    deletion_response = await deletion_flow.initiate_deletion_flow(
                        user_id=user_id,
                        message_id=message_id,
                        search_query=message_content,
                        matched_transactions=transactions,
                        user_currency=user_currency,
                    )

                    raise HITLInterrupt(
                        response=LLMResponse(
                            message_id=message_id,
                            user_id=user_id,
                            content=deletion_response.message,
                            tool_calls=[],
                            model=model_name,
                            timestamp=datetime.utcnow(),
                            hitl_data={"matches_found": deletion_response.matches_found},
                        )
                    )

            return result

        return [registry_toolset, hitl_toolset]

    def _extract_balance(self, result: Any) -> Optional[Any]:
        """Pull UserBalance out of tool-result messages, if present."""
        for msg in result.all_messages():
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "content") and isinstance(part.content, dict):
                        if "balance" in part.content:
                            from schemas.api_tools import UserBalance
                            return UserBalance(**part.content["balance"])
        return None

    async def _execute_stream(
        self,
        agent: Any,
        user_prompt: str,
        on_token: Callable[[str], Awaitable[None]],
    ) -> tuple[str, Optional[Any]]:
        """Run an agent stream, calling on_token per delta. Returns (content, balance)."""
        content = ""
        user_balance = None
        async with agent.run_stream(user_prompt) as result:
            async for delta in result.stream_text(delta=True):
                await on_token(delta)
                content += delta
            user_balance = self._extract_balance(result)
        return content, user_balance

    async def process_message(self, message: MessageInput) -> LLMResponse:
        """
        Process user message with main agent.

        PydanticAI handles the tool loop automatically.
        """
        try:
            logger.info(f"Processing message with MainAgent for user {message.user_id}")

            # Fetch categories to guide the LLM in category selection
            try:
                categories = await self.api_client.get_categories(message.user_id)
            except Exception:
                categories = []
            if categories:
                category_lines = []
                for c in categories:
                    line = f"- {c['title']}"
                    if c.get("title_ru"):
                        line += f" ({c['title_ru']})"
                    category_lines.append(line)
                category_context = f"Available transaction categories:\n" + "\n".join(category_lines)
            else:
                category_context = "No categories defined yet. You may suggest a suitable category name."

            # Extract user profile from message context
            ctx = message.user_metadata or {}
            user_currency = ctx.get("user_currency", "USD")
            user_fullname = ctx.get("user_fullname", "User")

            # Build additional context
            additional_context = (
                f"User name: {user_fullname}\n"
                f"User currency: {user_currency}\n\n"
                f"{category_context}"
            )

            # Build conversation history for the user prompt
            history_parts = []
            if message.user_metadata and "conversation_history" in message.user_metadata:
                for msg in message.user_metadata["conversation_history"]:
                    if msg.get("role") == "user":
                        history_parts.append(f"User: {msg.get('content', '')}")

            user_prompt = message.content
            if history_parts:
                history_text = "\n".join(history_parts)
                user_prompt = f"Conversation history:\n{history_text}\n\nCurrent message: {message.content}"

            # Build system prompt with context
            system_prompt = self.get_system_prompt() + f"\n\nAdditional Context:\n{additional_context}"

            # Create PydanticAI agent with tools
            model = create_model(model_name=self.model_name)
            toolsets = self._create_toolsets(
                user_id=message.user_id,
                user_currency=user_currency,
                message_id=message.message_id,
                message_content=message.content,
            )

            agent = Agent(
                model,
                system_prompt=system_prompt,
                toolsets=toolsets,
                model_settings=ModelSettings(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ),
            )

            try:
                try:
                    result = await asyncio.wait_for(
                        agent.run(user_prompt), timeout=settings.LLM_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[MainAgent] Primary LLM timed out ({settings.LLM_TIMEOUT}s), "
                        "falling back to OpenAI"
                    )
                    fallback_agent = Agent(
                        create_openai_model(),
                        system_prompt=system_prompt,
                        toolsets=toolsets,
                        model_settings=ModelSettings(
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        ),
                    )
                    result = await asyncio.wait_for(
                        fallback_agent.run(user_prompt), timeout=settings.LLM_TIMEOUT
                    )
                content = result.output
            except HITLInterrupt as hitl:
                return hitl.response

            if not content or not content.strip():
                logger.warning("MainAgent produced empty content")
                content = "I'm sorry, I wasn't able to complete that request. Could you please try again?"

            user_balance = self._extract_balance(result)

            return self.build_response(
                message=message,
                content=content,
                tokens_used=result.usage().total_tokens if result.usage() else None,
                balance=user_balance,
            )

        except Exception as e:
            return await self.handle_error(message, e)

    async def stream_message(
        self,
        message: MessageInput,
        on_token: Callable[[str], Awaitable[None]],
    ) -> LLMResponse:
        """
        Process user message with token-level streaming.

        Calls on_token(delta) for each text chunk as it is generated.
        Falls back to the full response if a HITL interrupt is raised
        (tool calls complete before text generation starts, so no spurious
        tokens are emitted before a HITL interrupt).
        """
        try:
            logger.info(f"Streaming message with MainAgent for user {message.user_id}")

            try:
                categories = await self.api_client.get_categories(message.user_id)
            except Exception:
                categories = []
            if categories:
                category_lines = []
                for c in categories:
                    line = f"- {c['title']}"
                    if c.get("title_ru"):
                        line += f" ({c['title_ru']})"
                    category_lines.append(line)
                category_context = "Available transaction categories:\n" + "\n".join(category_lines)
            else:
                category_context = "No categories defined yet. You may suggest a suitable category name."

            ctx = message.user_metadata or {}
            user_currency = ctx.get("user_currency", "USD")
            user_fullname = ctx.get("user_fullname", "User")

            additional_context = (
                f"User name: {user_fullname}\n"
                f"User currency: {user_currency}\n\n"
                f"{category_context}"
            )

            user_prompt = message.content
            system_prompt = self.get_system_prompt() + f"\n\nAdditional Context:\n{additional_context}"

            model = create_model(model_name=self.model_name)
            toolsets = self._create_toolsets(
                user_id=message.user_id,
                user_currency=user_currency,
                message_id=message.message_id,
                message_content=message.content,
            )

            agent = Agent(
                model,
                system_prompt=system_prompt,
                toolsets=toolsets,
                model_settings=ModelSettings(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ),
            )

            content = ""
            user_balance = None

            # Track emitted tokens to decide whether fallback is safe
            emitted: list[str] = []

            async def tracked_on_token(delta: str) -> None:
                emitted.append(delta)
                await on_token(delta)

            try:
                try:
                    content, user_balance = await asyncio.wait_for(
                        self._execute_stream(agent, user_prompt, tracked_on_token),
                        timeout=settings.LLM_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    if not emitted:
                        logger.warning(
                            f"[MainAgent] Primary LLM timed out ({settings.LLM_TIMEOUT}s) "
                            "with no tokens — falling back to OpenAI"
                        )
                        fallback_agent = Agent(
                            create_openai_model(),
                            system_prompt=system_prompt,
                            toolsets=toolsets,
                            model_settings=ModelSettings(
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                            ),
                        )
                        content, user_balance = await self._execute_stream(
                            fallback_agent, user_prompt, on_token
                        )
                    else:
                        logger.warning(
                            f"[MainAgent] LLM timed out mid-stream after {len(emitted)} tokens"
                        )
                        content = "".join(emitted)

            except HITLInterrupt as hitl:
                return hitl.response

            if not content or not content.strip():
                logger.warning("MainAgent stream produced empty content")
                content = "I'm sorry, I wasn't able to complete that request. Could you please try again?"

            return self.build_response(
                message=message,
                content=content,
                balance=user_balance,
            )

        except Exception as e:
            return await self.handle_error(message, e)
