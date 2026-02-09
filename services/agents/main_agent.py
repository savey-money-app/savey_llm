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

import logging
from typing import Any, Dict, List

from core.config import settings
from langchain_core.messages import HumanMessage
from schemas.message import MessageInput
from schemas.response import LLMResponse, ToolCall
from services.agents.base_agent import BaseAgent
from services.hitl_flows.transaction_deletion import TransactionDeletionFlow
from services.hitl_manager import HITLManager
from services.api_client import APIClient
from services.model_factory import get_model_name
from services.prompt_manager import prompt_manager
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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

    def _define_tools(self) -> List[Any]:
        """Return LangChain tool definitions from the tool registry."""
        return self.tool_registry.get_definitions()

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Execute a tool call via the tool registry."""
        from uuid import UUID

        user_uuid = UUID(user_id)
        return await self.tool_registry.execute(tool_name, user_uuid, arguments)

    async def process_message(self, message: MessageInput) -> LLMResponse:
        """
        Process user message with main agent

        Args:
            message: Input message

        Returns:
            LLM response
        """
        try:
            logger.info(f"🤖 Processing message with MainAgent for user {message.user_id}")

            # Initialize model with tools
            tools = self._define_tools()
            model = self.initialize_model(tools=tools)

            # Fetch categories to guide the LLM in category selection
            try:
                categories = await self.api_client.get_categories(message.user_id)
            except Exception:
                categories = []
            if categories:
                category_lines = "\n".join(f"- {c['title']}" for c in categories)
                category_context = f"Available transaction categories:\n{category_lines}"
            else:
                category_context = "No categories defined yet. You may suggest a suitable category name."

            # Extract user profile from message context
            ctx = message.user_metadata or {}
            user_currency = ctx.get("user_currency", "USD")
            user_fullname = ctx.get("user_fullname", "User")

            # Combine all additional context for the LLM
            additional_context = (
                f"User name: {user_fullname}\n"
                f"User currency: {user_currency}\n\n"
                f"{category_context}"
            )

            # Build messages with additional context
            messages = self.build_messages(message, additional_context=additional_context)

            # Tool execution loop: keep invoking the model and executing
            # tool calls until the LLM returns a pure text response (or we
            # hit the safety limit).
            max_tool_rounds = 5
            executed_tools: List[ToolCall] = []
            content = ""
            response = await model.ainvoke(messages)
            tokens_used = None

            for _round in range(max_tool_rounds):
                content = self.extract_content(response.content if hasattr(response, "content") else "")
                tool_calls = self.extract_tool_calls(response)

                if not tool_calls:
                    # No more tool calls -- we have the final text response
                    break

                # Execute every tool call in this round
                for tc in tool_calls:
                    result = await self._execute_tool(tc.name, tc.arguments, str(message.user_id))
                    tc.result = result
                    executed_tools.append(tc)

                    # Check if this is a transaction deletion request that needs HITL
                    if tc.name == "get_user_transactions" and "delete" in message.content.lower():
                        transactions_data = result.get("transactions", [])
                        if transactions_data:
                            from schemas.api_tools import TransactionRead

                            transactions = [TransactionRead(**t) for t in transactions_data]

                            deletion_response = await self.deletion_flow.initiate_deletion_flow(
                                user_id=message.user_id,
                                message_id=message.message_id,
                                search_query=message.content,
                                matched_transactions=transactions,
                                user_currency=user_currency,
                            )

                            return self.build_response(
                                message=message,
                                content=deletion_response.message,
                                tool_calls=executed_tools,
                                hitl_flow_id=deletion_response.flow_id,
                                hitl_required=True,
                                hitl_data={"matches_found": deletion_response.matches_found},
                            )

                # Feed tool results back to the LLM for the next round
                tool_results_context = "Tool execution results:\n"
                for tc in tool_calls:
                    tool_results_context += f"- {tc.name}: {tc.result}\n"

                messages = messages + [response, HumanMessage(content=tool_results_context)]
                response = await model.ainvoke(messages)

            # After the loop, extract final content from the last response
            content = self.extract_content(response.content if hasattr(response, "content") else content)

            # Extract balance from tool results if present
            user_balance = None
            for tc in executed_tools:
                if tc.result and "balance" in tc.result:
                    from schemas.api_tools import UserBalance
                    user_balance = UserBalance(**tc.result["balance"])
                    break

            # Calculate token usage (approximate)
            if hasattr(response, "response_metadata"):
                tokens_used = response.response_metadata.get("token_usage", {}).get("total_tokens")

            return self.build_response(
                message=message, content=content, tool_calls=executed_tools, tokens_used=tokens_used, balance=user_balance
            )

        except Exception as e:
            return await self.handle_error(message, e)
