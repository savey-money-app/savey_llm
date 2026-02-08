"""
Main Conversational Agent

Handles general user interactions with access to all 7 MCP tools:
1. save_transaction - Save new transaction and return UserBalance
2. delete_transaction - Delete specific transaction by ID
3. get_user_transactions - Get user transactions with filters
4. delete_last_transaction - Delete most recent transaction
5. delete_last_statement_transactions - Delete last statement transactions
6. create_transactions_from_statement - Bulk create from statement
7. mcc_lookup - Lookup Merchant Category Code

Also initiates HITL flows when needed for transaction deletion.
"""

import logging
from typing import Any, Dict, List

from core.config import settings
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from schemas.message import MessageInput
from schemas.api_tools import (
    DeleteLastStatementTransactionsTool,
    DeleteLastTransactionTool,
    DeleteTransactionTool,
    GetUserTransactionsTool,
    MCCLookupTool,
    SaveTransactionTool,
)
from schemas.response import LLMResponse, ToolCall
from services.agents.base_agent import BaseAgent
from services.hitl_flows.transaction_deletion import TransactionDeletionFlow
from services.hitl_manager import HITLManager
from services.api_client import APIClient
from services.prompt_manager import prompt_manager

logger = logging.getLogger(__name__)


class MainAgent(BaseAgent):
    """Main conversational agent with all MCP tools"""

    def __init__(self):
        super().__init__(
            model_name=settings.GEMINI_MODEL_MAIN,
            temperature=settings.GEMINI_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
        self.api_client = APIClient()
        self.hitl_manager = HITLManager()
        self.deletion_flow = TransactionDeletionFlow(self.hitl_manager, self.api_client)

    def get_agent_name(self) -> str:
        return "main"

    def get_system_prompt(self) -> str:
        return prompt_manager.get("main_agent")

    def _define_tools(self) -> List[Any]:
        """
        Define all MCP tools for the main agent

        Returns:
            List of LangChain tool definitions
        """

        @tool
        def save_transaction(
            amount: float,
            category: str,
            description: str,
            transaction_type: str,
            date: str = None,
        ) -> Dict[str, Any]:
            """Save a new transaction and return updated user balance with spending limits.

            Args:
                amount: Transaction amount (positive for income, negative for expense)
                category: Transaction category (e.g., 'Food', 'Transport', 'Salary')
                description: Description of the transaction
                transaction_type: Type of transaction - either 'income' or 'expense'
                date: Optional transaction date in ISO format (defaults to now)

            Returns:
                UserBalance with balance, monthly_spending, monthly_limit, daily_spending, daily_limit
            """
            # This will be replaced with actual implementation at runtime
            pass

        @tool
        def get_user_transactions(
            limit: int = 20,
            transaction_type: str = None,
            start_date: str = None,
            end_date: str = None,
            category: str = None,
        ) -> List[Dict[str, Any]]:
            """Get user's recent transactions with optional filters.

            Args:
                limit: Maximum number of transactions to return (default: 20)
                transaction_type: Filter by 'income' or 'expense'
                start_date: Filter transactions after this date (ISO format)
                end_date: Filter transactions before this date (ISO format)
                category: Filter by category name

            Returns:
                List of transactions with id, amount, category, description, date, etc.
            """
            pass

        @tool
        def delete_transaction(transaction_id: str) -> str:
            """Delete a specific transaction by its ID.

            IMPORTANT: This should only be called after the user has confirmed which
            transaction to delete via HITL flow. Do not call this directly based on
            user description - use get_user_transactions first to find matches.

            Args:
                transaction_id: UUID of the transaction to delete

            Returns:
                Success message
            """
            pass

        @tool
        def delete_last_transaction() -> str:
            """Delete the most recently created transaction for the user.

            Use this when user says something like "delete my last transaction"
            or "undo my last entry".

            Returns:
                Success message
            """
            pass

        @tool
        def delete_last_statement_transactions() -> Dict[str, Any]:
            """Delete all transactions created from the last bank statement upload.

            Use this when user wants to undo a bank statement import.

            Returns:
                Dict with deleted_count
            """
            pass

        @tool
        def mcc_lookup(mcc_code: str) -> Dict[str, Any]:
            """Look up information about a Merchant Category Code (MCC).

            Args:
                mcc_code: 4-digit MCC code to lookup

            Returns:
                Dict with code, description, and category information
            """
            pass

        return [
            save_transaction,
            get_user_transactions,
            delete_transaction,
            delete_last_transaction,
            delete_last_statement_transactions,
            mcc_lookup,
        ]

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Execute a tool call via MCP service

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            user_id: User ID

        Returns:
            Tool execution result
        """
        from datetime import datetime
        from uuid import UUID

        user_uuid = UUID(user_id)

        try:
            if tool_name == "save_transaction":
                # Parse date if provided
                date = None
                if arguments.get("date"):
                    date = datetime.fromisoformat(arguments["date"])

                transaction, balance = await self.api_client.save_transaction(
                    user_id=user_uuid,
                    amount=arguments["amount"],
                    category=arguments["category"],
                    description=arguments["description"],
                    transaction_type=arguments["transaction_type"],
                    date=date,
                )
                result = {"transaction": transaction, "success": True}
                if balance:
                    result["balance"] = balance.model_dump()
                return result

            elif tool_name == "get_user_transactions":
                # Parse dates if provided
                start_date = None
                end_date = None
                if arguments.get("start_date"):
                    start_date = datetime.fromisoformat(arguments["start_date"])
                if arguments.get("end_date"):
                    end_date = datetime.fromisoformat(arguments["end_date"])

                transactions = await self.api_client.get_user_transactions(
                    user_id=user_uuid,
                    limit=arguments.get("limit", 20),
                    transaction_type=arguments.get("transaction_type"),
                    start_date=start_date,
                    end_date=end_date,
                    category=arguments.get("category"),
                )
                return {"transactions": [t.model_dump() for t in transactions]}

            elif tool_name == "delete_transaction":
                transaction_id = UUID(arguments["transaction_id"])
                balance = await self.api_client.delete_transaction(user_uuid, transaction_id)
                result = {"success": True, "message": "Transaction deleted successfully"}
                if balance:
                    result["balance"] = balance.model_dump()
                return result

            elif tool_name == "delete_last_transaction":
                await self.api_client.delete_last_transaction(user_uuid)
                return {"success": True, "message": "Last transaction deleted successfully"}

            elif tool_name == "delete_last_statement_transactions":
                deleted_count = await self.api_client.delete_last_statement_transactions(user_uuid)
                return {"success": True, "deleted_count": deleted_count}

            elif tool_name == "mcc_lookup":
                # For now, return simple lookup (can be implemented in API later)
                mcc_codes = {
                    "5411": {"code": "5411", "description": "Grocery Stores", "category": "Food"},
                    "5812": {"code": "5812", "description": "Restaurants", "category": "Food"},
                }
                result = mcc_codes.get(arguments["mcc_code"], {"code": arguments["mcc_code"], "description": "Unknown"})
                return result

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"❌ Tool execution failed ({tool_name}): {e}")
            return {"error": str(e), "success": False}

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

            # Build messages with category context
            messages = self.build_messages(message, additional_context=category_context)

            # Invoke model
            response = await model.ainvoke(messages)

            # Extract content and tool calls
            content = self.extract_content(response.content if hasattr(response, "content") else "")
            tool_calls = self.extract_tool_calls(response)

            # Execute tool calls if any
            executed_tools = []
            for tc in tool_calls:
                result = await self._execute_tool(tc.name, tc.arguments, str(message.user_id))
                tc.result = result
                executed_tools.append(tc)

                # Check if this is a transaction deletion request that needs HITL
                if tc.name == "get_user_transactions" and "delete" in message.content.lower():
                    # This is likely a deletion request - check if we have matches
                    transactions_data = result.get("transactions", [])
                    if transactions_data:
                        from schemas.api_tools import TransactionRead

                        transactions = [TransactionRead(**t) for t in transactions_data]

                        # Initiate HITL flow
                        deletion_response = await self.deletion_flow.initiate_deletion_flow(
                            user_id=message.user_id,
                            message_id=message.message_id,
                            search_query=message.content,
                            matched_transactions=transactions,
                        )

                        # Return HITL response
                        return self.build_response(
                            message=message,
                            content=deletion_response.message,
                            tool_calls=executed_tools,
                            hitl_flow_id=deletion_response.flow_id,
                            hitl_required=True,
                            hitl_data={"matches_found": deletion_response.matches_found},
                        )

            # Extract balance from tool results if present
            user_balance = None
            for tc in executed_tools:
                if tc.result and "balance" in tc.result:
                    from schemas.api_tools import UserBalance
                    user_balance = UserBalance(**tc.result["balance"])
                    break

            # If we executed tools, regenerate response with results
            if executed_tools:
                # Add tool results to context and regenerate
                tool_results_context = "Tool execution results:\n"
                for tc in executed_tools:
                    tool_results_context += f"- {tc.name}: {tc.result}\n"

                final_messages = messages + [
                    response,
                    HumanMessage(content=tool_results_context),
                ]

                final_response = await model.ainvoke(final_messages)
                content = self.extract_content(final_response.content if hasattr(final_response, "content") else content)

            # Calculate token usage (approximate)
            tokens_used = None
            if hasattr(response, "response_metadata"):
                tokens_used = response.response_metadata.get("token_usage", {}).get("total_tokens")

            return self.build_response(
                message=message, content=content, tool_calls=executed_tools, tokens_used=tokens_used, balance=user_balance
            )

        except Exception as e:
            return await self.handle_error(message, e)
