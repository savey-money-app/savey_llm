"""
LLM Orchestration Service with Multi-Agent Architecture

Routes messages to appropriate specialized agents:
- MainAgent: General conversation and transaction management
- StatementParserAgent: Bank statement parsing with vision

Also handles HITL (Human-in-the-Loop) flow responses using LLM-based
action inference (no client-provided hitl_action required).
"""

import json
import logging
from collections.abc import Callable, Awaitable
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings

from core.config import settings
from schemas.api_tools import TransactionCreateShort
from schemas.hitl import (
    HITLFlowState,
    HITLFlowType,
    StatementParsingFlowData,
    TransactionDeletionFlowData,
)
from schemas.message import MessageInput
from schemas.response import LLMResponse
from services.agents.main_agent import MainAgent
from services.agents.statement_parser_agent import StatementParserAgent
from services.hitl_flows.statement_parsing import StatementParsingFlow
from services.hitl_flows.transaction_deletion import TransactionDeletionFlow
from services.hitl_manager import HITLManager
from services.api_client import APIClient
from services.model_factory import create_model, get_model_name
from services.prompt_manager import prompt_manager

logger = logging.getLogger(__name__)

# ============================================================================
# Structured output Pydantic models for the HITL resolver LLM call
# ============================================================================


class DeletionDecision(BaseModel):
    """LLM decision for a transaction deletion HITL flow."""
    action: Literal["confirm", "select", "cancel"]
    selected_number: Optional[int] = None
    user_message: str


class ModifiedTransaction(BaseModel):
    """A transaction modified by the user during HITL."""
    amount: float
    category: str
    description: str
    date: str


class StatementDecision(BaseModel):
    """LLM decision for a statement parsing HITL flow."""
    action: Literal["confirm", "cancel", "modify"]
    modified_transactions: Optional[List[ModifiedTransaction]] = None
    user_message: str


class LLMService:
    """
    Service for processing messages with multi-agent architecture

    Handles:
    - Agent routing (main vs statement parser)
    - HITL flow continuation (LLM-driven action inference)
    - Error handling and fallbacks
    """

    def __init__(self):
        # Initialize agents
        self.main_agent = MainAgent()
        self.statement_parser_agent = StatementParserAgent()

        # Initialize HITL components
        self.hitl_manager = HITLManager()
        self.api_client = APIClient()
        self.deletion_flow = TransactionDeletionFlow(self.hitl_manager, self.api_client)
        self.parsing_flow = StatementParsingFlow(self.hitl_manager, self.api_client)

        self.multi_agent_enabled = settings.ENABLE_MULTI_AGENT

    async def _route_to_agent(self, message: MessageInput) -> LLMResponse:
        """Route message to appropriate agent"""
        if message.attachments and len(message.attachments) > 0:
            logger.info("Routing to StatementParserAgent (attachment detected)")
            return await self.statement_parser_agent.process_message(message)

        logger.info("Routing to MainAgent")
        return await self.main_agent.process_message(message)

    # ------------------------------------------------------------------
    # HITL resolution via LLM
    # ------------------------------------------------------------------

    def _build_flow_context(
        self,
        flow_type: HITLFlowType,
        flow_data: dict,
        user_currency: str = "USD",
        user_fullname: str = "User",
    ) -> str:
        """Build a human-readable context string describing the HITL flow."""
        header = [
            f"User name: {user_fullname}",
            f"User currency: {user_currency}",
            "",
        ]

        if flow_type == HITLFlowType.TRANSACTION_DELETION:
            data = TransactionDeletionFlowData(**flow_data)
            lines = [f"Flow type: transaction_deletion"]
            if data.selected_transaction_id:
                lines.append("A single transaction has already been selected for deletion.")
            lines.append(f"Number of candidate transactions: {len(data.matched_transactions)}")
            for i, t in enumerate(data.matched_transactions[:20], 1):
                lines.append(
                    f"  {i}. {abs(t.amount):.2f} {user_currency} ({t.transaction_type}) "
                    f"- {t.category.title} - {t.description} ({t.date})"
                )
            return "\n".join(header + lines)

        elif flow_type == HITLFlowType.STATEMENT_PARSING:
            data = StatementParsingFlowData(**flow_data)
            lines = [f"Flow type: statement_parsing"]
            lines.append(f"Number of transactions: {len(data.transactions)}")
            lines.append(f"Iteration: {data.iteration}")
            for i, t in enumerate(data.transactions, 1):
                lines.append(
                    f"  {i}. {t.date} | {t.amount:+.2f} {user_currency} | {t.category} | {t.description}"
                )
            return "\n".join(header + lines)

        return "\n".join(header) + f"\nFlow type: {flow_type.value}\nData: {json.dumps(flow_data, default=str)}"

    async def _resolve_hitl_action(
        self,
        flow_type: HITLFlowType,
        flow_data: dict,
        user_message: str,
        user_currency: str = "USD",
        user_fullname: str = "User",
    ) -> dict:
        """
        Call the LLM with structured output to infer the user's HITL action.

        Returns:
            Parsed dict with 'action', 'user_message', and optional fields.
        """
        system_prompt = prompt_manager.get("hitl_resolver")
        flow_context = self._build_flow_context(
            flow_type, flow_data, user_currency=user_currency, user_fullname=user_fullname
        )

        model = create_model()

        # Pick the right output type based on flow type
        if flow_type == HITLFlowType.TRANSACTION_DELETION:
            output_type = DeletionDecision
        else:
            output_type = StatementDecision

        agent = Agent(
            model,
            system_prompt=system_prompt,
            output_type=output_type,
            model_settings=ModelSettings(
                temperature=0.3,
                max_tokens=settings.MAX_TOKENS,
            ),
        )

        result = await agent.run(
            f"Flow context:\n{flow_context}\n\nUser message: {user_message}"
        )

        # Return as dict for backward compatibility with the handler
        return result.output.model_dump()

    async def _handle_hitl_response(self, message: MessageInput, flow: dict) -> LLMResponse:
        """
        Handle HITL flow continuation using LLM-based action inference.
        """
        flow_id = flow["flow_id"]
        user_id_str = str(message.user_id)

        logger.info(f"Handling HITL response for flow {flow_id} (user {user_id_str})")

        flow_type = HITLFlowType(flow["flow_type"])

        ctx = message.user_metadata or {}
        user_currency = ctx.get("user_currency", "USD")
        user_fullname = ctx.get("user_fullname", "User")

        try:
            decision = await self._resolve_hitl_action(
                flow_type,
                flow["data"],
                message.content,
                user_currency=user_currency,
                user_fullname=user_fullname,
            )
        except Exception as e:
            logger.error(f"HITL resolver LLM call failed: {e}")
            return LLMResponse(
                message_id=message.message_id,
                user_id=message.user_id,
                content="I couldn't understand your response. Please reply with 'confirm', 'cancel', or describe your changes.",
                tool_calls=[],
                model=get_model_name("main"),
                timestamp=datetime.utcnow(),
            )

        action = decision.get("action", "confirm")
        user_msg = decision.get("user_message", "")

        logger.info(f"HITL resolver decided: action={action} for flow {flow_id}")

        content = user_msg
        balance = None

        if action == "cancel":
            await self.hitl_manager.update_flow_state(user_id_str, HITLFlowState.CANCELLED)
            content = content or "Cancelled. No changes were made."

        elif action == "confirm":
            await self.hitl_manager.update_flow_state(user_id_str, HITLFlowState.CONFIRMED)
            if flow_type == HITLFlowType.TRANSACTION_DELETION:
                content, balance = await self.deletion_flow.execute_deletion(message.user_id)
            elif flow_type == HITLFlowType.STATEMENT_PARSING:
                content, balance = await self.parsing_flow.execute_bulk_creation(message.user_id)

        elif action == "select" and flow_type == HITLFlowType.TRANSACTION_DELETION:
            selected_number = decision.get("selected_number")
            flow_data = TransactionDeletionFlowData(**flow["data"])

            if (
                selected_number is not None
                and 1 <= selected_number <= len(flow_data.matched_transactions)
            ):
                selected = flow_data.matched_transactions[selected_number - 1]
                flow_data.selected_transaction_id = selected.id
                await self.hitl_manager.update_flow_state(
                    user_id_str,
                    HITLFlowState.CONFIRMED,
                    flow_data.model_dump(mode="json"),
                )
                content, balance = await self.deletion_flow.execute_deletion(message.user_id)
            else:
                count = len(flow_data.matched_transactions)
                content = (
                    f"I couldn't determine which transaction you meant. "
                    f"Please pick a number between 1 and {min(count, 20)}, or reply 'cancel'."
                )

        elif action == "modify" and flow_type == HITLFlowType.STATEMENT_PARSING:
            raw_transactions = decision.get("modified_transactions", [])
            if not raw_transactions:
                content = "I couldn't apply the modifications. Could you describe the changes again?"
            else:
                current_iteration = int(flow.get("iteration", 1))
                if current_iteration >= self.hitl_manager.max_iterations:
                    await self.hitl_manager.update_flow_state(user_id_str, HITLFlowState.EXPIRED)
                    content = (
                        f"Maximum modification iterations ({self.hitl_manager.max_iterations}) "
                        f"reached. Please start over by uploading the statement again."
                    )
                else:
                    modified = []
                    for t in raw_transactions:
                        modified.append(
                            TransactionCreateShort(
                                amount=t["amount"],
                                category=t["category"],
                                description=t["description"],
                                date=datetime.fromisoformat(t["date"]),
                            )
                        )

                    presentation = await self.parsing_flow.handle_modification_iteration(
                        user_id=message.user_id,
                        modified_transactions=modified,
                        remarks=message.content,
                    )

                    return LLMResponse(
                        message_id=message.message_id,
                        user_id=message.user_id,
                        content=presentation.message,
                        tool_calls=[],
                        model=get_model_name("main"),
                        timestamp=datetime.utcnow(),
                        hitl_data={"transaction_count": presentation.transaction_count},
                    )

        else:
            content = "I'm not sure what you'd like to do. Please reply 'confirm', 'cancel', or describe any changes."

        return LLMResponse(
            message_id=message.message_id,
            user_id=message.user_id,
            content=content,
            tool_calls=[],
            model=get_model_name("main"),
            timestamp=datetime.utcnow(),
            balance=balance,
        )

    async def stream_message(
        self,
        message: MessageInput,
        on_token: Callable[[str], Awaitable[None]],
    ) -> LLMResponse:
        """
        Stream message processing, calling on_token for each text delta.

        HITL flows and statement parsing are non-streaming and return a
        complete LLMResponse without calling on_token.
        """
        try:
            logger.info(f"Streaming message {message.message_id} from user {message.user_id}")

            active_flow = await self.hitl_manager.get_active_flow(str(message.user_id))
            if active_flow:
                logger.info("Active HITL flow — falling back to non-streaming path")
                return await self._handle_hitl_response(message, active_flow)

            if message.attachments and len(message.attachments) > 0:
                logger.info("Attachment detected — falling back to non-streaming StatementParserAgent")
                return await self.statement_parser_agent.process_message(message)

            logger.info("Routing to MainAgent (streaming)")
            return await self.main_agent.stream_message(message, on_token)

        except Exception as e:
            logger.error(f"Error streaming message {message.message_id}: {e}", exc_info=True)
            return LLMResponse(
                message_id=message.message_id,
                user_id=message.user_id,
                content="I apologize, but I encountered an error processing your request. Please try again.",
                tool_calls=[],
                model=get_model_name("main"),
                timestamp=datetime.utcnow(),
                error=str(e),
            )

    async def process_message(self, message: MessageInput) -> LLMResponse:
        """
        Process a message with multi-agent architecture
        """
        try:
            logger.info(f"Processing message {message.message_id} from user {message.user_id}")

            # Check if user has an active HITL flow
            active_flow = await self.hitl_manager.get_active_flow(str(message.user_id))
            if active_flow:
                logger.info(f"Active HITL flow detected for user {message.user_id}: {active_flow.get('flow_id')}")
                return await self._handle_hitl_response(message, active_flow)

            # Route to appropriate agent
            if self.multi_agent_enabled:
                return await self._route_to_agent(message)
            else:
                logger.info("Multi-agent disabled, using MainAgent only")
                return await self.main_agent.process_message(message)

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}", exc_info=True)
            return LLMResponse(
                message_id=message.message_id,
                user_id=message.user_id,
                content="I apologize, but I encountered an error processing your request. Please try again.",
                tool_calls=[],
                model=get_model_name("main"),
                timestamp=datetime.utcnow(),
                error=str(e)
            )
