"""
LLM Orchestration Service with Multi-Agent Architecture

Routes messages to appropriate specialized agents:
- MainAgent: General conversation and transaction management
- StatementParserAgent: Bank statement parsing with OCR/vision

Also handles HITL (Human-in-the-Loop) flow responses using LLM-based
action inference (no client-provided hitl_action required).
"""

import json
import logging
from datetime import datetime

from core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
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
from services.prompt_manager import prompt_manager

logger = logging.getLogger(__name__)

# ============================================================================
# Structured output JSON schemas for the HITL resolver LLM call
# ============================================================================

_DELETION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["confirm", "select", "cancel"]},
        "selected_number": {"type": "integer"},
        "user_message": {"type": "string"},
    },
    "required": ["action", "user_message"],
}

_STATEMENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["confirm", "cancel", "modify"]},
        "modified_transactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["amount", "category", "description", "date"],
            },
        },
        "user_message": {"type": "string"},
    },
    "required": ["action", "user_message"],
}


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
        """
        Route message to appropriate agent

        Args:
            message: Input message

        Returns:
            LLMResponse from the selected agent
        """
        # Check for bank statement attachment
        if message.attachments and len(message.attachments) > 0:
            logger.info("📄 Routing to StatementParserAgent (attachment detected)")
            return await self.statement_parser_agent.process_message(message)

        # Check for bank statement keywords
        statement_keywords = ["bank statement", "statement", "parse my statement", "uploaded statement"]
        if any(keyword in message.content.lower() for keyword in statement_keywords):
            if not message.attachments:
                # User mentioned statement but didn't attach - ask for it
                return LLMResponse(
                    message_id=message.message_id,
                    user_id=message.user_id,
                    content="📎 Please attach your bank statement (image or PDF) so I can parse it for you.",
                    tool_calls=[],
                    model=settings.GEMINI_MODEL_MAIN,
                    timestamp=datetime.utcnow(),
                    agent_type="router",
                )

        # Default to main agent for all other requests
        logger.info("🤖 Routing to MainAgent")
        return await self.main_agent.process_message(message)

    # ------------------------------------------------------------------
    # HITL resolution via LLM
    # ------------------------------------------------------------------

    def _build_flow_context(self, flow_type: HITLFlowType, flow_data: dict) -> str:
        """Build a human-readable context string describing the HITL flow."""

        if flow_type == HITLFlowType.TRANSACTION_DELETION:
            data = TransactionDeletionFlowData(**flow_data)
            lines = [f"Flow type: transaction_deletion"]
            if data.selected_transaction_id:
                lines.append("A single transaction has already been selected for deletion.")
            lines.append(f"Number of candidate transactions: {len(data.matched_transactions)}")
            for i, t in enumerate(data.matched_transactions[:20], 1):
                lines.append(
                    f"  {i}. {abs(t.amount):.2f} ({t.transaction_type}) "
                    f"- {t.category.title} - {t.description} ({t.date})"
                )
            return "\n".join(lines)

        elif flow_type == HITLFlowType.STATEMENT_PARSING:
            data = StatementParsingFlowData(**flow_data)
            lines = [f"Flow type: statement_parsing"]
            lines.append(f"Number of transactions: {len(data.transactions)}")
            lines.append(f"Iteration: {data.iteration}")
            for i, t in enumerate(data.transactions, 1):
                lines.append(
                    f"  {i}. {t.date} | {t.amount:+.2f} | {t.category} | {t.description}"
                )
            return "\n".join(lines)

        return f"Flow type: {flow_type.value}\nData: {json.dumps(flow_data, default=str)}"

    async def _resolve_hitl_action(
        self, flow_type: HITLFlowType, flow_data: dict, user_message: str
    ) -> dict:
        """
        Call the LLM with structured output to infer the user's HITL action.

        Returns:
            Parsed JSON dict with 'action', 'user_message', and optional fields.
        """
        system_prompt = prompt_manager.get("hitl_resolver")
        flow_context = self._build_flow_context(flow_type, flow_data)

        # Pick the right JSON schema based on flow type
        if flow_type == HITLFlowType.TRANSACTION_DELETION:
            response_schema = _DELETION_RESPONSE_SCHEMA
        else:
            response_schema = _STATEMENT_RESPONSE_SCHEMA

        model = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_MAIN,
            temperature=0.3,
            google_api_key=settings.GEMINI_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Flow context:\n{flow_context}\n\nUser message: {user_message}"},
        ]

        response = await model.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Extract text from list-of-parts if needed
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        return json.loads(content)

    async def _handle_hitl_response(self, message: MessageInput) -> LLMResponse:
        """
        Handle HITL flow continuation using LLM-based action inference.

        The LLM reads the flow context + user message and returns a structured
        JSON decision (confirm / cancel / select / modify).
        """
        flow_id = message.hitl_flow_id

        logger.info(f"🔄 Handling HITL response for flow {flow_id}")

        # 1. Load flow from Redis
        flow = await self.hitl_manager.get_flow(flow_id)
        if not flow:
            return LLMResponse(
                message_id=message.message_id,
                user_id=message.user_id,
                content="❌ This confirmation flow has expired or doesn't exist. Please start over.",
                tool_calls=[],
                model=settings.GEMINI_MODEL_MAIN,
                timestamp=datetime.utcnow(),
                agent_type="hitl_manager",
            )

        flow_type = HITLFlowType(flow["flow_type"])
        current_state = HITLFlowState(flow["state"])

        # Check if flow already completed
        if current_state in (HITLFlowState.CONFIRMED, HITLFlowState.CANCELLED, HITLFlowState.EXPIRED):
            return LLMResponse(
                message_id=message.message_id,
                user_id=message.user_id,
                content=f"This flow has already been {current_state.value}.",
                tool_calls=[],
                model=settings.GEMINI_MODEL_MAIN,
                timestamp=datetime.utcnow(),
                agent_type="hitl_manager",
            )

        # 2. Ask the LLM to interpret the user's reply
        try:
            decision = await self._resolve_hitl_action(flow_type, flow["data"], message.content)
        except Exception as e:
            logger.error(f"❌ HITL resolver LLM call failed: {e}")
            return LLMResponse(
                message_id=message.message_id,
                user_id=message.user_id,
                content="I couldn't understand your response. Please reply with 'confirm', 'cancel', or describe your changes.",
                tool_calls=[],
                model=settings.GEMINI_MODEL_MAIN,
                timestamp=datetime.utcnow(),
                agent_type="hitl_manager",
            )

        action = decision.get("action", "confirm")
        user_msg = decision.get("user_message", "")

        logger.info(f"🤖 HITL resolver decided: action={action} for flow {flow_id}")

        # 3. Execute based on the LLM's decision
        content = user_msg

        if action == "cancel":
            await self.hitl_manager.update_flow_state(flow_id, HITLFlowState.CANCELLED)
            content = content or "❌ Cancelled. No changes were made."

        elif action == "confirm":
            await self.hitl_manager.update_flow_state(flow_id, HITLFlowState.CONFIRMED)
            if flow_type == HITLFlowType.TRANSACTION_DELETION:
                result = await self.deletion_flow.execute_deletion(flow_id, message.user_id)
                content = result
            elif flow_type == HITLFlowType.STATEMENT_PARSING:
                result = await self.parsing_flow.execute_bulk_creation(flow_id, message.user_id)
                content = result

        elif action == "select" and flow_type == HITLFlowType.TRANSACTION_DELETION:
            # User selected a specific transaction from the list
            selected_number = decision.get("selected_number")
            flow_data = TransactionDeletionFlowData(**flow["data"])

            if (
                selected_number is not None
                and 1 <= selected_number <= len(flow_data.matched_transactions)
            ):
                selected = flow_data.matched_transactions[selected_number - 1]
                flow_data.selected_transaction_id = selected.id
                await self.hitl_manager.update_flow_state(
                    flow_id,
                    HITLFlowState.CONFIRMED,
                    flow_data.model_dump(mode="json"),
                )
                result = await self.deletion_flow.execute_deletion(flow_id, message.user_id)
                content = result
            else:
                count = len(flow_data.matched_transactions)
                content = (
                    f"I couldn't determine which transaction you meant. "
                    f"Please pick a number between 1 and {min(count, 20)}, or reply 'cancel'."
                )

        elif action == "modify" and flow_type == HITLFlowType.STATEMENT_PARSING:
            # User requested modifications to the parsed statement
            raw_transactions = decision.get("modified_transactions", [])
            if not raw_transactions:
                content = "I couldn't apply the modifications. Could you describe the changes again?"
            else:
                # Check iteration limit
                current_iteration = int(flow.get("iteration", 1))
                if current_iteration >= self.hitl_manager.max_iterations:
                    await self.hitl_manager.update_flow_state(flow_id, HITLFlowState.EXPIRED)
                    content = (
                        f"Maximum modification iterations ({self.hitl_manager.max_iterations}) "
                        f"reached. Please start over by uploading the statement again."
                    )
                else:
                    # Parse LLM-produced transactions into TransactionCreateShort
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
                        flow_id=flow_id,
                        user_id=message.user_id,
                        modified_transactions=modified,
                        remarks=message.content,
                    )

                    return LLMResponse(
                        message_id=message.message_id,
                        user_id=message.user_id,
                        content=presentation.message,
                        tool_calls=[],
                        model=settings.GEMINI_MODEL_MAIN,
                        timestamp=datetime.utcnow(),
                        agent_type="hitl_manager",
                        hitl_flow_id=flow_id,
                        hitl_required=True,
                        hitl_data={"transaction_count": presentation.transaction_count},
                    )

        else:
            # Unknown combination -- ask user to clarify
            content = "I'm not sure what you'd like to do. Please reply 'confirm', 'cancel', or describe any changes."

        return LLMResponse(
            message_id=message.message_id,
            user_id=message.user_id,
            content=content,
            tool_calls=[],
            model=settings.GEMINI_MODEL_MAIN,
            timestamp=datetime.utcnow(),
            agent_type="hitl_manager",
        )

    async def process_message(self, message: MessageInput) -> LLMResponse:
        """
        Process a message with multi-agent architecture

        Args:
            message: Input message from user

        Returns:
            LLMResponse with content and tool call results
        """
        try:
            logger.info(f"📨 Processing message {message.message_id} from user {message.user_id}")

            # Check if this is a HITL flow response
            if message.hitl_flow_id:
                logger.info(f"🔄 Message is HITL flow response: {message.hitl_flow_id}")
                return await self._handle_hitl_response(message)

            # Route to appropriate agent
            if self.multi_agent_enabled:
                return await self._route_to_agent(message)
            else:
                # Fall back to main agent only
                logger.info("⚠️ Multi-agent disabled, using MainAgent only")
                return await self.main_agent.process_message(message)

        except Exception as e:
            logger.error(f"❌ Error processing message {message.message_id}: {e}", exc_info=True)
            # Return error response
            return LLMResponse(
                message_id=message.message_id,
                user_id=message.user_id,
                content="I apologize, but I encountered an error processing your request. Please try again.",
                tool_calls=[],
                model=settings.GEMINI_MODEL_MAIN,
                timestamp=datetime.utcnow(),
                error=str(e),
                agent_type="error_handler",
            )
