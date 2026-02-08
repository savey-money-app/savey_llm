"""
LLM Orchestration Service with Multi-Agent Architecture

Routes messages to appropriate specialized agents:
- MainAgent: General conversation and transaction management
- StatementParserAgent: Bank statement parsing with OCR/vision

Also handles HITL (Human-in-the-Loop) flow responses.
"""

import logging
from datetime import datetime

from core.config import settings
from schemas.hitl import HITLUserResponse
from schemas.message import MessageInput
from schemas.response import LLMResponse
from services.agents.main_agent import MainAgent
from services.agents.statement_parser_agent import StatementParserAgent
from services.hitl_flows.statement_parsing import StatementParsingFlow
from services.hitl_flows.transaction_deletion import TransactionDeletionFlow
from services.hitl_manager import HITLManager
from services.api_client import APIClient

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for processing messages with multi-agent architecture

    Handles:
    - Agent routing (main vs statement parser)
    - HITL flow continuation
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

    async def _handle_hitl_response(self, message: MessageInput) -> LLMResponse:
        """
        Handle HITL flow continuation

        Args:
            message: Message with HITL flow ID and action

        Returns:
            LLMResponse based on HITL action
        """
        flow_id = message.hitl_flow_id
        action = message.hitl_action or "confirm"  # Default to confirm

        logger.info(f"🔄 Handling HITL response for flow {flow_id}: {action}")

        # Get flow data
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

        from schemas.hitl import HITLFlowType, HITLFlowState

        flow_type = HITLFlowType(flow["flow_type"])

        # Create user response
        user_response = HITLUserResponse(
            flow_id=flow_id, user_id=message.user_id, action=action, comment=message.content
        )

        # Process the response
        hitl_response = await self.hitl_manager.process_user_response(flow_id, user_response)

        # Handle based on flow type and state
        if hitl_response.state == HITLFlowState.CONFIRMED:
            # Execute the confirmed action
            if flow_type == HITLFlowType.TRANSACTION_DELETION:
                result = await self.deletion_flow.execute_deletion(flow_id, message.user_id)
                content = result

            elif flow_type == HITLFlowType.STATEMENT_PARSING:
                result = await self.parsing_flow.execute_bulk_creation(flow_id, message.user_id)
                content = result

            else:
                content = hitl_response.message

        elif hitl_response.state == HITLFlowState.IN_PROGRESS:
            # Modification requested - need to re-parse or re-present
            content = "🔄 Processing your modifications..."
            # TODO: Implement modification logic

        else:
            # Cancelled, expired, or other state
            content = hitl_response.message

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
