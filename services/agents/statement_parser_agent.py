"""
Bank Statement Parser Agent

Uses multimodal capability to parse bank statements from
images and PDFs directly — no OCR preprocessing needed.
"""

import base64
import logging
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelSettings

from core.config import settings
from schemas.bank_statement import ParsedStatement
from schemas.api_tools import TransactionCreateShort
from schemas.message import MessageInput
from schemas.response import LLMResponse
from services.agents.base_agent import BaseAgent
from services.hitl_flows.statement_parsing import StatementParsingFlow
from services.hitl_manager import HITLManager
from services.api_client import APIClient
from services.model_factory import create_model, get_model_name
from services.prompt_manager import prompt_manager

logger = logging.getLogger(__name__)


# Structured output model for the LLM response
class StatementTransaction(BaseModel):
    """A single transaction parsed from a bank statement."""
    date: str = Field(..., description="Transaction date in ISO format")
    amount: float = Field(..., description="Transaction amount (positive for income, negative for expense)")
    description: str = Field(..., description="Transaction description")
    category: str = Field(..., description="Transaction category")
    mcc: Optional[str] = Field(None, description="Merchant Category Code if available")


class StatementParsingOutput(BaseModel):
    """Structured output from bank statement parsing."""
    statement_date: Optional[str] = Field(None, description="Date of the bank statement")
    confidence: float = Field(default=0.9, description="Confidence score for the parsing")
    transactions: List[StatementTransaction] = Field(..., description="Extracted transactions")


class StatementParserAgent(BaseAgent):
    """Agent for parsing bank statements using multimodal LLM"""

    def __init__(self):
        super().__init__(
            model_name=get_model_name("vision"),
            temperature=0.3,
            max_tokens=settings.MAX_TOKENS_VISION,
        )
        self.api_client = APIClient()
        self.hitl_manager = HITLManager()
        self.parsing_flow = StatementParsingFlow(self.hitl_manager, self.api_client)

    def get_agent_name(self) -> str:
        return "statement_parser"

    def get_system_prompt(self) -> str:
        return prompt_manager.get("statement_parser_agent")

    def _build_parsed_statement(self, output: StatementParsingOutput) -> ParsedStatement:
        """Convert StatementParsingOutput to ParsedStatement."""
        transactions = []
        for t in output.transactions:
            amount = float(t.amount)
            transactions.append(
                TransactionCreateShort(
                    amount=amount,
                    category=t.category,
                    description=t.description,
                    date=datetime.fromisoformat(t.date),
                    mcc=t.mcc,
                )
            )

        statement_date = None
        if output.statement_date:
            statement_date = datetime.fromisoformat(output.statement_date)

        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_expenses = sum(abs(t.amount) for t in transactions if t.amount < 0)

        return ParsedStatement(
            transactions=transactions,
            statement_date=statement_date,
            total_income=total_income,
            total_expenses=total_expenses,
            confidence=output.confidence,
        )

    async def parse_statement(self, data: str, mime_type: str) -> ParsedStatement:
        """
        Parse a bank statement by passing the file directly to the LLM.

        Uses PydanticAI's output_type for structured output and BinaryContent
        for multimodal input.

        Args:
            data: Base64-encoded file data
            mime_type: MIME type (e.g. 'application/pdf', 'image/png')

        Returns:
            ParsedStatement with extracted transactions
        """
        logger.info(f"Parsing bank statement via {settings.LLM_PROVIDER} multimodal ({mime_type})")

        model = create_model(model_name=self.model_name)

        agent = Agent(
            model,
            system_prompt=self.get_system_prompt(),
            output_type=StatementParsingOutput,
            model_settings=ModelSettings(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
        )

        # Decode base64 data and pass as BinaryContent
        binary_data = base64.b64decode(data)

        result = await agent.run([
            "Extract all transactions from this bank statement.",
            BinaryContent(data=binary_data, media_type=mime_type),
        ])

        output = result.output  # Already a validated StatementParsingOutput
        parsed_statement = self._build_parsed_statement(output)

        logger.info(
            f"Parsed {len(parsed_statement.transactions)} transactions "
            f"(confidence: {parsed_statement.confidence})"
        )
        return parsed_statement

    async def process_message(self, message: MessageInput) -> LLMResponse:
        """
        Process bank statement parsing request.

        Args:
            message: Input message with attachment

        Returns:
            LLM response with HITL flow initiation
        """
        try:
            logger.info(f"Processing bank statement for user {message.user_id}")

            if not message.attachments or len(message.attachments) == 0:
                return self.build_response(
                    message=message,
                    content="No bank statement attached. Please upload an image or PDF of your bank statement.",
                )

            attachment = message.attachments[0]

            try:
                parsed_statement = await self.parse_statement(attachment.data, attachment.mime_type)
            except Exception as e:
                logger.error(f"Failed to parse bank statement: {e}")
                return self.build_response(
                    message=message,
                    content=f"Failed to parse bank statement: {e}",
                    error=str(e),
                )

            if not parsed_statement or not parsed_statement.transactions:
                return self.build_response(
                    message=message,
                    content="No transactions found in the statement. Please check the file and try again.",
                )

            ctx = message.user_metadata or {}
            user_currency = ctx.get("user_currency", "USD")

            presentation = await self.parsing_flow.initiate_parsing_flow(
                user_id=message.user_id,
                message_id=message.message_id,
                parsed_statement=parsed_statement,
                user_currency=user_currency,
            )

            return self.build_response(
                message=message,
                content=presentation.message,
                hitl_data={
                    "transaction_count": presentation.transaction_count,
                    "total_income": presentation.total_income,
                    "total_expenses": presentation.total_expenses,
                },
            )

        except Exception as e:
            return await self.handle_error(message, e)
