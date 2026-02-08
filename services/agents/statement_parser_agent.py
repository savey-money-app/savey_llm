"""
Bank Statement Parser Agent

Uses Gemini's native multimodal capability to parse bank statements from
images and PDFs directly — no OCR preprocessing needed.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.bank_statement import ParsedStatement, StatementParsingResponse
from schemas.api_tools import TransactionCreateShort
from schemas.message import MessageInput
from schemas.response import LLMResponse
from services.agents.base_agent import BaseAgent
from services.hitl_flows.statement_parsing import StatementParsingFlow
from services.hitl_manager import HITLManager
from services.api_client import APIClient
from services.prompt_manager import prompt_manager

logger = logging.getLogger(__name__)


class StatementParserAgent(BaseAgent):
    """Agent for parsing bank statements using Gemini multimodal"""

    def __init__(self):
        super().__init__(
            model_name=settings.GEMINI_MODEL_VISION,
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

    def _parse_llm_json(self, content: str) -> dict:
        """Extract and parse JSON from LLM response, handling markdown code blocks."""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)

    def _build_parsed_statement(self, parsed_data: dict) -> ParsedStatement:
        """Convert parsed JSON dict to ParsedStatement."""
        transactions = []
        for t in parsed_data.get("transactions", []):
            amount = float(t["amount"])
            transactions.append(
                TransactionCreateShort(
                    amount=amount,
                    category=t.get("category", "Uncategorized"),
                    description=t.get("description", ""),
                    date=datetime.fromisoformat(t["date"]),
                    mcc=t.get("mcc"),
                )
            )

        statement_date = None
        if parsed_data.get("statement_date"):
            statement_date = datetime.fromisoformat(parsed_data["statement_date"])

        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_expenses = sum(abs(t.amount) for t in transactions if t.amount < 0)

        return ParsedStatement(
            transactions=transactions,
            statement_date=statement_date,
            total_income=total_income,
            total_expenses=total_expenses,
            confidence=parsed_data.get("confidence", 0.9),
        )

    # JSON schema for structured output — avoids truncated responses
    _RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "statement_date": {"type": "string", "nullable": True},
            "confidence": {"type": "number"},
            "transactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "amount": {"type": "number"},
                        "description": {"type": "string"},
                        "category": {"type": "string"},
                    },
                    "required": ["date", "amount", "description", "category"],
                },
            },
        },
        "required": ["transactions", "confidence"],
    }

    async def parse_statement(self, data: str, mime_type: str) -> ParsedStatement:
        """
        Parse a bank statement by passing the file directly to Gemini.

        Uses structured output (response schema) to prevent truncated JSON.
        Supports PDFs and images natively — no OCR preprocessing.

        Args:
            data: Base64-encoded file data
            mime_type: MIME type (e.g. 'application/pdf', 'image/png')

        Returns:
            ParsedStatement with extracted transactions
        """
        logger.info(f"📄 Parsing bank statement via Gemini multimodal ({mime_type})")

        model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=settings.GEMINI_API_KEY,
            max_tokens=self.max_tokens,
            response_mime_type="application/json",
            response_schema=self._RESPONSE_SCHEMA,
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "media",
                        "mime_type": mime_type,
                        "data": data,
                    },
                    {
                        "type": "text",
                        "text": "Extract all transactions from this bank statement.",
                    },
                ],
            },
        ]

        response = await model.ainvoke(messages)
        content = self.extract_content(response.content if hasattr(response, "content") else str(response))

        try:
            parsed_data = json.loads(content)
            parsed_statement = self._build_parsed_statement(parsed_data)
            logger.info(
                f"✅ Parsed {len(parsed_statement.transactions)} transactions "
                f"(confidence: {parsed_statement.confidence})"
            )
            return parsed_statement
        except Exception as e:
            logger.error(f"❌ Failed to parse Gemini response: {e}")
            logger.debug(f"Response content: {content}")
            raise Exception(f"Failed to parse statement: {e}")

    async def process_message(self, message: MessageInput) -> LLMResponse:
        """
        Process bank statement parsing request.

        Args:
            message: Input message with attachment

        Returns:
            LLM response with HITL flow initiation
        """
        try:
            logger.info(f"📄 Processing bank statement for user {message.user_id}")

            if not message.attachments or len(message.attachments) == 0:
                return self.build_response(
                    message=message,
                    content="❌ No bank statement attached. Please upload an image or PDF of your bank statement.",
                )

            attachment = message.attachments[0]

            try:
                parsed_statement = await self.parse_statement(attachment.data, attachment.mime_type)
            except Exception as e:
                logger.error(f"❌ Failed to parse bank statement: {e}")
                return self.build_response(
                    message=message,
                    content=f"❌ Failed to parse bank statement: {e}",
                    error=str(e),
                )

            if not parsed_statement or not parsed_statement.transactions:
                return self.build_response(
                    message=message,
                    content="❌ No transactions found in the statement. Please check the file and try again.",
                )

            presentation = await self.parsing_flow.initiate_parsing_flow(
                user_id=message.user_id,
                message_id=message.message_id,
                parsed_statement=parsed_statement,
            )

            return self.build_response(
                message=message,
                content=presentation.message,
                hitl_flow_id=presentation.flow_id,
                hitl_required=True,
                hitl_data={
                    "transaction_count": presentation.transaction_count,
                    "total_income": presentation.total_income,
                    "total_expenses": presentation.total_expenses,
                },
            )

        except Exception as e:
            return await self.handle_error(message, e)
