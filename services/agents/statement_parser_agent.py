"""
Bank Statement Parser Agent

Specialized agent using GPT-4o vision for parsing bank statements from images/PDFs.
Extracts transactions and initiates HITL flow for user confirmation.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from core.config import settings
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from schemas.bank_statement import ParsedStatement, StatementParsingResponse
from schemas.api_tools import TransactionCreateShort
from schemas.message import MessageInput
from schemas.response import LLMResponse
from services.agents.base_agent import BaseAgent
from services.hitl_flows.statement_parsing import StatementParsingFlow
from services.hitl_manager import HITLManager
from services.api_client import APIClient
from services.ocr_service import OCRService

logger = logging.getLogger(__name__)


class StatementParserAgent(BaseAgent):
    """Agent for parsing bank statements using GPT-4o vision"""

    def __init__(self):
        super().__init__(
            model_name=settings.GEMINI_MODEL_VISION,
            temperature=0.3,  # Lower temperature for more deterministic parsing
            max_tokens=settings.MAX_TOKENS_VISION,
        )
        self.ocr_service = OCRService()
        self.api_client = APIClient()
        self.hitl_manager = HITLManager()
        self.parsing_flow = StatementParsingFlow(self.hitl_manager, self.api_client)

    def get_agent_name(self) -> str:
        return "statement_parser"

    def get_system_prompt(self) -> str:
        return """You are a specialized bank statement parser for Savey money tracking app.

Your task is to extract transaction data from bank statements (images or PDFs).

Extract the following for each transaction:
1. **date**: Transaction date (ISO format: YYYY-MM-DD)
2. **amount**: Transaction amount (float, negative for expenses, positive for income)
3. **description**: Transaction description/merchant name
4. **category**: Inferred category (e.g., Food, Transport, Shopping, Salary, etc.)
5. **mcc**: Merchant Category Code if visible (optional)

Guidelines:
- Identify the statement date/period
- Extract ALL transactions visible in the statement
- Infer transaction type from amount (debits = expenses = negative, credits = income = positive)
- Categorize intelligently based on merchant name and description
- If amount has a debit/credit indicator, respect it
- Return transactions in chronological order
- Skip header rows, totals, and balances

Output format:
Return a JSON object with:
```json
{
  "statement_date": "YYYY-MM-DD",
  "transactions": [
    {
      "date": "YYYY-MM-DD",
      "amount": -45.50,
      "description": "Starbucks Coffee",
      "category": "Food & Dining"
    },
    ...
  ],
  "confidence": 0.95
}
```

Be accurate and thorough. If text is unclear, use OCR context provided."""

    async def parse_statement_with_vision(
        self, image_data: str, mime_type: str, ocr_text: Optional[str] = None
    ) -> ParsedStatement:
        """
        Parse bank statement using GPT-4o vision

        Args:
            image_data: Base64-encoded image data
            mime_type: MIME type of the image
            ocr_text: Optional OCR-extracted text for context

        Returns:
            ParsedStatement with extracted transactions
        """
        logger.info("🔍 Parsing bank statement with GPT-4o vision")

        # Prepare vision message
        data_url = f"data:{mime_type};base64,{image_data}"

        message_content = [
            {
                "type": "text",
                "text": "Parse this bank statement and extract all transactions in JSON format as specified in the system prompt.",
            },
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ]

        # Add OCR context if available
        if ocr_text:
            message_content.append(
                {
                    "type": "text",
                    "text": f"OCR Text Context (for reference):\n```\n{ocr_text[:2000]}\n```",
                }
            )

        # Initialize vision model (without tools)
        vision_model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=settings.GEMINI_API_KEY,
            max_tokens=self.max_tokens,
        )

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": message_content},
        ]

        # Invoke vision model
        response = await vision_model.ainvoke(messages)

        # Parse JSON response
        try:
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed_data = json.loads(content)

            # Convert to TransactionCreateShort objects
            transactions = []
            for t in parsed_data.get("transactions", []):
                # Parse date
                date = datetime.fromisoformat(t["date"])

                # Determine transaction type from amount
                amount = float(t["amount"])
                transaction_type = "income" if amount > 0 else "expense"

                transactions.append(
                    TransactionCreateShort(
                        amount=amount,
                        category=t.get("category", "Uncategorized"),
                        description=t.get("description", ""),
                        date=date,
                        mcc=t.get("mcc"),
                    )
                )

            # Parse statement date
            statement_date = None
            if parsed_data.get("statement_date"):
                statement_date = datetime.fromisoformat(parsed_data["statement_date"])

            # Calculate totals
            total_income = sum(t.amount for t in transactions if t.amount > 0)
            total_expenses = sum(abs(t.amount) for t in transactions if t.amount < 0)

            parsed_statement = ParsedStatement(
                transactions=transactions,
                statement_date=statement_date,
                total_income=total_income,
                total_expenses=total_expenses,
                raw_text=ocr_text,
                confidence=parsed_data.get("confidence", 0.9),
            )

            logger.info(
                f"✅ Parsed {len(transactions)} transactions from statement (confidence: {parsed_statement.confidence})"
            )
            return parsed_statement

        except Exception as e:
            logger.error(f"❌ Failed to parse vision model response: {e}")
            logger.debug(f"Response content: {content}")
            raise Exception(f"Failed to parse statement: {e}")

    async def parse_statement_with_ocr(self, data: str, mime_type: str) -> ParsedStatement:
        """
        Parse bank statement using OCR text extraction + LLM

        Args:
            data: Base64-encoded file data
            mime_type: MIME type

        Returns:
            ParsedStatement with extracted transactions
        """
        logger.info("📄 Parsing bank statement with OCR + LLM")

        # Extract text via OCR
        ocr_text = await self.ocr_service.extract_text(mime_type, data)

        if not ocr_text or len(ocr_text.strip()) < 50:
            raise Exception("Failed to extract meaningful text from statement")

        # Use LLM to parse the OCR text
        model = self.initialize_model()

        parsing_prompt = f"""Parse the following bank statement text and extract all transactions.

Bank Statement Text:
```
{ocr_text}
```

Extract all transactions in the JSON format specified in the system prompt."""

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": parsing_prompt},
        ]

        response = await model.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse JSON (same logic as vision parsing)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed_data = json.loads(content)

        # Convert to TransactionCreateShort objects (same as vision method)
        transactions = []
        for t in parsed_data.get("transactions", []):
            date = datetime.fromisoformat(t["date"])
            amount = float(t["amount"])

            transactions.append(
                TransactionCreateShort(
                    amount=amount,
                    category=t.get("category", "Uncategorized"),
                    description=t.get("description", ""),
                    date=date,
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
            raw_text=ocr_text,
            confidence=parsed_data.get("confidence", 0.85),
        )

    async def process_message(self, message: MessageInput) -> LLMResponse:
        """
        Process bank statement parsing request

        Args:
            message: Input message with attachment

        Returns:
            LLM response with HITL flow initiation
        """
        try:
            logger.info(f"📄 Processing bank statement for user {message.user_id}")

            # Check for attachments
            if not message.attachments or len(message.attachments) == 0:
                return self.build_response(
                    message=message,
                    content="❌ No bank statement attached. Please upload an image or PDF of your bank statement.",
                )

            attachment = message.attachments[0]  # Use first attachment

            # Parse statement (try vision first, fall back to OCR)
            parsed_statement = None
            try:
                if attachment.mime_type in ["image/png", "image/jpeg", "image/jpg"]:
                    # Try vision first for images
                    parsed_statement = await self.parse_statement_with_vision(
                        attachment.data, attachment.mime_type
                    )
                else:
                    # Use OCR for PDFs
                    parsed_statement = await self.parse_statement_with_ocr(
                        attachment.data, attachment.mime_type
                    )
            except Exception as vision_error:
                logger.warning(f"Vision/OCR parsing failed, trying alternative: {vision_error}")
                # Fallback to OCR if vision fails
                try:
                    parsed_statement = await self.parse_statement_with_ocr(
                        attachment.data, attachment.mime_type
                    )
                except Exception as ocr_error:
                    logger.error(f"Both parsing methods failed: {ocr_error}")
                    return self.build_response(
                        message=message,
                        content=f"❌ Failed to parse bank statement: {ocr_error}",
                        error=str(ocr_error),
                    )

            if not parsed_statement or not parsed_statement.transactions:
                return self.build_response(
                    message=message,
                    content="❌ No transactions found in the statement. Please check the file and try again.",
                )

            # Initiate HITL flow for confirmation
            presentation = await self.parsing_flow.initiate_parsing_flow(
                user_id=message.user_id, message_id=message.message_id, parsed_statement=parsed_statement
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
