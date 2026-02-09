"""Schemas for savey_llm service"""
from schemas.message import MessageInput
from schemas.response import LLMResponse, ToolCall
from schemas.api_tools import (
    TransactionCreateShort,
    UserBalance,
    TransactionRead,
    CategoryResponse,
)
from schemas.bank_statement import (
    BankStatement,
    ParsedStatement,
    StatementParsingRequest,
    StatementParsingResponse,
)
from schemas.hitl import (
    HITLFlowType,
    HITLFlowState,
    HITLRequest,
    HITLResponse,
    HITLUserResponse,
    TransactionDeletionFlowData,
    TransactionDeletionResponse,
    StatementParsingFlowData,
    StatementParsingConfirmation,
    StatementParsingPresentationList,
)

__all__ = [
    # Message and Response
    "MessageInput",
    "LLMResponse",
    "ToolCall",
    # API Schemas
    "TransactionCreateShort",
    "UserBalance",
    "TransactionRead",
    "CategoryResponse",
    # Bank Statement
    "BankStatement",
    "ParsedStatement",
    "StatementParsingRequest",
    "StatementParsingResponse",
    # HITL
    "HITLFlowType",
    "HITLFlowState",
    "HITLRequest",
    "HITLResponse",
    "HITLUserResponse",
    "TransactionDeletionFlowData",
    "TransactionDeletionResponse",
    "StatementParsingFlowData",
    "StatementParsingConfirmation",
    "StatementParsingPresentationList",
]
