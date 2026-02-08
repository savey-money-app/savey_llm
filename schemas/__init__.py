"""Schemas for savey_llm service"""
from schemas.message import MessageInput
from schemas.response import LLMResponse, ToolCall
from schemas.api_tools import (
    SaveTransactionTool,
    DeleteTransactionTool,
    GetUserTransactionsTool,
    DeleteLastTransactionTool,
    DeleteLastStatementTransactionsTool,
    CreateTransactionsFromStatementTool,
    MCCLookupTool,
    TransactionCreateShort,
    UserBalance,
    TransactionRead,
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
    # API Tools
    "SaveTransactionTool",
    "DeleteTransactionTool",
    "GetUserTransactionsTool",
    "DeleteLastTransactionTool",
    "DeleteLastStatementTransactionsTool",
    "CreateTransactionsFromStatementTool",
    "MCCLookupTool",
    "TransactionCreateShort",
    "UserBalance",
    "TransactionRead",
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
