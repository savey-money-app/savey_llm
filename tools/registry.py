"""Tool registry -- single entry point for tool definitions and execution"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from services.api_client import APIClient
from tools.base import BaseTool
from tools.mcc_tool import MccLookupTool
from tools.transaction_tools import (
    DeleteLastStatementTransactionsTool,
    DeleteLastTransactionTool,
    DeleteTransactionTool,
    GetUserTransactionsTool,
    SaveTransactionTool,
)

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry that owns every tool instance.

    Usage::

        registry = ToolRegistry(api_client)
        model.bind_tools(registry.get_definitions())
        result = await registry.execute("save_transaction", user_id, args)
    """

    def __init__(self, api_client: APIClient):
        self._tools: Dict[str, BaseTool] = {}
        self._register_all(api_client)

    def _register_all(self, api_client: APIClient) -> None:
        """Instantiate and register every tool."""
        tool_classes = [
            SaveTransactionTool,
            GetUserTransactionsTool,
            DeleteTransactionTool,
            DeleteLastTransactionTool,
            DeleteLastStatementTransactionsTool,
            MccLookupTool,
        ]
        for cls in tool_classes:
            tool = cls(api_client)
            self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_definitions(self) -> List[Any]:
        """Return LangChain-compatible tool definitions for ``bind_tools``."""
        return [tool.definition() for tool in self._tools.values()]

    def get_tool(self, name: str) -> BaseTool:
        """Look up a tool by name. Raises ``KeyError`` if not found."""
        return self._tools[name]

    async def execute(self, name: str, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Look up a tool by *name* and execute it.

        Returns the result dict on success, or an error dict on failure.
        """
        try:
            tool = self._tools[name]
        except KeyError:
            raise ValueError(f"Unknown tool: {name}")

        try:
            return await tool.execute(user_id, arguments)
        except Exception as e:
            logger.error(f"Tool execution failed ({name}): {e}")
            return {"error": str(e), "success": False}
