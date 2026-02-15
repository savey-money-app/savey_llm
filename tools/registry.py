"""Tool registry -- single entry point for tool definitions and execution"""

import inspect
import logging
from typing import Any, Callable, Dict, List
from uuid import UUID

from pydantic_ai.toolsets.function import FunctionToolset

from services.api_client import APIClient
from tools.base import BaseTool
from tools.currency_tool import CurrencyConverterTool
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
        toolset = registry.create_toolset(user_id)  # For PydanticAI Agent
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
            CurrencyConverterTool,
        ]
        for cls in tool_classes:
            tool = cls(api_client)
            self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_toolset(self, user_id: UUID) -> FunctionToolset:
        """
        Build a PydanticAI ``FunctionToolset`` from all registered tools.

        The ``user_id`` is captured in each wrapper closure so the tools
        don't need ``RunContext`` — this avoids introspection issues with
        dynamically-generated signatures.
        """
        toolset: FunctionToolset = FunctionToolset()

        for tool in self._tools.values():
            wrapper = self._make_tool_wrapper(tool, user_id)
            toolset.tool(wrapper, name=tool.name, description=tool.description)

        return toolset

    @staticmethod
    def _make_tool_wrapper(tool: BaseTool, user_id: UUID) -> Callable:
        """
        Create an async wrapper function for a BaseTool that PydanticAI can
        introspect.  The ``user_id`` is baked into the closure; the wrapper's
        visible parameters come from the tool's ``args_schema``.
        """
        schema = tool.args_schema
        fields = schema.model_fields

        # Build parameter list dynamically from the Pydantic model fields
        params: list[inspect.Parameter] = []
        annotations: dict = {}

        for field_name, field_info in fields.items():
            default = inspect.Parameter.empty
            if not field_info.is_required():
                default = field_info.default
            params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=field_info.annotation,
                )
            )
            annotations[field_name] = field_info.annotation

        async def _wrapper(**kwargs):
            return await tool.execute(user_id, kwargs)

        # Set proper signature so PydanticAI can extract parameter schemas
        _wrapper.__signature__ = inspect.Signature(params)
        _wrapper.__name__ = tool.name
        _wrapper.__doc__ = tool.description
        _wrapper.__annotations__ = annotations

        return _wrapper

    def get_tool(self, name: str) -> BaseTool:
        """Look up a tool by name. Raises ``KeyError`` if not found."""
        return self._tools[name]

    def get_tool_names(self) -> List[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

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
