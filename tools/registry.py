"""Tool registry -- single entry point for tool definitions and execution"""

import inspect
import logging
from typing import Any, Callable, Dict, List
from uuid import UUID

from pydantic_ai import RunContext
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
        toolset = registry.create_toolset()  # For PydanticAI Agent
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

    def create_toolset(self) -> FunctionToolset:
        """
        Build a PydanticAI ``FunctionToolset`` from all registered tools.

        Each ``BaseTool`` is wrapped in an async function whose signature
        is derived from the tool's ``args_schema`` so PydanticAI can
        generate correct JSON-schema definitions for the LLM.
        """
        toolset: FunctionToolset = FunctionToolset()

        for tool in self._tools.values():
            wrapper = self._make_tool_wrapper(tool)
            toolset.tool(wrapper, name=tool.name, description=tool.description)

        return toolset

    @staticmethod
    def _make_tool_wrapper(tool: BaseTool) -> Callable:
        """
        Create an async wrapper function for a BaseTool that PydanticAI can
        introspect. The wrapper accepts ``RunContext`` as first arg (for deps)
        plus all fields from the tool's ``args_schema`` as keyword arguments.
        """
        schema = tool.args_schema
        fields = schema.model_fields

        # Build parameter list dynamically from the Pydantic model fields
        params: list[inspect.Parameter] = [
            inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
        for field_name, field_info in fields.items():
            default = field_info.default if field_info.default is not None else inspect.Parameter.empty
            if field_info.is_required():
                default = inspect.Parameter.empty
            params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=field_info.annotation,
                )
            )

        async def _wrapper(ctx, **kwargs):
            user_id: UUID = ctx.deps.user_id
            return await tool.execute(user_id, kwargs)

        # Set proper signature so PydanticAI can extract parameter schemas
        _wrapper.__signature__ = inspect.Signature(params)
        _wrapper.__name__ = tool.name
        _wrapper.__doc__ = tool.description
        # Attach annotations for PydanticAI schema generation
        annotations = {"ctx": RunContext}
        for field_name, field_info in fields.items():
            annotations[field_name] = field_info.annotation
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
