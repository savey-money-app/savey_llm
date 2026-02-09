"""Base tool class for LLM function calling"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from uuid import UUID

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from services.api_client import APIClient

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Each tool bundles its parameter schema, LangChain definition, and
    execution logic in one place.
    """

    name: str
    description: str
    args_schema: Type[BaseModel]

    def __init__(self, api_client: APIClient):
        self.api_client = api_client

    def definition(self) -> StructuredTool:
        """
        Return a LangChain StructuredTool suitable for ``bind_tools``.

        The underlying function is a no-op placeholder; actual execution
        goes through :meth:`execute`.
        """

        def _placeholder(**kwargs: Any) -> None:  # noqa: ARG001
            pass

        return StructuredTool.from_function(
            func=_placeholder,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    @abstractmethod
    async def execute(self, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with the given arguments.

        Args:
            user_id: Authenticated user UUID.
            arguments: Tool arguments as parsed by the LLM.

        Returns:
            Result dict that will be fed back into the LLM context.
        """
        ...
