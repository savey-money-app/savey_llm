"""Base tool class for LLM function calling"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from uuid import UUID

from pydantic import BaseModel

from services.api_client import APIClient

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Each tool bundles its parameter schema and execution logic in one place.
    Tool registration for PydanticAI is handled by the ToolRegistry.
    """

    name: str
    description: str
    args_schema: Type[BaseModel]

    def __init__(self, api_client: APIClient):
        self.api_client = api_client

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
