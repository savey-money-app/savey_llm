"""Base tool class for function calling"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel


class BaseTool(ABC):
    """Base class for all tools"""

    def __init__(self, name: str, description: str, parameters_schema: type[BaseModel]):
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for LangChain binding"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema.model_json_schema()
        }
