"""
Prompt Manager

Loads system prompts from .txt files in the prompts/ directory.
Prompts are cached in memory after the first load.
"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# Default prompts directory: savey_llm/prompts/
_DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class PromptManager:
    """Loads and caches system prompts from the prompts/ directory."""

    def __init__(self, prompts_dir: Path = _DEFAULT_PROMPTS_DIR):
        self._dir = prompts_dir
        self._cache: Dict[str, str] = {}

    def get(self, name: str) -> str:
        """
        Return the content of prompts/{name}.txt.

        The file is read once and cached for subsequent calls.

        Args:
            name: Prompt name (without .txt extension)

        Returns:
            Prompt text

        Raises:
            FileNotFoundError: If the prompt file does not exist
        """
        if name not in self._cache:
            path = self._dir / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            self._cache[name] = path.read_text(encoding="utf-8").strip()
            logger.debug(f"Loaded prompt '{name}' from {path}")
        return self._cache[name]

    def reload(self, name: str) -> str:
        """Force reload a prompt from disk, bypassing the cache."""
        self._cache.pop(name, None)
        return self.get(name)


# Module-level singleton used by agents
prompt_manager = PromptManager()
