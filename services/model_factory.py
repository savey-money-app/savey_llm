"""
LLM Model Factory

Provider-agnostic factory for creating PydanticAI model instances.
Switch between Gemini and OpenAI by setting the ``LLM_PROVIDER`` env var.

Supported providers:
- ``gemini`` (default) — Google Gemini via ``pydantic-ai[google]``
- ``openai`` — OpenAI (GPT-4o, etc.) via ``pydantic-ai[openai]``
"""

import logging

from pydantic_ai.models import Model

from core.config import settings

logger = logging.getLogger(__name__)


def get_model_name(model_type: str = "main") -> str:
    """
    Return the model name for the configured provider.

    Args:
        model_type: ``"main"`` or ``"vision"``
    """
    if settings.LLM_PROVIDER == "openai":
        return settings.OPENAI_MODEL_MAIN if model_type == "main" else settings.OPENAI_MODEL_VISION
    return settings.GEMINI_MODEL_MAIN if model_type == "main" else settings.GEMINI_MODEL_VISION


def create_openai_model(model_type: str = "main") -> Model:
    """Always create an OpenAI model, regardless of the configured provider."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    name = settings.OPENAI_MODEL_MAIN if model_type == "main" else settings.OPENAI_MODEL_VISION
    return OpenAIChatModel(name, provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY))


def create_gemini_model(model_type: str = "main") -> Model:
    """Always create a Gemini model, regardless of the configured provider.
    Used as the fallback when the primary LLM times out."""
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    name = settings.GEMINI_MODEL_MAIN if model_type == "main" else settings.GEMINI_MODEL_VISION
    return GoogleModel(name, provider=GoogleProvider(api_key=settings.GEMINI_API_KEY))


def create_model(model_type: str = "main", model_name: str | None = None) -> Model:
    """
    Create a PydanticAI Model instance for the configured provider.

    Args:
        model_type: ``"main"`` or ``"vision"`` (ignored when ``model_name`` set).
        model_name: Explicit model name (overrides ``model_type`` lookup).

    Returns:
        A PydanticAI ``Model`` instance (``GoogleModel`` or ``OpenAIChatModel``).
    """
    name = model_name or get_model_name(model_type)
    provider = settings.LLM_PROVIDER

    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(name, provider=OpenAIProvider(api_key=settings.OPENAI_API_KEY))
    else:
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(name, provider=GoogleProvider(api_key=settings.GEMINI_API_KEY))
