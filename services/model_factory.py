"""
LLM Model Factory

Provider-agnostic factory for creating LangChain chat models.
Switch between Gemini and OpenAI by setting the ``LLM_PROVIDER`` env var.

Supported providers:
- ``gemini`` (default) — Google Gemini via ``langchain-google-genai``
- ``openai`` — OpenAI (GPT-4o, etc.) via ``langchain-openai``
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_model_name(model_type: str = "main") -> str:
    """
    Return the model name for the configured provider.

    Args:
        model_type: ``"main"`` or ``"vision"``
    """
    if settings.LLM_PROVIDER == "openai":
        return settings.OPENAI_MODEL_MAIN if model_type == "main" else settings.OPENAI_MODEL_VISION
    return settings.GEMINI_MODEL_MAIN if model_type == "main" else settings.GEMINI_MODEL_VISION


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_chat_model(
    *,
    model_name: Optional[str] = None,
    model_type: str = "main",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    tools: Optional[List[Any]] = None,
) -> BaseChatModel:
    """
    Create a LangChain chat model for the configured provider.

    Args:
        model_name: Explicit model name (overrides ``model_type`` lookup).
        model_type: ``"main"`` or ``"vision"`` (ignored when ``model_name`` set).
        temperature: Sampling temperature (defaults to ``LLM_TEMPERATURE``).
        max_tokens: Max output tokens (defaults to ``MAX_TOKENS``).
        tools: Optional LangChain tools to bind.

    Returns:
        Configured ``BaseChatModel`` instance.
    """
    name = model_name or get_model_name(model_type)
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    tokens = max_tokens or settings.MAX_TOKENS

    provider = settings.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=name,
            temperature=temp,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=tokens,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=name,
            temperature=temp,
            google_api_key=settings.GEMINI_API_KEY,
            max_tokens=tokens,
        )

    if tools:
        model = model.bind_tools(tools)

    return model


def create_structured_model(
    *,
    response_schema: Dict[str, Any],
    model_name: Optional[str] = None,
    model_type: str = "main",
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Create a chat model configured for structured JSON output.

    - **Gemini**: uses native ``response_mime_type`` + ``response_schema``.
    - **OpenAI**: uses ``response_format={"type": "json_object"}``.
      The prompt must explicitly instruct the model to output JSON.

    Args:
        response_schema: JSON schema describing the expected response shape.
        model_name: Explicit model name.
        model_type: ``"main"`` or ``"vision"``.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.

    Returns:
        Model configured for structured JSON output.
    """
    name = model_name or get_model_name(model_type)
    tokens = max_tokens or settings.MAX_TOKENS

    provider = settings.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=name,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=tokens,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=name,
            temperature=temperature,
            google_api_key=settings.GEMINI_API_KEY,
            max_tokens=tokens,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

    return model


def format_multimodal_content(data: str, mime_type: str, text: str) -> list:
    """
    Format multimodal (vision) message content for the configured provider.

    Args:
        data: Base64-encoded file data.
        mime_type: MIME type (e.g. ``image/png``, ``application/pdf``).
        text: Text prompt to accompany the media.

    Returns:
        Provider-appropriate content list for a ``HumanMessage``.

    Note:
        OpenAI vision does not natively support PDFs — only images.
        Gemini supports both images and PDFs natively.
    """
    provider = settings.LLM_PROVIDER

    if provider == "openai":
        data_uri = f"data:{mime_type};base64,{data}"
        return [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": text},
        ]
    else:
        return [
            {"type": "media", "mime_type": mime_type, "data": data},
            {"type": "text", "text": text},
        ]
