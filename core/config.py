"""Configuration settings for savey_llm service"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis Configuration
    REDIS_URL: str = "redis://redis:6379"
    REDIS_CHANNEL_INPUT: str = "llm:messages:input"
    REDIS_CHANNEL_OUTPUT: str = "llm:messages:output"
    REDIS_HITL_PREFIX: str = "hitl:flow:"  # Prefix for HITL flow state keys

    # Google Gemini Configuration
    GEMINI_API_KEY: str
    # Main conversational agent (1M context, multimodal)
    GEMINI_MODEL_MAIN: str = "gemini-3-flash-preview"
    # Vision agent - same model (Gemini 2.0 Flash is natively multimodal)
    GEMINI_MODEL_VISION: str = "gemini-3-flash-preview"
    GEMINI_TEMPERATURE: float = 0.7

    # LLM Settings
    MAX_TOKENS: int = 2000
    MAX_TOKENS_VISION: int = 4000  # Higher limit for vision tasks
    ENABLE_FUNCTION_CALLING: bool = True

    # API Client Settings
    SAVEY_API_URL: str = "http://savey_api:8000"
    API_TIMEOUT: int = 30  # Timeout for API calls in seconds
    INTERNAL_API_TOKEN: str = "change-me-internal-secret"

    # HITL (Human-in-the-Loop) Settings
    HITL_FLOW_TTL: int = 3600  # TTL for HITL flows in Redis (1 hour)
    HITL_MAX_ITERATIONS: int = 5  # Max iterations for HITL confirmation loops

    # OCR Settings
    OCR_SUPPORTED_MIME_TYPES: list[str] = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "application/pdf",
    ]
    OCR_MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB max file size

    # Agent Settings
    DEFAULT_AGENT: str = "main"  # Default agent to use
    ENABLE_MULTI_AGENT: bool = True  # Enable multi-agent routing

    # Service Settings
    LOG_LEVEL: str = "INFO"

    DOCKER_CONFIG: str = "docker"
    APP_NAME: str = "savey_llm"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
