import os
from dataclasses import dataclass


class ConfigurationError(RuntimeError):
    """Raised when the LLM provider is not configured for a run."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model_name: str


def get_llm_settings() -> LLMSettings:
    """Read LLM settings lazily so the FastAPI app can expose health checks without a key."""
    api_key = os.getenv("MODEL_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_or_gemini_api_key_here":
        raise ConfigurationError(
            "MODEL_API_KEY is not configured. Create .env from .env.example and set a valid API key."
        )

    return LLMSettings(
        api_key=api_key,
        base_url=os.getenv(
            "BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        ).strip(),
        model_name=os.getenv("NAME_MODEL", "gemini-2.5-flash").strip(),
    )


def llm_is_configured() -> bool:
    try:
        get_llm_settings()
        return True
    except ConfigurationError:
        return False
