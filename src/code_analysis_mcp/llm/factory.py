import os
import logging
from .base import LLMProvider

logger = logging.getLogger(__name__)

_provider: LLMProvider | None = None


def create_llm_provider() -> LLMProvider | None:
    provider_type = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider_type == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider()
    elif provider_type == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider()
    else:
        logger.warning(
            f"Unknown LLM_PROVIDER '{provider_type}'. Falling back to gemini."
        )
        from .gemini import GeminiProvider

        return GeminiProvider()


def init_llm_provider():
    global _provider
    _provider = create_llm_provider()
    if _provider and _provider.is_available:
        logger.info(f"LLM provider initialized: {_provider.name}")
    else:
        logger.warning("No LLM provider available. LLM features disabled.")


def get_llm_provider() -> LLMProvider | None:
    global _provider
    if _provider is None:
        init_llm_provider()
    return _provider
