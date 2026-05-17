from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""


class RateLimitError(LLMProviderError):
    """Raised when the provider rate-limits the request."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    @abstractmethod
    def embed(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[float]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_available(self) -> bool: ...
