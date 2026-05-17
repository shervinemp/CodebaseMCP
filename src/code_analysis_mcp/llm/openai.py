import os
import logging
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from openai import APIError as OpenAIAPIError
from .base import LLMProvider, LLMProviderError, RateLimitError

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._model_name = None
        self._embedding_model_name = None

        if self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key)
                self._model_name = os.getenv("GENERATION_MODEL_NAME", "gpt-4o")
                self._embedding_model_name = os.getenv(
                    "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
                )
                logger.info(
                    f"OpenAIProvider: generation={self._model_name}, embedding={self._embedding_model_name}"
                )
            except Exception as e:
                logger.error(f"OpenAIProvider init failed: {e}")

    def generate(self, prompt: str) -> str:
        if not self._client:
            raise LLMProviderError("OpenAI client not initialized")
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e)) from e
        except OpenAIAPIError as e:
            raise LLMProviderError(str(e)) from e

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        if not self._client:
            raise LLMProviderError("OpenAI client not initialized")
        try:
            response = self._client.embeddings.create(
                model=self._embedding_model_name,
                input=text,
            )
            return response.data[0].embedding
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e)) from e
        except OpenAIAPIError as e:
            raise LLMProviderError(str(e)) from e

    @property
    def name(self) -> str:
        return "openai"

    @property
    def is_available(self) -> bool:
        return self._client is not None
