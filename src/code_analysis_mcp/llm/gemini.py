import os
import google.generativeai as genai
import google.api_core.exceptions
import logging
from .base import LLMProvider, LLMProviderError, RateLimitError

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._model = None
        self._embedding_model_name = None

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                model_name = os.getenv(
                    "GENERATION_MODEL_NAME", "models/gemini-2.0-flash-001"
                )
                self._model = genai.GenerativeModel(model_name)
                self._embedding_model_name = os.getenv(
                    "EMBEDDING_MODEL_NAME", "models/embedding-001"
                )
                logger.info(
                    f"GeminiProvider: generation={model_name}, embedding={self._embedding_model_name}"
                )
            except Exception as e:
                logger.error(f"GeminiProvider init failed: {e}")

    def generate(self, prompt: str) -> str:
        if not self._model:
            raise LLMProviderError("Gemini model not initialized")
        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except google.api_core.exceptions.ResourceExhausted as e:
            raise RateLimitError(str(e)) from e
        except google.api_core.exceptions.GoogleAPIError as e:
            raise LLMProviderError(str(e)) from e

    def embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        if not self._embedding_model_name:
            raise LLMProviderError("Gemini embedding model not initialized")
        try:
            result = genai.embed_content(
                model=self._embedding_model_name,
                content=text,
                task_type=task_type,
            )
            return result.get("embedding")
        except google.api_core.exceptions.ResourceExhausted as e:
            raise RateLimitError(str(e)) from e
        except google.api_core.exceptions.GoogleAPIError as e:
            raise LLMProviderError(str(e)) from e

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def is_available(self) -> bool:
        return self._model is not None
