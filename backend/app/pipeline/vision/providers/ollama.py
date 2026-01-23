"""
Ollama vision provider implementation.
Uses native /api/chat endpoint with images field for reliable image processing.
"""

import logging
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger("app.pipeline.vision.ollama")
from app.pipeline.vision.providers.base import VisionProviderBase
from app.pipeline.vision.schemas import ClassificationResult, ExtractionResult


def _load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Looks in the backend/prompts/ directory.
    """
    # Navigate from backend/app/pipeline/vision/providers to backend/
    backend_root = Path(__file__).parent.parent.parent.parent.parent
    prompt_path = backend_root / "prompts" / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _strip_base64_prefix(base64_string: str) -> str:
    """Strip data URI prefix from base64 string if present."""
    if "," in base64_string and base64_string.startswith("data:"):
        return base64_string.split(",", 1)[1]
    return base64_string


class OllamaVisionProvider(VisionProviderBase):
    """Ollama vision provider using native /api/chat endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama API base URL (default: http://localhost:11434/v1)
            model: Model to use (default: ministral-3:latest)
        """
        configured_url = base_url or settings.ollama_base_url
        self.model = model or settings.vision_model

        # Derive native API base URL by stripping /v1 suffix
        # e.g., http://192.168.10.3:11436/v1 -> http://192.168.10.3:11436
        self.base_url = configured_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]

        # Create async HTTP client for native Ollama API
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=settings.api_request_timeout,
        )

        # Load prompts lazily
        self._classifier_prompt: Optional[str] = None
        self._extract_prompt: Optional[str] = None

    @property
    def classifier_prompt(self) -> str:
        """Load classifier prompt lazily."""
        if self._classifier_prompt is None:
            self._classifier_prompt = _load_prompt("vision-classifier.txt")
        return self._classifier_prompt

    @property
    def extract_prompt(self) -> str:
        """Load extraction prompt lazily.

        Uses simplified prompt (vision-extract-ollama.txt) to avoid
        overwhelming the model's vision attention with long examples.
        """
        if self._extract_prompt is None:
            self._extract_prompt = _load_prompt("vision-extract-ollama.txt")
        return self._extract_prompt

    async def classify(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ClassificationResult:
        """Classify whether an image contains graphical content.

        Uses native Ollama /api/chat with images field.
        """
        image_data = _strip_base64_prefix(image_base64)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self.classifier_prompt,
                    "images": [image_data],
                }
            ],
            "stream": False,
            "options": {
                "num_predict": 10,  # Only need YES/NO
            },
        }

        for attempt in range(settings.api_max_retries):
            try:
                response = await self._client.post("/api/chat", json=payload)
                response.raise_for_status()

                data = response.json()
                result_text = data["message"]["content"].strip().upper()

                has_graphics = result_text.startswith("YES")

                # Native API token fields
                prompt_tokens = data.get("prompt_eval_count", 0) or 0
                completion_tokens = data.get("eval_count", 0) or 0
                total_tokens = prompt_tokens + completion_tokens

                return ClassificationResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    has_graphics=has_graphics,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except httpx.ConnectError as e:
                if attempt == settings.api_max_retries - 1:
                    return ClassificationResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        has_graphics=True,
                        error=f"Connection error: {str(e)}",
                    )
            except Exception as e:
                if attempt == settings.api_max_retries - 1:
                    return ClassificationResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        has_graphics=True,
                        error=f"Error: {str(e)}",
                    )

        return ClassificationResult(
            page_number=page_number,
            chunk_id=chunk_id,
            has_graphics=True,
            error="Max retries exceeded",
        )

    async def extract(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ExtractionResult:
        """Extract visual content from an image using native Ollama API."""
        image_data = _strip_base64_prefix(image_base64)

        logger.info(
            "[OLLAMA_EXTRACT_START] page=%d, model=%s, base_url=%s, image_base64_len=%d, prompt_len=%d",
            page_number, self.model, self.base_url, len(image_data), len(self.extract_prompt),
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self.extract_prompt,
                    "images": [image_data],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4000,
                "seed": 42,
            },
        }

        for attempt in range(settings.api_max_retries):
            try:
                response = await self._client.post("/api/chat", json=payload)
                response.raise_for_status()

                data = response.json()
                image_text = data["message"]["content"]

                # Native API token fields
                prompt_tokens = data.get("prompt_eval_count", 0) or 0
                completion_tokens = data.get("eval_count", 0) or 0
                total_tokens = prompt_tokens + completion_tokens

                logger.info(
                    "[OLLAMA_EXTRACT_COMPLETE] page=%d, prompt_tokens=%d, completion_tokens=%d, response_len=%d",
                    page_number, prompt_tokens, completion_tokens, len(image_text) if image_text else 0,
                )
                logger.debug("[OLLAMA_EXTRACT_RESPONSE] page=%d, response=%s", page_number, image_text)

                return ExtractionResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    image_text=image_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except httpx.ConnectError as e:
                if attempt == settings.api_max_retries - 1:
                    return ExtractionResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        image_text=None,
                        error=f"Connection error: {str(e)}",
                    )
            except Exception as e:
                if attempt == settings.api_max_retries - 1:
                    return ExtractionResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        image_text=None,
                        error=f"Error: {str(e)}",
                    )

        return ExtractionResult(
            page_number=page_number,
            chunk_id=chunk_id,
            image_text=None,
            error="Max retries exceeded",
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
