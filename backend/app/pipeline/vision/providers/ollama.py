"""
Ollama vision provider implementation.
Uses OpenAI-compatible /v1/chat/completions endpoint.
"""

from pathlib import Path
from typing import Optional

import httpx

from app.config import settings
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


def _prepare_image_data_uri(base64_string: str) -> str:
    """Prepare base64 string for OpenAI-compatible API."""
    # Remove existing data URI prefix if present
    if "," in base64_string and base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    # Return with proper data URI prefix
    return f"data:image/jpeg;base64,{base64_string}"


class OllamaVisionProvider(VisionProviderBase):
    """Ollama vision provider using OpenAI-compatible API."""

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
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.vision_model

        # Create async HTTP client
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
        """Load extraction prompt lazily."""
        if self._extract_prompt is None:
            self._extract_prompt = _load_prompt("vision-extract.txt")
        return self._extract_prompt

    async def classify(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ClassificationResult:
        """Classify whether an image contains graphical content.

        Uses a quick YES/NO prompt to determine if visual elements exist.
        """
        image_data_uri = _prepare_image_data_uri(image_base64)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        },
                        {"type": "text", "text": self.classifier_prompt},
                    ],
                }
            ],
            "max_tokens": 10,  # Only need YES/NO
        }

        for attempt in range(settings.api_max_retries):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                response.raise_for_status()

                data = response.json()
                result_text = data["choices"][0]["message"]["content"].strip().upper()

                has_graphics = result_text.startswith("YES")

                # Extract token usage if available
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0
                total_tokens = usage.get("total_tokens", 0) or 0

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
                    # Default to True on connection error to avoid missing content
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

        # Should not reach here, but default to True for safety
        return ClassificationResult(
            page_number=page_number,
            chunk_id=chunk_id,
            has_graphics=True,
            error="Max retries exceeded",
        )

    async def extract(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ExtractionResult:
        """Extract visual content from an image."""
        image_data_uri = _prepare_image_data_uri(image_base64)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri},
                        },
                        {"type": "text", "text": self.extract_prompt},
                    ],
                }
            ],
            "temperature": 0.1,  # Low temp for factual extraction
            "max_tokens": 4000,  # Sufficient for detailed structured output
            "seed": 42,  # Reproducible results
        }

        for attempt in range(settings.api_max_retries):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                response.raise_for_status()

                data = response.json()
                image_text = data["choices"][0]["message"]["content"]

                # Extract token usage if available
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0
                total_tokens = usage.get("total_tokens", 0) or 0

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
