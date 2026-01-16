"""
Google Gemini vision provider implementation.
Uses google-genai SDK with native async support.
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted

from app.config import settings

logger = logging.getLogger(__name__)
from app.pipeline.vision.providers.base import VisionProviderBase
from app.pipeline.vision.schemas import ClassificationResult, ExtractionResult


def _load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the backend/prompts directory."""
    backend_root = Path(__file__).parent.parent.parent.parent.parent
    prompt_path = backend_root / "prompts" / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _decode_base64_to_bytes(base64_string: str) -> bytes:
    """Decode base64 string to bytes, handling data URI format."""
    if "," in base64_string and base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]
    return base64.b64decode(base64_string)


class GeminiVisionProvider(VisionProviderBase):
    """Google Gemini vision provider using native async SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize Gemini provider.

        Args:
            api_key: Gemini API key (default: from GEMINI_API_KEY env)
            model: Model to use (default: gemini-2.5-flash)
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_vision_model

        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY env var.")

        self._client = genai.Client(api_key=self.api_key)

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
        """Classify whether an image contains graphical content."""
        for attempt in range(settings.api_max_retries):
            try:
                image_bytes = _decode_base64_to_bytes(image_base64)
                image_part = types.Part.from_bytes(
                    data=image_bytes, mime_type="image/jpeg"
                )

                # Per Gemini docs: image must come BEFORE text prompt
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=[image_part, self.classifier_prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=10,
                    ),
                )

                result_text = response.text.strip().upper() if response.text else ""

                # Handle empty response - retry instead of defaulting to False
                if not result_text:
                    logger.warning(
                        "Empty response from Gemini classify (page=%d, attempt=%d/%d)",
                        page_number,
                        attempt + 1,
                        settings.api_max_retries,
                    )
                    if attempt < settings.api_max_retries - 1:
                        await asyncio.sleep(settings.api_retry_delay)
                        continue
                    return ClassificationResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        has_graphics=True,  # Conservative default
                        error="Empty response from API",
                    )

                has_graphics = result_text.startswith("YES")

                # Extract token usage from response
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ) or 0
                    completion_tokens = getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ) or 0
                    total_tokens = getattr(
                        response.usage_metadata, "total_token_count", 0
                    ) or 0

                return ClassificationResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    has_graphics=has_graphics,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except ResourceExhausted as e:
                logger.warning(
                    "Gemini rate limit hit (page=%d, attempt=%d/%d): %s",
                    page_number,
                    attempt + 1,
                    settings.api_max_retries,
                    str(e),
                )
                if attempt < settings.api_max_retries - 1:
                    await asyncio.sleep(settings.api_retry_delay * 2)
                    continue
                return ClassificationResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    has_graphics=True,
                    error=f"Rate limit exceeded: {str(e)}",
                )

            except GoogleAPICallError as e:
                logger.warning(
                    "Gemini API error (page=%d, attempt=%d/%d): %s",
                    page_number,
                    attempt + 1,
                    settings.api_max_retries,
                    str(e),
                )
                if attempt < settings.api_max_retries - 1:
                    await asyncio.sleep(settings.api_retry_delay)
                    continue
                return ClassificationResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    has_graphics=True,
                    error=f"API error: {str(e)}",
                )

            except Exception as e:
                logger.error(
                    "Gemini classify unexpected error (page=%d, attempt=%d/%d): %s",
                    page_number,
                    attempt + 1,
                    settings.api_max_retries,
                    str(e),
                )
                if attempt < settings.api_max_retries - 1:
                    await asyncio.sleep(settings.api_retry_delay)
                    continue
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
        """Extract visual content from an image."""
        for attempt in range(settings.api_max_retries):
            try:
                image_bytes = _decode_base64_to_bytes(image_base64)
                image_part = types.Part.from_bytes(
                    data=image_bytes, mime_type="image/jpeg"
                )

                # Per Gemini docs: image must come BEFORE text prompt
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=[image_part, self.extract_prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=4000,
                    ),
                )

                image_text = response.text if response.text else None

                # Handle empty response - retry
                if not image_text:
                    logger.warning(
                        "Empty response from Gemini extract (page=%d, attempt=%d/%d)",
                        page_number,
                        attempt + 1,
                        settings.api_max_retries,
                    )
                    if attempt < settings.api_max_retries - 1:
                        await asyncio.sleep(settings.api_retry_delay)
                        continue
                    return ExtractionResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        image_text=None,
                        error="Empty response from API",
                    )

                # Extract token usage from response
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ) or 0
                    completion_tokens = getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ) or 0
                    total_tokens = getattr(
                        response.usage_metadata, "total_token_count", 0
                    ) or 0

                return ExtractionResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    image_text=image_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except ResourceExhausted as e:
                logger.warning(
                    "Gemini rate limit hit (page=%d, attempt=%d/%d): %s",
                    page_number,
                    attempt + 1,
                    settings.api_max_retries,
                    str(e),
                )
                if attempt < settings.api_max_retries - 1:
                    await asyncio.sleep(settings.api_retry_delay * 2)
                    continue
                return ExtractionResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    image_text=None,
                    error=f"Rate limit exceeded: {str(e)}",
                )

            except GoogleAPICallError as e:
                logger.warning(
                    "Gemini API error (page=%d, attempt=%d/%d): %s",
                    page_number,
                    attempt + 1,
                    settings.api_max_retries,
                    str(e),
                )
                if attempt < settings.api_max_retries - 1:
                    await asyncio.sleep(settings.api_retry_delay)
                    continue
                return ExtractionResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    image_text=None,
                    error=f"API error: {str(e)}",
                )

            except Exception as e:
                logger.error(
                    "Gemini extract unexpected error (page=%d, attempt=%d/%d): %s",
                    page_number,
                    attempt + 1,
                    settings.api_max_retries,
                    str(e),
                )
                if attempt < settings.api_max_retries - 1:
                    await asyncio.sleep(settings.api_retry_delay)
                    continue
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
