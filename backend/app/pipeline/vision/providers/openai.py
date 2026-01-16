"""
OpenAI vision provider implementation.
Uses official OpenAI Python SDK.
"""

import asyncio
from pathlib import Path
from typing import Optional

from openai import OpenAI, APIError, OpenAIError

from app.config import settings
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


def _prepare_image_data_uri(base64_string: str) -> str:
    """Prepare base64 string for OpenAI API."""
    if "," in base64_string and base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]
    return f"data:image/jpeg;base64,{base64_string}"


class OpenAIVisionProvider(VisionProviderBase):
    """OpenAI vision provider using official SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env)
            model: Model to use (default: gpt-4.1-mini)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_vision_model

        # Create sync client (we wrap calls with asyncio.to_thread)
        self._client = OpenAI(api_key=self.api_key)

        # Load prompts lazily
        self._classifier_prompt: Optional[str] = None
        self._extract_prompt: Optional[str] = None
        self._synthesis_prompt: Optional[str] = None

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

    @property
    def synthesis_prompt(self) -> str:
        """Load synthesis prompt lazily."""
        if self._synthesis_prompt is None:
            self._synthesis_prompt = _load_prompt("vision-synthesis.txt")
        return self._synthesis_prompt

    def _sync_classify(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ClassificationResult:
        """Synchronous classify implementation."""
        image_data_uri = _prepare_image_data_uri(image_base64)

        for attempt in range(settings.api_max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.classifier_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_uri,
                                        "detail": "low",  # Low detail for classification
                                    },
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=10,
                )

                result_text = response.choices[0].message.content.strip().upper()
                has_graphics = result_text.startswith("YES")

                # Extract token usage if available
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                    total_tokens = getattr(response.usage, "total_tokens", 0) or 0

                return ClassificationResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    has_graphics=has_graphics,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except (APIError, OpenAIError) as e:
                if attempt == settings.api_max_retries - 1:
                    return ClassificationResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        has_graphics=True,
                        error=f"API error: {str(e)}",
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

    def _sync_extract(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ExtractionResult:
        """Synchronous extract implementation."""
        image_data_uri = _prepare_image_data_uri(image_base64)

        for attempt in range(settings.api_max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.extract_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_uri,
                                        "detail": "high",  # High detail for extraction
                                    },
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=4000,
                )

                image_text = response.choices[0].message.content

                # Extract token usage if available
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                    total_tokens = getattr(response.usage, "total_tokens", 0) or 0

                return ExtractionResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    image_text=image_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except (APIError, OpenAIError) as e:
                if attempt == settings.api_max_retries - 1:
                    return ExtractionResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        image_text=None,
                        error=f"API error: {str(e)}",
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

    async def classify(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ClassificationResult:
        """Classify whether an image contains graphical content."""
        return await asyncio.to_thread(
            self._sync_classify, image_base64, page_number, chunk_id
        )

    async def extract(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ExtractionResult:
        """Extract visual content from an image."""
        return await asyncio.to_thread(
            self._sync_extract, image_base64, page_number, chunk_id
        )

    def _sync_synthesize(
        self,
        extraction_1: str,
        extraction_2: str,
        extraction_3: str,
        page_number: int,
        chunk_id: str,
    ) -> ExtractionResult:
        """Synchronous synthesize implementation."""
        # Build the input with all 3 extractions
        synthesis_input = f"""## EXTRACTION 1
{extraction_1}

## EXTRACTION 2
{extraction_2}

## EXTRACTION 3
{extraction_3}"""

        for attempt in range(settings.api_max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.synthesis_prompt,
                        },
                        {
                            "role": "user",
                            "content": synthesis_input,
                        },
                    ],
                    max_completion_tokens=6000,
                )

                image_text = response.choices[0].message.content

                # Extract token usage if available
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                    total_tokens = getattr(response.usage, "total_tokens", 0) or 0

                return ExtractionResult(
                    page_number=page_number,
                    chunk_id=chunk_id,
                    image_text=image_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            except (APIError, OpenAIError) as e:
                if attempt == settings.api_max_retries - 1:
                    return ExtractionResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        image_text=extraction_1,  # Fallback to first extraction
                        error=f"Synthesis API error: {str(e)}",
                    )
            except Exception as e:
                if attempt == settings.api_max_retries - 1:
                    return ExtractionResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        image_text=extraction_1,  # Fallback to first extraction
                        error=f"Synthesis error: {str(e)}",
                    )

        return ExtractionResult(
            page_number=page_number,
            chunk_id=chunk_id,
            image_text=extraction_1,  # Fallback to first extraction
            error="Synthesis max retries exceeded",
        )

    async def synthesize(
        self,
        extraction_1: str,
        extraction_2: str,
        extraction_3: str,
        page_number: int,
        chunk_id: str,
    ) -> ExtractionResult:
        """Synthesize 3 extractions into a comprehensive result."""
        return await asyncio.to_thread(
            self._sync_synthesize,
            extraction_1,
            extraction_2,
            extraction_3,
            page_number,
            chunk_id,
        )
