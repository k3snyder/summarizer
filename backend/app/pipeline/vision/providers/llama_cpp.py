"""
llama.cpp vision provider implementation.
Uses the OpenAI-compatible /v1/chat/completions endpoint for multimodal requests.
"""

import asyncio
import re
from typing import Any, Optional

from openai import APIError, OpenAI, OpenAIError

from app.config import settings
from app.pipeline.vision.providers.base import VisionProviderBase
from app.pipeline.vision.providers.openai import _load_prompt, _prepare_image_data_uri
from app.pipeline.vision.schemas import ClassificationResult, ExtractionResult


class LlamaCppVisionProvider(VisionProviderBase):
    """llama.cpp vision provider using the OpenAI-compatible server API."""

    _THOUGHT_PREFIX_PATTERN = re.compile(
        r"^(?:<\|channel\>thought\s*<channel\|>\s*)+",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize llama.cpp provider."""
        self.base_url = base_url or settings.effective_llama_cpp_vision_base_url
        self.api_key = api_key or settings.llama_cpp_api_key
        self.model = model or settings.llama_cpp_vision_model

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

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

    @staticmethod
    def _request_options(max_tokens: int, temperature: float) -> dict[str, Any]:
        """Build llama.cpp-compatible request options."""
        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": {
                "reasoning": "off",
                "reasoning_budget": 0,
                "reasoning_in_content": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_format": "none",
            },
        }

    @staticmethod
    def _extract_usage(response: Any) -> tuple[int, int, int]:
        """Extract token usage from an OpenAI-compatible response."""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            total_tokens = getattr(response.usage, "total_tokens", 0) or 0

        return prompt_tokens, completion_tokens, total_tokens

    @classmethod
    def _clean_message_content(cls, content: Optional[str]) -> str:
        """Remove llama.cpp/Gemma thought-channel prefixes from response content."""
        if not content:
            return ""
        return cls._THOUGHT_PREFIX_PATTERN.sub("", content).strip()

    def _sync_classify(
        self,
        image_base64: str,
        page_number: int,
        chunk_id: str,
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
                                {"type": "image_url", "image_url": {"url": image_data_uri}},
                            ],
                        }
                    ],
                    **self._request_options(max_tokens=10, temperature=0.0),
                )

                result_text = self._clean_message_content(
                    response.choices[0].message.content
                ).upper()
                has_graphics = result_text.startswith("YES")
                prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)

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
        self,
        image_base64: str,
        page_number: int,
        chunk_id: str,
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
                                {"type": "image_url", "image_url": {"url": image_data_uri}},
                            ],
                        }
                    ],
                    **self._request_options(max_tokens=4000, temperature=0.1),
                )

                image_text = self._clean_message_content(response.choices[0].message.content)
                prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)

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
        self,
        image_base64: str,
        page_number: int,
        chunk_id: str,
    ) -> ClassificationResult:
        """Classify whether an image contains graphical content."""
        return await asyncio.to_thread(
            self._sync_classify,
            image_base64,
            page_number,
            chunk_id,
        )

    async def extract(
        self,
        image_base64: str,
        page_number: int,
        chunk_id: str,
    ) -> ExtractionResult:
        """Extract visual content from an image."""
        return await asyncio.to_thread(
            self._sync_extract,
            image_base64,
            page_number,
            chunk_id,
        )

    def _sync_synthesize(
        self,
        extraction_1: str,
        extraction_2: str,
        extraction_3: str,
        page_number: int,
        chunk_id: str,
    ) -> ExtractionResult:
        """Synthesize three extraction passes into a single result."""
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
                    **self._request_options(max_tokens=6000, temperature=0.1),
                )

                image_text = self._clean_message_content(response.choices[0].message.content)
                prompt_tokens, completion_tokens, total_tokens = self._extract_usage(response)

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
                        image_text=extraction_1,
                        error=f"Synthesis API error: {str(e)}",
                    )
            except Exception as e:
                if attempt == settings.api_max_retries - 1:
                    return ExtractionResult(
                        page_number=page_number,
                        chunk_id=chunk_id,
                        image_text=extraction_1,
                        error=f"Synthesis error: {str(e)}",
                    )

        return ExtractionResult(
            page_number=page_number,
            chunk_id=chunk_id,
            image_text=extraction_1,
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
