"""
VisionProcessor service for processing document pages.
Handles classification and extraction with rate-limited concurrency.
Supports separate providers for classification vs extraction.
"""

import asyncio
from typing import Optional, Callable, Awaitable

from app.pipeline.vision.schemas import (
    VisionConfig,
    VisionProvider,
    ClassificationResult,
    ExtractionResult,
)
from app.pipeline.vision.providers.base import VisionProviderBase
from app.pipeline.vision.providers.ollama import OllamaVisionProvider
from app.pipeline.vision.providers.openai import OpenAIVisionProvider
from app.pipeline.vision.providers.gemini import GeminiVisionProvider
from app.pipeline.vision.providers.codex_cli import CLIVisionProvider


class VisionProcessor:
    """Service for processing document pages with vision models.

    Handles batch processing with rate limiting using asyncio.Semaphore.
    Supports separate providers for classification and extraction.
    """

    def __init__(self, config: VisionConfig):
        """Initialize VisionProcessor with configuration.

        Args:
            config: VisionConfig with provider settings and batch size
        """
        self.config = config

        # Create separate provider instances for classifier and extractor
        self._classifier_provider = self._create_provider(
            config.classifier_provider,
            config.classifier_model,
        )
        self._extractor_provider = self._create_provider(
            config.extractor_provider,
            config.extractor_model,
        )

        # Separate semaphores for rate limiting each provider
        # Use extractor_batch_size if specified (e.g., 1 for OpenAI), otherwise batch_size
        self._classifier_semaphore = asyncio.Semaphore(config.batch_size)
        extractor_concurrency = config.extractor_batch_size or config.batch_size
        self._extractor_semaphore = asyncio.Semaphore(extractor_concurrency)

        # Track if using same provider for both (optimization info)
        self._providers_are_same = (
            config.classifier_provider == config.extractor_provider
            and config.classifier_model == config.extractor_model
        )

    def _create_provider(
        self,
        provider: VisionProvider,
        model: Optional[str],
    ) -> VisionProviderBase:
        """Create a vision provider instance.

        Args:
            provider: The provider type to create
            model: The model name to use

        Returns:
            Configured provider instance
        """
        if provider == VisionProvider.OLLAMA:
            return OllamaVisionProvider(
                base_url=self.config.ollama_base_url,
                model=model,
            )
        elif provider == VisionProvider.OPENAI:
            return OpenAIVisionProvider(
                api_key=self.config.openai_api_key,
                model=model,
            )
        elif provider == VisionProvider.GEMINI:
            return GeminiVisionProvider(
                api_key=self.config.gemini_api_key,
                model=model,
            )
        elif provider == VisionProvider.CODEX_CLI:
            return CLIVisionProvider(cli_provider=self.config.cli_provider)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def classify_pages(
        self,
        pages: list[dict],
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> list[ClassificationResult]:
        """Classify pages to determine which have graphical content.

        Uses the classifier provider configured in VisionConfig.

        Args:
            pages: List of page dicts with page_number, chunk_id, image_base64
            progress_callback: Optional async callback(current, total) for progress

        Returns:
            List of ClassificationResult for each page
        """
        async def classify_with_semaphore(page: dict, index: int) -> ClassificationResult:
            async with self._classifier_semaphore:
                image_base64 = page.get("image_base64", [])
                if not image_base64:
                    return ClassificationResult(
                        page_number=page["page_number"],
                        chunk_id=page["chunk_id"],
                        has_graphics=False,
                    )

                # Use first image for classification
                result = await self._classifier_provider.classify(
                    image_base64[0],
                    page["page_number"],
                    page["chunk_id"],
                )

                if progress_callback:
                    await progress_callback(index + 1, len(pages))

                return result

        tasks = [classify_with_semaphore(page, i) for i, page in enumerate(pages)]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def extract_visual_content(
        self,
        pages: list[dict],
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> list[ExtractionResult]:
        """Extract visual content from pages.

        Uses the extractor provider configured in VisionConfig.
        When detailed_extraction is enabled, runs extraction 3 times
        and synthesizes results for comprehensive coverage.

        Args:
            pages: List of page dicts with page_number, chunk_id, image_base64
            progress_callback: Optional async callback(current, total) for progress

        Returns:
            List of ExtractionResult for each page
        """
        if self.config.detailed_extraction:
            return await self._extract_detailed(pages, progress_callback)
        else:
            return await self._extract_standard(pages, progress_callback)

    async def _extract_standard(
        self,
        pages: list[dict],
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> list[ExtractionResult]:
        """Standard single-pass extraction."""
        async def extract_with_semaphore(page: dict, index: int) -> ExtractionResult:
            async with self._extractor_semaphore:
                image_base64 = page.get("image_base64", [])
                if not image_base64:
                    return ExtractionResult(
                        page_number=page["page_number"],
                        chunk_id=page["chunk_id"],
                        image_text=None,
                        skipped=True,
                    )

                # Use first image for extraction
                result = await self._extractor_provider.extract(
                    image_base64[0],
                    page["page_number"],
                    page["chunk_id"],
                )

                if progress_callback:
                    await progress_callback(index + 1, len(pages))

                return result

        tasks = [extract_with_semaphore(page, i) for i, page in enumerate(pages)]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def _extract_detailed(
        self,
        pages: list[dict],
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> list[ExtractionResult]:
        """Detailed 3x extraction with synthesis for comprehensive coverage."""
        async def extract_page_detailed(page: dict, index: int) -> ExtractionResult:
            async with self._extractor_semaphore:
                image_base64 = page.get("image_base64", [])
                if not image_base64:
                    return ExtractionResult(
                        page_number=page["page_number"],
                        chunk_id=page["chunk_id"],
                        image_text=None,
                        skipped=True,
                    )

                page_num = page["page_number"]
                chunk_id = page["chunk_id"]
                image = image_base64[0]

                # Run extraction 3 times
                result_1 = await self._extractor_provider.extract(image, page_num, chunk_id)
                result_2 = await self._extractor_provider.extract(image, page_num, chunk_id)
                result_3 = await self._extractor_provider.extract(image, page_num, chunk_id)

                # Collect token usage from all 3 extractions
                total_prompt_tokens = (
                    result_1.prompt_tokens + result_2.prompt_tokens + result_3.prompt_tokens
                )
                total_completion_tokens = (
                    result_1.completion_tokens + result_2.completion_tokens + result_3.completion_tokens
                )

                # Get the extraction texts
                text_1 = result_1.image_text or ""
                text_2 = result_2.image_text or ""
                text_3 = result_3.image_text or ""

                # Synthesize the 3 extractions
                synthesis_result = await self._extractor_provider.synthesize(
                    text_1, text_2, text_3, page_num, chunk_id
                )

                # Add synthesis tokens to total
                total_prompt_tokens += synthesis_result.prompt_tokens
                total_completion_tokens += synthesis_result.completion_tokens

                if progress_callback:
                    await progress_callback(index + 1, len(pages))

                # Return result with all intermediate results stored
                return ExtractionResult(
                    page_number=page_num,
                    chunk_id=chunk_id,
                    image_text=synthesis_result.image_text,
                    image_text_1=text_1 if text_1 else None,
                    image_text_2=text_2 if text_2 else None,
                    image_text_3=text_3 if text_3 else None,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                )

        tasks = [extract_page_detailed(page, i) for i, page in enumerate(pages)]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def process_pages(
        self,
        pages: list[dict],
        progress_callback: Optional[Callable[[int, int, str], Awaitable[None]]] = None,
    ) -> list[dict]:
        """Full vision processing: classify then extract graphics pages.

        Args:
            pages: List of page dicts
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of page dicts with image_classifier and image_text added
        """
        total_pages = len(pages)

        # Phase 1: Classification
        if progress_callback:
            await progress_callback(0, total_pages, "Classifying pages...")

        if not self.config.skip_classification:
            classification_results = await self.classify_pages(pages)

            # Update pages with classification results
            for page, result in zip(pages, classification_results):
                page["image_classifier"] = result.has_graphics
        else:
            # Skip classification, assume all have graphics
            for page in pages:
                page["image_classifier"] = True

        # Phase 2: Extract only pages with graphics
        graphics_pages = [p for p in pages if p.get("image_classifier", False)]

        if progress_callback:
            await progress_callback(
                0, len(graphics_pages), f"Extracting visual content from {len(graphics_pages)} pages..."
            )

        if graphics_pages:
            extraction_results = await self.extract_visual_content(graphics_pages)

            # Create lookup by chunk_id
            extraction_by_chunk = {r.chunk_id: r for r in extraction_results}

            # Update pages with extraction results
            for page in pages:
                chunk_id = page["chunk_id"]
                if chunk_id in extraction_by_chunk:
                    result = extraction_by_chunk[chunk_id]
                    page["image_text"] = result.image_text
                else:
                    page["image_text"] = None

        return pages
