"""
PipelineOrchestrator - Coordinates extraction, vision, and summarization stages.

Replaces subprocess-based execution with direct module calls,
passing data between stages in-memory with progress callbacks.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from app.pipeline.models import PipelineStage, ProgressEvent
from app.pipeline.extraction.extractor import DocumentExtractor
from app.pipeline.extraction.schemas import ExtractionConfig
from app.pipeline.vision.processor import VisionProcessor
from app.pipeline.vision.schemas import VisionConfig, VisionProvider
from app.pipeline.summarization.summarizer import SummarizeService
from app.pipeline.summarization.codex_summarizer import CLISummarizer
from app.pipeline.summarization.schemas import SummarizerConfig, PageContext


@dataclass
class StageMetricsData:
    """Metrics data for a single pipeline stage."""

    duration_ms: float = 0.0
    pages_processed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Stage-specific fields
    pages_with_images: int = 0  # Vision stage
    classified_count: int = 0  # Vision stage
    extracted_count: int = 0  # Vision stage
    avg_relevancy: float = 0.0  # Summarization stage
    total_attempts: int = 0  # Summarization stage


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""

    extraction: StageMetricsData = field(default_factory=StageMetricsData)
    vision: StageMetricsData = field(default_factory=StageMetricsData)
    summarization: StageMetricsData = field(default_factory=StageMetricsData)

# Stage-specific loggers for per-stage log files
extraction_logger = logging.getLogger("app.pipeline.extraction")
vision_logger = logging.getLogger("app.pipeline.vision")
summarization_logger = logging.getLogger("app.pipeline.summarization")

# Type alias for progress callback
ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


class PipelineOrchestrator:
    """Orchestrates the document processing pipeline.

    Coordinates three stages:
    1. Extraction: PDF/text parsing to extract content
    2. Vision: Visual content classification and extraction
    3. Summarization: Generate summaries with quality validation
    """

    def __init__(self):
        """Initialize the orchestrator."""
        pass

    @staticmethod
    def _strip_base64_images(pages: list[dict[str, Any]]) -> None:
        """Remove base64 image data from all pages to reduce output size."""
        for page in pages:
            page.pop("image_base64", None)

    async def run(
        self,
        job_id: str,
        input_file_path: Path,
        config: dict[str, Any],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> tuple[dict[str, Any], PipelineMetrics]:
        """
        Run the complete pipeline on a document.

        Args:
            job_id: Unique job identifier
            input_file_path: Path to input document
            config: Pipeline configuration dict
            progress_callback: Optional callback for progress events

        Returns:
            Tuple of (output_data, metrics):
            - output_data: JSON-serializable dict with document and pages
            - metrics: PipelineMetrics with timing and token usage
        """
        input_file_path = Path(input_file_path)
        metrics = PipelineMetrics()

        # Phase 1: Extraction
        extraction_result, extraction_metrics = await self._run_extraction(
            input_file_path, config, progress_callback
        )
        metrics.extraction = extraction_metrics

        # Convert to page dicts for further processing
        pages = [page.to_dict() for page in extraction_result.pages]

        # Add page_number to each page dict for later stages
        for i, page in enumerate(pages):
            page["page_number"] = extraction_result.pages[i].page_number

        # Check if extract_only mode
        if config.get("extract_only", False):
            # Strip base64 images unless explicitly requested to keep
            if not config.get("keep_base64_images", False):
                self._strip_base64_images(pages)
            return {
                "document": {
                    "document_id": extraction_result.document_id,
                    "filename": extraction_result.filename,
                    "total_pages": extraction_result.total_pages,
                    "metadata": extraction_result.metadata,
                },
                "pages": pages,
            }, metrics

        # Phase 2: Vision (only for PDFs with images)
        # Skip vision for text files (no images to process) or when images are disabled
        is_text_file = extraction_result.metadata.get("source_type") == "text"
        skip_images = config.get("skip_images", False) or config.get("text_only", False)
        vision_mode = config.get("vision_mode", "ollama")

        if not is_text_file and not skip_images and vision_mode != "none":
            try:
                pages, vision_metrics = await self._run_vision(pages, config, progress_callback)
                metrics.vision = vision_metrics
            except Exception as e:
                vision_logger.warning("Vision stage failed, continuing: %s", e)
                # Continue without vision data
                if progress_callback:
                    await progress_callback(
                        ProgressEvent(
                            event_type="stage_changed",
                            stage=PipelineStage.VISION,
                            message=f"Vision failed: {e}",
                            error=str(e),
                        )
                    )

        # Phase 3: Summarization (unless run_summarization is False)
        run_summarization = config.get("run_summarization", True)

        if run_summarization:
            pages, summarization_metrics = await self._run_summarization(pages, config, progress_callback)
            metrics.summarization = summarization_metrics

        # Strip base64 images unless explicitly requested to keep
        if not config.get("keep_base64_images", False):
            self._strip_base64_images(pages)

        return {
            "document": {
                "document_id": extraction_result.document_id,
                "filename": extraction_result.filename,
                "total_pages": extraction_result.total_pages,
                "metadata": extraction_result.metadata,
            },
            "pages": pages,
        }, metrics

    async def _run_extraction(
        self,
        input_file_path: Path,
        config: dict[str, Any],
        progress_callback: Optional[ProgressCallback],
    ):
        """Run the extraction stage.

        Returns:
            Tuple of (extraction_result, metrics)
        """
        stage_start = time.perf_counter()
        extraction_logger.info(
            "Extraction started - file=%s, config=%s",
            input_file_path.name,
            {
                "skip_tables": config.get("skip_tables"),
                "skip_images": config.get("skip_images"),
                "text_only": config.get("text_only"),
                "pdf_image_dpi": config.get("pdf_image_dpi"),
            },
        )

        if progress_callback:
            await progress_callback(
                ProgressEvent(
                    event_type="started",
                    stage=PipelineStage.EXTRACTION,
                    message="Starting extraction",
                )
            )

        extraction_config = ExtractionConfig(
            skip_tables=config.get("skip_tables", False),
            skip_images=config.get("skip_images", False),
            skip_pptx_tables=config.get("skip_pptx_tables", False),
            text_only=config.get("text_only", False),
            pdf_image_dpi=config.get("pdf_image_dpi", 200),
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
        )

        extractor = DocumentExtractor()

        async def extraction_progress(current: int, total: int, message: str):
            extraction_logger.debug(
                "Extraction progress - page=%d/%d, message=%s",
                current,
                total,
                message,
            )
            if progress_callback:
                await progress_callback(
                    ProgressEvent(
                        event_type="progress",
                        stage=PipelineStage.EXTRACTION,
                        message=message,
                        current=current,
                        total=total,
                    )
                )

        result = await extractor.extract(
            input_file_path, extraction_config, progress_callback=extraction_progress
        )

        stage_duration = time.perf_counter() - stage_start
        extraction_logger.info(
            "Extraction completed - pages=%d, duration=%.2fs, avg=%.2fms/page",
            result.total_pages,
            stage_duration,
            (stage_duration * 1000) / max(result.total_pages, 1),
        )

        if progress_callback:
            await progress_callback(
                ProgressEvent(
                    event_type="stage_changed",
                    stage=PipelineStage.EXTRACTION,
                    message="Extraction complete",
                    current=result.total_pages,
                    total=result.total_pages,
                )
            )

        metrics = StageMetricsData(
            duration_ms=stage_duration * 1000,
            pages_processed=result.total_pages,
            # Extraction doesn't use LLM, so no tokens
        )

        return result, metrics

    async def _run_vision(
        self,
        pages: list[dict],
        config: dict[str, Any],
        progress_callback: Optional[ProgressCallback],
    ) -> tuple[list[dict], StageMetricsData]:
        """Run the vision stage.

        Returns:
            Tuple of (updated pages, metrics)
        """
        from app.config import settings

        stage_start = time.perf_counter()

        # Determine classifier provider (inherits from vision_mode if not set)
        vision_mode = config.get("vision_mode", "ollama")
        classifier_mode = config.get("vision_classifier_mode") or vision_mode
        extractor_mode = config.get("vision_extractor_mode") or vision_mode

        def get_provider_and_model(mode: str) -> tuple[VisionProvider, str]:
            """Get provider enum and model name for a mode."""
            if mode == "openai":
                return VisionProvider.OPENAI, settings.openai_vision_model
            elif mode == "gemini":
                return VisionProvider.GEMINI, settings.gemini_vision_model
            elif mode in ("codex", "claude"):
                return VisionProvider.CODEX_CLI, f"{mode}-cli"
            else:
                return VisionProvider.OLLAMA, settings.vision_model

        # If extractor is CLI-based (codex or claude), force classifier to ollama (CLI doesn't do classification)
        if extractor_mode in ("codex", "claude"):
            classifier_mode = "ollama"

        classifier_provider, classifier_model = get_provider_and_model(classifier_mode)
        extractor_provider, extractor_model = get_provider_and_model(extractor_mode)

        # Count pages with images
        pages_with_images = sum(1 for p in pages if p.get("image_base64"))
        vision_logger.info(
            "Vision started - pages=%d, pages_with_images=%d, classifier=%s/%s, extractor=%s/%s",
            len(pages),
            pages_with_images,
            classifier_mode,
            classifier_model,
            extractor_mode,
            extractor_model,
        )

        if progress_callback:
            await progress_callback(
                ProgressEvent(
                    event_type="started",
                    stage=PipelineStage.VISION,
                    message="Starting vision processing",
                )
            )

        # Get CLI provider for vision (codex or claude)
        # Use explicit config, or derive from extractor_mode if it's a CLI mode
        vision_cli_provider = config.get("vision_cli_provider")
        if not vision_cli_provider:
            if extractor_mode in ("codex", "claude"):
                vision_cli_provider = extractor_mode
            else:
                vision_cli_provider = settings.vision_cli_provider

        # CLI providers should run sequentially (1 at a time) to avoid hanging
        extractor_batch_size = 1 if extractor_provider == VisionProvider.CODEX_CLI else None

        vision_config = VisionConfig(
            classifier_provider=classifier_provider,
            classifier_model=classifier_model,
            extractor_provider=extractor_provider,
            extractor_model=extractor_model,
            batch_size=5,
            extractor_batch_size=extractor_batch_size,
            detailed_extraction=config.get("vision_detailed_extraction", False),
            cli_provider=vision_cli_provider,
            ollama_base_url=settings.ollama_base_url,
            openai_api_key=settings.openai_api_key,
            gemini_api_key=settings.gemini_api_key,
        )

        processor = VisionProcessor(vision_config)

        # Convert pages to format expected by vision processor
        vision_pages = []
        for page in pages:
            # Get image_base64, wrap in list if it's a string
            image_base64 = page.get("image_base64")
            if image_base64 and isinstance(image_base64, str):
                image_base64 = [image_base64]

            vision_pages.append({
                "page_number": page.get("page_number", 0),
                "chunk_id": page.get("chunk_id", ""),
                "image_base64": image_base64 or [],
            })

        async def vision_progress(current: int, total: int, message: str):
            vision_logger.debug(
                "Vision progress - page=%d/%d, message=%s",
                current,
                total,
                message,
            )
            if progress_callback:
                await progress_callback(
                    ProgressEvent(
                        event_type="progress",
                        stage=PipelineStage.VISION,
                        message=message,
                        current=current,
                        total=total,
                    )
                )

        # Collect token metrics from classification and extraction
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        # Run classification
        if not vision_config.skip_classification:
            classification_results = await processor.classify_pages(vision_pages)
            for result in classification_results:
                total_prompt_tokens += result.prompt_tokens
                total_completion_tokens += result.completion_tokens
                total_tokens += result.total_tokens

            # Update pages with classification and log each result
            graphics_page_nums = []
            no_graphics_page_nums = []
            for page, result in zip(pages, classification_results):
                page["image_classifier"] = result.has_graphics
                if result.has_graphics:
                    graphics_page_nums.append(result.page_number)
                else:
                    no_graphics_page_nums.append(result.page_number)
                vision_logger.debug(
                    "Classification - page=%d, chunk=%s, has_graphics=%s",
                    result.page_number,
                    result.chunk_id,
                    result.has_graphics,
                )

            # Log classification summary at INFO level
            vision_logger.info(
                "Classification complete - model=%s/%s, has_graphics=%s, no_graphics=%s",
                classifier_mode,
                classifier_model,
                graphics_page_nums if graphics_page_nums else "none",
                no_graphics_page_nums if no_graphics_page_nums else "none",
            )
        else:
            for page in pages:
                page["image_classifier"] = True

        # Filter pages with graphics for extraction
        graphics_pages = [p for p in vision_pages if pages[vision_pages.index(p)].get("image_classifier", False)]

        # Run extraction on graphics pages
        classified_count = sum(1 for p in pages if p.get("image_classifier"))
        extracted_count = 0

        if graphics_pages:
            vision_logger.info(
                "Extraction starting - model=%s/%s, pages=%d",
                extractor_mode,
                extractor_model,
                len(graphics_pages),
            )
            extraction_results = await processor.extract_visual_content(graphics_pages)
            for result in extraction_results:
                total_prompt_tokens += result.prompt_tokens
                total_completion_tokens += result.completion_tokens
                total_tokens += result.total_tokens

            # Create lookup by chunk_id
            extraction_by_chunk = {r.chunk_id: r for r in extraction_results}

            # Update pages with extraction results
            for page in pages:
                chunk_id = page["chunk_id"]
                if chunk_id in extraction_by_chunk:
                    result = extraction_by_chunk[chunk_id]
                    page["image_text"] = result.image_text
                    # Store intermediate results if detailed extraction was used
                    if result.image_text_1 is not None:
                        page["image_text_1"] = result.image_text_1
                    if result.image_text_2 is not None:
                        page["image_text_2"] = result.image_text_2
                    if result.image_text_3 is not None:
                        page["image_text_3"] = result.image_text_3
                    if result.image_text:
                        extracted_count += 1
                else:
                    page["image_text"] = None

        stage_duration = time.perf_counter() - stage_start
        vision_logger.info(
            "Vision completed - pages=%d, classified=%d, extracted=%d, duration=%.2fs, tokens=%d, classifier=%s/%s, extractor=%s/%s",
            len(pages),
            classified_count,
            extracted_count,
            stage_duration,
            total_tokens,
            classifier_mode,
            classifier_model,
            extractor_mode,
            extractor_model,
        )

        if progress_callback:
            await progress_callback(
                ProgressEvent(
                    event_type="stage_changed",
                    stage=PipelineStage.VISION,
                    message="Vision processing complete",
                    current=len(pages),
                    total=len(pages),
                )
            )

        metrics = StageMetricsData(
            duration_ms=stage_duration * 1000,
            pages_processed=len(pages),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            pages_with_images=pages_with_images,
            classified_count=classified_count,
            extracted_count=extracted_count,
        )

        return pages, metrics

    async def _run_summarization(
        self,
        pages: list[dict],
        config: dict[str, Any],
        progress_callback: Optional[ProgressCallback],
    ) -> tuple[list[dict], StageMetricsData]:
        """Run the summarization stage.

        Returns:
            Tuple of (updated pages, metrics)
        """
        from app.config import settings

        stage_start = time.perf_counter()
        summarizer_mode = config.get("summarizer_mode", "full")
        summarizer_provider = config.get("summarizer_provider", "ollama")

        summarization_logger.info(
            "Summarization started - pages=%d, mode=%s, provider=%s, model=%s",
            len(pages),
            summarizer_mode,
            summarizer_provider,
            settings.summarizer_model_tier_1,
        )

        if progress_callback:
            await progress_callback(
                ProgressEvent(
                    event_type="started",
                    stage=PipelineStage.SUMMARIZATION,
                    message="Starting summarization",
                )
            )

        summarizer_config = SummarizerConfig(
            mode=summarizer_mode,
            provider=summarizer_provider,
            batch_size=5,
            detailed_extraction=config.get("summarizer_detailed_extraction", False),
        )

        # Use CLISummarizer for codex/claude providers, otherwise SummarizeService
        if summarizer_provider in ("codex", "claude"):
            # Get CLI provider for summarization (codex or claude)
            # Use explicit config, or derive from summarizer_provider
            summarizer_cli_provider = config.get("summarizer_cli_provider")
            if not summarizer_cli_provider:
                summarizer_cli_provider = summarizer_provider  # codex or claude
            summarizer = CLISummarizer(summarizer_config, cli_provider=summarizer_cli_provider)
        else:
            summarizer = SummarizeService(summarizer_config)

        # Convert pages to PageContext
        contexts = []
        for page in pages:
            context = PageContext(
                page_number=page.get("page_number", 0),
                chunk_id=page.get("chunk_id", ""),
                text=page.get("text", ""),
                tables=page.get("tables"),
                image_text=page.get("image_text"),
                image_classifier=page.get("image_classifier"),
            )
            contexts.append(context)

        async def summarizer_progress(current: int, total: int):
            summarization_logger.debug(
                "Summarization progress - page=%d/%d",
                current,
                total,
            )
            if progress_callback:
                await progress_callback(
                    ProgressEvent(
                        event_type="progress",
                        stage=PipelineStage.SUMMARIZATION,
                        message=f"Summarizing page {current} of {total}",
                        current=current,
                        total=total,
                    )
                )

        results = await summarizer.summarize_pages_batch(
            contexts, progress_callback=summarizer_progress
        )

        # Merge summarization results back into pages and calculate stats
        total_attempts = 0
        avg_relevancy = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for i, page in enumerate(pages):
            if i < len(results):
                result = results[i]
                page["summary_notes"] = result.summary_notes
                page["summary_topics"] = result.summary_topics
                page["summary_relevancy"] = result.summary_relevancy
                # Store intermediate results if detailed extraction was used
                if result.summary_notes_1 is not None:
                    page["summary_notes_1"] = result.summary_notes_1
                if result.summary_notes_2 is not None:
                    page["summary_notes_2"] = result.summary_notes_2
                if result.summary_notes_3 is not None:
                    page["summary_notes_3"] = result.summary_notes_3
                total_attempts += result.attempts_used
                avg_relevancy += result.summary_relevancy
                total_prompt_tokens += result.prompt_tokens
                total_completion_tokens += result.completion_tokens
                total_tokens += result.total_tokens

        if len(results) > 0:
            avg_relevancy /= len(results)

        stage_duration = time.perf_counter() - stage_start
        summarization_logger.info(
            "Summarization completed - pages=%d, duration=%.2fs, avg_relevancy=%.1f%%, total_attempts=%d, tokens=%d, model=%s",
            len(pages),
            stage_duration,
            avg_relevancy,
            total_attempts,
            total_tokens,
            settings.summarizer_model_tier_1,
        )

        if progress_callback:
            await progress_callback(
                ProgressEvent(
                    event_type="stage_changed",
                    stage=PipelineStage.SUMMARIZATION,
                    message="Summarization complete",
                    current=len(pages),
                    total=len(pages),
                )
            )

        metrics = StageMetricsData(
            duration_ms=stage_duration * 1000,
            pages_processed=len(pages),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            avg_relevancy=avg_relevancy,
            total_attempts=total_attempts,
        )

        return pages, metrics
