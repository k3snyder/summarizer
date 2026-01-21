"""
Pipeline runner service using orchestrator directly.

Runs the full pipeline in-process via the orchestrator.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from app.db.database import AsyncSessionLocal
from app.logging import set_job_id, reset_job_id, TokenMetrics, JobMetrics, JobMetricsLogger
from app.models.config import PipelineConfig
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.models import ProgressEvent, PipelineStage
from app.services.job_manager import JobManager
from app.services.file_storage import FileStorage

logger = logging.getLogger(__name__)
metrics_logger = JobMetricsLogger()


class PipelineRunner:
    """
    Execute document processing pipeline via orchestrator.

    Uses the PipelineOrchestrator directly, passing data between
    stages in-memory with progress callbacks.
    """

    def __init__(self):
        """Initialize PipelineRunner."""
        pass

    def _config_to_dict(self, config: PipelineConfig) -> dict[str, Any]:
        """
        Convert PipelineConfig to dict for orchestrator.

        Args:
            config: PipelineConfig model

        Returns:
            Dict with configuration values
        """
        return {
            "extract_only": config.extract_only,
            "text_only": config.text_only,
            "skip_tables": config.skip_tables,
            "skip_images": config.skip_images,
            "skip_pptx_tables": config.skip_pptx_tables,
            "pdf_image_dpi": config.pdf_image_dpi,
            "vision_mode": config.vision_mode,
            "vision_classifier_mode": config.vision_classifier_mode,
            "vision_extractor_mode": config.vision_extractor_mode,
            "vision_detailed_extraction": config.vision_detailed_extraction,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "run_summarization": config.run_summarization,
            "summarizer_mode": config.summarizer_mode,
            "summarizer_provider": config.summarizer_provider,
            "summarizer_detailed_extraction": config.summarizer_detailed_extraction,
            "summarizer_insight_mode": config.summarizer_insight_mode,
            "keep_base64_images": config.keep_base64_images,
        }

    async def run(
        self,
        job_id: str,
        file_path: str,
        config: PipelineConfig,
    ) -> None:
        """
        Execute pipeline for a job using orchestrator directly.

        Args:
            job_id: Job identifier
            file_path: Path to input file
            config: Pipeline configuration

        The method updates job status via JobManager during execution,
        stores output path on completion, and captures errors.
        """
        # Set job_id in logging context for all subsequent logs
        token = set_job_id(job_id)
        pipeline_start = time.perf_counter()

        logger.info(
            "Pipeline started - file=%s, config=%s",
            Path(file_path).name,
            {k: v for k, v in self._config_to_dict(config).items()},
        )

        try:
            # Update job to processing status using fresh session
            async with AsyncSessionLocal() as start_db:
                await JobManager.update_status(
                    start_db,
                    job_id,
                    status="processing",
                    progress=0,
                    stage="extraction",
                    message="Starting pipeline execution",
                )

            # Create progress callback with lock to prevent concurrent db access
            current_progress = 0
            progress_lock = asyncio.Lock()

            # Stage weights for overall progress calculation
            # Extraction: 0-20%, Vision: 20-50%, Summarization: 50-100%
            stage_weights = {
                PipelineStage.EXTRACTION: (0, 20),
                PipelineStage.VISION: (20, 50),
                PipelineStage.SUMMARIZATION: (50, 100),
            }

            async def progress_callback(event: ProgressEvent) -> None:
                """Handle progress events from orchestrator."""
                nonlocal current_progress
                stage = self._stage_to_string(event.stage) if event.stage else None

                if event.current is not None and event.total is not None and event.total > 0:
                    # Calculate progress within the current stage
                    stage_progress = event.current / event.total

                    # Map to overall progress using stage weights
                    if event.stage and event.stage in stage_weights:
                        start_pct, end_pct = stage_weights[event.stage]
                        current_progress = int(start_pct + stage_progress * (end_pct - start_pct))
                    else:
                        # Fallback for unknown stages
                        current_progress = int(stage_progress * 100)

                # Use lock and fresh session to avoid concurrent access issues
                async with progress_lock:
                    async with AsyncSessionLocal() as progress_db:
                        await JobManager.update_status(
                            progress_db,
                            job_id,
                            status="processing",
                            progress=current_progress,
                            stage=stage,
                            message=event.message,
                        )

            # Convert config to dict
            config_dict = self._config_to_dict(config)

            # Run orchestrator
            orchestrator = PipelineOrchestrator()
            output_data, pipeline_metrics = await orchestrator.run(
                job_id=job_id,
                input_file_path=Path(file_path),
                config=config_dict,
                progress_callback=progress_callback,
            )

            pipeline_duration = time.perf_counter() - pipeline_start

            # Add metrics to output data
            output_data["metrics"] = {
                "total_duration_ms": round(pipeline_duration * 1000, 2),
                "total_tokens": (
                    pipeline_metrics.extraction.total_tokens
                    + pipeline_metrics.vision.total_tokens
                    + pipeline_metrics.summarization.total_tokens
                ),
                "stages": {
                    "extraction": {
                        "duration_ms": round(pipeline_metrics.extraction.duration_ms, 2),
                        "pages_processed": pipeline_metrics.extraction.pages_processed,
                        "tokens": pipeline_metrics.extraction.total_tokens,
                    },
                    "vision": {
                        "duration_ms": round(pipeline_metrics.vision.duration_ms, 2),
                        "pages_processed": pipeline_metrics.vision.pages_processed,
                        "pages_with_images": pipeline_metrics.vision.pages_with_images,
                        "classified_count": pipeline_metrics.vision.classified_count,
                        "extracted_count": pipeline_metrics.vision.extracted_count,
                        "tokens": pipeline_metrics.vision.total_tokens,
                    },
                    "summarization": {
                        "duration_ms": round(pipeline_metrics.summarization.duration_ms, 2),
                        "pages_processed": pipeline_metrics.summarization.pages_processed,
                        "avg_relevancy": round(pipeline_metrics.summarization.avg_relevancy, 2),
                        "total_attempts": pipeline_metrics.summarization.total_attempts,
                        "tokens": pipeline_metrics.summarization.total_tokens,
                    },
                },
                "config": {
                    "vision_mode": config_dict.get("vision_mode"),
                    "summarizer_provider": config_dict.get("summarizer_provider"),
                    "summarizer_mode": config_dict.get("summarizer_mode"),
                },
            }

            # Save output to job directory
            output_path = await FileStorage.save_output(job_id, output_data)

            # Store output path and mark as completed using fresh session
            async with AsyncSessionLocal() as complete_db:
                await JobManager.set_output(complete_db, job_id, str(output_path))
                await JobManager.update_status(
                    complete_db,
                    job_id,
                    status="completed",
                    progress=100,
                    stage=None,
                    message="Processing completed successfully",
                )

            logger.info(
                "Pipeline completed - duration=%.2fs, output=%s",
                pipeline_duration,
                output_path,
            )

            # Log job metrics summary
            total_pages = output_data.get("document", {}).get("total_pages", 0)
            job_metrics = JobMetrics(
                job_id=job_id,
                filename=Path(file_path).name,
                status="completed",
                total_pages=total_pages,
                total_duration_ms=pipeline_duration * 1000,
                config=config_dict,
            )
            # Copy stage metrics
            job_metrics.extraction.duration_ms = pipeline_metrics.extraction.duration_ms
            job_metrics.extraction.pages_processed = pipeline_metrics.extraction.pages_processed
            job_metrics.extraction.tokens = TokenMetrics(
                prompt_tokens=pipeline_metrics.extraction.prompt_tokens,
                completion_tokens=pipeline_metrics.extraction.completion_tokens,
                total_tokens=pipeline_metrics.extraction.total_tokens,
            )
            job_metrics.vision.duration_ms = pipeline_metrics.vision.duration_ms
            job_metrics.vision.pages_processed = pipeline_metrics.vision.pages_processed
            job_metrics.vision.pages_with_images = pipeline_metrics.vision.pages_with_images
            job_metrics.vision.classified_count = pipeline_metrics.vision.classified_count
            job_metrics.vision.extracted_count = pipeline_metrics.vision.extracted_count
            job_metrics.vision.tokens = TokenMetrics(
                prompt_tokens=pipeline_metrics.vision.prompt_tokens,
                completion_tokens=pipeline_metrics.vision.completion_tokens,
                total_tokens=pipeline_metrics.vision.total_tokens,
            )
            job_metrics.summarization.duration_ms = pipeline_metrics.summarization.duration_ms
            job_metrics.summarization.pages_processed = pipeline_metrics.summarization.pages_processed
            job_metrics.summarization.avg_relevancy = pipeline_metrics.summarization.avg_relevancy
            job_metrics.summarization.total_attempts = pipeline_metrics.summarization.total_attempts
            job_metrics.summarization.tokens = TokenMetrics(
                prompt_tokens=pipeline_metrics.summarization.prompt_tokens,
                completion_tokens=pipeline_metrics.summarization.completion_tokens,
                total_tokens=pipeline_metrics.summarization.total_tokens,
            )
            metrics_logger.log_job_summary(job_metrics)

        except Exception as e:
            pipeline_duration = time.perf_counter() - pipeline_start
            error_message = str(e)

            logger.error(
                "Pipeline failed - duration=%.2fs, error=%s",
                pipeline_duration,
                error_message,
                exc_info=True,
            )

            # Mark job as failed using fresh session
            async with AsyncSessionLocal() as error_db:
                await JobManager.update_status(
                    error_db,
                    job_id,
                    status="failed",
                    progress=0,
                    stage=None,
                    message=f"Pipeline failed: {error_message}",
                )

                # Update error field in database
                from sqlalchemy import update
                from app.db.database import Job

                await error_db.execute(
                    update(Job).where(Job.id == job_id).values(error=error_message)
                )
                await error_db.commit()

            # Log failed job metrics
            job_metrics = JobMetrics(
                job_id=job_id,
                filename=Path(file_path).name,
                status="failed",
                total_duration_ms=pipeline_duration * 1000,
                config=self._config_to_dict(config),
                error=error_message,
            )
            metrics_logger.log_job_summary(job_metrics)

        finally:
            # Reset job_id context
            reset_job_id(token)

    def _stage_to_string(self, stage: PipelineStage) -> str:
        """Convert PipelineStage enum to string."""
        if stage == PipelineStage.EXTRACTION:
            return "extraction"
        elif stage == PipelineStage.VISION:
            return "vision"
        elif stage == PipelineStage.SUMMARIZATION:
            return "summarization"
        return "unknown"
