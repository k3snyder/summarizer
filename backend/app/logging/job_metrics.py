"""Job-level metrics collection and logging.

Provides structured job summary logging with per-stage timing and token usage.
Writes to logs/jobs.metrics.jsonl in NDJSON format.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _utc_now_iso() -> str:
    """Get current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

logger = logging.getLogger(__name__)


@dataclass
class TokenMetrics:
    """Token usage metrics for an API call or stage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenMetrics") -> None:
        """Add another TokenMetrics to this one."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> dict[str, int]:
        """Convert to dict."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    duration_ms: float = 0.0
    pages_processed: int = 0
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    success: bool = True
    error: Optional[str] = None

    # Stage-specific fields
    pages_with_images: Optional[int] = None  # Vision stage
    classified_count: Optional[int] = None  # Vision stage
    extracted_count: Optional[int] = None  # Vision stage
    avg_relevancy: Optional[float] = None  # Summarization stage
    total_attempts: Optional[int] = None  # Summarization stage

    @property
    def ms_per_page(self) -> float:
        """Calculate ms per page."""
        if self.pages_processed <= 0:
            return 0.0
        return self.duration_ms / self.pages_processed

    @property
    def tokens_per_page(self) -> float:
        """Calculate tokens per page."""
        if self.pages_processed <= 0:
            return 0.0
        return self.tokens.total_tokens / self.pages_processed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result = {
            "duration_ms": round(self.duration_ms, 2),
            "pages_processed": self.pages_processed,
            "ms_per_page": round(self.ms_per_page, 2),
            "tokens": self.tokens.to_dict(),
            "tokens_per_page": round(self.tokens_per_page, 2),
            "success": self.success,
        }
        if self.error:
            result["error"] = self.error
        if self.pages_with_images is not None:
            result["pages_with_images"] = self.pages_with_images
        if self.classified_count is not None:
            result["classified_count"] = self.classified_count
        if self.extracted_count is not None:
            result["extracted_count"] = self.extracted_count
        if self.avg_relevancy is not None:
            result["avg_relevancy"] = round(self.avg_relevancy, 2)
        if self.total_attempts is not None:
            result["total_attempts"] = self.total_attempts
        return result


@dataclass
class JobMetrics:
    """Complete metrics for a pipeline job."""

    job_id: str
    filename: str
    status: str
    total_pages: int = 0

    # Timestamps
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Total duration
    total_duration_ms: float = 0.0

    # Per-stage metrics
    extraction: StageMetrics = field(default_factory=StageMetrics)
    vision: StageMetrics = field(default_factory=StageMetrics)
    summarization: StageMetrics = field(default_factory=StageMetrics)

    # Configuration used
    config: dict[str, Any] = field(default_factory=dict)

    # Error if failed
    error: Optional[str] = None

    @property
    def total_tokens(self) -> TokenMetrics:
        """Sum of tokens across all stages."""
        total = TokenMetrics()
        total.add(self.extraction.tokens)
        total.add(self.vision.tokens)
        total.add(self.summarization.tokens)
        return total

    @property
    def ms_per_page(self) -> float:
        """Total ms per page."""
        if self.total_pages <= 0:
            return 0.0
        return self.total_duration_ms / self.total_pages

    @property
    def tokens_per_page(self) -> float:
        """Total tokens per page."""
        if self.total_pages <= 0:
            return 0.0
        return self.total_tokens.total_tokens / self.total_pages

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "timestamp": _utc_now_iso(),
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status,
            "total_pages": self.total_pages,
            "timing": {
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "total_duration_ms": round(self.total_duration_ms, 2),
                "ms_per_page": round(self.ms_per_page, 2),
            },
            "total_tokens": self.total_tokens.to_dict(),
            "tokens_per_page": round(self.tokens_per_page, 2),
            "stages": {
                "extraction": self.extraction.to_dict(),
                "vision": self.vision.to_dict(),
                "summarization": self.summarization.to_dict(),
            },
            "config": self.config,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


class JobMetricsLogger:
    """Logger for writing job metrics to JSONL file."""

    def __init__(self, logger_name: str = "app.pipeline.metrics"):
        """Initialize the metrics logger.

        Args:
            logger_name: Name of the logger to use for metrics
        """
        self._logger = logging.getLogger(logger_name)

    def log_job_summary(self, metrics: JobMetrics) -> None:
        """Log job summary metrics.

        Args:
            metrics: JobMetrics to log
        """
        self._logger.info(
            "Job summary - job_id=%s, status=%s, pages=%d, duration=%.2fms, tokens=%d",
            metrics.job_id,
            metrics.status,
            metrics.total_pages,
            metrics.total_duration_ms,
            metrics.total_tokens.total_tokens,
            extra={"job_metrics": metrics.to_dict()},
        )


class JobMetricsAggregator:
    """Collects metrics during pipeline execution."""

    def __init__(self, job_id: str, filename: str, config: dict[str, Any]):
        """Initialize the aggregator.

        Args:
            job_id: Job identifier
            filename: Input filename
            config: Pipeline configuration
        """
        self.metrics = JobMetrics(
            job_id=job_id,
            filename=filename,
            status="processing",
            config=config,
            started_at=_utc_now_iso(),
        )

    def set_extraction_metrics(
        self,
        duration_ms: float,
        pages_processed: int,
        tokens: Optional[TokenMetrics] = None,
        error: Optional[str] = None,
    ) -> None:
        """Set extraction stage metrics."""
        self.metrics.extraction = StageMetrics(
            duration_ms=duration_ms,
            pages_processed=pages_processed,
            tokens=tokens or TokenMetrics(),
            success=error is None,
            error=error,
        )
        self.metrics.total_pages = pages_processed

    def set_vision_metrics(
        self,
        duration_ms: float,
        pages_processed: int,
        pages_with_images: int,
        classified_count: int,
        extracted_count: int,
        tokens: Optional[TokenMetrics] = None,
        error: Optional[str] = None,
    ) -> None:
        """Set vision stage metrics."""
        self.metrics.vision = StageMetrics(
            duration_ms=duration_ms,
            pages_processed=pages_processed,
            pages_with_images=pages_with_images,
            classified_count=classified_count,
            extracted_count=extracted_count,
            tokens=tokens or TokenMetrics(),
            success=error is None,
            error=error,
        )

    def set_summarization_metrics(
        self,
        duration_ms: float,
        pages_processed: int,
        avg_relevancy: float,
        total_attempts: int,
        tokens: Optional[TokenMetrics] = None,
        error: Optional[str] = None,
    ) -> None:
        """Set summarization stage metrics."""
        self.metrics.summarization = StageMetrics(
            duration_ms=duration_ms,
            pages_processed=pages_processed,
            avg_relevancy=avg_relevancy,
            total_attempts=total_attempts,
            tokens=tokens or TokenMetrics(),
            success=error is None,
            error=error,
        )

    def finalize(
        self,
        status: str,
        total_duration_ms: float,
        error: Optional[str] = None,
    ) -> JobMetrics:
        """Finalize metrics after job completion.

        Args:
            status: Final job status (completed, failed)
            total_duration_ms: Total job duration
            error: Error message if failed

        Returns:
            Finalized JobMetrics
        """
        self.metrics.status = status
        self.metrics.total_duration_ms = total_duration_ms
        self.metrics.completed_at = _utc_now_iso()
        self.metrics.error = error
        return self.metrics
