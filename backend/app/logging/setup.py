"""Logging setup for the document summarizer pipeline.

Configures async-safe logging with QueueHandler/QueueListener pattern,
per-stage log files, and size-based rotation.
"""

import atexit
import json
import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Optional

from app.config import settings
from app.logging.correlation import JobIdFilter, StageFilter


class JobMetricsFilter(logging.Filter):
    """Filter that only passes log records with job_metrics data."""

    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, "job_metrics") and record.job_metrics is not None


class JobMetricsFormatter(logging.Formatter):
    """Formatter that outputs job_metrics as a single JSON line (NDJSON format)."""

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "job_metrics") and record.job_metrics:
            return json.dumps(record.job_metrics, separators=(",", ":"))
        return ""

# Module-level listener reference for shutdown
_queue_listener: Optional[QueueListener] = None
_log_queue: Optional[queue.Queue] = None


def setup_logging() -> QueueListener:
    """Set up async-safe logging with per-stage files.

    Creates:
    - logs/extraction.log - Extraction stage logs
    - logs/vision.log - Vision processing logs
    - logs/summarization.log - Summarization logs
    - logs/pipeline.log - Combined pipeline logs
    - Console output for all logs

    Returns:
        QueueListener that should be started and stopped with the application
    """
    global _queue_listener, _log_queue

    # Create logs directory
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create queue for async-safe logging
    _log_queue = queue.Queue(-1)  # No size limit

    # Create formatter
    formatter = logging.Formatter(
        fmt=settings.log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create handlers for each stage
    handlers = []

    # Extraction handler
    extraction_handler = RotatingFileHandler(
        log_dir / "extraction.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    extraction_handler.setFormatter(formatter)
    extraction_handler.setLevel(getattr(logging, settings.effective_log_level_extraction))
    extraction_handler.addFilter(StageFilter("extraction"))
    handlers.append(extraction_handler)

    # Vision handler
    vision_handler = RotatingFileHandler(
        log_dir / "vision.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    vision_handler.setFormatter(formatter)
    vision_handler.setLevel(getattr(logging, settings.effective_log_level_vision))
    vision_handler.addFilter(StageFilter("vision"))
    handlers.append(vision_handler)

    # Summarization handler
    summarization_handler = RotatingFileHandler(
        log_dir / "summarization.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    summarization_handler.setFormatter(formatter)
    summarization_handler.setLevel(getattr(logging, settings.effective_log_level_summarization))
    summarization_handler.addFilter(StageFilter("summarization"))
    handlers.append(summarization_handler)

    # Combined pipeline handler (all stages)
    pipeline_handler = RotatingFileHandler(
        log_dir / "pipeline.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    pipeline_handler.setFormatter(formatter)
    pipeline_handler.setLevel(getattr(logging, settings.log_level))
    handlers.append(pipeline_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.log_level))
    handlers.append(console_handler)

    # Job metrics handler (NDJSON format for job summaries)
    metrics_handler = RotatingFileHandler(
        log_dir / "jobs.metrics.jsonl",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    metrics_handler.setFormatter(JobMetricsFormatter())
    metrics_handler.setLevel(logging.INFO)
    metrics_handler.addFilter(JobMetricsFilter())
    handlers.append(metrics_handler)

    # Create queue listener (runs handlers on separate thread)
    _queue_listener = QueueListener(
        _log_queue,
        *handlers,
        respect_handler_level=True,
    )

    # Configure root logger with queue handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Let handlers filter

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create queue handler with JobIdFilter
    # The filter must be on the queue handler to inject job_id before record goes to queue
    queue_handler = QueueHandler(_log_queue)
    queue_handler.addFilter(JobIdFilter())
    root_logger.addHandler(queue_handler)

    # Register shutdown handler
    atexit.register(shutdown_logging)

    return _queue_listener


def shutdown_logging() -> None:
    """Shutdown logging gracefully.

    Stops the queue listener, ensuring all pending logs are written.
    """
    global _queue_listener

    if _queue_listener:
        _queue_listener.stop()
        _queue_listener = None


def get_stage_logger(stage: str) -> logging.Logger:
    """Get a logger for a specific pipeline stage.

    Args:
        stage: Stage name (extraction, vision, summarization)

    Returns:
        Logger configured for the stage
    """
    return logging.getLogger(f"app.pipeline.{stage}")
