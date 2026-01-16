"""Job ID correlation for log tracing.

Uses contextvars to propagate job_id through async task boundaries,
allowing all logs from a single job to be traced across pipeline stages.
"""

import contextvars
import logging
from typing import Optional

# Context variable for job ID - automatically propagates through async boundaries
job_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "job_id", default="N/A"
)


def set_job_id(job_id: str) -> contextvars.Token[str]:
    """Set job_id for the current async context.

    Args:
        job_id: The unique job identifier

    Returns:
        Token that can be used to reset the context variable
    """
    return job_id_var.set(job_id)


def reset_job_id(token: contextvars.Token[str]) -> None:
    """Reset job_id using the token from set_job_id().

    Args:
        token: The token returned by set_job_id()
    """
    job_id_var.reset(token)


def get_job_id() -> str:
    """Get the current job_id from context.

    Returns:
        The current job_id or "N/A" if not set
    """
    return job_id_var.get()


class JobIdFilter(logging.Filter):
    """Logging filter that injects job_id into all log records.

    This filter automatically adds the current job_id from contextvars
    to every log record, enabling job-correlated logging without
    explicitly passing job_id through all function calls.

    Usage:
        # Add to root logger at startup
        root_logger = logging.getLogger()
        root_logger.addFilter(JobIdFilter())

        # Then in your code
        token = set_job_id("abc-123")
        try:
            logger.info("Processing")  # Will include [abc-123]
        finally:
            reset_job_id(token)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add job_id to the log record.

        Args:
            record: The log record to modify

        Returns:
            Always True (don't filter out any records)
        """
        record.job_id = job_id_var.get()
        return True


class StageFilter(logging.Filter):
    """Filter logs by pipeline stage name.

    Used to route logs to stage-specific files based on logger name.
    """

    def __init__(self, stage: str):
        """Initialize filter for a specific stage.

        Args:
            stage: Stage name to filter for (e.g., "extraction", "vision")
        """
        super().__init__()
        self.stage = stage

    def filter(self, record: logging.LogRecord) -> bool:
        """Only allow logs from the specified stage.

        Args:
            record: The log record to check

        Returns:
            True if the log is from this stage, False otherwise
        """
        return self.stage in record.name
