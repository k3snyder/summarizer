"""Performance metrics logging utilities.

Provides timing decorators and token extraction for measuring
pipeline performance and LLM API usage.
"""

import contextlib
import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@contextlib.asynccontextmanager
async def timed_operation(
    operation_name: str,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    **context_data: Any,
):
    """Async context manager for timing operations with structured logging.

    Args:
        operation_name: Name of the operation being timed
        logger_instance: Logger to use (defaults to module logger)
        level: Log level for the timing message
        **context_data: Additional context to include in the log

    Yields:
        Dict that can be updated with additional metrics during the operation

    Example:
        async with timed_operation("vision_classification", pages=10) as metrics:
            results = await classify_pages(pages)
            metrics["pages_processed"] = len(results)
    """
    log = logger_instance or logger
    start = time.perf_counter()
    result_data = dict(context_data)

    try:
        yield result_data
    except Exception as e:
        result_data["error"] = str(e)
        raise
    finally:
        elapsed = time.perf_counter() - start
        result_data["operation"] = operation_name
        result_data["duration_seconds"] = round(elapsed, 4)
        result_data["duration_ms"] = round(elapsed * 1000, 2)

        log.log(
            level,
            "%s completed in %.2fms",
            operation_name,
            elapsed * 1000,
            extra={"metrics": result_data},
        )


def async_timed(operation_name: str, level: int = logging.DEBUG) -> Callable[[F], F]:
    """Decorator for timing async functions.

    Args:
        operation_name: Name of the operation for logging
        level: Log level for the timing message

    Returns:
        Decorated function that logs timing on completion

    Example:
        @async_timed("extract_page")
        async def extract_page(page_num: int) -> PageData:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                logger.log(
                    level,
                    "%s completed in %.2fms",
                    operation_name,
                    elapsed * 1000,
                    extra={
                        "metrics": {
                            "operation": operation_name,
                            "duration_seconds": round(elapsed, 4),
                            "duration_ms": round(elapsed * 1000, 2),
                        }
                    },
                )

        return wrapper  # type: ignore

    return decorator


def extract_token_usage(response: Any) -> Optional[dict[str, int]]:
    """Extract token usage from API response.

    Handles both OpenAI SDK response objects and raw JSON dicts.

    Args:
        response: API response (OpenAI ChatCompletion or dict)

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens
        or None if not available
    """
    # Handle OpenAI SDK response object
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    # Handle dict response (Ollama/raw JSON)
    if isinstance(response, dict) and "usage" in response:
        usage = response["usage"]
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }

    return None


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for text.

    Uses ~1.3 tokens per word as a rough estimate.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


class MetricsLogger:
    """Structured metrics logger for pipeline performance tracking."""

    def __init__(self, name: str = "metrics"):
        """Initialize metrics logger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)

    def log_stage_timing(
        self,
        stage: str,
        duration_seconds: float,
        page_count: int,
        success: bool,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log timing for a pipeline stage.

        Args:
            stage: Stage name (extraction, vision, summarization)
            duration_seconds: Time taken in seconds
            page_count: Number of pages processed
            success: Whether the stage completed successfully
            error: Error message if failed
            **kwargs: Additional metrics
        """
        metrics = {
            "event": "stage_completed",
            "stage": stage,
            "duration_seconds": round(duration_seconds, 4),
            "duration_ms": round(duration_seconds * 1000, 2),
            "pages_processed": page_count,
            "avg_ms_per_page": (
                round((duration_seconds * 1000) / page_count, 2)
                if page_count > 0
                else 0
            ),
            "success": success,
            "error": error,
            **kwargs,
        }
        self.logger.info(
            "%s stage completed: %d pages in %.2fms",
            stage,
            page_count,
            duration_seconds * 1000,
            extra={"metrics": metrics},
        )

    def log_api_call(
        self,
        provider: str,
        operation: str,
        model: str,
        duration_seconds: float,
        retry_count: int = 0,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log individual API call metrics.

        Args:
            provider: API provider (ollama, openai)
            operation: Operation type (classify, extract, summarize)
            model: Model used
            duration_seconds: API call duration
            retry_count: Number of retries needed
            prompt_tokens: Input tokens (if available)
            completion_tokens: Output tokens (if available)
            error: Error message if failed
            **kwargs: Additional metrics
        """
        metrics = {
            "event": "api_call",
            "provider": provider,
            "operation": operation,
            "model": model,
            "duration_seconds": round(duration_seconds, 4),
            "duration_ms": round(duration_seconds * 1000, 2),
            "retry_count": retry_count,
            "error": error,
            **kwargs,
        }

        if prompt_tokens is not None or completion_tokens is not None:
            metrics["tokens"] = {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": (prompt_tokens or 0) + (completion_tokens or 0),
            }

        level = logging.WARNING if error else logging.DEBUG
        self.logger.log(
            level,
            "%s %s via %s (%s) in %.2fms",
            operation,
            "failed" if error else "completed",
            provider,
            model,
            duration_seconds * 1000,
            extra={"metrics": metrics},
        )
