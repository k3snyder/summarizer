"""
Shared models for pipeline stages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PipelineStage(Enum):
    """Pipeline processing stages."""

    EXTRACTION = "extraction"
    VISION = "vision"
    SUMMARIZATION = "summarization"


@dataclass
class ProgressEvent:
    """Real-time progress update from pipeline stages.

    Event types:
    - started: Stage has begun
    - progress: Work in progress with current/total counts
    - stage_changed: Moving to a new stage
    - completed: Pipeline finished successfully
    - failed: Pipeline encountered an error
    """

    event_type: str
    stage: PipelineStage
    message: Optional[str] = None
    current: Optional[int] = None
    total: Optional[int] = None
    error: Optional[str] = None

    @property
    def percentage(self) -> float:
        """Calculate progress percentage. Returns 0.0 if total is 0 or None."""
        if self.total is None or self.total == 0:
            return 0.0
        if self.current is None:
            return 0.0
        return (self.current / self.total) * 100.0
