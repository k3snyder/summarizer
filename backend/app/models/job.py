"""Job status and response models"""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class JobStatus(BaseModel):
    """
    Current status and progress of a processing job.

    Tracks job lifecycle from creation through completion or failure.
    """

    job_id: str = Field(
        description="Unique identifier for the job"
    )

    status: Literal['pending', 'processing', 'completed', 'failed'] = Field(
        description="Current job status"
    )

    progress: int = Field(
        description="Progress percentage (0-100)",
        ge=0,
        le=100
    )

    current_stage: Optional[str] = Field(
        default=None,
        description="Current processing stage (extraction, vision, summarization)"
    )

    message: Optional[str] = Field(
        default=None,
        description="Human-readable status message"
    )

    file_name: str = Field(
        description="Original filename of the document being processed"
    )

    created_at: datetime = Field(
        description="Job creation timestamp"
    )

    started_at: Optional[datetime] = Field(
        default=None,
        description="Job processing start timestamp"
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="Job completion timestamp"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if job failed"
    )


class JobCreateResponse(BaseModel):
    """
    Response from job creation endpoint.

    Provides job ID and initial status information.
    """

    job_id: str = Field(
        description="Unique identifier for the created job"
    )

    status: str = Field(
        description="Initial job status (typically 'pending')"
    )

    message: str = Field(
        description="Human-readable confirmation message"
    )
