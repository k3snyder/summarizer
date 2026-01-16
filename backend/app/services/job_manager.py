"""Job management service for CRUD operations"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import Job
from app.models.config import PipelineConfig
from app.models.job import JobStatus
from app.services.file_storage import FileStorage


class JobManager:
    """
    Manages job lifecycle and persistence.

    Provides CRUD operations for processing jobs, including status updates,
    output storage, and cleanup.
    """

    @staticmethod
    async def create(
        db: AsyncSession,
        job_id: str,
        file_name: str,
        file_path: str,
        config: PipelineConfig
    ) -> JobStatus:
        """
        Create a new job with pending status.

        Args:
            db: Database session
            job_id: Unique job identifier
            file_name: Original filename
            file_path: Path to uploaded file
            config: Pipeline configuration

        Returns:
            JobStatus with initial pending state
        """
        # Create job record
        job = Job(
            id=job_id,
            status="pending",
            progress=0,
            current_stage=None,
            message=f"Job created. Processing {file_name}",
            file_name=file_name,
            file_path=file_path,
            config_json=config.model_dump_json(),
            output_path=None,
            created_at=datetime.now(timezone.utc),
            started_at=None,
            completed_at=None,
            error=None
        )

        # Add to database
        db.add(job)
        await db.commit()
        await db.refresh(job)

        # Return as JobStatus
        return JobStatus(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            current_stage=job.current_stage,
            message=job.message,
            file_name=job.file_name,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error
        )

    @staticmethod
    async def get(db: AsyncSession, job_id: str) -> Optional[JobStatus]:
        """
        Retrieve a job by ID.

        Args:
            db: Database session
            job_id: Job identifier

        Returns:
            JobStatus if found, None otherwise
        """
        # Query database
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

        if job is None:
            return None

        # Convert to JobStatus
        return JobStatus(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            current_stage=job.current_stage,
            message=job.message,
            file_name=job.file_name,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error
        )

    @staticmethod
    async def update_status(
        db: AsyncSession,
        job_id: str,
        status: str,
        progress: int,
        stage: Optional[str] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Update job status and progress.

        Args:
            db: Database session
            job_id: Job identifier
            status: New status (pending, processing, completed, failed)
            progress: Progress percentage (0-100)
            stage: Current processing stage (extraction, vision, summarization)
            message: Human-readable status message
        """
        # Get job
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()

        # Update fields
        job.status = status
        job.progress = progress

        if stage is not None:
            job.current_stage = stage

        if message is not None:
            job.message = message

        # Set started_at if transitioning to processing
        if status == "processing" and job.started_at is None:
            job.started_at = datetime.now(timezone.utc)

        # Set completed_at if finished
        if status in ("completed", "failed") and job.completed_at is None:
            job.completed_at = datetime.now(timezone.utc)

        # Commit changes
        await db.commit()

    @staticmethod
    async def set_output(
        db: AsyncSession,
        job_id: str,
        output_path: str
    ) -> None:
        """
        Store output file path for a job.

        Args:
            db: Database session
            job_id: Job identifier
            output_path: Path to output JSON file
        """
        # Get job
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one()

        # Update output path
        job.output_path = output_path

        # Commit changes
        await db.commit()

    @staticmethod
    async def get_output(db: AsyncSession, job_id: str) -> Optional[dict]:
        """
        Retrieve output JSON for a completed job.

        Args:
            db: Database session
            job_id: Job identifier

        Returns:
            Output data as dictionary, None if no output exists
        """
        # Get job
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

        if job is None or job.output_path is None:
            return None

        # Read output file
        output_path = Path(job.output_path)
        if not output_path.exists():
            return None

        with open(output_path, "r") as f:
            return json.load(f)

    @staticmethod
    async def delete(db: AsyncSession, job_id: str) -> None:
        """
        Delete a job and clean up associated files.

        Args:
            db: Database session
            job_id: Job identifier
        """
        # Get job
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

        if job is None:
            return

        # Delete from database
        await db.delete(job)
        await db.commit()

        # Clean up files
        FileStorage.cleanup(job_id)
