"""
Background cleanup task for removing old job directories.

Periodically scans the job temp directory and removes jobs older
than the configured max age.
"""

import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.database import Job

logger = logging.getLogger(__name__)

# Global task reference for cleanup
_cleanup_task: Optional[asyncio.Task] = None


async def cleanup_old_jobs(db: AsyncSession, max_age_hours: int = 24) -> int:
    """
    Remove job directories and database records older than max_age_hours.

    Args:
        db: Database session
        max_age_hours: Maximum age in hours before cleanup (default 24)

    Returns:
        Number of jobs deleted
    """
    cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

    # Query for old jobs
    result = await db.execute(
        select(Job).where(Job.created_at < cutoff_time)
    )
    old_jobs = result.scalars().all()

    deleted_count = 0

    for job in old_jobs:
        # Remove job directory if it exists
        job_dir = Path(settings.job_temp_dir) / job.id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up job directory: {job_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove job directory {job_dir}: {e}")

        # Remove database record
        await db.delete(job)
        deleted_count += 1

    if deleted_count > 0:
        await db.commit()
        logger.info(f"Cleaned up {deleted_count} old jobs")

    return deleted_count


async def _cleanup_loop() -> None:
    """
    Background loop that periodically runs cleanup.
    """
    from app.db.database import async_session_factory

    interval_seconds = settings.cleanup_interval_minutes * 60

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            async with async_session_factory() as db:
                await cleanup_old_jobs(db, max_age_hours=settings.cleanup_max_age_hours)

        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            # Continue running even on error


def start_cleanup_task() -> None:
    """
    Start the background cleanup task.

    Should be called from the FastAPI startup event.
    """
    global _cleanup_task

    if _cleanup_task is not None:
        logger.warning("Cleanup task already running")
        return

    _cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info(
        f"Started cleanup task (interval: {settings.cleanup_interval_minutes}min, "
        f"max_age: {settings.cleanup_max_age_hours}h)"
    )


def stop_cleanup_task() -> None:
    """
    Stop the background cleanup task.

    Should be called from the FastAPI shutdown event.
    """
    global _cleanup_task

    if _cleanup_task is None:
        return

    _cleanup_task.cancel()
    _cleanup_task = None
    logger.info("Stopped cleanup task")
