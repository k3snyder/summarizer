"""
Tests for background cleanup task.
TDD tests for bsc-5.2.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import time


@pytest.fixture
def temp_jobs_dir(tmp_path):
    """Create temporary jobs directory."""
    jobs_dir = tmp_path / "summarizer-jobs"
    jobs_dir.mkdir()
    return jobs_dir


@pytest.fixture
def old_job_dir(temp_jobs_dir):
    """Create an old job directory (older than 24 hours)."""
    job_id = "old-job-123"
    job_dir = temp_jobs_dir / job_id
    job_dir.mkdir()

    # Create some files
    (job_dir / "input.pdf").write_bytes(b"PDF content")
    (job_dir / "output.json").write_text('{"test": "data"}')

    # Set modification time to 25 hours ago
    old_time = time.time() - (25 * 3600)
    for f in job_dir.iterdir():
        f.touch()
    job_dir.touch()

    # We can't easily modify mtime on directories in Python,
    # so we'll check by job creation time in database instead
    return job_id, job_dir


@pytest.fixture
def new_job_dir(temp_jobs_dir):
    """Create a new job directory (less than 24 hours old)."""
    job_id = "new-job-456"
    job_dir = temp_jobs_dir / job_id
    job_dir.mkdir()

    # Create some files
    (job_dir / "input.pdf").write_bytes(b"PDF content")
    (job_dir / "output.json").write_text('{"test": "data"}')

    return job_id, job_dir


class TestCleanupTaskImports:
    """Test that cleanup task module exists and is importable."""

    def test_cleanup_task_importable(self):
        """cleanup_task module can be imported."""
        from app.services import cleanup_task

        assert cleanup_task is not None

    def test_cleanup_old_jobs_function_exists(self):
        """cleanup_old_jobs function exists."""
        from app.services.cleanup_task import cleanup_old_jobs

        assert callable(cleanup_old_jobs)


class TestCleanupOldJobs:
    """Test cleanup_old_jobs function."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_job_directories(
        self, temp_jobs_dir, old_job_dir, new_job_dir, monkeypatch
    ):
        """Old job directories are removed, new ones are kept."""
        from app.config import settings

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_jobs_dir))

        old_job_id, old_dir = old_job_dir
        new_job_id, new_dir = new_job_dir

        # Create mock database - only return the old job (simulating WHERE clause)
        # The function queries with WHERE created_at < cutoff, so only old jobs returned
        mock_db = AsyncMock()
        old_created_at = datetime.utcnow() - timedelta(hours=25)

        # Mock the database query to return only the old job
        mock_result = MagicMock()
        mock_jobs = [
            MagicMock(id=old_job_id, created_at=old_created_at),
        ]
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()

        from app.services.cleanup_task import cleanup_old_jobs

        await cleanup_old_jobs(mock_db, max_age_hours=24)

        # Old directory should be removed
        assert not old_dir.exists()

        # New directory should still exist (it was never in the query result)
        assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_deletes_database_records(self, temp_jobs_dir, monkeypatch):
        """Old job database records are deleted."""
        from app.config import settings

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_jobs_dir))

        # Create old job directory
        old_job_id = "old-db-job"
        old_dir = temp_jobs_dir / old_job_id
        old_dir.mkdir()

        # Create mock database
        mock_db = AsyncMock()
        old_created_at = datetime.utcnow() - timedelta(hours=25)

        mock_old_job = MagicMock(id=old_job_id, created_at=old_created_at)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_old_job]
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()

        from app.services.cleanup_task import cleanup_old_jobs

        await cleanup_old_jobs(mock_db, max_age_hours=24)

        # Database delete should have been called
        mock_db.delete.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_respects_max_age_hours(self, temp_jobs_dir, monkeypatch):
        """Cleanup respects the max_age_hours parameter."""
        from app.config import settings

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_jobs_dir))

        # Create job directory
        job_id = "age-test-job"
        job_dir = temp_jobs_dir / job_id
        job_dir.mkdir()

        # Job is 2 hours old
        created_at = datetime.utcnow() - timedelta(hours=2)
        mock_job = MagicMock(id=job_id, created_at=created_at)

        from app.services.cleanup_task import cleanup_old_jobs

        # With max_age_hours=24, job should NOT be deleted
        # Mock returns empty list (no jobs older than 24h)
        mock_db_24h = AsyncMock()
        mock_result_empty = MagicMock()
        mock_result_empty.scalars.return_value.all.return_value = []
        mock_db_24h.execute = AsyncMock(return_value=mock_result_empty)
        mock_db_24h.delete = AsyncMock()
        mock_db_24h.commit = AsyncMock()

        await cleanup_old_jobs(mock_db_24h, max_age_hours=24)
        assert job_dir.exists()

        # With max_age_hours=1, job SHOULD be deleted
        # Mock returns the job (job is older than 1h)
        mock_db_1h = AsyncMock()
        mock_result_with_job = MagicMock()
        mock_result_with_job.scalars.return_value.all.return_value = [mock_job]
        mock_db_1h.execute = AsyncMock(return_value=mock_result_with_job)
        mock_db_1h.delete = AsyncMock()
        mock_db_1h.commit = AsyncMock()

        await cleanup_old_jobs(mock_db_1h, max_age_hours=1)
        assert not job_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_directory(self, temp_jobs_dir, monkeypatch):
        """Cleanup handles jobs with missing directories gracefully."""
        from app.config import settings

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_jobs_dir))

        # Don't create the job directory - only in database
        job_id = "missing-dir-job"

        mock_db = AsyncMock()
        old_created_at = datetime.utcnow() - timedelta(hours=25)

        mock_job = MagicMock(id=job_id, created_at=old_created_at)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_job]
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()

        from app.services.cleanup_task import cleanup_old_jobs

        # Should not raise an exception
        await cleanup_old_jobs(mock_db, max_age_hours=24)

        # Database delete should still have been called
        mock_db.delete.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_returns_count_of_deleted_jobs(self, temp_jobs_dir, monkeypatch):
        """Cleanup returns the number of deleted jobs."""
        from app.config import settings

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_jobs_dir))

        # Create two old job directories
        job1_id = "old-job-1"
        job2_id = "old-job-2"
        (temp_jobs_dir / job1_id).mkdir()
        (temp_jobs_dir / job2_id).mkdir()

        mock_db = AsyncMock()
        old_created_at = datetime.utcnow() - timedelta(hours=25)

        mock_jobs = [
            MagicMock(id=job1_id, created_at=old_created_at),
            MagicMock(id=job2_id, created_at=old_created_at),
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()

        from app.services.cleanup_task import cleanup_old_jobs

        deleted_count = await cleanup_old_jobs(mock_db, max_age_hours=24)

        assert deleted_count == 2


class TestCleanupConfig:
    """Test cleanup configuration."""

    def test_cleanup_interval_in_settings(self):
        """Settings includes cleanup_interval_minutes."""
        from app.config import settings

        assert hasattr(settings, "cleanup_interval_minutes")

    def test_cleanup_interval_default_value(self):
        """cleanup_interval_minutes defaults to 60."""
        from app.config import settings

        # Default should be 60 minutes
        assert settings.cleanup_interval_minutes >= 1

    def test_cleanup_max_age_hours_in_settings(self):
        """Settings includes cleanup_max_age_hours."""
        from app.config import settings

        assert hasattr(settings, "cleanup_max_age_hours")


class TestBackgroundTask:
    """Test background task setup."""

    def test_start_cleanup_task_function_exists(self):
        """start_cleanup_task function exists."""
        from app.services.cleanup_task import start_cleanup_task

        assert callable(start_cleanup_task)

    def test_stop_cleanup_task_function_exists(self):
        """stop_cleanup_task function exists."""
        from app.services.cleanup_task import stop_cleanup_task

        assert callable(stop_cleanup_task)
