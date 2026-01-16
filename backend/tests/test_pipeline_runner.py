"""
Tests for updated PipelineRunner that uses orchestrator directly.
TDD red phase for bsc-4.5.
"""

import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from contextlib import asynccontextmanager


@pytest.fixture
def mock_session():
    """Mock database session that works with async context manager."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_config():
    """Sample pipeline configuration."""
    from app.models.config import PipelineConfig

    return PipelineConfig(
        extract_only=False,
        text_only=False,
        skip_tables=False,
        skip_images=False,
        pdf_image_dpi=200,
        vision_mode="ollama",
        summarizer_mode="full",
    )


@pytest.fixture
def mock_orchestrator():
    """Mock PipelineOrchestrator."""
    orchestrator = AsyncMock()
    orchestrator.run = AsyncMock(
        return_value={
            "document": {
                "document_id": "doc_123",
                "filename": "test.pdf",
                "total_pages": 2,
            },
            "pages": [
                {"chunk_id": "chunk_1", "summary_notes": ["Note 1"]},
                {"chunk_id": "chunk_2", "summary_notes": ["Note 2"]},
            ],
        }
    )
    return orchestrator


def create_mock_session_local(mock_session):
    """Create a mock AsyncSessionLocal that returns mock_session."""
    @asynccontextmanager
    async def mock_session_local():
        yield mock_session
    return mock_session_local


class TestPipelineRunnerUsesOrchestrator:
    """Test that PipelineRunner uses orchestrator directly."""

    @pytest.mark.asyncio
    async def test_run_calls_orchestrator(
        self, mock_session, mock_config, mock_orchestrator, tmp_path
    ):
        """Test that run() calls orchestrator.run()."""
        from app.services.pipeline_runner import PipelineRunner

        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        with patch(
            "app.services.pipeline_runner.PipelineOrchestrator",
            return_value=mock_orchestrator,
        ), patch(
            "app.services.pipeline_runner.AsyncSessionLocal",
            create_mock_session_local(mock_session),
        ), patch("app.services.pipeline_runner.JobManager") as mock_job_manager:
            mock_job_manager.update_status = AsyncMock()
            mock_job_manager.set_output = AsyncMock()

            runner = PipelineRunner()
            await runner.run(
                job_id="job_123",
                file_path=str(test_file),
                config=mock_config,
            )

            mock_orchestrator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_passes_config_to_orchestrator(
        self, mock_session, mock_config, mock_orchestrator, tmp_path
    ):
        """Test that config is properly passed to orchestrator."""
        from app.services.pipeline_runner import PipelineRunner

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        with patch(
            "app.services.pipeline_runner.PipelineOrchestrator",
            return_value=mock_orchestrator,
        ), patch(
            "app.services.pipeline_runner.AsyncSessionLocal",
            create_mock_session_local(mock_session),
        ), patch("app.services.pipeline_runner.JobManager") as mock_job_manager:
            mock_job_manager.update_status = AsyncMock()
            mock_job_manager.set_output = AsyncMock()

            runner = PipelineRunner()
            await runner.run(
                job_id="job_123",
                file_path=str(test_file),
                config=mock_config,
            )

            # Check orchestrator was called with config dict
            call_kwargs = mock_orchestrator.run.call_args.kwargs
            assert "config" in call_kwargs
            assert call_kwargs["config"]["vision_mode"] == "ollama"

    @pytest.mark.asyncio
    async def test_run_saves_output_via_file_storage(
        self, mock_session, mock_config, mock_orchestrator, tmp_path
    ):
        """Test that output is saved via FileStorage."""
        from app.services.pipeline_runner import PipelineRunner

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        with patch(
            "app.services.pipeline_runner.PipelineOrchestrator",
            return_value=mock_orchestrator,
        ), patch(
            "app.services.pipeline_runner.AsyncSessionLocal",
            create_mock_session_local(mock_session),
        ), patch(
            "app.services.pipeline_runner.FileStorage"
        ) as mock_storage, patch(
            "app.services.pipeline_runner.JobManager"
        ) as mock_job_manager:
            mock_storage.save_output = AsyncMock(return_value=Path("/tmp/output.json"))
            mock_job_manager.update_status = AsyncMock()
            mock_job_manager.set_output = AsyncMock()

            runner = PipelineRunner()
            await runner.run(
                job_id="job_123",
                file_path=str(test_file),
                config=mock_config,
            )

            # FileStorage.save_output should be called with output data
            mock_storage.save_output.assert_called_once()


class TestPipelineRunnerProgressUpdates:
    """Test progress callback updates job status."""

    @pytest.mark.asyncio
    async def test_run_updates_job_progress(
        self, mock_session, mock_config, tmp_path
    ):
        """Test that job status is updated during processing."""
        from app.services.pipeline_runner import PipelineRunner
        from app.pipeline.models import ProgressEvent, PipelineStage

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        async def mock_run(**kwargs):
            """Simulate orchestrator with progress callbacks."""
            progress_callback = kwargs.get("progress_callback")
            if progress_callback:
                await progress_callback(
                    ProgressEvent(
                        event_type="started",
                        stage=PipelineStage.EXTRACTION,
                        message="Starting extraction",
                    )
                )
            return {"document": {}, "pages": []}

        mock_orchestrator = AsyncMock()
        mock_orchestrator.run = mock_run

        with patch(
            "app.services.pipeline_runner.PipelineOrchestrator",
            return_value=mock_orchestrator,
        ), patch(
            "app.services.pipeline_runner.AsyncSessionLocal",
            create_mock_session_local(mock_session),
        ), patch(
            "app.services.pipeline_runner.FileStorage"
        ) as mock_storage, patch(
            "app.services.pipeline_runner.JobManager"
        ) as mock_job_manager:
            mock_storage.save_output = AsyncMock(return_value=Path("/tmp/output.json"))
            mock_job_manager.update_status = AsyncMock()
            mock_job_manager.set_output = AsyncMock()

            runner = PipelineRunner()
            await runner.run(
                job_id="job_123",
                file_path=str(test_file),
                config=mock_config,
            )

            # JobManager.update_status should have been called
            assert mock_job_manager.update_status.call_count >= 1


class TestPipelineRunnerErrorHandling:
    """Test error handling in PipelineRunner."""

    @pytest.mark.asyncio
    async def test_run_handles_orchestrator_failure(self, mock_session, mock_config, tmp_path):
        """Test that orchestrator failures are handled."""
        from app.services.pipeline_runner import PipelineRunner

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        failing_orchestrator = AsyncMock()
        failing_orchestrator.run = AsyncMock(side_effect=Exception("Pipeline failed"))

        with patch(
            "app.services.pipeline_runner.PipelineOrchestrator",
            return_value=failing_orchestrator,
        ), patch(
            "app.services.pipeline_runner.AsyncSessionLocal",
            create_mock_session_local(mock_session),
        ), patch("app.services.pipeline_runner.JobManager") as mock_job_manager:
            mock_job_manager.update_status = AsyncMock()

            runner = PipelineRunner()
            await runner.run(
                job_id="job_123",
                file_path=str(test_file),
                config=mock_config,
            )

            # Job should be marked as failed
            failed_calls = [
                c for c in mock_job_manager.update_status.call_args_list
                if c.kwargs.get("status") == "failed"
            ]
            assert len(failed_calls) >= 1

    @pytest.mark.asyncio
    async def test_run_handles_file_not_found(self, mock_session, mock_config):
        """Test that missing file is handled."""
        from app.services.pipeline_runner import PipelineRunner

        with patch(
            "app.services.pipeline_runner.AsyncSessionLocal",
            create_mock_session_local(mock_session),
        ), patch("app.services.pipeline_runner.JobManager") as mock_job_manager, \
             patch("app.services.pipeline_runner.PipelineOrchestrator") as mock_orch:
            mock_job_manager.update_status = AsyncMock()
            mock_orch.return_value.run = AsyncMock(
                side_effect=FileNotFoundError("File not found")
            )

            runner = PipelineRunner()
            await runner.run(
                job_id="job_123",
                file_path="/nonexistent/file.pdf",
                config=mock_config,
            )

            # Job should be marked as failed
            failed_calls = [
                c for c in mock_job_manager.update_status.call_args_list
                if c.kwargs.get("status") == "failed"
            ]
            assert len(failed_calls) >= 1


class TestPipelineRunnerNoSubprocess:
    """Test that PipelineRunner no longer uses subprocess."""

    def test_runner_has_no_subprocess_code(self):
        """Test that PipelineRunner doesn't import subprocess."""
        from app.services import pipeline_runner
        import inspect

        source = inspect.getsource(pipeline_runner)

        # Should not have subprocess-related code
        assert "subprocess" not in source
        assert "create_subprocess_exec" not in source

    def test_runner_has_no_run_py_path(self):
        """Test that PipelineRunner doesn't reference run.py."""
        from app.services import pipeline_runner
        import inspect

        source = inspect.getsource(pipeline_runner)

        # Should not reference run.py
        assert "run.py" not in source
