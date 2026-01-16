"""
Tests for job-scoped file storage.
TDD red phase for bsc-4.3.
"""

import pytest
import tempfile
from pathlib import Path
from io import BytesIO
import json


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directory."""
    return tmp_path / "job_storage"


class TestFileStorageJobDirectory:
    """Test job directory creation and paths."""

    def test_get_path_returns_job_specific_directory(self, temp_storage_dir, monkeypatch):
        """Test that get_path returns /tmp/summarizer-jobs/{job_id}/"""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "test_job_123"
        path = FileStorage.get_path(job_id)

        assert str(temp_storage_dir) in str(path)
        assert job_id in str(path)

    def test_get_path_is_consistent(self, temp_storage_dir, monkeypatch):
        """Test that get_path returns same path for same job_id."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "consistent_job"
        path1 = FileStorage.get_path(job_id)
        path2 = FileStorage.get_path(job_id)

        assert path1 == path2


class TestFileStorageSave:
    """Test file saving functionality."""

    @pytest.mark.asyncio
    async def test_save_creates_job_directory(self, temp_storage_dir, monkeypatch):
        """Test that save creates job directory if not exists."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "new_job"
        file = BytesIO(b"test content")
        filename = "test.pdf"

        await FileStorage.save(job_id, file, filename)

        job_dir = FileStorage.get_path(job_id)
        assert job_dir.exists()

    @pytest.mark.asyncio
    async def test_save_returns_file_path(self, temp_storage_dir, monkeypatch):
        """Test that save returns the correct file path."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "path_test"
        file = BytesIO(b"content")
        filename = "document.pdf"

        result_path = await FileStorage.save(job_id, file, filename)

        assert result_path.exists()
        assert result_path.name == filename
        assert job_id in str(result_path)

    @pytest.mark.asyncio
    async def test_save_file_content_is_preserved(self, temp_storage_dir, monkeypatch):
        """Test that file content is preserved during save."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "content_test"
        content = b"This is test content for verification"
        file = BytesIO(content)
        filename = "test.txt"

        result_path = await FileStorage.save(job_id, file, filename)

        with open(result_path, "rb") as f:
            saved_content = f.read()

        assert saved_content == content


class TestFileStorageOutput:
    """Test output file handling."""

    @pytest.mark.asyncio
    async def test_save_output_creates_output_file(self, temp_storage_dir, monkeypatch):
        """Test that save_output creates the output file."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "output_test"
        output_data = {"document": {"id": "123"}, "pages": []}

        # Create job directory first
        job_dir = FileStorage.get_path(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        await FileStorage.save_output(job_id, output_data)

        output_path = FileStorage.get_output_path(job_id)
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_save_output_content_is_json(self, temp_storage_dir, monkeypatch):
        """Test that output file is valid JSON."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "json_test"
        output_data = {"document": {"id": "doc_1"}, "pages": [{"chunk_id": "1"}]}

        job_dir = FileStorage.get_path(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        await FileStorage.save_output(job_id, output_data)

        output_path = FileStorage.get_output_path(job_id)
        with open(output_path, "r") as f:
            loaded = json.load(f)

        assert loaded == output_data

    def test_get_output_path_returns_json_file(self, temp_storage_dir, monkeypatch):
        """Test that get_output_path returns path to output.json."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "output_path_test"
        output_path = FileStorage.get_output_path(job_id)

        assert output_path.name == "output.json"
        assert job_id in str(output_path)


class TestFileStorageCleanup:
    """Test cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_job_directory(self, temp_storage_dir, monkeypatch):
        """Test that cleanup removes the entire job directory."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "cleanup_test"
        file = BytesIO(b"content")
        await FileStorage.save(job_id, file, "test.pdf")

        job_dir = FileStorage.get_path(job_id)
        assert job_dir.exists()

        FileStorage.cleanup(job_id)

        assert not job_dir.exists()

    def test_cleanup_handles_nonexistent_directory(self, temp_storage_dir, monkeypatch):
        """Test that cleanup handles non-existent directory gracefully."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        # Should not raise an exception
        FileStorage.cleanup("nonexistent_job_id")


class TestFileStorageList:
    """Test file listing functionality."""

    @pytest.mark.asyncio
    async def test_list_files_returns_saved_files(self, temp_storage_dir, monkeypatch):
        """Test that list_files returns all files in job directory."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        job_id = "list_test"
        await FileStorage.save(job_id, BytesIO(b"content1"), "file1.pdf")
        await FileStorage.save(job_id, BytesIO(b"content2"), "file2.txt")

        files = FileStorage.list_files(job_id)

        filenames = [f.name for f in files]
        assert "file1.pdf" in filenames
        assert "file2.txt" in filenames

    def test_list_files_returns_empty_for_nonexistent_job(
        self, temp_storage_dir, monkeypatch
    ):
        """Test that list_files returns empty list for non-existent job."""
        from app.config import settings
        from app.services.file_storage import FileStorage

        monkeypatch.setattr(settings, "job_temp_dir", str(temp_storage_dir))

        files = FileStorage.list_files("nonexistent_job")

        assert files == []
