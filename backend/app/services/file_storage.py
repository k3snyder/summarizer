"""File storage service for uploaded documents"""

import json
import shutil
from pathlib import Path
from typing import Any, BinaryIO

import aiofiles

from app.config import settings


class FileStorage:
    """
    Manages file storage for processing jobs.

    Stores uploaded files in job-specific directories under the configured
    temp directory. Provides methods for saving, retrieving paths, and cleanup.

    Directory structure:
    /tmp/summarizer-jobs/{job_id}/
        input.pdf           # Uploaded file
        output.json         # Final enriched output
    """

    OUTPUT_FILENAME = "output.json"

    @staticmethod
    def get_path(job_id: str) -> Path:
        """
        Get the directory path for a job's files.

        Args:
            job_id: Unique job identifier

        Returns:
            Path object for the job's storage directory
        """
        return Path(settings.job_temp_dir) / job_id

    @staticmethod
    def get_output_path(job_id: str) -> Path:
        """
        Get the path for the job's output JSON file.

        Args:
            job_id: Unique job identifier

        Returns:
            Path object for the output.json file
        """
        return FileStorage.get_path(job_id) / FileStorage.OUTPUT_FILENAME

    @staticmethod
    async def save(job_id: str, file: BinaryIO, filename: str) -> Path:
        """
        Save an uploaded file to the job's directory.

        Creates the job directory if it doesn't exist, then streams the file
        contents asynchronously to avoid blocking.

        Args:
            job_id: Unique job identifier
            file: File-like object to read from
            filename: Original filename to preserve

        Returns:
            Path object pointing to the saved file
        """
        # Get job directory and create it
        job_dir = FileStorage.get_path(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Determine file path
        file_path = job_dir / filename

        # Stream file contents asynchronously
        async with aiofiles.open(file_path, "wb") as f:
            # Read in chunks to handle large files efficiently
            chunk_size = 8192
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)

        return file_path

    @staticmethod
    async def save_output(job_id: str, output_data: dict[str, Any]) -> Path:
        """
        Save the pipeline output as JSON.

        Args:
            job_id: Unique job identifier
            output_data: Output dict to save as JSON

        Returns:
            Path object pointing to the output.json file
        """
        output_path = FileStorage.get_output_path(job_id)

        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(output_data, indent=2, ensure_ascii=False))

        return output_path

    @staticmethod
    def list_files(job_id: str) -> list[Path]:
        """
        List all files in a job's directory.

        Args:
            job_id: Unique job identifier

        Returns:
            List of Path objects for files in the job directory
        """
        job_dir = FileStorage.get_path(job_id)
        if not job_dir.exists():
            return []

        return [f for f in job_dir.iterdir() if f.is_file()]

    @staticmethod
    def cleanup(job_id: str) -> None:
        """
        Remove the job's directory and all contents.

        Args:
            job_id: Unique job identifier
        """
        job_dir = FileStorage.get_path(job_id)
        if job_dir.exists():
            shutil.rmtree(job_dir)
