"""Jobs API endpoints"""

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.config import PipelineConfig
from app.models.job import JobCreateResponse, JobStatus
from app.services.file_storage import FileStorage
from app.services.job_manager import JobManager
from app.services.pipeline_runner import PipelineRunner


router = APIRouter()


async def run_pipeline_background(job_id: str, file_path: str, config: PipelineConfig):
    """Background task to execute the pipeline"""
    runner = PipelineRunner()
    await runner.run(job_id, file_path, config)


@router.post("/jobs", response_model=JobCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF, PPTX, TXT, or MD file to process")],
    config: Annotated[str, Form(description="Pipeline configuration as JSON string")],
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new document processing job.

    Accepts a file upload and pipeline configuration, stores the file,
    creates a job record, and returns the job ID.

    Args:
        file: Uploaded document file (PDF, PPTX, TXT, or MD)
        config: JSON string of PipelineConfig
        db: Database session (injected)

    Returns:
        JobCreateResponse with job_id, status, and message

    Raises:
        HTTPException 400: Invalid file type or config format
        HTTPException 413: File too large (>100MB)
    """
    # Validate filename exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    # Validate file type
    allowed_extensions = {".pdf", ".pptx", ".txt", ".md"}
    file_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if f".{file_ext}" not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Validate file size (100MB max)
    max_size = 100 * 1024 * 1024  # 100MB in bytes
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_size / 1024 / 1024}MB"
        )

    # Parse and validate config
    try:
        pipeline_config = PipelineConfig.model_validate_json(config)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid config format: {str(e)}"
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save file
    file_path = await FileStorage.save(job_id, file.file, file.filename)

    # Create job record
    job_status = await JobManager.create(
        db=db,
        job_id=job_id,
        file_name=file.filename,
        file_path=str(file_path),
        config=pipeline_config
    )

    # Start pipeline execution in background
    background_tasks.add_task(
        run_pipeline_background,
        job_id=job_id,
        file_path=str(file_path),
        config=pipeline_config
    )

    # Return response
    return JobCreateResponse(
        job_id=job_status.job_id,
        status=job_status.status,
        message=f"Job created. Processing {file.filename}"
    )


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get job status and progress.

    Args:
        job_id: Unique job identifier
        db: Database session (injected)

    Returns:
        JobStatus with current state, progress, and metadata

    Raises:
        HTTPException 404: Job not found
    """
    # Retrieve job
    job_status = await JobManager.get(db, job_id)

    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    return job_status


@router.get("/jobs/{job_id}/output")
async def get_job_output(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get processing results for completed job.

    Returns the full output JSON with document metadata, pages, and summaries.

    Args:
        job_id: Unique job identifier
        db: Database session (injected)

    Returns:
        Output JSON dictionary with document and pages

    Raises:
        HTTPException 404: Job not found
        HTTPException 400: Job not completed or output not available
    """
    # Retrieve job
    job_status = await JobManager.get(db, job_id)

    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Check if job is completed
    if job_status.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job not completed. Current status: {job_status.status}"
        )

    # Retrieve output
    output = await JobManager.get_output(db, job_id)

    if output is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Output not available for this job"
        )

    return output


def _generate_markdown_report(output: dict) -> str:
    """
    Generate a markdown report from pipeline output.

    Extracts summary_notes and summary_topics from each page into a
    structured markdown document.

    Args:
        output: Pipeline output dictionary with document and pages

    Returns:
        Formatted markdown string
    """
    document = output.get("document", {})
    pages = output.get("pages", [])
    filename = document.get("filename", "Unknown Document")

    lines = [f"# {filename}", ""]

    for i, page in enumerate(pages, start=1):
        summary_notes = page.get("summary_notes") or []
        summary_topics = page.get("summary_topics") or []

        # Skip pages without any summary content
        if not summary_notes and not summary_topics:
            continue

        lines.append(f"## Page {i}")
        lines.append("")

        if summary_topics:
            lines.append("### Topics")
            for topic in summary_topics:
                lines.append(f"- {topic}")
            lines.append("")

        if summary_notes:
            lines.append("### Summary Notes")
            for note in summary_notes:
                lines.append(f"- {note}")
            lines.append("")

    return "\n".join(lines)


@router.get("/jobs/{job_id}/output/markdown")
async def get_job_output_markdown(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get processing results as a markdown report.

    Extracts summary_notes and summary_topics from each page into a
    formatted markdown document for download.

    Args:
        job_id: Unique job identifier
        db: Database session (injected)

    Returns:
        Markdown file as attachment

    Raises:
        HTTPException 404: Job not found
        HTTPException 400: Job not completed or output not available
    """
    # Retrieve job
    job_status = await JobManager.get(db, job_id)

    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Check if job is completed
    if job_status.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job not completed. Current status: {job_status.status}"
        )

    # Retrieve output
    output = await JobManager.get_output(db, job_id)

    if output is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Output not available for this job"
        )

    # Generate markdown
    markdown_content = _generate_markdown_report(output)

    # Generate filename from original document
    original_filename = output.get("document", {}).get("filename", "output")
    # Remove extension and add _summary.md
    base_name = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    download_filename = f"{base_name}_summary.md"

    return Response(
        content=markdown_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{download_filename}"'
        }
    )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a running job or delete a completed job.

    Removes job record from database and cleans up associated files.

    Args:
        job_id: Unique job identifier
        db: Database session (injected)

    Returns:
        204 No Content on success

    Raises:
        HTTPException 404: Job not found
    """
    # Check if job exists
    job_status = await JobManager.get(db, job_id)

    if job_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )

    # Delete job (includes file cleanup)
    await JobManager.delete(db, job_id)

    # Return 204 No Content (no response body)
    return None
