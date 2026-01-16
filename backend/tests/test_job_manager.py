"""Tests for JobManager service"""

import json
import pytest
import pytest_asyncio
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

from app.db.database import Base, Job
from app.models.config import PipelineConfig
from app.models.job import JobStatus
from app.services.job_manager import JobManager
from app.services.file_storage import FileStorage


# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_db(test_engine):
    """Create test database session"""
    # test_engine is already awaited by pytest-asyncio
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session
        await session.rollback()  # Rollback any uncommitted changes


@pytest.fixture
def test_config():
    """Create test pipeline configuration"""
    return PipelineConfig(
        extract_only=False,
        skip_tables=False,
        skip_images=False,
        text_only=False,
        pdf_image_dpi=200,
        vision_mode='none',
        chunk_size=3000,
        chunk_overlap=80,
        run_summarization=True,
        summarizer_mode='full'
    )


@pytest.fixture
def test_output_data():
    """Create test output data"""
    return {
        "document": {
            "document_id": "test-doc-id",
            "filename": "test.pdf",
            "total_pages": 5,
            "metadata": {}
        },
        "pages": [
            {
                "chunk_id": "page-1",
                "doc_title": "test.pdf",
                "text": "Sample text",
                "tables": [],
                "summary_notes": ["Note 1"],
                "summary_topics": ["Topic 1"],
                "summary_relevancy": 0.92
            }
        ]
    }


@pytest.mark.asyncio
async def test_create_job(test_db: AsyncSession, test_config: PipelineConfig):
    """Test JobManager.create() inserts job with pending status"""
    job_id = "test-job-123"
    file_name = "test_document.pdf"
    file_path = "/tmp/test/test_document.pdf"

    # Create job
    job = await JobManager.create(
        db=test_db,
        job_id=job_id,
        file_name=file_name,
        file_path=file_path,
        config=test_config
    )

    # Verify job was created with correct attributes
    assert job.job_id == job_id
    assert job.status == "pending"
    assert job.progress == 0
    assert job.file_name == file_name
    assert job.created_at is not None

    # Verify job persists in database
    result = await test_db.execute(select(Job).where(Job.id == job_id))
    db_job = result.scalar_one_or_none()

    assert db_job is not None
    assert db_job.id == job_id
    assert db_job.status == "pending"
    assert db_job.file_name == file_name


@pytest.mark.asyncio
async def test_get_job_exists(test_db: AsyncSession, test_config: PipelineConfig):
    """Test JobManager.get() returns job when it exists"""
    job_id = "test-job-456"
    file_name = "document.pdf"
    file_path = "/tmp/test/document.pdf"

    # Create job first
    await JobManager.create(
        db=test_db,
        job_id=job_id,
        file_name=file_name,
        file_path=file_path,
        config=test_config
    )

    # Retrieve job
    job_status = await JobManager.get(db=test_db, job_id=job_id)

    # Verify job was retrieved
    assert job_status is not None
    assert job_status.job_id == job_id
    assert job_status.file_name == file_name
    assert job_status.status == "pending"


@pytest.mark.asyncio
async def test_get_job_not_exists(test_db: AsyncSession):
    """Test JobManager.get() returns None when job doesn't exist"""
    job_status = await JobManager.get(db=test_db, job_id="nonexistent-job")

    assert job_status is None


@pytest.mark.asyncio
async def test_update_status(test_db: AsyncSession, test_config: PipelineConfig):
    """Test JobManager.update_status() changes status and progress"""
    job_id = "test-job-789"
    file_name = "document.pdf"
    file_path = "/tmp/test/document.pdf"

    # Create job
    await JobManager.create(
        db=test_db,
        job_id=job_id,
        file_name=file_name,
        file_path=file_path,
        config=test_config
    )

    # Update status
    await JobManager.update_status(
        db=test_db,
        job_id=job_id,
        status="processing",
        progress=50,
        stage="vision",
        message="Processing page 5 of 10"
    )

    # Retrieve updated job
    job_status = await JobManager.get(db=test_db, job_id=job_id)

    # Verify updates
    assert job_status is not None
    assert job_status.status == "processing"
    assert job_status.progress == 50
    assert job_status.current_stage == "vision"
    assert job_status.message == "Processing page 5 of 10"
    assert job_status.started_at is not None


@pytest.mark.asyncio
async def test_set_output(test_db: AsyncSession, test_config: PipelineConfig):
    """Test JobManager.set_output() stores output path"""
    job_id = "test-job-output"
    file_name = "document.pdf"
    file_path = "/tmp/test/document.pdf"
    output_path = "/tmp/output/result.json"

    # Create job
    await JobManager.create(
        db=test_db,
        job_id=job_id,
        file_name=file_name,
        file_path=file_path,
        config=test_config
    )

    # Set output path
    await JobManager.set_output(
        db=test_db,
        job_id=job_id,
        output_path=output_path
    )

    # Verify output path was stored
    result = await test_db.execute(select(Job).where(Job.id == job_id))
    db_job = result.scalar_one()

    assert db_job.output_path == output_path


@pytest.mark.asyncio
async def test_get_output(
    test_db: AsyncSession,
    test_config: PipelineConfig,
    test_output_data: dict,
    tmp_path: Path
):
    """Test JobManager.get_output() returns output JSON"""
    job_id = "test-job-get-output"
    file_name = "document.pdf"
    file_path = "/tmp/test/document.pdf"

    # Create output file
    output_path = tmp_path / "output.json"
    with open(output_path, "w") as f:
        json.dump(test_output_data, f)

    # Create job and set output
    await JobManager.create(
        db=test_db,
        job_id=job_id,
        file_name=file_name,
        file_path=file_path,
        config=test_config
    )
    await JobManager.set_output(
        db=test_db,
        job_id=job_id,
        output_path=str(output_path)
    )

    # Retrieve output
    output = await JobManager.get_output(db=test_db, job_id=job_id)

    # Verify output matches
    assert output is not None
    assert output["document"]["filename"] == "test.pdf"
    assert len(output["pages"]) == 1


@pytest.mark.asyncio
async def test_delete_job(test_db: AsyncSession, test_config: PipelineConfig, tmp_path: Path):
    """Test JobManager.delete() removes job and files"""
    job_id = "test-job-delete"
    file_name = "document.pdf"

    # Create job directory with a file
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    file_path = job_dir / file_name
    file_path.write_text("test content")

    # Create job
    await JobManager.create(
        db=test_db,
        job_id=job_id,
        file_name=file_name,
        file_path=str(file_path),
        config=test_config
    )

    # Verify job exists
    job_status = await JobManager.get(db=test_db, job_id=job_id)
    assert job_status is not None

    # Delete job (mock FileStorage.cleanup to use tmp_path)
    await JobManager.delete(db=test_db, job_id=job_id)

    # Verify job no longer exists in database
    job_status = await JobManager.get(db=test_db, job_id=job_id)
    assert job_status is None


@pytest.mark.asyncio
async def test_jobs_persist_across_sessions(test_engine, test_config: PipelineConfig):
    """Test jobs persist across service restarts (SQLite)"""
    job_id = "test-job-persist"
    file_name = "persistent.pdf"
    file_path = "/tmp/test/persistent.pdf"

    # Create first session and add job
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session1:
        await JobManager.create(
            db=session1,
            job_id=job_id,
            file_name=file_name,
            file_path=file_path,
            config=test_config
        )
        await session1.commit()

    # Create second session and verify job exists
    async with async_session() as session2:
        job_status = await JobManager.get(db=session2, job_id=job_id)
        assert job_status is not None
        assert job_status.job_id == job_id
        assert job_status.file_name == file_name
        assert job_status.status == "pending"
