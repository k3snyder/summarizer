"""Tests for /api/jobs endpoints"""

import io
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.database import Base, get_db
from app.main import app
from app.models.config import PipelineConfig


# Create test database engine
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_db():
    """Create a test database for each test"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_db():
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    yield engine

    app.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def client(test_db):
    """Create test client with test database"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestCreateJob:
    """Tests for POST /api/jobs endpoint"""

    @pytest.mark.asyncio
    async def test_create_job_with_pdf_returns_201(self, client):
        """POST /api/jobs with PDF returns 201 with job_id"""
        # Create a mock PDF file
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        # Mock the pipeline runner to avoid actual execution
        with patch("app.routers.jobs.run_pipeline_background"):
            response = await client.post("/api/jobs", files=files, data=data)

        assert response.status_code == 201
        body = response.json()
        assert "job_id" in body
        assert body["status"] == "pending"
        assert "message" in body

    @pytest.mark.asyncio
    async def test_create_job_with_txt_returns_201(self, client):
        """POST /api/jobs with TXT file returns 201"""
        txt_content = b"Sample text document content"
        files = {"file": ("document.txt", io.BytesIO(txt_content), "text/plain")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            response = await client.post("/api/jobs", files=files, data=data)

        assert response.status_code == 201
        body = response.json()
        assert "job_id" in body

    @pytest.mark.asyncio
    async def test_create_job_with_md_returns_201(self, client):
        """POST /api/jobs with MD file returns 201"""
        md_content = b"# Markdown Document\n\nSome content here."
        files = {"file": ("readme.md", io.BytesIO(md_content), "text/markdown")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            response = await client.post("/api/jobs", files=files, data=data)

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_create_job_invalid_file_type_returns_400(self, client):
        """POST /api/jobs with invalid file type returns 400"""
        jpg_content = b"fake image content"
        files = {"file": ("image.jpg", io.BytesIO(jpg_content), "image/jpeg")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        response = await client.post("/api/jobs", files=files, data=data)

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_job_invalid_config_returns_400(self, client):
        """POST /api/jobs with invalid config returns 400"""
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        data = {"config": "not valid json"}

        response = await client.post("/api/jobs", files=files, data=data)

        assert response.status_code == 400
        assert "Invalid config format" in response.json()["detail"]


class TestGetJobStatus:
    """Tests for GET /api/jobs/{job_id} endpoint"""

    @pytest.mark.asyncio
    async def test_get_job_status_returns_job(self, client):
        """GET /api/jobs/{id} returns JobStatus"""
        # First create a job
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Now get the status
        response = await client.get(f"/api/jobs/{job_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["job_id"] == job_id
        assert body["status"] == "pending"
        assert "progress" in body
        assert "file_name" in body
        assert body["file_name"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_get_job_status_not_found_returns_404(self, client):
        """GET /api/jobs/{id} returns 404 for non-existent job"""
        response = await client.get("/api/jobs/nonexistent-job-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestGetJobOutput:
    """Tests for GET /api/jobs/{job_id}/output endpoint"""

    @pytest.mark.asyncio
    async def test_get_output_not_completed_returns_400(self, client):
        """GET /api/jobs/{id}/output returns 400 for non-completed job"""
        # Create a job (will be in pending status)
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Try to get output before completion
        response = await client.get(f"/api/jobs/{job_id}/output")

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_output_not_found_returns_404(self, client):
        """GET /api/jobs/{id}/output returns 404 for non-existent job"""
        response = await client.get("/api/jobs/nonexistent-job-id/output")

        assert response.status_code == 404


class TestDeleteJob:
    """Tests for DELETE /api/jobs/{job_id} endpoint"""

    @pytest.mark.asyncio
    async def test_delete_job_returns_204(self, client):
        """DELETE /api/jobs/{id} returns 204"""
        # Create a job first
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Delete the job
        response = await client.delete(f"/api/jobs/{job_id}")

        assert response.status_code == 204

        # Verify job is actually deleted
        get_response = await client.get(f"/api/jobs/{job_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_job_not_found_returns_404(self, client):
        """DELETE /api/jobs/{id} returns 404 for non-existent job"""
        response = await client.delete("/api/jobs/nonexistent-job-id")

        assert response.status_code == 404
