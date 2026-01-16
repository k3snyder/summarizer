"""End-to-end integration tests for the backend API flow"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.database import Base, Job, get_db
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

    # Store session factory for direct DB access in tests
    app.state.test_session_factory = TestSessionLocal
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


class TestCORS:
    """Tests for CORS configuration"""

    @pytest.mark.asyncio
    async def test_cors_preflight_request(self, client):
        """OPTIONS request returns proper CORS headers"""
        response = await client.options(
            "/api/jobs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            }
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "access-control-allow-methods" in response.headers

    @pytest.mark.asyncio
    async def test_cors_actual_request_includes_headers(self, client):
        """Actual request includes CORS headers in response"""
        response = await client.get(
            "/api/health",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_cors_with_credentials(self, client):
        """CORS allows credentials from allowed origins"""
        response = await client.options(
            "/api/jobs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )

        assert "access-control-allow-credentials" in response.headers
        assert response.headers["access-control-allow-credentials"] == "true"


class TestFullJobFlow:
    """Tests for complete job lifecycle from upload to output retrieval"""

    @pytest.mark.asyncio
    async def test_full_job_lifecycle_with_polling(self, client):
        """Test complete flow: upload PDF, poll status, simulate completion, get output"""
        # Step 1: Upload PDF and create job
        pdf_content = b"%PDF-1.4 mock pdf content for testing"
        files = {"file": ("integration_test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        assert create_response.status_code == 201
        job_data = create_response.json()
        job_id = job_data["job_id"]
        assert job_data["status"] == "pending"
        assert "message" in job_data

        # Step 2: Poll for status (initial pending)
        status_response = await client.get(f"/api/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert status_data["file_name"] == "integration_test.pdf"
        assert "created_at" in status_data

        # Step 3: Simulate job completing by directly updating database
        async with app.state.test_session_factory() as db:
            from sqlalchemy import select
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one()

            # Create a mock output file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False
            ) as f:
                mock_output = {
                    "document": {
                        "document_id": "test-doc",
                        "filename": "integration_test.pdf",
                        "total_pages": 3,
                        "metadata": {}
                    },
                    "pages": [
                        {
                            "chunk_id": "page-1",
                            "doc_title": "Test Document",
                            "text": "Page 1 content",
                            "summary_notes": ["First note"],
                            "summary_topics": ["testing"]
                        }
                    ]
                }
                json.dump(mock_output, f)
                output_path = f.name

            # Update job to completed
            job.status = "completed"
            job.progress = 100
            job.current_stage = "summarization"
            job.message = "Pipeline complete!"
            job.output_path = output_path
            await db.commit()

        # Step 4: Poll again to see completed status
        status_response = await client.get(f"/api/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] == "completed"
        assert status_data["progress"] == 100

        # Step 5: Get output
        output_response = await client.get(f"/api/jobs/{job_id}/output")
        assert output_response.status_code == 200
        output_data = output_response.json()
        assert "document" in output_data
        assert "pages" in output_data
        assert output_data["document"]["filename"] == "integration_test.pdf"
        assert len(output_data["pages"]) == 1

        # Step 6: Clean up - delete job
        delete_response = await client.delete(f"/api/jobs/{job_id}")
        assert delete_response.status_code == 204

        # Step 7: Verify job is gone
        get_response = await client.get(f"/api/jobs/{job_id}")
        assert get_response.status_code == 404

        # Clean up temp file
        Path(output_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_job_progress_updates_during_processing(self, client):
        """Test that job progress can be tracked during processing"""
        # Create job
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("progress_test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Simulate progress updates
        progress_stages = [
            ("processing", 10, "extraction", "Starting extraction..."),
            ("processing", 30, "extraction", "Processing page 1 of 3"),
            ("processing", 50, "vision", "Running vision classifier"),
            ("processing", 75, "summarization", "Generating summaries"),
            ("processing", 90, "summarization", "Finalizing output"),
        ]

        async with app.state.test_session_factory() as db:
            from sqlalchemy import select
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one()

            for status, progress, stage, message in progress_stages:
                job.status = status
                job.progress = progress
                job.current_stage = stage
                job.message = message
                await db.commit()

                # Poll and verify
                response = await client.get(f"/api/jobs/{job_id}")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == status
                assert data["progress"] == progress
                assert data["current_stage"] == stage
                assert data["message"] == message

        # Clean up
        await client.delete(f"/api/jobs/{job_id}")


class TestJobCleanup:
    """Tests for job deletion and cleanup functionality"""

    @pytest.mark.asyncio
    async def test_delete_pending_job(self, client):
        """Can delete a job that is still pending"""
        # Create job
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("delete_test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Delete immediately
        delete_response = await client.delete(f"/api/jobs/{job_id}")
        assert delete_response.status_code == 204

        # Verify gone
        get_response = await client.get(f"/api/jobs/{job_id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_completed_job(self, client):
        """Can delete a completed job and its output"""
        # Create job
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("cleanup_test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Mark as completed with output
        async with app.state.test_session_factory() as db:
            from sqlalchemy import select
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({"test": "output"}, f)
                output_path = f.name

            job.status = "completed"
            job.progress = 100
            job.output_path = output_path
            await db.commit()

        # Delete job
        delete_response = await client.delete(f"/api/jobs/{job_id}")
        assert delete_response.status_code == 204

        # Verify job gone
        get_response = await client.get(f"/api/jobs/{job_id}")
        assert get_response.status_code == 404

        # Clean up temp file
        Path(output_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_delete_failed_job(self, client):
        """Can delete a failed job"""
        # Create job
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("failed_test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Mark as failed
        async with app.state.test_session_factory() as db:
            from sqlalchemy import select
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one()
            job.status = "failed"
            job.error = "Pipeline execution failed"
            await db.commit()

        # Delete job
        delete_response = await client.delete(f"/api/jobs/{job_id}")
        assert delete_response.status_code == 204


class TestErrorHandling:
    """Tests for error scenarios"""

    @pytest.mark.asyncio
    async def test_get_output_on_pending_job_returns_400(self, client):
        """Requesting output before completion returns 400"""
        # Create job
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("pending_output.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Try to get output
        output_response = await client.get(f"/api/jobs/{job_id}/output")
        assert output_response.status_code == 400
        assert "not completed" in output_response.json()["detail"]

        # Clean up
        await client.delete(f"/api/jobs/{job_id}")

    @pytest.mark.asyncio
    async def test_get_output_on_failed_job_returns_400(self, client):
        """Requesting output on failed job returns 400"""
        # Create job
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("failed_output.pdf", io.BytesIO(pdf_content), "application/pdf")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            create_response = await client.post("/api/jobs", files=files, data=data)

        job_id = create_response.json()["job_id"]

        # Mark as failed
        async with app.state.test_session_factory() as db:
            from sqlalchemy import select
            result = await db.execute(select(Job).where(Job.id == job_id))
            job = result.scalar_one()
            job.status = "failed"
            job.error = "Test failure"
            await db.commit()

        # Try to get output
        output_response = await client.get(f"/api/jobs/{job_id}/output")
        assert output_response.status_code == 400

        # Clean up
        await client.delete(f"/api/jobs/{job_id}")

    @pytest.mark.asyncio
    async def test_invalid_job_id_returns_404(self, client):
        """Non-existent job ID returns 404"""
        response = await client.get("/api/jobs/invalid-uuid-that-does-not-exist")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_invalid_config_json_returns_400(self, client):
        """Malformed config JSON returns 400"""
        pdf_content = b"%PDF-1.4 mock pdf content"
        files = {"file": ("bad_config.pdf", io.BytesIO(pdf_content), "application/pdf")}
        data = {"config": "{invalid json"}

        response = await client.post("/api/jobs", files=files, data=data)
        assert response.status_code == 400
        assert "Invalid config format" in response.json()["detail"]


class TestMultipleFileTypes:
    """Tests for different file type handling"""

    @pytest.mark.asyncio
    async def test_txt_file_upload(self, client):
        """TXT file upload creates job successfully"""
        txt_content = b"This is a plain text document.\n\nWith multiple paragraphs."
        files = {"file": ("document.txt", io.BytesIO(txt_content), "text/plain")}
        config = PipelineConfig(text_only=True)
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            response = await client.post("/api/jobs", files=files, data=data)

        assert response.status_code == 201
        job_data = response.json()
        assert "job_id" in job_data

        # Verify job exists
        status_response = await client.get(f"/api/jobs/{job_data['job_id']}")
        assert status_response.status_code == 200
        assert status_response.json()["file_name"] == "document.txt"

        # Clean up
        await client.delete(f"/api/jobs/{job_data['job_id']}")

    @pytest.mark.asyncio
    async def test_md_file_upload(self, client):
        """Markdown file upload creates job successfully"""
        md_content = b"# Markdown Document\n\n## Section 1\n\nSome content here."
        files = {"file": ("readme.md", io.BytesIO(md_content), "text/markdown")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        with patch("app.routers.jobs.run_pipeline_background"):
            response = await client.post("/api/jobs", files=files, data=data)

        assert response.status_code == 201
        job_data = response.json()

        status_response = await client.get(f"/api/jobs/{job_data['job_id']}")
        assert status_response.json()["file_name"] == "readme.md"

        # Clean up
        await client.delete(f"/api/jobs/{job_data['job_id']}")

    @pytest.mark.asyncio
    async def test_unsupported_file_type_returns_400(self, client):
        """Unsupported file type returns 400"""
        exe_content = b"MZ\x90\x00fake executable"
        files = {"file": ("virus.exe", io.BytesIO(exe_content), "application/x-msdownload")}
        config = PipelineConfig()
        data = {"config": config.model_dump_json()}

        response = await client.post("/api/jobs", files=files, data=data)
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
