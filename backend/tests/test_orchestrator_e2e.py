"""
End-to-end integration tests for PipelineOrchestrator.

These tests verify the full pipeline flow from extraction through
summarization using mocked LLM providers.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.models import ProgressEvent, PipelineStage
from app.pipeline.vision.schemas import ClassificationResult, ExtractionResult


@pytest.fixture
def text_file(tmp_path):
    """Create a test text file."""
    file_path = tmp_path / "test_document.txt"
    content = """
    Document Title: Test Document

    This is the first section of the document. It contains important
    information about testing the pipeline orchestrator.

    Section 2: More Content

    This section has additional content that will be processed
    by the summarization stage.
    """
    file_path.write_text(content)
    return file_path


@pytest.fixture
def pdf_file(tmp_path):
    """Create a minimal test PDF file."""
    file_path = tmp_path / "test_document.pdf"
    # Minimal PDF with text content
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Test PDF content) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000359 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
434
%%EOF"""
    file_path.write_bytes(pdf_content)
    return file_path


@pytest.fixture
def mock_vision_provider():
    """Mock vision provider that returns classification and extraction results."""
    provider = AsyncMock()
    provider.classify = AsyncMock(return_value=ClassificationResult(
        page_number=1,
        chunk_id="chunk_1",
        has_graphics=False,
        confidence=0.95,
    ))
    provider.extract = AsyncMock(return_value=ExtractionResult(
        page_number=1,
        chunk_id="chunk_1",
        extracted_text="Extracted visual content",
        skipped=False,
    ))
    return provider


@pytest.fixture
def mock_summarize_response():
    """Mock LLM response for summarization."""
    return {
        "summary_notes": ["This is a test summary note", "Another important point"],
        "summary_topics": ["testing", "pipeline", "integration"],
        "summary_relevancy": 92,
    }


class TestOrchestratorTextExtraction:
    """Test orchestrator with text file extraction only."""

    @pytest.mark.asyncio
    async def test_extraction_only_mode_text_file(self, text_file):
        """Extraction-only mode returns document structure without vision/summarization."""
        orchestrator = PipelineOrchestrator()

        config = {
            "extract_only": True,
            "text_only": True,
        }

        result, metrics = await orchestrator.run(
            job_id="test-job-1",
            input_file_path=text_file,
            config=config,
        )

        # Verify document structure
        assert "document" in result
        assert "pages" in result
        assert result["document"]["filename"] == "test_document.txt"
        assert result["document"]["total_pages"] >= 1

        # Verify pages have expected fields
        for page in result["pages"]:
            assert "chunk_id" in page
            assert "text" in page

    @pytest.mark.asyncio
    async def test_full_pipeline_text_file_skips_vision(self, text_file, mock_summarize_response):
        """Text file with text_only=True skips vision stage."""
        orchestrator = PipelineOrchestrator()

        config = {
            "text_only": True,
            "run_summarization": True,
        }

        progress_events = []

        async def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        # Mock the summarizer to return predictable results
        with patch("app.pipeline.orchestrator.SummarizeService") as mock_summarizer_class:
            mock_summarizer = AsyncMock()
            mock_summarizer.summarize_pages_batch = AsyncMock(return_value=[
                MagicMock(
                    summary_notes=mock_summarize_response["summary_notes"],
                    summary_topics=mock_summarize_response["summary_topics"],
                    summary_relevancy=mock_summarize_response["summary_relevancy"],
                )
            ])
            mock_summarizer_class.return_value = mock_summarizer

            result, metrics = await orchestrator.run(
                job_id="test-job-2",
                input_file_path=text_file,
                config=config,
                progress_callback=capture_progress,
            )

        # Verify extraction stage was called
        extraction_events = [e for e in progress_events if e.stage == PipelineStage.EXTRACTION]
        assert len(extraction_events) > 0

        # Verify vision stage was NOT called (text_only mode)
        vision_events = [e for e in progress_events if e.stage == PipelineStage.VISION]
        assert len(vision_events) == 0

        # Verify summarization stage was called
        summarization_events = [e for e in progress_events if e.stage == PipelineStage.SUMMARIZATION]
        assert len(summarization_events) > 0


class TestOrchestratorSkipModes:
    """Test orchestrator with various skip configurations."""

    @pytest.mark.asyncio
    async def test_skip_summarization_mode(self, text_file):
        """run_summarization=False skips summarization stage."""
        orchestrator = PipelineOrchestrator()

        config = {
            "text_only": True,
            "run_summarization": False,
        }

        progress_events = []

        async def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        result, metrics = await orchestrator.run(
            job_id="test-job-3",
            input_file_path=text_file,
            config=config,
            progress_callback=capture_progress,
        )

        # Verify summarization stage was NOT called
        summarization_events = [e for e in progress_events if e.stage == PipelineStage.SUMMARIZATION]
        assert len(summarization_events) == 0

        # Pages should not have summary fields
        for page in result["pages"]:
            assert "summary_notes" not in page or page.get("summary_notes") is None

    @pytest.mark.asyncio
    async def test_skip_images_mode(self, text_file):
        """skip_images=True skips vision stage."""
        orchestrator = PipelineOrchestrator()

        config = {
            "skip_images": True,
            "run_summarization": False,
        }

        progress_events = []

        async def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        result, metrics = await orchestrator.run(
            job_id="test-job-4",
            input_file_path=text_file,
            config=config,
            progress_callback=capture_progress,
        )

        # Verify vision stage was NOT called
        vision_events = [e for e in progress_events if e.stage == PipelineStage.VISION]
        assert len(vision_events) == 0


class TestOrchestratorOutputSchema:
    """Test that orchestrator output matches expected schema."""

    @pytest.mark.asyncio
    async def test_output_matches_current_schema_exactly(self, text_file):
        """Output structure matches the expected document schema."""
        orchestrator = PipelineOrchestrator()

        config = {
            "extract_only": True,
            "text_only": True,
        }

        result, metrics = await orchestrator.run(
            job_id="test-job-5",
            input_file_path=text_file,
            config=config,
        )

        # Verify top-level structure
        assert set(result.keys()) == {"document", "pages"}

        # Verify document fields
        document = result["document"]
        assert "document_id" in document
        assert "filename" in document
        assert "total_pages" in document
        assert "metadata" in document

        # Verify document_id is a string
        assert isinstance(document["document_id"], str)
        assert len(document["document_id"]) > 0

        # Verify pages is a list
        assert isinstance(result["pages"], list)

        # Verify each page has required fields
        for page in result["pages"]:
            assert "chunk_id" in page
            assert "text" in page

    @pytest.mark.asyncio
    async def test_document_id_is_consistent_for_same_file(self, text_file):
        """Same file produces consistent document_id."""
        orchestrator = PipelineOrchestrator()

        config = {
            "extract_only": True,
            "text_only": True,
        }

        result1, _ = await orchestrator.run(
            job_id="test-job-6a",
            input_file_path=text_file,
            config=config,
        )

        result2, _ = await orchestrator.run(
            job_id="test-job-6b",
            input_file_path=text_file,
            config=config,
        )

        # document_ids should be the same for same input file
        assert result1["document"]["document_id"] == result2["document"]["document_id"]

    @pytest.mark.asyncio
    async def test_document_id_differs_for_different_files(self, text_file, tmp_path):
        """Different files produce different document_ids."""
        orchestrator = PipelineOrchestrator()

        # Create a second file
        second_file = tmp_path / "another_document.txt"
        second_file.write_text("Different content")

        config = {
            "extract_only": True,
            "text_only": True,
        }

        result1, _ = await orchestrator.run(
            job_id="test-job-6c",
            input_file_path=text_file,
            config=config,
        )

        result2, _ = await orchestrator.run(
            job_id="test-job-6d",
            input_file_path=second_file,
            config=config,
        )

        # document_ids should be different for different files
        assert result1["document"]["document_id"] != result2["document"]["document_id"]


class TestOrchestratorProgressCallbacks:
    """Test progress callback behavior."""

    @pytest.mark.asyncio
    async def test_progress_callback_receives_all_stages(self, text_file, mock_summarize_response):
        """Progress callback receives events for all enabled stages."""
        orchestrator = PipelineOrchestrator()

        config = {
            "text_only": True,
            "run_summarization": True,
        }

        progress_events = []

        async def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        with patch("app.pipeline.orchestrator.SummarizeService") as mock_summarizer_class:
            mock_summarizer = AsyncMock()
            mock_summarizer.summarize_pages_batch = AsyncMock(return_value=[
                MagicMock(
                    summary_notes=mock_summarize_response["summary_notes"],
                    summary_topics=mock_summarize_response["summary_topics"],
                    summary_relevancy=mock_summarize_response["summary_relevancy"],
                )
            ])
            mock_summarizer_class.return_value = mock_summarizer

            await orchestrator.run(
                job_id="test-job-7",
                input_file_path=text_file,
                config=config,
                progress_callback=capture_progress,
            )

        # Should have at least one event per enabled stage
        stages_seen = set(e.stage for e in progress_events if e.stage is not None)
        assert PipelineStage.EXTRACTION in stages_seen
        assert PipelineStage.SUMMARIZATION in stages_seen

        # Should have started events
        started_events = [e for e in progress_events if e.event_type == "started"]
        assert len(started_events) >= 2

    @pytest.mark.asyncio
    async def test_progress_callback_optional(self, text_file):
        """Orchestrator works without progress callback."""
        orchestrator = PipelineOrchestrator()

        config = {
            "extract_only": True,
            "text_only": True,
        }

        # Should not raise when progress_callback is None
        result, metrics = await orchestrator.run(
            job_id="test-job-8",
            input_file_path=text_file,
            config=config,
            progress_callback=None,
        )

        assert "document" in result
        assert "pages" in result


class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator."""

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_error(self, tmp_path):
        """Orchestrator raises error for non-existent file."""
        orchestrator = PipelineOrchestrator()

        nonexistent_path = tmp_path / "does_not_exist.txt"

        config = {
            "extract_only": True,
            "text_only": True,
        }

        with pytest.raises(Exception):  # Could be FileNotFoundError or similar
            await orchestrator.run(
                job_id="test-job-9",
                input_file_path=nonexistent_path,
                config=config,
            )

    @pytest.mark.asyncio
    async def test_vision_failure_continues_processing(self, text_file, mock_summarize_response):
        """Vision stage failure does not stop the pipeline."""
        orchestrator = PipelineOrchestrator()

        config = {
            "text_only": False,  # Enable vision
            "skip_images": False,
            "run_summarization": True,
        }

        progress_events = []

        async def capture_progress(event: ProgressEvent):
            progress_events.append(event)

        # Mock vision processor to fail
        with patch("app.pipeline.orchestrator.VisionProcessor") as mock_vision_class, \
             patch("app.pipeline.orchestrator.SummarizeService") as mock_summarizer_class:

            mock_vision = MagicMock()
            mock_vision.process_pages = AsyncMock(side_effect=Exception("Vision failed"))
            mock_vision_class.return_value = mock_vision

            mock_summarizer = AsyncMock()
            mock_summarizer.summarize_pages_batch = AsyncMock(return_value=[
                MagicMock(
                    summary_notes=mock_summarize_response["summary_notes"],
                    summary_topics=mock_summarize_response["summary_topics"],
                    summary_relevancy=mock_summarize_response["summary_relevancy"],
                )
            ])
            mock_summarizer_class.return_value = mock_summarizer

            result, metrics = await orchestrator.run(
                job_id="test-job-10",
                input_file_path=text_file,
                config=config,
                progress_callback=capture_progress,
            )

        # Pipeline should still complete
        assert "document" in result
        assert "pages" in result

        # Summarization should still have been called
        summarization_events = [e for e in progress_events if e.stage == PipelineStage.SUMMARIZATION]
        assert len(summarization_events) > 0


class TestOrchestratorWithMockedSummarization:
    """Test full pipeline with mocked summarization."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_summarization(self, text_file, mock_summarize_response):
        """Full pipeline produces complete output with summaries."""
        orchestrator = PipelineOrchestrator()

        config = {
            "text_only": True,
            "run_summarization": True,
            "summarizer_mode": "full",
        }

        with patch("app.pipeline.orchestrator.SummarizeService") as mock_summarizer_class:
            mock_summarizer = AsyncMock()
            mock_result = MagicMock()
            mock_result.summary_notes = mock_summarize_response["summary_notes"]
            mock_result.summary_topics = mock_summarize_response["summary_topics"]
            mock_result.summary_relevancy = mock_summarize_response["summary_relevancy"]
            mock_summarizer.summarize_pages_batch = AsyncMock(return_value=[mock_result])
            mock_summarizer_class.return_value = mock_summarizer

            result, metrics = await orchestrator.run(
                job_id="test-job-11",
                input_file_path=text_file,
                config=config,
            )

        # Verify pages have summary fields
        assert len(result["pages"]) >= 1
        page = result["pages"][0]
        assert "summary_notes" in page
        assert "summary_topics" in page
        assert "summary_relevancy" in page
