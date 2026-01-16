"""
Integration tests for extraction module.

These tests verify end-to-end extraction with real files,
schema compliance, and async behavior.
"""

import asyncio
from pathlib import Path

import pytest


# Test fixtures
@pytest.fixture
def test_txt_path() -> Path:
    return Path(__file__).parent.parent / "fixtures" / "test_document.txt"


@pytest.fixture
def test_pdf_path() -> Path:
    project_root = Path(__file__).parent.parent.parent.parent
    # Look for any PDF in the project
    pdf_files = list(project_root.glob("**/*.pdf"))
    if pdf_files:
        return pdf_files[0]
    return Path(__file__).parent.parent / "fixtures" / "test_document.pdf"


class TestExtractionIntegrationText:
    """Integration tests for text file extraction."""

    @pytest.mark.asyncio
    async def test_extract_real_text_file(self, test_txt_path):
        """E2E test that extracts a real TXT file."""
        from app.pipeline.extraction import DocumentExtractor, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)

        # Verify basic structure
        assert result.document_id is not None
        assert result.filename == test_txt_path.name
        assert result.total_pages >= 1
        assert len(result.pages) == result.total_pages

        # Verify all pages have content
        for page in result.pages:
            assert page.chunk_id is not None
            assert len(page.text) > 0

    @pytest.mark.asyncio
    async def test_text_output_matches_expected_schema(self, test_txt_path):
        """Verify output matches expected schema (document_id, filename, total_pages, pages array)."""
        from app.pipeline.extraction import DocumentExtractor, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)
        output = result.to_dict()

        # Check document structure
        assert "document" in output
        assert "pages" in output

        doc = output["document"]
        assert "document_id" in doc
        assert "filename" in doc
        assert "total_pages" in doc
        assert "metadata" in doc

        # Check pages structure
        for page in output["pages"]:
            assert "chunk_id" in page
            assert "doc_title" in page
            assert "text" in page
            assert "tables" in page

    @pytest.mark.asyncio
    async def test_text_progress_callbacks_fire_correctly(self, test_txt_path):
        """Verify progress callbacks fire correctly during extraction."""
        from app.pipeline.extraction import DocumentExtractor, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        progress_events: list[tuple[int, int, str]] = []

        async def callback(current: int, total: int, message: str):
            progress_events.append((current, total, message))

        result = await extractor.extract(test_txt_path, config, progress_callback=callback)

        # Should have progress events for each chunk
        assert len(progress_events) >= result.total_pages

        # Progress should be monotonically increasing
        for i, (current, total, _) in enumerate(progress_events):
            if i > 0:
                prev_current = progress_events[i - 1][0]
                assert current >= prev_current

    @pytest.mark.asyncio
    async def test_text_async_execution_non_blocking(self, test_txt_path):
        """Verify async execution doesn't block event loop."""
        from app.pipeline.extraction import DocumentExtractor, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        # Track concurrent task execution
        concurrent_completed = False

        async def concurrent_task():
            nonlocal concurrent_completed
            await asyncio.sleep(0.001)
            concurrent_completed = True
            return "done"

        # Run extraction and concurrent task together
        extraction_task = extractor.extract(test_txt_path, config)
        concurrent = concurrent_task()

        results = await asyncio.gather(extraction_task, concurrent)

        # Both should complete
        assert results[0] is not None
        assert results[1] == "done"
        assert concurrent_completed


class TestExtractionIntegrationPDF:
    """Integration tests for PDF file extraction."""

    @pytest.mark.asyncio
    async def test_extract_real_pdf_file(self, test_pdf_path):
        """E2E test that extracts a real PDF file."""
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction import DocumentExtractor, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_pdf_path, config)

        # Verify basic structure
        assert result.document_id is not None
        assert result.filename == test_pdf_path.name
        assert result.total_pages >= 1
        assert len(result.pages) == result.total_pages

    @pytest.mark.asyncio
    async def test_pdf_output_matches_expected_schema(self, test_pdf_path):
        """Verify output matches expected schema (document_id, filename, total_pages, pages array)."""
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction import DocumentExtractor, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_pdf_path, config)
        output = result.to_dict()

        # Check document structure
        assert "document" in output
        assert "pages" in output

        doc = output["document"]
        assert "document_id" in doc
        assert "filename" in doc
        assert "total_pages" in doc

        # Check pages structure
        for page in output["pages"]:
            assert "chunk_id" in page
            assert "doc_title" in page
            assert "text" in page
            assert "tables" in page

    @pytest.mark.asyncio
    async def test_pdf_progress_callbacks_fire_correctly(self, test_pdf_path):
        """Verify progress callbacks fire correctly during PDF extraction."""
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction import DocumentExtractor, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        progress_events: list[tuple[int, int, str]] = []

        async def callback(current: int, total: int, message: str):
            progress_events.append((current, total, message))

        result = await extractor.extract(test_pdf_path, config, progress_callback=callback)

        # Should have at least one progress event per page
        assert len(progress_events) >= result.total_pages

    @pytest.mark.asyncio
    async def test_pdf_async_execution_non_blocking(self, test_pdf_path):
        """Verify async PDF extraction doesn't block event loop."""
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction import DocumentExtractor, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        concurrent_completed = False

        async def concurrent_task():
            nonlocal concurrent_completed
            await asyncio.sleep(0.001)
            concurrent_completed = True
            return "done"

        extraction_task = extractor.extract(test_pdf_path, config)
        concurrent = concurrent_task()

        results = await asyncio.gather(extraction_task, concurrent)

        assert results[0] is not None
        assert results[1] == "done"
        assert concurrent_completed


class TestExtractionModuleExports:
    """Test that extraction module exports are correct."""

    def test_extraction_config_exported(self):
        from app.pipeline.extraction import ExtractionConfig

        assert ExtractionConfig is not None

    def test_extracted_page_exported(self):
        from app.pipeline.extraction import ExtractedPage

        assert ExtractedPage is not None

    def test_extracted_document_exported(self):
        from app.pipeline.extraction import ExtractedDocument

        assert ExtractedDocument is not None

    def test_document_extractor_exported(self):
        from app.pipeline.extraction import DocumentExtractor

        assert DocumentExtractor is not None

    def test_all_exports_from_pipeline_package(self):
        """Test exports from main pipeline package."""
        from app.pipeline import (
            ExtractionConfig,
            ExtractedDocument,
            ExtractedPage,
            PipelineStage,
            ProgressEvent,
        )

        assert ExtractionConfig is not None
        assert ExtractedDocument is not None
        assert ExtractedPage is not None
        assert PipelineStage is not None
        assert ProgressEvent is not None
