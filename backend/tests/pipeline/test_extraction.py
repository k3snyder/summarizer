"""
TDD Red Phase: Tests for DocumentExtractor.

These tests define the expected behavior for the DocumentExtractor service
which handles PDF and text file extraction with async support.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Test fixtures
@pytest.fixture
def test_txt_path() -> Path:
    return Path(__file__).parent.parent / "fixtures" / "test_document.txt"


@pytest.fixture
def test_pdf_path() -> Path:
    # Use a real PDF from the project if available
    project_root = Path(__file__).parent.parent.parent.parent
    # Look for any PDF in the project
    pdf_files = list(project_root.glob("**/*.pdf"))
    if pdf_files:
        return pdf_files[0]
    # Return a placeholder path (tests will skip if no PDF)
    return Path(__file__).parent.parent / "fixtures" / "test_document.pdf"


class TestDocumentExtractorImports:
    """Tests for DocumentExtractor module imports."""

    def test_document_extractor_importable(self):
        from app.pipeline.extraction.extractor import DocumentExtractor

        assert DocumentExtractor is not None

    def test_document_extractor_is_class(self):
        from app.pipeline.extraction.extractor import DocumentExtractor

        assert isinstance(DocumentExtractor, type)


class TestDocumentExtractorTextExtraction:
    """Tests for text file extraction."""

    @pytest.mark.asyncio
    async def test_extract_text_returns_extracted_document(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractedDocument, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)

        assert isinstance(result, ExtractedDocument)
        assert result.filename == test_txt_path.name
        assert result.total_pages >= 1
        assert len(result.pages) >= 1

    @pytest.mark.asyncio
    async def test_text_extraction_creates_document_id(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)

        assert result.document_id is not None
        assert len(result.document_id) > 0
        # Document ID should include filename stem
        assert "test_document" in result.document_id

    @pytest.mark.asyncio
    async def test_text_extraction_pages_have_chunk_ids(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)

        for page in result.pages:
            assert page.chunk_id is not None
            assert len(page.chunk_id) > 0

    @pytest.mark.asyncio
    async def test_text_extraction_pages_have_text(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)

        for page in result.pages:
            assert page.text is not None
            assert len(page.text) > 0

    @pytest.mark.asyncio
    async def test_text_extraction_no_tables_or_images(self, test_txt_path):
        """Text files should have empty tables and no images."""
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)

        for page in result.pages:
            assert page.tables == []
            assert page.image_base64 is None


class TestDocumentExtractorPDFExtraction:
    """Tests for PDF file extraction."""

    @pytest.mark.asyncio
    async def test_extract_pdf_returns_extracted_document(self, test_pdf_path):
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractedDocument, ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_pdf_path, config)

        assert isinstance(result, ExtractedDocument)
        assert result.filename == test_pdf_path.name
        assert result.total_pages >= 1
        assert len(result.pages) >= 1

    @pytest.mark.asyncio
    async def test_pdf_extraction_pages_match_total(self, test_pdf_path):
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_pdf_path, config)

        assert len(result.pages) == result.total_pages


class TestDocumentExtractorProgressCallback:
    """Tests for progress callback functionality."""

    @pytest.mark.asyncio
    async def test_progress_callback_called_for_each_page(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        progress_calls: list[tuple[int, int, str]] = []

        async def progress_callback(current: int, total: int, message: str):
            progress_calls.append((current, total, message))

        result = await extractor.extract(
            test_txt_path, config, progress_callback=progress_callback
        )

        # Should have at least one progress call per page
        assert len(progress_calls) >= result.total_pages

    @pytest.mark.asyncio
    async def test_progress_callback_receives_valid_counts(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        progress_calls: list[tuple[int, int, str]] = []

        async def progress_callback(current: int, total: int, message: str):
            progress_calls.append((current, total, message))

        await extractor.extract(test_txt_path, config, progress_callback=progress_callback)

        for current, total, message in progress_calls:
            assert current >= 0
            assert total > 0
            assert current <= total

    @pytest.mark.asyncio
    async def test_extraction_works_without_callback(self, test_txt_path):
        """Extraction should work even if no callback is provided."""
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        # Should not raise
        result = await extractor.extract(test_txt_path, config)

        assert result is not None


class TestDocumentExtractorConfigOptions:
    """Tests for extraction configuration options."""

    @pytest.mark.asyncio
    async def test_skip_tables_config_excludes_tables(self, test_pdf_path):
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig(skip_tables=True)

        result = await extractor.extract(test_pdf_path, config)

        # With skip_tables=True, all pages should have empty tables
        for page in result.pages:
            assert page.tables == []

    @pytest.mark.asyncio
    async def test_skip_images_config_excludes_images(self, test_pdf_path):
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig(skip_images=True)

        result = await extractor.extract(test_pdf_path, config)

        # With skip_images=True, all pages should have no image
        for page in result.pages:
            assert page.image_base64 is None

    @pytest.mark.asyncio
    async def test_text_only_mode_skips_tables_and_images(self, test_pdf_path):
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig(text_only=True)

        result = await extractor.extract(test_pdf_path, config)

        for page in result.pages:
            assert page.tables == []
            assert page.image_base64 is None

    @pytest.mark.asyncio
    async def test_chunk_size_affects_text_splitting(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()

        # Small chunk size should create more pages
        small_chunk_config = ExtractionConfig(chunk_size=100, chunk_overlap=20)
        small_result = await extractor.extract(test_txt_path, small_chunk_config)

        # Large chunk size should create fewer pages
        large_chunk_config = ExtractionConfig(chunk_size=5000, chunk_overlap=500)
        large_result = await extractor.extract(test_txt_path, large_chunk_config)

        # Small chunks should result in more pages (or equal if content is small)
        assert small_result.total_pages >= large_result.total_pages

    @pytest.mark.asyncio
    async def test_pdf_image_dpi_config_used(self, test_pdf_path):
        """Test that pdf_image_dpi config is respected (affects image quality)."""
        if not test_pdf_path.exists():
            pytest.skip("No test PDF available")

        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()

        # Low DPI
        low_dpi_config = ExtractionConfig(pdf_image_dpi=72)
        low_result = await extractor.extract(test_pdf_path, low_dpi_config)

        # High DPI
        high_dpi_config = ExtractionConfig(pdf_image_dpi=300)
        high_result = await extractor.extract(test_pdf_path, high_dpi_config)

        # Both should succeed; higher DPI might produce larger base64 strings
        assert low_result is not None
        assert high_result is not None


class TestDocumentExtractorOutputSchema:
    """Tests for output schema compliance."""

    @pytest.mark.asyncio
    async def test_to_dict_produces_correct_schema(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)
        output = result.to_dict()

        # Check top-level structure
        assert "document" in output
        assert "pages" in output

        # Check document fields
        doc = output["document"]
        assert "document_id" in doc
        assert "filename" in doc
        assert "total_pages" in doc

        # Check pages structure
        assert isinstance(output["pages"], list)
        for page in output["pages"]:
            assert "chunk_id" in page
            assert "doc_title" in page
            assert "text" in page
            assert "tables" in page

    @pytest.mark.asyncio
    async def test_metadata_included_in_output(self, test_txt_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        result = await extractor.extract(test_txt_path, config)
        output = result.to_dict()

        # Metadata should be present (may be empty dict)
        assert "metadata" in output["document"]
        # Should include source type
        assert "source_type" in result.metadata or result.metadata == {}


class TestDocumentExtractorAsyncBehavior:
    """Tests for async/non-blocking behavior."""

    @pytest.mark.asyncio
    async def test_extraction_is_async(self, test_txt_path):
        """Verify extraction can run concurrently with other async tasks."""
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        # Run extraction alongside another task
        async def other_task():
            await asyncio.sleep(0.01)
            return "other"

        extraction_task = extractor.extract(test_txt_path, config)
        other = other_task()

        # Both should complete without blocking
        results = await asyncio.gather(extraction_task, other)

        assert results[1] == "other"
        assert results[0] is not None


class TestDocumentExtractorErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_error(self):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        with pytest.raises((FileNotFoundError, ValueError)):
            await extractor.extract(Path("/nonexistent/file.txt"), config)

    @pytest.mark.asyncio
    async def test_unsupported_file_type_raises_error(self, tmp_path):
        from app.pipeline.extraction.extractor import DocumentExtractor
        from app.pipeline.extraction.schemas import ExtractionConfig

        # Create a file with unsupported extension
        unsupported_file = tmp_path / "test.jpg"
        unsupported_file.write_text("not a document")

        extractor = DocumentExtractor()
        config = ExtractionConfig()

        with pytest.raises(ValueError) as exc_info:
            await extractor.extract(unsupported_file, config)

        assert "unsupported" in str(exc_info.value).lower() or "invalid" in str(
            exc_info.value
        ).lower()
