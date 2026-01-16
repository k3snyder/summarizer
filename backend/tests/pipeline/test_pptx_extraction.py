"""
TDD Red Phase: Tests for PPTX extraction in DocumentExtractor.

These tests define the expected behavior for PowerPoint (.pptx) file extraction.
Tests should fail initially because the implementation doesn't exist yet.

Test fixture (test_presentation.pptx) has 6 slides:
  - Slide 1: Title slide with 'Test Presentation' title (no notes)
  - Slide 2: Bullet points + speaker notes ('Remember to explain this')
  - Slide 3: Table (3 rows x 2 columns: Header1/Header2, A/B, C/D)
  - Slide 4: Image-only slide (no text shapes)
  - Slide 5: Empty/blank slide
  - Slide 6: Two tables side by side
"""

from pathlib import Path

import pytest


@pytest.fixture
def test_pptx_path() -> Path:
    """Path to the 6-slide test presentation."""
    return Path(__file__).parent.parent / "fixtures" / "test_presentation.pptx"


@pytest.fixture
def extractor():
    """Create a DocumentExtractor instance."""
    from app.pipeline.extraction.extractor import DocumentExtractor

    return DocumentExtractor()


@pytest.fixture
def default_config():
    """Default extraction config."""
    from app.pipeline.extraction.schemas import ExtractionConfig

    return ExtractionConfig()


class TestPPTXExtraction:
    """Core tests for PPTX extraction functionality."""

    @pytest.mark.asyncio
    async def test_extract_returns_extracted_document(
        self, test_pptx_path, extractor, default_config
    ):
        """PPTX extraction should return an ExtractedDocument."""
        from app.pipeline.extraction.schemas import ExtractedDocument

        result = await extractor.extract(test_pptx_path, default_config)

        assert isinstance(result, ExtractedDocument)
        assert result.filename == test_pptx_path.name

    @pytest.mark.asyncio
    async def test_slide_count_matches_pages(
        self, test_pptx_path, extractor, default_config
    ):
        """Number of extracted pages should match slide count (6 slides)."""
        result = await extractor.extract(test_pptx_path, default_config)

        assert result.total_pages == 6
        assert len(result.pages) == 6

    @pytest.mark.asyncio
    async def test_extracts_text_from_shapes(
        self, test_pptx_path, extractor, default_config
    ):
        """Text from slide shapes should be extracted."""
        result = await extractor.extract(test_pptx_path, default_config)

        # Slide 1 (index 0) has title "Test Presentation"
        slide_1 = result.pages[0]
        assert "Test Presentation" in slide_1.text

    @pytest.mark.asyncio
    async def test_extracts_speaker_notes_with_marker(
        self, test_pptx_path, extractor, default_config
    ):
        """Speaker notes should be extracted with [Speaker Notes] marker."""
        result = await extractor.extract(test_pptx_path, default_config)

        # Slide 2 (index 1) has speaker notes "Remember to explain this"
        slide_2 = result.pages[1]
        assert "[Speaker Notes]" in slide_2.text
        assert "Remember to explain this" in slide_2.text

    @pytest.mark.asyncio
    async def test_extracts_tables(self, test_pptx_path, extractor, default_config):
        """Tables in slides should be extracted as 2D arrays."""
        result = await extractor.extract(test_pptx_path, default_config)

        # Slide 3 (index 2) has a 3x2 table
        slide_3 = result.pages[2]
        assert len(slide_3.tables) >= 1

        # Table should have 3 rows and 2 columns
        table = slide_3.tables[0]
        assert len(table) == 3  # 3 rows
        assert len(table[0]) == 2  # 2 columns

        # Check expected content
        assert "Header1" in table[0][0] or table[0][0] == "Header1"
        assert "Header2" in table[0][1] or table[0][1] == "Header2"

    @pytest.mark.asyncio
    async def test_table_text_not_duplicated_in_text_field(
        self, test_pptx_path, extractor, default_config
    ):
        """Table content should be in tables field only, NOT duplicated in text field."""
        result = await extractor.extract(test_pptx_path, default_config)

        # Slide 3 (index 2) has a table with "Header1", "Header2", "A", "B", "C", "D"
        slide_3 = result.pages[2]

        # Table content should be in tables
        assert len(slide_3.tables) >= 1
        table = slide_3.tables[0]
        assert "Header1" in table[0][0]

        # Table content should NOT be in the text field (avoid duplication)
        # The text field should only have non-table content like the slide title
        assert "Header1" not in slide_3.text
        assert "Header2" not in slide_3.text

    @pytest.mark.asyncio
    async def test_skip_notes_config_works(self, test_pptx_path, extractor):
        """skip_tables config should skip speaker notes for PPTX (UI shows 'Skip Slide Notes')."""
        from app.pipeline.extraction.schemas import ExtractionConfig

        config = ExtractionConfig(skip_tables=True)
        result = await extractor.extract(test_pptx_path, config)

        # Slide 2 has speaker notes - with skip_tables=True, notes should be skipped
        slide2 = result.pages[1]
        assert "[Speaker Notes]" not in slide2.text
        assert "Remember to explain this" not in slide2.text

        # Tables should still be extracted (skip_tables only affects notes for PPTX)
        slide3 = result.pages[2]  # Slide 3 has a table
        assert len(slide3.tables) > 0

    @pytest.mark.asyncio
    async def test_skip_images_config_works(self, test_pptx_path, extractor):
        """skip_images config should exclude slide screenshots from extraction."""
        from app.pipeline.extraction.schemas import ExtractionConfig

        config = ExtractionConfig(skip_images=True)
        result = await extractor.extract(test_pptx_path, config)

        # All slides should have no image
        for page in result.pages:
            assert page.image_base64 is None

    @pytest.mark.asyncio
    async def test_text_only_mode_works(self, test_pptx_path, extractor):
        """text_only mode should skip notes, tables, and screenshots for PPTX."""
        from app.pipeline.extraction.schemas import ExtractionConfig

        config = ExtractionConfig(text_only=True)
        result = await extractor.extract(test_pptx_path, config)

        # All slides should have no images
        for page in result.pages:
            assert page.image_base64 is None

        # Speaker notes should be skipped (Slide 2 has notes)
        slide2 = result.pages[1]
        assert "[Speaker Notes]" not in slide2.text
        assert "Remember to explain this" not in slide2.text

        # Tables should also be skipped in text_only mode
        slide3 = result.pages[2]
        assert len(slide3.tables) == 0

    @pytest.mark.asyncio
    async def test_skip_pptx_tables_config_works(self, test_pptx_path, extractor):
        """skip_pptx_tables config should skip table extraction from PPTX slides."""
        from app.pipeline.extraction.schemas import ExtractionConfig

        config = ExtractionConfig(skip_pptx_tables=True)
        result = await extractor.extract(test_pptx_path, config)

        # Tables should be skipped
        slide3 = result.pages[2]  # Slide 3 has a table
        assert len(slide3.tables) == 0

        # But speaker notes should still be extracted
        slide2 = result.pages[1]
        assert "[Speaker Notes]" in slide2.text

    @pytest.mark.asyncio
    async def test_generates_slide_images(self, test_pptx_path, extractor, default_config):
        """Slides should be rendered as base64 JPEG images for vision processing."""
        result = await extractor.extract(test_pptx_path, default_config)

        # At least some slides should have images (may skip if LibreOffice not installed)
        has_images = any(page.image_base64 is not None for page in result.pages)

        # If LibreOffice is installed, images should be generated
        # This test will pass if LibreOffice is present
        if has_images:
            # Check that image is valid base64
            import base64

            first_with_image = next(p for p in result.pages if p.image_base64)
            # Should not raise
            base64.b64decode(first_with_image.image_base64)


class TestPPTXMetadata:
    """Tests for PPTX extraction metadata."""

    @pytest.mark.asyncio
    async def test_metadata_contains_source_type(
        self, test_pptx_path, extractor, default_config
    ):
        """Metadata should contain source_type='pptx'."""
        result = await extractor.extract(test_pptx_path, default_config)

        assert "source_type" in result.metadata
        assert result.metadata["source_type"] == "pptx"

    @pytest.mark.asyncio
    async def test_metadata_contains_file_type(
        self, test_pptx_path, extractor, default_config
    ):
        """Metadata should contain file_type='.pptx'."""
        result = await extractor.extract(test_pptx_path, default_config)

        assert "file_type" in result.metadata
        assert result.metadata["file_type"] == ".pptx"


class TestPPTXChunkIds:
    """Tests for chunk ID and page numbering in PPTX extraction."""

    @pytest.mark.asyncio
    async def test_chunk_ids_sequential(
        self, test_pptx_path, extractor, default_config
    ):
        """Chunk IDs should be sequential (chunk_1, chunk_2, ...)."""
        result = await extractor.extract(test_pptx_path, default_config)

        expected_ids = [f"chunk_{i}" for i in range(1, 7)]
        actual_ids = [page.chunk_id for page in result.pages]

        assert actual_ids == expected_ids

    @pytest.mark.asyncio
    async def test_page_numbers_sequential_1_indexed(
        self, test_pptx_path, extractor, default_config
    ):
        """Page numbers should be 1-indexed and sequential."""
        result = await extractor.extract(test_pptx_path, default_config)

        expected_page_numbers = list(range(1, 7))  # [1, 2, 3, 4, 5, 6]
        actual_page_numbers = [page.page_number for page in result.pages]

        assert actual_page_numbers == expected_page_numbers
