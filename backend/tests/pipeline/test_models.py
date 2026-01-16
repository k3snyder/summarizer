"""
TDD Red Phase: Tests for pipeline package structure and models.

These tests define the expected behavior for:
- ExtractionConfig: Configuration for document extraction
- ExtractedPage: Individual page data from extraction
- ExtractedDocument: Complete document with pages
- ProgressEvent: Real-time progress updates
- PipelineStage: Pipeline stage enum
"""

import pytest
from dataclasses import fields, is_dataclass
from enum import Enum


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_pipeline_stage_is_enum(self):
        from app.pipeline.models import PipelineStage

        assert issubclass(PipelineStage, Enum)

    def test_pipeline_stage_has_extraction(self):
        from app.pipeline.models import PipelineStage

        assert hasattr(PipelineStage, "EXTRACTION")
        assert PipelineStage.EXTRACTION.value == "extraction"

    def test_pipeline_stage_has_vision(self):
        from app.pipeline.models import PipelineStage

        assert hasattr(PipelineStage, "VISION")
        assert PipelineStage.VISION.value == "vision"

    def test_pipeline_stage_has_summarization(self):
        from app.pipeline.models import PipelineStage

        assert hasattr(PipelineStage, "SUMMARIZATION")
        assert PipelineStage.SUMMARIZATION.value == "summarization"


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_progress_event_is_dataclass(self):
        from app.pipeline.models import ProgressEvent

        assert is_dataclass(ProgressEvent)

    def test_progress_event_started_type(self):
        from app.pipeline.models import ProgressEvent, PipelineStage

        event = ProgressEvent(
            event_type="started",
            stage=PipelineStage.EXTRACTION,
            message="Starting extraction",
        )
        assert event.event_type == "started"
        assert event.stage == PipelineStage.EXTRACTION
        assert event.message == "Starting extraction"

    def test_progress_event_progress_type(self):
        from app.pipeline.models import ProgressEvent, PipelineStage

        event = ProgressEvent(
            event_type="progress",
            stage=PipelineStage.EXTRACTION,
            current=5,
            total=10,
            message="Processing page 5 of 10",
        )
        assert event.event_type == "progress"
        assert event.current == 5
        assert event.total == 10

    def test_progress_event_stage_changed_type(self):
        from app.pipeline.models import ProgressEvent, PipelineStage

        event = ProgressEvent(
            event_type="stage_changed",
            stage=PipelineStage.VISION,
            message="Entering vision stage",
        )
        assert event.event_type == "stage_changed"
        assert event.stage == PipelineStage.VISION

    def test_progress_event_completed_type(self):
        from app.pipeline.models import ProgressEvent, PipelineStage

        event = ProgressEvent(
            event_type="completed",
            stage=PipelineStage.SUMMARIZATION,
            message="Pipeline completed successfully",
        )
        assert event.event_type == "completed"

    def test_progress_event_failed_type(self):
        from app.pipeline.models import ProgressEvent, PipelineStage

        event = ProgressEvent(
            event_type="failed",
            stage=PipelineStage.EXTRACTION,
            message="Failed to parse PDF",
            error="Invalid PDF format",
        )
        assert event.event_type == "failed"
        assert event.error == "Invalid PDF format"

    def test_progress_event_percentage_property(self):
        from app.pipeline.models import ProgressEvent, PipelineStage

        event = ProgressEvent(
            event_type="progress",
            stage=PipelineStage.EXTRACTION,
            current=3,
            total=10,
        )
        assert event.percentage == 30.0

    def test_progress_event_percentage_with_zero_total(self):
        from app.pipeline.models import ProgressEvent, PipelineStage

        event = ProgressEvent(
            event_type="progress",
            stage=PipelineStage.EXTRACTION,
            current=0,
            total=0,
        )
        assert event.percentage == 0.0


class TestExtractionConfig:
    """Tests for ExtractionConfig dataclass."""

    def test_extraction_config_is_dataclass(self):
        from app.pipeline.extraction.schemas import ExtractionConfig

        assert is_dataclass(ExtractionConfig)

    def test_extraction_config_default_values(self):
        from app.pipeline.extraction.schemas import ExtractionConfig

        config = ExtractionConfig()
        assert config.skip_tables is False
        assert config.skip_images is False
        assert config.text_only is False
        assert config.pdf_image_dpi == 200
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_extraction_config_custom_values(self):
        from app.pipeline.extraction.schemas import ExtractionConfig

        config = ExtractionConfig(
            skip_tables=True,
            skip_images=True,
            text_only=True,
            pdf_image_dpi=300,
            chunk_size=2000,
            chunk_overlap=400,
        )
        assert config.skip_tables is True
        assert config.skip_images is True
        assert config.text_only is True
        assert config.pdf_image_dpi == 300
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 400


class TestExtractedPage:
    """Tests for ExtractedPage dataclass."""

    def test_extracted_page_is_dataclass(self):
        from app.pipeline.extraction.schemas import ExtractedPage

        assert is_dataclass(ExtractedPage)

    def test_extracted_page_required_fields(self):
        from app.pipeline.extraction.schemas import ExtractedPage

        page = ExtractedPage(
            page_number=1,
            chunk_id="doc_abc_page_001",
            doc_title="Test Document",
            text="Page content here",
        )
        assert page.page_number == 1
        assert page.chunk_id == "doc_abc_page_001"
        assert page.doc_title == "Test Document"
        assert page.text == "Page content here"

    def test_extracted_page_optional_fields_default_to_empty(self):
        from app.pipeline.extraction.schemas import ExtractedPage

        page = ExtractedPage(
            page_number=1,
            chunk_id="doc_abc_page_001",
            doc_title="Test Document",
            text="Page content here",
        )
        assert page.tables == []
        assert page.image_base64 is None
        assert page.image_text is None
        assert page.image_classifier is None

    def test_extracted_page_with_tables(self):
        from app.pipeline.extraction.schemas import ExtractedPage

        tables = [
            [["Header1", "Header2"], ["Row1Col1", "Row1Col2"]],
            [["A", "B"], ["C", "D"]],
        ]
        page = ExtractedPage(
            page_number=1,
            chunk_id="doc_abc_page_001",
            doc_title="Test Document",
            text="Page content here",
            tables=tables,
        )
        assert len(page.tables) == 2
        assert page.tables[0][0] == ["Header1", "Header2"]

    def test_extracted_page_with_image(self):
        from app.pipeline.extraction.schemas import ExtractedPage

        page = ExtractedPage(
            page_number=1,
            chunk_id="doc_abc_page_001",
            doc_title="Test Document",
            text="Page content here",
            image_base64="base64encodedstring",
        )
        assert page.image_base64 == "base64encodedstring"


class TestExtractedDocument:
    """Tests for ExtractedDocument dataclass."""

    def test_extracted_document_is_dataclass(self):
        from app.pipeline.extraction.schemas import ExtractedDocument

        assert is_dataclass(ExtractedDocument)

    def test_extracted_document_required_fields(self):
        from app.pipeline.extraction.schemas import ExtractedDocument

        doc = ExtractedDocument(
            document_id="abc123",
            filename="test.pdf",
            total_pages=5,
            pages=[],
        )
        assert doc.document_id == "abc123"
        assert doc.filename == "test.pdf"
        assert doc.total_pages == 5
        assert doc.pages == []

    def test_extracted_document_with_pages(self):
        from app.pipeline.extraction.schemas import ExtractedDocument, ExtractedPage

        pages = [
            ExtractedPage(
                page_number=1,
                chunk_id="doc_abc_page_001",
                doc_title="Test",
                text="Content 1",
            ),
            ExtractedPage(
                page_number=2,
                chunk_id="doc_abc_page_002",
                doc_title="Test",
                text="Content 2",
            ),
        ]
        doc = ExtractedDocument(
            document_id="abc123",
            filename="test.pdf",
            total_pages=2,
            pages=pages,
        )
        assert len(doc.pages) == 2
        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2

    def test_extracted_document_optional_metadata(self):
        from app.pipeline.extraction.schemas import ExtractedDocument

        doc = ExtractedDocument(
            document_id="abc123",
            filename="test.pdf",
            total_pages=5,
            pages=[],
            metadata={"author": "John", "created": "2026-01-08"},
        )
        assert doc.metadata == {"author": "John", "created": "2026-01-08"}

    def test_extracted_document_metadata_defaults_to_empty_dict(self):
        from app.pipeline.extraction.schemas import ExtractedDocument

        doc = ExtractedDocument(
            document_id="abc123",
            filename="test.pdf",
            total_pages=5,
            pages=[],
        )
        assert doc.metadata == {}

    def test_extracted_document_to_dict(self):
        """Test that ExtractedDocument can be serialized to a dict matching the output schema."""
        from app.pipeline.extraction.schemas import ExtractedDocument, ExtractedPage

        pages = [
            ExtractedPage(
                page_number=1,
                chunk_id="doc_abc_page_001",
                doc_title="Test Doc",
                text="Page 1 content",
                tables=[],
            ),
        ]
        doc = ExtractedDocument(
            document_id="abc123",
            filename="test.pdf",
            total_pages=1,
            pages=pages,
        )

        # Should have a to_dict method for serialization
        result = doc.to_dict()

        assert "document" in result
        assert "pages" in result
        assert result["document"]["document_id"] == "abc123"
        assert result["document"]["filename"] == "test.pdf"
        assert result["document"]["total_pages"] == 1
        assert len(result["pages"]) == 1
        assert result["pages"][0]["chunk_id"] == "doc_abc_page_001"
