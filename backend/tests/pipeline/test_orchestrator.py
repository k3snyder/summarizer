"""
Tests for PipelineOrchestrator.
TDD red phase for bsc-4.1.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile


@pytest.fixture
def sample_pdf_path():
    """Create a simple test file path."""
    return Path("/tmp/test_document.pdf")


@pytest.fixture
def sample_config():
    """Sample pipeline configuration."""
    return {
        "extract_only": False,
        "skip_tables": False,
        "skip_images": False,
        "vision_mode": "ollama",
        "run_summarization": True,
        "summarizer_mode": "full",
    }


@pytest.fixture
def mock_extractor():
    """Mock DocumentExtractor."""
    from app.pipeline.extraction.schemas import ExtractedDocument, ExtractedPage

    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value=ExtractedDocument(
            document_id="doc_123",
            filename="test.pdf",
            total_pages=2,
            pages=[
                ExtractedPage(
                    page_number=1,
                    chunk_id="chunk_1",
                    doc_title="test.pdf",
                    text="Page 1 content",
                    image_base64="base64data1",
                ),
                ExtractedPage(
                    page_number=2,
                    chunk_id="chunk_2",
                    doc_title="test.pdf",
                    text="Page 2 content",
                    image_base64="base64data2",
                ),
            ],
        )
    )
    return extractor


@pytest.fixture
def mock_vision_processor():
    """Mock VisionProcessor."""
    from app.pipeline.vision.schemas import ClassificationResult, ExtractionResult

    processor = AsyncMock()
    processor.classify_pages = AsyncMock(
        side_effect=lambda pages, **kwargs: [
            ClassificationResult(
                page_number=page.get("page_number", i + 1),
                chunk_id=page.get("chunk_id", f"chunk_{i + 1}"),
                has_graphics=True,
            )
            for i, page in enumerate(pages)
        ]
    )
    processor.extract_visual_content = AsyncMock(
        side_effect=lambda pages, **kwargs: [
            ExtractionResult(
                page_number=page.get("page_number", i + 1),
                chunk_id=page.get("chunk_id", f"chunk_{i + 1}"),
                image_text="Visual content",
            )
            for i, page in enumerate(pages)
        ]
    )
    return processor


@pytest.fixture
def mock_summarize_service():
    """Mock SummarizeService."""
    from app.pipeline.summarization.schemas import SummaryResult

    service = AsyncMock()
    service.summarize_pages_batch = AsyncMock(
        side_effect=lambda contexts, **kwargs: [
            SummaryResult(
                page_number=ctx.page_number,
                chunk_id=ctx.chunk_id,
                summary_notes=["Note 1"],
                summary_topics=["Topic A"],
                summary_relevancy=92.0,
            )
            for ctx in contexts
        ]
    )
    return service


class TestOrchestratorStages:
    """Test orchestrator runs all stages."""

    @pytest.mark.asyncio
    async def test_orchestrator_runs_all_stages(
        self, sample_pdf_path, sample_config, mock_extractor, mock_vision_processor, mock_summarize_service
    ):
        """Test that orchestrator runs extraction, vision, and summarization."""
        from app.pipeline.orchestrator import PipelineOrchestrator

        with patch(
            "app.pipeline.orchestrator.DocumentExtractor", return_value=mock_extractor
        ), patch(
            "app.pipeline.orchestrator.VisionProcessor", return_value=mock_vision_processor
        ), patch(
            "app.pipeline.orchestrator.SummarizeService", return_value=mock_summarize_service
        ):
            orchestrator = PipelineOrchestrator()
            result, metrics = await orchestrator.run(
                job_id="job_123",
                input_file_path=sample_pdf_path,
                config=sample_config,
            )

            # All stages should have been called
            mock_extractor.extract.assert_called_once()
            mock_vision_processor.classify_pages.assert_called_once()
            mock_vision_processor.extract_visual_content.assert_called_once()
            mock_summarize_service.summarize_pages_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_calls_progress_callback_for_each_stage(
        self, sample_pdf_path, sample_config, mock_extractor, mock_vision_processor, mock_summarize_service
    ):
        """Test that progress callback is called for each stage."""
        from app.pipeline.orchestrator import PipelineOrchestrator
        from app.pipeline.models import PipelineStage

        progress_events = []

        async def progress_callback(event):
            progress_events.append(event)

        with patch(
            "app.pipeline.orchestrator.DocumentExtractor", return_value=mock_extractor
        ), patch(
            "app.pipeline.orchestrator.VisionProcessor", return_value=mock_vision_processor
        ), patch(
            "app.pipeline.orchestrator.SummarizeService", return_value=mock_summarize_service
        ):
            orchestrator = PipelineOrchestrator()
            await orchestrator.run(
                job_id="job_123",
                input_file_path=sample_pdf_path,
                config=sample_config,
                progress_callback=progress_callback,
            )

            # Should have events for all three stages
            stages = [e.stage for e in progress_events]
            assert PipelineStage.EXTRACTION in stages
            assert PipelineStage.VISION in stages
            assert PipelineStage.SUMMARIZATION in stages


class TestOrchestratorDataPassing:
    """Test data flow between stages."""

    @pytest.mark.asyncio
    async def test_orchestrator_passes_data_between_stages(
        self, sample_pdf_path, sample_config, mock_extractor, mock_vision_processor, mock_summarize_service
    ):
        """Test that data flows from extraction to vision to summarization."""
        from app.pipeline.orchestrator import PipelineOrchestrator

        with patch(
            "app.pipeline.orchestrator.DocumentExtractor", return_value=mock_extractor
        ), patch(
            "app.pipeline.orchestrator.VisionProcessor", return_value=mock_vision_processor
        ), patch(
            "app.pipeline.orchestrator.SummarizeService", return_value=mock_summarize_service
        ):
            orchestrator = PipelineOrchestrator()
            result, metrics = await orchestrator.run(
                job_id="job_123",
                input_file_path=sample_pdf_path,
                config=sample_config,
            )

            # Verify output has all enriched data
            assert "document" in result
            assert "pages" in result
            assert len(result["pages"]) == 2
            # Should have vision data
            assert result["pages"][0].get("image_text") is not None
            # Should have summarization data
            assert result["pages"][0].get("summary_notes") is not None


class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_handles_extraction_failure(
        self, sample_pdf_path, sample_config
    ):
        """Test that extraction failures are propagated."""
        from app.pipeline.orchestrator import PipelineOrchestrator

        failing_extractor = AsyncMock()
        failing_extractor.extract = AsyncMock(side_effect=Exception("Extraction failed"))

        with patch(
            "app.pipeline.orchestrator.DocumentExtractor", return_value=failing_extractor
        ):
            orchestrator = PipelineOrchestrator()

            with pytest.raises(Exception, match="Extraction failed"):
                await orchestrator.run(
                    job_id="job_123",
                    input_file_path=sample_pdf_path,
                    config=sample_config,
                )

    @pytest.mark.asyncio
    async def test_orchestrator_handles_vision_failure_gracefully(
        self, sample_pdf_path, sample_config, mock_extractor, mock_summarize_service
    ):
        """Test that vision failures don't stop the pipeline."""
        from app.pipeline.orchestrator import PipelineOrchestrator

        failing_vision = AsyncMock()
        failing_vision.process_pages = AsyncMock(side_effect=Exception("Vision failed"))

        with patch(
            "app.pipeline.orchestrator.DocumentExtractor", return_value=mock_extractor
        ), patch(
            "app.pipeline.orchestrator.VisionProcessor", return_value=failing_vision
        ), patch(
            "app.pipeline.orchestrator.SummarizeService", return_value=mock_summarize_service
        ):
            orchestrator = PipelineOrchestrator()
            result, metrics = await orchestrator.run(
                job_id="job_123",
                input_file_path=sample_pdf_path,
                config=sample_config,
            )

            # Should still complete with extraction data
            assert "pages" in result
            assert len(result["pages"]) == 2


class TestOrchestratorSkipModes:
    """Test skip configurations."""

    @pytest.mark.asyncio
    async def test_orchestrator_skips_vision_when_configured(
        self, sample_pdf_path, mock_extractor, mock_summarize_service
    ):
        """Test that vision is skipped when configured."""
        from app.pipeline.orchestrator import PipelineOrchestrator

        config = {
            "extract_only": False,
            "skip_images": True,  # Skip vision
            "run_summarization": True,
        }

        with patch(
            "app.pipeline.orchestrator.DocumentExtractor", return_value=mock_extractor
        ), patch(
            "app.pipeline.orchestrator.VisionProcessor"
        ) as MockVision, patch(
            "app.pipeline.orchestrator.SummarizeService", return_value=mock_summarize_service
        ):
            orchestrator = PipelineOrchestrator()
            await orchestrator.run(
                job_id="job_123",
                input_file_path=sample_pdf_path,
                config=config,
            )

            # Vision processor should not be instantiated
            MockVision.return_value.classify_pages.assert_not_called()

    @pytest.mark.asyncio
    async def test_orchestrator_skips_summarization_when_configured(
        self, sample_pdf_path, mock_extractor, mock_vision_processor
    ):
        """Test that summarization is skipped when configured."""
        from app.pipeline.orchestrator import PipelineOrchestrator

        config = {
            "extract_only": False,
            "run_summarization": False,  # Skip summarization
        }

        with patch(
            "app.pipeline.orchestrator.DocumentExtractor", return_value=mock_extractor
        ), patch(
            "app.pipeline.orchestrator.VisionProcessor", return_value=mock_vision_processor
        ), patch(
            "app.pipeline.orchestrator.SummarizeService"
        ) as MockSummarize:
            orchestrator = PipelineOrchestrator()
            await orchestrator.run(
                job_id="job_123",
                input_file_path=sample_pdf_path,
                config=config,
            )

            # Summarize service should not be called
            MockSummarize.return_value.summarize_pages_batch.assert_not_called()
