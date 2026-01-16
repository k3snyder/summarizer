"""
Tests for VisionProcessor service.
TDD red phase for bsc-2.7.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional


@pytest.fixture
def sample_pages():
    """Sample pages with image data."""
    return [
        {
            "page_number": 1,
            "chunk_id": "chunk_1",
            "image_base64": ["base64data1"],
        },
        {
            "page_number": 2,
            "chunk_id": "chunk_2",
            "image_base64": ["base64data2"],
        },
        {
            "page_number": 3,
            "chunk_id": "chunk_3",
            "image_base64": ["base64data3"],
        },
    ]


@pytest.fixture
def mock_provider():
    """Create a mock vision provider."""
    from app.pipeline.vision.schemas import ClassificationResult, ExtractionResult

    provider = AsyncMock()
    provider.classify = AsyncMock(
        side_effect=lambda img, pn, cid: ClassificationResult(
            page_number=pn,
            chunk_id=cid,
            has_graphics=True,
        )
    )
    provider.extract = AsyncMock(
        side_effect=lambda img, pn, cid: ExtractionResult(
            page_number=pn,
            chunk_id=cid,
            image_text=f"Extracted from {cid}",
        )
    )
    return provider


class TestVisionProcessorClassify:
    """Test VisionProcessor classify_pages method."""

    @pytest.mark.asyncio
    async def test_classify_pages_batches_correctly(self, sample_pages, mock_provider):
        """Test that pages are batched correctly for classification."""
        from app.pipeline.vision.processor import VisionProcessor
        from app.pipeline.vision.schemas import VisionConfig, VisionProvider

        config = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            extractor_provider=VisionProvider.OLLAMA,
            batch_size=2,
        )

        with patch.object(
            VisionProcessor, "_create_provider", return_value=mock_provider
        ):
            processor = VisionProcessor(config)
            results = await processor.classify_pages(sample_pages)

            assert len(results) == 3
            assert mock_provider.classify.call_count == 3

    @pytest.mark.asyncio
    async def test_classify_returns_classification_results(
        self, sample_pages, mock_provider
    ):
        """Test that classify_pages returns list of ClassificationResult."""
        from app.pipeline.vision.processor import VisionProcessor
        from app.pipeline.vision.schemas import (
            VisionConfig,
            VisionProvider,
            ClassificationResult,
        )

        config = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            extractor_provider=VisionProvider.OLLAMA,
        )

        with patch.object(
            VisionProcessor, "_create_provider", return_value=mock_provider
        ):
            processor = VisionProcessor(config)
            results = await processor.classify_pages(sample_pages)

            assert all(isinstance(r, ClassificationResult) for r in results)
            assert results[0].chunk_id == "chunk_1"
            assert results[1].chunk_id == "chunk_2"


class TestVisionProcessorExtract:
    """Test VisionProcessor extract_visual_content method."""

    @pytest.mark.asyncio
    async def test_extract_visual_content_batches_correctly(
        self, sample_pages, mock_provider
    ):
        """Test that pages are batched correctly for extraction."""
        from app.pipeline.vision.processor import VisionProcessor
        from app.pipeline.vision.schemas import VisionConfig, VisionProvider

        config = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            extractor_provider=VisionProvider.OLLAMA,
            batch_size=2,
        )

        with patch.object(
            VisionProcessor, "_create_provider", return_value=mock_provider
        ):
            processor = VisionProcessor(config)
            results = await processor.extract_visual_content(sample_pages)

            assert len(results) == 3
            assert mock_provider.extract.call_count == 3

    @pytest.mark.asyncio
    async def test_extract_returns_extraction_results(
        self, sample_pages, mock_provider
    ):
        """Test that extract_visual_content returns list of ExtractionResult."""
        from app.pipeline.vision.processor import VisionProcessor
        from app.pipeline.vision.schemas import (
            VisionConfig,
            VisionProvider,
            ExtractionResult,
        )

        config = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            extractor_provider=VisionProvider.OLLAMA,
        )

        with patch.object(
            VisionProcessor, "_create_provider", return_value=mock_provider
        ):
            processor = VisionProcessor(config)
            results = await processor.extract_visual_content(sample_pages)

            assert all(isinstance(r, ExtractionResult) for r in results)
            assert results[0].image_text == "Extracted from chunk_1"


class TestVisionProcessorProvider:
    """Test VisionProcessor provider selection."""

    def test_processor_uses_configured_provider(self):
        """Test that processor creates the correct provider types."""
        from app.pipeline.vision.processor import VisionProcessor
        from app.pipeline.vision.schemas import VisionConfig, VisionProvider
        from app.pipeline.vision.providers.ollama import OllamaVisionProvider
        from app.pipeline.vision.providers.openai import OpenAIVisionProvider

        # Test Ollama for both
        config_ollama = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            extractor_provider=VisionProvider.OLLAMA,
        )
        processor_ollama = VisionProcessor(config_ollama)
        assert isinstance(processor_ollama._classifier_provider, OllamaVisionProvider)
        assert isinstance(processor_ollama._extractor_provider, OllamaVisionProvider)

        # Test mixed providers
        config_mixed = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            extractor_provider=VisionProvider.OPENAI,
            openai_api_key="test-key",
        )
        processor_mixed = VisionProcessor(config_mixed)
        assert isinstance(processor_mixed._classifier_provider, OllamaVisionProvider)
        assert isinstance(processor_mixed._extractor_provider, OpenAIVisionProvider)


class TestVisionProcessorErrorHandling:
    """Test VisionProcessor error handling."""

    @pytest.mark.asyncio
    async def test_processor_handles_provider_failure_gracefully(self, sample_pages):
        """Test that provider failures are handled gracefully."""
        from app.pipeline.vision.processor import VisionProcessor
        from app.pipeline.vision.schemas import (
            VisionConfig,
            VisionProvider,
            ClassificationResult,
        )

        failing_provider = AsyncMock()
        failing_provider.classify = AsyncMock(
            side_effect=lambda img, pn, cid: ClassificationResult(
                page_number=pn,
                chunk_id=cid,
                has_graphics=True,
                error="Provider failed",
            )
        )

        config = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            extractor_provider=VisionProvider.OLLAMA,
        )

        with patch.object(
            VisionProcessor, "_create_provider", return_value=failing_provider
        ):
            processor = VisionProcessor(config)
            results = await processor.classify_pages(sample_pages)

            # Should still return results with errors
            assert len(results) == 3
            assert all(r.error == "Provider failed" for r in results)


class TestVisionProcessorConcurrency:
    """Test VisionProcessor concurrent processing."""

    @pytest.mark.asyncio
    async def test_concurrent_processing_respects_batch_size(
        self, sample_pages, mock_provider
    ):
        """Test that concurrent processing respects batch size limits."""
        from app.pipeline.vision.processor import VisionProcessor
        from app.pipeline.vision.schemas import VisionConfig, VisionProvider
        import asyncio

        config = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            extractor_provider=VisionProvider.OLLAMA,
            batch_size=1,
        )

        # Track concurrent calls
        concurrent_calls = 0
        max_concurrent = 0

        async def track_concurrency(img, pn, cid):
            nonlocal concurrent_calls, max_concurrent
            from app.pipeline.vision.schemas import ClassificationResult

            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)
            await asyncio.sleep(0.01)  # Simulate work
            concurrent_calls -= 1
            return ClassificationResult(
                page_number=pn,
                chunk_id=cid,
                has_graphics=True,
            )

        mock_provider.classify = track_concurrency

        with patch.object(
            VisionProcessor, "_create_provider", return_value=mock_provider
        ):
            processor = VisionProcessor(config)
            await processor.classify_pages(sample_pages)

            # With batch_size=1, max concurrent should be 1
            assert max_concurrent == 1
