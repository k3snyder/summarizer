"""
Tests for Ollama vision provider.
TDD red phase for bsc-2.3.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


@pytest.fixture
def sample_base64_image():
    """Simple 1x1 white JPEG as base64."""
    return "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="


@pytest.fixture
def ollama_provider():
    """Create Ollama provider with test configuration."""
    from app.pipeline.vision.providers.ollama import OllamaVisionProvider

    return OllamaVisionProvider(
        base_url="http://localhost:11434/v1",
        model="test-model",
    )


class TestOllamaClassify:
    """Test Ollama classify method."""

    @pytest.mark.asyncio
    async def test_ollama_classify_returns_classification_result(
        self, ollama_provider, sample_base64_image
    ):
        """Test that classify returns a ClassificationResult."""
        from app.pipeline.vision.schemas import ClassificationResult

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "YES"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_provider._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await ollama_provider.classify(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            assert isinstance(result, ClassificationResult)
            assert result.page_number == 1
            assert result.chunk_id == "chunk_1"
            assert result.has_graphics is True

    @pytest.mark.asyncio
    async def test_ollama_classify_no_graphics(
        self, ollama_provider, sample_base64_image
    ):
        """Test that classify returns False for NO response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "NO"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_provider._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await ollama_provider.classify(
                sample_base64_image, page_number=2, chunk_id="chunk_2"
            )

            assert result.has_graphics is False

    @pytest.mark.asyncio
    async def test_ollama_handles_connection_error(
        self, ollama_provider, sample_base64_image
    ):
        """Test that connection errors are handled gracefully."""
        with patch.object(
            ollama_provider._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            result = await ollama_provider.classify(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            # Should return result with error, defaulting to has_graphics=True
            assert result.has_graphics is True
            assert result.error is not None
            assert "Connection" in result.error

    @pytest.mark.asyncio
    async def test_ollama_uses_correct_endpoint(
        self, ollama_provider, sample_base64_image
    ):
        """Test that correct OpenAI-compatible endpoint is used."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "YES"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_provider._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            await ollama_provider.classify(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            # Verify POST was called with correct endpoint
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/chat/completions" in str(call_args)


class TestOllamaExtract:
    """Test Ollama extract method."""

    @pytest.mark.asyncio
    async def test_ollama_extract_returns_extraction_result(
        self, ollama_provider, sample_base64_image
    ):
        """Test that extract returns an ExtractionResult."""
        from app.pipeline.vision.schemas import ExtractionResult

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "## Visual Elements\nChart showing data"}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            ollama_provider._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            result = await ollama_provider.extract(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            assert isinstance(result, ExtractionResult)
            assert result.page_number == 1
            assert result.chunk_id == "chunk_1"
            assert result.image_text == "## Visual Elements\nChart showing data"

    @pytest.mark.asyncio
    async def test_ollama_extract_handles_connection_error(
        self, ollama_provider, sample_base64_image
    ):
        """Test that connection errors in extract are handled."""
        with patch.object(
            ollama_provider._client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            result = await ollama_provider.extract(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            assert result.image_text is None
            assert result.error is not None
