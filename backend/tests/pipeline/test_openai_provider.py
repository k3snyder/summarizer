"""
Tests for OpenAI vision provider.
TDD red phase for bsc-2.5.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def sample_base64_image():
    """Simple 1x1 white JPEG as base64."""
    return "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="


@pytest.fixture
def openai_provider():
    """Create OpenAI provider with test configuration."""
    from app.pipeline.vision.providers.openai import OpenAIVisionProvider

    return OpenAIVisionProvider(api_key="test-api-key")


class TestOpenAIClassify:
    """Test OpenAI classify method."""

    @pytest.mark.asyncio
    async def test_openai_classify_returns_classification_result(
        self, openai_provider, sample_base64_image
    ):
        """Test that classify returns a ClassificationResult."""
        from app.pipeline.vision.schemas import ClassificationResult

        # Mock the OpenAI client response
        mock_message = MagicMock()
        mock_message.content = "YES"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            openai_provider._client.chat.completions,
            "create",
            return_value=mock_response,
        ):
            result = await openai_provider.classify(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            assert isinstance(result, ClassificationResult)
            assert result.page_number == 1
            assert result.chunk_id == "chunk_1"
            assert result.has_graphics is True

    @pytest.mark.asyncio
    async def test_openai_classify_no_graphics(
        self, openai_provider, sample_base64_image
    ):
        """Test that classify returns False for NO response."""
        mock_message = MagicMock()
        mock_message.content = "NO"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            openai_provider._client.chat.completions,
            "create",
            return_value=mock_response,
        ):
            result = await openai_provider.classify(
                sample_base64_image, page_number=2, chunk_id="chunk_2"
            )

            assert result.has_graphics is False

    @pytest.mark.asyncio
    async def test_openai_handles_api_error(self, openai_provider, sample_base64_image):
        """Test that API errors are handled gracefully."""
        from openai import APIError

        with patch.object(
            openai_provider._client.chat.completions,
            "create",
            side_effect=APIError(
                message="API Error", request=MagicMock(), body=None
            ),
        ):
            result = await openai_provider.classify(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            # Should return result with error, defaulting to has_graphics=True
            assert result.has_graphics is True
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_openai_uses_gpt4o_model(self, openai_provider, sample_base64_image):
        """Test that GPT-4o model is used by default."""
        mock_message = MagicMock()
        mock_message.content = "YES"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            openai_provider._client.chat.completions,
            "create",
            return_value=mock_response,
        ) as mock_create:
            await openai_provider.classify(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            # Verify model parameter
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert "gpt-4" in call_kwargs.get("model", "").lower()


class TestOpenAIExtract:
    """Test OpenAI extract method."""

    @pytest.mark.asyncio
    async def test_openai_extract_returns_extraction_result(
        self, openai_provider, sample_base64_image
    ):
        """Test that extract returns an ExtractionResult."""
        from app.pipeline.vision.schemas import ExtractionResult

        mock_message = MagicMock()
        mock_message.content = "## Visual Elements\nChart showing data"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            openai_provider._client.chat.completions,
            "create",
            return_value=mock_response,
        ):
            result = await openai_provider.extract(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            assert isinstance(result, ExtractionResult)
            assert result.page_number == 1
            assert result.chunk_id == "chunk_1"
            assert result.image_text == "## Visual Elements\nChart showing data"

    @pytest.mark.asyncio
    async def test_openai_extract_handles_api_error(
        self, openai_provider, sample_base64_image
    ):
        """Test that API errors in extract are handled."""
        from openai import APIError

        with patch.object(
            openai_provider._client.chat.completions,
            "create",
            side_effect=APIError(
                message="API Error", request=MagicMock(), body=None
            ),
        ):
            result = await openai_provider.extract(
                sample_base64_image, page_number=1, chunk_id="chunk_1"
            )

            assert result.image_text is None
            assert result.error is not None
