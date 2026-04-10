"""
Tests for llama.cpp vision provider.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_base64_image():
    """Simple 1x1 white JPEG as base64."""
    return "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="


@pytest.fixture
def llama_cpp_provider():
    """Create llama.cpp provider with test configuration."""
    from app.pipeline.vision.providers.llama_cpp import LlamaCppVisionProvider

    return LlamaCppVisionProvider(
        base_url="http://localhost:11439/v1",
        api_key="sk-test",
        model="model.gguf",
    )


class TestLlamaCppClassify:
    """Test llama.cpp classify method."""

    @pytest.mark.asyncio
    async def test_llama_cpp_classify_returns_classification_result(
        self, llama_cpp_provider, sample_base64_image
    ):
        """Test that classify returns a ClassificationResult."""
        from app.pipeline.vision.schemas import ClassificationResult

        mock_message = MagicMock()
        mock_message.content = "YES"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            llama_cpp_provider._client.chat.completions,
            "create",
            return_value=mock_response,
        ):
            result = await llama_cpp_provider.classify(
                sample_base64_image,
                page_number=1,
                chunk_id="chunk_1",
            )

            assert isinstance(result, ClassificationResult)
            assert result.page_number == 1
            assert result.chunk_id == "chunk_1"
            assert result.has_graphics is True

    @pytest.mark.asyncio
    async def test_llama_cpp_classify_uses_openai_compatible_payload(
        self, llama_cpp_provider, sample_base64_image
    ):
        """Test that llama.cpp uses image_url content blocks and max_tokens."""
        mock_message = MagicMock()
        mock_message.content = "YES"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            llama_cpp_provider._client.chat.completions,
            "create",
            return_value=mock_response,
        ) as mock_create:
            await llama_cpp_provider.classify(
                sample_base64_image,
                page_number=1,
                chunk_id="chunk_1",
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "model.gguf"
            assert call_kwargs["max_tokens"] == 10
            assert call_kwargs["messages"][0]["content"][1]["type"] == "image_url"
            assert call_kwargs["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
            assert call_kwargs["extra_body"]["reasoning"] == "off"
            assert call_kwargs["extra_body"]["reasoning_budget"] == 0
            assert call_kwargs["extra_body"]["reasoning_in_content"] is False
            assert call_kwargs["extra_body"]["reasoning_format"] == "none"
            assert call_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_llama_cpp_handles_api_error(
        self, llama_cpp_provider, sample_base64_image
    ):
        """Test that API errors are handled gracefully."""
        from openai import APIError

        with patch.object(
            llama_cpp_provider._client.chat.completions,
            "create",
            side_effect=APIError(
                message="API Error",
                request=MagicMock(),
                body=None,
            ),
        ):
            result = await llama_cpp_provider.classify(
                sample_base64_image,
                page_number=1,
                chunk_id="chunk_1",
            )

            assert result.has_graphics is True
            assert result.error is not None


class TestLlamaCppExtract:
    """Test llama.cpp extract method."""

    @pytest.mark.asyncio
    async def test_llama_cpp_extract_returns_extraction_result(
        self, llama_cpp_provider, sample_base64_image
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
            llama_cpp_provider._client.chat.completions,
            "create",
            return_value=mock_response,
        ):
            result = await llama_cpp_provider.extract(
                sample_base64_image,
                page_number=1,
                chunk_id="chunk_1",
            )

            assert isinstance(result, ExtractionResult)
            assert result.page_number == 1
            assert result.chunk_id == "chunk_1"
            assert result.image_text == "## Visual Elements\nChart showing data"

    @pytest.mark.asyncio
    async def test_llama_cpp_extract_strips_thought_prefix(
        self, llama_cpp_provider, sample_base64_image
    ):
        """Test that llama.cpp thought-channel prefixes are removed from output."""
        mock_message = MagicMock()
        mock_message.content = "<|channel>thought\n<channel|>## Visual Elements\nChart showing data"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.object(
            llama_cpp_provider._client.chat.completions,
            "create",
            return_value=mock_response,
        ):
            result = await llama_cpp_provider.extract(
                sample_base64_image,
                page_number=1,
                chunk_id="chunk_1",
            )

            assert result.image_text == "## Visual Elements\nChart showing data"
