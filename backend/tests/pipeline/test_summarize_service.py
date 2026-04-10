"""
Tests for SummarizeService.
TDD red phase for bsc-3.5.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def sample_page_context():
    """Sample page context for testing."""
    from app.pipeline.summarization.schemas import PageContext

    return PageContext(
        page_number=1,
        chunk_id="chunk_1",
        text="This is sample text content for summarization.",
        tables=None,
        image_text=None,
    )


@pytest.fixture
def sample_page_contexts():
    """Multiple page contexts for batch testing."""
    from app.pipeline.summarization.schemas import PageContext

    return [
        PageContext(page_number=1, chunk_id="chunk_1", text="Text for page 1"),
        PageContext(page_number=2, chunk_id="chunk_2", text="Text for page 2"),
        PageContext(page_number=3, chunk_id="chunk_3", text="Text for page 3"),
    ]


def make_llm_response(content: str):
    """Create a mock LLMResponse with the given content."""
    from app.pipeline.summarization.summarizer import LLMResponse
    return LLMResponse(content=content, prompt_tokens=10, completion_tokens=5, total_tokens=15)


class TestSummarizePageBasic:
    """Test basic summarize_page functionality."""

    @pytest.mark.asyncio
    async def test_summarize_page_returns_summary_result(self, sample_page_context):
        """Test that summarize_page returns a SummaryResult."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig, SummaryResult

        config = SummarizerConfig()

        # Mock the LLM calls
        with patch.object(
            SummarizeService, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = [
                make_llm_response("* Key point 1\n* Key point 2"),  # Notes response
                make_llm_response("95"),  # Quality check
                make_llm_response("Topic A, Topic B"),  # Topics response
            ]

            service = SummarizeService(config)
            result = await service.summarize_page(sample_page_context)

            assert isinstance(result, SummaryResult)
            assert result.page_number == 1
            assert result.chunk_id == "chunk_1"
            assert result.summary_notes is not None
            assert result.summary_topics is not None


class TestSummarizePageRetry:
    """Test retry behavior in summarize_page."""

    @pytest.mark.asyncio
    async def test_summarize_page_retries_on_low_quality(self, sample_page_context):
        """Test that low quality triggers retry."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig()

        with patch.object(
            SummarizeService, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = [
                make_llm_response("* Bad summary"),  # First attempt notes
                make_llm_response("50"),  # Low quality score (fails)
                make_llm_response("* Better summary"),  # Second attempt notes
                make_llm_response("92"),  # Good quality score (passes)
                make_llm_response("Topic A"),  # Topics
            ]

            service = SummarizeService(config)
            result = await service.summarize_page(sample_page_context)

            # Should have retried (at least 2 notes calls + quality checks)
            assert result.attempts_used >= 2

    @pytest.mark.asyncio
    async def test_summarize_page_respects_30_attempt_limit(self, sample_page_context):
        """Test that max 30 attempts is respected."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig(max_attempts=5)  # Use smaller limit for test

        with patch.object(
            SummarizeService, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            # Always return low quality to exhaust attempts
            mock_llm.side_effect = [
                make_llm_response("* Summary"),
                make_llm_response("50"),
            ] * 10  # More than max_attempts

            service = SummarizeService(config)
            result = await service.summarize_page(sample_page_context)

            # Should stop at max_attempts
            assert result.attempts_used <= config.max_attempts


class TestSummarizePageBatch:
    """Test batch processing in summarize_pages_batch."""

    @pytest.mark.asyncio
    async def test_summarize_pages_batch_processes_concurrently(
        self, sample_page_contexts
    ):
        """Test that batch processing works for multiple pages."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig, SummaryResult

        config = SummarizerConfig(batch_size=2)

        with patch.object(
            SummarizeService, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = [
                make_llm_response("* Note 1"),
                make_llm_response("95"),
                make_llm_response("Topic 1"),
                make_llm_response("* Note 2"),
                make_llm_response("95"),
                make_llm_response("Topic 2"),
                make_llm_response("* Note 3"),
                make_llm_response("95"),
                make_llm_response("Topic 3"),
            ]

            service = SummarizeService(config)
            results = await service.summarize_pages_batch(sample_page_contexts)

            assert len(results) == 3
            assert all(isinstance(r, SummaryResult) for r in results)

    @pytest.mark.asyncio
    async def test_summarize_pages_batch_calls_progress_callback(
        self, sample_page_contexts
    ):
        """Test that progress callback is called during batch processing."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig()
        progress_calls = []

        async def progress_callback(current: int, total: int):
            progress_calls.append((current, total))

        with patch.object(
            SummarizeService, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = [
                make_llm_response("* Note"),
                make_llm_response("95"),
                make_llm_response("Topic"),
            ] * 3

            service = SummarizeService(config)
            await service.summarize_pages_batch(
                sample_page_contexts, progress_callback=progress_callback
            )

            assert len(progress_calls) == 3
            assert progress_calls[-1] == (3, 3)


class TestTopicsOnlyMode:
    """Test topics-only mode."""

    @pytest.mark.asyncio
    async def test_topics_only_mode_skips_notes(self, sample_page_context):
        """Test that topics-only mode skips note generation."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig(mode="topics-only")

        with patch.object(
            SummarizeService, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = make_llm_response("Topic A, Topic B")

            service = SummarizeService(config)
            result = await service.summarize_page(sample_page_context)

            # Notes should be None in topics-only mode
            assert result.summary_notes is None
            assert result.summary_topics is not None


class TestSkipMode:
    """Test skip mode."""

    @pytest.mark.asyncio
    async def test_skip_mode_returns_empty_result(self, sample_page_context):
        """Test that skip mode returns empty result without LLM calls."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig(mode="skip")

        with patch.object(
            SummarizeService, "_call_llm", new_callable=AsyncMock
        ) as mock_llm:
            service = SummarizeService(config)
            result = await service.summarize_page(sample_page_context)

            # No LLM calls should be made
            mock_llm.assert_not_called()

            assert result.summary_notes is None
            assert result.summary_topics is None
            assert result.summary_relevancy == 0


class TestLlamaCppProvider:
    """Test llama.cpp-specific summarizer behavior."""

    def test_llama_cpp_provider_uses_gpu0_text_endpoint(self):
        """Test that llama.cpp summarizer uses the configured text endpoint."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig(
            provider="llama_cpp",
            llama_cpp_base_url_1="http://localhost:11440/v1",
            llama_cpp_base_url_2="http://localhost:11440/v1",
            llama_cpp_api_key="sk-test",
        )

        service = SummarizeService(config)

        assert str(service._client_1.base_url) == "http://localhost:11440/v1/"
        assert str(service._client_2.base_url) == "http://localhost:11440/v1/"

    @pytest.mark.asyncio
    async def test_llama_cpp_provider_disables_reasoning_and_strips_thought_prefix(
        self, sample_page_context
    ):
        """Test that llama.cpp calls disable reasoning and normalize response text."""
        from app.pipeline.summarization.summarizer import SummarizeService
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig(provider="llama_cpp")
        service = SummarizeService(config)

        mock_message = MagicMock()
        mock_message.content = "<|channel>thought\n<channel|>* Key point 1\n* Key point 2"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        with patch.object(
            service._client_1.chat.completions,
            "create",
            return_value=mock_response,
        ) as mock_create:
            response = await service._call_llm("Summarize this", tier=1)

            assert response.content == "* Key point 1\n* Key point 2"
            assert mock_create.call_args.kwargs["extra_body"]["reasoning"] == "off"
            assert mock_create.call_args.kwargs["extra_body"]["reasoning_budget"] == 0
            assert mock_create.call_args.kwargs["extra_body"]["reasoning_in_content"] is False
            assert mock_create.call_args.kwargs["extra_body"]["reasoning_format"] == "none"
            assert mock_create.call_args.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
