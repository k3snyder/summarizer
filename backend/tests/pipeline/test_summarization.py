"""
Tests for summarization schemas.
TDD red phase for bsc-3.1.
"""

import pytest


class TestSummarizerConfig:
    """Test SummarizerConfig dataclass."""

    def test_quality_threshold_high_default_is_90(self):
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig()
        assert config.quality_threshold_high == 90

    def test_quality_threshold_low_default_is_85(self):
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig()
        assert config.quality_threshold_low == 85

    def test_max_attempts_default_is_30(self):
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig()
        assert config.max_attempts == 30

    def test_summarizer_config_has_model_tiers(self):
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig()
        assert config.model_tier_1 is not None
        assert config.model_tier_2 is not None
        assert config.model_tier_3 is not None

    def test_summarizer_config_has_mode(self):
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig()
        assert config.mode == "full"

        config_topics = SummarizerConfig(mode="topics-only")
        assert config_topics.mode == "topics-only"

    def test_summarizer_config_custom_values(self):
        from app.pipeline.summarization.schemas import SummarizerConfig

        config = SummarizerConfig(
            quality_threshold_high=95,
            quality_threshold_low=88,
            max_attempts=20,
            mode="topics-only",
        )
        assert config.quality_threshold_high == 95
        assert config.quality_threshold_low == 88
        assert config.max_attempts == 20


class TestPageContext:
    """Test PageContext dataclass."""

    def test_page_context_required_fields(self):
        from app.pipeline.summarization.schemas import PageContext

        context = PageContext(
            page_number=1,
            chunk_id="chunk_1",
            text="Sample text content",
        )
        assert context.page_number == 1
        assert context.chunk_id == "chunk_1"
        assert context.text == "Sample text content"

    def test_page_context_optional_fields(self):
        from app.pipeline.summarization.schemas import PageContext

        context = PageContext(
            page_number=1,
            chunk_id="chunk_1",
            text="Sample text",
            tables=[],
            image_text="Visual content",
            image_classifier=True,
        )
        assert context.tables == []
        assert context.image_text == "Visual content"
        assert context.image_classifier is True

    def test_page_context_default_optional_fields(self):
        from app.pipeline.summarization.schemas import PageContext

        context = PageContext(
            page_number=1,
            chunk_id="chunk_1",
            text="Sample text",
        )
        assert context.tables is None
        assert context.image_text is None
        assert context.image_classifier is None


class TestSummaryResult:
    """Test SummaryResult dataclass."""

    def test_summary_result_with_notes_and_topics(self):
        from app.pipeline.summarization.schemas import SummaryResult

        result = SummaryResult(
            page_number=1,
            chunk_id="chunk_1",
            summary_notes=["Note 1", "Note 2"],
            summary_topics=["Topic A", "Topic B"],
            summary_relevancy=92.5,
        )
        assert result.page_number == 1
        assert result.summary_notes == ["Note 1", "Note 2"]
        assert result.summary_topics == ["Topic A", "Topic B"]
        assert result.summary_relevancy == 92.5

    def test_summary_result_with_attempts_used(self):
        from app.pipeline.summarization.schemas import SummaryResult

        result = SummaryResult(
            page_number=1,
            chunk_id="chunk_1",
            summary_notes=["Note"],
            summary_topics=["Topic"],
            summary_relevancy=90.0,
            attempts_used=3,
        )
        assert result.attempts_used == 3

    def test_summary_result_optional_notes(self):
        from app.pipeline.summarization.schemas import SummaryResult

        # Topics-only mode returns None for notes
        result = SummaryResult(
            page_number=1,
            chunk_id="chunk_1",
            summary_notes=None,
            summary_topics=["Topic"],
            summary_relevancy=0,
        )
        assert result.summary_notes is None

    def test_summary_result_with_error(self):
        from app.pipeline.summarization.schemas import SummaryResult

        result = SummaryResult(
            page_number=1,
            chunk_id="chunk_1",
            summary_notes=None,
            summary_topics=None,
            summary_relevancy=0,
            error="Failed to summarize",
        )
        assert result.error == "Failed to summarize"
