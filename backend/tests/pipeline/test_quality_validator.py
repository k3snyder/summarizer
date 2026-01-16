"""
Tests for quality validation logic.
TDD red phase for bsc-3.3.
"""

import pytest


class TestQualityValidatorThreshold:
    """Test threshold calculation based on attempt number."""

    def test_relevancy_above_threshold_passes(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator()
        # At attempt 1, threshold is 90%
        assert validator.validate(relevancy=91.0, attempt=1) is True
        assert validator.validate(relevancy=90.0, attempt=1) is True

    def test_relevancy_below_threshold_fails(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator()
        # At attempt 1, threshold is 90%
        assert validator.validate(relevancy=89.0, attempt=1) is False
        assert validator.validate(relevancy=50.0, attempt=1) is False

    def test_threshold_drops_after_attempt_5(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator()

        # First 5 attempts use 90% threshold
        assert validator.get_threshold_for_attempt(1) == 90
        assert validator.get_threshold_for_attempt(5) == 90

        # Attempt 6+ uses 85% threshold
        assert validator.get_threshold_for_attempt(6) == 85
        assert validator.get_threshold_for_attempt(10) == 85
        assert validator.get_threshold_for_attempt(30) == 85

    def test_edge_case_at_threshold(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator()
        # At attempt 6, threshold is 85%
        assert validator.validate(relevancy=85.0, attempt=6) is True
        assert validator.validate(relevancy=84.9, attempt=6) is False


class TestQualityValidatorModelTier:
    """Test model tier rotation based on attempt number."""

    def test_model_tier_changes_after_attempt_10(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator()

        # Attempts 1-10: Tier 1
        assert validator.get_model_tier_for_attempt(1) == 1
        assert validator.get_model_tier_for_attempt(10) == 1

        # Attempts 11-20: Tier 2
        assert validator.get_model_tier_for_attempt(11) == 2
        assert validator.get_model_tier_for_attempt(20) == 2

    def test_model_tier_changes_after_attempt_20(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator()

        # Attempts 21-30: Tier 3
        assert validator.get_model_tier_for_attempt(21) == 3
        assert validator.get_model_tier_for_attempt(30) == 3


class TestQualityValidatorPromptKey:
    """Test prompt key selection based on attempt number."""

    def test_fallback_prompt_used_after_attempt_5(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator()

        # Attempts 1-5: primary prompt
        assert validator.get_prompt_key_for_attempt(1) == "primary"
        assert validator.get_prompt_key_for_attempt(5) == "primary"

        # Attempts 6+: fallback prompt
        assert validator.get_prompt_key_for_attempt(6) == "fallback"
        assert validator.get_prompt_key_for_attempt(30) == "fallback"


class TestQualityValidatorCustomConfig:
    """Test QualityValidator with custom thresholds."""

    def test_custom_thresholds(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator(
            threshold_high=95,
            threshold_low=88,
        )

        assert validator.get_threshold_for_attempt(1) == 95
        assert validator.get_threshold_for_attempt(6) == 88

    def test_validate_with_custom_thresholds(self):
        from app.pipeline.summarization.quality_validator import QualityValidator

        validator = QualityValidator(
            threshold_high=95,
            threshold_low=88,
        )

        # At attempt 1, threshold is 95%
        assert validator.validate(relevancy=94.0, attempt=1) is False
        assert validator.validate(relevancy=95.0, attempt=1) is True

        # At attempt 6, threshold is 88%
        assert validator.validate(relevancy=87.0, attempt=6) is False
        assert validator.validate(relevancy=88.0, attempt=6) is True
