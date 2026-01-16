"""
Tests for vision provider interface and schemas.
TDD red phase for bsc-2.1.
"""

import pytest
from dataclasses import FrozenInstanceError


class TestVisionProvider:
    """Test VisionProvider enum."""

    def test_ollama_provider_exists(self):
        from app.pipeline.vision.schemas import VisionProvider
        assert VisionProvider.OLLAMA.value == "ollama"

    def test_openai_provider_exists(self):
        from app.pipeline.vision.schemas import VisionProvider
        assert VisionProvider.OPENAI.value == "openai"

    def test_provider_enum_has_three_values(self):
        from app.pipeline.vision.schemas import VisionProvider
        assert len(VisionProvider) == 3
        assert VisionProvider.GEMINI.value == "gemini"


class TestVisionConfig:
    """Test VisionConfig dataclass."""

    def test_vision_config_default_provider(self):
        from app.pipeline.vision.schemas import VisionConfig, VisionProvider
        config = VisionConfig()
        assert config.classifier_provider == VisionProvider.OLLAMA
        assert config.extractor_provider == VisionProvider.OLLAMA

    def test_vision_config_default_model(self):
        from app.pipeline.vision.schemas import VisionConfig
        config = VisionConfig()
        assert config.classifier_model is None  # Uses provider default
        assert config.extractor_model is None  # Uses provider default

    def test_vision_config_default_batch_size(self):
        from app.pipeline.vision.schemas import VisionConfig
        config = VisionConfig()
        assert config.batch_size == 5

    def test_vision_config_custom_values(self):
        from app.pipeline.vision.schemas import VisionConfig, VisionProvider
        config = VisionConfig(
            classifier_provider=VisionProvider.OLLAMA,
            classifier_model="llava:latest",
            extractor_provider=VisionProvider.OPENAI,
            extractor_model="gpt-4o",
            batch_size=3,
            skip_classification=True,
        )
        assert config.classifier_provider == VisionProvider.OLLAMA
        assert config.classifier_model == "llava:latest"
        assert config.extractor_provider == VisionProvider.OPENAI
        assert config.extractor_model == "gpt-4o"
        assert config.batch_size == 3
        assert config.skip_classification is True

    def test_vision_config_ollama_base_url(self):
        from app.pipeline.vision.schemas import VisionConfig
        config = VisionConfig(ollama_base_url="http://localhost:11434")
        assert config.ollama_base_url == "http://localhost:11434"


class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    def test_classification_result_has_graphics(self):
        from app.pipeline.vision.schemas import ClassificationResult
        result = ClassificationResult(
            page_number=1,
            chunk_id="chunk_1",
            has_graphics=True,
        )
        assert result.page_number == 1
        assert result.chunk_id == "chunk_1"
        assert result.has_graphics is True

    def test_classification_result_no_graphics(self):
        from app.pipeline.vision.schemas import ClassificationResult
        result = ClassificationResult(
            page_number=2,
            chunk_id="chunk_2",
            has_graphics=False,
        )
        assert result.has_graphics is False

    def test_classification_result_optional_error(self):
        from app.pipeline.vision.schemas import ClassificationResult
        result = ClassificationResult(
            page_number=1,
            chunk_id="chunk_1",
            has_graphics=True,
            error=None,
        )
        assert result.error is None

        result_with_error = ClassificationResult(
            page_number=1,
            chunk_id="chunk_1",
            has_graphics=False,
            error="Connection failed",
        )
        assert result_with_error.error == "Connection failed"


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_extraction_result_with_text(self):
        from app.pipeline.vision.schemas import ExtractionResult
        result = ExtractionResult(
            page_number=1,
            chunk_id="chunk_1",
            image_text="Extracted visual content",
        )
        assert result.page_number == 1
        assert result.chunk_id == "chunk_1"
        assert result.image_text == "Extracted visual content"

    def test_extraction_result_skipped(self):
        from app.pipeline.vision.schemas import ExtractionResult
        result = ExtractionResult(
            page_number=2,
            chunk_id="chunk_2",
            image_text=None,
            skipped=True,
        )
        assert result.image_text is None
        assert result.skipped is True

    def test_extraction_result_optional_error(self):
        from app.pipeline.vision.schemas import ExtractionResult
        result = ExtractionResult(
            page_number=1,
            chunk_id="chunk_1",
            image_text=None,
            error="API error",
        )
        assert result.error == "API error"


class TestVisionProviderBase:
    """Test abstract VisionProviderBase class."""

    def test_provider_base_is_abstract(self):
        from app.pipeline.vision.providers.base import VisionProviderBase
        import abc

        # Verify VisionProviderBase is abstract
        assert abc.ABC in VisionProviderBase.__mro__

    def test_provider_base_has_classify_method(self):
        from app.pipeline.vision.providers.base import VisionProviderBase
        import inspect

        # Verify classify is an abstract method
        assert hasattr(VisionProviderBase, "classify")
        assert getattr(VisionProviderBase.classify, "__isabstractmethod__", False)

    def test_provider_base_has_extract_method(self):
        from app.pipeline.vision.providers.base import VisionProviderBase
        import inspect

        # Verify extract is an abstract method
        assert hasattr(VisionProviderBase, "extract")
        assert getattr(VisionProviderBase.extract, "__isabstractmethod__", False)

    def test_cannot_instantiate_abstract_provider(self):
        from app.pipeline.vision.providers.base import VisionProviderBase

        with pytest.raises(TypeError):
            VisionProviderBase()
