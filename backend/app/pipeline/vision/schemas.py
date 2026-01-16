"""
Schemas for vision processing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VisionProvider(Enum):
    """Supported vision providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    CODEX_CLI = "codex"


@dataclass
class VisionConfig:
    """Configuration for vision processing with separate classifier and extractor."""

    # Classifier configuration
    classifier_provider: VisionProvider = VisionProvider.OLLAMA
    classifier_model: Optional[str] = None

    # Extractor configuration
    extractor_provider: VisionProvider = VisionProvider.OLLAMA
    extractor_model: Optional[str] = None

    # Shared settings
    batch_size: int = 5
    extractor_batch_size: Optional[int] = None  # If None, uses batch_size. Set to 1 for OpenAI.
    skip_classification: bool = False
    detailed_extraction: bool = False  # Run extraction 3x and synthesize

    # CLI provider selection (for CODEX_CLI provider: 'codex' or 'claude')
    cli_provider: str = "codex"

    # Provider credentials (shared across both)
    ollama_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None


@dataclass
class ClassificationResult:
    """Result from vision classification."""

    page_number: int
    chunk_id: str
    has_graphics: bool
    error: Optional[str] = None

    # Token usage metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ExtractionResult:
    """Result from vision extraction."""

    page_number: int
    chunk_id: str
    image_text: Optional[str] = None
    skipped: bool = False
    error: Optional[str] = None

    # Detailed extraction intermediate results (when detailed_extraction=True)
    image_text_1: Optional[str] = None
    image_text_2: Optional[str] = None
    image_text_3: Optional[str] = None

    # Token usage metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
