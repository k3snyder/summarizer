"""Pipeline configuration models"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """
    Configuration for the document processing pipeline.

    Controls extraction, vision processing, and summarization stages.
    """

    extract_only: bool = Field(
        default=False,
        description="Stop after extraction stage, skip vision and summarization"
    )

    skip_tables: bool = Field(
        default=False,
        description="Skip table extraction from documents"
    )

    skip_images: bool = Field(
        default=False,
        description="Skip image extraction from documents"
    )

    skip_pptx_tables: bool = Field(
        default=False,
        description="Skip table extraction from PPTX slides (PPTX only)"
    )

    text_only: bool = Field(
        default=False,
        description="Extract only text, skip tables and images"
    )

    pdf_image_dpi: Literal[72, 144, 200, 300] = Field(
        default=200,
        description="DPI for PDF image extraction (72=fast, 300=high quality)"
    )

    vision_mode: Literal['none', 'deepseek', 'gemini', 'openai', 'ollama', 'codex', 'claude'] = Field(
        default='none',
        description="Vision processing provider (none disables vision processing)"
    )

    vision_classifier_mode: Optional[Literal['none', 'ollama', 'openai', 'gemini', 'codex', 'claude']] = Field(
        default=None,
        description="Vision classifier provider (inherits from vision_mode if not set)"
    )

    vision_extractor_mode: Optional[Literal['none', 'ollama', 'openai', 'gemini', 'codex', 'claude']] = Field(
        default=None,
        description="Vision extractor provider (inherits from vision_mode if not set)"
    )

    vision_cli_provider: Optional[Literal['codex', 'claude']] = Field(
        default=None,
        description="CLI provider for vision (codex or claude, used when vision_mode is codex/claude)"
    )

    vision_detailed_extraction: bool = Field(
        default=False,
        description="Run vision extraction 3 times per page and synthesize results for comprehensive coverage"
    )

    chunk_size: int = Field(
        default=3000,
        description="Text chunk size for summarization",
        gt=0
    )

    chunk_overlap: int = Field(
        default=80,
        description="Overlap between text chunks",
        ge=0
    )

    run_summarization: bool = Field(
        default=True,
        description="Enable summarization stage"
    )

    summarizer_mode: Literal['full', 'topics-only', 'skip'] = Field(
        default='full',
        description="Summarization mode (full=notes+topics, topics-only=fast, skip=passthrough)"
    )

    summarizer_provider: Literal['ollama', 'openai', 'codex', 'claude'] = Field(
        default='ollama',
        description="Summarizer provider (ollama=local privacy-first, openai=cloud API, codex/claude=CLI)"
    )

    summarizer_detailed_extraction: bool = Field(
        default=False,
        description="Run summarization 3 times per page and synthesize results for comprehensive coverage"
    )

    summarizer_cli_provider: Optional[Literal['codex', 'claude']] = Field(
        default=None,
        description="CLI provider for summarization (codex or claude, used when summarizer_provider is codex/claude)"
    )

    keep_base64_images: bool = Field(
        default=False,
        description="Keep base64-encoded images in final output (default: strip to reduce size)"
    )
