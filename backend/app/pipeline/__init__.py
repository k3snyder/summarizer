"""
Pipeline package for self-contained document processing.

This package provides a fully async pipeline for document extraction,
vision processing, and summarization without subprocess dependencies.
"""

from app.pipeline.models import PipelineStage, ProgressEvent
from app.pipeline.extraction.schemas import (
    ExtractionConfig,
    ExtractedPage,
    ExtractedDocument,
)

__all__ = [
    "PipelineStage",
    "ProgressEvent",
    "ExtractionConfig",
    "ExtractedPage",
    "ExtractedDocument",
]
