"""
Schemas for document extraction.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExtractionConfig:
    """Configuration for document extraction."""

    skip_tables: bool = False
    skip_images: bool = False
    skip_pptx_tables: bool = False
    text_only: bool = False
    pdf_image_dpi: int = 200
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class ExtractedPage:
    """Individual page data from extraction."""

    page_number: int
    chunk_id: str
    doc_title: str
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    image_base64: Optional[str] = None
    image_text: Optional[str] = None
    image_classifier: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching output schema."""
        return {
            "chunk_id": self.chunk_id,
            "doc_title": self.doc_title,
            "text": self.text,
            "tables": self.tables,
            "image_base64": self.image_base64,
            "image_text": self.image_text,
            "image_classifier": self.image_classifier,
        }


@dataclass
class ExtractedDocument:
    """Complete document with extracted pages."""

    document_id: str
    filename: str
    total_pages: int
    pages: list[ExtractedPage]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching pipeline output schema."""
        return {
            "document": {
                "document_id": self.document_id,
                "filename": self.filename,
                "total_pages": self.total_pages,
                "metadata": self.metadata,
            },
            "pages": [page.to_dict() for page in self.pages],
        }
