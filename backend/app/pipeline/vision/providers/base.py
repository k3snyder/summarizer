"""
Abstract base class for vision providers.
"""

from abc import ABC, abstractmethod

from app.pipeline.vision.schemas import ClassificationResult, ExtractionResult


class VisionProviderBase(ABC):
    """Abstract base class for vision providers.

    All vision providers must implement classify and extract methods.
    """

    @abstractmethod
    async def classify(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ClassificationResult:
        """Classify whether an image contains graphical content.

        Args:
            image_base64: Base64-encoded image data
            page_number: Page number in document
            chunk_id: Unique chunk identifier

        Returns:
            ClassificationResult with has_graphics boolean
        """
        pass

    @abstractmethod
    async def extract(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ExtractionResult:
        """Extract visual content from an image.

        Args:
            image_base64: Base64-encoded image data
            page_number: Page number in document
            chunk_id: Unique chunk identifier

        Returns:
            ExtractionResult with extracted image_text
        """
        pass

    async def synthesize(
        self,
        extraction_1: str,
        extraction_2: str,  # noqa: ARG002 - used by subclasses
        extraction_3: str,  # noqa: ARG002 - used by subclasses
        page_number: int,
        chunk_id: str,
    ) -> ExtractionResult:
        """Synthesize 3 extractions into a comprehensive result.

        Default implementation returns the first extraction.
        Subclasses should override to provide actual synthesis.

        Args:
            extraction_1: First extraction text
            extraction_2: Second extraction text
            extraction_3: Third extraction text
            page_number: Page number in document
            chunk_id: Unique chunk identifier

        Returns:
            ExtractionResult with synthesized image_text
        """
        # Default: just return first extraction (extraction_2, extraction_3 used by subclasses)
        return ExtractionResult(
            page_number=page_number,
            chunk_id=chunk_id,
            image_text=extraction_1,
        )
