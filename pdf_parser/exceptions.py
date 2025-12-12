"""Custom exceptions for PDF parser."""


class PDFParserError(Exception):
    """Base exception for PDF parser errors."""
    pass


class PDFFileError(PDFParserError):
    """Raised when there's an issue with the PDF file itself."""
    pass


class PDFNotFoundError(PDFFileError):
    """Raised when the PDF file is not found."""
    pass


class PDFCorruptedError(PDFFileError):
    """Raised when the PDF file is corrupted or unreadable."""
    pass


class PageProcessingError(PDFParserError):
    """Raised when there's an error processing a specific page."""
    
    def __init__(self, message: str, page_number: int, original_error: Exception = None):
        super().__init__(message)
        self.page_number = page_number
        self.original_error = original_error


class TextExtractionError(PageProcessingError):
    """Raised when text extraction fails for a page."""
    
    def __init__(self, message: str, page_number: int, original_error: Exception = None):
        super().__init__(message, page_number, original_error)


class TableExtractionError(PageProcessingError):
    """Raised when table extraction fails for a page."""
    
    def __init__(self, message: str, page_number: int, original_error: Exception = None):
        super().__init__(message, page_number, original_error)


class ImageExtractionError(PageProcessingError):
    """Raised when image extraction fails for a page."""
    
    def __init__(self, message: str, page_number: int, original_error: Exception = None):
        super().__init__(message, page_number, original_error)


class ValidationError(PDFParserError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(PDFParserError):
    """Raised when there's an issue with configuration."""
    pass