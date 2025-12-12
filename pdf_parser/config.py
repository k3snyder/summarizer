"""Configuration management for PDF parser."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ParsingConfig:
    """Configuration for PDF parsing operations."""
    
    # Image settings
    image_dpi: int = 144
    ocr_resolution: int = 150
    image_format: str = "JPEG"
    
    # Table detection settings
    table_y_tolerance: int = 3
    table_bottom_threshold: int = 75
    table_height_threshold_ratio: float = 0.2
    table_bottom_margin: int = 100
    
    # Processing settings
    max_concurrent_pages: int = 4
    enable_ocr_fallback: bool = True
    enable_camelot_fallback: bool = True
    
    # Output settings
    output_dir: str = "output"
    output_filename: str = "output_parsed.json"
    json_indent: int = 2
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def from_env(cls) -> 'ParsingConfig':
        """Create configuration from environment variables."""
        return cls(
            image_dpi=int(os.getenv('PDF_IMAGE_DPI', cls.image_dpi)),
            ocr_resolution=int(os.getenv('PDF_OCR_RESOLUTION', cls.ocr_resolution)),
            image_format=os.getenv('PDF_IMAGE_FORMAT', cls.image_format),
            table_y_tolerance=int(os.getenv('PDF_TABLE_Y_TOLERANCE', cls.table_y_tolerance)),
            table_bottom_threshold=int(os.getenv('PDF_TABLE_BOTTOM_THRESHOLD', cls.table_bottom_threshold)),
            table_height_threshold_ratio=float(os.getenv('PDF_TABLE_HEIGHT_RATIO', cls.table_height_threshold_ratio)),
            table_bottom_margin=int(os.getenv('PDF_TABLE_BOTTOM_MARGIN', cls.table_bottom_margin)),
            max_concurrent_pages=int(os.getenv('PDF_MAX_CONCURRENT_PAGES', cls.max_concurrent_pages)),
            enable_ocr_fallback=os.getenv('PDF_ENABLE_OCR_FALLBACK', 'true').lower() == 'true',
            enable_camelot_fallback=os.getenv('PDF_ENABLE_CAMELOT_FALLBACK', 'true').lower() == 'true',
            output_dir=os.getenv('PDF_OUTPUT_DIR', cls.output_dir),
            output_filename=os.getenv('PDF_OUTPUT_FILENAME', cls.output_filename),
            json_indent=int(os.getenv('PDF_JSON_INDENT', cls.json_indent)),
            log_level=os.getenv('PDF_LOG_LEVEL', cls.log_level),
            log_format=os.getenv('PDF_LOG_FORMAT', cls.log_format),
        )
    
    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """Get the full output path for parsed results."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir / (filename or self.output_filename)


# Global configuration instance
config = ParsingConfig.from_env()