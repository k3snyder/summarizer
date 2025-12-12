"""
Text Parser Module

Converts text and markdown files into unified JSON schema compatible with PDF parser output.
Each text chunk becomes equivalent to a PDF page for unified downstream processing.
"""

from .main import TextParser, process_text_file

__all__ = ['TextParser', 'process_text_file']
