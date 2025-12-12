#!/usr/bin/env python3
"""
Text File Parser - Converts text/markdown files to unified JSON schema.

Produces the same output schema as pdf_parser to enable unified processing.
Each text chunk becomes equivalent to a PDF page.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


class TextParser:
    """Parse text files and chunk them with unified JSON output."""

    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 80):
        """
        Initialize text parser.

        Args:
            chunk_size: Maximum characters per chunk (default: 3000)
            chunk_overlap: Character overlap between chunks (default: 80)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """
        Parse text file and return unified JSON structure.

        Args:
            filepath: Path to .txt or .md file

        Returns:
            Dict matching pdf_parser output schema with pages array
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Text file not found: {filepath}")

        # Validate file type
        if filepath.suffix.lower() not in ['.txt', '.md']:
            raise ValueError(f"Unsupported file type: {filepath.suffix}. Use .txt or .md")

        logger.info(f"Parsing text file: {filepath.name}")

        # Read file content
        with open(filepath, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Split into chunks
        chunks = self.text_splitter.split_text(text_content)
        total_chunks = len(chunks)

        logger.info(f"Split {filepath.name} into {total_chunks} chunks")

        # Build document metadata
        document_id = f"doc_{filepath.stem}"
        filename = filepath.name

        # Create pages array (chunks as pages)
        pages = []
        for idx, chunk_text in enumerate(chunks, start=1):
            page = {
                "chunk_id": f"chunk_{idx}",
                "doc_title": filename,
                "text": chunk_text,
                "tables": [],  # Text files don't have tables
                "image_base64": []  # Text files don't have images
            }
            pages.append(page)

        # Build unified JSON structure (matches pdf_parser output)
        output = {
            "document": {
                "document_id": document_id,
                "filename": filename,
                "total_pages": total_chunks,
                "metadata": {
                    "source_type": "text",
                    "file_type": filepath.suffix,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
            },
            "pages": pages
        }

        return output

    def save_output(self, output: Dict[str, Any], output_path: str = "./output/output_parsed.json"):
        """
        Save parsed output to JSON file.

        Args:
            output: Parsed document structure
            output_path: Path to save JSON file
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved parsed output to {output_path}")


def process_text_file(
    filepath: str,
    output_path: str = "./output/output_parsed.json",
    chunk_size: int = 3000,
    chunk_overlap: int = 80
) -> Dict[str, Any]:
    """
    Process a text file and save to unified JSON format.

    Args:
        filepath: Path to .txt or .md file
        output_path: Where to save parsed output
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks

    Returns:
        Parsed document structure
    """
    parser = TextParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    output = parser.parse_file(filepath)
    parser.save_output(output, output_path)

    return output


def main():
    """CLI entry point for text parser."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Text File Parser - Convert text/markdown to unified JSON schema"
    )
    parser.add_argument(
        "filepath",
        help="Path to .txt or .md file"
    )
    parser.add_argument(
        "--output",
        default="./output/output_parsed.json",
        help="Output JSON path (default: ./output/output_parsed.json)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3000,
        help="Maximum characters per chunk (default: 3000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help="Character overlap between chunks (default: 80)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        output = process_text_file(
            args.filepath,
            output_path=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        total_chunks = output['document']['total_pages']
        print(f"\n✅ Successfully parsed {args.filepath}")
        print(f"   Chunks created: {total_chunks}")
        print(f"   Output saved to: {args.output}")

    except Exception as e:
        print(f"\n❌ Error parsing text file: {e}")
        logger.error(f"Failed to parse text file: {e}")
        exit(1)


if __name__ == "__main__":
    main()
