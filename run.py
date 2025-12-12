#!/usr/bin/env python3
"""
Interactive PDF Processing Pipeline with Enhanced UX.

Provides a user-friendly console interface for configuring and running
the 2-step PDF processing pipeline with intelligent defaults and validation.
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
import subprocess
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    # Input
    input_file: Optional[str] = None
    batch_mode: bool = False
    file_type: Optional[str] = None  # 'pdf', 'text', or 'markdown'

    # Extraction options (only for PDFs)
    extract_only: bool = False
    skip_tables: bool = False
    skip_images: bool = False
    text_only: bool = False
    pdf_image_dpi: int = 144  # PDF image resolution (72=low, 144=medium, 200+=high)

    # Vision processing options (only for PDFs with images)
    vision_mode: str = 'none'  # 'none', 'deepseek', 'gemini', 'openai', or 'ollama'

    # Text chunking options
    chunk_size: int = 3000
    chunk_overlap: int = 80

    # Summarization options
    run_summarization: bool = True
    summarizer_mode: str = 'full'  # 'full', 'topics-only', or 'skip'

    def __post_init__(self):
        """Normalize configuration after initialization."""
        # Auto-detect file type if not set
        if self.input_file and not self.file_type:
            ext = Path(self.input_file).suffix.lower()
            if ext == '.pdf':
                self.file_type = 'pdf'
            elif ext == '.txt':
                self.file_type = 'text'
            elif ext == '.md':
                self.file_type = 'markdown'

        # text_only is a shortcut for both skip flags
        if self.text_only:
            self.skip_tables = True
            self.skip_images = True

        # Vision processing requires images to be extracted
        if self.skip_images:
            self.vision_mode = 'none'

        # extract_only means don't run summarization
        if self.extract_only:
            self.run_summarization = False
            self.summarizer_mode = 'skip'

    def is_text_file(self) -> bool:
        """Check if input is a text or markdown file."""
        return self.file_type in ('text', 'markdown')

    def summary(self) -> str:
        """Generate human-readable configuration summary."""
        lines = [
            "\n" + "="*60,
            "📋 PIPELINE CONFIGURATION SUMMARY",
            "="*60,
        ]

        # Input mode
        if self.batch_mode:
            lines.append("📂 Mode: Batch processing (all files in ./input)")
        else:
            file_type_emoji = "📄" if self.is_text_file() else "📕"
            file_type_name = self.file_type.upper() if self.file_type else "Unknown"
            lines.append(f"{file_type_emoji} Mode: Single {file_type_name} file ({self.input_file})")

        # Pipeline stages
        lines.append("\n🔧 Processing Stages:")

        if self.is_text_file():
            # Text file processing
            lines.append("  ✓ Step 1: Text Chunking")
            lines.append(f"    └─ Chunk size: {self.chunk_size} chars, Overlap: {self.chunk_overlap} chars")
        else:
            # PDF processing
            lines.append("  ✓ Step 1: PDF Extraction")

            # Extraction options
            extraction_features = []
            if not self.skip_tables:
                extraction_features.append("Tables")
            if not self.skip_images:
                extraction_features.append("Images")
            extraction_features.append("Text")

            lines.append(f"    └─ Extracting: {', '.join(extraction_features)}")

            # Vision processing step (only for PDFs with images)
            if self.vision_mode == 'deepseek':
                lines.append("  ✓ Step 2: Vision OCR (DeepSeek)")
                lines.append("    └─ Extract text from page images via self-hosted server")
            elif self.vision_mode == 'gemini':
                lines.append("  ✓ Step 2: Vision AI (Gemini)")
                lines.append("    └─ Classify pages → Extract visual content (charts, graphs, diagrams)")
            elif self.vision_mode == 'openai':
                lines.append("  ✓ Step 2: Vision AI (OpenAI)")
                lines.append("    └─ Classify pages → Extract visual content (charts, graphs, diagrams)")
            elif self.vision_mode == 'ollama':
                lines.append("  ✓ Step 2: Vision AI (Ollama)")
                lines.append("    └─ Classify pages → Extract visual content (local Ollama, zero cost)")

        # Summarization step
        step_num = 3 if (not self.is_text_file() and self.vision_mode != 'none') else 2
        if self.run_summarization:
            if self.summarizer_mode == 'full':
                lines.append(f"  ✓ Step {step_num}: AI Summarization (Full: notes + topics)")
                lines.append("    └─ Quality threshold: 85-90%")
            elif self.summarizer_mode == 'topics-only':
                lines.append(f"  ✓ Step {step_num}: AI Summarization (Topics only)")
                lines.append("    └─ Faster processing, no quality checks")
            else:  # skip
                lines.append(f"  ✗ Step {step_num}: Summarization (SKIPPED)")
        else:
            lines.append(f"  ✗ Step {step_num}: Summarization (SKIPPED)")

        # Output
        lines.append("\n📤 Output:")
        lines.append("  • Parsed content: ./output/output_parsed.json")
        if self.run_summarization:
            filename_base = Path(self.input_file).stem if self.input_file else "output"
            lines.append(f"  • Summaries: ./output/{filename_base}_enriched.json")

        lines.append("="*60 + "\n")

        return "\n".join(lines)


class InteractivePrompt:
    """Handle interactive console prompts with validation."""

    @staticmethod
    def clear_screen():
        """Clear console screen (optional, can be disabled)."""
        # Uncomment if you want screen clearing
        # os.system('cls' if os.name == 'nt' else 'clear')
        pass

    @staticmethod
    def print_header():
        """Print application header."""
        print("\n" + "="*60)
        print("🤖 DOCUMENT INTELLIGENCE PIPELINE")
        print("="*60)
        print("Privacy-first AI-powered document processing")
        print("Supports: PDF, TXT, MD")
        print("="*60 + "\n")

    @staticmethod
    def yes_no_prompt(question: str, default: bool = True) -> bool:
        """
        Prompt for yes/no answer with default.

        Args:
            question: Question to ask
            default: Default answer if user just presses Enter

        Returns:
            bool: True for yes, False for no
        """
        default_str = "Y/n" if default else "y/N"
        while True:
            response = input(f"{question} [{default_str}]: ").strip().lower()

            if response == '':
                return default
            elif response in ('y', 'yes'):
                return True
            elif response in ('n', 'no'):
                return False
            else:
                print("❌ Please enter 'y' or 'n' (or press Enter for default)")

    @staticmethod
    def select_file() -> Optional[str]:
        """
        Prompt user to select a document file (PDF, TXT, or MD).

        Returns:
            Path to selected file or None
        """
        print("\n📁 FILE SELECTION")
        print("-" * 40)

        # Check for supported files in input directory
        input_dir = Path('./input')
        pdf_files = sorted(input_dir.glob('*.pdf'))
        txt_files = sorted(input_dir.glob('*.txt'))
        md_files = sorted(input_dir.glob('*.md'))

        all_files = pdf_files + txt_files + md_files

        if all_files:
            print(f"\nFound {len(all_files)} file(s) in ./input directory:")
            for i, file in enumerate(all_files, 1):
                size_mb = file.stat().st_size / (1024 * 1024)
                file_type_emoji = "📕" if file.suffix == '.pdf' else "📄"
                print(f"  {i}. {file_type_emoji} {file.name} ({size_mb:.2f} MB)")
            print(f"  {len(all_files) + 1}. Enter custom path")
            print(f"  {len(all_files) + 2}. Batch mode (process all)")

            while True:
                choice = input(f"\nSelect option [1-{len(all_files) + 2}]: ").strip()

                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(all_files):
                        selected = str(all_files[choice_num - 1])
                        print(f"✓ Selected: {selected}")
                        return selected
                    elif choice_num == len(all_files) + 1:
                        break  # Continue to custom path
                    elif choice_num == len(all_files) + 2:
                        return "BATCH_MODE"
                    else:
                        print(f"❌ Please enter a number between 1 and {len(all_files) + 2}")
                except ValueError:
                    print("❌ Please enter a valid number")

        # Custom path input
        while True:
            path = input("\n📄 Enter path to file (PDF/TXT/MD) or 'q' to quit: ").strip()

            if path.lower() == 'q':
                return None

            if os.path.exists(path):
                ext = Path(path).suffix.lower()
                if ext in ['.pdf', '.txt', '.md']:
                    print(f"✓ Valid file: {path}")
                    return path
                else:
                    print("❌ Unsupported file type. Use .pdf, .txt, or .md")
            else:
                print("❌ File not found. Please try again.")

    @staticmethod
    def configure_extraction() -> Dict[str, Any]:
        """
        Prompt for extraction configuration.

        Returns:
            Dict with skip_tables, skip_images, and pdf_image_dpi
        """
        print("\n⚙️  EXTRACTION CONFIGURATION")
        print("-" * 40)
        print("Configure what content to extract from the PDF.")
        print()

        # Quick presets
        print("Quick presets:")
        print("  1. Full extraction (text + tables + images) - Recommended")
        print("  2. Text and images only (skip tables)")
        print("  3. Text only (fastest, skip tables and images)")
        print("  4. Custom configuration")

        skip_tables = False
        skip_images = False

        while True:
            choice = input("\nSelect preset [1-4] or press Enter for Full: ").strip()

            if choice == '' or choice == '1':
                print("✓ Selected: Full extraction")
                skip_tables, skip_images = False, False
                break
            elif choice == '2':
                print("✓ Selected: Text and images only")
                skip_tables, skip_images = True, False
                break
            elif choice == '3':
                print("✓ Selected: Text only")
                skip_tables, skip_images = True, True
                break
            elif choice == '4':
                # Custom configuration
                print("\n📝 Custom Configuration:")

                extract_tables = InteractivePrompt.yes_no_prompt(
                    "  Extract tables?",
                    default=True
                )

                extract_images = InteractivePrompt.yes_no_prompt(
                    "  Extract images?",
                    default=True
                )

                skip_tables = not extract_tables
                skip_images = not extract_images
                break
            else:
                print("❌ Please enter 1, 2, 3, or 4")

        # PDF Image Resolution (only if extracting images)
        pdf_image_dpi = 144  # default
        if not skip_images:
            print("\n📸 PDF IMAGE RESOLUTION")
            print("-" * 40)
            print("Choose PDF image extraction quality:")
            print("  1. Low (72 DPI) - Smallest files, basic quality")
            print("  2. Medium (144 DPI) - Balanced (Recommended)")
            print("  3. High (200 DPI) - Better quality, larger files")
            print("  4. Very High (300 DPI) - Best quality, largest files")
            print()
            print("Note: Vision models work best with higher resolution")
            print("      (vision_rag_ollama will optimize to 2048px)")
            print()

            while True:
                dpi_choice = input("Select resolution [1-4] or press Enter for Medium: ").strip()

                if dpi_choice == '' or dpi_choice == '2':
                    pdf_image_dpi = 144
                    print("✓ Selected: Medium (144 DPI)")
                    break
                elif dpi_choice == '1':
                    pdf_image_dpi = 72
                    print("✓ Selected: Low (72 DPI) - will be upscaled by vision processing")
                    break
                elif dpi_choice == '3':
                    pdf_image_dpi = 200
                    print("✓ Selected: High (200 DPI) - recommended for charts/graphs")
                    break
                elif dpi_choice == '4':
                    pdf_image_dpi = 300
                    print("✓ Selected: Very High (300 DPI) - best quality")
                    break
                else:
                    print("❌ Please enter 1, 2, 3, or 4")

        return {
            "skip_tables": skip_tables,
            "skip_images": skip_images,
            "pdf_image_dpi": pdf_image_dpi
        }

    @staticmethod
    def configure_vision_processing() -> str:
        """
        Prompt for Vision processing configuration.

        Returns:
            str: Vision mode - 'none', 'deepseek', 'gemini', 'openai', or 'ollama'
        """
        print("\n👁️  VISION PROCESSING (Optional)")
        print("-" * 40)
        print("Extract visual content from page images using AI:")
        print()
        print("Vision processing options:")
        print("  1. None - Skip vision processing")
        print("  2. DeepSeek OCR - Extract text from images (self-hosted, zero cost)")
        print("  3. Gemini Vision - Extract visual insights: charts, graphs, diagrams (API cost)")
        print("  4. OpenAI Vision - Extract visual insights with OpenAI (API cost)")
        print("  5. Ollama Vision - Extract visual insights with Ollama (local, zero cost)")
        print()
        print("Note: Gemini, OpenAI, and Ollama automatically filter pages to process only visual content.")
        print()

        while True:
            choice = input("Select option [1-5] or press Enter to skip: ").strip()

            if choice == '' or choice == '1':
                print("✓ Skipping vision processing")
                return 'none'
            elif choice == '2':
                print("✓ Selected: DeepSeek OCR (requires server at 192.168.10.3:8000)")
                return 'deepseek'
            elif choice == '3':
                print("✓ Selected: Gemini Vision (classifier → visual content extraction)")
                return 'gemini'
            elif choice == '4':
                print("✓ Selected: OpenAI Vision (classifier → visual content extraction)")
                return 'openai'
            elif choice == '5':
                print("✓ Selected: Ollama Vision (classifier → visual content extraction, local Ollama)")
                return 'ollama'
            else:
                print("❌ Please enter 1, 2, 3, 4, or 5")

    @staticmethod
    def configure_pipeline() -> Dict[str, Any]:
        """
        Prompt for pipeline stage configuration.

        Returns:
            Dict with extract_only, run_summarization, and summarizer_mode
        """
        print("\n🔄 PIPELINE STAGES")
        print("-" * 40)

        run_both = InteractivePrompt.yes_no_prompt(
            "Run both extraction AND summarization?",
            default=True
        )

        if not run_both:
            print("✓ Will run: Extraction only (faster)")
            return {
                "extract_only": True,
                "run_summarization": False,
                "summarizer_mode": "skip"
            }

        # Configure summarization mode
        print("\n📝 SUMMARIZATION MODE")
        print("-" * 40)
        print("Choose summarization level:")
        print("  1. Full (notes + topics) - Best quality, slower")
        print("  2. Topics only - Faster, no quality checks")
        print()

        while True:
            choice = input("Select mode [1-2] or press Enter for Full: ").strip()

            if choice == '' or choice == '1':
                print("✓ Selected: Full summarization (notes + topics)")
                return {
                    "extract_only": False,
                    "run_summarization": True,
                    "summarizer_mode": "full"
                }
            elif choice == '2':
                print("✓ Selected: Topics only (faster)")
                return {
                    "extract_only": False,
                    "run_summarization": True,
                    "summarizer_mode": "topics-only"
                }
            else:
                print("❌ Please enter 1 or 2")


def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    logger.debug("Verified directory structure")


def run_pdf_parser(pdf_path: str, skip_tables: bool = False, skip_images: bool = False, pdf_image_dpi: int = 144) -> bool:
    """
    Run PDF parser on a single PDF file.

    Args:
        pdf_path: Path to the PDF file
        skip_tables: If True, skip table extraction
        skip_images: If True, skip image extraction
        pdf_image_dpi: PDF image resolution DPI (72=low, 144=medium, 200+=high)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"STEP 1: Running PDF parser on {pdf_path}")
        if skip_tables:
            logger.info("Skipping table extraction")
        if skip_images:
            logger.info("Skipping image extraction")
        else:
            logger.info(f"PDF image resolution: {pdf_image_dpi} DPI")

        # Import and run PDF parser
        sys.path.insert(0, os.path.dirname(__file__))
        from pdf_parser.main import process_pdf

        # Run the PDF parser
        asyncio.run(process_pdf(pdf_path, skip_tables=skip_tables, skip_images=skip_images, image_dpi=pdf_image_dpi))

        # Check if output was created
        output_path = './output/output_parsed.json'
        if os.path.exists(output_path):
            logger.info(f"PDF parsing complete. Output: {output_path}")
            return True
        else:
            logger.error("PDF parsing failed - no output file created")
            return False

    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        return False


def run_text_parser(text_path: str, chunk_size: int = 3000, chunk_overlap: int = 80) -> bool:
    """
    Run text parser on a single text/markdown file.

    Args:
        text_path: Path to the text file (.txt or .md)
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"STEP 1: Running text parser on {text_path}")
        logger.info(f"Chunk settings: size={chunk_size}, overlap={chunk_overlap}")

        # Import and run text parser
        sys.path.insert(0, os.path.dirname(__file__))
        from text_parser.main import process_text_file

        # Run the text parser
        process_text_file(
            text_path,
            output_path='./output/output_parsed.json',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Check if output was created
        output_path = './output/output_parsed.json'
        if os.path.exists(output_path):
            logger.info(f"Text parsing complete. Output: {output_path}")
            return True
        else:
            logger.error("Text parsing failed - no output file created")
            return False

    except Exception as e:
        logger.error(f"Text parsing failed: {e}")
        return False


def run_vision_ocr_deepseek(json_input: str = './output/output_parsed.json') -> bool:
    """
    Run Vision OCR (DeepSeek) to extract text from page images.

    Args:
        json_input: Path to parsed JSON with image_base64 data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("STEP 2: Running Vision OCR (DeepSeek)")

        # Run vision_ocr_deepseek as subprocess
        cmd = [sys.executable, '-u', 'vision_ocr_deepseek.py', '--input', json_input]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        returncode = process.wait()

        if returncode == 0:
            logger.info("Vision OCR complete")
            return True
        else:
            logger.error(f"Vision OCR failed with return code {returncode}")
            return False

    except Exception as e:
        logger.error(f"Vision OCR failed: {e}")
        return False


def run_vision_classifier_gemini(json_input: str = './output/output_parsed.json') -> bool:
    """
    Run Vision Classifier (Gemini) to identify pages with visual content.

    This pre-filters pages to identify which ones contain meaningful visual
    elements (charts, graphs, diagrams) worth processing with Vision RAG.
    Saves API costs by skipping text-only pages.

    Args:
        json_input: Path to parsed JSON with image_base64 data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("STEP 2a: Running Vision Classifier (Gemini)")
        logger.info("Identifying pages with visual content (charts, graphs, diagrams)...")

        # Run vision_classifier_gemini as subprocess
        cmd = [sys.executable, '-u', 'vision_classifier_gemini.py', '--input', json_input]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        returncode = process.wait()

        if returncode == 0:
            logger.info("Vision classification complete")
            return True
        else:
            logger.error(f"Vision classification failed with return code {returncode}")
            return False

    except Exception as e:
        logger.error(f"Vision classification failed: {e}")
        return False


def run_vision_rag_gemini(json_input: str = './output/output_parsed.json') -> bool:
    """
    Run Vision RAG (Gemini) to extract visual content from classified pages.

    This runs only on pages that were identified by the classifier as containing
    meaningful visual content. Extracts insights from charts, graphs, diagrams, etc.

    Args:
        json_input: Path to parsed JSON with image_base64 data and classification results

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("STEP 2b: Running Vision RAG (Gemini)")
        logger.info("Extracting visual content from classified pages...")

        # Run vision_rag_gemini as subprocess
        cmd = [sys.executable, '-u', 'vision_rag_gemini.py', '--input', json_input]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        returncode = process.wait()

        if returncode == 0:
            logger.info("Vision RAG complete")
            return True
        else:
            logger.error(f"Vision RAG failed with return code {returncode}")
            return False

    except Exception as e:
        logger.error(f"Vision RAG failed: {e}")
        return False


def run_vision_classifier_openai(json_input: str = './output/output_parsed.json') -> bool:
    """
    Run Vision Classifier (OpenAI) to identify pages with visual content.

    This pre-filters pages to identify which ones contain meaningful visual
    elements (charts, graphs, diagrams) worth processing with Vision RAG.
    Saves API costs by skipping text-only pages.

    Args:
        json_input: Path to parsed JSON with image_base64 data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("STEP 2a: Running Vision Classifier (OpenAI)")
        logger.info("Identifying pages with visual content (charts, graphs, diagrams)...")

        # Run vision_classifier_openai as subprocess
        cmd = [sys.executable, '-u', 'vision_classifier_openai.py', '--input', json_input]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        returncode = process.wait()

        if returncode == 0:
            logger.info("Vision classification complete")
            return True
        else:
            logger.error(f"Vision classification failed with return code {returncode}")
            return False

    except Exception as e:
        logger.error(f"Vision classification failed: {e}")
        return False


def run_vision_rag_openai(json_input: str = './output/output_parsed.json') -> bool:
    """
    Run Vision RAG (OpenAI) to extract visual content from classified pages.

    This runs only on pages that were identified by the classifier as containing
    meaningful visual content. Extracts insights from charts, graphs, diagrams, etc.

    Args:
        json_input: Path to parsed JSON with image_base64 data and classification results

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("STEP 2b: Running Vision RAG (OpenAI)")
        logger.info("Extracting visual content from classified pages...")

        # Run vision_rag_openai as subprocess
        cmd = [sys.executable, '-u', 'vision_rag_openai.py', '--input', json_input]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        returncode = process.wait()

        if returncode == 0:
            logger.info("Vision RAG complete")
            return True
        else:
            logger.error(f"Vision RAG failed with return code {returncode}")
            return False

    except Exception as e:
        logger.error(f"Vision RAG failed: {e}")
        return False


def run_vision_classifier_ollama(json_input: str = './output/output_parsed.json') -> bool:
    """
    Run Vision Classifier (Ollama) to identify pages with visual content.

    This pre-filters pages to identify which ones contain meaningful visual
    elements (charts, graphs, diagrams) worth processing with Vision RAG.
    Uses local Ollama with Ollama vision model.

    Args:
        json_input: Path to parsed JSON with image_base64 data

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("STEP 2a: Running Vision Classifier (Ollama)")
        logger.info("Identifying pages with visual content (charts, graphs, diagrams)...")

        # Run vision_classifier_ollama as subprocess
        cmd = [sys.executable, '-u', 'vision_classifier_ollama.py', '--input', json_input]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        returncode = process.wait()

        if returncode == 0:
            logger.info("Vision classification complete")
            return True
        else:
            logger.error(f"Vision classification failed with return code {returncode}")
            return False

    except Exception as e:
        logger.error(f"Vision classification failed: {e}")
        return False


def run_vision_rag_ollama(json_input: str = './output/output_parsed.json') -> bool:
    """
    Run Vision RAG (Ollama) to extract visual content from classified pages.

    This runs only on pages that were identified by the classifier as containing
    meaningful visual content. Uses local Ollama with Ollama vision model.

    Args:
        json_input: Path to parsed JSON with image_base64 data and classification results

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("STEP 2b: Running Vision RAG (Ollama)")
        logger.info("Extracting visual content from classified pages...")

        # Run vision_rag_ollama as subprocess
        cmd = [sys.executable, '-u', 'vision_rag_ollama.py', '--input', json_input]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        returncode = process.wait()

        if returncode == 0:
            logger.info("Vision RAG complete")
            return True
        else:
            logger.error(f"Vision RAG failed with return code {returncode}")
            return False

    except Exception as e:
        logger.error(f"Vision RAG failed: {e}")
        return False


def run_summarizer(mode: str = 'json', json_input: str = './output/output_parsed.json', summarizer_mode: str = 'full') -> bool:
    """
    Run RAG summarizer.

    Args:
        mode: Processing mode ('json', 'text', or 'pdf')
        json_input: Path to PDF parser JSON output (for json mode)
        summarizer_mode: Summarizer processing mode ('full', 'topics-only', 'skip')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        mode_desc = {
            'full': 'full summarization (notes + topics)',
            'topics-only': 'topics only',
            'skip': 'skipped'
        }
        logger.info(f"STEP 2: Running summarizer in {mode} mode - {mode_desc.get(summarizer_mode, summarizer_mode)}")

        # Run summarizer as subprocess to avoid import conflicts
        cmd = [sys.executable, '-u', 'summarizer.py', '--mode', mode, '--summarizer-mode', summarizer_mode]
        if mode == 'json':
            cmd.extend(['--json-input', json_input])

        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        # Wait for process to complete
        returncode = process.wait()

        if returncode == 0:
            logger.info("Summarization complete")
            return True
        else:
            logger.error(f"Summarization failed with return code {returncode}")
            return False

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return False


def process_single_file(config: PipelineConfig) -> bool:
    """
    Process a single file (PDF, TXT, or MD) through the pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {config.input_file}")
    logger.info(f"{'='*60}\n")

    # Step 1: Parse file (PDF or text)
    if config.is_text_file():
        # Text/Markdown processing
        if not run_text_parser(
            config.input_file,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        ):
            logger.error("Pipeline failed at text parsing step")
            return False
    else:
        # PDF processing
        if not run_pdf_parser(
            config.input_file,
            skip_tables=config.skip_tables,
            skip_images=config.skip_images,
            pdf_image_dpi=config.pdf_image_dpi
        ):
            logger.error("Pipeline failed at PDF parsing step")
            return False

        # Step 2: Vision Processing (optional, only for PDFs with images)
        if config.vision_mode == 'deepseek':
            # DeepSeek OCR: Extract text from page images
            if not run_vision_ocr_deepseek():
                logger.error("Pipeline failed at Vision OCR (DeepSeek) step")
                return False
        elif config.vision_mode == 'gemini':
            # Gemini Vision: Classify pages, then extract visual content
            if not run_vision_classifier_gemini():
                logger.error("Pipeline failed at Vision Classifier (Gemini) step")
                return False

            if not run_vision_rag_gemini():
                logger.error("Pipeline failed at Vision RAG (Gemini) step")
                return False
        elif config.vision_mode == 'openai':
            # OpenAI Vision: Classify pages, then extract visual content
            if not run_vision_classifier_openai():
                logger.error("Pipeline failed at Vision Classifier (OpenAI) step")
                return False

            if not run_vision_rag_openai():
                logger.error("Pipeline failed at Vision RAG (OpenAI) step")
                return False
        elif config.vision_mode == 'ollama':
            # Ollama Vision: Classify pages, then extract visual content with local Ollama
            if not run_vision_classifier_ollama():
                logger.error("Pipeline failed at Vision Classifier (Ollama) step")
                return False

            if not run_vision_rag_ollama():
                logger.error("Pipeline failed at Vision RAG (Ollama) step")
                return False

    if not config.run_summarization:
        logger.info(f"\n{'='*60}")
        logger.info("Content extraction complete (Summarization skipped)")
        logger.info(f"{'='*60}\n")
        return True

    # Final Step: Summarize
    # Determine which JSON file to use based on vision processing
    summarizer_input = './output/output_parsed.json'
    if config.vision_mode == 'gemini':
        summarizer_input = './output/output_vision.json'
    elif config.vision_mode == 'openai':
        summarizer_input = './output/output_vision.json'
    elif config.vision_mode == 'ollama':
        summarizer_input = './output/output_vision_ollama.json'
    # DeepSeek OCR writes back to output_parsed.json, so no change needed

    logger.info(f"Running summarizer on: {summarizer_input}")
    if not run_summarizer(mode='json', json_input=summarizer_input, summarizer_mode=config.summarizer_mode):
        logger.error("Pipeline failed at summarization step")
        return False

    logger.info(f"\n{'='*60}")
    logger.info("Pipeline complete!")
    logger.info(f"{'='*60}\n")
    return True


def process_batch(config: PipelineConfig):
    """
    Process all supported files in the input directory.

    Args:
        config: Pipeline configuration
    """
    input_dir = Path('./input')

    # Gather all supported files
    pdf_files = list(input_dir.glob('*.pdf'))
    txt_files = list(input_dir.glob('*.txt'))
    md_files = list(input_dir.glob('*.md'))

    all_files = pdf_files + txt_files + md_files

    if not all_files:
        logger.warning("No supported files found in ./input directory")
        return

    logger.info(f"Found {len(all_files)} file(s) to process:")
    logger.info(f"  PDFs: {len(pdf_files)}, TXT: {len(txt_files)}, MD: {len(md_files)}")

    success_count = 0
    for file_path in all_files:
        # Create a copy of config for this file
        file_config = PipelineConfig(**asdict(config))
        file_config.input_file = str(file_path)
        file_config.batch_mode = False

        if process_single_file(file_config):
            success_count += 1

    logger.info(f"\nBatch processing complete: {success_count}/{len(all_files)} successful")


def interactive_mode() -> Optional[PipelineConfig]:
    """
    Run interactive configuration mode.

    Returns:
        PipelineConfig or None if user cancels
    """
    InteractivePrompt.clear_screen()
    InteractivePrompt.print_header()

    # Step 1: Select File
    file_selection = InteractivePrompt.select_file()
    if file_selection is None:
        print("\n👋 Cancelled by user")
        return None

    # Handle batch mode
    batch_mode = (file_selection == "BATCH_MODE")
    input_file = None if batch_mode else file_selection

    # Detect file type
    file_type = None
    is_text_file = False
    if input_file:
        ext = Path(input_file).suffix.lower()
        if ext == '.pdf':
            file_type = 'pdf'
        elif ext == '.txt':
            file_type = 'text'
            is_text_file = True
        elif ext == '.md':
            file_type = 'markdown'
            is_text_file = True

    # Step 2: Configure extraction (only for PDFs)
    extraction_config = {"skip_tables": False, "skip_images": False}
    vision_mode = 'none'
    if not is_text_file and not batch_mode:
        extraction_config = InteractivePrompt.configure_extraction()

        # Step 3: Configure Vision Processing (only if images are being extracted)
        if not extraction_config["skip_images"]:
            vision_mode = InteractivePrompt.configure_vision_processing()

    # Step 4: Configure pipeline stages
    pipeline_config = InteractivePrompt.configure_pipeline()

    # Build configuration
    config = PipelineConfig(
        input_file=input_file,
        batch_mode=batch_mode,
        file_type=file_type,
        skip_tables=extraction_config["skip_tables"],
        skip_images=extraction_config["skip_images"],
        pdf_image_dpi=extraction_config.get("pdf_image_dpi", 144),
        vision_mode=vision_mode,
        extract_only=pipeline_config["extract_only"],
        run_summarization=pipeline_config["run_summarization"],
        summarizer_mode=pipeline_config["summarizer_mode"]
    )

    # Show summary and confirm
    print(config.summary())

    if not InteractivePrompt.yes_no_prompt("Proceed with this configuration?", default=True):
        print("\n👋 Cancelled by user")
        return None

    return config


def cli_mode(args) -> Optional[PipelineConfig]:
    """
    Parse command-line arguments into configuration.

    Args:
        args: Parsed argparse arguments

    Returns:
        PipelineConfig or None if invalid
    """
    # Validate inputs
    if args.batch:
        config = PipelineConfig(
            batch_mode=True,
            extract_only=args.extract_only,
            skip_tables=args.skip_tables or args.text_only,
            skip_images=args.skip_images or args.text_only,
            pdf_image_dpi=args.pdf_image_dpi,
            vision_mode=args.vision_mode,
            summarizer_mode=args.summarizer_mode
        )
    elif args.input_file:
        if not os.path.exists(args.input_file):
            logger.error(f"File not found: {args.input_file}")
            return None

        # Validate file type
        ext = Path(args.input_file).suffix.lower()
        if ext not in ['.pdf', '.txt', '.md']:
            logger.error(f"Unsupported file type: {ext}. Use .pdf, .txt, or .md")
            return None

        config = PipelineConfig(
            input_file=args.input_file,
            batch_mode=False,
            extract_only=args.extract_only,
            skip_tables=args.skip_tables or args.text_only,
            skip_images=args.skip_images or args.text_only,
            pdf_image_dpi=args.pdf_image_dpi,
            vision_mode=args.vision_mode,
            summarizer_mode=args.summarizer_mode
        )
    else:
        return None

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Document Processing Pipeline - Interactive & CLI modes (PDF/TXT/MD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (recommended):
    python run.py

  CLI mode - Process a single file:
    python run.py document.pdf
    python run.py transcript.txt
    python run.py notes.md

  CLI mode - Process all files in input directory:
    python run.py --batch

  CLI mode - Extract only (no summarization):
    python run.py document.pdf --extract-only

  CLI mode - PDF text-only extraction (skip tables and images):
    python run.py document.pdf --text-only

  CLI mode - Vision processing with DeepSeek OCR:
    python run.py document.pdf --vision-mode deepseek

  CLI mode - Vision processing with Gemini (classifier + RAG):
    python run.py document.pdf --vision-mode gemini

  CLI mode - Vision processing with OpenAI (classifier + RAG):
    python run.py document.pdf --vision-mode openai

  CLI mode - Vision processing with Ollama (classifier + RAG, local):
    python run.py document.pdf --vision-mode ollama
        """
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Path to file to process: PDF, TXT, or MD (omit for interactive mode)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all supported files in ./input directory (PDF/TXT/MD)'
    )
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Run only extraction/chunking (Step 1) and skip summarization'
    )
    parser.add_argument(
        '--skip-tables',
        action='store_true',
        help='Skip table extraction during PDF parsing (PDF only)'
    )
    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='Skip image extraction during PDF parsing (PDF only)'
    )
    parser.add_argument(
        '--text-only',
        action='store_true',
        help='Skip both table and image extraction (PDF only, equivalent to --skip-tables --skip-images)'
    )
    parser.add_argument(
        '--vision-mode',
        choices=['none', 'deepseek', 'gemini', 'openai', 'ollama'],
        default='none',
        help='Vision processing mode: none (default), deepseek (OCR via self-hosted server), gemini (AI visual content via Gemini API), openai (AI visual content via OpenAI API), ollama (AI visual content via local Ollama, zero cost)'
    )
    parser.add_argument(
        '--summarizer-mode',
        choices=['full', 'topics-only', 'skip'],
        default='full',
        help='Summarization mode: full (notes + topics, default), topics-only (faster), skip (no summarization)'
    )
    parser.add_argument(
        '--pdf-image-dpi',
        type=int,
        choices=[72, 144, 200, 300],
        default=144,
        help='PDF image extraction resolution: 72 (low), 144 (medium, default), 200 (high), 300 (very high)'
    )

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # Determine mode: interactive vs CLI
    if args.input_file or args.batch:
        # CLI mode
        config = cli_mode(args)
        if config is None:
            parser.print_help()
            sys.exit(1)
    else:
        # Interactive mode
        config = interactive_mode()
        if config is None:
            sys.exit(0)

    # Execute pipeline
    print("\n🚀 Starting pipeline execution...\n")

    if config.batch_mode:
        process_batch(config)
    else:
        success = process_single_file(config)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
