"""
Vision OCR - DeepSeek-based OCR text extraction from page images

Processes base64-encoded page images from pdf_parser output and extracts
all text content using OCR. This captures text that may be missed by
standard PDF text extraction (embedded in images, graphics, charts, etc.).
Writes image_text back to the same output_parsed.json for use by summarizer.

Uses your self-hosted DeepSeek-OCR server for zero-cost processing.

Usage:
    python vision_ocr_deepseek.py
    python vision_ocr_deepseek.py --input output/output_parsed.json
    python vision_ocr_deepseek.py --server http://192.168.10.3:8000
    python vision_ocr_deepseek.py --start-from 5  # Resume from chunk 5
"""

import os
import sys
import json
import logging
import time
import argparse
import tempfile
import re
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
DEFAULT_INPUT = 'output/output_parsed.json'
DEFAULT_OUTPUT = 'output/output_parsed.json'  # Write back to same file
DEFAULT_SERVER = os.getenv('DEEPSEEK_OCR_URL', 'http://192.168.10.3:8000')
REQUEST_TIMEOUT = 120  # DeepSeek can be slow on complex pages
BASE_SIZE = 1024
IMAGE_SIZE = 640


def setup_logger():
    """Initialize logging to logs/vision-ocr.log"""
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger('vision_ocr')
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler('logs/vision-ocr.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath):
    """Atomic save with temp file pattern"""
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', dir=os.path.dirname(filepath) or '.')
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, filepath)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def check_server_health(server_url: str, logger) -> bool:
    """Verify DeepSeek-OCR server is running and model is loaded."""
    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            model_loaded = health.get('model_loaded', False)
            if model_loaded:
                logger.info(f"Server healthy - Model loaded: {health.get('model_name', 'unknown')}")
                if health.get('gpu_available'):
                    logger.info(f"GPU: {health.get('gpu_name', 'unknown')}")
            return model_loaded
        else:
            logger.error(f"Health check returned status {response.status_code}")
            return False
    except requests.ConnectionError:
        logger.error(f"Cannot connect to server at {server_url}")
        return False
    except requests.Timeout:
        logger.error("Server health check timed out")
        return False
    except Exception as e:
        logger.error(f"Server health check failed: {e}")
        return False


def clean_ocr_output(text: str) -> str:
    """Clean DeepSeek-OCR output of special tokens and debug info."""
    if not text:
        return text

    # Remove end-of-sentence token
    text = text.replace('<｜end▁of▁sentence｜>', '')

    # Remove debug output patterns (torch.Size, etc.)
    text = re.sub(r'={5,}\n.*?torch\.Size.*?\n={5,}\n?', '', text, flags=re.DOTALL)

    # Normalize excessive newlines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()


def process_image_with_deepseek(image_base64: str, server_url: str, logger) -> dict:
    """
    Send image to DeepSeek-OCR server for full OCR text extraction.
    Returns dict with image_text (all extracted text from the page image).
    """
    try:
        response = requests.post(
            f"{server_url}/document-to-markdown",
            json={
                "image_base64": image_base64,
                "base_size": BASE_SIZE,
                "image_size": IMAGE_SIZE,
                "crop_mode": True
            },
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                markdown = result.get("result", "")
                markdown = clean_ocr_output(markdown)
                return {"image_text": markdown}
            else:
                return {
                    "image_text": None,
                    "error": result.get("error", "Unknown error")
                }
        else:
            return {
                "image_text": None,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }

    except requests.Timeout:
        return {
            "image_text": None,
            "error": "Request timeout (page may be too complex)"
        }
    except requests.ConnectionError:
        return {
            "image_text": None,
            "error": "Connection error - server may be down"
        }
    except Exception as e:
        return {
            "image_text": None,
            "error": str(e)
        }


def process_chunk(chunk: dict, server_url: str, logger) -> dict:
    """
    Process all images in a chunk with OCR.
    For multi-image chunks, combines results.
    """
    images = chunk.get('image_base64', [])

    if not images:
        return {"image_text": None}

    if len(images) == 1:
        return process_image_with_deepseek(images[0], server_url, logger)

    # Multiple images: process each, combine results
    results = []

    for i, img_b64 in enumerate(images):
        result = process_image_with_deepseek(img_b64, server_url, logger)

        if result.get("error"):
            results.append(f"[Image {i+1}]: Error - {result['error']}")
        elif result.get("image_text"):
            results.append(f"[Image {i+1}]:\n{result['image_text']}")
        else:
            results.append(f"[Image {i+1}]: No text extracted")

    return {"image_text": "\n\n".join(results) if results else None}


def main():
    parser = argparse.ArgumentParser(
        description='Vision OCR - DeepSeek-based visual content extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vision_ocr_deepseek.py
    python vision_ocr_deepseek.py --input output/output_parsed.json
    python vision_ocr_deepseek.py --server http://192.168.10.3:8000
    python vision_ocr_deepseek.py --start-from 5
    python vision_ocr_deepseek.py --dry-run
        """
    )
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help=f'Input JSON from pdf_parser (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output JSON with image_text (default: same as input)')
    parser.add_argument('--server', default=DEFAULT_SERVER,
                        help=f'DeepSeek-OCR server URL (default: {DEFAULT_SERVER})')
    parser.add_argument('--start-from', type=int, default=1,
                        help='Resume from specific chunk number (1-indexed)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process without API calls (for testing)')
    parser.add_argument('--skip-health-check', action='store_true',
                        help='Skip server health check')
    args = parser.parse_args()

    logger = setup_logger()

    logger.info(f"DeepSeek-OCR Server: {args.server}")

    # Check server health
    if not args.dry_run and not args.skip_health_check:
        if not check_server_health(args.server, logger):
            logger.error("DeepSeek-OCR server not available. Use --skip-health-check to bypass.")
            sys.exit(1)

    # Check input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load parsed document
    logger.info(f"Loading input: {args.input}")
    data = load_json(args.input)
    pages = data.get('pages', [])
    total_pages = len(pages)

    if total_pages == 0:
        logger.warning("No pages found in input document")
        sys.exit(0)

    logger.info(f"Starting Vision OCR processing for {total_pages} pages")
    if args.start_from > 1:
        logger.info(f"Resuming from chunk {args.start_from}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No API calls will be made")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Process each page sequentially
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for idx, chunk in enumerate(pages):
        chunk_num = idx + 1
        chunk_id = chunk.get('chunk_id', f'chunk_{chunk_num}')

        # Skip chunks before start_from
        if chunk_num < args.start_from:
            continue

        images = chunk.get('image_base64', [])

        if not images:
            logger.info(f"[{chunk_id}] No images to process, skipping")
            chunk['image_text'] = None
            skipped_count += 1
            continue

        logger.info(f"[{chunk_id}] Running OCR on {len(images)} image(s)... ({chunk_num}/{total_pages})")

        if args.dry_run:
            chunk['image_text'] = "[DRY RUN] Would process with DeepSeek-OCR"
            logger.info(f"[{chunk_id}] DRY RUN complete")
        else:
            # Process chunk with DeepSeek-OCR
            result = process_chunk(chunk, args.server, logger)

            chunk['image_text'] = result.get('image_text')

            if result.get('error'):
                logger.warning(f"[{chunk_id}] Error: {result['error']}")
                error_count += 1
            else:
                text_len = len(result.get('image_text', '') or '')
                logger.info(f"[{chunk_id}] Complete - extracted {text_len} chars")

        processed_count += 1

        # Save progress after each chunk (incremental saving pattern)
        save_json(data, args.output)

    logger.info(f"Vision OCR complete.")
    logger.info(f"  Processed: {processed_count}")
    logger.info(f"  Skipped: {skipped_count}")
    if error_count > 0:
        logger.info(f"  Errors: {error_count}")
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
