"""
Vision RAG - OpenAI Image Understanding

Processes base64-encoded page images from pdf_parser output and enriches
each chunk with visual content extraction using OpenAI GPT-4 Vision API.

Usage:
    python vision_rag_openai.py
    python vision_rag_openai.py --input output/output_parsed.json --output output/output_vision.json
    python vision_rag_openai.py --start-from 5  # Resume from chunk 5
"""

import os
import sys
import base64
import json
import logging
import time
import argparse
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
DEFAULT_INPUT = 'output/output_parsed.json'
DEFAULT_OUTPUT = 'output/output_vision.json'
MODEL = 'gpt-4.1-mini'  
MAX_RETRIES = 3
BASE_RETRY_DELAY = 5
REQUEST_DELAY = 1  # OpenAI has higher rate limits
SAVE_JPG = True  # Save converted JPG images to temp_images_rag/ for manual quality review


def setup_logger():
    """Initialize logging to logs/vision-rag-openai.log"""
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger('vision_rag_openai')
    logger.setLevel(logging.DEBUG)  # Enable DEBUG level for troubleshooting

    # File handler
    file_handler = logging.FileHandler('logs/vision-rag-openai.log')
    file_handler.setLevel(logging.DEBUG)  # Log DEBUG to file
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Keep console at INFO to reduce noise
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def open_file(filepath):
    """Read text file contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_parsed_document(filepath):
    """Load output_parsed.json"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_progress(data, filepath):
    """Atomic save with temp file pattern to prevent data loss"""
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', dir=os.path.dirname(filepath) or '.')
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, filepath)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def prepare_base64_image(base64_string):
    """Prepare base64 string for OpenAI API (add data URI prefix if needed)"""
    # Remove existing data URI prefix if present
    if ',' in base64_string and base64_string.startswith('data:'):
        base64_string = base64_string.split(',', 1)[1]

    # Return with proper data URI prefix
    return f"data:image/jpeg;base64,{base64_string}"


def save_base64_as_jpg(base64_string, chunk_id, image_index=0):
    """
    Save base64-encoded image as JPG file for manual quality review.

    Args:
        base64_string: Base64-encoded image data
        chunk_id: Chunk identifier for filename
        image_index: Index of image within chunk (default 0)

    Returns:
        Path to saved file, or None if saving disabled
    """
    if not SAVE_JPG:
        return None

    # Create temp_images_rag directory if it doesn't exist
    output_dir = 'temp_images_rag'
    os.makedirs(output_dir, exist_ok=True)

    # Remove data URI prefix if present
    if ',' in base64_string and base64_string.startswith('data:'):
        base64_string = base64_string.split(',', 1)[1]

    # Decode base64 to binary
    try:
        image_binary = base64.b64decode(base64_string)
    except Exception as e:
        return None

    # Generate filename
    filename = f"{chunk_id}_img{image_index}.jpg"
    filepath = os.path.join(output_dir, filename)

    # Save to file
    try:
        with open(filepath, 'wb') as f:
            f.write(image_binary)
        return filepath
    except Exception as e:
        return None


def process_image_with_openai(image_data_uri, prompt, client, logger, detail="high"):
    """
    Send image to OpenAI API with retry logic for rate limits.
    Returns extracted text description of visual elements.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri,
                                "detail": detail  # "high" for detailed analysis
                            }
                        }
                    ]
                }],
                max_completion_tokens=1000  # Sufficient for detailed visual descriptions
            )

            # Debug logging
            content = response.choices[0].message.content
            logger.debug(f"API Response - Model: {MODEL}")
            logger.debug(f"API Response - Content length: {len(content) if content else 0}")
            logger.debug(f"API Response - Content preview: {content[:200] if content else 'EMPTY/NULL'}")
            logger.debug(f"API Response - Finish reason: {response.choices[0].finish_reason}")

            return content

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (attempt + 1)
                logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {error_msg}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Max retries reached: {error_msg}")
                return f"[Vision extraction failed: {error_msg}]"


def process_chunk(chunk, client, prompt, logger):
    """
    Process all images in a chunk, combining results.
    Only processes chunks marked with image_classifier=True.
    Returns concatenated image_text for all images.
    """
    # Check if classifier marked this as having graphics
    has_graphics = chunk.get('image_classifier', True)  # Default True if not classified

    if not has_graphics:
        return "[Text-only page, skipped by classifier]"

    images = chunk.get('image_base64', [])
    chunk_id = chunk.get('chunk_id', 'unknown')

    if not images:
        return None

    if len(images) == 1:
        try:
            # Save JPG for manual quality review if enabled
            if SAVE_JPG:
                saved_path = save_base64_as_jpg(images[0], chunk_id, image_index=0)
                if saved_path:
                    logger.info(f"[{chunk_id}] Saved image to: {saved_path}")

            image_data_uri = prepare_base64_image(images[0])
            result = process_image_with_openai(image_data_uri, prompt, client, logger)
            time.sleep(REQUEST_DELAY)  # Rate limit delay
            return result
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return f"[Image processing failed: {str(e)}]"

    # Multiple images: process each, combine with separators
    results = []
    for i, img_b64 in enumerate(images):
        try:
            # Save JPG for manual quality review if enabled
            if SAVE_JPG:
                saved_path = save_base64_as_jpg(img_b64, chunk_id, image_index=i)
                if saved_path:
                    logger.info(f"[{chunk_id}] Saved image {i+1} to: {saved_path}")

            image_data_uri = prepare_base64_image(img_b64)
            result = process_image_with_openai(image_data_uri, prompt, client, logger)
            results.append(f"[Image {i+1}]: {result}")
            time.sleep(REQUEST_DELAY)  # Rate limit delay
        except Exception as e:
            logger.error(f"Failed to process image {i+1}: {e}")
            results.append(f"[Image {i+1}]: [Error: {str(e)}]")

    return "\n\n".join(results)


def main():
    parser = argparse.ArgumentParser(
        description='Vision RAG - OpenAI Image Understanding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vision_rag_openai.py
    python vision_rag_openai.py --input output/output_parsed.json
    python vision_rag_openai.py --start-from 5
    python vision_rag_openai.py --dry-run
        """
    )
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help=f'Input JSON from pdf_parser (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output JSON with image_text (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--start-from', type=int, default=1,
                        help='Resume from specific chunk number (1-indexed)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Process without API calls (for testing)')
    args = parser.parse_args()

    logger = setup_logger()

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key and not args.dry_run:
        logger.error("OPENAI_API_KEY not found in environment. Add it to .env file.")
        sys.exit(1)

    # Check input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Initialize OpenAI client
    client = None
    if not args.dry_run:
        client = OpenAI(api_key=api_key)

    # Load vision extraction prompt
    prompt_path = 'prompts/vision-extract.txt'
    if not os.path.exists(prompt_path):
        logger.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)
    prompt = open_file(prompt_path)

    # Load parsed document
    logger.info(f"Loading input: {args.input}")
    data = load_parsed_document(args.input)
    pages = data.get('pages', [])
    total_pages = len(pages)

    if total_pages == 0:
        logger.warning("No pages found in input document")
        sys.exit(0)

    logger.info(f"Starting Vision RAG processing for {total_pages} pages using OpenAI {MODEL}")
    if args.start_from > 1:
        logger.info(f"Resuming from chunk {args.start_from}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No API calls will be made")
    if SAVE_JPG:
        logger.info("Image saving ENABLED - JPG files will be saved to temp_images_rag/ folder")
    else:
        logger.info("Image saving DISABLED - No JPG files will be saved")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Process each page sequentially
    processed_count = 0
    skipped_count = 0
    classifier_skipped = 0

    for idx, chunk in enumerate(pages):
        chunk_num = idx + 1
        chunk_id = chunk.get('chunk_id', f'chunk_{chunk_num}')

        # Skip chunks before start_from
        if chunk_num < args.start_from:
            continue

        # Check if classifier marked this as text-only
        has_graphics = chunk.get('image_classifier', True)
        if not has_graphics:
            logger.info(f"[{chunk_id}] Skipped by classifier (text-only page)")
            chunk['image_text'] = "[Text-only page, skipped by classifier]"
            classifier_skipped += 1
            continue

        images = chunk.get('image_base64', [])

        if not images:
            logger.info(f"[{chunk_id}] No images to process, skipping")
            chunk['image_text'] = None
            skipped_count += 1
            continue

        logger.info(f"[{chunk_id}] Processing {len(images)} image(s)... ({chunk_num}/{total_pages})")

        if args.dry_run:
            # Simulate processing
            chunk['image_text'] = f"[DRY RUN] Would process {len(images)} image(s)"
            logger.info(f"[{chunk_id}] DRY RUN complete")
        else:
            # Process chunk images with OpenAI
            image_text = process_chunk(chunk, client, prompt, logger)
            chunk['image_text'] = image_text

            text_len = len(image_text) if image_text else 0
            logger.info(f"[{chunk_id}] Complete - extracted {text_len} chars")

        processed_count += 1

        # Save progress after each chunk (incremental saving pattern)
        save_progress(data, args.output)

    logger.info(f"Vision RAG complete.")
    logger.info(f"  Processed: {processed_count}")
    logger.info(f"  Skipped (no images): {skipped_count}")
    logger.info(f"  Skipped (classifier): {classifier_skipped}")
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
