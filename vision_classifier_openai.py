"""
Vision Classifier - Pre-filter pages with graphical content (OpenAI)

Classifies each page in output_parsed.json to determine if it contains
meaningful visual elements (charts, graphs, diagrams) worth processing
with the full Vision RAG pipeline. Saves API costs by skipping text-only pages.

Usage:
    python vision_classifier_openai.py
    python vision_classifier_openai.py --input output/output_parsed.json
    python vision_classifier_openai.py --dry-run
"""

import os
import sys
import base64
import json
import logging
import time
import argparse
import tempfile
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
DEFAULT_INPUT = 'output/output_parsed.json'
MODEL = 'gpt-4.1-mini'  # Fast and cost-effective for classification
MAX_RETRIES = 3
BASE_RETRY_DELAY = 5
REQUEST_DELAY = 1  # OpenAI has higher rate limits than Gemini
SAVE_JPG = True  # Save converted JPG images to temp_images_classifier/ for manual quality review


def setup_logger():
    """Initialize logging to logs/vision-classifier-openai.log"""
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger('vision_classifier_openai')
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler('logs/vision-classifier-openai.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def open_file(filepath):
    """Read text file contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


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

    # Create temp_images_classifier directory if it doesn't exist
    output_dir = 'temp_images_classifier'
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


def classify_image(image_data_uri, prompt, client, logger):
    """
    Send image to OpenAI for quick classification.
    Returns True if visual elements detected, False otherwise.
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
                                "detail": "low"  # Low detail is sufficient for classification
                            }
                        }
                    ]
                }],
                max_completion_tokens=10  # We only need a YES/NO response
            )

            result_text = response.choices[0].message.content.strip().upper()
            # Check for YES response
            return result_text.startswith('YES')

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (attempt + 1)
                logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {error_msg}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Max retries reached: {error_msg}")
                # Default to True on error to avoid missing important content
                return True


def classify_chunk(chunk, client, prompt, logger):
    """
    Classify a chunk's images to determine if visual elements exist.
    Returns True if any image has visual elements, False otherwise.
    """
    images = chunk.get('image_base64', [])
    chunk_id = chunk.get('chunk_id', 'unknown')

    if not images:
        return False

    # Check first image only (page render) - sufficient for classification
    try:
        # Save JPG for manual quality review if enabled
        if SAVE_JPG:
            saved_path = save_base64_as_jpg(images[0], chunk_id, image_index=0)
            if saved_path:
                logger.info(f"[{chunk_id}] Saved image to: {saved_path}")

        image_data_uri = prepare_base64_image(images[0])
        has_graphics = classify_image(image_data_uri, prompt, client, logger)
        time.sleep(REQUEST_DELAY)  # Rate limit delay
        return has_graphics
    except Exception as e:
        logger.error(f"Failed to classify image: {e}")
        # Default to True on error to avoid missing content
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Vision Classifier - Pre-filter pages with graphical content (OpenAI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vision_classifier_openai.py
    python vision_classifier_openai.py --input output/output_parsed.json
    python vision_classifier_openai.py --start-from 5
    python vision_classifier_openai.py --dry-run
        """
    )
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help=f'Input JSON from pdf_parser (default: {DEFAULT_INPUT})')
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

    # Load classifier prompt
    prompt_path = 'prompts/vision-classifier.txt'
    if not os.path.exists(prompt_path):
        logger.error(f"Prompt file not found: {prompt_path}")
        sys.exit(1)
    prompt = open_file(prompt_path)

    # Load parsed document
    logger.info(f"Loading input: {args.input}")
    data = load_json(args.input)
    pages = data.get('pages', [])
    total_pages = len(pages)

    if total_pages == 0:
        logger.warning("No pages found in input document")
        sys.exit(0)

    logger.info(f"Starting classification for {total_pages} pages using OpenAI {MODEL}")
    if args.start_from > 1:
        logger.info(f"Resuming from chunk {args.start_from}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No API calls will be made")
    if SAVE_JPG:
        logger.info("Image saving ENABLED - JPG files will be saved to temp_images_classifier/ folder")
    else:
        logger.info("Image saving DISABLED - No JPG files will be saved")

    # Process each page
    graphics_count = 0
    text_only_count = 0

    for idx, chunk in enumerate(pages):
        chunk_num = idx + 1
        chunk_id = chunk.get('chunk_id', f'chunk_{chunk_num}')

        # Skip chunks before start_from
        if chunk_num < args.start_from:
            continue

        images = chunk.get('image_base64', [])

        if not images:
            logger.info(f"[{chunk_id}] No images, marking as text-only")
            chunk['image_classifier'] = False
            text_only_count += 1
            continue

        logger.info(f"[{chunk_id}] Classifying... ({chunk_num}/{total_pages})")

        if args.dry_run:
            # Simulate - assume alternating for testing
            has_graphics = (chunk_num % 3 != 0)  # 2/3 have graphics in dry run
            chunk['image_classifier'] = has_graphics
            status = "HAS GRAPHICS" if has_graphics else "TEXT ONLY"
            logger.info(f"[{chunk_id}] DRY RUN - {status}")
        else:
            has_graphics = classify_chunk(chunk, client, prompt, logger)
            chunk['image_classifier'] = has_graphics
            status = "HAS GRAPHICS" if has_graphics else "TEXT ONLY"
            logger.info(f"[{chunk_id}] {status}")

        if has_graphics:
            graphics_count += 1
        else:
            text_only_count += 1

        # Save progress after each chunk
        save_json(data, args.input)

    logger.info(f"Classification complete.")
    logger.info(f"  Pages with graphics: {graphics_count}")
    logger.info(f"  Text-only pages: {text_only_count}")
    logger.info(f"  API calls saved by Vision RAG: {text_only_count}")
    logger.info(f"Output saved to: {args.input}")


if __name__ == "__main__":
    main()
