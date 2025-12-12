"""
Vision Classifier - Pre-filter pages with graphical content (Ollama)

Classifies each page in output_parsed.json to determine if it contains
meaningful visual elements (charts, graphs, diagrams) worth processing
with the full Vision RAG pipeline. Uses local Ollama with Gemma3 vision model.

PERFORMANCE OPTIMIZATION:
- Images downsized to 72 DPI (612px width) for faster classification
- Uses BILINEAR resampling (speed over quality)
- Lower JPEG quality (85%) for smaller payloads
- Classification doesn't need high resolution - just detect presence of visuals

Usage:
    python vision_classifier_ollama.py
    python vision_classifier_ollama.py --input output/output_parsed.json
    python vision_classifier_ollama.py --dry-run
"""

import os
import sys
import base64
import json
import logging
import time
import argparse
import tempfile
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

# Configuration
DEFAULT_INPUT = 'output/output_parsed.json'
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
MODEL = 'ministral-3:latest'  # 8B model recommended for accurate diagram/chart detection

# Image preprocessing - downsize for faster classification
CLASSIFIER_TARGET_SIZE = 612  # 72 DPI equivalent (Letter: 612×792)
CLASSIFIER_QUALITY = 85  # Lower quality for classifier (speed over quality)

MAX_RETRIES = 3
BASE_RETRY_DELAY = 5
REQUEST_DELAY = 0.5  # Local server can handle faster requests
SAVE_JPG = False  # Save converted JPG images to temp_images_classifier_ollama/ for manual quality review


def setup_logger():
    """Initialize logging to logs/vision-classifier-ollama.log"""
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger('vision_classifier_ollama')
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler('logs/vision-classifier-ollama.log')
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


def downsize_image_for_classifier(base64_string, target_size=CLASSIFIER_TARGET_SIZE):
    """
    Downsize image to 72 DPI equivalent for faster classification.

    Classifier doesn't need high resolution - just needs to detect if
    page has charts/graphs/diagrams vs text-only.

    Args:
        base64_string: Base64-encoded image
        target_size: Target width for downsizing (default: 612 = 72 DPI)

    Returns:
        Downsized image as base64 string
    """
    try:
        # Decode base64 to bytes
        if ',' in base64_string and base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]

        image_bytes = base64.b64decode(base64_string)

        # Open image with PIL
        img = Image.open(BytesIO(image_bytes))

        # Get original dimensions
        orig_width, orig_height = img.size

        # Only downsize if image is larger than target
        if orig_width <= target_size:
            return base64_string  # Already small enough

        # Calculate new dimensions maintaining aspect ratio
        new_width = target_size
        new_height = int((orig_height / orig_width) * target_size)

        # Resize image (use BILINEAR for speed, not quality)
        resized_img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Convert to RGB if necessary
        if resized_img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', resized_img.size, (255, 255, 255))
            if resized_img.mode == 'P':
                resized_img = resized_img.convert('RGBA')
            background.paste(resized_img, mask=resized_img.split()[-1] if resized_img.mode == 'RGBA' else None)
            resized_img = background

        # Save to bytes buffer as JPEG with lower quality
        buffer = BytesIO()
        resized_img.save(buffer, format='JPEG', quality=CLASSIFIER_QUALITY, optimize=False)
        resized_bytes = buffer.getvalue()

        # Encode back to base64
        resized_base64 = base64.b64encode(resized_bytes).decode('utf-8')

        return resized_base64

    except Exception as e:
        logging.warning(f"Image downsize failed: {e}. Using original image.")
        if ',' in base64_string and base64_string.startswith('data:'):
            return base64_string.split(',', 1)[1]
        return base64_string


def prepare_base64_image(base64_string):
    """
    Prepare base64 string for Ollama OpenAI-compatible API.
    Downsizes to 72 DPI for faster classification.
    """
    # Downsize for classifier performance
    base64_string = downsize_image_for_classifier(base64_string)

    # Remove existing data URI prefix if present
    if ',' in base64_string and base64_string.startswith('data:'):
        base64_string = base64_string.split(',', 1)[1]

    # Return with proper data URI prefix
    return f"data:image/jpeg;base64,{base64_string}"


def save_base64_as_jpg(base64_string, chunk_id, image_index=0):
    """
    Save base64-encoded image as JPG file for manual quality review.

    Args:
        base64_string: Base64-encoded image data (downsized for classifier)
        chunk_id: Chunk identifier for filename
        image_index: Index of image within chunk (default 0)

    Returns:
        Path to saved file, or None if saving disabled
    """
    if not SAVE_JPG:
        return None

    # Create temp_images_classifier_ollama directory if it doesn't exist
    output_dir = 'temp_images_classifier_ollama'
    os.makedirs(output_dir, exist_ok=True)

    # Remove data URI prefix if present
    if ',' in base64_string and base64_string.startswith('data:'):
        base64_string = base64_string.split(',', 1)[1]

    # Decode base64 to binary
    try:
        image_binary = base64.b64decode(base64_string)
    except Exception as e:
        return None

    # Get image dimensions for filename
    try:
        img = Image.open(BytesIO(image_binary))
        width, height = img.size
        dim_suffix = f"_{width}x{height}"
    except:
        dim_suffix = ""

    # Generate filename with dimensions (shows downsized resolution)
    filename = f"{chunk_id}_img{image_index}{dim_suffix}_classifier.jpg"
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
    Send image to Ollama for quick classification.
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
                                "url": image_data_uri
                            }
                        }
                    ]
                }],
                max_tokens=10  # We only need a YES/NO response
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
        # Prepare image (downsizes to 72 DPI for faster classification)
        image_data_uri = prepare_base64_image(images[0])

        # Save downsized JPG for manual quality review if enabled
        if SAVE_JPG:
            # Extract downsized base64 from data URI
            downsized_b64 = image_data_uri.split(',', 1)[1] if ',' in image_data_uri else image_data_uri
            saved_path = save_base64_as_jpg(downsized_b64, chunk_id, image_index=0)
            if saved_path:
                logger.info(f"[{chunk_id}] Saved downsized image: {saved_path}")

        has_graphics = classify_image(image_data_uri, prompt, client, logger)
        time.sleep(REQUEST_DELAY)  # Rate limit delay
        return has_graphics
    except Exception as e:
        logger.error(f"Failed to classify image: {e}")
        # Default to True on error to avoid missing content
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Vision Classifier - Pre-filter pages with graphical content (Ollama)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vision_classifier_ollama.py
    python vision_classifier_ollama.py --input output/output_parsed.json
    python vision_classifier_ollama.py --start-from 5
    python vision_classifier_ollama.py --dry-run
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

    # Check input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Initialize Ollama client (OpenAI-compatible)
    client = None
    if not args.dry_run:
        # Ollama doesn't require a real API key, but the client needs something
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key='ollama'  # Ollama ignores this but OpenAI client requires it
        )
        logger.info(f"Connected to Ollama at: {OLLAMA_BASE_URL}")

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

    logger.info(f"Starting classification for {total_pages} pages using Ollama {MODEL}")
    logger.info(f"Image downsizing: {CLASSIFIER_TARGET_SIZE}px (72 DPI equivalent) for faster classification")
    if args.start_from > 1:
        logger.info(f"Resuming from chunk {args.start_from}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No API calls will be made")
    if SAVE_JPG:
        logger.info("Image saving ENABLED - JPG files will be saved to temp_images_classifier_ollama/ folder")
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
