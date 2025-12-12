"""
Vision RAG - Ollama Image Understanding (Enhanced 2025)

Processes base64-encoded page images from pdf_parser output and enriches
each chunk with visual content extraction using local Ollama vision models.

FEATURES:
- Structured Markdown output with explicit sections
- Research-backed prompting for precise data extraction
- Low temperature (0.1) for consistent, factual results
- Anti-hallucination instructions with confidence markers
- Uses native PDF extraction resolution (user-configurable in run.py)
- Supports: Ministral-3, Llama3.2-Vision, Gemma3-Vision

BEST PRACTICES IMPLEMENTED (2025):
- Temperature: 0.1 for factual consistency
- Max Tokens: 2000 for detailed structured output
- Seed: 42 for reproducible results
- Native image resolution: Uses PDF extraction DPI (72/144/200/300)
- High-quality JPEG encoding (95% quality)
- Explicit handling of unclear/missing data
- Structured schema prevents hallucination

IMAGE RESOLUTION:
- Uses native PDF extraction resolution set in run.py
- No resizing/resampling (preserves original quality)
- User controls DPI via --pdf-image-dpi (72/144/200/300)
- Recommended: 200-300 DPI for optimal vision model performance
- Filenames include actual dimensions: chunk_X_imgY_1700x2200.jpg

Usage:
    python vision_rag_ollama.py
    python vision_rag_ollama.py --input output/output_parsed.json
    python vision_rag_ollama.py --start-from 5  # Resume from chunk 5
    VISION_MODEL=gemma3:12b python vision_rag_ollama.py  # Use different model
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
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

# Configuration
DEFAULT_INPUT = 'output/output_parsed.json'
DEFAULT_OUTPUT = 'output/output_vision_ollama.json'
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')

# Model Configuration (Best Practices for Vision Extraction)
MODEL = os.getenv('VISION_MODEL', 'ministral-3:latest')

# LLM Parameters (Research-backed best practices)
TEMPERATURE = 0.1  # Low temperature for consistent, factual extraction (2025 best practice)
MAX_TOKENS = 2000  # Sufficient for detailed structured markdown output
SEED = 42  # Reproducible results across runs

# Image Preprocessing (2025 Best Practices for Vision Models)
RESIZE_IMAGES = False  # Disable resizing - use native PDF extraction resolution
TARGET_SIZE = 2048  # Target resolution (only used if RESIZE_IMAGES=True)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB max size for safety
MAINTAIN_ASPECT_RATIO = True  # Keep aspect ratio close to 1:1 (best practice)
IMAGE_QUALITY = 95  # JPEG quality for saved/processed images

# Processing Configuration
MAX_RETRIES = 3
BASE_RETRY_DELAY = 5
REQUEST_DELAY = 0.5  # Local server can handle faster requests
SAVE_JPG = False  # Save converted JPG images to temp_images_rag_ollama/ for manual quality review


def setup_logger():
    """Initialize logging to logs/vision-rag-ollama.log"""
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger('vision_rag_ollama')
    logger.setLevel(logging.DEBUG)  # Enable DEBUG level for troubleshooting

    # File handler
    file_handler = logging.FileHandler('logs/vision-rag-ollama.log')
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


def resize_image_for_vision_model(base64_string, target_size=TARGET_SIZE):
    """
    Resize image to optimal resolution for vision models (2025 best practice).

    Research shows vision models perform best with:
    - Resolution: ~2048x2048 pixels
    - Aspect ratio: Close to 1:1
    - Format: JPEG with high quality

    Args:
        base64_string: Base64-encoded image
        target_size: Target resolution (default: 2048)

    Returns:
        Resized image as base64 string
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

        # Calculate new dimensions maintaining aspect ratio
        # Target: Square-ish aspect ratio (close to 1:1) with max dimension = target_size
        if orig_width > orig_height:
            # Landscape: make width = target_size, scale height proportionally
            new_width = target_size
            new_height = int((orig_height / orig_width) * target_size)
        else:
            # Portrait or square: make height = target_size, scale width proportionally
            new_height = target_size
            new_width = int((orig_width / orig_height) * target_size)

        # Resize image with high-quality resampling
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to RGB if necessary (remove alpha channel)
        if resized_img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', resized_img.size, (255, 255, 255))
            if resized_img.mode == 'P':
                resized_img = resized_img.convert('RGBA')
            background.paste(resized_img, mask=resized_img.split()[-1] if resized_img.mode == 'RGBA' else None)
            resized_img = background

        # Save to bytes buffer as JPEG
        buffer = BytesIO()
        resized_img.save(buffer, format='JPEG', quality=IMAGE_QUALITY, optimize=True)
        resized_bytes = buffer.getvalue()

        # Encode back to base64
        resized_base64 = base64.b64encode(resized_bytes).decode('utf-8')

        return resized_base64

    except Exception as e:
        # If resize fails, return original
        logging.warning(f"Image resize failed: {e}. Using original image.")
        if ',' in base64_string and base64_string.startswith('data:'):
            return base64_string.split(',', 1)[1]
        return base64_string


def prepare_base64_image(base64_string, resize=None):
    """
    Prepare base64 string for Ollama OpenAI-compatible API.

    Args:
        base64_string: Base64-encoded image
        resize: Whether to resize image (default: uses RESIZE_IMAGES config)

    Returns:
        Data URI formatted string
    """
    # Use global config if not specified
    if resize is None:
        resize = RESIZE_IMAGES

    # Resize image if enabled
    if resize:
        base64_string = resize_image_for_vision_model(base64_string)

    # Remove existing data URI prefix if present
    if ',' in base64_string and base64_string.startswith('data:'):
        base64_string = base64_string.split(',', 1)[1]

    # Return with proper data URI prefix
    return f"data:image/jpeg;base64,{base64_string}"


def save_base64_as_jpg(base64_string, chunk_id, image_index=0, resized=False):
    """
    Save base64-encoded image as JPG file for manual quality review.

    Args:
        base64_string: Base64-encoded image data (already resized if resized=True)
        chunk_id: Chunk identifier for filename
        image_index: Index of image within chunk (default 0)
        resized: Whether this is the resized version (affects filename)

    Returns:
        Path to saved file, or None if saving disabled
    """
    if not SAVE_JPG:
        return None

    # Create temp_images_rag_ollama directory if it doesn't exist
    output_dir = 'temp_images_rag_ollama'
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

    # Generate filename with dimensions
    suffix = "_resized" if resized else "_original"
    filename = f"{chunk_id}_img{image_index}{suffix}{dim_suffix}.jpg"
    filepath = os.path.join(output_dir, filename)

    # Save to file
    try:
        with open(filepath, 'wb') as f:
            f.write(image_binary)
        return filepath
    except Exception as e:
        return None


def process_image_with_ollama(image_data_uri, prompt, client, logger):
    """
    Send image to Ollama API with retry logic.
    Returns extracted text description of visual elements.

    Uses research-backed parameters:
    - Temperature 0.1 for consistent, factual extraction
    - High max_tokens for structured markdown output
    - Seed for reproducibility
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
                temperature=TEMPERATURE,  # Low temp for factual consistency (best practice)
                max_tokens=MAX_TOKENS,    # Enough for detailed structured markdown
                seed=SEED                  # Reproducible results
            )

            # Extract and validate response
            content = response.choices[0].message.content

            # Debug logging
            logger.debug(f"API Response - Model: {MODEL}")
            logger.debug(f"API Response - Temperature: {TEMPERATURE}")
            logger.debug(f"API Response - Max Tokens: {MAX_TOKENS}")
            logger.debug(f"API Response - Content length: {len(content) if content else 0}")
            logger.debug(f"API Response - Content preview: {content[:200] if content else 'EMPTY/NULL'}")
            logger.debug(f"API Response - Finish reason: {response.choices[0].finish_reason}")

            # Validate response
            if content and "No significant visual elements detected" in content:
                logger.debug("Model detected text-only page")

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
            # Prepare image for vision model (uses native PDF extraction resolution)
            image_data_uri = prepare_base64_image(images[0])

            # Save JPG for manual quality review if enabled
            if SAVE_JPG:
                # Extract base64 from data URI
                img_b64 = image_data_uri.split(',', 1)[1] if ',' in image_data_uri else image_data_uri
                saved_path = save_base64_as_jpg(img_b64, chunk_id, image_index=0, resized=False)
                if saved_path:
                    logger.info(f"[{chunk_id}] Saved image: {saved_path}")

            result = process_image_with_ollama(image_data_uri, prompt, client, logger)
            time.sleep(REQUEST_DELAY)  # Rate limit delay
            return result
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return f"[Image processing failed: {str(e)}]"

    # Multiple images: process each, combine with separators
    results = []
    for i, img_b64 in enumerate(images):
        try:
            # Prepare image for vision model (uses native PDF extraction resolution)
            image_data_uri = prepare_base64_image(img_b64)

            # Save JPG for manual quality review if enabled
            if SAVE_JPG:
                img_b64_clean = image_data_uri.split(',', 1)[1] if ',' in image_data_uri else image_data_uri
                saved_path = save_base64_as_jpg(img_b64_clean, chunk_id, image_index=i, resized=False)
                if saved_path:
                    logger.info(f"[{chunk_id}] Saved image {i+1}: {saved_path}")

            result = process_image_with_ollama(image_data_uri, prompt, client, logger)
            results.append(f"[Image {i+1}]: {result}")
            time.sleep(REQUEST_DELAY)  # Rate limit delay
        except Exception as e:
            logger.error(f"Failed to process image {i+1}: {e}")
            results.append(f"[Image {i+1}]: [Error: {str(e)}]")

    return "\n\n".join(results)


def main():
    parser = argparse.ArgumentParser(
        description='Vision RAG - Ollama Image Understanding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vision_rag_ollama.py
    python vision_rag_ollama.py --input output/output_parsed.json
    python vision_rag_ollama.py --start-from 5
    python vision_rag_ollama.py --dry-run
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

    logger.info(f"Starting Vision RAG processing for {total_pages} pages")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Parameters: temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}, seed={SEED}")
    if RESIZE_IMAGES:
        logger.info(f"Image preprocessing: resize to {TARGET_SIZE}px (aspect ratio maintained)")
    else:
        logger.info(f"Image preprocessing: DISABLED - using native PDF extraction resolution")
    if args.start_from > 1:
        logger.info(f"Resuming from chunk {args.start_from}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No API calls will be made")
    if SAVE_JPG:
        logger.info("Image saving ENABLED - JPG files will be saved to temp_images_rag_ollama/")
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
            # Process chunk images with Ollama
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
