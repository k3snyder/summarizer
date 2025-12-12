import os
import re
import argparse
import logging
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, OpenAIError
import time
import json

# Load environment variables from .env file
load_dotenv()

# Configuration
OLLAMA_BASE_URL_1 = os.getenv('OLLAMA_BASE_URL_1', 'http://localhost:11434/v1')
OLLAMA_BASE_URL_2 = os.getenv('OLLAMA_BASE_URL_2', 'http://localhost:11434/v1') # Optional secondary Ollama server
MODEL_TIER_1 = 'ministral-3:latest'  # Primary model for summarization (attempts 1-10)
MODEL_TIER_2 = 'ministral-3:latest'  # Fallback model (attempts 11-20)
MODEL_TIER_3 = 'ministral-3:latest'  # Validator and Final fallback model (attempts 21-30)
MAX_RETRIES = 3  # Retries per API call
RETRY_DELAY = 5  # Seconds between retries
QUALITY_THRESHOLD_HIGH = 90  # Initial quality threshold (%)
QUALITY_THRESHOLD_LOW = 85   # Reduced quality threshold after failures (%)
DEBUG_LOG = False  # (True or False) Enable detailed logging of prompts and responses for manual review

# Initialize OpenAI clients
client1 = OpenAI(base_url=OLLAMA_BASE_URL_1, api_key='ollama')
client2 = OpenAI(base_url=OLLAMA_BASE_URL_2, api_key='ollama')

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def save_json_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(content, outfile, indent=2)

def gpt_prompt1(prompt, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Primary model tier (client1) - attempts 1-10"""
    for attempt in range(max_retries):
        try:
            response = client1.chat.completions.create(
                model=MODEL_TIER_1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            if attempt < max_retries - 1:
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Error: {e}")
                return None

def gpt_prompt2(prompt, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Secondary model tier (client1) - attempts 11-20"""
    for attempt in range(max_retries):
        try:
            response = client1.chat.completions.create(
                model=MODEL_TIER_2,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            if attempt < max_retries - 1:
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Error: {e}")
                return None

def gpt_prompt3(prompt, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Tertiary model tier (client2) - attempts 21-30"""
    for attempt in range(max_retries):
        try:
            response = client2.chat.completions.create(
                model=MODEL_TIER_3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            if attempt < max_retries - 1:
                print(f"API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Error: {e}")
                return None


def quality_check(chunk, summary):
    prompt = f'''Compare the following original text and summary.

    On a scale from 0% to 100%, how accurately does the summary represent the original text?

    Original Text:
    """{chunk}"""


    Summary:
    """{summary}"""


    Provide only the percentage number.'''


    response = gpt_prompt3(prompt)
    # Remove any non-digit characters and convert to float
    try:
        return float(''.join(filter(str.isdigit, response)))
    except ValueError:
        return 0.0  # Return 0 if the response is invalid

def setup_logger():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename='logs/summarizer-debug.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

def clean_notes_output(output):
    # Remove any XML tags like <summary> and extract bullet points as array
    text = output.strip()
    # Remove <summary> and </summary> tags
    text = re.sub(r'</?summary>', '', text)
    # Extract lines that are bullet points, strip the "* " prefix
    lines = text.strip().split('\n')
    bullet_points = [line.strip()[2:] for line in lines if line.strip().startswith('* ')]
    return bullet_points

def clean_topics_output(output):
    # Clean and normalize comma-separated topics, return as array
    text = output.strip()
    # If there are newlines, take the last line (likely the actual topics)
    if '\n' in text:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # Find the line that looks like comma-separated topics
        for line in reversed(lines):
            if ',' in line and not line.endswith(':'):
                text = line
                break
    # Split by comma and clean each topic
    topics = [t.strip() for t in text.split(',') if t.strip()]
    return topics

def build_page_context(page: dict) -> str:
    """
    Build a comprehensive context string from page data.
    Combines text, tables, and image_text (from Vision OCR) into a single context.

    Args:
        page: Page dict with text, tables, and optional image_text

    Returns:
        Combined context string for summarization
    """
    sections = []

    # 1. Main text content
    text = page.get('text', '').strip()
    if text:
        sections.append(text)

    # 2. Table data (convert to readable format)
    tables = page.get('tables', [])
    if tables:
        table_texts = []
        for i, table in enumerate(tables):
            columns = table.get('columns', [])
            data = table.get('data', [])
            if columns and data:
                # Format as simple text table
                header = ' | '.join(str(c) for c in columns if c)
                rows = [' | '.join(str(cell) for cell in row) for row in data]
                table_text = f"Table {i+1}:\n{header}\n" + "\n".join(rows)
                table_texts.append(table_text)
        if table_texts:
            sections.append("\n\n[TABLES]\n" + "\n\n".join(table_texts))

    # 3. Vision OCR content (visual elements from page images)
    image_text = page.get('image_text', '')
    if image_text and image_text.strip():
        sections.append("\n\n[VISUAL CONTENT]\n" + image_text.strip())

    return "\n\n".join(sections)


def generate_summary_and_topics(text, logger, mode='full'):
    """
    Generate summary and topics for a given text.

    Args:
        text: Text to process
        logger: Logger instance
        mode: 'full' (summary+topics), 'topics-only', or 'skip' (no processing)

    Returns: (notes, topics, relevancy_percentage)
    """
    # Skip mode - return None for everything
    if mode == 'skip':
        return None, None, 0

    # Load prompt templates
    notes_prompt_template = open_file("./prompts/summarizer-notes-prompt.txt")

    notes = None
    topics = None
    relevancy_percentage = 0

    # Topics-only mode - skip summarization, go straight to topics
    if mode == 'topics-only':
        # Generate topics directly from the text (no notes step)
        keytw_template = open_file("./prompts/summarizer-topics-prompt.txt")
        # For topics-only, we use the raw text instead of notes
        keytw_modified = keytw_template.replace("<<NOTES>>", text[:2000])  # Limit context

        topic_attempt = 0
        max_topic_attempts = 3

        while topic_attempt < max_topic_attempts:
            if topic_attempt == 0:
                topics_response = gpt_prompt1(keytw_modified)
            elif topic_attempt == 1:
                topics_response = gpt_prompt2(keytw_modified)
            else:
                topics_response = gpt_prompt3(keytw_modified)

            if topics_response is not None:
                topics = clean_topics_output(topics_response)
                break

            topic_attempt += 1
            print(f"Failed to generate topics, attempt {topic_attempt}.")

        return None, topics, 0  # No notes, no relevancy check in topics-only mode

    # Full mode - generate both summary and topics
    attempt = 0
    max_attempts = 30
    relevancy_threshold = 90.0

    while attempt < max_attempts:
        attempt += 1

        # Adjust relevancy threshold after 5 attempts
        if attempt > 5:
            relevancy_threshold = 85.0

        prompt_modified = notes_prompt_template.format(chunk=text)

        # Debug logging: Log the prompt being sent
        if DEBUG_LOG:
            debug_log_path = './logs/summarizer-debug-prompts.log'
            os.makedirs('./logs', exist_ok=True)
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"ATTEMPT {attempt} - NOTES PROMPT\n")
                f.write(f"{'='*80}\n")
                f.write(prompt_modified)
                f.write(f"\n{'='*80}\n\n")

        # Use different models for different attempts
        if attempt <= 10:
            notes_response = gpt_prompt1(prompt_modified)
        elif attempt <= 20:
            notes_response = gpt_prompt2(prompt_modified)
        else:
            notes_response = gpt_prompt3(prompt_modified)

        # Debug logging: Log the response received
        if DEBUG_LOG and notes_response:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(f"ATTEMPT {attempt} - NOTES RESPONSE\n")
                f.write(f"{'='*80}\n")
                f.write(notes_response)
                f.write(f"\n{'='*80}\n\n")

        if notes_response is None:
            print(f"Failed to get response for attempt {attempt}. Continuing to next attempt.")
            continue

        relevancy_percentage = quality_check(text, notes_response)
        print(f"Relevancy Accuracy for Notes (Attempt {attempt}): {relevancy_percentage}%")

        if relevancy_percentage >= relevancy_threshold:
            notes = clean_notes_output(notes_response)
            break

    # Generate topics if notes were successfully generated
    if notes is not None:
        keytw_template = open_file("./prompts/summarizer-topics-prompt.txt")
        # Convert notes array back to string for the topics prompt
        notes_text = '\n'.join(f'* {note}' for note in notes)
        keytw_modified = keytw_template.replace("<<NOTES>>", notes_text)

        topic_attempt = 0
        max_topic_attempts = 3

        while topic_attempt < max_topic_attempts:
            # Debug logging: Log the topics prompt being sent
            if DEBUG_LOG and topic_attempt == 0:
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"TOPICS PROMPT (Attempt {topic_attempt + 1})\n")
                    f.write(f"{'='*80}\n")
                    f.write(keytw_modified)
                    f.write(f"\n{'='*80}\n\n")

            if topic_attempt == 0:
                topics_response = gpt_prompt1(keytw_modified)
            elif topic_attempt == 1:
                topics_response = gpt_prompt2(keytw_modified)
            else:
                topics_response = gpt_prompt3(keytw_modified)

            # Debug logging: Log the topics response received
            if DEBUG_LOG and topics_response:
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"TOPICS RESPONSE (Attempt {topic_attempt + 1})\n")
                    f.write(f"{'='*80}\n")
                    f.write(topics_response)
                    f.write(f"\n{'='*80}\n\n")

            if topics_response is not None:
                topics = clean_topics_output(topics_response)
                break

            topic_attempt += 1
            print(f"Failed to generate topics, attempt {topic_attempt}.")
            
    return notes, topics, relevancy_percentage

def process_json_mode(logger, json_path, summarizer_mode='full'):
    """
    Process PDF parser JSON output mode.

    Args:
        logger: Logger instance
        json_path: Path to input JSON
        summarizer_mode: 'full' (summary+topics), 'topics-only', or 'skip'
    """
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        logger.error(f"JSON file not found: {json_path}")
        return

    mode_desc = {
        'full': 'Full summarization (notes + topics)',
        'topics-only': 'Topics extraction only',
        'skip': 'No summarization'
    }

    print(f"\nProcessing PDF parser JSON output: {json_path}")
    print(f"Summarizer mode: {mode_desc.get(summarizer_mode, summarizer_mode)}")
    logger.info(f"Processing PDF parser JSON output: {json_path}")
    logger.info(f"Summarizer mode: {summarizer_mode}")

    # Load PDF parser JSON
    pdf_data = open_json_file(json_path)

    # Extract document metadata (supports both 'document' and legacy 'pdf_document' keys)
    document = pdf_data.get('document') or pdf_data.get('pdf_document', {})
    filename = document.get('filename', 'unknown.pdf')
    total_pages = document.get('total_pages', 0)

    print(f"Document: {filename} ({total_pages} pages)")
    logger.info(f"Processing document: {filename} with {total_pages} pages")

    pages = pdf_data.get('pages', [])
    
    # Process each page
    for i, page in enumerate(pages):
        chunk_id = page.get('chunk_id') or page.get('page_id', f'chunk_{i+1}')
        print(f"\nProcessing chunk {i+1}/{total_pages} ({chunk_id})...")

        # Build comprehensive context from text, tables, and image_text
        page_context = build_page_context(page)
        if not page_context.strip():
            print(f"Skipping chunk {chunk_id} - no content.")
            continue

        # Log if visual content is included
        if page.get('image_text'):
            print(f"  Including visual content from Vision OCR")

        # Debug logging: Log chunk header
        if DEBUG_LOG:
            debug_log_path = './logs/summarizer-debug-prompts.log'
            os.makedirs('./logs', exist_ok=True)
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(f"\n\n{'#'*80}\n")
                f.write(f"# PROCESSING CHUNK {i+1}/{total_pages}: {chunk_id}\n")
                f.write(f"# Document: {filename}\n")
                f.write(f"# Has visual content: {bool(page.get('image_text'))}\n")
                f.write(f"{'#'*80}\n\n")
                f.write(f"COMPREHENSIVE PAGE CONTEXT (sent to LLM):\n")
                f.write(f"{'='*80}\n")
                f.write(page_context)
                f.write(f"\n{'='*80}\n\n")

        # Generate summary and topics for the page (based on mode)
        notes, topics, relevancy = generate_summary_and_topics(page_context, logger, mode=summarizer_mode)

        # Enrich page object with metadata
        if summarizer_mode != 'skip':
            page['summary_notes'] = notes
            page['summary_topics'] = topics
            page['summary_relevancy'] = relevancy
        else:
            # Skip mode - just mark as skipped
            page['summary_notes'] = None
            page['summary_topics'] = None
            page['summary_relevancy'] = 0
        
        # Save incrementally (save the whole updated structure)
        output_filename = os.path.splitext(filename)[0] + '_enriched.json'
        output_path = os.path.join('./output', output_filename)
        save_json_file(output_path, pdf_data)

    print("\nProcessing complete! Enriched JSON saved.")
    logger.info("Processing complete! Enriched JSON saved.")

def main():
    parser = argparse.ArgumentParser(
        description="RAG Summarizer - Process documents into structured summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Summarizer Modes:
  full (default)   - Generate both summary notes and topics
  topics-only      - Extract topics only, skip summarization (faster)
  skip             - Skip all summarization (just pass through)

Examples:
  # Full summarization (notes + topics)
  python summarizer.py --mode json --json-input output/output_parsed.json

  # Topics only (faster, no quality checks)
  python summarizer.py --mode json --summarizer-mode topics-only

  # Skip summarizer entirely
  python summarizer.py --mode json --summarizer-mode skip
        """
    )
    parser.add_argument(
        '--mode',
        choices=['json'],
        default='json',
        help='Processing mode: json (PDF parser output). Legacy modes (text, pdf) removed.'
    )
    parser.add_argument(
        '--json-input',
        type=str,
        default='./output/output_parsed.json',
        help='Path to PDF parser JSON output'
    )
    parser.add_argument(
        '--summarizer-mode',
        choices=['full', 'topics-only', 'skip'],
        default='full',
        help='Summarizer processing mode (default: full)'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()

    # Run JSON mode
    if args.mode == 'json':
        process_json_mode(logger, args.json_input, summarizer_mode=args.summarizer_mode)
    else:
        print(f"Unsupported mode: {args.mode}. Only 'json' mode is supported.")

if __name__ == "__main__":
    main()