# Document Summarizer Pipeline

A privacy-first, locally-hosted document intelligence system that transforms PDF documents and text transcripts into structured summaries using a multi-stage pipeline with AI-powered analysis via local and cloud LLM models.

## Features

- **Multi-format support**: PDF, text files (.txt), markdown files (.md)
- **Advanced vision processing**: Extract content from charts, graphs, diagrams, and tables
- **Multiple LLM providers**: Ollama (local), OpenAI, Google Gemini, DeepSeek
- **Configurable PDF image quality**: User-selectable DPI (72/144/200/300)
- **Flexible summarization modes**: Full (notes + topics), topics-only, or skip
- **Privacy-first**: Local processing option with Ollama
- **Intelligent classification**: Pre-filter pages to identify visual content
- **Interactive CLI**: Guided configuration with smart defaults

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 1: Content Extraction                                          │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ PDF Parser (pdf_parser/)                                         │ │
│ │ • Multi-strategy table extraction (pdfplumber + Camelot)         │ │
│ │ • Chart data recognition with percentage extraction              │ │
│ │ • OCR fallback with Tesseract                                   │ │
│ │ • Full-page image capture (user-configurable 72-300 DPI)        │ │
│ │ • Cross-page table merging                                       │ │
│ │ Input:  PDF files                                                │ │
│ │ Output: output/output_parsed.json                                │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ Text Parser (text_parser/)                                       │ │
│ │ • Direct text processing for .txt and .md files                  │ │
│ │ • Preserves document structure                                   │ │
│ │ Input:  Text/Markdown files                                      │ │
│ │ Output: output/output_parsed.json                                │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 2: Vision Processing (Optional, PDF only)                      │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ A) Page Classification (vision_classifier_*)                     │ │
│ │    • Quick detection of pages with visual content                │ │
│ │    • Downsized to 72 DPI for speed (3-5x faster)                 │ │
│ │    • Filters text-only pages                                     │ │
│ │    Providers: Ollama (local), Gemini, OpenAI                     │ │
│ └──────────────────────────────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────────────────┐ │
│ │ B) Visual Content Extraction (vision_rag_* or vision_ocr_*)      │ │
│ │    Vision RAG: Structured extraction of charts/graphs/diagrams   │ │
│ │    • Native resolution (uses PDF extraction DPI)                 │ │
│ │    • Research-backed prompting (2025 best practices)             │ │
│ │    • Anti-hallucination with confidence markers                  │ │
│ │    Providers: Ollama (local), Gemini, OpenAI                     │ │
│ │                                                                   │ │
│ │    Vision OCR: Text extraction from images                       │ │
│ │    • DeepSeek-powered OCR                                        │ │
│ │    • Handles scanned documents                                   │ │
│ └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 3: Summarization (summarizer.py)                               │
│ • Text chunking with overlap                                         │
│ • Multi-attempt summarization (up to 30 attempts)                    │
│ • Quality validation (85-90% threshold)                              │
│ • Model rotation across attempts                                     │
│ • Three modes: full, topics-only, skip                               │
│ Input:  output/output_parsed.json or output/output_vision_*.json     │
│ Output: output/*_enriched.json                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements
- Python 3.8+
- Tesseract OCR (for PDF OCR fallback)
- At least one of:
  - Ollama (for local, privacy-first processing)
  - OpenAI API key
  - Google Gemini API key
  - DeepSeek API access

### Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### Install Ollama (Optional, for local processing)

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull vision models:
```bash
# For classification and extraction (recommended)
ollama pull ministral-3:14b

# Alternative vision models
ollama pull llama3.2-vision:11b
ollama pull llama3.2-vision:90b
ollama pull gemma3:12b

# For summarization
ollama pull gemma2:9b-instruct-q8_0
ollama pull gemma2:latest
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd summarizer-dev
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment (optional):
```bash
cp .example.env .env
# Edit .env with your API keys and server endpoints
```

## Usage

### Interactive Mode (Recommended)

Launch the interactive wizard with guided configuration:

```bash
python run.py
```

The wizard will guide you through:
1. **File Selection**: Choose a file or batch mode
2. **Extraction Config** (PDF only): Select what to extract (tables, images)
3. **PDF Image Quality** (if extracting images): Choose DPI (72/144/200/300)
4. **Vision Processing** (PDF only): Enable chart/graph extraction
5. **Pipeline Stages**: Full pipeline or extraction only
6. **Summarization Mode**: Full, topics-only, or skip

### Command Line Mode

**Process a single PDF:**
```bash
python run.py document.pdf
```

**Process with specific PDF image quality (recommended: 200 DPI for charts/graphs):**
```bash
python run.py document.pdf --pdf-image-dpi 200
```

**Process with vision extraction (Ollama - local, zero cost):**
```bash
python run.py document.pdf --vision-mode ollama --pdf-image-dpi 200
```

**Process with vision extraction (cloud providers):**
```bash
python run.py document.pdf --vision-mode gemini --pdf-image-dpi 200
python run.py document.pdf --vision-mode openai --pdf-image-dpi 200
python run.py document.pdf --vision-mode deepseek  # DeepSeek OCR
```

**Fast processing (topics only, no quality checks):**
```bash
python run.py document.pdf --summarizer-mode topics-only
```

**Batch process all PDFs in `./input` directory:**
```bash
python run.py --batch
```

**Process text or markdown files:**
```bash
python run.py transcript.txt
python run.py notes.md
```

**Extract only (skip summarization):**
```bash
python run.py document.pdf --extract-only
```

**Text-only extraction (fastest, skip tables and images):**
```bash
python run.py document.pdf --text-only
```

### Advanced Options

**Selective extraction:**
```bash
# Skip table extraction
python run.py document.pdf --skip-tables

# Skip image extraction
python run.py document.pdf --skip-images

# Combine options
python run.py document.pdf --pdf-image-dpi 300 --vision-mode ollama --summarizer-mode full
```

### Individual Components

**Run PDF parser directly:**
```bash
python pdf_parser/main.py ./input/document.pdf
```

**Run summarizer directly:**
```bash
# Full summarization (notes + topics)
python summarizer.py --mode json --json-input output/output_parsed.json

# Topics only (faster)
python summarizer.py --mode json --summarizer-mode topics-only

# Skip summarization
python summarizer.py --mode json --summarizer-mode skip
```

**Run vision classification (Ollama):**
```bash
python vision_classifier_ollama.py --input output/output_parsed.json
```

**Run vision extraction (Ollama):**
```bash
python vision_rag_ollama.py --input output/output_parsed.json
```

## Directory Structure

```
summarizer-dev/
├── run.py                      # Main pipeline orchestrator (interactive + CLI)
├── summarizer.py               # AI summarization engine
│
├── pdf_parser/                 # PDF extraction module
│   ├── main.py                 # PDF parser entry point
│   ├── parsers.py              # Multi-strategy extraction logic
│   ├── models.py               # Pydantic data models
│   ├── config.py               # Configuration management
│   ├── utils.py                # Image conversion helpers
│   └── exceptions.py           # Custom exceptions
│
├── text_parser/                # Text/Markdown parser
│   └── main.py                 # Text parser entry point
│
├── vision_classifier_*.py      # Page classification (detect visuals)
│   ├── vision_classifier_ollama.py     # Local Ollama (recommended)
│   ├── vision_classifier_gemini.py     # Google Gemini
│   └── vision_classifier_openai.py     # OpenAI GPT-4V
│
├── vision_rag_*.py             # Visual content extraction
│   ├── vision_rag_ollama.py            # Local Ollama (recommended)
│   ├── vision_rag_gemini.py            # Google Gemini
│   └── vision_rag_openai.py            # OpenAI GPT-4V
│
├── vision_ocr_deepseek.py      # DeepSeek OCR for scanned documents
│
├── prompts/                    # LLM prompt templates
│   ├── summarizer-notes-prompt.txt     # Primary summarization
│   ├── summarizer-topics-prompt.txt    # Topic extraction
│   ├── vision-classifier.txt           # Visual content detection
│   └── vision-extract.txt              # Structured visual extraction
│
├── input/                      # Place source files here
├── output/                     # Processed outputs
│   ├── output_parsed.json              # PDF/text extraction results
│   ├── output_vision_ollama.json       # Vision extraction results
│   └── *_enriched.json                 # Final summarized output
│
├── logs/                       # Processing logs
│   ├── summarizer-debug.log
│   ├── vision-rag-ollama.log
│   └── vision-classifier-ollama.log
│
├── archive/                    # Legacy/deprecated code
└── requirements.txt            # Python dependencies
```

## Core Features

### PDF Parser
- **Multi-strategy table extraction**: pdfplumber strategies + Camelot fallback
- **Chart data recognition**: Extracts percentages and data from bar charts
- **Cross-page table merging**: Automatic detection and merging
- **OCR fallback**: Tesseract for scanned/image-only PDFs
- **Configurable image quality**: User-selectable DPI (72/144/200/300)
- **Concurrent processing**: Async page processing for performance

### Vision Processing

**Classification (Pre-filter):**
- Quickly identifies pages with charts, graphs, diagrams
- Downsized to 72 DPI for 3-5x faster classification
- Filters out text-only pages (saves processing time)
- Providers: Ollama (local), Gemini, OpenAI

**Visual Content Extraction:**
- Structured Markdown output with explicit sections
- Extracts data from charts, graphs, tables, diagrams
- Research-backed prompting (2025 best practices)
- Anti-hallucination with confidence markers
- Native resolution processing (no quality loss)
- Providers: Ollama (local), Gemini, OpenAI, DeepSeek (OCR)

### Summarization
- **Three modes**:
  - **Full**: Summary notes + topics with quality validation (default)
  - **Topics-only**: Fast topic extraction, no quality checks (~10x faster)
  - **Skip**: Pass-through mode (no summarization)
- **Quality-driven**: Multi-attempt (up to 30) with 85-90% relevancy thresholds
- **Model rotation**: Graduated fallback across 3 model tiers
- **Adaptive thresholds**: 90% initially, drops to 85% after 5 attempts
- **Incremental saving**: Progress saved after each chunk

## Output Formats

### Parser Output (`output_parsed.json`)
```json
{
  "document": {
    "document_id": "doc_filename.pdf",
    "filename": "filename.pdf",
    "total_pages": 10,
    "metadata": {}
  },
  "pages": [
    {
      "chunk_id": "chunk_1",
      "doc_title": "filename.pdf",
      "text": "Extracted text content...",
      "tables": [
        {
          "columns": ["Column1", "Column2"],
          "data": [["Row1Col1", "Row1Col2"]],
          "extends_to_bottom": false
        }
      ],
      "image_base64": ["base64_encoded_page_image"]
    }
  ]
}
```

### Vision-Enhanced Output (`output_vision_ollama.json`)
```json
{
  "pages": [
    {
      "chunk_id": "chunk_1",
      "text": "...",
      "image_classifier": true,
      "image_text": "## Visual Elements Summary\n[Structured markdown extraction]"
    }
  ]
}
```

### Final Enriched Output (`*_enriched.json`)
```json
{
  "pages": [
    {
      "chunk_id": "chunk_1",
      "text": "...",
      "image_text": "...",
      "summary_notes": ["Note 1", "Note 2", "Note 3"],
      "summary_topics": ["Topic 1", "Topic 2"],
      "summary_relevancy": 92.0
    }
  ]
}
```

## Configuration

### PDF Image Quality

Choose resolution when extracting PDF images:

| DPI | Resolution (Letter) | Best For | File Size |
|-----|-------------------|----------|-----------|
| 72  | 612 × 792 | Text-heavy documents, quick testing | ~50 KB |
| 144 | 1224 × 1584 | General documents (default) | ~150 KB |
| 200 | 1700 × 2200 | **Charts, graphs, small text** (recommended) | ~300 KB |
| 300 | 2550 × 3300 | Complex diagrams, maximum quality | ~650 KB |

**Recommendation**: Use 200-300 DPI for documents with charts/graphs.

### Summarizer Modes

| Mode | Speed | Output | Quality Checks | Best For |
|------|-------|--------|----------------|----------|
| **full** | Slow | Notes + Topics | ✅ 85-90% | High-quality summaries |
| **topics-only** | Fast (~10x) | Topics only | ❌ None | Quick categorization |
| **skip** | Instant | None | ❌ None | Extraction-only workflows |

### Vision Processing Providers

| Provider | Cost | Speed | Quality | Privacy |
|----------|------|-------|---------|---------|
| **Ollama** | Free | Medium | Good | ✅ Local |
| Gemini | Paid | Fast | Excellent | ❌ Cloud |
| OpenAI | Paid | Fast | Excellent | ❌ Cloud |
| DeepSeek | Paid | Fast | Good (OCR) | ❌ Cloud |

### Environment Variables

```bash
# Ollama servers
OLLAMA_BASE_URL=http://localhost:11434/v1

# PDF parser
PDF_IMAGE_DPI=200              # Default PDF image DPI
PDF_OCR_RESOLUTION=150         # OCR resolution
PDF_MAX_CONCURRENT_PAGES=4     # Parallel processing

# Vision models
VISION_MODEL=ministral-3:14b   # Ollama vision model

# API keys (if using cloud providers)
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

## Privacy & Security

**Local Processing (Ollama):**
- ✅ All processing on local/private servers
- ✅ No cloud dependencies
- ✅ Data never leaves your network
- ✅ Network isolation (private IPs)
- ✅ Zero API costs

**Cloud Processing (Gemini/OpenAI/DeepSeek):**
- ⚠️ Data sent to external APIs
- ⚠️ Review provider terms of service
- ⚠️ API usage costs apply
- ℹ️ Typically faster and higher quality

## Development

### Adding Custom Prompts

Edit prompt templates in `./prompts/`:
- `summarizer-notes-prompt.txt` - Primary summarization
- `summarizer-topics-prompt.txt` - Topic extraction
- `vision-classifier.txt` - Visual content detection
- `vision-extract.txt` - Structured visual extraction (research-backed, 2025 best practices)

### Vision Model Selection

**Ollama models** (`vision_rag_ollama.py`):
```python
MODEL = os.getenv('VISION_MODEL', 'ministral-3:14b')  # Recommended
# Alternatives: gemma3:12b
```

**Configuration**:
```bash
VISION_MODEL=gemma3:12b python vision_rag_ollama.py
```

## License

[Your License Here]

## Acknowledgments

- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF parsing
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Text recognition
- [pdf2image](https://github.com/Belval/pdf2image) - Page rendering
- [Camelot](https://github.com/camelot-dev/camelot) - Table extraction
- [LangChain](https://github.com/langchain-ai/langchain) - Text splitting
- [Ollama](https://ollama.ai/) - Local LLM inference
- [OpenAI](https://openai.com/) - GPT-4 Vision
- [Google Gemini](https://ai.google.dev/) - Gemini Vision
- [Pillow (PIL)](https://python-pillow.org/) - Image processing
