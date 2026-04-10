# AI-powered Document Intelligence Pipeline

Extracts and structures document content for AI-ready consumption. A privacy-first, locally-hosted intelligence system that transforms PDF, PPTX, and text documents into structured data with text, tables, visual content, and summarization.

## Features

- **Multi-Format Support** - PDF, PPTX, TXT, and Markdown files
- **Vision Processing** - Intelligent classification and extraction of diagrams, screenshots, photos, and visually meaningful tables
- **Quality-Validated Summarization** - Multi-attempt processing with relevancy thresholds
- **Privacy-First** - Local processing with llama.cpp or Ollama, zero cloud exposure option
- **Multi-Provider** - llama.cpp (local), Ollama (fallback local), OpenAI, Gemini, DeepSeek, Codex CLI, Claude CLI
- **Detailed Extraction** - Optional 3x extraction with synthesis for comprehensive coverage
- **Web UI + REST API** - Next.js frontend with FastAPI backend

## Architecture

```
STAGE 1: EXTRACTION → STAGE 2: VISION → STAGE 3: SUMMARIZATION
     PDF/PPTX/TXT        Classify & Extract      Notes & Topics
```

The pipeline produces a unified JSON schema at each stage:

```json
{
  "document": { "document_id", "filename", "total_pages", "metadata" },
  "pages": [
    {
      "chunk_id", "text", "tables",
      "image_base64", "image_text", "image_classifier",
      "summary_notes", "summary_topics", "summary_relevancy"
    }
  ]
}
```

## Requirements

- Python 3.8+
- Node.js 18+
- Tesseract OCR
- Poppler (for PDF rendering)
- LibreOffice (for PPTX processing)
- llama.cpp server (recommended local default)
- Ollama (optional fallback local LLM)
- Codex CLI (optional, for Codex provider)
- Claude CLI (optional, for Claude CLI provider)

### macOS

```bash
brew install tesseract poppler libreoffice
```

### Ubuntu/Debian

```bash
sudo apt-get install tesseract-ocr poppler-utils libreoffice
```

### CLI Tools (Optional)

```bash
# Codex CLI
npm install -g @openai/codex

# Claude CLI
npm install -g @anthropic-ai/claude-code
```

## Installation

### Backend

```bash
cd backend
python -m venv venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` file:

```env
# Server
HOST=0.0.0.0
PORT=8000

# Local defaults
VISION_PROVIDER=llama_cpp
SUMMARIZER_PROVIDER=llama_cpp

# llama.cpp text + vision split
LLAMA_CPP_BASE_URL=http://localhost:11440/v1
LLAMA_CPP_VISION_BASE_URL=http://localhost:11439/v1
LLAMA_CPP_MODEL=model.gguf
LLAMA_CPP_VISION_MODEL=model.gguf
LLAMA_CPP_API_KEY=sk-no-key-required

# Optional fallback Ollama
OLLAMA_BASE_URL=http://localhost:11434/v1

# Optional: Cloud providers
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Logging
LOG_LEVEL=INFO
```

### Frontend

```bash
cd frontend
npm install
```

Create `.env.local`:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## Usage

### Start Services

Use the root Makefile to bring both dev services up or down:

```bash
make services-up
make services-down
```

`make services-up` starts the FastAPI backend and Next.js frontend in the background, writes PID files under `.run/`, and logs to `.run/backend.log` and `.run/frontend.log`.

Manual startup is still available:

```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

### Recommended Local Topology

The current local-first setup is:

- `llama.cpp` on `11440` for primary text + summarization
- `llama.cpp` on `11439` for multimodal vision
- `Ollama` as an optional fallback local provider

This split lets summarization and vision run on separate local inference endpoints while keeping cloud providers optional.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/jobs` | Create processing job with file upload |
| GET | `/api/jobs/{job_id}` | Get job status and progress |
| GET | `/api/jobs/{job_id}/output` | Get processing results (JSON) |
| GET | `/api/jobs/{job_id}/output/markdown` | Get results as Markdown report |
| DELETE | `/api/jobs/{job_id}` | Cancel or delete job |
| GET | `/api/health` | Health check |

### Create a Job

```bash
curl -X POST http://localhost:8000/api/jobs \
  -F "file=@document.pdf" \
  -F 'config={"vision_mode":"llama_cpp","summarizer_provider":"llama_cpp"}'
```

### Pipeline Configuration

```json
{
  "extract_only": false,
  "skip_tables": false,
  "skip_images": false,
  "text_only": false,
  "pdf_image_dpi": 200,
  "vision_mode": "llama_cpp",
  "vision_classifier_mode": null,
  "vision_extractor_mode": null,
  "vision_cli_provider": null,
  "vision_detailed_extraction": false,
  "chunk_size": 3000,
  "chunk_overlap": 80,
  "run_summarization": true,
  "summarizer_mode": "full",
  "summarizer_provider": "llama_cpp",
  "summarizer_cli_provider": null,
  "summarizer_detailed_extraction": false,
  "summarizer_insight_mode": false,
  "keep_base64_images": false
}
```

**Vision Modes**: `none`, `llama_cpp`, `ollama`, `openai`, `gemini`, `deepseek`, `codex`, `claude`

**Vision Classifier / Extractor Modes**: `none`, `llama_cpp`, `ollama`, `openai`, `gemini`, `codex`, `claude`

**Vision CLI Providers**: `codex`, `claude` (used when `vision_mode` is `codex` or `claude`)

**Summarizer Modes**: `full` (notes + topics), `topics-only`, `skip`

**Summarizer Providers**: `llama_cpp`, `ollama`, `openai`, `codex`, `claude`

**Summarizer CLI Providers**: `codex`, `claude` (used when `summarizer_provider` is `codex` or `claude`)

**Detailed Extraction**: When enabled, runs extraction 3 times and synthesizes results for comprehensive coverage

**Vision Classification Policy**: The classifier is intentionally conservative about page furniture. Footer/header logos, page numbers, copyright lines, decorative cover backgrounds, branding accents, and simple table-of-contents leader lines should not trigger extraction by themselves. Pages are classified `YES` when they contain substantive visual information that would be lost with text-only extraction.

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Environment settings
│   │   ├── routers/             # API endpoints
│   │   ├── pipeline/            # Core processing
│   │   │   ├── orchestrator.py  # Pipeline coordination
│   │   │   ├── extraction/      # Document parsing
│   │   │   ├── vision/          # Vision processing
│   │   │   └── summarization/   # Summary generation
│   │   ├── models/              # Pydantic schemas
│   │   ├── services/            # Business logic
│   │   └── db/                  # Database layer
│   ├── prompts/                 # LLM prompt templates
│   └── requirements.txt
├── frontend/
│   ├── app/                     # Next.js pages
│   ├── components/              # React components
│   ├── types/                   # TypeScript definitions
│   └── package.json
```

## Tech Stack

### Backend
- FastAPI, Uvicorn
- pdfplumber, pytesseract, python-pptx, pdf2image
- OpenAI SDK, google-generativeai, LangChain
- SQLAlchemy 2.0, aiosqlite
- Pydantic 2.0

### Frontend
- Next.js 15, React 19
- Tailwind CSS v4, shadcn/ui
- TypeScript

### LLM Providers
- **llama.cpp** - Primary local inference for summarization and multimodal vision
- **Ollama** - Fallback local inference
- **OpenAI** - GPT-4.1 family
- **Gemini** - Google AI
- **DeepSeek** - Specialized OCR
- **Codex CLI** - OpenAI Codex (subprocess execution)
- **Claude CLI** - Anthropic Claude (subprocess execution)

## Development

## Environment Variables

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `VISION_PROVIDER` | `llama_cpp` | Default vision provider |
| `SUMMARIZER_PROVIDER` | `llama_cpp` | Default summarizer provider |
| `LLAMA_CPP_BASE_URL` | `http://localhost:11440/v1` | Primary llama.cpp text/summarization endpoint |
| `LLAMA_CPP_VISION_BASE_URL` | `http://localhost:11439/v1` | llama.cpp multimodal vision endpoint |
| `LLAMA_CPP_MODEL` | `model.gguf` | Default llama.cpp text model |
| `LLAMA_CPP_VISION_MODEL` | `model.gguf` | Default llama.cpp vision model |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Fallback Ollama API URL |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `LOG_LEVEL` | `INFO` | Logging level |
| `JOB_RETENTION_HOURS` | `24` | Job data retention |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Backend API URL |
