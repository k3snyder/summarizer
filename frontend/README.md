# Document Summarizer Pipeline - Frontend

Web UI for the privacy-first document intelligence system.

## Requirements

- Node.js 18+
- npm
- Backend server running on port 8000

## Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Configure environment:

The frontend needs to know where the backend API is running. Create `.env.local` if it does not exist:

```bash
echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" > .env.local
```

3. Start the development server:

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`.

## Running with Backend

Both the frontend and backend need to be running for the full application to work:

**Terminal 1 - Backend:**
```bash
cd backend
source .venv/bin/activate  # or: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Usage

1. Navigate to `http://localhost:3000`
2. Drag and drop or click to upload a PDF, TXT, or MD file (max 100MB)
3. Configure the pipeline options:
   - **Extraction**: Skip tables, images, or enable text-only mode
   - **Vision**: Choose a vision processing provider (Ollama, Gemini, OpenAI, DeepSeek)
   - **Chunking**: Set chunk size and overlap for text processing
   - **Summarization**: Enable or disable summarization, choose mode
4. Click Submit to start processing
5. View progress in real-time as the document is processed
6. View results with extracted text, tables, images, and summaries
7. Export results as JSON

## Features

- Drag-and-drop file upload
- Step-by-step configuration wizard
- Real-time progress tracking with stage indicators
- Results viewer with:
  - Document overview
  - Individual page cards with collapsible text
  - Table visualization
  - Summary notes and topics
  - Relevancy scores
- JSON export functionality
- Error handling with user-friendly messages

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Backend API URL |

## Tech Stack

- Next.js 15 with App Router
- TypeScript (strict mode)
- Tailwind CSS v4
- shadcn/ui components
- React Server Components

## Build for Production

```bash
npm run build
npm start
```

## API Endpoints

The frontend communicates with the backend via these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/jobs` | Create new processing job |
| GET | `/api/jobs/{id}` | Get job status |
| GET | `/api/jobs/{id}/output` | Get processing results |
| DELETE | `/api/jobs/{id}` | Cancel or delete job |
