---
name: summarizer-cli
description: Submit local documents to the summarizer-app-v3 FastAPI pipeline and retrieve the completed output JSON and Markdown report. Use when Codex needs to process a PDF, PPTX, TXT, or Markdown file through the app's existing /api/jobs workflow, especially to run the same parameters the frontend would use without browser automation, poll job completion, and return artifact paths or parsed results.
---

# Summarizer CLI

## Overview

Use the backend REST API, not the Next.js UI. This skill submits a document to `/api/jobs`, waits for completion, downloads the output JSON and Markdown report, and returns stable artifact paths the agent can inspect.

## Workflow

### 1. Confirm the backend is reachable

- Prefer `--backend-url` when the user gives one.
- Otherwise the script falls back to `SUMMARIZER_BACKEND_URL`, then `NEXT_PUBLIC_BACKEND_URL`, then `http://localhost:8000`.
- If you are inside the `summarizer-app-v3` repo and the backend is not up, run `make services-up` before retrying.

### 2. Choose how to express pipeline settings

- Use friendly flags for normal work.
- Use `--config-json` when the user already has an exact config object.
- Use `--set key=value` for one-off overrides that are not covered by a dedicated flag.
- Read `references/config-schema.md` only when you need the full current config surface or the default values.

### 3. Submit and wait

Run:

```bash
~/.codex/skills/summarizer-cli/scripts/run_job.py \
  --file /path/to/document.pdf \
  --backend-url http://localhost:8000
```

The script:

- checks `/api/health`
- submits the file with multipart form data
- polls `/api/jobs/{job_id}`
- downloads `/api/jobs/{job_id}/output`
- downloads `/api/jobs/{job_id}/output/markdown`
- writes a manifest plus artifact files to an output directory

### 4. Read the artifacts before answering

After the script finishes:

- read `output.json` for the structured result
- read `output.md` for the report form
- read `job-meta.json` for the resolved config, job id, and final status

Do not dump large payloads directly to the user if a concise summary plus file paths is enough.

### 5. Failure handling

- If health check fails, start the local services or ask for the correct backend URL.
- If the job fails, inspect `job-meta.json` and the final job status/error before deciding whether to retry.
- If the request shape seems wrong, read `references/api-contract.md`.

## Examples

Use default local llama.cpp settings:

```bash
~/.codex/skills/summarizer-cli/scripts/run_job.py \
  --file /path/to/document.pdf
```

Vision on llama.cpp, summarization on llama.cpp, no images kept in final JSON:

```bash
~/.codex/skills/summarizer-cli/scripts/run_job.py \
  --file /path/to/document.pdf \
  --vision-mode llama_cpp \
  --summarizer-provider llama_cpp \
  --no-keep-base64-images
```

Vision disabled, summarization only:

```bash
~/.codex/skills/summarizer-cli/scripts/run_job.py \
  --file /path/to/document.md \
  --vision-mode none \
  --summarizer-provider llama_cpp
```

Exact override with JSON:

```bash
~/.codex/skills/summarizer-cli/scripts/run_job.py \
  --file /path/to/document.pptx \
  --config-json '{"vision_mode":"llama_cpp","summarizer_provider":"llama_cpp","pdf_image_dpi":300}'
```

## Resources

### `scripts/run_job.py`

Submit a file, poll completion, download JSON and Markdown outputs, and print a manifest to stdout.

### `references/config-schema.md`

Current pipeline config keys, defaults, and common local-provider choices.

### `references/api-contract.md`

Current job endpoints, request shape, and download flow.
