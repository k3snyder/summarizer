# Config Schema

Current pipeline config shape, based on:
- `backend/app/models/config.py`
- `frontend/types/pipeline.ts`

## Current defaults

```json
{
  "extract_only": false,
  "skip_tables": false,
  "skip_images": false,
  "skip_pptx_tables": false,
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

## Field summary

### Extraction

- `extract_only`: stop after extraction
- `skip_tables`
- `skip_images`
- `skip_pptx_tables`
- `text_only`
- `pdf_image_dpi`: `72 | 144 | 200 | 300`

### Vision

- `vision_mode`: `none | deepseek | gemini | openai | ollama | llama_cpp | codex | claude`
- `vision_classifier_mode`: same choices, optional
- `vision_extractor_mode`: same choices, optional
- `vision_cli_provider`: `codex | claude`, optional
- `vision_detailed_extraction`: run extraction 3 times and synthesize

### Summarization

- `run_summarization`
- `summarizer_mode`: `full | topics-only | skip`
- `summarizer_provider`: `ollama | llama_cpp | openai | codex | claude`
- `summarizer_cli_provider`: `codex | claude`, optional
- `summarizer_detailed_extraction`
- `summarizer_insight_mode`

### Output

- `keep_base64_images`: keep base64 images in output JSON

## Current local-provider policy

- Preferred local vision: `vision_mode=llama_cpp`
- Preferred local summarizer: `summarizer_provider=llama_cpp`
- Ollama remains available as a fallback local provider

## Common recipes

### Local default full pipeline

```json
{
  "vision_mode": "llama_cpp",
  "summarizer_provider": "llama_cpp",
  "summarizer_mode": "full"
}
```

### Summarization only

```json
{
  "vision_mode": "none",
  "summarizer_provider": "llama_cpp",
  "summarizer_mode": "full"
}
```

### Extraction only

```json
{
  "extract_only": true,
  "run_summarization": false
}
```

### CLI vision extraction

```json
{
  "vision_mode": "codex",
  "vision_cli_provider": "codex"
}
```

When the user needs exact control, prefer `--config-json` or `--set key=value` over guessing.
