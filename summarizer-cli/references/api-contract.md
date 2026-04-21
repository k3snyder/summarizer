# API Contract

Current backend contract for the summarizer app.

Canonical source files in the repo:
- `backend/app/routers/jobs.py`
- `backend/app/models/job.py`
- `frontend/lib/api-client.ts`

## Base URL

- Default local backend: `http://localhost:8000`
- Endpoints are mounted under `/api`

## Endpoints

### `GET /api/health`

Used as a fast connectivity check before job submission.

Expected success:

```json
{ "status": "ok" }
```

### `POST /api/jobs`

Create a processing job.

Request type:
- `multipart/form-data`

Form fields:
- `file`: uploaded `PDF`, `PPTX`, `TXT`, or `MD`
- `config`: JSON string matching `PipelineConfig`

Success response:

```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Job created. Processing file.pdf"
}
```

Common failures:
- `400` invalid filename, type, or config
- `413` file too large

### `GET /api/jobs/{job_id}`

Return job status and progress.

Important fields:
- `status`: `pending` | `processing` | `completed` | `failed`
- `progress`: `0..100`
- `current_stage`
- `message`
- `error`

### `GET /api/jobs/{job_id}/output`

Return the completed output JSON.

Precondition:
- job status must be `completed`

### `GET /api/jobs/{job_id}/output/markdown`

Return the markdown summary report.

Precondition:
- job status must be `completed`

### `DELETE /api/jobs/{job_id}`

Delete/cancel a job. This skill does not call it by default.

## Polling pattern

Recommended flow:

1. `POST /api/jobs`
2. Poll `GET /api/jobs/{job_id}` until `completed` or `failed`
3. Download JSON from `/output`
4. Download Markdown from `/output/markdown`

## Notes

- The frontend already uses this REST flow. Do not automate the browser unless the user explicitly asks for UI automation.
- The upload path is the only multipart request; all other job endpoints are standard JSON or plain text responses.
