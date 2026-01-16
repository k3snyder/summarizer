/**
 * Typed API client for backend communication.
 *
 * Uses native fetch with proper error handling.
 * Backend URL is configurable via NEXT_PUBLIC_BACKEND_URL env var.
 */

import type {
  PipelineConfig,
  JobStatus,
  JobCreateResponse,
  DocumentOutput,
} from '@/types';

/**
 * API error with structured details from backend.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly detail: string | null = null
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Get the backend base URL from environment or use default.
 */
function getBaseUrl(): string {
  return process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000';
}

/**
 * Parse error response from backend.
 */
async function parseErrorResponse(response: Response): Promise<string | null> {
  try {
    const data: unknown = await response.json();
    if (
      typeof data === 'object' &&
      data !== null &&
      'detail' in data &&
      typeof (data as Record<string, unknown>).detail === 'string'
    ) {
      return (data as Record<string, string>).detail;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Handle non-OK responses by throwing typed ApiError.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new ApiError(
      `Request failed with status ${response.status}`,
      response.status,
      detail
    );
  }
  return response.json() as Promise<T>;
}

/**
 * Create a new document processing job.
 *
 * @param file - PDF, TXT, or MD file to process
 * @param config - Pipeline configuration options
 * @returns Job creation response with job_id
 * @throws ApiError on request failure
 */
async function createJob(
  file: File,
  config: PipelineConfig
): Promise<JobCreateResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('config', JSON.stringify(config));

  const response = await fetch(`${getBaseUrl()}/api/jobs`, {
    method: 'POST',
    body: formData,
  });

  return handleResponse<JobCreateResponse>(response);
}

/**
 * Get current status and progress of a job.
 *
 * @param jobId - Unique job identifier
 * @returns Current job status
 * @throws ApiError on request failure (404 if job not found)
 */
async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${getBaseUrl()}/api/jobs/${jobId}`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  return handleResponse<JobStatus>(response);
}

/**
 * Get processing output for a completed job.
 *
 * @param jobId - Unique job identifier
 * @returns Document output with pages and summaries
 * @throws ApiError on request failure (404 if not found, 400 if not completed)
 */
async function getJobOutput(jobId: string): Promise<DocumentOutput> {
  const response = await fetch(`${getBaseUrl()}/api/jobs/${jobId}/output`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  return handleResponse<DocumentOutput>(response);
}

/**
 * Cancel a running job or delete a completed job.
 *
 * @param jobId - Unique job identifier
 * @throws ApiError on request failure (404 if job not found)
 */
async function deleteJob(jobId: string): Promise<void> {
  const response = await fetch(`${getBaseUrl()}/api/jobs/${jobId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new ApiError(
      `Request failed with status ${response.status}`,
      response.status,
      detail
    );
  }
}

/**
 * API client for backend communication.
 */
export const api = {
  createJob,
  getJobStatus,
  getJobOutput,
  deleteJob,
} as const;
