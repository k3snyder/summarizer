/**
 * Job status and response models.
 *
 * Matches backend/app/models/job.py
 */

/**
 * Valid job status values.
 */
export type JobStatusValue = 'pending' | 'processing' | 'completed' | 'failed';

/**
 * Current status and progress of a processing job.
 * Tracks job lifecycle from creation through completion or failure.
 */
export interface JobStatus {
  /** Unique identifier for the job */
  job_id: string;

  /** Current job status */
  status: JobStatusValue;

  /** Progress percentage (0-100) */
  progress: number;

  /** Current processing stage (extraction, vision, summarization) */
  current_stage: string | null;

  /** Human-readable status message */
  message: string | null;

  /** Original filename of the document being processed */
  file_name: string;

  /** Job creation timestamp (ISO 8601 string) */
  created_at: string;

  /** Job processing start timestamp (ISO 8601 string) */
  started_at: string | null;

  /** Job completion timestamp (ISO 8601 string) */
  completed_at: string | null;

  /** Error message if job failed */
  error: string | null;
}

/**
 * Response from job creation endpoint.
 * Provides job ID and initial status information.
 */
export interface JobCreateResponse {
  /** Unique identifier for the created job */
  job_id: string;

  /** Initial job status (typically 'pending') */
  status: string;

  /** Human-readable confirmation message */
  message: string;
}
