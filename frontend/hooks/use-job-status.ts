/**
 * Hook for polling job status from the backend.
 *
 * Polls GET /api/jobs/{id} at a configurable interval and stops
 * automatically when the job reaches a terminal state.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError } from '@/lib/api-client';
import type { JobStatus } from '@/types';

/**
 * Options for useJobStatus hook.
 */
interface UseJobStatusOptions {
  /** Polling interval in milliseconds. Defaults to 2000 (2 seconds). */
  interval?: number;
}

/**
 * Return value from useJobStatus hook.
 */
interface UseJobStatusResult {
  /** Current job status, or null if no job or not yet fetched. */
  status: JobStatus | null;
  /** Error from the most recent fetch attempt, or null. */
  error: Error | null;
  /** True while the initial fetch is in progress. */
  isLoading: boolean;
}

/**
 * Terminal job statuses that stop polling.
 */
const TERMINAL_STATUSES = new Set(['completed', 'failed']);

/**
 * Poll job status from the backend API.
 *
 * @param jobId - The job ID to poll, or null for idle state
 * @param options - Optional configuration (polling interval)
 * @returns Object with status, error, and loading state
 *
 * @example
 * ```tsx
 * const { status, error, isLoading } = useJobStatus(jobId);
 *
 * if (isLoading) return <Spinner />;
 * if (error) return <Error message={error.message} />;
 * if (status?.status === 'completed') return <Results />;
 * ```
 */
export function useJobStatus(
  jobId: string | null,
  options?: UseJobStatusOptions
): UseJobStatusResult {
  const { interval = 2000 } = options ?? {};

  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Track if component is mounted to prevent state updates after unmount
  const isMountedRef = useRef<boolean>(true);
  // Track the current timeout for cleanup
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * Fetch the current job status.
   */
  const fetchStatus = useCallback(async (): Promise<JobStatus | null> => {
    if (!jobId) return null;

    try {
      const result = await api.getJobStatus(jobId);
      return result;
    } catch (err) {
      if (err instanceof ApiError || err instanceof Error) {
        throw err;
      }
      throw new Error('Unknown error fetching job status');
    }
  }, [jobId]);

  /**
   * Schedule the next poll using setTimeout.
   */
  const scheduleNextPoll = useCallback(
    (pollFn: () => void) => {
      // Clear any existing timeout
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(pollFn, interval);
    },
    [interval]
  );

  useEffect(() => {
    isMountedRef.current = true;

    // Reset state when jobId changes
    setStatus(null);
    setError(null);
    setIsLoading(false);

    // If no jobId, stay in idle state
    if (!jobId) {
      return;
    }

    setIsLoading(true);

    /**
     * Poll function that fetches status and schedules next poll if needed.
     */
    const poll = async () => {
      if (!isMountedRef.current) return;

      try {
        const result = await fetchStatus();

        if (!isMountedRef.current) return;

        setStatus(result);
        setError(null);
        setIsLoading(false);

        // Schedule next poll if not in terminal state
        if (result && !TERMINAL_STATUSES.has(result.status)) {
          scheduleNextPoll(poll);
        }
      } catch (err) {
        if (!isMountedRef.current) return;

        const errorObj =
          err instanceof Error ? err : new Error('Unknown error');
        setError(errorObj);
        setIsLoading(false);
        // Stop polling on error
      }
    };

    // Start polling immediately
    poll();

    // Cleanup on unmount or jobId change
    return () => {
      isMountedRef.current = false;
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [jobId, fetchStatus, scheduleNextPoll]);

  return { status, error, isLoading };
}
