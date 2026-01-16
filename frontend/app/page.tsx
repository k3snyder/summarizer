"use client";

import { useState, useCallback, useEffect } from "react";
import { FileUpload } from "@/components/pipeline/file-upload";
import { ConfigForm } from "@/components/pipeline/config-form";
import { ProgressTracker } from "@/components/pipeline/progress-tracker";
import { ProcessingSummary } from "@/components/pipeline/processing-summary";
import { api, ApiError } from "@/lib/api-client";
import { useJobStatus } from "@/hooks/use-job-status";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { PipelineConfig, PipelineMetrics } from "@/types";

/**
 * Pipeline state machine.
 * upload -> configure -> processing -> done
 */
type PipelineState = "upload" | "configure" | "processing" | "done";

export default function Home() {
  const [state, setState] = useState<PipelineState>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submittedConfig, setSubmittedConfig] = useState<PipelineConfig | null>(null);
  const [metrics, setMetrics] = useState<PipelineMetrics | null>(null);

  // Poll job status when we have a jobId
  const { status, error: statusError, isLoading: isStatusLoading } = useJobStatus(
    state === "processing" || state === "done" ? jobId : null
  );

  // Transition to done when job completes or fails
  if (state === "processing" && status?.status === "completed") {
    setState("done");
  }
  if (state === "processing" && status?.status === "failed") {
    setState("done");
  }

  // Fetch metrics when job completes
  useEffect(() => {
    if (state === "done" && status?.status === "completed" && jobId && !metrics) {
      const fetchMetrics = async () => {
        try {
          const output = await api.getJobOutput(jobId);
          if (output.metrics) {
            setMetrics(output.metrics);
          }
        } catch (err) {
          console.error("Failed to fetch metrics:", err);
        }
      };
      fetchMetrics();
    }
  }, [state, status?.status, jobId, metrics]);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setState("configure");
    setSubmitError(null);
  }, []);

  const handleFileRemove = useCallback(() => {
    setSelectedFile(null);
    setState("upload");
    setJobId(null);
    setSubmitError(null);
  }, []);

  const handleConfigSubmit = useCallback(
    async (config: PipelineConfig) => {
      if (!selectedFile) return;

      setIsSubmitting(true);
      setSubmitError(null);

      try {
        const response = await api.createJob(selectedFile, config);
        setJobId(response.job_id);
        setSubmittedConfig(config);
        setState("processing");
      } catch (err) {
        if (err instanceof ApiError) {
          setSubmitError(err.detail ?? err.message);
        } else if (err instanceof Error) {
          setSubmitError(err.message);
        } else {
          setSubmitError("An unexpected error occurred");
        }
      } finally {
        setIsSubmitting(false);
      }
    },
    [selectedFile]
  );

  const handleRetry = useCallback(() => {
    setJobId(null);
    setSubmitError(null);
    setState("configure");
  }, []);

  const handleStartNew = useCallback(() => {
    setSelectedFile(null);
    setJobId(null);
    setSubmitError(null);
    setMetrics(null);
    setState("upload");
  }, []);

  const handleDownload = useCallback(async () => {
    if (!jobId) return;

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'}/api/jobs/${jobId}/output`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch output');
      }

      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedFile?.name.replace(/\.[^/.]+$/, '') ?? 'output'}_summary.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
    }
  }, [jobId, selectedFile?.name]);

  const handleDownloadMarkdown = useCallback(async () => {
    if (!jobId) return;

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'}/api/jobs/${jobId}/output/markdown`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch markdown');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedFile?.name.replace(/\.[^/.]+$/, '') ?? 'output'}_summary.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Markdown download failed:', err);
    }
  }, [jobId, selectedFile?.name]);

  const isCompleted = status?.status === "completed";
  const isFailed = status?.status === "failed";

  return (
    <div className="min-h-screen p-8">
      <main className="max-w-4xl mx-auto space-y-8">
        <div>
          <h1 className="text-4xl font-bold mb-2">
            AI-powered Document Intelligence Pipeline
          </h1>
          <p className="text-lg text-muted-foreground">
            Extracts and structures document content for AI-ready consumption.
          </p>
        </div>

        {/* Upload State */}
        {state === "upload" && (
          <FileUpload
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onRemove={handleFileRemove}
          />
        )}

        {/* Configure State */}
        {state === "configure" && selectedFile && (
          <div className="space-y-6">
            <FileUpload
              onFileSelect={handleFileSelect}
              selectedFile={selectedFile}
              onRemove={handleFileRemove}
            />

            {submitError && (
              <Card className="border-destructive bg-destructive/10">
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <svg
                      className="w-5 h-5 text-destructive mt-0.5 flex-shrink-0"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                      />
                    </svg>
                    <div>
                      <p className="text-sm font-medium text-destructive">
                        Failed to create job
                      </p>
                      <p className="text-sm text-destructive/80 mt-1">
                        {submitError}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            <ConfigForm onSubmit={handleConfigSubmit} filename={selectedFile?.name} />

            {isSubmitting && (
              <div className="flex items-center justify-center gap-3 py-4">
                <svg
                  className="w-5 h-5 animate-spin text-primary"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="3"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                <span className="text-sm text-muted-foreground">
                  Creating job...
                </span>
              </div>
            )}
          </div>
        )}

        {/* Processing State */}
        {state === "processing" && (
          <div className="space-y-6">
            <ProgressTracker
              status={status}
              isLoading={isStatusLoading}
              extractOnly={submittedConfig?.extract_only}
              skipVision={submittedConfig?.vision_mode === "none" || submittedConfig?.skip_images || submittedConfig?.text_only}
              runSummarization={submittedConfig?.run_summarization}
            />

            {statusError && (
              <Card className="border-destructive bg-destructive/10">
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <svg
                      className="w-5 h-5 text-destructive mt-0.5 flex-shrink-0"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                      />
                    </svg>
                    <div>
                      <p className="text-sm font-medium text-destructive">
                        Failed to get job status
                      </p>
                      <p className="text-sm text-destructive/80 mt-1">
                        {statusError.message}
                      </p>
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-3"
                        onClick={handleRetry}
                      >
                        Try Again
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            <p className="text-sm text-muted-foreground text-center">
              Processing your document. This may take a few minutes.
            </p>
          </div>
        )}

        {/* Done State */}
        {state === "done" && status && (
          <div className="space-y-6">
            <ProgressTracker
              status={status}
              extractOnly={submittedConfig?.extract_only}
              skipVision={submittedConfig?.vision_mode === "none" || submittedConfig?.skip_images || submittedConfig?.text_only}
              runSummarization={submittedConfig?.run_summarization}
            />

            {isCompleted && metrics && <ProcessingSummary metrics={metrics} />}

            {isCompleted && (
              <Card className="border-primary/50 bg-primary/5">
                <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                      <svg
                        className="w-6 h-6 text-primary"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-foreground">
                        Processing Complete
                      </h3>
                      <p className="text-sm text-muted-foreground mt-1">
                        Your document has been successfully processed.
                      </p>
                      <div className="flex flex-wrap gap-3 mt-4">
                        <Button variant="outline" onClick={handleDownload}>
                          <svg
                            className="w-4 h-4 mr-2"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={2}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                            />
                          </svg>
                          Download JSON
                        </Button>
                        <Button variant="outline" onClick={handleDownloadMarkdown}>
                          <svg
                            className="w-4 h-4 mr-2"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={2}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                            />
                          </svg>
                          Download Markdown
                        </Button>
                        <Button variant="outline" onClick={handleStartNew}>
                          Process Another Document
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {isFailed && (
              <Card className="border-destructive bg-destructive/10">
                <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-full bg-destructive/20 flex items-center justify-center flex-shrink-0">
                      <svg
                        className="w-6 h-6 text-destructive"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-destructive">
                        Processing Failed
                      </h3>
                      <p className="text-sm text-muted-foreground mt-1">
                        {status.error ?? "An error occurred while processing your document."}
                      </p>
                      <div className="flex gap-3 mt-4">
                        <Button onClick={handleRetry}>
                          <svg
                            className="w-4 h-4 mr-2"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={2}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                            />
                          </svg>
                          Try Again
                        </Button>
                        <Button variant="outline" onClick={handleStartNew}>
                          Start Over
                        </Button>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
