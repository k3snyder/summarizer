"use client";

import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { JobStatus } from "@/types";

interface ProgressTrackerProps {
  status: JobStatus | null;
  isLoading?: boolean;
  extractOnly?: boolean;
  skipVision?: boolean;
  runSummarization?: boolean;
}

type Stage = { id: string; label: string };

function getStages(extractOnly?: boolean, skipVision?: boolean, runSummarization?: boolean): Stage[] {
  if (extractOnly) {
    return [{ id: "extraction", label: "Extraction" }];
  }

  const stages: Stage[] = [{ id: "extraction", label: "Extraction" }];

  if (!skipVision) {
    stages.push({ id: "vision", label: "Vision" });
  }

  if (runSummarization !== false) {
    stages.push({ id: "summarization", label: "Summarization" });
  }

  return stages;
}

function getStageIndex(stages: readonly Stage[], stageId: string | null): number {
  if (!stageId) return -1;
  return stages.findIndex((s) => s.id === stageId);
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2.5}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M5 13l4 4L19 7"
      />
    </svg>
  );
}

function AlertIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
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
  );
}

function LoadingSpinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn("animate-spin", className)}
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
  );
}

export function ProgressTracker({ status, isLoading, extractOnly, skipVision, runSummarization }: ProgressTrackerProps) {
  if (!status && !isLoading) {
    return null;
  }

  const stages = getStages(extractOnly, skipVision, runSummarization);
  const currentStageIndex = getStageIndex(stages, status?.current_stage ?? null);
  const isFailed = status?.status === "failed";
  const isCompleted = status?.status === "completed";
  const isPending = status?.status === "pending";
  const isProcessing = status?.status === "processing";

  function getStageState(stageIndex: number): "completed" | "current" | "pending" | "failed" {
    if (isFailed && stageIndex === currentStageIndex) {
      return "failed";
    }
    if (isCompleted) {
      return "completed";
    }
    if (stageIndex < currentStageIndex) {
      return "completed";
    }
    if (stageIndex === currentStageIndex && isProcessing) {
      return "current";
    }
    return "pending";
  }

  function getStatusBadgeVariant(): "default" | "secondary" | "destructive" | "outline" {
    if (isFailed) return "destructive";
    if (isCompleted) return "default";
    if (isProcessing) return "secondary";
    return "outline";
  }

  function getStatusLabel(): string {
    if (isFailed) return "Failed";
    if (isCompleted) return "Completed";
    if (isProcessing) return "Processing";
    if (isPending) return "Pending";
    return "Unknown";
  }

  return (
    <Card className="border border-border/50 bg-card/50 backdrop-blur-sm">
      <CardContent className="p-6">
        {/* Header with status badge */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-sm font-medium text-foreground">Pipeline Progress</h3>
          <Badge variant={getStatusBadgeVariant()}>
            {isLoading && !status ? (
              <span className="flex items-center gap-1.5">
                <LoadingSpinner className="w-3 h-3" />
                Loading
              </span>
            ) : (
              getStatusLabel()
            )}
          </Badge>
        </div>

        {/* Stage indicators */}
        <div className="flex items-center justify-between mb-6">
          {stages.map((stage, index) => {
            const state = status ? getStageState(index) : "pending";
            const isLast = index === stages.length - 1;

            return (
              <div key={stage.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center">
                  {/* Stage circle */}
                  <div
                    className={cn(
                      "w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300",
                      state === "completed" && "bg-primary border-primary text-primary-foreground",
                      state === "current" && "border-primary bg-primary/10 text-primary",
                      state === "pending" && "border-muted-foreground/30 bg-muted/50 text-muted-foreground",
                      state === "failed" && "border-destructive bg-destructive/10 text-destructive"
                    )}
                  >
                    {state === "completed" && <CheckIcon className="w-5 h-5" />}
                    {state === "current" && <LoadingSpinner className="w-5 h-5" />}
                    {state === "pending" && (
                      <span className="text-sm font-medium">{index + 1}</span>
                    )}
                    {state === "failed" && <AlertIcon className="w-5 h-5" />}
                  </div>
                  {/* Stage label */}
                  <span
                    className={cn(
                      "text-xs mt-2 font-medium transition-colors duration-300",
                      state === "completed" && "text-foreground",
                      state === "current" && "text-primary",
                      state === "pending" && "text-muted-foreground",
                      state === "failed" && "text-destructive"
                    )}
                  >
                    {stage.label}
                  </span>
                </div>
                {/* Connector line */}
                {!isLast && (
                  <div
                    className={cn(
                      "flex-1 h-0.5 mx-3 transition-colors duration-300",
                      index < currentStageIndex || isCompleted
                        ? "bg-primary"
                        : "bg-muted-foreground/20"
                    )}
                  />
                )}
              </div>
            );
          })}
        </div>

        {/* Progress bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Overall Progress</span>
            <span className="font-medium text-foreground">
              {status?.progress ?? 0}%
            </span>
          </div>
          <Progress
            value={status?.progress ?? 0}
            className={cn(
              "h-2",
              isFailed && "[&>div]:bg-destructive"
            )}
          />
        </div>

        {/* Status message */}
        {status?.message && (
          <p className="mt-4 text-sm text-muted-foreground">
            {status.message}
          </p>
        )}

        {/* Error display */}
        {isFailed && status?.error && (
          <div className="mt-4 p-3 rounded-lg bg-destructive/10 border border-destructive/20">
            <div className="flex items-start gap-2">
              <AlertIcon className="w-4 h-4 text-destructive mt-0.5 flex-shrink-0" />
              <p className="text-sm text-destructive">{status.error}</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
