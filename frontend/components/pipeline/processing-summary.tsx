"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { PipelineMetrics } from "@/types";

interface ProcessingSummaryProps {
  metrics: PipelineMetrics;
}

function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  const seconds = ms / 1000;
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
}

function formatTokens(tokens: number): string {
  if (tokens >= 1000000) {
    return `${(tokens / 1000000).toFixed(2)}M`;
  }
  if (tokens >= 1000) {
    return `${(tokens / 1000).toFixed(1)}K`;
  }
  return tokens.toString();
}

function ChevronIcon({ className, expanded }: { className?: string; expanded: boolean }) {
  return (
    <svg
      className={cn("transition-transform duration-200", expanded && "rotate-180", className)}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}

export function ProcessingSummary({ metrics }: ProcessingSummaryProps) {
  const [expanded, setExpanded] = useState(false);
  const { stages, config } = metrics;

  return (
    <Card className="border border-border/50 bg-card/50 backdrop-blur-sm">
      <CardContent className="p-6">
        {/* Collapsible Header */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center justify-between text-left"
        >
          <h3 className="text-sm font-medium text-foreground">Processing Summary</h3>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">
              {formatDuration(metrics.total_duration_ms)} · {formatTokens(metrics.total_tokens)} tokens
            </span>
            <ChevronIcon className="w-4 h-4 text-muted-foreground" expanded={expanded} />
          </div>
        </button>

        {/* Expandable Content */}
        <div
          className={cn(
            "overflow-hidden transition-all duration-200",
            expanded ? "max-h-96 opacity-100 mt-6" : "max-h-0 opacity-0"
          )}
        >
          {/* Totals Row */}
          <div className="grid grid-cols-2 gap-4 pb-4 border-b border-border/50">
            <div>
              <p className="text-xs text-muted-foreground">Total Duration</p>
              <p className="text-lg font-semibold text-foreground">{formatDuration(metrics.total_duration_ms)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Total Tokens</p>
              <p className="text-lg font-semibold text-foreground">{formatTokens(metrics.total_tokens)}</p>
            </div>
          </div>

          {/* Stage Breakdown */}
          <div className="pt-4 space-y-3">
            <p className="text-xs font-medium text-muted-foreground">Stage Breakdown</p>

            <div className="grid grid-cols-3 gap-3 text-sm">
              {/* Extraction */}
              <div className="space-y-1">
                <p className="font-medium text-foreground">Extraction</p>
                <p className="text-muted-foreground">
                  {formatDuration(stages.extraction.duration_ms)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {stages.extraction.pages_processed} pages
                </p>
              </div>

              {/* Vision */}
              <div className="space-y-1">
                <p className="font-medium text-foreground">Vision</p>
                <p className="text-muted-foreground">
                  {formatDuration(stages.vision.duration_ms)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {stages.vision.extracted_count ?? 0}/{stages.vision.pages_with_images ?? 0} extracted
                </p>
                {stages.vision.tokens > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {formatTokens(stages.vision.tokens)} tokens
                  </p>
                )}
              </div>

              {/* Summarization */}
              <div className="space-y-1">
                <p className="font-medium text-foreground">Summarization</p>
                <p className="text-muted-foreground">
                  {formatDuration(stages.summarization.duration_ms)}
                </p>
                {stages.summarization.avg_relevancy !== undefined && stages.summarization.avg_relevancy > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {stages.summarization.avg_relevancy}% relevancy
                  </p>
                )}
                {stages.summarization.tokens > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {formatTokens(stages.summarization.tokens)} tokens
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Configuration */}
          {(config.vision_mode || config.summarizer_provider) && (
            <div className="pt-4 mt-4 border-t border-border/50">
              <p className="text-xs font-medium text-muted-foreground mb-2">Configuration</p>
              <div className="flex flex-wrap gap-2 text-xs">
                {config.vision_mode && config.vision_mode !== "none" && (
                  <span className="px-2 py-1 bg-muted/50 rounded text-muted-foreground">
                    Vision: {config.vision_mode}
                  </span>
                )}
                {config.summarizer_provider && (
                  <span className="px-2 py-1 bg-muted/50 rounded text-muted-foreground">
                    Summarizer: {config.summarizer_provider}
                  </span>
                )}
                {config.summarizer_mode && (
                  <span className="px-2 py-1 bg-muted/50 rounded text-muted-foreground">
                    Mode: {config.summarizer_mode}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
