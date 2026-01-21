"use client";

import * as React from "react";
import { Badge } from "@/components/ui/badge";
import { PipelineConfig } from "@/types";

interface ConfigSummaryProps {
  config: PipelineConfig;
  fileType?: "pdf" | "pptx" | "text";
}

interface SummaryRowProps {
  label: string;
  value: React.ReactNode;
}

function SummaryRow({ label, value }: SummaryRowProps) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-medium">{value}</span>
    </div>
  );
}

function BooleanBadge({ value, trueLabel = "Yes", falseLabel = "No" }: {
  value: boolean;
  trueLabel?: string;
  falseLabel?: string;
}) {
  return (
    <Badge variant={value ? "default" : "secondary"}>
      {value ? trueLabel : falseLabel}
    </Badge>
  );
}

function getDpiDescription(dpi: number): string {
  switch (dpi) {
    case 72:
      return "72 (Fast)";
    case 144:
      return "144 (Balanced)";
    case 200:
      return "200 (Default)";
    case 300:
      return "300 (High Quality)";
    default:
      return String(dpi);
  }
}

function getVisionModeDescription(mode: string): string {
  switch (mode) {
    case "none":
      return "Disabled";
    case "ollama":
      return "Ollama (Local)";
    case "openai":
      return "OpenAI GPT-4V";
    case "gemini":
      return "Google Gemini";
    default:
      return mode;
  }
}

function getSummarizerModeDescription(mode: string): string {
  switch (mode) {
    case "full":
      return "Full (Notes + Topics)";
    case "topics-only":
      return "Topics Only";
    case "skip":
      return "Skip";
    default:
      return mode;
  }
}

function getSummarizerProviderDescription(provider: string): string {
  switch (provider) {
    case "ollama":
      return "Ollama (Local)";
    case "openai":
      return "OpenAI GPT-4";
    default:
      return provider;
  }
}

export function ConfigSummary({ config, fileType = "pdf" }: ConfigSummaryProps) {
  // Extract Only skips vision and summarization
  const showVisionAndSummarization = !config.extract_only;

  return (
    <div className="space-y-4">
      {/* PDF/PPTX: Extraction Settings */}
      {(fileType === "pdf" || fileType === "pptx") && (
        <div className="rounded-lg border bg-card">
          <div className="p-4 border-b">
            <h4 className="font-semibold flex items-center gap-2">
              <svg
                className="w-4 h-4 text-muted-foreground"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              {fileType === "pptx" ? "PowerPoint Extraction" : "PDF Extraction"}
            </h4>
          </div>
          <div className="p-4 space-y-1">
            <SummaryRow
              label="Extract Only"
              value={<BooleanBadge value={config.extract_only} />}
            />
            {!config.extract_only && (
              <>
                <SummaryRow
                  label="Text Only"
                  value={<BooleanBadge value={config.text_only} />}
                />
                <SummaryRow
                  label={fileType === "pptx" ? "Skip Slide Notes" : "Skip Tables"}
                  value={<BooleanBadge value={config.skip_tables} />}
                />
                {fileType === "pptx" && (
                  <SummaryRow
                    label="Skip Slide Tables"
                    value={<BooleanBadge value={config.skip_pptx_tables} />}
                  />
                )}
                <SummaryRow
                  label={fileType === "pptx" ? "Skip Slide Screenshots" : "Skip Images"}
                  value={<BooleanBadge value={config.skip_images} />}
                />
                <SummaryRow
                  label="Keep Base64 Images"
                  value={<BooleanBadge value={config.keep_base64_images} />}
                />
              </>
            )}
          </div>
        </div>
      )}

      {/* PDF/PPTX: Vision Processing (only if not extract_only) */}
      {(fileType === "pdf" || fileType === "pptx") && showVisionAndSummarization && (
        <div className="rounded-lg border bg-card">
          <div className="p-4 border-b">
            <h4 className="font-semibold flex items-center gap-2">
              <svg
                className="w-4 h-4 text-muted-foreground"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
              Vision Processing
            </h4>
          </div>
          <div className="p-4 space-y-1">
            {config.vision_classifier_mode || config.vision_extractor_mode ? (
              <>
                <SummaryRow
                  label="Classification"
                  value={
                    <Badge variant={(config.vision_classifier_mode || config.vision_mode) === "none" ? "secondary" : "default"}>
                      {getVisionModeDescription(config.vision_classifier_mode || config.vision_mode)}
                    </Badge>
                  }
                />
                <SummaryRow
                  label="Extraction"
                  value={
                    <Badge variant={(config.vision_extractor_mode || config.vision_mode) === "none" ? "secondary" : "default"}>
                      {getVisionModeDescription(config.vision_extractor_mode || config.vision_mode)}
                    </Badge>
                  }
                />
              </>
            ) : (
              <SummaryRow
                label="Provider"
                value={
                  <Badge variant={config.vision_mode === "none" ? "secondary" : "default"}>
                    {getVisionModeDescription(config.vision_mode)}
                  </Badge>
                }
              />
            )}
            {config.vision_mode !== "none" && (
              <SummaryRow
                label="PDF Image DPI"
                value={getDpiDescription(config.pdf_image_dpi)}
              />
            )}
          </div>
        </div>
      )}

      {/* Text: Chunking Settings */}
      {fileType === "text" && (
        <div className="rounded-lg border bg-card">
          <div className="p-4 border-b">
            <h4 className="font-semibold flex items-center gap-2">
              <svg
                className="w-4 h-4 text-muted-foreground"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16m-7 6h7"
                />
              </svg>
              Text Chunking
            </h4>
          </div>
          <div className="p-4 space-y-1">
            <SummaryRow
              label="Chunk Size"
              value={`${config.chunk_size.toLocaleString()} chars`}
            />
            <SummaryRow
              label="Chunk Overlap"
              value={`${config.chunk_overlap} chars`}
            />
          </div>
        </div>
      )}

      {/* Summarization (only if not extract_only) */}
      {showVisionAndSummarization && (
        <div className="rounded-lg border bg-card">
          <div className="p-4 border-b">
            <h4 className="font-semibold flex items-center gap-2">
              <svg
                className="w-4 h-4 text-muted-foreground"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                />
              </svg>
              Summarization
            </h4>
          </div>
          <div className="p-4 space-y-1">
            <SummaryRow
              label="Enabled"
              value={<BooleanBadge value={config.run_summarization} />}
            />
            {config.run_summarization && (
              <>
                <SummaryRow
                  label="Provider"
                  value={
                    <Badge variant="default">
                      {getSummarizerProviderDescription(config.summarizer_provider)}
                    </Badge>
                  }
                />
                <SummaryRow
                  label="Mode"
                  value={
                    <Badge variant="default">
                      {getSummarizerModeDescription(config.summarizer_mode)}
                    </Badge>
                  }
                />
                <SummaryRow
                  label="Detailed Extraction"
                  value={<BooleanBadge value={config.summarizer_detailed_extraction} />}
                />
                <SummaryRow
                  label="Insight Mode"
                  value={<BooleanBadge value={config.summarizer_insight_mode} />}
                />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
