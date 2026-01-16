"use client";

import * as React from "react";
import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import {
  PipelineConfig,
  DEFAULT_PIPELINE_CONFIG,
  PdfImageDpi,
  VisionMode,
  SummarizerMode,
  SummarizerProvider,
  CLIProvider,
} from "@/types";
import { ConfigSummary } from "./config-summary";

interface ConfigFormProps {
  onSubmit: (config: PipelineConfig) => void;
  initialConfig?: PipelineConfig;
  filename?: string;
}

type StepId = "extraction" | "vision" | "chunking" | "summarization" | "review";

interface StepDef {
  id: StepId;
  title: string;
  description: string;
}

const DPI_OPTIONS: { value: PdfImageDpi; label: string }[] = [
  { value: 72, label: "72 DPI (Fast)" },
  { value: 144, label: "144 DPI (Balanced)" },
  { value: 200, label: "200 DPI (Default)" },
  { value: 300, label: "300 DPI (High Quality)" },
];

const VISION_MODE_OPTIONS: { value: VisionMode; label: string; description: string }[] = [
  { value: "none", label: "None", description: "Skip vision processing" },
  { value: "ollama", label: "Ollama", description: "Locally hosted model" },
  { value: "openai", label: "OpenAI", description: "Cloud hosted model" },
  { value: "gemini", label: "Gemini", description: "Google Gemini Flash" },
  { value: "codex", label: "Codex CLI", description: "OpenAI Codex via CLI" },
  { value: "claude", label: "Claude CLI", description: "Anthropic Claude via CLI" },
];

const CLI_PROVIDER_OPTIONS: { value: CLIProvider; label: string; description: string }[] = [
  { value: "codex", label: "Codex CLI", description: "OpenAI Codex CLI" },
  { value: "claude", label: "Claude CLI", description: "Anthropic Claude CLI" },
];

const SUMMARIZER_MODE_OPTIONS: { value: SummarizerMode; label: string; description: string }[] = [
  { value: "full", label: "Full", description: "Generate notes and topics" },
  { value: "topics-only", label: "Topics Only", description: "Faster, topics extraction" },
];

const SUMMARIZER_PROVIDER_OPTIONS: { value: SummarizerProvider; label: string; description: string }[] = [
  { value: "ollama", label: "Ollama", description: "Locally hosted model" },
  { value: "openai", label: "OpenAI", description: "Cloud hosted model" },
  { value: "codex", label: "Codex CLI", description: "OpenAI Codex via CLI" },
  { value: "claude", label: "Claude CLI", description: "Anthropic Claude via CLI" },
];

function getFileType(filename?: string): "pdf" | "pptx" | "text" {
  if (!filename) return "pdf";
  const ext = filename.toLowerCase().split(".").pop();
  if (ext === "pptx") return "pptx";
  if (ext === "md" || ext === "txt") return "text";
  return "pdf";
}

export function ConfigForm({ onSubmit, initialConfig, filename }: ConfigFormProps) {
  const fileType = getFileType(filename);

  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [config, setConfig] = useState<PipelineConfig>(
    initialConfig ?? { ...DEFAULT_PIPELINE_CONFIG }
  );
  const [advancedVisionConfig, setAdvancedVisionConfig] = useState(false);
  const [advancedExtractionConfig, setAdvancedExtractionConfig] = useState(false);

  // Dynamic steps based on file type AND config options
  const steps = useMemo<StepDef[]>(() => {
    if (fileType === "pdf" || fileType === "pptx") {
      const extractionDescription = fileType === "pptx"
        ? "Configure PowerPoint parsing options"
        : "Configure PDF parsing options";
      // Extract Only: skip vision and summarization
      if (config.extract_only) {
        return [
          { id: "extraction", title: "Extraction", description: extractionDescription },
          { id: "review", title: "Review", description: "Review and submit configuration" },
        ];
      }
      return [
        { id: "extraction", title: "Extraction", description: extractionDescription },
        { id: "vision", title: "Vision", description: "" },
        { id: "summarization", title: "Summarization", description: "Configure summarization options" },
        { id: "review", title: "Review", description: "Review and submit configuration" },
      ];
    }
    // Text files (.md, .txt)
    return [
      { id: "chunking", title: "Chunking", description: "Set text chunking parameters" },
      { id: "summarization", title: "Summarization", description: "Configure summarization options" },
      { id: "review", title: "Review", description: "Review and submit configuration" },
    ];
  }, [fileType, config.extract_only]);

  // Clamp step index if steps array shrinks (e.g., Extract Only enabled)
  const safeStepIndex = Math.min(currentStepIndex, steps.length - 1);
  const currentStep = steps[safeStepIndex];
  const isLastStep = safeStepIndex === steps.length - 1;
  const isFirstStep = safeStepIndex === 0;

  // Sync state if clamping occurred
  React.useEffect(() => {
    if (currentStepIndex !== safeStepIndex) {
      setCurrentStepIndex(safeStepIndex);
    }
  }, [currentStepIndex, safeStepIndex]);

  const updateConfig = <K extends keyof PipelineConfig>(
    key: K,
    value: PipelineConfig[K]
  ) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleNext = () => {
    if (!isLastStep) {
      setCurrentStepIndex((prev) => prev + 1);
    }
  };

  const handleBack = () => {
    if (!isFirstStep) {
      setCurrentStepIndex((prev) => prev - 1);
    }
  };

  const handleSubmit = () => {
    onSubmit(config);
  };

  const renderStepIndicator = () => (
    <div className="flex items-center justify-between mb-8">
      {steps.map((step, index) => (
        <div key={step.id} className="flex items-center">
          <div
            className={cn(
              "flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium transition-colors",
              safeStepIndex === index
                ? "bg-primary text-primary-foreground"
                : safeStepIndex > index
                ? "bg-primary/20 text-primary"
                : "bg-muted text-muted-foreground"
            )}
          >
            {safeStepIndex > index ? (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              index + 1
            )}
          </div>
          <span
            className={cn(
              "ml-2 text-sm font-medium hidden sm:inline",
              safeStepIndex === index ? "text-foreground" : "text-muted-foreground"
            )}
          >
            {step.title}
          </span>
          {index < steps.length - 1 && (
            <div
              className={cn(
                "w-8 sm:w-16 h-0.5 mx-2",
                safeStepIndex > index ? "bg-primary/40" : "bg-muted"
              )}
            />
          )}
        </div>
      ))}
    </div>
  );

  const renderExtractionStep = () => {
    // Dynamic labels based on file type
    const isPptx = fileType === "pptx";
    const skipTablesLabel = isPptx ? "Skip Slide Notes" : "Skip Tables";
    const skipTablesDescription = isPptx
      ? "Do not extract speaker notes from slides"
      : "Do not extract tables from PDF";
    const skipImagesLabel = isPptx ? "Skip Slide Screenshots" : "Skip Images";
    const skipImagesDescription = isPptx
      ? "Do not render slide screenshots for vision processing"
      : "Do not extract images from PDF";
    const textOnlyDescription = isPptx
      ? "Extract only text, skip notes and slide screenshots"
      : "Extract only text, skip tables and images";

    return (
      <div className="space-y-6">
        <div className="space-y-4">
          {/* Extract Only - always visible */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="extract_only" className="text-base">Extract Only</Label>
              <p className="text-sm text-muted-foreground">
                Stop after extraction, skip vision and summarization
              </p>
            </div>
            <Switch
              id="extract_only"
              checked={config.extract_only}
              onCheckedChange={(checked) => updateConfig("extract_only", checked)}
            />
          </div>

          {/* Advanced Mode toggle */}
          <div className="flex items-center justify-between pt-4 border-t">
            <div className="space-y-0.5">
              <Label htmlFor="advanced_extraction" className="text-sm font-medium">Advanced Mode</Label>
              <p className="text-xs text-muted-foreground">
                Fine-tune extraction options
              </p>
            </div>
            <Switch
              id="advanced_extraction"
              checked={advancedExtractionConfig}
              onCheckedChange={setAdvancedExtractionConfig}
            />
          </div>

          {/* Advanced options - only visible when Advanced Mode is on */}
          {advancedExtractionConfig && (
            <div className="space-y-4 pt-4 border-t">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="text_only" className="text-base">Text Only</Label>
                  <p className="text-sm text-muted-foreground">
                    {textOnlyDescription}
                  </p>
                </div>
                <Switch
                  id="text_only"
                  checked={config.text_only}
                  onCheckedChange={(checked) => updateConfig("text_only", checked)}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="skip_tables" className="text-base">{skipTablesLabel}</Label>
                  <p className="text-sm text-muted-foreground">
                    {skipTablesDescription}
                  </p>
                </div>
                <Switch
                  id="skip_tables"
                  checked={config.skip_tables}
                  onCheckedChange={(checked) => updateConfig("skip_tables", checked)}
                  disabled={config.text_only}
                />
              </div>

              {/* PPTX-only: Skip Slide Tables option */}
              {isPptx && (
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="skip_pptx_tables" className="text-base">Skip Slide Tables</Label>
                    <p className="text-sm text-muted-foreground">
                      Do not extract tables from slides
                    </p>
                  </div>
                  <Switch
                    id="skip_pptx_tables"
                    checked={config.skip_pptx_tables}
                    onCheckedChange={(checked) => updateConfig("skip_pptx_tables", checked)}
                    disabled={config.text_only}
                  />
                </div>
              )}

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="skip_images" className="text-base">{skipImagesLabel}</Label>
                  <p className="text-sm text-muted-foreground">
                    {skipImagesDescription}
                  </p>
                </div>
                <Switch
                  id="skip_images"
                  checked={config.skip_images}
                  onCheckedChange={(checked) => updateConfig("skip_images", checked)}
                  disabled={config.text_only}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="keep_base64_images" className="text-base">Keep Base64 Images</Label>
                  <p className="text-sm text-muted-foreground">
                    Include image data in output JSON (increases file size)
                  </p>
                </div>
                <Switch
                  id="keep_base64_images"
                  checked={config.keep_base64_images}
                  onCheckedChange={(checked) => updateConfig("keep_base64_images", checked)}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderVisionProviderSelector = (
    selectedValue: VisionMode,
    onSelect: (value: VisionMode) => void,
    excludeNone = false
  ) => {
    const options = excludeNone
      ? VISION_MODE_OPTIONS.filter((opt) => opt.value !== "none")
      : VISION_MODE_OPTIONS;

    return (
      <div className="grid gap-3">
        {options.map(({ value, label, description }) => (
          <div
            key={value}
            onClick={() => onSelect(value)}
            className={cn(
              "flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-colors",
              selectedValue === value
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50"
            )}
          >
            <div className="space-y-0.5">
              <p className="font-medium">{label}</p>
              <p className="text-sm text-muted-foreground">{description}</p>
            </div>
            <div
              className={cn(
                "w-4 h-4 rounded-full border-2 transition-colors",
                selectedValue === value
                  ? "border-primary bg-primary"
                  : "border-muted-foreground"
              )}
            >
              {selectedValue === value && (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary-foreground" />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const visionEnabled = config.vision_mode !== "none";

  const renderVisionStep = () => (
    <div className="space-y-6">
      {/* Vision Enabled Toggle */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground">
            Select to analyze images and visual content in PDF pages
          </p>
        </div>
        <Switch
          id="vision_enabled"
          checked={visionEnabled}
          onCheckedChange={(checked) => {
            if (checked) {
              updateConfig("vision_mode", "ollama");
            } else {
              updateConfig("vision_mode", "none");
              setAdvancedVisionConfig(false);
              updateConfig("vision_classifier_mode", undefined);
              updateConfig("vision_extractor_mode", undefined);
            }
          }}
        />
      </div>

      {visionEnabled && (
        <>
          {/* Settings row: Advanced toggle + DPI */}
          <div className="flex flex-col sm:flex-row gap-4 pt-4 border-t">
            <div className="flex items-center justify-between sm:justify-start gap-4 flex-1">
              <div className="space-y-0.5">
                <Label htmlFor="advanced_vision" className="text-sm font-medium">Advanced Mode</Label>
                <p className="text-xs text-muted-foreground">
                  Separate classifier & extractor
                </p>
              </div>
              <Switch
                id="advanced_vision"
                checked={advancedVisionConfig}
                onCheckedChange={(checked) => {
                  setAdvancedVisionConfig(checked);
                  if (!checked) {
                    updateConfig("vision_classifier_mode", undefined);
                    updateConfig("vision_extractor_mode", undefined);
                  }
                }}
              />
            </div>

            <div className="flex-1">
              <Label htmlFor="pdf_image_dpi" className="text-sm font-medium">PDF Image DPI</Label>
              <Select
                value={config.pdf_image_dpi.toString()}
                onValueChange={(value) => updateConfig("pdf_image_dpi", parseInt(value, 10) as PdfImageDpi)}
              >
                <SelectTrigger id="pdf_image_dpi" className="w-full mt-1">
                  <SelectValue placeholder="Select DPI" />
                </SelectTrigger>
                <SelectContent>
                  {DPI_OPTIONS.map(({ value, label }) => (
                    <SelectItem key={value} value={value.toString()}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Provider Selection */}
          {!advancedVisionConfig ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label className="text-base">Vision Provider</Label>
                <p className="text-sm text-muted-foreground">
                  Choose a provider for image and visual content analysis
                </p>
              </div>
              {renderVisionProviderSelector(
                config.vision_mode,
                (value) => updateConfig("vision_mode", value),
                true
              )}
            </div>
          ) : (
            <div className="space-y-6">
              <div className="p-4 rounded-lg bg-muted/50">
                <p className="text-sm text-muted-foreground">
                  Use a fast local model for classification (YES/NO detection) and a quality cloud model for extraction (detailed content analysis).
                </p>
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-base">Classification Provider</Label>
                  <p className="text-sm text-muted-foreground">
                    Fast identification of pages with visual content (lower cost)
                  </p>
                </div>
                {renderVisionProviderSelector(
                  config.vision_classifier_mode || config.vision_mode,
                  (value) => updateConfig("vision_classifier_mode", value),
                  true
                )}
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label className="text-base">Extraction Provider</Label>
                  <p className="text-sm text-muted-foreground">
                    Detailed visual content analysis (used only on flagged pages)
                  </p>
                </div>
                {renderVisionProviderSelector(
                  config.vision_extractor_mode || config.vision_mode,
                  (value) => {
                    updateConfig("vision_extractor_mode", value);
                    // When codex is selected for extraction, force classifier to ollama
                    if (value === "codex") {
                      updateConfig("vision_classifier_mode", "ollama");
                    }
                  },
                  true
                )}
              </div>

              {/* Detailed Extraction Toggle */}
              <div className="flex items-center justify-between p-4 rounded-lg bg-muted/30 border">
                <div className="space-y-0.5">
                  <Label htmlFor="vision_detailed_extraction" className="text-base">
                    Detailed Extraction
                  </Label>
                  <p className="text-sm text-muted-foreground">
                    Run extraction 3 times per page and synthesize results for comprehensive coverage
                  </p>
                </div>
                <Switch
                  id="vision_detailed_extraction"
                  checked={config.vision_detailed_extraction}
                  onCheckedChange={(checked) => updateConfig("vision_detailed_extraction", checked)}
                />
              </div>

            </div>
          )}
        </>
      )}
    </div>
  );

  const renderChunkingStep = () => (
    <div className="space-y-6">
      <div className="p-4 rounded-lg bg-muted/50 mb-4">
        <p className="text-sm text-muted-foreground">
          Text documents are split into chunks for processing. Each chunk will be summarized individually.
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="chunk_size" className="text-base">Chunk Size</Label>
        <p className="text-sm text-muted-foreground">
          Maximum characters per text chunk (1000-10000)
        </p>
        <Input
          id="chunk_size"
          type="number"
          min={1000}
          max={10000}
          step={100}
          value={config.chunk_size}
          onChange={(e) => {
            const value = parseInt(e.target.value, 10);
            if (!isNaN(value) && value >= 1000 && value <= 10000) {
              updateConfig("chunk_size", value);
            }
          }}
        />
        <p className="text-xs text-muted-foreground">
          Current: {config.chunk_size} characters
        </p>
      </div>

      <div className="space-y-2">
        <Label htmlFor="chunk_overlap" className="text-base">Chunk Overlap</Label>
        <p className="text-sm text-muted-foreground">
          Characters of overlap between chunks (0-500)
        </p>
        <Input
          id="chunk_overlap"
          type="number"
          min={0}
          max={500}
          step={10}
          value={config.chunk_overlap}
          onChange={(e) => {
            const value = parseInt(e.target.value, 10);
            if (!isNaN(value) && value >= 0 && value <= 500) {
              updateConfig("chunk_overlap", value);
            }
          }}
        />
        <p className="text-xs text-muted-foreground">
          Current: {config.chunk_overlap} characters overlap
        </p>
      </div>
    </div>
  );

  const renderSummarizationStep = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          <Label htmlFor="run_summarization" className="text-base">Enable Summarization</Label>
          <p className="text-sm text-muted-foreground">
            Generate summaries for {fileType === "pdf" || fileType === "pptx" ? "extracted pages" : "text chunks"}
          </p>
        </div>
        <Switch
          id="run_summarization"
          checked={config.run_summarization}
          onCheckedChange={(checked) => updateConfig("run_summarization", checked)}
        />
      </div>

      {config.run_summarization && (
        <div className="space-y-6">
          {/* AI Provider Selection */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-base">AI Provider</Label>
              <p className="text-sm text-muted-foreground">
                Choose the LLM provider for summarization
              </p>
            </div>
            <div className="grid gap-3">
              {SUMMARIZER_PROVIDER_OPTIONS.map(({ value, label, description }) => (
                <div
                  key={value}
                  onClick={() => updateConfig("summarizer_provider", value)}
                  className={cn(
                    "flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-colors",
                    config.summarizer_provider === value
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  )}
                >
                  <div className="space-y-0.5">
                    <p className="font-medium">{label}</p>
                    <p className="text-sm text-muted-foreground">{description}</p>
                  </div>
                  <div
                    className={cn(
                      "w-4 h-4 rounded-full border-2 transition-colors",
                      config.summarizer_provider === value
                        ? "border-primary bg-primary"
                        : "border-muted-foreground"
                    )}
                  >
                    {config.summarizer_provider === value && (
                      <div className="w-full h-full flex items-center justify-center">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary-foreground" />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Summarization Mode Selection */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-base">Summarization Mode</Label>
              <p className="text-sm text-muted-foreground">
                Choose the type of summary output
              </p>
            </div>
            <div className="grid gap-3">
              {SUMMARIZER_MODE_OPTIONS.map(({ value, label, description }) => (
                <div
                  key={value}
                  onClick={() => updateConfig("summarizer_mode", value)}
                  className={cn(
                    "flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-colors",
                    config.summarizer_mode === value
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  )}
                >
                  <div className="space-y-0.5">
                    <p className="font-medium">{label}</p>
                    <p className="text-sm text-muted-foreground">{description}</p>
                  </div>
                  <div
                    className={cn(
                      "w-4 h-4 rounded-full border-2 transition-colors",
                      config.summarizer_mode === value
                        ? "border-primary bg-primary"
                        : "border-muted-foreground"
                    )}
                  >
                    {config.summarizer_mode === value && (
                      <div className="w-full h-full flex items-center justify-center">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary-foreground" />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Detailed Summarization Toggle */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-muted/30 border">
            <div className="space-y-0.5">
              <Label htmlFor="summarizer_detailed_extraction" className="text-base">
                Detailed Summarization
              </Label>
              <p className="text-sm text-muted-foreground">
                Run summarization 3 times per page and synthesize results for comprehensive coverage
              </p>
            </div>
            <Switch
              id="summarizer_detailed_extraction"
              checked={config.summarizer_detailed_extraction}
              onCheckedChange={(checked) => updateConfig("summarizer_detailed_extraction", checked)}
            />
          </div>
        </div>
      )}
    </div>
  );

  const renderReviewStep = () => (
    <div className="space-y-6">
      <ConfigSummary config={config} fileType={fileType} />
      <div className="p-4 rounded-lg bg-muted/50">
        <p className="text-sm text-muted-foreground">
          Review your configuration above. Click Submit to start processing your document
          with these settings.
        </p>
      </div>
    </div>
  );

  const renderCurrentStep = () => {
    switch (currentStep.id) {
      case "extraction":
        return renderExtractionStep();
      case "vision":
        return renderVisionStep();
      case "chunking":
        return renderChunkingStep();
      case "summarization":
        return renderSummarizationStep();
      case "review":
        return renderReviewStep();
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Configure Pipeline</CardTitle>
        <CardDescription>
          Set up your document processing options in {steps.length} steps
        </CardDescription>
      </CardHeader>
      <CardContent>
        {renderStepIndicator()}

        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-1">{currentStep.title}</h3>
          <p className="text-sm text-muted-foreground">{currentStep.description}</p>
        </div>

        <div className="min-h-[300px]">{renderCurrentStep()}</div>

        <div className="flex justify-between mt-8 pt-6 border-t">
          <Button
            variant="outline"
            onClick={handleBack}
            disabled={isFirstStep}
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
            Back
          </Button>

          {!isLastStep ? (
            <Button onClick={handleNext}>
              Next
              <svg
                className="w-4 h-4 ml-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </Button>
          ) : (
            <Button onClick={handleSubmit}>
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              Submit
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
