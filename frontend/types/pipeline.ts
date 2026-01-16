/**
 * Pipeline configuration for document processing.
 *
 * Matches backend/app/models/config.py PipelineConfig
 */

/**
 * Valid DPI values for PDF image extraction.
 * 72 = fast, 300 = high quality
 */
export type PdfImageDpi = 72 | 144 | 200 | 300;

/**
 * Vision processing provider options.
 * 'none' disables vision processing.
 * 'codex' and 'claude' use CLI executors.
 */
export type VisionMode = 'none' | 'deepseek' | 'gemini' | 'openai' | 'ollama' | 'codex' | 'claude';

/**
 * CLI provider options for vision and summarization.
 * Used when vision_mode or summarizer_provider is 'codex' or 'claude'.
 */
export type CLIProvider = 'codex' | 'claude';

/**
 * Summarization mode options.
 * full = notes + topics, topics-only = fast, skip = passthrough
 */
export type SummarizerMode = 'full' | 'topics-only' | 'skip';

/**
 * Summarizer provider options.
 * ollama = local privacy-first, openai = cloud API, codex/claude = CLI executors
 */
export type SummarizerProvider = 'ollama' | 'openai' | 'codex' | 'claude';

/**
 * Configuration for the document processing pipeline.
 * Controls extraction, vision processing, and summarization stages.
 */
export interface PipelineConfig {
  /** Stop after extraction stage, skip vision and summarization */
  extract_only: boolean;

  /** Skip table extraction from documents (for PDF); skip speaker notes (for PPTX) */
  skip_tables: boolean;

  /** Skip image extraction from documents */
  skip_images: boolean;

  /** Skip table extraction from PPTX slides (PPTX only) */
  skip_pptx_tables: boolean;

  /** Extract only text, skip tables and images */
  text_only: boolean;

  /** DPI for PDF image extraction (72=fast, 300=high quality) */
  pdf_image_dpi: PdfImageDpi;

  /** Vision processing provider (none disables vision processing) */
  vision_mode: VisionMode;

  /** Vision classifier provider (inherits from vision_mode if not set) */
  vision_classifier_mode?: VisionMode;

  /** Vision extractor provider (inherits from vision_mode if not set) */
  vision_extractor_mode?: VisionMode;

  /** Run vision extraction 3 times per page and synthesize results for comprehensive coverage */
  vision_detailed_extraction: boolean;

  /** CLI provider for vision processing (when vision_mode is 'codex' or 'claude') */
  vision_cli_provider?: CLIProvider;

  /** Text chunk size for summarization */
  chunk_size: number;

  /** Overlap between text chunks */
  chunk_overlap: number;

  /** Enable summarization stage */
  run_summarization: boolean;

  /** Summarization mode (full=notes+topics, topics-only=fast, skip=passthrough) */
  summarizer_mode: SummarizerMode;

  /** Summarizer provider (ollama=local, openai=cloud) */
  summarizer_provider: SummarizerProvider;

  /** Run summarization 3 times per page and synthesize results for comprehensive coverage */
  summarizer_detailed_extraction: boolean;

  /** CLI provider for summarization (when summarizer_provider is 'codex' or 'claude') */
  summarizer_cli_provider?: CLIProvider;

  /** Keep base64-encoded images in final output (default: strip to reduce size) */
  keep_base64_images: boolean;
}

/**
 * Default pipeline configuration values.
 * Matches backend Pydantic model defaults.
 */
export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  extract_only: false,
  skip_tables: false,
  skip_images: false,
  skip_pptx_tables: false,
  text_only: false,
  pdf_image_dpi: 200,
  vision_mode: 'ollama',
  vision_detailed_extraction: false,
  chunk_size: 3000,
  chunk_overlap: 80,
  run_summarization: true,
  summarizer_mode: 'full',
  summarizer_provider: 'ollama',
  summarizer_detailed_extraction: false,
  keep_base64_images: false,
};
