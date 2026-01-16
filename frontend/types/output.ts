/**
 * Pipeline output models.
 *
 * Matches the JSON output schema from run.py pipeline.
 * See CLAUDE.md Output Schema section.
 */

/**
 * Document-level metadata from the pipeline output.
 */
export interface DocumentMetadata {
  /** Unique document identifier (format: doc_{filename}) */
  document_id: string;

  /** Original filename of the processed document */
  filename: string;

  /** Total number of pages/chunks in the document */
  total_pages: number;

  /** Additional document metadata (extraction-specific) */
  metadata: Record<string, unknown>;
}

/**
 * Individual page/chunk output from the pipeline.
 * Contains extracted content and optional summarization results.
 */
export interface PageOutput {
  /** Unique chunk identifier (format: chunk_{n}) */
  chunk_id: string;

  /** Document title/filename reference */
  doc_title: string;

  /** Extracted text content from the page */
  text: string;

  /** Extracted tables as 2D string arrays */
  tables: string[][];

  /** Base64-encoded images extracted from the page */
  image_base64: string[];

  /** Vision-extracted text description of images (if vision processing enabled) */
  image_text?: string | null;

  /** Detailed extraction intermediate results (if vision_detailed_extraction enabled) */
  image_text_1?: string | null;
  image_text_2?: string | null;
  image_text_3?: string | null;

  /** Vision classifier result: true if page contains meaningful graphics */
  image_classifier?: boolean;

  /** Summary notes as bullet points (if summarization enabled) */
  summary_notes?: string[] | null;

  /** Topic tags extracted from the content (if summarization enabled) */
  summary_topics?: string[] | null;

  /** Relevancy score 0-1 indicating summary quality (if summarization enabled) */
  summary_relevancy?: number;

  /** Detailed summarization intermediate results (if summarizer_detailed_extraction enabled) */
  summary_notes_1?: string[] | null;
  summary_notes_2?: string[] | null;
  summary_notes_3?: string[] | null;
}

/**
 * Stage-specific metrics from pipeline execution.
 */
export interface StageMetrics {
  /** Duration in milliseconds */
  duration_ms: number;
  /** Number of pages processed */
  pages_processed: number;
  /** Total tokens used */
  tokens: number;
  /** Pages with images (vision stage only) */
  pages_with_images?: number;
  /** Pages classified as having graphics (vision stage only) */
  classified_count?: number;
  /** Pages with extracted visual content (vision stage only) */
  extracted_count?: number;
  /** Average relevancy score (summarization stage only) */
  avg_relevancy?: number;
  /** Total summarization attempts (summarization stage only) */
  total_attempts?: number;
}

/**
 * Pipeline execution metrics.
 */
export interface PipelineMetrics {
  /** Total pipeline duration in milliseconds */
  total_duration_ms: number;
  /** Total tokens used across all stages */
  total_tokens: number;
  /** Per-stage metrics */
  stages: {
    extraction: StageMetrics;
    vision: StageMetrics;
    summarization: StageMetrics;
  };
  /** Configuration used for processing */
  config: {
    vision_mode: string | null;
    summarizer_provider: string | null;
    summarizer_mode: string | null;
  };
}

/**
 * Complete pipeline output structure.
 * Contains document metadata and all processed pages.
 */
export interface DocumentOutput {
  /** Document-level metadata */
  document: DocumentMetadata;

  /** Array of processed pages/chunks */
  pages: PageOutput[];

  /** Pipeline execution metrics (if available) */
  metrics?: PipelineMetrics;
}
