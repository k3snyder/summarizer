/**
 * Frontend type definitions.
 *
 * Re-exports all types for convenient imports:
 * import { PipelineConfig, JobStatus, DocumentOutput } from '@/types';
 */

export type {
  PdfImageDpi,
  VisionMode,
  SummarizerMode,
  SummarizerProvider,
  CLIProvider,
  PipelineConfig,
} from './pipeline';

export { DEFAULT_PIPELINE_CONFIG } from './pipeline';

export type {
  JobStatusValue,
  JobStatus,
  JobCreateResponse,
} from './job';

export type {
  DocumentMetadata,
  PageOutput,
  DocumentOutput,
  StageMetrics,
  PipelineMetrics,
} from './output';
