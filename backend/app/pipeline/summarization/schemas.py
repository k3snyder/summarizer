"""
Schemas for summarization.
"""

from dataclasses import dataclass, field
from typing import Optional


def _get_default_model_tier_1() -> str:
    from app.config import settings
    return settings.summarizer_model_tier_1


def _get_default_model_tier_2() -> str:
    from app.config import settings
    return settings.summarizer_model_tier_2


def _get_default_model_tier_3() -> str:
    from app.config import settings
    return settings.summarizer_model_tier_3


def _get_default_quality_high() -> int:
    from app.config import settings
    return settings.summarizer_quality_threshold_high


def _get_default_quality_low() -> int:
    from app.config import settings
    return settings.summarizer_quality_threshold_low


def _get_default_max_attempts() -> int:
    from app.config import settings
    return settings.summarizer_max_attempts


@dataclass
class SummarizerConfig:
    """Configuration for summarization service.

    Matches the quality loop behavior from summarizer.py:
    - 30 max attempts with decreasing quality threshold
    - Model tier rotation: tier 1 (1-10), tier 2 (11-20), tier 3 (21-30)
    - Quality threshold: 90% for first 5 attempts, 85% thereafter

    Defaults are loaded from centralized settings in app.config.
    """

    # Quality thresholds
    quality_threshold_high: int = field(default_factory=_get_default_quality_high)
    quality_threshold_low: int = field(default_factory=_get_default_quality_low)

    # Retry configuration
    max_attempts: int = field(default_factory=_get_default_max_attempts)
    max_retries_per_call: int = 3
    retry_delay: float = 5.0

    # Model configuration (tiers for fallback rotation)
    model_tier_1: str = field(default_factory=_get_default_model_tier_1)
    model_tier_2: str = field(default_factory=_get_default_model_tier_2)
    model_tier_3: str = field(default_factory=_get_default_model_tier_3)

    # Ollama configuration
    ollama_base_url_1: Optional[str] = None
    ollama_base_url_2: Optional[str] = None

    # Processing mode
    mode: str = "full"  # "full" or "topics-only"

    # Provider selection
    provider: str = "ollama"  # "ollama" or "openai"

    # Batch processing
    batch_size: int = 5

    # Sub-chunking configuration
    sub_chunk_size: int = 3000
    sub_chunk_overlap: int = 120

    # Debug logging
    debug_log: bool = False

    # Detailed extraction mode
    detailed_extraction: bool = False  # Run summarization 3x and synthesize

    # Insight mode: two-stage synthesis + insight extraction for knowledge base
    insight_mode: bool = False  # Stage 1: synthesize all context, Stage 2: extract insights


def _is_noise_table(table: list | dict, empty_threshold: float = 0.50) -> bool:
    """Check if a table is likely OCR noise rather than real tabular data.

    Detects tables that are actually fragmented body text incorrectly
    detected as tables by PDF extraction tools.

    Args:
        table: Table data (2D list or dict format)
        empty_threshold: Fraction of empty cells that indicates noise (default 0.50)

    Returns:
        True if table appears to be noise, False if it looks like real data
    """
    total_cells = 0
    empty_cells = 0

    if isinstance(table, list):
        for row in table:
            if isinstance(row, list):
                for cell in row:
                    total_cells += 1
                    if not str(cell).strip():
                        empty_cells += 1
    elif isinstance(table, dict):
        # Dict format: {"columns": [...], "data": [[...], ...]}
        data = table.get("data", [])
        for row in data:
            if isinstance(row, list):
                for cell in row:
                    total_cells += 1
                    if not str(cell).strip():
                        empty_cells += 1

    if total_cells == 0:
        return True  # Empty table is noise

    empty_ratio = empty_cells / total_cells
    return empty_ratio > empty_threshold


@dataclass
class PageContext:
    """Context for a page to be summarized.

    Contains all content extracted from the page that should be
    considered during summarization.
    """

    page_number: int
    chunk_id: str
    text: str

    # Optional content from other stages
    tables: Optional[list] = None
    image_text: Optional[str] = None
    image_classifier: Optional[bool] = None

    def build_context(self) -> str:
        """Build comprehensive context string for summarization.

        Combines text, tables, and image_text into a single context.
        Filters out noise tables (>50% empty cells) that are typically
        OCR artifacts from PDF extraction.
        """
        sections = []

        # Main text content
        if self.text:
            sections.append(self.text.strip())

        # Table data (supports both dict format and 2D array format)
        # Filter out noise tables that are just fragmented OCR text
        if self.tables:
            table_texts = []
            for i, table in enumerate(self.tables):
                # Skip noise tables (>50% empty cells)
                if _is_noise_table(table):
                    continue

                # Handle 2D array format (from PPTX): [["H1", "H2"], ["A", "B"]]
                if isinstance(table, list):
                    if len(table) >= 1:
                        header = " | ".join(str(c) for c in table[0] if c)
                        rows = [" | ".join(str(cell) for cell in row) for row in table[1:]]
                        table_text = f"Table {i+1}:\n{header}\n" + "\n".join(rows)
                        table_texts.append(table_text)
                # Handle dict format (from PDF): {"columns": [...], "data": [...]}
                elif isinstance(table, dict):
                    columns = table.get("columns", [])
                    data = table.get("data", [])
                    if columns and data:
                        header = " | ".join(str(c) for c in columns if c)
                        rows = [" | ".join(str(cell) for cell in row) for row in data]
                        table_text = f"Table {i+1}:\n{header}\n" + "\n".join(rows)
                        table_texts.append(table_text)
            if table_texts:
                sections.append("\n\n[TABLES]\n" + "\n\n".join(table_texts))

        # Vision OCR content
        if self.image_text and self.image_text.strip():
            sections.append("\n\n[VISUAL CONTENT]\n" + self.image_text.strip())

        return "\n\n".join(sections)


@dataclass
class SummaryResult:
    """Result from summarizing a single page."""

    page_number: int
    chunk_id: str
    summary_notes: Optional[list[str]]
    summary_topics: Optional[list[str]]
    summary_relevancy: float

    # Debugging info
    attempts_used: int = 1
    error: Optional[str] = None

    # Detailed extraction intermediate results (when detailed_extraction=True)
    summary_notes_1: Optional[list[str]] = None
    summary_notes_2: Optional[list[str]] = None
    summary_notes_3: Optional[list[str]] = None

    # Insight mode results (when insight_mode=True)
    # Stage 1: Comprehensive synthesis of all context
    context_synthesis: Optional[str] = None
    # Stage 2: Focused insights for knowledge base
    core_understanding: Optional[str] = None
    key_insights: Optional[list[str]] = None
    connections: Optional[list[str]] = None
    implications: Optional[dict[str, list[str]]] = None  # {"users": [...], "technical": [...], "business": [...]}
    knowledge_tags: Optional[dict[str, list[str]]] = None  # {"primary": [...], "related": [...], "entities": [...]}
    summary_statement: Optional[str] = None

    # Token usage metrics (total across all API calls for this page)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
