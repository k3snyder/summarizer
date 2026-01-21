"""
CLISummarizer - Summarization using CLI executors (Codex or Claude).

Bypasses the quality loop and trusts CLI output directly (relevancy=100).
Uses the same prompt templates as the standard summarizer for consistency.
"""

import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional, Callable, Awaitable

from app.pipeline.cli import CLIExecutionError, CLIExecutorBase, get_cli_executor
from app.pipeline.summarization.schemas import (
    SummarizerConfig,
    PageContext,
    SummaryResult,
)


def _load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the backend/prompts directory."""
    # Navigate from backend/app/pipeline/summarization to backend/
    backend_root = Path(__file__).parent.parent.parent.parent
    prompt_path = backend_root / "prompts" / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _clean_notes_output(output: str) -> list[str]:
    """Extract bullet points from notes response."""
    text = output.strip()
    # Remove <summary> tags if present
    text = re.sub(r"</?summary>", "", text)
    # Extract lines that are bullet points
    lines = text.strip().split("\n")
    bullet_points = [line.strip()[2:] for line in lines if line.strip().startswith("* ")]
    return bullet_points


def _clean_topics_output(output: str) -> list[str]:
    """Clean and normalize comma-separated topics."""
    text = output.strip()
    # If there are newlines, take the last line
    if "\n" in text:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in reversed(lines):
            if "," in line and not line.endswith(":"):
                text = line
                break
    # Split by comma and clean
    topics = [t.strip() for t in text.split(",") if t.strip()]
    return topics

logger = logging.getLogger("app.pipeline.summarization")


class CLISummarizer:
    """Summarizer using CLI executors (Codex or Claude).

    Unlike SummarizeService, this class bypasses the quality loop
    and trusts CLI output directly (relevancy=100).

    Uses the same prompt templates as the standard summarizer for
    consistent, high-quality summarization output.
    """

    def __init__(
        self,
        config: SummarizerConfig,
        cli_provider: str = "codex",
        executor: Optional[CLIExecutorBase] = None,
    ):
        """Initialize CLISummarizer.

        Args:
            config: SummarizerConfig with batch settings
            cli_provider: CLI provider name ('codex' or 'claude')
            executor: Optional pre-configured executor (for testing)
        """
        self.config = config
        self._cli_provider = cli_provider
        self._executor = executor or get_cli_executor(cli_provider)
        self._semaphore = asyncio.Semaphore(config.batch_size)

        # Load prompts lazily
        self._notes_prompt: Optional[str] = None
        self._topics_prompt: Optional[str] = None
        self._context_synthesis_prompt: Optional[str] = None
        self._insight_prompt: Optional[str] = None

    @property
    def notes_prompt(self) -> str:
        """Load notes prompt lazily."""
        if self._notes_prompt is None:
            self._notes_prompt = _load_prompt("summarizer-notes-prompt.txt")
        return self._notes_prompt

    @property
    def topics_prompt(self) -> str:
        """Load topics prompt lazily."""
        if self._topics_prompt is None:
            self._topics_prompt = _load_prompt("summarizer-topics-prompt.txt")
        return self._topics_prompt

    @property
    def context_synthesis_prompt(self) -> str:
        """Load context synthesis prompt lazily (for insight mode stage 1)."""
        if self._context_synthesis_prompt is None:
            self._context_synthesis_prompt = _load_prompt("summarizer-context-synthesis.txt")
        return self._context_synthesis_prompt

    @property
    def insight_prompt(self) -> str:
        """Load insight extraction prompt lazily (for insight mode stage 2)."""
        if self._insight_prompt is None:
            self._insight_prompt = _load_prompt("summarizer-insight-prompt.txt")
        return self._insight_prompt

    async def summarize_page(self, context: PageContext) -> SummaryResult:
        """Summarize a single page using CLI executor.

        Uses the same prompt templates as the standard summarizer for
        high-quality, detailed summaries that capture names, numbers,
        and nuanced insights.

        Args:
            context: PageContext with page content

        Returns:
            SummaryResult with notes, topics, and relevancy=100
        """
        # Skip mode
        if self.config.mode == "skip":
            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                summary_notes=None,
                summary_topics=None,
                summary_relevancy=0,
                attempts_used=0,
            )

        # Insight mode: two-stage synthesis + insight extraction
        if self.config.insight_mode:
            return await self._summarize_page_insight(context)

        # Build comprehensive context from page data
        full_context = context.build_context()

        # Create temp directory for CLI working directory
        temp_dir = tempfile.mkdtemp(prefix=f"{self._executor.name}_summarize_")

        try:
            notes = None
            topics = None

            # Topics-only mode
            if self.config.mode == "topics-only":
                # Use topics prompt template
                topics_prompt = self.topics_prompt.replace("<<NOTES>>", full_context[:2000])
                response = await self._executor.execute(
                    prompt=topics_prompt,
                    working_dir=temp_dir,
                )
                if response:
                    topics = _clean_topics_output(response)

                return SummaryResult(
                    page_number=context.page_number,
                    chunk_id=context.chunk_id,
                    summary_notes=None,
                    summary_topics=topics,
                    summary_relevancy=100,  # Trust CLI output
                    attempts_used=1,
                )

            # Full mode: use notes prompt template for detailed summarization
            notes_prompt = self.notes_prompt.format(chunk=full_context)
            response = await self._executor.execute(
                prompt=notes_prompt,
                working_dir=temp_dir,
            )

            if response:
                # Parse bullet point output (same as standard summarizer)
                notes = _clean_notes_output(response)

                # If no bullet points found, try JSON fallback
                if not notes:
                    try:
                        data = json.loads(response)
                        notes = data.get("notes")
                    except json.JSONDecodeError:
                        # Try extracting JSON from response
                        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
                        if json_match:
                            try:
                                data = json.loads(json_match.group())
                                notes = data.get("notes")
                            except json.JSONDecodeError:
                                pass

            # Generate topics from notes
            if notes:
                notes_text = "\n".join(f"* {note}" for note in notes)
                topics_prompt = self.topics_prompt.replace("<<NOTES>>", notes_text)
                topics_response = await self._executor.execute(
                    prompt=topics_prompt,
                    working_dir=temp_dir,
                )
                if topics_response:
                    topics = _clean_topics_output(topics_response)

            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                summary_notes=notes,
                summary_topics=topics,
                summary_relevancy=100,  # Trust CLI output
                attempts_used=1,
            )

        except CLIExecutionError as e:
            logger.error(
                "%s CLI summarization failed (page=%d): %s",
                self._executor.name,
                context.page_number,
                str(e),
            )
            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                summary_notes=None,
                summary_topics=None,
                summary_relevancy=0,
                attempts_used=1,
                error=f"{self._executor.name} CLI error: {str(e)}",
            )

        except Exception as e:
            logger.error(
                "%s CLI summarization unexpected error (page=%d): %s",
                self._executor.name,
                context.page_number,
                str(e),
            )
            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                summary_notes=None,
                summary_topics=None,
                summary_relevancy=0,
                attempts_used=1,
                error=f"Error: {str(e)}",
            )

        finally:
            # Cleanup temp directory
            if temp_dir:
                try:
                    Path(temp_dir).rmdir()
                except OSError:
                    pass

    async def _summarize_page_insight(self, context: PageContext) -> SummaryResult:
        """Summarize a page using standard flow + two-stage insight extraction via CLI.

        First runs the standard notes/topics flow, then enriches with insight extraction:
        - Standard: Generate notes and topics
        - Stage 1: Synthesize all context into comprehensive inventory
        - Stage 2: Extract focused insights, connections, and implications
        - Append key_insights to summary_notes for enriched output

        Args:
            context: PageContext with page content

        Returns:
            SummaryResult with both standard notes and insight mode fields populated
        """
        logger.info(
            "Starting CLI insight extraction - page=%d, running standard + 2 insight stages",
            context.page_number,
        )

        # Build comprehensive context from all sources
        full_context = context.build_context()

        # Create temp directory for CLI working directory
        temp_dir = tempfile.mkdtemp(prefix=f"{self._executor.name}_insight_")

        try:
            # === STANDARD FLOW: Notes and Topics ===
            logger.debug(
                "Standard flow: Generating notes - page=%d",
                context.page_number,
            )

            notes_prompt = self.notes_prompt.format(chunk=full_context)
            notes_response = await self._executor.execute(
                prompt=notes_prompt,
                working_dir=temp_dir,
            )

            notes = []
            if notes_response:
                notes = _clean_notes_output(notes_response)

            # Generate topics from notes
            topics = []
            if notes:
                notes_text = "\n".join(f"* {note}" for note in notes)
                topics_prompt = self.topics_prompt.replace("<<NOTES>>", notes_text)
                topics_response = await self._executor.execute(
                    prompt=topics_prompt,
                    working_dir=temp_dir,
                )
                if topics_response:
                    topics = _clean_topics_output(topics_response)

            logger.debug(
                "Standard flow complete - page=%d, notes=%d, topics=%d",
                context.page_number,
                len(notes),
                len(topics),
            )

            # === INSIGHT STAGE 1: Context Synthesis ===
            logger.debug(
                "Insight Stage 1: Context synthesis - page=%d, context_len=%d",
                context.page_number,
                len(full_context),
            )

            synthesis_prompt = self.context_synthesis_prompt.replace("{chunk}", full_context)
            context_synthesis = await self._executor.execute(
                prompt=synthesis_prompt,
                working_dir=temp_dir,
            )

            if not context_synthesis:
                context_synthesis = ""

            logger.debug(
                "Insight Stage 1 complete - page=%d, synthesis_len=%d",
                context.page_number,
                len(context_synthesis),
            )

            # === INSIGHT STAGE 2: Insight Extraction ===
            logger.debug(
                "Insight Stage 2: Insight extraction - page=%d",
                context.page_number,
            )

            insight_prompt = self.insight_prompt.replace("{chunk}", context_synthesis)
            insight_output = await self._executor.execute(
                prompt=insight_prompt,
                working_dir=temp_dir,
            )

            if not insight_output:
                insight_output = ""

            # Parse insight output
            parsed = self._parse_insight_output(insight_output)

            logger.info(
                "CLI insight extraction complete - page=%d, notes=%d, insights=%d, connections=%d",
                context.page_number,
                len(notes),
                len(parsed.get("key_insights", [])),
                len(parsed.get("connections", [])),
            )

            # Append key_insights to notes for enriched output
            key_insights = parsed.get("key_insights", [])
            combined_notes = notes + key_insights if key_insights else notes

            # Use primary tags from insights to enrich topics if available
            insight_topics = parsed.get("knowledge_tags", {}).get("primary", [])
            combined_topics = topics + [t for t in insight_topics if t not in topics] if insight_topics else topics

            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                # Combined fields: standard notes + key_insights appended
                summary_notes=combined_notes if combined_notes else None,
                summary_topics=combined_topics if combined_topics else None,
                summary_relevancy=100,  # Trust CLI output
                attempts_used=4,  # notes + topics + synthesis + insights
                # Insight mode specific fields
                context_synthesis=context_synthesis if context_synthesis else None,
                core_understanding=parsed.get("core_understanding"),
                key_insights=key_insights if key_insights else None,
                connections=parsed.get("connections"),
                implications=parsed.get("implications"),
                knowledge_tags=parsed.get("knowledge_tags"),
                summary_statement=parsed.get("summary_statement"),
            )

        except CLIExecutionError as e:
            logger.error(
                "%s CLI insight extraction failed (page=%d): %s",
                self._executor.name,
                context.page_number,
                str(e),
            )
            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                summary_notes=None,
                summary_topics=None,
                summary_relevancy=0,
                attempts_used=2,
                error=f"{self._executor.name} CLI error: {str(e)}",
            )

        except Exception as e:
            logger.error(
                "%s CLI insight extraction unexpected error (page=%d): %s",
                self._executor.name,
                context.page_number,
                str(e),
            )
            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                summary_notes=None,
                summary_topics=None,
                summary_relevancy=0,
                attempts_used=2,
                error=f"Error: {str(e)}",
            )

        finally:
            # Cleanup temp directory
            if temp_dir:
                try:
                    Path(temp_dir).rmdir()
                except OSError:
                    pass

    def _parse_insight_output(self, output: str) -> dict:
        """Parse structured insight output into component fields.

        Args:
            output: Raw markdown output from insight prompt

        Returns:
            Dict with parsed fields: core_understanding, key_insights, connections,
            implications, knowledge_tags, summary_statement
        """
        result: dict = {
            "core_understanding": None,
            "key_insights": [],
            "connections": [],
            "implications": {"users": [], "technical": [], "business": []},
            "knowledge_tags": {"primary": [], "related": [], "entities": []},
            "summary_statement": None,
        }

        if not output:
            return result

        lines = output.split("\n")
        current_section = None
        current_subsection = None
        buffer: list[str] = []

        for line in lines:
            line_stripped = line.strip()

            # Detect section headers
            if line_stripped.startswith("## CORE UNDERSTANDING"):
                current_section = "core_understanding"
                current_subsection = None
                buffer = []
            elif line_stripped.startswith("## KEY INSIGHTS"):
                # Save previous section
                if current_section == "core_understanding" and buffer:
                    result["core_understanding"] = "\n".join(buffer).strip()
                current_section = "key_insights"
                current_subsection = None
                buffer = []
            elif line_stripped.startswith("## CONNECTIONS"):
                current_section = "connections"
                current_subsection = None
                buffer = []
            elif line_stripped.startswith("## IMPLICATIONS"):
                current_section = "implications"
                current_subsection = None
                buffer = []
            elif line_stripped.startswith("## KNOWLEDGE BASE TAGS"):
                current_section = "knowledge_tags"
                current_subsection = None
                buffer = []
            elif line_stripped.startswith("## SUMMARY STATEMENT"):
                current_section = "summary_statement"
                current_subsection = None
                buffer = []

            # Detect subsections within implications and knowledge_tags
            elif current_section == "implications":
                if "For Users" in line_stripped or "Customers" in line_stripped:
                    current_subsection = "users"
                elif "Technical" in line_stripped:
                    current_subsection = "technical"
                elif "Business" in line_stripped or "Strategy" in line_stripped:
                    current_subsection = "business"
                elif line_stripped.startswith("- ") and current_subsection:
                    result["implications"][current_subsection].append(line_stripped[2:])

            elif current_section == "knowledge_tags":
                if "Primary" in line_stripped:
                    current_subsection = "primary"
                elif "Related" in line_stripped:
                    current_subsection = "related"
                elif "Entity" in line_stripped or "Entities" in line_stripped:
                    current_subsection = "entities"
                elif "Domain" in line_stripped:
                    current_subsection = "domain"
                elif current_subsection and ":" in line_stripped:
                    # Parse inline tags like "**Primary topics**: topic1, topic2"
                    parts = line_stripped.split(":", 1)
                    if len(parts) > 1:
                        tags = [t.strip() for t in parts[1].split(",") if t.strip()]
                        if current_subsection in result["knowledge_tags"]:
                            result["knowledge_tags"][current_subsection].extend(tags)

            # Collect content for current section
            elif current_section == "core_understanding":
                if line_stripped and not line_stripped.startswith("#"):
                    buffer.append(line_stripped)

            elif current_section == "key_insights":
                if line_stripped.startswith("* **"):
                    # Extract insight text after the bold label
                    insight = line_stripped[2:]  # Remove "* "
                    result["key_insights"].append(insight)
                elif line_stripped.startswith("- **"):
                    insight = line_stripped[2:]  # Remove "- "
                    result["key_insights"].append(insight)

            elif current_section == "connections":
                if line_stripped.startswith("- "):
                    result["connections"].append(line_stripped[2:])

            elif current_section == "summary_statement":
                if line_stripped and not line_stripped.startswith("#"):
                    buffer.append(line_stripped)

        # Save final section if it was summary_statement
        if current_section == "summary_statement" and buffer:
            result["summary_statement"] = "\n".join(buffer).strip()
        elif current_section == "core_understanding" and buffer:
            result["core_understanding"] = "\n".join(buffer).strip()

        return result

    async def summarize_pages_batch(
        self,
        contexts: list[PageContext],
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> list[SummaryResult]:
        """Summarize multiple pages with rate limiting.

        Args:
            contexts: List of PageContext objects
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of SummaryResult for each page
        """
        completed = 0
        results = []

        async def process_with_semaphore(context: PageContext) -> SummaryResult:
            nonlocal completed
            async with self._semaphore:
                result = await self.summarize_page(context)
                completed += 1
                if progress_callback:
                    await progress_callback(completed, len(contexts))
                return result

        tasks = [process_with_semaphore(ctx) for ctx in contexts]
        results = await asyncio.gather(*tasks)

        return list(results)


# Backwards compatibility alias
CodexSummarizer = CLISummarizer
