"""
CLISummarizer - Summarization using CLI executors (Codex or Claude).

Bypasses the quality loop and trusts CLI output directly (relevancy=100).
Uses temp files for input/output with CLI subprocess.
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

logger = logging.getLogger("app.pipeline.summarization")


class CLISummarizer:
    """Summarizer using CLI executors (Codex or Claude).

    Unlike SummarizeService, this class bypasses the quality loop
    and trusts CLI output directly (relevancy=100).
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

    async def summarize_page(self, context: PageContext) -> SummaryResult:
        """Summarize a single page using CLI executor.

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

        temp_dir = None
        temp_file = None

        try:
            # Create temp directory for CLI to work in
            temp_dir = tempfile.mkdtemp(prefix=f"{self._executor.name}_summarize_")
            temp_file = Path(temp_dir) / f"page_{context.page_number}.json"

            # Build comprehensive context
            full_context = context.build_context()

            # Save context to temp file
            page_data = {
                "page_number": context.page_number,
                "chunk_id": context.chunk_id,
                "text": context.text,
                "tables": context.tables,
                "image_text": context.image_text,
                "full_context": full_context,
            }
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(page_data, f)

            # Build prompt based on mode
            if self.config.mode == "topics-only":
                prompt = f"""Read the page content from {temp_file.name} and extract key topics.

The JSON file contains 'full_context' which is the complete page content.

Return a JSON object with exactly this structure:
{{
  "topics": ["topic1", "topic2", "topic3"]
}}

Extract 3-8 key topics that represent the main subjects discussed.
Topics should be concise (1-3 words each).
Return ONLY the JSON, no explanations or markdown."""
            else:
                prompt = f"""Read the page content from {temp_file.name} and create a summary.

The JSON file contains 'full_context' which is the complete page content.

Return a JSON object with exactly this structure:
{{
  "notes": [
    "First key point or finding",
    "Second key point or finding",
    "Additional important information"
  ],
  "topics": ["topic1", "topic2", "topic3"]
}}

Notes should be:
- 3-10 bullet points capturing the most important information
- Complete sentences that stand alone
- Focused on facts, findings, and key details

Topics should be:
- 3-8 key subjects or themes
- Concise (1-3 words each)

Return ONLY the JSON, no explanations or markdown."""

            # Call CLI executor
            response = await self._executor.execute(
                prompt=prompt,
                working_dir=temp_dir,
            )

            # Parse JSON response
            notes = None
            topics = None

            if response:
                try:
                    # Try to parse as JSON
                    data = json.loads(response)
                    notes = data.get("notes")
                    topics = data.get("topics")
                except json.JSONDecodeError:
                    # If not valid JSON, try to extract manually
                    logger.warning(
                        "%s response not valid JSON (page=%d), attempting extraction",
                        self._executor.name,
                        context.page_number,
                    )
                    # Look for JSON in the response
                    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                            notes = data.get("notes")
                            topics = data.get("topics")
                        except json.JSONDecodeError:
                            pass

            # Topics-only mode: notes should be None
            if self.config.mode == "topics-only":
                notes = None

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
            # Cleanup temp files
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            if temp_dir:
                try:
                    Path(temp_dir).rmdir()
                except OSError:
                    pass

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
