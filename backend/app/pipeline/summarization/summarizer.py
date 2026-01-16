"""
SummarizeService for document summarization.

Ports the summarization logic from summarizer.py into an async service
with the 30-attempt quality loop, model tier rotation, and progress callbacks.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Awaitable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI, OpenAIError

# Logger for summarization stage
summarization_logger = logging.getLogger("app.pipeline.summarization")

from app.config import settings
from app.pipeline.summarization.schemas import (
    SummarizerConfig,
    PageContext,
    SummaryResult,
)
from app.pipeline.summarization.quality_validator import QualityValidator


@dataclass
class LLMResponse:
    """Response from an LLM call with token usage."""

    content: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in reversed(lines):
            if "," in line and not line.endswith(":"):
                text = line
                break
    # Split by comma and clean
    topics = [t.strip() for t in text.split(",") if t.strip()]
    return topics


class SummarizeService:
    """Service for summarizing document pages.

    Implements the quality loop from summarizer.py:
    - 30 max attempts with decreasing quality threshold
    - Model tier rotation: tier 1 (1-10), tier 2 (11-20), tier 3 (21-30)
    - Quality check via LLM after each summary attempt
    """

    def __init__(self, config: SummarizerConfig):
        """Initialize SummarizeService.

        Args:
            config: SummarizerConfig with model and quality settings
        """
        self.config = config
        self.validator = QualityValidator(
            threshold_high=config.quality_threshold_high,
            threshold_low=config.quality_threshold_low,
        )

        # Create clients based on provider
        self._provider = config.provider

        if self._provider == "openai":
            # Use OpenAI API directly
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
            self._client_1 = OpenAI(api_key=api_key)
            self._client_2 = self._client_1  # OpenAI doesn't need separate clients
            self._openai_model = settings.openai_summarizer_model
            summarization_logger.info("Using OpenAI provider with model: %s", self._openai_model)
        else:
            # Use Ollama (OpenAI-compatible API)
            base_url_1 = config.ollama_base_url_1 or settings.ollama_url_1
            base_url_2 = config.ollama_base_url_2 or settings.ollama_url_2
            self._client_1 = OpenAI(base_url=base_url_1, api_key="ollama")
            self._client_2 = OpenAI(base_url=base_url_2, api_key="ollama")
            self._openai_model = None
            summarization_logger.info("Using Ollama provider with URLs: %s, %s", base_url_1, base_url_2)

        # Load prompts lazily
        self._notes_prompt: Optional[str] = None
        self._notes_prompt_fallback: Optional[str] = None
        self._topics_prompt: Optional[str] = None
        self._synthesis_prompt: Optional[str] = None

        # Semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(config.batch_size)

        # Text splitter for sub-chunking large page contexts
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.sub_chunk_size,
            chunk_overlap=config.sub_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    @property
    def notes_prompt(self) -> str:
        """Load notes prompt lazily."""
        if self._notes_prompt is None:
            self._notes_prompt = _load_prompt("summarizer-notes-prompt.txt")
        return self._notes_prompt

    @property
    def notes_prompt_fallback(self) -> str:
        """Load fallback notes prompt lazily."""
        if self._notes_prompt_fallback is None:
            try:
                self._notes_prompt_fallback = _load_prompt(
                    "summarizer-notes-prompt-fallback.txt"
                )
            except FileNotFoundError:
                # Fall back to primary prompt if fallback doesn't exist
                self._notes_prompt_fallback = self.notes_prompt
        return self._notes_prompt_fallback

    @property
    def topics_prompt(self) -> str:
        """Load topics prompt lazily."""
        if self._topics_prompt is None:
            self._topics_prompt = _load_prompt("summarizer-topics-prompt.txt")
        return self._topics_prompt

    @property
    def synthesis_prompt(self) -> str:
        """Load synthesis prompt lazily."""
        if self._synthesis_prompt is None:
            self._synthesis_prompt = _load_prompt("summarizer-synthesis.txt")
        return self._synthesis_prompt

    def _get_client_for_tier(self, tier: int) -> OpenAI:
        """Get the appropriate client for a model tier."""
        if tier <= 2:
            return self._client_1
        return self._client_2

    def _get_model_for_tier(self, tier: int) -> str:
        """Get the model name for a tier."""
        # OpenAI provider uses a single model for all tiers
        if self._provider == "openai" and self._openai_model:
            return self._openai_model
        # Ollama uses tiered models
        if tier == 1:
            return self.config.model_tier_1
        elif tier == 2:
            return self.config.model_tier_2
        return self.config.model_tier_3

    def _split_context(self, context: str) -> list[str]:
        """Split context into overlapping sub-chunks.

        Uses RecursiveCharacterTextSplitter to split at semantic boundaries
        (paragraphs, lines, words) while maintaining overlap for context continuity.

        Args:
            context: Full page context string

        Returns:
            List of sub-chunks (always at least one, even for small contexts)
        """
        if not context or not context.strip():
            return []

        chunks = self._splitter.split_text(context)
        return chunks if chunks else [context]

    async def _call_llm(
        self, prompt: str, tier: int = 1, max_retries: int = 3
    ) -> LLMResponse:
        """Call LLM with retry logic.

        Uses asyncio.to_thread to avoid blocking the event loop.
        Returns LLMResponse with content and token usage.
        """
        client = self._get_client_for_tier(tier)
        model = self._get_model_for_tier(tier)

        def sync_call() -> LLMResponse:
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw_content = response.choices[0].message.content
                    content = raw_content.strip() if raw_content else ""

                    # Extract token usage if available
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
                    if hasattr(response, "usage") and response.usage:
                        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                        total_tokens = getattr(response.usage, "total_tokens", 0) or 0

                    return LLMResponse(
                        content=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                except OpenAIError:
                    if attempt < max_retries - 1:
                        continue
                    return LLMResponse(content=None)
            return LLMResponse(content=None)

        return await asyncio.to_thread(sync_call)

    async def _quality_check(
        self, text: str, summary: str, tier: int
    ) -> tuple[float, LLMResponse]:
        """Check quality of summary against original text.

        Returns:
            Tuple of (relevancy score, LLMResponse with token usage)
        """
        prompt = f"""Compare the following original text and summary.

On a scale from 0% to 100%, how accurately does the summary represent the original text?

Original Text:
\"\"\"{text}\"\"\"


Summary:
\"\"\"{summary}\"\"\"


Provide only the percentage number."""

        llm_response = await self._call_llm(prompt, tier=tier)
        if llm_response.content is None:
            return 0.0, llm_response

        try:
            # Extract digits only
            score = float("".join(filter(str.isdigit, llm_response.content)))
            return score, llm_response
        except ValueError:
            return 0.0, llm_response

    async def summarize_page(self, context: PageContext) -> SummaryResult:
        """Summarize a single page with quality loop.

        Args:
            context: PageContext with page content

        Returns:
            SummaryResult with notes, topics, relevancy, and token usage
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

        # Detailed extraction mode - run 3 passes and synthesize
        if self.config.detailed_extraction:
            return await self._summarize_page_detailed(context)

        # Token accumulators
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        # Topics-only mode
        if self.config.mode == "topics-only":
            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                summary_notes=None,
                summary_topics=None,
                summary_relevancy=0,
                attempts_used=0,
            )

        # Build comprehensive context
        full_context = context.build_context()

        # Topics-only mode
        if self.config.mode == "topics-only":
            topics, topic_tokens = await self._generate_topics(full_context[:2000], tier=1)
            total_prompt_tokens += topic_tokens.prompt_tokens
            total_completion_tokens += topic_tokens.completion_tokens
            total_tokens += topic_tokens.total_tokens
            return SummaryResult(
                page_number=context.page_number,
                chunk_id=context.chunk_id,
                summary_notes=None,
                summary_topics=topics,
                summary_relevancy=0,
                attempts_used=1,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
            )

        # Full mode: split context into sub-chunks and process each
        sub_chunks = self._split_context(full_context)

        # Log sub-chunking info
        summarization_logger.debug(
            "Sub-chunking - page=%d, context_len=%d, chunks=%d",
            context.page_number,
            len(full_context),
            len(sub_chunks),
        )

        all_notes: list[str] = []
        total_attempts = 0
        chunk_relevancies: list[float] = []

        for chunk_idx, sub_chunk in enumerate(sub_chunks):
            chunk_notes = None
            chunk_relevancy = 0.0
            attempt = 0

            while attempt < self.config.max_attempts:
                attempt += 1
                tier = self.validator.get_model_tier_for_attempt(attempt)
                prompt_key = self.validator.get_prompt_key_for_attempt(attempt)

                # Select prompt
                if prompt_key == "primary":
                    prompt = self.notes_prompt.format(chunk=sub_chunk)
                else:
                    prompt = self.notes_prompt_fallback.format(chunk=sub_chunk)

                # Generate summary
                llm_response = await self._call_llm(prompt, tier=tier)
                total_prompt_tokens += llm_response.prompt_tokens
                total_completion_tokens += llm_response.completion_tokens
                total_tokens += llm_response.total_tokens

                if llm_response.content is None:
                    continue

                # Quality check against the sub-chunk
                chunk_relevancy, qc_response = await self._quality_check(
                    sub_chunk, llm_response.content, tier=3
                )
                total_prompt_tokens += qc_response.prompt_tokens
                total_completion_tokens += qc_response.completion_tokens
                total_tokens += qc_response.total_tokens

                if self.validator.validate(chunk_relevancy, attempt):
                    chunk_notes = _clean_notes_output(llm_response.content)
                    break

            total_attempts += attempt
            if chunk_notes:
                all_notes.extend(chunk_notes)
                chunk_relevancies.append(chunk_relevancy)

            summarization_logger.debug(
                "Sub-chunk complete - page=%d, chunk=%d/%d, notes=%d, relevancy=%.1f%%, attempts=%d",
                context.page_number,
                chunk_idx + 1,
                len(sub_chunks),
                len(chunk_notes) if chunk_notes else 0,
                chunk_relevancy,
                attempt,
            )

        # Calculate average relevancy across all chunks
        avg_relevancy = (
            sum(chunk_relevancies) / len(chunk_relevancies)
            if chunk_relevancies
            else 0.0
        )

        summarization_logger.info(
            "Page summarization complete - page=%d, chunks=%d, total_notes=%d, avg_relevancy=%.1f%%",
            context.page_number,
            len(sub_chunks),
            len(all_notes),
            avg_relevancy,
        )

        # Generate topics from combined notes
        topics = None
        if all_notes:
            notes_text = "\n".join(f"* {note}" for note in all_notes)
            topics, topic_response = await self._generate_topics(notes_text, tier=1)
            total_prompt_tokens += topic_response.prompt_tokens
            total_completion_tokens += topic_response.completion_tokens
            total_tokens += topic_response.total_tokens

        return SummaryResult(
            page_number=context.page_number,
            chunk_id=context.chunk_id,
            summary_notes=all_notes if all_notes else None,
            summary_topics=topics,
            summary_relevancy=avg_relevancy,
            attempts_used=total_attempts,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
        )

    async def _generate_topics(
        self, notes_text: str, tier: int = 1, max_attempts: int = 3
    ) -> tuple[Optional[list[str]], LLMResponse]:
        """Generate topics from notes text.

        Returns:
            Tuple of (topics list, LLMResponse with token usage)
        """
        prompt = self.topics_prompt.replace("<<NOTES>>", notes_text)

        # Accumulate tokens across all attempts
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for attempt in range(max_attempts):
            current_tier = min(tier + attempt, 3)
            llm_response = await self._call_llm(prompt, tier=current_tier)
            total_prompt_tokens += llm_response.prompt_tokens
            total_completion_tokens += llm_response.completion_tokens
            total_tokens += llm_response.total_tokens

            if llm_response.content:
                return _clean_topics_output(llm_response.content), LLMResponse(
                    content=None,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_tokens,
                )

        return None, LLMResponse(
            content=None,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
        )

    async def _synthesize_notes(
        self,
        notes_1: list[str],
        notes_2: list[str],
        notes_3: list[str],
        tier: int = 1,
    ) -> tuple[list[str], LLMResponse]:
        """Synthesize 3 sets of notes into a comprehensive result.

        Args:
            notes_1: First extraction notes
            notes_2: Second extraction notes
            notes_3: Third extraction notes
            tier: Model tier to use

        Returns:
            Tuple of (synthesized notes list, LLMResponse with token usage)
        """
        # Format the 3 extractions
        extraction_1 = "\n".join(f"* {note}" for note in notes_1) if notes_1 else "* (no notes extracted)"
        extraction_2 = "\n".join(f"* {note}" for note in notes_2) if notes_2 else "* (no notes extracted)"
        extraction_3 = "\n".join(f"* {note}" for note in notes_3) if notes_3 else "* (no notes extracted)"

        synthesis_input = f"""## EXTRACTION 1
{extraction_1}

## EXTRACTION 2
{extraction_2}

## EXTRACTION 3
{extraction_3}"""

        prompt = f"{self.synthesis_prompt}\n\n{synthesis_input}"

        llm_response = await self._call_llm(prompt, tier=tier)

        if llm_response.content:
            synthesized_notes = _clean_notes_output(llm_response.content)
            return synthesized_notes, llm_response

        # Fallback: return union of all notes if synthesis fails
        all_notes = list(notes_1 or []) + list(notes_2 or []) + list(notes_3 or [])
        return all_notes, llm_response

    async def _summarize_page_single_pass(
        self, context: PageContext
    ) -> tuple[list[str], float, int, int, int, int]:
        """Run a single pass of note extraction for a page.

        Args:
            context: PageContext with page content

        Returns:
            Tuple of (notes, relevancy, attempts, prompt_tokens, completion_tokens, total_tokens)
        """
        # Build comprehensive context
        full_context = context.build_context()

        # Split context into sub-chunks
        sub_chunks = self._split_context(full_context)

        all_notes: list[str] = []
        total_attempts = 0
        chunk_relevancies: list[float] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for sub_chunk in sub_chunks:
            chunk_notes = None
            chunk_relevancy = 0.0
            attempt = 0

            while attempt < self.config.max_attempts:
                attempt += 1
                tier = self.validator.get_model_tier_for_attempt(attempt)
                prompt_key = self.validator.get_prompt_key_for_attempt(attempt)

                # Select prompt
                if prompt_key == "primary":
                    prompt = self.notes_prompt.format(chunk=sub_chunk)
                else:
                    prompt = self.notes_prompt_fallback.format(chunk=sub_chunk)

                # Generate summary
                llm_response = await self._call_llm(prompt, tier=tier)
                total_prompt_tokens += llm_response.prompt_tokens
                total_completion_tokens += llm_response.completion_tokens
                total_tokens += llm_response.total_tokens

                if llm_response.content is None:
                    continue

                # Quality check against the sub-chunk
                chunk_relevancy, qc_response = await self._quality_check(
                    sub_chunk, llm_response.content, tier=3
                )
                total_prompt_tokens += qc_response.prompt_tokens
                total_completion_tokens += qc_response.completion_tokens
                total_tokens += qc_response.total_tokens

                if self.validator.validate(chunk_relevancy, attempt):
                    chunk_notes = _clean_notes_output(llm_response.content)
                    break

            total_attempts += attempt
            if chunk_notes:
                all_notes.extend(chunk_notes)
                chunk_relevancies.append(chunk_relevancy)

        # Calculate average relevancy
        avg_relevancy = (
            sum(chunk_relevancies) / len(chunk_relevancies)
            if chunk_relevancies
            else 0.0
        )

        return (
            all_notes,
            avg_relevancy,
            total_attempts,
            total_prompt_tokens,
            total_completion_tokens,
            total_tokens,
        )

    async def _summarize_page_detailed(self, context: PageContext) -> SummaryResult:
        """Summarize a page using detailed extraction (3 passes + synthesis).

        Runs extraction 3 times and synthesizes the results for comprehensive coverage.

        Args:
            context: PageContext with page content

        Returns:
            SummaryResult with notes, topics, relevancy, intermediate results, and token usage
        """
        summarization_logger.info(
            "Starting detailed extraction - page=%d, running 3 passes",
            context.page_number,
        )

        # Token accumulators
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_attempts = 0

        # Run extraction 3 times
        notes_1, rel_1, attempts_1, pt_1, ct_1, tt_1 = await self._summarize_page_single_pass(context)
        total_prompt_tokens += pt_1
        total_completion_tokens += ct_1
        total_tokens += tt_1
        total_attempts += attempts_1

        summarization_logger.debug(
            "Detailed extraction pass 1 complete - page=%d, notes=%d, relevancy=%.1f%%",
            context.page_number,
            len(notes_1),
            rel_1,
        )

        notes_2, rel_2, attempts_2, pt_2, ct_2, tt_2 = await self._summarize_page_single_pass(context)
        total_prompt_tokens += pt_2
        total_completion_tokens += ct_2
        total_tokens += tt_2
        total_attempts += attempts_2

        summarization_logger.debug(
            "Detailed extraction pass 2 complete - page=%d, notes=%d, relevancy=%.1f%%",
            context.page_number,
            len(notes_2),
            rel_2,
        )

        notes_3, rel_3, attempts_3, pt_3, ct_3, tt_3 = await self._summarize_page_single_pass(context)
        total_prompt_tokens += pt_3
        total_completion_tokens += ct_3
        total_tokens += tt_3
        total_attempts += attempts_3

        summarization_logger.debug(
            "Detailed extraction pass 3 complete - page=%d, notes=%d, relevancy=%.1f%%",
            context.page_number,
            len(notes_3),
            rel_3,
        )

        # Synthesize the 3 extractions
        synthesized_notes, synthesis_response = await self._synthesize_notes(
            notes_1, notes_2, notes_3, tier=1
        )
        total_prompt_tokens += synthesis_response.prompt_tokens
        total_completion_tokens += synthesis_response.completion_tokens
        total_tokens += synthesis_response.total_tokens

        # Calculate average relevancy from all 3 passes
        avg_relevancy = (rel_1 + rel_2 + rel_3) / 3.0

        summarization_logger.info(
            "Detailed extraction complete - page=%d, synthesized_notes=%d, avg_relevancy=%.1f%%",
            context.page_number,
            len(synthesized_notes),
            avg_relevancy,
        )

        # Generate topics from synthesized notes
        topics = None
        if synthesized_notes:
            notes_text = "\n".join(f"* {note}" for note in synthesized_notes)
            topics, topic_response = await self._generate_topics(notes_text, tier=1)
            total_prompt_tokens += topic_response.prompt_tokens
            total_completion_tokens += topic_response.completion_tokens
            total_tokens += topic_response.total_tokens

        return SummaryResult(
            page_number=context.page_number,
            chunk_id=context.chunk_id,
            summary_notes=synthesized_notes if synthesized_notes else None,
            summary_topics=topics,
            summary_relevancy=avg_relevancy,
            attempts_used=total_attempts,
            summary_notes_1=notes_1 if notes_1 else None,
            summary_notes_2=notes_2 if notes_2 else None,
            summary_notes_3=notes_3 if notes_3 else None,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
        )

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
