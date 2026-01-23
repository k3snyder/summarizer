"""
CLI vision provider implementation.

Uses CLI executors (Codex or Claude) for vision extraction via subprocess.
Classification always returns True (CLI handles full extraction).
"""

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

from app.pipeline.cli import CLIExecutionError, CLIExecutorBase, get_cli_executor
from app.pipeline.vision.providers.base import VisionProviderBase
from app.pipeline.vision.schemas import ClassificationResult, ExtractionResult


logger = logging.getLogger(__name__)


def _load_prompt(prompt_name: str) -> str:
    """Load a prompt template from the backend/prompts directory."""
    backend_root = Path(__file__).parent.parent.parent.parent.parent
    prompt_path = backend_root / "prompts" / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


class CLIVisionProvider(VisionProviderBase):
    """CLI vision provider.

    Uses CLI executors (Codex or Claude) for vision extraction.
    Classification always returns True since CLI handles full extraction.
    """

    def __init__(
        self,
        cli_provider: str = "codex",
        executor: Optional[CLIExecutorBase] = None,
    ):
        """Initialize CLI vision provider.

        Args:
            cli_provider: CLI provider name ('codex' or 'claude')
            executor: Optional pre-configured executor (for testing)
        """
        self._cli_provider = cli_provider
        self._executor = executor or get_cli_executor(cli_provider)

        # Load extraction prompt lazily
        self._extract_prompt: Optional[str] = None

    @property
    def extract_prompt(self) -> str:
        """Load extraction prompt lazily."""
        if self._extract_prompt is None:
            self._extract_prompt = _load_prompt("vision-extract.txt")
        return self._extract_prompt

    async def classify(
        self, image_base64: str, page_number: int, chunk_id: str  # noqa: ARG002
    ) -> ClassificationResult:
        """Classify whether an image contains graphical content.

        For CLI providers, always returns True since classification is bypassed.
        CLI handles full extraction directly.
        """
        return ClassificationResult(
            page_number=page_number,
            chunk_id=chunk_id,
            has_graphics=True,
        )

    async def extract(
        self, image_base64: str, page_number: int, chunk_id: str
    ) -> ExtractionResult:
        """Extract visual content from an image using CLI executor.

        Saves image as PNG file, invokes CLI with extraction prompt,
        and returns the response.
        """
        temp_dir = None
        temp_file = None

        try:
            # Create temp directory for CLI to work in
            temp_dir = tempfile.mkdtemp(prefix=f"{self._executor.name}_vision_")
            temp_file = Path(temp_dir) / f"page_{page_number}.png"

            logger.info(
                "CLIVisionProvider.extract starting (page=%d, chunk=%s, cli=%s, temp_dir=%s)",
                page_number,
                chunk_id,
                self._executor.name,
                temp_dir,
            )

            # Decode base64 and save as proper PNG file
            # Source data is JPEG from extraction stage - convert to actual PNG
            # so file extension matches format and Codex can view correctly
            if "," in image_base64 and image_base64.startswith("data:"):
                image_base64 = image_base64.split(",", 1)[1]

            image_bytes = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_bytes))
            img.save(temp_file, format="PNG")

            file_size = temp_file.stat().st_size
            logger.debug(
                "Saved image to %s (size: %d bytes, dimensions: %dx%d)",
                temp_file, file_size, img.width, img.height,
            )

            # Build prompt - simple and direct
            prompt = f"""Analyze the image file {temp_file.name} in the current directory.

{self.extract_prompt}

Return ONLY the markdown analysis, no explanations or preamble."""

            # Call CLI executor
            logger.info("Calling %s executor for page %d...", self._executor.name, page_number)
            response = await self._executor.execute(
                prompt=prompt,
                working_dir=temp_dir,
            )
            logger.info("CLI executor returned for page %d (response length: %d)", page_number, len(response) if response else 0)

            return ExtractionResult(
                page_number=page_number,
                chunk_id=chunk_id,
                image_text=response if response else None,
            )

        except CLIExecutionError as e:
            logger.error(
                "%s CLI extraction failed (page=%d): %s",
                self._executor.name,
                page_number,
                str(e),
            )
            return ExtractionResult(
                page_number=page_number,
                chunk_id=chunk_id,
                image_text=None,
                error=f"{self._executor.name} CLI error: {str(e)}",
            )

        except Exception as e:
            logger.error(
                "%s CLI extraction unexpected error (page=%d): %s",
                self._executor.name,
                page_number,
                str(e),
            )
            return ExtractionResult(
                page_number=page_number,
                chunk_id=chunk_id,
                image_text=None,
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


# Backwards compatibility alias
CodexCLIVisionProvider = CLIVisionProvider
