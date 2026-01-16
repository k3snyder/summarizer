"""
Claude CLI executor implementation.

Invokes Claude CLI via subprocess with --print flag for output capture.
Uses --dangerously-skip-permissions for autonomous execution.
"""

import asyncio
import logging
import shutil
import subprocess

from app.pipeline.cli.base import CLIExecutionError, CLIExecutorBase

logger = logging.getLogger("app.pipeline.cli.claude")


class ClaudeCLIExecutor(CLIExecutorBase):
    """Claude CLI executor implementation.

    Executes prompts via the Claude CLI tool with stdout capture.
    """

    def __init__(
        self,
        model: str = "sonnet",
        timeout: int = 600,
    ):
        """Initialize Claude CLI executor.

        Args:
            model: Claude model to use (default 'sonnet', options: opus, sonnet, haiku)
            timeout: Default timeout in seconds (default 600 = 10 minutes)
        """
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "claude"

    def is_available(self) -> bool:
        return shutil.which("claude") is not None

    async def execute(
        self,
        prompt: str,
        working_dir: str,
        timeout: int | None = None,
    ) -> str:
        """Execute a prompt via Claude CLI.

        Args:
            prompt: The prompt to send to Claude
            working_dir: Working directory for Claude to operate in
            timeout: Timeout in seconds (None uses instance default)

        Returns:
            The response text from Claude

        Raises:
            CLIExecutionError: If Claude CLI is not found, times out, or returns error
        """
        if not self.is_available():
            raise CLIExecutionError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
                exit_code=-1,
                stderr="Claude CLI not installed",
            )

        effective_timeout = timeout if timeout is not None else self._timeout

        # Build command
        cmd = [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--model",
            self._model,
            "-p",
            prompt,
        ]

        logger.info(
            "Executing claude command in %s (model=%s, timeout=%ds)",
            working_dir,
            self._model,
            effective_timeout,
        )
        logger.debug("Claude prompt length: %d chars", len(prompt))

        def run_claude() -> subprocess.CompletedProcess[str]:
            """Synchronous subprocess execution."""
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=working_dir,
                stdin=subprocess.DEVNULL,  # Prevent waiting for stdin
            )

        try:
            result = await asyncio.to_thread(run_claude)
            logger.info("Claude command completed with exit code %d", result.returncode)
        except subprocess.TimeoutExpired as e:
            logger.error("Claude CLI timed out after %d seconds", effective_timeout)
            raise CLIExecutionError(
                f"Claude CLI timed out after {effective_timeout} seconds",
                exit_code=-2,
                stderr=str(e),
            )
        except FileNotFoundError:
            raise CLIExecutionError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
                exit_code=-1,
                stderr="Claude CLI not installed",
            )

        if result.returncode != 0:
            raise CLIExecutionError(
                f"Claude CLI failed with exit code {result.returncode}",
                exit_code=result.returncode,
                stderr=result.stderr,
            )

        # Claude CLI outputs plain text to stdout (and sometimes stderr)
        # Combine both for completeness
        response = (result.stdout + result.stderr).strip()

        if not response:
            logger.warning("Claude returned empty response")

        return response
