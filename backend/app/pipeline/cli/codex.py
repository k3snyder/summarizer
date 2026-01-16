"""
Codex CLI executor implementation.

Invokes Codex CLI via subprocess with file-based I/O, parsing JSONL output
to extract AI responses.
"""

import asyncio
import json
import logging
import shutil
import subprocess

from app.pipeline.cli.base import CLIExecutionError, CLIExecutorBase

logger = logging.getLogger("app.pipeline.cli.codex")


def _parse_jsonl_output(stdout: str) -> str:
    """Extract response content from JSONL output.

    Codex CLI outputs JSONL events. We look for:
    1. 'item.completed' events with agent_message type (current format)
    2. 'message' events with assistant role (legacy/alternative format)

    Args:
        stdout: Raw JSONL output from codex exec

    Returns:
        Concatenated content from assistant/agent message events
    """
    content_parts = []

    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            event_type = event.get("type", "")

            # Current Codex CLI format: item.completed with agent_message
            if event_type == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")
                    if text:
                        content_parts.append(text)

            # Legacy/alternative format: message events with assistant role
            elif event_type == "message":
                message = event.get("message", {})
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                content_parts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                content_parts.append(block)
                    elif isinstance(content, str):
                        content_parts.append(content)

        except json.JSONDecodeError:
            continue

    return "\n".join(content_parts).strip()


class CodexCLIExecutor(CLIExecutorBase):
    """Codex CLI executor implementation.

    Executes prompts via the Codex CLI tool with JSONL output parsing.
    """

    def __init__(
        self,
        timeout: int = 300,
        sandbox_policy: str = "workspace-write",
    ):
        """Initialize Codex CLI executor.

        Args:
            timeout: Default timeout in seconds (default 300 = 5 minutes)
            sandbox_policy: Sandbox policy for Codex (default 'workspace-write')
        """
        self._timeout = timeout
        self._sandbox_policy = sandbox_policy

    @property
    def name(self) -> str:
        return "codex"

    def is_available(self) -> bool:
        return shutil.which("codex") is not None

    async def execute(
        self,
        prompt: str,
        working_dir: str,
        timeout: int | None = None,
    ) -> str:
        """Execute a prompt via Codex CLI.

        Args:
            prompt: The prompt to send to Codex
            working_dir: Working directory for Codex to operate in
            timeout: Timeout in seconds (None uses instance default)

        Returns:
            The extracted response text from Codex

        Raises:
            CLIExecutionError: If Codex CLI is not found, times out, or returns error
        """
        if not self.is_available():
            raise CLIExecutionError(
                "Codex CLI not found. Install with: npm install -g @openai/codex",
                exit_code=-1,
                stderr="Codex CLI not installed",
            )

        effective_timeout = timeout if timeout is not None else self._timeout

        # Build command - use "-" to read prompt from stdin (avoids command line length limits)
        cmd = [
            "codex",
            "exec",
            "-C",
            working_dir,
            "-s",
            self._sandbox_policy,
            "--full-auto",
            "--skip-git-repo-check",
            "--json",
            "-",  # Read prompt from stdin
        ]

        logger.info(
            "Executing codex command in %s (timeout=%ds, sandbox=%s, prompt_len=%d)",
            working_dir,
            effective_timeout,
            self._sandbox_policy,
            len(prompt),
        )

        def run_codex() -> subprocess.CompletedProcess[str]:
            """Synchronous subprocess execution."""
            return subprocess.run(
                cmd,
                input=prompt,  # Pass prompt via stdin
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=working_dir,
            )

        try:
            result = await asyncio.to_thread(run_codex)
            logger.info("Codex command completed with exit code %d", result.returncode)
        except subprocess.TimeoutExpired as e:
            logger.error("Codex CLI timed out after %d seconds", effective_timeout)
            raise CLIExecutionError(
                f"Codex CLI timed out after {effective_timeout} seconds",
                exit_code=-2,
                stderr=str(e),
            )

        if result.returncode != 0:
            raise CLIExecutionError(
                f"Codex CLI failed with exit code {result.returncode}",
                exit_code=result.returncode,
                stderr=result.stderr,
            )

        # Parse JSONL output
        response = _parse_jsonl_output(result.stdout)

        if not response:
            logger.warning("Codex returned empty response. stdout: %s", result.stdout[:500])

        return response


# Backwards compatibility: expose module-level query function
async def query(
    prompt: str,
    working_dir: str,
    timeout: int = 300,
    sandbox_policy: str = "workspace-write",
) -> str:
    """Execute a Codex CLI query and return the response.

    This function is provided for backwards compatibility.
    New code should use CodexCLIExecutor directly.

    Args:
        prompt: The prompt to send to Codex
        working_dir: Working directory for Codex to operate in
        timeout: Timeout in seconds (default 300 = 5 minutes)
        sandbox_policy: Sandbox policy for Codex (default 'workspace-write')

    Returns:
        The extracted response text from Codex

    Raises:
        CLIExecutionError: If Codex CLI is not found, times out, or returns error
    """
    executor = CodexCLIExecutor(timeout=timeout, sandbox_policy=sandbox_policy)
    return await executor.execute(prompt, working_dir, timeout)


# Backwards compatibility alias
CodexExecutionError = CLIExecutionError
