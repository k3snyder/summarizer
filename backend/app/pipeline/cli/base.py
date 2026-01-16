"""
Base class for CLI executors.

Defines the abstract interface that all CLI executor implementations must follow.
"""

from abc import ABC, abstractmethod


class CLIExecutionError(Exception):
    """Exception raised when CLI execution fails."""

    def __init__(self, message: str, exit_code: int = 1, stderr: str = ""):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


class CLIExecutorBase(ABC):
    """Abstract base class for CLI executors.

    Provides a unified interface for executing prompts via different CLI tools.
    Implementations handle CLI-specific command building, output parsing, and
    error handling.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the CLI executor name for logging."""
        pass

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        working_dir: str,
        timeout: int | None = None,
    ) -> str:
        """Execute a prompt via the CLI and return the response.

        Args:
            prompt: The prompt to send to the CLI
            working_dir: Working directory for the CLI to operate in
            timeout: Timeout in seconds (None uses executor default)

        Returns:
            The extracted response text from the CLI

        Raises:
            CLIExecutionError: If CLI is not found, times out, or returns error
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the CLI tool is installed and available.

        Returns:
            True if the CLI is available, False otherwise
        """
        pass
