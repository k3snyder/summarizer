"""
CLI executor factory.

Provides a factory function to create CLI executor instances based on provider name.
"""

from enum import Enum
from typing import TYPE_CHECKING

from app.pipeline.cli.base import CLIExecutorBase
from app.pipeline.cli.claude import ClaudeCLIExecutor
from app.pipeline.cli.codex import CodexCLIExecutor

if TYPE_CHECKING:
    from app.config import Settings


class CLIProvider(str, Enum):
    """Available CLI provider options."""

    CODEX = "codex"
    CLAUDE = "claude"


def get_cli_executor(
    provider: str | CLIProvider,
    settings: "Settings | None" = None,
) -> CLIExecutorBase:
    """Create a CLI executor instance for the specified provider.

    Args:
        provider: CLI provider name ('codex' or 'claude')
        settings: Optional settings object. If not provided, uses global settings.

    Returns:
        CLIExecutorBase instance for the specified provider

    Raises:
        ValueError: If provider is unknown
    """
    # Import settings lazily to avoid circular imports
    if settings is None:
        from app.config import settings as global_settings

        settings = global_settings

    # Normalize provider to enum
    if isinstance(provider, str):
        provider = provider.lower()

    if provider == CLIProvider.CODEX or provider == "codex":
        return CodexCLIExecutor(
            timeout=settings.codex_cli_timeout,
            sandbox_policy=settings.codex_cli_sandbox_policy,
        )
    elif provider == CLIProvider.CLAUDE or provider == "claude":
        return ClaudeCLIExecutor(
            model=settings.claude_cli_model,
            timeout=settings.claude_cli_timeout,
        )
    else:
        raise ValueError(f"Unknown CLI provider: {provider}")
