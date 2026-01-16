"""
CLI executor abstraction layer.

Provides a unified interface for executing prompts via different CLI tools
(Codex CLI, Claude CLI) with consistent error handling and output parsing.
"""

from app.pipeline.cli.base import CLIExecutionError, CLIExecutorBase
from app.pipeline.cli.factory import CLIProvider, get_cli_executor

__all__ = [
    "CLIExecutorBase",
    "CLIExecutionError",
    "CLIProvider",
    "get_cli_executor",
]
