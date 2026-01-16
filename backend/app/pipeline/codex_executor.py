"""
Codex CLI executor - DEPRECATED: Use app.pipeline.cli instead.

This module is maintained for backwards compatibility.
New code should import from app.pipeline.cli.
"""

# Re-export from new location for backwards compatibility
from app.pipeline.cli.base import CLIExecutionError
from app.pipeline.cli.codex import CodexCLIExecutor, query, _parse_jsonl_output

# Backwards compatibility alias
CodexExecutionError = CLIExecutionError

__all__ = [
    "query",
    "CodexExecutionError",
    "CLIExecutionError",
    "CodexCLIExecutor",
    "_parse_jsonl_output",
]
