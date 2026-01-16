"""
E2E tests for Codex CLI executor.

These tests require Codex CLI to be installed and authenticated.
They will be skipped if codex is not available.
"""

import shutil
from unittest.mock import patch

import pytest

from app.pipeline.codex_executor import query, CodexExecutionError, _parse_jsonl_output


# Check if codex CLI is installed
CODEX_INSTALLED = shutil.which("codex") is not None


class TestParseJsonlOutput:
    """Unit tests for JSONL parsing (don't require codex CLI)."""

    def test_extracts_item_completed_agent_message(self):
        """Extract content from item.completed with agent_message type (current format)."""
        jsonl = '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"Hello world"}}'
        result = _parse_jsonl_output(jsonl)
        assert result == "Hello world"

    def test_extracts_assistant_message_string_content(self):
        """Extract content from assistant message with string content (legacy format)."""
        jsonl = '{"type":"message","message":{"role":"assistant","content":"Hello world"}}'
        result = _parse_jsonl_output(jsonl)
        assert result == "Hello world"

    def test_extracts_assistant_message_array_content(self):
        """Extract content from assistant message with array content."""
        jsonl = '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Hello world"}]}}'
        result = _parse_jsonl_output(jsonl)
        assert result == "Hello world"

    def test_ignores_non_message_events(self):
        """Ignore events that are not message or item.completed type."""
        jsonl = """{"type":"thread.started","thread_id":"123"}
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"agent_message","text":"Result"}}
{"type":"turn.completed","usage":{}}"""
        result = _parse_jsonl_output(jsonl)
        assert result == "Result"

    def test_ignores_non_assistant_messages(self):
        """Ignore messages that are not from assistant."""
        jsonl = """{"type":"message","message":{"role":"user","content":"Query"}}
{"type":"message","message":{"role":"assistant","content":"Response"}}"""
        result = _parse_jsonl_output(jsonl)
        assert result == "Response"

    def test_handles_empty_input(self):
        """Handle empty input gracefully."""
        result = _parse_jsonl_output("")
        assert result == ""

    def test_handles_malformed_json(self):
        """Skip malformed JSON lines."""
        jsonl = """not valid json
{"type":"item.completed","item":{"type":"agent_message","text":"Good"}}"""
        result = _parse_jsonl_output(jsonl)
        assert result == "Good"

    def test_concatenates_multiple_messages(self):
        """Concatenate content from multiple agent messages."""
        jsonl = """{"type":"item.completed","item":{"type":"agent_message","text":"First"}}
{"type":"item.completed","item":{"type":"agent_message","text":"Second"}}"""
        result = _parse_jsonl_output(jsonl)
        assert "First" in result
        assert "Second" in result


class TestCodexExecutionError:
    """Tests for CodexExecutionError exception."""

    def test_has_exit_code_attribute(self):
        """Exception should have exit_code attribute."""
        e = CodexExecutionError("test", exit_code=42)
        assert e.exit_code == 42

    def test_has_stderr_attribute(self):
        """Exception should have stderr attribute."""
        e = CodexExecutionError("test", stderr="error output")
        assert e.stderr == "error output"

    def test_default_values(self):
        """Default values for exit_code and stderr."""
        e = CodexExecutionError("test")
        assert e.exit_code == 1
        assert e.stderr == ""


@pytest.mark.skipif(not CODEX_INSTALLED, reason="Codex CLI not installed")
class TestCodexExecutorE2E:
    """E2E tests that require Codex CLI to be installed."""

    @pytest.mark.asyncio
    async def test_query_returns_response(self):
        """Query should return a response for a simple prompt."""
        result = await query("What is 2+2? Reply with just the number.", "/tmp")
        assert "4" in result

    @pytest.mark.asyncio
    async def test_handles_timeout(self):
        """Query should raise error on timeout."""
        with pytest.raises(CodexExecutionError) as exc_info:
            await query(
                "Count to 1 million slowly",
                "/tmp",
                timeout=1,  # Very short timeout
            )
        assert "timeout" in str(exc_info.value).lower() or exc_info.value.exit_code == -2


class TestCodexMissingCLI:
    """Tests for handling missing Codex CLI."""

    @pytest.mark.asyncio
    async def test_handles_missing_cli(self):
        """Raise CodexExecutionError when CLI is not installed."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(CodexExecutionError) as exc_info:
                await query("test", "/tmp")
            assert "not found" in str(exc_info.value).lower()
            assert exc_info.value.exit_code == -1
