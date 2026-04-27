"""Tests for ToolExecutor: the stage-4 tool dispatch engine.

ToolExecutor replaces the scattered _invoke_tool, _execute_tool_calls,
_execute_tool_calls_concurrent, and checkpoint interception logic
in run_agent.py (860+ lines combined).
"""

from __future__ import annotations

import json
import pytest
from typing import Any, Optional
from unittest.mock import MagicMock

from agent.orchestrator.context import ParsedResponse
from agent.orchestrator.stages import StageAction, ToolDispatchResult


# ============================================================================
# Helpers
# ============================================================================

def _make_parsed_with_tools(*tool_names):
    """Build a ParsedResponse with tool calls for given names."""
    tool_calls = []
    for i, name in enumerate(tool_names):
        tool_calls.append({
            "id": f"call_{i+1}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps({"arg": i})},
        })
    return ParsedResponse(
        message={"role": "assistant", "content": None, "tool_calls": tool_calls},
        tool_calls=tool_calls,
        finish_reason="tool_calls",
    )


def _make_parsed_text(content="Hello"):
    """Build a ParsedResponse with text only, no tool calls."""
    return ParsedResponse(
        message={"role": "assistant", "content": content},
        tool_calls=[],
        finish_reason="stop",
    )


# ============================================================================
# ToolExecutor creation and registration
# ============================================================================

class TestToolExecutorCreation:
    """ToolExecutor manages the mapping of tool names to callables."""

    def test_create_empty_executor(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()
        assert executor.tool_count == 0

    def test_register_tool(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()
        executor.register("read_file", lambda args: "file contents")
        assert executor.tool_count == 1

    def test_register_multiple_tools(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()
        executor.register("read_file", lambda args: "contents")
        executor.register("write_file", lambda args: "ok")
        executor.register("patch", lambda args: "patched")
        assert executor.tool_count == 3

    def test_register_overwrites_existing(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()
        executor.register("test", lambda args: "v1")
        executor.register("test", lambda args: "v2")
        assert executor.tool_count == 1

    def test_has_tool(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()
        executor.register("read_file", lambda args: "ok")
        assert executor.has_tool("read_file") is True
        assert executor.has_tool("write_file") is False


# ============================================================================
# ToolExecutor dispatch
# ============================================================================

class TestToolExecutorDispatch:
    """dispatch() routes tool calls to registered handlers."""

    def test_dispatch_single_tool(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()
        executor.register("read_file", lambda args: f"read:{args.get('path', '?')}")

        parsed = _make_parsed_with_tools("read_file")
        result = executor.dispatch(parsed)

        assert isinstance(result, ToolDispatchResult)
        assert len(result.tool_results) == 1
        assert result.tool_results[0]["role"] == "tool"
        assert result.tool_results[0]["name"] == "read_file"
        assert result.action == StageAction.CONTINUE

    def test_dispatch_multiple_tools_sequential(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()
        executor.register("tool_a", lambda args: "a_result")
        executor.register("tool_b", lambda args: "b_result")

        parsed = _make_parsed_with_tools("tool_a", "tool_b")
        result = executor.dispatch(parsed)

        assert len(result.tool_results) == 2
        assert result.tool_results[0]["name"] == "tool_a"
        assert result.tool_results[1]["name"] == "tool_b"

    def test_dispatch_no_tool_calls_yields(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()

        parsed = _make_parsed_text()
        result = executor.dispatch(parsed)

        assert result.tool_results == []
        assert result.action == StageAction.YIELD

    def test_dispatch_unregistered_tool(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()

        parsed = _make_parsed_with_tools("unknown_tool")
        result = executor.dispatch(parsed)

        assert len(result.tool_results) == 1
        # Should contain an error message
        assert "error" in result.tool_results[0]["content"].lower() or \
               "unknown" in result.tool_results[0]["content"].lower() or \
               "not found" in result.tool_results[0]["content"].lower()

    def test_dispatch_tool_error_returns_error_content(self):
        from agent.orchestrator.tools import ToolExecutor
        def failing_tool(args):
            raise RuntimeError("Tool crashed")

        executor = ToolExecutor()
        executor.register("failing_tool", failing_tool)

        parsed = _make_parsed_with_tools("failing_tool")
        result = executor.dispatch(parsed)

        assert len(result.tool_results) == 1
        assert "error" in result.tool_results[0]["content"].lower() or \
               "crashed" in result.tool_results[0]["content"].lower()

    def test_dispatch_preserves_tool_call_id(self):
        from agent.orchestrator.tools import ToolExecutor
        executor = ToolExecutor()
        executor.register("test_tool", lambda args: "ok")

        parsed = ParsedResponse(
            message={"role": "assistant", "content": None},
            tool_calls=[{
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }],
            finish_reason="tool_calls",
        )
        result = executor.dispatch(parsed)

        assert result.tool_results[0]["tool_call_id"] == "call_abc123"


# ============================================================================
# Pre-dispatch hooks (checkpoint interception)
# ============================================================================

class TestPreDispatchHooks:
    """Pre-dispatch hooks run before tool execution (e.g. checkpointing).

    DESIGN.md specifies: "Pre-dispatch hook: if tool is write_file or
    patch, ensure checkpoint via CheckpointManager."
    """

    def test_pre_dispatch_hook_called_before_tool(self):
        from agent.orchestrator.tools import ToolExecutor
        call_order = []

        def pre_hook(name, args):
            call_order.append(("pre", name))

        def handler(args):
            call_order.append(("exec", "test"))
            return "result"

        executor = ToolExecutor(pre_dispatch_hooks=[pre_hook])
        executor.register("test_tool", handler)

        parsed = _make_parsed_with_tools("test_tool")
        result = executor.dispatch(parsed)

        assert call_order[0] == ("pre", "test_tool")
        assert call_order[1] == ("exec", "test")

    def test_multiple_pre_dispatch_hooks(self):
        from agent.orchestrator.tools import ToolExecutor
        hook_calls = []

        def hook1(name, args):
            hook_calls.append(("hook1", name))

        def hook2(name, args):
            hook_calls.append(("hook2", name))

        executor = ToolExecutor(pre_dispatch_hooks=[hook1, hook2])
        executor.register("test_tool", lambda args: "ok")

        parsed = _make_parsed_with_tools("test_tool")
        result = executor.dispatch(parsed)

        assert len(hook_calls) == 2
        assert hook_calls[0] == ("hook1", "test_tool")
        assert hook_calls[1] == ("hook2", "test_tool")

    def test_checkpoint_hook_for_write_operations(self):
        """Simulate the CheckpointManager pre-dispatch pattern from DESIGN.md."""
        from agent.orchestrator.tools import ToolExecutor
        checkpoint_called_for = []

        def checkpoint_hook(name, args):
            # DESIGN.md: checkpoint for write_file, patch
            if name in ("write_file", "patch"):
                checkpoint_called_for.append(name)

        executor = ToolExecutor(pre_dispatch_hooks=[checkpoint_hook])
        executor.register("write_file", lambda args: "written")
        executor.register("read_file", lambda args: "contents")
        executor.register("patch", lambda args: "patched")

        # Checkpoint for write_file
        parsed = _make_parsed_with_tools("write_file")
        executor.dispatch(parsed)
        assert "write_file" in checkpoint_called_for

        # No checkpoint for read_file
        parsed = _make_parsed_with_tools("read_file")
        executor.dispatch(parsed)
        assert "read_file" not in checkpoint_called_for

        # Checkpoint for patch
        parsed = _make_parsed_with_tools("patch")
        executor.dispatch(parsed)
        assert "patch" in checkpoint_called_for

    def test_pre_dispatch_hook_failure_does_not_block_execution(self):
        """A failing pre-dispatch hook should not prevent tool execution."""
        from agent.orchestrator.tools import ToolExecutor
        executed = []

        def failing_hook(name, args):
            raise RuntimeError("Hook crashed")

        executor = ToolExecutor(pre_dispatch_hooks=[failing_hook])
        executor.register("test_tool", lambda args: executed.append("ran") or "ok")

        parsed = _make_parsed_with_tools("test_tool")
        result = executor.dispatch(parsed)

        # Tool still executes despite hook failure
        assert len(executed) == 1
        assert len(result.tool_results) == 1


# ============================================================================
# Post-dispatch callbacks
# ============================================================================

class TestPostDispatchCallbacks:
    """Post-dispatch callbacks run after tool execution.

    DESIGN.md specifies:
    - Reset nudge counters on memory/skill tool use
    - Attach write metadata to memory tool calls
    """

    def test_post_dispatch_callback_called(self):
        from agent.orchestrator.tools import ToolExecutor
        post_calls = []

        def post_callback(name, args, result):
            post_calls.append((name, result))

        executor = ToolExecutor(post_dispatch_callbacks=[post_callback])
        executor.register("test_tool", lambda args: "tool_result")

        parsed = _make_parsed_with_tools("test_tool")
        result = executor.dispatch(parsed)

        assert len(post_calls) == 1
        assert post_calls[0][0] == "test_tool"

    def test_post_dispatch_receives_tool_result(self):
        from agent.orchestrator.tools import ToolExecutor
        results = []

        def post_callback(name, args, result):
            results.append(result)

        executor = ToolExecutor(post_dispatch_callbacks=[post_callback])
        executor.register("test_tool", lambda args: "special_result")

        parsed = _make_parsed_with_tools("test_tool")
        executor.dispatch(parsed)

        assert results[0] == "special_result"


# ============================================================================
# Argument parsing
# ============================================================================

class TestArgumentParsing:
    """Tool call arguments may be JSON strings or dicts."""

    def test_json_string_arguments(self):
        from agent.orchestrator.tools import ToolExecutor
        received_args = {}

        def handler(args):
            received_args.update(args)
            return "ok"

        executor = ToolExecutor()
        executor.register("read_file", handler)

        parsed = ParsedResponse(
            message={"role": "assistant", "content": None},
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/tmp/test"}'},
            }],
            finish_reason="tool_calls",
        )
        result = executor.dispatch(parsed)
        assert received_args.get("path") == "/tmp/test"

    def test_dict_arguments(self):
        from agent.orchestrator.tools import ToolExecutor
        received_args = {}

        def handler(args):
            received_args.update(args)
            return "ok"

        executor = ToolExecutor()
        executor.register("read_file", handler)

        parsed = ParsedResponse(
            message={"role": "assistant", "content": None},
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "/tmp/test"}},
            }],
            finish_reason="tool_calls",
        )
        result = executor.dispatch(parsed)
        assert received_args.get("path") == "/tmp/test"

    def test_empty_arguments(self):
        from agent.orchestrator.tools import ToolExecutor
        received_args = {}

        def handler(args):
            received_args.update(args)
            return "ok"

        executor = ToolExecutor()
        executor.register("list_files", handler)

        parsed = ParsedResponse(
            message={"role": "assistant", "content": None},
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_files", "arguments": "{}"},
            }],
            finish_reason="tool_calls",
        )
        result = executor.dispatch(parsed)
        assert received_args == {}

    def test_malformed_json_arguments(self):
        from agent.orchestrator.tools import ToolExecutor
        received_args = {}

        def handler(args):
            received_args.update(args)
            return "ok"

        executor = ToolExecutor()
        executor.register("test", handler)

        parsed = ParsedResponse(
            message={"role": "assistant", "content": None},
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "test", "arguments": "{broken json"},
            }],
            finish_reason="tool_calls",
        )
        result = executor.dispatch(parsed)
        # Should handle gracefully — empty dict fallback
        assert received_args == {}
        # Tool still called (with empty args), not skipped