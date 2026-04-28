"""Tests for write provenance and tool dispatch wiring (Phase 7 Tasks 3-4).

CompatShim's _dispatch_tool calls memory.record_tool_use() for
memory/skill tools, and memory.write_metadata.build_metadata()
attaches provenance to memory writes.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from agent.orchestrator.compat import AIAgentCompatShim
from agent.orchestrator.memory import MemoryCoordinator, WriteMetadataTracker


class TestDispatchRecordsToolUse:
    """_dispatch_tool records tool use for nudge tracking."""

    def test_dispatch_records_memory_tool_use(self):
        """Dispatching a memory tool resets the memory nudge counter."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-prov")

        # Burn turns to trigger memory nudge
        for i in range(12):
            shim.memory.on_turn_start(i, f"msg{i}")

        # Nudge should be due
        assert shim.memory.should_nudge() is not None

        # Dispatch a memory tool — mock the agent-loop dispatch since
        # we're only testing nudge tracking, not the actual memory tool.
        with patch.object(shim, "_dispatch_agent_loop_tool", return_value='{"success": true}'):
            result = shim._dispatch_tool("memory", {"action": "add", "content": "test"})

        # Memory nudge should be reset
        assert shim.memory.nudge_tracker._turns_since_memory == 0

    def test_dispatch_records_skill_tool_use(self):
        """Dispatching a skill tool resets the skill nudge counter."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-prov")

        # Burn turns to trigger skill nudge
        for i in range(12):
            shim.memory.on_turn_start(i, f"msg{i}")

        # Confirm skill nudge is due
        assert shim.memory.nudge_tracker._turns_since_skill >= 10

        # Dispatch a skill tool
        with patch("model_tools.handle_function_call", return_value="ok"):
            result = shim._dispatch_tool("skill_view", {"name": "test"})

        # Skill counter should be reset
        assert shim.memory.nudge_tracker._turns_since_skill == 0

    def test_dispatch_does_not_reset_nudge_for_regular_tools(self):
        """Regular tools like web_search don't reset nudge counters."""
        mock_parent = MagicMock()
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-prov")

        # Burn turns to trigger memory nudge
        for i in range(12):
            shim.memory.on_turn_start(i, f"msg{i}")

        # Confirm nudge is due
        assert shim.memory.should_nudge() is not None

        # Dispatch a regular tool
        with patch("model_tools.handle_function_call", return_value="ok"):
            result = shim._dispatch_tool("web_search", {"query": "test"})

        # Nudge should STILL be due (not reset by regular tool)
        assert shim.memory.should_nudge() is not None


class TestWriteProvenance:
    """WriteMetadataTracker builds provenance dicts for memory writes."""

    def test_metadata_includes_session_id(self):
        tracker = WriteMetadataTracker(session_id="sess-123")
        meta = tracker.build_metadata()
        assert meta["session_id"] == "sess-123"

    def test_metadata_includes_platform(self):
        tracker = WriteMetadataTracker(session_id="s", platform="telegram")
        meta = tracker.build_metadata()
        assert meta["platform"] == "telegram"

    def test_metadata_excludes_empty_parent_session(self):
        tracker = WriteMetadataTracker(session_id="s")
        meta = tracker.build_metadata()
        assert "parent_session_id" not in meta

    def test_metadata_includes_parent_session_when_set(self):
        tracker = WriteMetadataTracker(session_id="s", parent_session_id="parent-1")
        meta = tracker.build_metadata()
        assert meta["parent_session_id"] == "parent-1"

    def test_set_origin_changes_write_origin(self):
        tracker = WriteMetadataTracker(session_id="s")
        tracker.set_origin("background_review", "auto_review")
        meta = tracker.build_metadata()
        assert meta["write_origin"] == "background_review"
        assert meta["execution_context"] == "auto_review"