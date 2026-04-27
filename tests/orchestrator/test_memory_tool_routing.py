"""Tests for MemoryCoordinator tool routing (Phase 7 Task 2).

MemoryCoordinator.handle_tool_call() routes memory/external-memory tool
calls to the appropriate handler (MemoryStore or MemoryManager).
MemoryCoordinator.get_tool_schemas() returns merged tool schemas.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent.orchestrator.memory import MemoryCoordinator


class TestHandleToolCall:
    """MemoryCoordinator routes tool calls to store or manager."""

    def test_routes_memory_add_to_store(self):
        """memory tool add/replace/delete/search/consolidate go to store."""
        mock_store = MagicMock()
        mock_store.handle_tool_call.return_value = "added"
        
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.store = mock_store
        coordinator._memory_enabled = True
        
        result = coordinator.handle_tool_call(
            "memory", {"action": "add", "content": "test"}, metadata={}
        )
        mock_store.handle_tool_call.assert_called_once()

    def test_routes_external_tool_to_manager(self):
        """Tools that manager.has_tool() are routed to manager."""
        mock_manager = MagicMock()
        mock_manager.has_tool.return_value = True
        mock_manager.handle_tool_call.return_value = "external_result"
        
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.manager = mock_manager
        
        result = coordinator.handle_tool_call(
            "external_memory", {"action": "query"}, metadata={}
        )
        assert result == "external_result"

    def test_raises_for_unknown_tool(self):
        """Unknown tools raise ValueError."""
        coordinator = MemoryCoordinator(session_id="t")
        with pytest.raises(ValueError, match="Unknown memory tool"):
            coordinator.handle_tool_call("unknown_tool", {}, metadata={})

    def test_routes_memory_search_to_store(self):
        """memory tool search action goes to store."""
        mock_store = MagicMock()
        mock_store.handle_tool_call.return_value = "results"
        
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.store = mock_store
        coordinator._memory_enabled = True
        
        result = coordinator.handle_tool_call(
            "memory", {"action": "search", "query": "test"}, metadata={}
        )
        mock_store.handle_tool_call.assert_called_once()


class TestGetToolSchemas:
    """MemoryCoordinator merges built-in and external tool schemas."""

    def test_returns_empty_when_no_store_or_manager(self):
        coordinator = MemoryCoordinator(session_id="t")
        assert coordinator.get_tool_schemas() == []

    def test_includes_store_schemas(self):
        """When store exists and memory_enabled, store schemas are included."""
        mock_store = MagicMock()
        mock_store.get_tool_schema.return_value = {"type": "function", "name": "memory"}
        
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.store = mock_store
        coordinator._memory_enabled = True
        
        schemas = coordinator.get_tool_schemas()
        assert len(schemas) >= 1

    def test_includes_manager_schemas(self):
        """When manager exists, its schemas are merged in."""
        mock_manager = MagicMock()
        mock_manager.get_all_tool_schemas.return_value = [
            {"type": "function", "name": "external_memory"}
        ]
        
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.manager = mock_manager
        
        schemas = coordinator.get_tool_schemas()
        assert len(schemas) == 1


class TestRecordToolUse:
    """MemoryCoordinator resets nudge counters on tool use."""

    def test_memory_tool_resets_nudge(self):
        coordinator = MemoryCoordinator(session_id="t", nudge_interval=3, skill_interval=100)
        # Burn through 3 turns to trigger memory nudge (skill interval is 100 so it won't fire)
        for i in range(4):
            coordinator.on_turn_start(i, f"msg{i}")
        assert coordinator.should_nudge() is not None
        
        # Using memory tool resets counter
        coordinator.record_tool_use("memory")
        assert coordinator.should_nudge() is None

    def test_skill_tool_resets_skill_nudge(self):
        coordinator = MemoryCoordinator(session_id="t")
        for i in range(10):
            coordinator.on_turn_start(i, f"msg{i}")
        assert coordinator.should_nudge() is not None
        
        coordinator.record_tool_use("skill_manage")
        # Skill nudge is reset; memory nudge may still fire
        # But at least the skill counter is reset
        coordinator.nudge_tracker.on_memory_use()  # also clear memory
        assert coordinator.should_nudge() is None