"""Tests for MemoryCoordinator wiring into CompatShim.

Phase 7, Task 1: CompatShim gets a MemoryCoordinator that references the
parent AIAgent's MemoryStore and MemoryManager. The coordinator's
build_prompt_blocks() produces the same system prompt blocks that
_build_system_prompt was producing.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent.orchestrator.memory import (
    MemoryCoordinator,
    NudgeTracker,
    WriteMetadataTracker,
)


class TestMemoryCoordinatorInit:
    """MemoryCoordinator is wired during CompatShim.__init__."""

    def test_compat_shim_has_memory_coordinator(self):
        """CompatShim exposes a .memory property that is a MemoryCoordinator."""
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(parent_agent=None, session_id="test")
        assert hasattr(shim, "memory")
        assert isinstance(shim.memory, MemoryCoordinator)

    def test_coordinator_receives_store_from_parent(self):
        """When parent has a MemoryStore, coordinator.store points to it."""
        from agent.orchestrator.compat import AIAgentCompatShim
        mock_parent = MagicMock()
        mock_parent._memory_store = "mock_store"
        mock_parent._memory_enabled = True
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test")
        assert shim.memory.store == "mock_store"

    def test_coordinator_receives_manager_from_parent(self):
        """When parent has a MemoryManager, coordinator.manager points to it."""
        from agent.orchestrator.compat import AIAgentCompatShim
        mock_parent = MagicMock()
        mock_parent._memory_manager = "mock_manager"
        mock_parent._memory_enabled = True
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test")
        assert shim.memory.manager == "mock_manager"


class TestMemoryCoordinatorNudgeTracker:
    """MemoryCoordinator's NudgeTracker is wired for turn counting."""

    def test_coordinator_has_nudge_tracker(self):
        coordinator = MemoryCoordinator(session_id="t")
        assert isinstance(coordinator.nudge_tracker, NudgeTracker)

    def test_on_turn_start_returns_nudge_when_due(self):
        coordinator = MemoryCoordinator(session_id="t")
        # Not due on first turns
        assert coordinator.on_turn_start(1, "hello") is None
        assert coordinator.on_turn_start(2, "hello") is None
        # Due after 10 turns (default interval)
        for i in range(3, 12):
            result = coordinator.on_turn_start(i, f"msg{i}")
        # The 11th call (total) should trigger nudge
        result = coordinator.on_turn_start(12, "msg12")
        assert result is not None
        assert "memory" in result.lower() or "skill" in result.lower()


class TestMemoryCoordinatorWriteProvenance:
    """MemoryCoordinator's WriteMetadataTracker builds provenance dicts."""

    def test_coordinator_has_write_metadata(self):
        coordinator = MemoryCoordinator(session_id="s1")
        assert isinstance(coordinator.write_metadata, WriteMetadataTracker)

    def test_build_metadata_includes_required_fields(self):
        coordinator = MemoryCoordinator(session_id="s1")
        meta = coordinator.write_metadata.build_metadata()
        assert "write_origin" in meta
        assert "session_id" in meta
        assert meta["session_id"] == "s1"


class TestMemoryCoordinatorPromptBlocks:
    """build_prompt_blocks() returns memory/user/recent-context blocks."""

    def test_empty_when_no_store(self):
        """With no store, build_prompt_blocks returns empty list."""
        coordinator = MemoryCoordinator(session_id="t")
        blocks = coordinator.build_prompt_blocks()
        assert blocks == []

    def test_delegates_to_store_format_for_system_prompt(self):
        """When store exists, build_prompt_blocks calls store.format_for_system_prompt."""
        mock_store = MagicMock()
        mock_store.format_for_system_prompt.side_effect = lambda cat: f"<{cat}_block>"
        
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.store = mock_store
        coordinator._memory_enabled = True
        coordinator._user_profile_enabled = True
        
        blocks = coordinator.build_prompt_blocks()
        # Should get memory, user, and recent_context blocks
        assert len(blocks) >= 2
        mock_store.format_for_system_prompt.assert_called()