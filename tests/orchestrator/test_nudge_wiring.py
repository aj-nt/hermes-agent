"""Tests for nudge tracking integration (Phase 7 Task 3).

CompatShim delegates nudge tracking through MemoryCoordinator.
Tool dispatch records usage. Streaming bridge checks for nudges.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent.orchestrator.compat import AIAgentCompatShim
from agent.orchestrator.memory import MemoryCoordinator, NudgeTracker


class TestDispatchRecordsToolUse:
    """_dispatch_tool records tool use via MemoryCoordinator."""

    def test_memory_tool_resets_nudge_counter(self):
        """Dispatching a memory tool resets the memory nudge counter."""
        mock_parent = MagicMock()
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="t")
        
        # Burn turns to trigger nudge
        for i in range(12):
            shim.memory.on_turn_start(i, f"msg{i}")
        
        # Nudge should be due
        nudge = shim.memory.should_nudge()
        assert nudge is not None
        
        # Now dispatch a memory tool
        # (This only tests the wiring; actual dispatch requires handle_function_call)
        shim.memory.record_tool_use("memory")
        
        # Memory nudge should be reset
        assert shim.memory.nudge_tracker._turns_since_memory == 0

    def test_skill_tool_resets_skill_nudge_counter(self):
        """Dispatching a skill tool resets the skill nudge counter."""
        mock_parent = MagicMock()
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="t")
        
        # Burn turns to trigger skill nudge
        for i in range(12):
            shim.memory.on_turn_start(i, f"msg{i}")
        
        # Skill nudge should be due
        assert shim.memory.nudge_tracker._turns_since_skill >= 10
        
        # Dispatch a skill tool
        shim.memory.record_tool_use("skill_manage")
        
        # Skill counter reset
        assert shim.memory.nudge_tracker._turns_since_skill == 0


class TestNudgeTrackerOnTurnStart:
    """on_turn_start increments counters and returns nudge text."""

    def test_on_turn_start_increments_counters(self):
        coordinator = MemoryCoordinator(session_id="t")
        assert coordinator.nudge_tracker._turns_since_memory == 0
        coordinator.on_turn_start(1, "hello")
        assert coordinator.nudge_tracker._turns_since_memory == 1
        coordinator.on_turn_start(2, "world")
        assert coordinator.nudge_tracker._turns_since_memory == 2

    def test_on_turn_start_returns_nudge_text(self):
        """After enough turns without memory use, nudge fires."""
        coordinator = MemoryCoordinator(session_id="t", nudge_interval=3, skill_interval=100)
        assert coordinator.on_turn_start(1, "hello") is None
        assert coordinator.on_turn_start(2, "hello") is None
        result = coordinator.on_turn_start(4, "hello")
        assert result is not None
        assert "memory" in result.lower()

    def test_on_turn_start_returns_skill_nudge_if_memory_not_due(self):
        """Skill nudge fires when memory nudge has been used recently."""
        coordinator = MemoryCoordinator(session_id="t", nudge_interval=100, skill_interval=3)
        coordinator.memory_nudge = False  # won't fire memory
        assert coordinator.on_turn_start(1, "hello") is None
        assert coordinator.on_turn_start(2, "hello") is None
        result = coordinator.on_turn_start(4, "hello")
        assert result is not None
        assert "skill" in result.lower()


class TestNudgeViaCompatShim:
    """CompatShim exposes memory.on_turn_start for pipeline loop integration."""

    def test_compat_shim_memory_tracks_turns(self):
        mock_parent = MagicMock()
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="t")
        
        # Pipeline loop calls on_turn_start each iteration
        shim.memory.on_turn_start(1, "hello")
        assert shim.memory.nudge_tracker._turns_since_memory == 1
        
        shim.memory.on_turn_start(2, "describe yourself")
        assert shim.memory.nudge_tracker._turns_since_memory == 2