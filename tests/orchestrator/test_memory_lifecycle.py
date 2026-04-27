"""Tests for MemoryCoordinator lifecycle methods (Phase 7 Task 5).

MemoryCoordinator.on_turn_end(), on_session_end(), and shutdown()
route through to the underlying store/manager.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call

from agent.orchestrator.memory import MemoryCoordinator


class TestOnTurnEnd:
    """on_turn_end syncs to external providers and prefetches."""

    def test_calls_manager_sync_all(self):
        """on_turn_end calls manager.sync_all with user/assistant messages."""
        mock_manager = MagicMock()
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.manager = mock_manager

        coordinator.on_turn_end("user message", "assistant response")

        mock_manager.sync_all.assert_called_once_with("user message", "assistant response")

    def test_calls_manager_queue_prefetch(self):
        """on_turn_end calls manager.queue_prefetch_all."""
        mock_manager = MagicMock()
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.manager = mock_manager

        coordinator.on_turn_end("user message", "assistant response")

        mock_manager.queue_prefetch_all.assert_called_once_with("user message")

    def test_noop_when_no_manager(self):
        """on_turn_end is a no-op when no manager exists."""
        coordinator = MemoryCoordinator(session_id="t")
        # Should not raise
        coordinator.on_turn_end("user message", "assistant response")

    def test_skips_when_no_final_response(self):
        """on_turn_end with interrupted=True skips sync."""
        mock_manager = MagicMock()
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.manager = mock_manager

        coordinator.on_turn_end("user message", "", interrupted=True)

        # sync_all should NOT be called for interrupted turns
        mock_manager.sync_all.assert_not_called()


class TestOnSessionEnd:
    """on_session_end does final extraction pass (no teardown)."""

    def test_calls_manager_on_session_end(self):
        """on_session_end calls manager.on_session_end with messages."""
        mock_manager = MagicMock()
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.manager = mock_manager

        coordinator.on_session_end([{"role": "user", "content": "hi"}])

        mock_manager.on_session_end.assert_called_once()

    def test_noop_when_no_manager(self):
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.on_session_end([])  # Should not raise


class TestShutdown:
    """shutdown does extraction + full provider teardown."""

    def test_calls_manager_on_session_end_then_shutdown_all(self):
        """shutdown calls on_session_end then shutdown_all."""
        mock_manager = MagicMock()
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.manager = mock_manager

        coordinator.shutdown(messages=[{"role": "user", "content": "hi"}])

        mock_manager.on_session_end.assert_called_once()
        mock_manager.shutdown_all.assert_called_once()

    def test_calls_manager_shutdown_all_with_no_messages(self):
        """shutdown with no messages still tears down."""
        mock_manager = MagicMock()
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.manager = mock_manager

        coordinator.shutdown()

        mock_manager.on_session_end.assert_called_once_with([])
        mock_manager.shutdown_all.assert_called_once()

    def test_noop_when_no_manager(self):
        coordinator = MemoryCoordinator(session_id="t")
        coordinator.shutdown()  # Should not raise