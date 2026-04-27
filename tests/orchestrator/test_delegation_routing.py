"""Tests for AIAgent delegation to the new pipeline.

Phase 6B-2: The old code paths are removed. Delegation to CompatShim is
now unconditional — switch_model(), interrupt(), run_conversation(), and
chat() always route through the new pipeline orchestrator.

These tests require venv Python 3.11+.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

try:
    from run_agent import AIAgent
    from agent.orchestrator.compat import AIAgentCompatShim
    HAS_RUN_AGENT = True
except Exception:
    HAS_RUN_AGENT = False

pytestmark = pytest.mark.skipif(not HAS_RUN_AGENT, reason="run_agent requires venv Python 3.11+")


class TestUnconditionalDelegation:
    """Delegation to CompatShim is now unconditional (no feature flag)."""

    def test_run_conversation_delegates_to_shim(self):
        """run_conversation() delegates to shim.run_conversation()."""
        mock_shim = MagicMock(spec=AIAgentCompatShim)
        mock_shim.run_conversation.return_value = {
            "final_response": "Mocked response",
            "messages": [{"role": "assistant", "content": "Mocked response"}],
            "iterations": 1,
            "interrupted": False,
        }

        agent = object.__new__(AIAgent)
        agent.session_id = "test-session"
        agent.model = "test-model"
        agent._new_pipeline = mock_shim

        result = agent.run_conversation("Hello")
        mock_shim.run_conversation.assert_called_once()
        assert result["final_response"] == "Mocked response"

    def test_chat_delegates_through_run_conversation(self):
        """chat() calls run_conversation(), which delegates to the shim."""
        mock_shim = MagicMock(spec=AIAgentCompatShim)
        mock_shim.run_conversation.return_value = {
            "final_response": "Chat response",
            "messages": [],
            "iterations": 1,
            "interrupted": False,
        }

        agent = object.__new__(AIAgent)
        agent.session_id = "test-session"
        agent.model = "test-model"
        agent._new_pipeline = mock_shim

        result = agent.chat("Hello")
        mock_shim.run_conversation.assert_called_once()
        assert result == "Chat response"

    def test_interrupt_delegates_to_shim(self):
        """interrupt() delegates to shim.interrupt()."""
        mock_shim = MagicMock(spec=AIAgentCompatShim)

        agent = object.__new__(AIAgent)
        agent.session_id = "test-session"
        agent.model = "test-model"
        agent._new_pipeline = mock_shim

        agent.interrupt("stop please")
        mock_shim.interrupt.assert_called_once_with("stop please")

    def test_switch_model_delegates_to_shim(self):
        """switch_model() delegates to shim.switch_model()."""
        mock_shim = MagicMock(spec=AIAgentCompatShim)

        agent = object.__new__(AIAgent)
        agent.session_id = "test-session"
        agent.model = "old-model"
        agent._new_pipeline = mock_shim

        agent.switch_model("new-model", "anthropic", api_key="sk-test")
        mock_shim.switch_model.assert_called_once_with(
            "new-model", "anthropic", api_key="sk-test", base_url="", api_mode=""
        )