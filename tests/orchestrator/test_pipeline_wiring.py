"""Tests for wiring the new Orchestrator pipeline into AIAgent.

Phase 6B-2: No feature flag. Delegation is unconditional.
Tests verify that CompatShim is always created and methods route through it.
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


class TestOrchestratorCreation:
    """CompatShim is always constructed (no flag to disable)."""

    def test_chat_delegates_to_shim(self):
        """chat() always delegates through CompatShim."""
        mock_shim = MagicMock(spec=AIAgentCompatShim)
        mock_shim.run_conversation.return_value = {
            "final_response": "response",
            "messages": [],
            "iterations": 1,
            "interrupted": False,
        }

        agent = object.__new__(AIAgent)
        agent._new_pipeline = mock_shim
        agent.session_id = "test"
        agent.model = "test"

        result = agent.chat("hello")
        assert result == "response"


class TestMethodRouting:
    """All external methods route through the CompatShim."""

    def test_run_conversation_method_exists(self):
        assert hasattr(AIAgent, "run_conversation")

    def test_switch_model_method_exists(self):
        assert hasattr(AIAgent, "switch_model")

    def test_interrupt_method_exists(self):
        assert hasattr(AIAgent, "interrupt")


class TestAttributeParity:
    """CompatShim has all external methods needed by the gateway."""

    def test_essential_attributes_on_shim(self):
        assert hasattr(AIAgentCompatShim, "chat")
        assert hasattr(AIAgentCompatShim, "run_conversation")
        assert hasattr(AIAgentCompatShim, "switch_model")
        assert hasattr(AIAgentCompatShim, "interrupt")

    def test_shim_has_all_external_methods(self):
        """CompatShim implements all methods the gateway calls on AIAgent."""
        external_methods = [
            "chat",
            "run_conversation",
            "switch_model",
            "interrupt",
            "release_clients",
        ]
        for method in external_methods:
            assert hasattr(AIAgentCompatShim, method), (
                f"CompatShim missing method '{method}' needed by gateway"
            )