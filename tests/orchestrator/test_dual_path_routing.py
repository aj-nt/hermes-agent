"""Tests for AIAgent delegation to the new pipeline.

Phase 6B-2: Delegation is unconditional. No dual-path routing.
All methods route through the CompatShim orchestrator.
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


class TestChatDelegation:
    """chat() routes through CompatShim."""

    def test_chat_delegates_to_shim(self):
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
        mock_shim.run_conversation.assert_called_once()
        assert result == "response"

    def test_chat_returns_string_via_shim(self):
        mock_shim = MagicMock(spec=AIAgentCompatShim)
        mock_shim.run_conversation.return_value = {
            "final_response": "yes",
            "messages": [],
            "iterations": 1,
            "interrupted": False,
        }

        agent = object.__new__(AIAgent)
        agent._new_pipeline = mock_shim
        agent.session_id = "test"
        agent.model = "test"

        assert agent.chat("q") == "yes"


class TestRunConversationDelegation:
    """run_conversation() routes through CompatShim."""

    def test_run_conversation_returns_dict_via_shim(self):
        mock_shim = MagicMock(spec=AIAgentCompatShim)
        expected = {
            "final_response": "test",
            "messages": [{"role": "assistant", "content": "test"}],
            "iterations": 1,
            "interrupted": False,
        }
        mock_shim.run_conversation.return_value = expected

        agent = object.__new__(AIAgent)
        agent._new_pipeline = mock_shim
        agent.session_id = "test"
        agent.model = "test"

        result = agent.run_conversation("hello")
        assert result == expected


class TestInterruptDelegation:
    """interrupt() routes through CompatShim."""

    def test_interrupt_delegates_to_shim(self):
        mock_shim = MagicMock(spec=AIAgentCompatShim)

        agent = object.__new__(AIAgent)
        agent._new_pipeline = mock_shim

        agent.interrupt("stop")
        mock_shim.interrupt.assert_called_once_with("stop")


class TestSwitchModelDelegation:
    """switch_model() routes through CompatShim."""

    def test_switch_model_via_shim(self):
        mock_shim = MagicMock(spec=AIAgentCompatShim)

        agent = object.__new__(AIAgent)
        agent._new_pipeline = mock_shim

        agent.switch_model("new-model", "openai", api_key="sk-test")
        mock_shim.switch_model.assert_called_once_with(
            "new-model", "openai", api_key="sk-test", base_url="", api_mode=""
        )