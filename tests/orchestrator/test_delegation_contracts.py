"""Tests for type contracts at CompatShim delegation boundaries.

Every method on CompatShim that delegates to AIAgent must validate
its inputs before delegating and validate the return type when
coming back.  These tests verify the contracts exist and catch
type errors early.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agent.orchestrator.compat import AIAgentCompatShim
from agent.orchestrator.memory import MemoryCoordinator


def _make_shim():
    """Create a CompatShim with a mocked AIAgent."""
    mock_agent = MagicMock()
    mock_agent._memory_store = MagicMock()
    mock_agent._memory_manager = MagicMock()
    mock_agent._memory_enabled = True
    mock_agent._user_profile_enabled = False
    mock_agent.session_id = "test-session"
    mock_agent._memory_nudge_interval = 5
    mock_agent._skill_nudge_interval = 5
    shim = AIAgentCompatShim(parent_agent=mock_agent)
    return shim, mock_agent


class TestDelegationContracts(unittest.TestCase):
    """Type contracts at the CompatShim -> AIAgent boundary."""

    # ---- chat() must return str and validate inputs ----

    def test_chat_returns_string(self):
        """chat() return type is enforced as str."""
        shim, mock_agent = _make_shim()
        mock_agent.chat.return_value = "Hello"
        result = shim.chat("hello")
        self.assertIsInstance(result, str)

    def test_chat_rejects_none_message(self):
        """chat() must reject None message."""
        shim, _ = _make_shim()
        with self.assertRaises(TypeError):
            shim.chat(None)

    def test_chat_rejects_empty_message(self):
        """chat() must reject empty string message."""
        shim, _ = _make_shim()
        with self.assertRaises(ValueError):
            shim.chat("   ")

    # ---- switch_model() must validate args ----

    def test_switch_model_requires_model_and_provider(self):
        """switch_model must have both model and provider."""
        shim, _ = _make_shim()
        with self.assertRaises(TypeError):
            shim.switch_model()  # no args at all

    # ---- interrupt() must accept valid signal ----

    def test_interrupt_returns_none(self):
        """interrupt() returns None (fire-and-forget signal)."""
        shim, mock_agent = _make_shim()
        result = shim.interrupt("user_stop")
        self.assertIsNone(result)

    # ---- run_conversation delegates correctly ----

    def test_run_conversation_passes_message(self):
        """run_conversation forwards the user message."""
        shim, mock_agent = _make_shim()
        mock_agent.run_conversation.return_value = {"result": "ok"}
        result = shim.run_conversation(
            "hello",
            ephemeral_system_prompt="You are helpful.",
        )
        self.assertIsNotNone(result)

    # ---- release_clients/commit_memory_session/shutdown ----

    def test_release_clients_is_idempotent(self):
        """release_clients can be called multiple times safely."""
        shim, mock_agent = _make_shim()
        shim.release_clients()
        shim.release_clients()
        self.assertEqual(mock_agent.release_clients.call_count, 2)

    def test_commit_memory_session_delegates(self):
        """commit_memory_session forwards to agent."""
        shim, mock_agent = _make_shim()
        shim.commit_memory_session()
        mock_agent.commit_memory_session.assert_called_once()

    def test_shutdown_memory_provider_delegates(self):
        """shutdown_memory_provider forwards to agent."""
        shim, mock_agent = _make_shim()
        shim.shutdown_memory_provider()
        mock_agent.shutdown_memory_provider.assert_called_once()


class TestMemoryCoordinatorContracts(unittest.TestCase):
    """Type contracts on MemoryCoordinator method signatures."""

    def _make_coordinator(self):
        coord = MemoryCoordinator(session_id="test-session")
        coord.store = MagicMock()
        coord.manager = MagicMock()
        coord._memory_enabled = True
        coord._user_profile_enabled = False
        return coord

    def test_handle_tool_call_rejects_none_name(self):
        """handle_tool_call must reject None tool name."""
        coord = self._make_coordinator()
        with self.assertRaises(ValueError):
            coord.handle_tool_call(None, {}, metadata={})

    def test_handle_tool_call_rejects_empty_name(self):
        """handle_tool_call must reject empty string tool name."""
        coord = self._make_coordinator()
        with self.assertRaises(ValueError):
            coord.handle_tool_call("", {}, metadata={})

    def test_handle_tool_call_rejects_whitespace_name(self):
        """handle_tool_call must reject whitespace-only tool name."""
        coord = self._make_coordinator()
        with self.assertRaises(ValueError):
            coord.handle_tool_call("   ", {}, metadata={})

    def test_handle_tool_call_rejects_non_dict_args(self):
        """handle_tool_call must reject non-dict args."""
        coord = self._make_coordinator()
        with self.assertRaises(TypeError):
            coord.handle_tool_call("memory_add", "not a dict", metadata={})

    def test_build_prompt_blocks_returns_list(self):
        """build_prompt_blocks must return a list."""
        coord = self._make_coordinator()
        coord.store.format_for_system_prompt.return_value = "- mem line 1\n- mem line 2"
        result = coord.build_prompt_blocks()
        self.assertIsInstance(result, list)

    def test_on_turn_end_accepts_strings(self):
        """on_turn_end accepts user_msg and assistant_msg strings."""
        coord = self._make_coordinator()
        coord.on_turn_end(user_msg="hello", assistant_msg="hi there")

    def test_on_session_end_no_required_args(self):
        """on_session_end needs no arguments."""
        coord = self._make_coordinator()
        coord.on_session_end(messages=[])


if __name__ == "__main__":
    unittest.main()