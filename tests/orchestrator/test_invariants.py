"""Tests for AIAgent.__post_init__ invariant checks.

These tests verify that structural invariants are enforced at
instantiation time.  They would have caught the 6B-2 indent bug
where CompatShim was never created in chat_completions mode.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestAIAgentInvariants(unittest.TestCase):
    """Every AIAgent must satisfy these invariants after __init__."""

    def _make_agent(self, **overrides):
        from run_agent import AIAgent
        defaults = dict(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="gemma4:26b",
            ephemeral_system_prompt="You are a test assistant.",
        )
        defaults.update(overrides)
        return AIAgent(**defaults)

    # ---- _new_pipeline must exist ----

    def test_compat_shim_exists_after_init(self):
        """Invariant: agent._new_pipeline is a CompatShim, never None."""
        agent = self._make_agent()
        from agent.orchestrator.compat import AIAgentCompatShim
        self.assertIsInstance(agent._new_pipeline, AIAgentCompatShim)

    def test_compat_shim_exists_for_chat_completions(self):
        """The bug: CompatShim was inside the anthropic_messages if-block."""
        agent = self._make_agent(api_mode="chat_completions")
        from agent.orchestrator.compat import AIAgentCompatShim
        self.assertIsNotNone(agent._new_pipeline)

    def test_compat_shim_exists_for_anthropic(self):
        """CompatShim must also exist for anthropic_messages mode."""
        agent = self._make_agent(api_mode="anthropic_messages")
        from agent.orchestrator.compat import AIAgentCompatShim
        self.assertIsNotNone(agent._new_pipeline)

    # ---- MemoryCoordinator must exist and be wired ----

    def test_memory_coordinator_attached(self):
        """Invariant: agent._new_pipeline.memory is a MemoryCoordinator."""
        agent = self._make_agent()
        from agent.orchestrator.memory import MemoryCoordinator
        self.assertIsInstance(agent._new_pipeline.memory, MemoryCoordinator)

    def test_memory_coordinator_references_store(self):
        """MemoryCoordinator.store must point to the agent's MemoryStore."""
        agent = self._make_agent()
        self.assertIs(agent._new_pipeline.memory.store, agent._memory_store)

    def test_memory_coordinator_references_manager(self):
        """MemoryCoordinator.manager must point to the agent's MemoryManager."""
        agent = self._make_agent()
        self.assertIs(agent._new_pipeline.memory.manager, agent._memory_manager)

    # ---- Session state must be initialized ----

    def test_session_id_not_empty(self):
        """Invariant: agent.session_id must be a non-empty string."""
        agent = self._make_agent()
        self.assertIsInstance(agent.session_id, str)
        self.assertTrue(len(agent.session_id) > 0)

    # ---- No None delegation targets ----

    def test_no_none_delegation_targets(self):
        """Critical methods must not delegate to None."""
        agent = self._make_agent()
        shim = agent._new_pipeline
        # These must be callable, not None
        for method_name in ("switch_model", "interrupt", "chat",
                            "run_conversation", "commit_memory_session",
                            "shutdown_memory_provider", "release_clients"):
            method = getattr(shim, method_name, None)
            self.assertIsNotNone(method, f"shim.{method_name} is None")
            self.assertTrue(callable(method), f"shim.{method_name} not callable")


if __name__ == "__main__":
    unittest.main()