#!/usr/bin/env python3
"""Extended integration smoke tests — multi-turn, switch, invariants, error recovery.

Exercises the pipeline at a deeper level than the basic smoke test.
Run standalone:  python tests/orchestrator/test_integration_extended.py
Not part of pytest suite (makes real LLM calls, ~2 min total).
"""
import os
import sys
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("TEST_MODEL", "glm-5.1:cloud")


def _make_agent(**overrides):
    from run_agent import AIAgent
    defaults = dict(
        base_url=BASE_URL,
        api_key="ollama",
        model=MODEL,
        ephemeral_system_prompt="You are a helpful math test assistant. Answer concisely.",
    )
    defaults.update(overrides)
    return AIAgent(**defaults)


class TestExtendedIntegration(unittest.TestCase):
    """Deeper integration tests — multi-turn, switch, invariants, error recovery."""

    def test_01_invariants_after_init(self):
        """Structural invariants hold after agent creation."""
        agent = _make_agent()
        from agent.orchestrator.compat import AIAgentCompatShim
        from agent.orchestrator.memory import MemoryCoordinator
        self.assertIsInstance(agent._new_pipeline, AIAgentCompatShim)
        self.assertIsInstance(agent._new_pipeline.memory, MemoryCoordinator)
        self.assertTrue(agent.session_id)

    def test_02_sequential_chat_turns(self):
        """Two chat() calls on the same agent maintain independence."""
        agent = _make_agent()
        r1 = agent.chat("What is 3+4?")
        self.assertIn("7", r1, f"Expected '7' in response, got: {r1[:200]}")
        r2 = agent.chat("What is 8+9?")
        self.assertIn("17", r2, f"Expected '17' in response, got: {r2[:200]}")

    def test_03_interrupt_does_not_crash(self):
        """Calling interrupt() on an idle agent doesn't crash."""
        agent = _make_agent()
        result = agent._new_pipeline.interrupt("test_interrupt")
        self.assertIsNone(result)

    def test_04_memory_coordinator_handles_unknown_tool(self):
        """MemoryCoordinator raises on unknown tool calls."""
        agent = _make_agent()
        coord = agent._new_pipeline.memory
        with self.assertRaises(ValueError):
            coord.handle_tool_call("nonexistent_tool", {}, metadata={})

    def test_05_compat_shim_chat_rejects_none(self):
        """CompatShim.chat() rejects None message with TypeError."""
        agent = _make_agent()
        with self.assertRaises(TypeError):
            agent._new_pipeline.chat(None)

    def test_06_compat_shim_chat_rejects_empty(self):
        """CompatShim.chat() rejects whitespace-only message with ValueError."""
        agent = _make_agent()
        with self.assertRaises(ValueError):
            agent._new_pipeline.chat("   ")

    def test_07_release_clients_idempotent(self):
        """release_clients can be called multiple times without error."""
        agent = _make_agent()
        agent._new_pipeline.release_clients()
        agent._new_pipeline.release_clients()

    def test_08_sequential_turns_no_state_corruption(self):
        """Multiple sequential chats on same agent don't corrupt state."""
        agent = _make_agent()
        results = []
        for i in range(3):
            r = agent.chat(f"What is {i}+{i}? Just give me the number.")
            results.append(r)
        for i, r in enumerate(results):
            self.assertTrue(len(r) > 0, f"Turn {i} returned empty")


if __name__ == "__main__":
    print(f"Running extended integration tests against {MODEL}")
    unittest.main(verbosity=2)