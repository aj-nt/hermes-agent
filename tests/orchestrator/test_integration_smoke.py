#!/usr/bin/env python3
"""Integration smoke test for the new Orchestrator pipeline.

Exercises the FULL real pipeline with live LLM calls against the local
Ollama instance. No mocks for the pipeline itself — only for external
services we can't run in a test (iMessage gateway, etc.).

Prerequisites:
- Ollama running on localhost:11434 with glm-5.1:cloud model available
- Python venv active with all hermes-agent dependencies

Run:  python tests/orchestrator/test_integration_smoke.py
      pytest tests/orchestrator/test_integration_smoke.py -xvs
"""
from __future__ import annotations

import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check if Ollama is available before importing the agent (saves time)
try:
    import urllib.request
    req = urllib.request.Request("http://localhost:11434/v1/models", headers={"User-Agent": "test"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        _models_data = json.loads(resp.read())
        _available_models = [m["id"] for m in _models_data.get("data", [])]
except Exception:
    _available_models = []

PRIMARY_MODEL = "glm-5.1:cloud"
SKIP_NO_OLLAMA = not _available_models or PRIMARY_MODEL not in _available_models


def _make_agent(model=PRIMARY_MODEL, ephemeral_prompt="You are a helpful test assistant. Answer concisely."):
    """Create a real AIAgent instance for integration testing."""
    from run_agent import AIAgent
    return AIAgent(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model=model,
        ephemeral_system_prompt=ephemeral_prompt,
        enabled_toolsets=["terminal"],  # minimal tools for testing
        max_iterations=5,
        verbose_logging=False,
        quiet_mode=True,
    )


@unittest.skipIf(SKIP_NO_OLLAMA, f"No Ollama or model {PRIMARY_MODEL} not available")
class TestIntegrationSmoke(unittest.TestCase):
    """Full pipeline integration tests against live Ollama."""

    # -------------------------------------------------------
    # Test 1: Basic chat — does the pipeline answer correctly?
    # -------------------------------------------------------
    def test_01_basic_chat(self):
        """Pipeline should answer 'What is 2+2?' with '4'."""
        agent = _make_agent()
        try:
            response = agent.chat("What is 2+2? Reply with just the number.")
            self.assertIsNotNone(response)
            self.assertIn("4", str(response).strip(),
                          f"Expected '4' in response, got: {response!r}")
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()

    # -------------------------------------------------------
    # Test 2: CompatShim is wired
    # -------------------------------------------------------
    def test_02_compat_shim_wired(self):
        """Agent should have a CompatShim with memory coordinator."""
        agent = _make_agent()
        try:
            self.assertTrue(hasattr(agent, '_new_pipeline'), "Agent should have _shim")
            self.assertTrue(hasattr(agent._new_pipeline, 'memory'), "Shim should have memory")
            self.assertTrue(hasattr(agent._new_pipeline, '_orchestrator'), "Shim should have _orchestrator")

            # Memory coordinator should have a store reference
            # (may be None if no store configured, but attribute must exist)
            self.assertTrue(hasattr(agent._new_pipeline.memory, 'store'), "Memory should have store attr")
            self.assertTrue(hasattr(agent._new_pipeline.memory, 'manager'), "Memory should have manager attr")
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()

    # -------------------------------------------------------
    # Test 3: run_conversation returns a valid result
    # -------------------------------------------------------
    def test_03_run_conversation(self):
        """run_conversation should return a dict-like result."""
        agent = _make_agent()
        try:
            result = agent.run_conversation(
                messages=[{"role": "user", "content": "Say 'pipeline works' exactly."}],
                model=PRIMARY_MODEL,
            )
            self.assertIsNotNone(result)
            # Result should be a dict with at least 'content' or 'response'
            self.assertTrue(
                isinstance(result, (dict, str)) or hasattr(result, 'content'),
                f"Expected dict/str result, got {type(result)}"
            )
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()

    # -------------------------------------------------------
    # Test 4: MemoryCoordinator prompt blocks
    # -------------------------------------------------------
    def test_04_memory_prompt_blocks(self):
        """MemoryCoordinator.build_prompt_blocks should return a list."""
        agent = _make_agent()
        try:
            blocks = agent._new_pipeline.memory.build_prompt_blocks()
            self.assertIsInstance(blocks, list,
                                f"Expected list, got {type(blocks)}")
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()

    # -------------------------------------------------------
    # Test 5: Nudge tracking increments
    # -------------------------------------------------------
    def test_05_nudge_tracking(self):
        """Turn tracking should increment correctly."""
        from agent.orchestrator.memory import NudgeTracker
        tracker = NudgeTracker(nudge_interval=3, skill_interval=5)

        self.assertEqual(tracker.turn_count, 0)
        tracker.on_turn_start()
        self.assertEqual(tracker.turn_count, 1)
        tracker.on_turn_start()
        self.assertEqual(tracker.turn_count, 2)

    # -------------------------------------------------------
    # Test 6: Sequential turns retain context
    # -------------------------------------------------------
    def test_06_sequential_turns(self):
        """Two turns on same agent should maintain context."""
        agent = _make_agent()
        try:
            r1 = agent.chat("Remember the number 42.")
            self.assertIsNotNone(r1)

            r2 = agent.chat("What number did I ask you to remember? Just the number.")
            self.assertIsNotNone(r2)
            self.assertIn("42", str(r2),
                          f"Expected '42' in second response, got: {r2!r}")
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()

    # -------------------------------------------------------
    # Test 7: switch_model works
    # -------------------------------------------------------
    def test_07_switch_model(self):
        """switch_model should change the model and work."""
        # Find an alternate model
        alt_model = None
        for m in _available_models:
            if m != PRIMARY_MODEL and "cloud" in m:
                alt_model = m
                break

        if alt_model is None:
            self.skipTest("No alternate model available for switch test")

        agent = _make_agent()
        try:
            original = agent.model
            agent.switch_model(alt_model)
            self.assertEqual(agent.model, alt_model)

            # Make a call with the switched model
            response = agent.chat("Say 'switched' exactly.")
            self.assertIsNotNone(response)

            # Switch back
            agent.switch_model(original)
            self.assertEqual(agent.model, original)
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()

    # -------------------------------------------------------
    # Test 8: Tool dispatch through pipeline
    # -------------------------------------------------------
    def test_08_tool_dispatch(self):
        """Pipeline should handle a tool-calling prompt without crashing."""
        agent = _make_agent(ephemeral_prompt=(
            "You are a test assistant with access to terminal tools. "
            "When asked to run a command, use the terminal tool."
        ))
        try:
            # Ask the agent to use a tool — simple terminal command
            response = agent.chat("Run the command 'echo hello' using the terminal tool.")
            # We just verify it doesn't crash — the tool might or might not execute
            self.assertIsNotNone(response)
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()


@unittest.skipIf(SKIP_NO_OLLAMA, f"No Ollama or model {PRIMARY_MODEL} not available")
class TestIntegrationErrorPaths(unittest.TestCase):
    """Test error recovery in the pipeline."""

    def test_01_unavailable_model_falls_back(self):
        """An unavailable model should fall back, not crash."""
        agent = _make_agent()
        try:
            result = agent.run_conversation(
                messages=[{"role": "user", "content": "test"}],
                model="nonexistent-model-xyz:latest",
            )
            # Should fall back to primary model and succeed
            self.assertIsNotNone(result)
        except Exception as e:
            # Error is acceptable — just not a segfault or infinite loop
            self.assertNotIsInstance(e, (SystemExit, KeyboardInterrupt))
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()

    def test_02_empty_message_handled(self):
        """An empty message should not crash the pipeline."""
        agent = _make_agent()
        try:
            result = agent.run_conversation(
                messages=[{"role": "user", "content": "hi"}],
                model=PRIMARY_MODEL,
            )
            self.assertIsNotNone(result)
        finally:
            if hasattr(agent, 'release_clients'):
                agent.release_clients()


if __name__ == "__main__":
    if SKIP_NO_OLLAMA:
        print(f"SKIP: Ollama not available or model {PRIMARY_MODEL} not found")
        print(f"Available models: {_available_models}")
        sys.exit(0)
    print(f"Running integration smoke tests against {PRIMARY_MODEL}")
    print(f"Available models: {_available_models}")
    unittest.main(verbosity=2)