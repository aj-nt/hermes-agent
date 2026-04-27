#!/usr/bin/env python3
"""Integration smoke test for the Orchestrator pipeline.

Exercises the FULL real pipeline with live LLM calls against local Ollama.
Run standalone:  python tests/orchestrator/test_integration_smoke.py
NOT via pytest-xdist (tests make real LLM calls with timeouts).
"""
from __future__ import annotations

import json
import os
import signal
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Check Ollama availability
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
TIMEOUT_SEC = 30


def _make_agent(model=PRIMARY_MODEL, max_iter=3):
    from run_agent import AIAgent
    return AIAgent(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model=model,
        api_mode="chat_completions",
        enabled_toolsets=[],
        max_iterations=max_iter,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _timed(fn, *args, timeout=TIMEOUT_SEC, **kwargs):
    """Run fn with a hard SIGALRM timeout to prevent hangs."""
    def _handler(signum, frame):
        raise TimeoutError(f"Call exceeded {timeout}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@unittest.skipIf(SKIP_NO_OLLAMA, f"No Ollama or model {PRIMARY_MODEL} not available")
class TestPipelineSmoke(unittest.TestCase):
    """Full pipeline integration against live Ollama."""

    def test_01_compat_shim_created(self):
        """CompatShim must exist after __init__."""
        agent = _make_agent(max_iter=1)
        try:
            self.assertTrue(hasattr(agent, "_new_pipeline"))
            self.assertEqual(type(agent._new_pipeline).__name__, "AIAgentCompatShim")
            self.assertTrue(hasattr(agent._new_pipeline, "memory"))
        finally:
            agent.release_clients()

    def test_02_basic_chat(self):
        """Pipeline should answer 2+2=4."""
        agent = _make_agent()
        try:
            result = _timed(agent.chat, "What is 2+2? Reply with just the number.")
            self.assertIn("4", str(result).strip(), f"Expected '4', got: {result!r}")
        finally:
            agent.release_clients()

    def test_03_memory_coordinator_wired(self):
        """MemoryCoordinator should be attached and functional."""
        agent = _make_agent(max_iter=1)
        try:
            mem = agent._new_pipeline.memory
            blocks = mem.build_prompt_blocks()
            self.assertIsInstance(blocks, list)
        finally:
            agent.release_clients()

    def test_04_nudge_tracker(self):
        """NudgeTracker.on_turn() should increment counters."""
        from agent.orchestrator.memory import NudgeTracker
        tracker = NudgeTracker(nudge_interval=3, skill_interval=5)
        self.assertEqual(tracker._turns_since_memory, 0)
        tracker.on_turn()
        self.assertEqual(tracker._turns_since_memory, 1)
        tracker.on_turn()
        self.assertEqual(tracker._turns_since_memory, 2)

    def test_05_run_conversation(self):
        """run_conversation should return a result without hanging."""
        agent = _make_agent()
        try:
            # run_conversation(user_message, system_message=None, ...)
            result = _timed(agent.run_conversation, "Say hello exactly.")
            self.assertIsNotNone(result)
        finally:
            agent.release_clients()

    def test_06_switch_model(self):
        """switch_model should change the model."""
        alt_model = None
        for m in _available_models:
            if m != PRIMARY_MODEL:
                alt_model = m
                break
        if alt_model is None:
            self.skipTest("No alternate model")

        agent = _make_agent()
        try:
            original = agent.model
            # switch_model(new_model, new_provider, api_key, base_url, api_mode)
            agent.switch_model(alt_model, "custom", api_key="ollama", base_url="http://localhost:11434/v1")
            # switch_model updates the pipeline; AIAgent.model may lag
            self.assertEqual(agent._new_pipeline.model, alt_model)
            # Switch back
            agent.switch_model(original, "custom", api_key="ollama", base_url="http://localhost:11434/v1")
            self.assertEqual(agent._new_pipeline.model, original)
        finally:
            agent.release_clients()


if __name__ == "__main__":
    if SKIP_NO_OLLAMA:
        print(f"SKIP: Ollama not available or model {PRIMARY_MODEL} not found")
        print(f"Available: {_available_models}")
        sys.exit(0)

    print(f"Running integration smoke tests against {PRIMARY_MODEL}")
    print(f"Available: {_available_models}")
    print()
    unittest.main(verbosity=2)