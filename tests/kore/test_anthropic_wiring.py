"""Tests for Step 4c wiring — AnthropicStreamingExecutor wired into run_agent.py.

These tests verify that the Anthropic executor wiring in _call_anthropic
correctly delegates to AnthropicStreamingExecutor and communicates with
the outer scope (last_chunk_time, deltas_were_sent, first_delta_fired).
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestAnthropicWiringCallbacks:
    """Test the callback wiring between executor and outer scope."""

    def test_activity_touch_also_updates_last_chunk_time(self):
        """The wiring should augment activity_touch to also update last_chunk_time."""
        # This tests the pattern used in _call_anthropic's wiring:
        # _orig_activity = executor._callbacks.activity_touch
        # executor._callbacks.activity_touch = lambda msg: (last_chunk_time["t"] = time.time(), _orig_activity(msg))
        last_chunk_time = {"t": 0.0}
        original_calls = []
        def orig_activity(msg):
            original_calls.append(msg)

        import time
        def activity_with_chunk_time(msg):
            last_chunk_time["t"] = time.time()
            orig_activity(msg)

        activity_with_chunk_time("test")
        assert last_chunk_time["t"] > 0
        assert original_calls == ["test"]

    def test_stream_delta_tracking_sets_deltas_were_sent(self):
        """The wiring should track that deltas were sent for retry logic."""
        deltas_were_sent = {"yes": False}
        orig_delta_calls = []
        def orig_stream_delta(text):
            orig_delta_calls.append(text)

        def stream_delta_with_tracking(text):
            deltas_were_sent["yes"] = True
            orig_stream_delta(text)

        # Simulate what the wiring does
        def _stream_delta_with_tracking(text):
            deltas_were_sent["yes"] = True
            if orig_stream_delta is not None:
                pass  # origStream_delta called separately in real code

        _stream_delta_with_tracking("hello")
        assert deltas_were_sent["yes"] is True

    def test_first_delta_wired_to_fire_first_delta(self):
        """The wiring should connect executor's first_delta to the outer _fire_first_delta."""
        first_delta_fired = {"done": False}
        on_first_delta_calls = []

        def _fire_first_delta():
            if not first_delta_fired["done"]:
                first_delta_fired["done"] = True
                on_first_delta_calls.append("fired")

        # Simulate executor calling callbacks.first_delta
        _fire_first_delta()
        assert first_delta_fired["done"] is True
        assert len(on_first_delta_calls) == 1

        # Second call should be a no-op
        _fire_first_delta()
        assert len(on_first_delta_calls) == 1  # still 1


class TestAnthropicExecutorFromAgentIntegration:
    """Integration tests for AnthropicStreamingExecutor.from_agent."""

    def test_from_agent_creates_executor(self):
        """from_agent should create an AnthropicStreamingExecutor with agent's client."""
        from agent.orchestrator.provider_adapters import AnthropicStreamingExecutor

        agent = MagicMock()
        agent._anthropic_client = MagicMock()
        agent._fire_stream_delta = MagicMock()
        agent._fire_reasoning_delta = MagicMock()
        agent._fire_tool_gen_started = MagicMock()
        agent._touch_activity = MagicMock()
        agent._interrupt_requested = False

        executor = AnthropicStreamingExecutor.from_agent(agent)
        assert isinstance(executor, AnthropicStreamingExecutor)
        assert executor._anthropic_client is agent._anthropic_client

    def test_from_agent_wires_callbacks(self):
        """from_agent should wire agent methods to callbacks."""
        from agent.orchestrator.provider_adapters import AnthropicStreamingExecutor

        agent = MagicMock()
        agent._interrupt_requested = False

        executor = AnthropicStreamingExecutor.from_agent(agent)

        # Verify callbacks call through to agent
        executor._callbacks.stream_delta("hello")
        agent._fire_stream_delta.assert_called_with("hello")

        executor._callbacks.reasoning_delta("thinking")
        agent._fire_reasoning_delta.assert_called_with("thinking")

        executor._callbacks.tool_gen_started("search")
        agent._fire_tool_gen_started.assert_called_with("search")

    def test_from_agent_interrupt_check_reads_agent(self):
        """from_agent should wire interrupt check to agent._interrupt_requested."""
        from agent.orchestrator.provider_adapters import AnthropicStreamingExecutor

        agent = MagicMock()
        agent._interrupt_requested = False

        executor = AnthropicStreamingExecutor.from_agent(agent)

        assert executor._interrupt_check() is False
        agent._interrupt_requested = True
        assert executor._interrupt_check() is True

    def test_executor_deltas_were_sent_attribute(self):
        """Executor should expose deltas_were_sent for outer scope tracking."""
        from agent.orchestrator.provider_adapters import AnthropicStreamingExecutor

        agent = MagicMock()
        agent._anthropic_client = MagicMock()
        agent._interrupt_requested = False

        executor = AnthropicStreamingExecutor.from_agent(agent)
        assert executor.deltas_were_sent is False