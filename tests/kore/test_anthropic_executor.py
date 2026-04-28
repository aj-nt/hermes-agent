"""Tests for AnthropicStreamingExecutor (Kore Step 4).

The Anthropic streaming path uses the SDK's context manager
(messages.stream) and returns a native Message object.
"""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agent.orchestrator.provider_adapters import (
    AnthropicStreamingExecutor,
    StreamCallbacks,
)


# --- Mock helpers for Anthropic SDK events ---

class MockAnthropicStreamEvent:
    """A single event from Anthropic's streaming API."""

    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockAnthropicStreamManager:
    """Mock for the Anthropic SDK's messages.stream() context manager.

    Iterates through events and returns a final Message from get_final_message().
    """

    def __init__(self, events: list, final_message: Any = None):
        self._events = events
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final_message


class MockAnthropicMessages:
    """Messages namespace with .stream() method."""

    def __init__(self, stream_events: list, final_message: Any = None):
        self._stream_events = stream_events
        self._final_message = final_message

    def stream(self, **kwargs):
        return MockAnthropicStreamManager(
            self._stream_events, self._final_message
        )


class MockAnthropicClient:
    """Mock Anthropic client with .messages attribute."""

    def __init__(self, stream_events: list, final_message: Any = None):
        self.messages = MockAnthropicMessages(stream_events, final_message)


def _make_text_delta_event(text: str) -> MockAnthropicStreamEvent:
    """Create a content_block_delta event with text_delta."""
    delta = SimpleNamespace(type="text_delta", text=text)
    return MockAnthropicStreamEvent("content_block_delta", delta=delta)


def _make_thinking_delta_event(thinking: str) -> MockAnthropicStreamEvent:
    """Create a content_block_delta event with thinking_delta."""
    delta = SimpleNamespace(type="thinking_delta", thinking=thinking)
    return MockAnthropicStreamEvent("content_block_delta", delta=delta)


def _make_tool_use_start_event(name: str, tool_id: str = "tool_0") -> MockAnthropicStreamEvent:
    """Create a content_block_start event for a tool_use block."""
    block = SimpleNamespace(type="tool_use", name=name, id=tool_id)
    return MockAnthropicStreamEvent("content_block_start", content_block=block)


def _make_message_event(model: str = "claude-3-opus") -> MockAnthropicStreamEvent:
    """Create a message_start event."""
    return MockAnthropicStreamEvent("message_start", message=SimpleNamespace(model=model))


def _make_message_delta_event(stop_reason: str = "end_turn") -> MockAnthropicStreamEvent:
    """Create a message_delta event with stop_reason."""
    delta = SimpleNamespace(stop_reason=stop_reason)
    return MockAnthropicStreamEvent("message_delta", delta=delta, usage=SimpleNamespace(output_tokens=50))


# --- AnthropicStreamingExecutor ---

class TestAnthropicStreamingExecutorInit:
    """Test constructor and basic properties."""

    def test_init_stores_client(self):
        client = MagicMock()
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=StreamCallbacks(),
            interrupt_check=lambda: False,
        )
        assert executor._anthropic_client is client

    def test_init_stores_callbacks(self):
        callbacks = StreamCallbacks()
        executor = AnthropicStreamingExecutor(
            anthropic_client=MagicMock(),
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        assert executor._callbacks is callbacks

    def test_init_defaults(self):
        executor = AnthropicStreamingExecutor(
            anthropic_client=MagicMock(),
            callbacks=StreamCallbacks(),
            interrupt_check=lambda: False,
        )
        assert executor._interrupt_check() is False


class TestAnthropicStreamingExecution:
    """Test the execute_streaming method with various event sequences."""

    def test_text_only_stream(self):
        """Stream text content fires stream_delta and returns final message."""
        final_msg = SimpleNamespace(
            id="msg_1", content=[SimpleNamespace(type="text", text="Hello world")],
            model="claude-3-opus", stop_reason="end_turn", usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        deltas_fired = []
        callbacks = StreamCallbacks(
            stream_delta=lambda t: deltas_fired.append(t),
        )
        events = [
            _make_message_event(),
            _make_text_delta_event("Hello "),
            _make_text_delta_event("world"),
            _make_message_delta_event(),
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        result = executor.execute_streaming(api_kwargs={"model": "claude-3-opus"})
        assert result is final_msg
        assert deltas_fired == ["Hello ", "world"]

    def test_thinking_delta_fires_reasoning_callback(self):
        """Thinking deltas fire reasoning_delta callback."""
        reasoning_fired = []
        callbacks = StreamCallbacks(
            reasoning_delta=lambda t: reasoning_fired.append(t),
        )
        final_msg = SimpleNamespace(id="msg_2", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_thinking_delta_event("Let me think..."),
            _make_message_delta_event(),
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        result = executor.execute_streaming(api_kwargs={})
        assert reasoning_fired == ["Let me think..."]

    def test_tool_use_start_fires_callback(self):
        """Tool use content_block_start fires tool_gen_started and first_delta."""
        tools_started = []
        first_delta_fired = []
        callbacks = StreamCallbacks(
            tool_gen_started=lambda n: tools_started.append(n),
            first_delta=lambda: first_delta_fired.append(True),
        )
        final_msg = SimpleNamespace(id="msg_3", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_tool_use_start_event("read_file", "tool_0"),
            _make_message_delta_event(),
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        result = executor.execute_streaming(api_kwargs={})
        assert "read_file" in tools_started
        assert len(first_delta_fired) == 1

    def test_first_delta_fired_once(self):
        """first_delta should fire only once even with multiple events."""
        first_delta_count = []
        callbacks = StreamCallbacks(
            first_delta=lambda: first_delta_count.append(1),
            stream_delta=lambda t: None,
            reasoning_delta=lambda t: None,
        )
        final_msg = SimpleNamespace(id="msg_4", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_text_delta_event("A"),
            _make_text_delta_event("B"),
            _make_text_delta_event("C"),
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        executor.execute_streaming(api_kwargs={})
        assert len(first_delta_count) == 1

    def test_interrupt_stops_stream(self):
        """Interrupt check should stop iteration mid-stream."""
        deltas_fired = []
        call_count = 0
        def interrupt_after_2():
            nonlocal call_count
            call_count += 1
            return call_count > 2

        callbacks = StreamCallbacks(
            stream_delta=lambda t: deltas_fired.append(t),
        )
        final_msg = SimpleNamespace(id="msg_5", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_text_delta_event("A"),
            _make_text_delta_event("B"),
            _make_text_delta_event("C"),  # should not fire
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=interrupt_after_2,
        )
        result = executor.execute_streaming(api_kwargs={})
        # Only "A" should have been streamed before interrupt
        assert deltas_fired == ["A"]

    def test_text_suppressed_during_tool_use(self):
        """Text deltas after tool_use starts should NOT fire stream_delta."""
        deltas_fired = []
        tools_started = []
        callbacks = StreamCallbacks(
            stream_delta=lambda t: deltas_fired.append(t),
            tool_gen_started=lambda n: tools_started.append(n),
        )
        final_msg = SimpleNamespace(id="msg_6", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_text_delta_event("Thinking about it..."),  # should fire
            _make_tool_use_start_event("search", "tool_0"),
            _make_text_delta_event("Using tool now"),  # should NOT fire (has_tool_use=True)
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        executor.execute_streaming(api_kwargs={})
        assert "search" in tools_started
        # Only the first text delta should fire stream_delta
        assert deltas_fired == ["Thinking about it..."]


class TestAnthropicFromAgent:
    """Test the from_agent factory method."""

    def test_from_agent_creates_executor_with_agent_callbacks(self):
        """from_agent should wire agent methods to executor callbacks."""
        agent = MagicMock()
        agent._fire_stream_delta = MagicMock()
        agent._fire_reasoning_delta = MagicMock()
        agent._fire_tool_gen_started = MagicMock()
        agent._touch_activity = MagicMock()

        executor = AnthropicStreamingExecutor.from_agent(agent)

        # Verify callbacks are wired through to the agent
        executor._callbacks.stream_delta("hello")
        agent._fire_stream_delta.assert_called_with("hello")

        executor._callbacks.reasoning_delta("thinking")
        agent._fire_reasoning_delta.assert_called_with("thinking")

        executor._callbacks.tool_gen_started("search")
        agent._fire_tool_gen_started.assert_called_with("search")

    def test_from_agent_wires_interrupt_check(self):
        """from_agent should wire agent._interrupt_requested as interrupt check."""
        agent = MagicMock()
        agent._interrupt_requested = False
        executor = AnthropicStreamingExecutor.from_agent(agent)

        # Interrupt check should read agent._interrupt_requested
        assert executor._interrupt_check() is False
        agent._interrupt_requested = True
        assert executor._interrupt_check() is True

    def test_from_agent_uses_agent_anthropic_client(self):
        """from_agent should use agent._anthropic_client."""
        agent = MagicMock()
        executor = AnthropicStreamingExecutor.from_agent(agent)
        assert executor._anthropic_client is agent._anthropic_client


class TestAnthropicActivityResultTouch:
    """Test that activity_touch and on_last_chunk_time callbacks work."""

    def test_activity_touch_callback_fires(self):
        """activity_touch callback should fire on each event."""
        activities = []
        callbacks = StreamCallbacks(
            activity_touch=lambda msg: activities.append(msg),
        )
        final_msg = SimpleNamespace(id="msg_7", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_text_delta_event("hi"),
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        executor.execute_streaming(api_kwargs={})
        assert len(activities) >= 1  # At least "receiving stream response"

    def test_on_last_chunk_time_callback(self):
        """on_last_chunk_time should be called on every event iteration."""
        chunk_time_calls = 0
        callbacks = StreamCallbacks(
            on_last_chunk_time=lambda: None,  # just count calls
            stream_delta=lambda t: None,
        )
        final_msg = SimpleNamespace(id="msg_8", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_text_delta_event("A"),
            _make_text_delta_event("B"),
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        # Patch on_last_chunk_time to count calls
        original_olt = executor._callbacks.on_last_chunk_time
        def counting_olt():
            nonlocal chunk_time_calls
            chunk_time_calls += 1
        executor._callbacks.on_last_chunk_time = counting_olt
        executor.execute_streaming(api_kwargs={})
        # on_last_chunk_time fires on every event iteration (3 events = 3 calls)
        assert chunk_time_calls == 3

    def test_deltas_were_sent_tracking(self):
        """deltas_were_sent should be True when stream_delta fires for text."""
        callbacks = StreamCallbacks(stream_delta=lambda t: None)
        final_msg = SimpleNamespace(id="msg_9", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_text_delta_event("hello"),
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        assert executor.deltas_were_sent is False
        executor.execute_streaming(api_kwargs={})
        assert executor.deltas_were_sent is True

    def test_deltas_were_sent_false_when_only_thinking(self):
        """deltas_were_sent should stay False when only thinking deltas fire."""
        callbacks = StreamCallbacks(reasoning_delta=lambda t: None)
        final_msg = SimpleNamespace(id="msg_10", content=[], model="claude-3-opus")
        events = [
            _make_message_event(),
            _make_thinking_delta_event("thinking..."),
        ]
        client = MockAnthropicClient(events, final_msg)
        executor = AnthropicStreamingExecutor(
            anthropic_client=client,
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        executor.execute_streaming(api_kwargs={})
        assert executor.deltas_were_sent is False