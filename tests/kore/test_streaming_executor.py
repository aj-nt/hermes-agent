"""Tests for StreamingChatCompletionsExecutor — Step 3b extraction.

The executor encapsulates the streaming call logic currently in
_call_chat_completions (run_agent.py ~4435-4686). It:
- Builds stream kwargs with timeouts
- Creates a per-request OpenAI client
- Iterates SSE chunks, accumulating content/tool_calls/reasoning/usage
- Fires callbacks (stream_delta, reasoning_delta, tool_gen_started, on_first_delta)
- Handles Ollama index-reuse for parallel tool calls
- Repairs truncated tool-call JSON arguments
- Returns a SimpleNamespace matching non-streaming response shape

TDD: write failing tests first, then implement to make them pass.
"""

import json
import uuid
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agent.orchestrator.provider_adapters import (
    OpenAICompatibleProvider,
    RequestConfig,
    StreamingChatCompletionsExecutor,
    StreamCallbacks,
    StreamResult,
)


# ============================================================================
# Helpers
# ============================================================================

def make_delta(
    content: Optional[str] = None,
    tool_calls: Optional[List] = None,
    reasoning: Optional[str] = None,
    reasoning_content: Optional[str] = None,
):
    """Create a mock delta object matching OpenAI ChatCompletionChunk delta."""
    delta = SimpleNamespace()
    if content is not None:
        delta.content = content
    else:
        delta.content = None
    if reasoning is not None:
        delta.reasoning = reasoning
    elif reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    else:
        delta.reasoning_content = None
        delta.reasoning = None
    if tool_calls is not None:
        delta.tool_calls = tool_calls
    else:
        delta.tool_calls = None
    return delta


def make_chunk(
    delta: SimpleNamespace,
    finish_reason: Optional[str] = None,
    model: Optional[str] = None,
    usage=None,
):
    """Create a mock SSE chunk."""
    chunk = SimpleNamespace()
    chunk.choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    chunk.model = model
    chunk.usage = usage
    return chunk


class MockStream:
    """A mock OpenAI Stream object that is iterable and has a .response attribute."""

    def __init__(self, chunks, response_headers=None):
        self._chunks = chunks
        self.response = SimpleNamespace(headers=response_headers or {})

    def __iter__(self):
        return iter(self._chunks)


def make_tool_call_delta(
    index: int = 0,
    id: str = "",
    name: str = "",
    arguments: str = "",
    extra_content=None,
):
    """Create a mock tool call delta."""
    tc = SimpleNamespace()
    tc.index = index
    tc.id = id
    tc.name = name
    tc.function = SimpleNamespace(name=name, arguments=arguments)
    tc.extra_content = extra_content
    # model_extra for OpenRouter Gemini compatibility
    tc.model_extra = {"extra_content": extra_content} if extra_content else None
    return tc


def make_tool_call_delta_with_model_extra(
    index: int = 0,
    id: str = "",
    name: str = "",
    arguments: str = "",
    model_extra: Optional[Dict] = None,
):
    """Create a mock tool call delta with explicit model_extra."""
    tc = SimpleNamespace()
    tc.index = index
    tc.id = id
    tc.name = name
    tc.function = SimpleNamespace(name=name, arguments=arguments)
    tc.extra_content = None
    tc.model_extra = model_extra
    return tc


# ============================================================================
# StreamCallbacks
# ============================================================================

class TestStreamCallbacks:
    """StreamCallbacks bundles the callbacks for streaming execution."""

    def test_stream_callbacks_creation(self):
        """StreamCallbacks should be constructable with all callbacks."""
        on_delta = MagicMock()
        on_reasoning = MagicMock()
        on_tool_gen = MagicMock()
        on_first_delta = MagicMock()
        on_activity = MagicMock()
        cb = StreamCallbacks(
            stream_delta=on_delta,
            reasoning_delta=on_reasoning,
            tool_gen_started=on_tool_gen,
            first_delta=on_first_delta,
            activity_touch=on_activity,
        )
        assert cb.stream_delta is on_delta
        assert cb.reasoning_delta is on_reasoning
        assert cb.tool_gen_started is on_tool_gen
        assert cb.first_delta is on_first_delta
        assert cb.activity_touch is on_activity

    def test_stream_callbacks_defaults_to_none(self):
        """StreamCallbacks should default all callbacks to None."""
        cb = StreamCallbacks()
        assert cb.stream_delta is None
        assert cb.reasoning_delta is None
        assert cb.tool_gen_started is None
        assert cb.first_delta is None
        assert cb.activity_touch is None


# ============================================================================
# StreamResult
# ============================================================================

class TestStreamResult:
    """StreamResult holds the accumulated response from streaming execution."""

    def test_stream_result_fields(self):
        """StreamResult should hold content, tool_calls, reasoning, usage, model."""
        result = StreamResult(
            content="Hello",
            tool_calls=[{"id": "tc_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
            reasoning="Let me think...",
            finish_reason="stop",
            model_name="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            partial_tool_names=["test"],
        )
        assert result.content == "Hello"
        assert len(result.tool_calls) == 1
        assert result.reasoning == "Let me think..."
        assert result.finish_reason == "stop"
        assert result.model_name == "gpt-4o"
        assert result.partial_tool_names == ["test"]

    def test_stream_result_defaults(self):
        """StreamResult should have sensible defaults."""
        result = StreamResult()
        assert result.content is None
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.finish_reason == "stop"
        assert result.model_name is None
        assert result.usage is None
        assert result.partial_tool_names == []

    def test_stream_result_to_response_namespace(self):
        """StreamResult should produce a SimpleNamespace matching OpenAI shape."""
        result = StreamResult(
            content="Hello",
            finish_reason="stop",
            model_name="gpt-4o",
        )
        ns = result.to_response()
        # Should match non-streaming response shape
        assert hasattr(ns, "id")
        assert hasattr(ns, "model")
        assert ns.model == "gpt-4o"
        assert len(ns.choices) == 1
        msg = ns.choices[0].message
        assert msg.role == "assistant"
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert ns.choices[0].finish_reason == "stop"

    def test_stream_result_to_response_with_tool_calls(self):
        """StreamResult with tool_calls should produce proper SimpleNamespace."""
        tc = SimpleNamespace(
            id="tc_123",
            type="function",
            extra_content=None,
            function=SimpleNamespace(name="read_file", arguments='{"path": "/tmp/x"}'),
        )
        result = StreamResult(
            content=None,
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        ns = result.to_response()
        msg = ns.choices[0].message
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "read_file"

    def test_stream_result_to_response_with_reasoning(self):
        """StreamResult with reasoning should include it in the message."""
        result = StreamResult(
            content="The answer is 42.",
            reasoning="I need to think about this...",
            finish_reason="stop",
        )
        ns = result.to_response()
        msg = ns.choices[0].message
        assert msg.reasoning_content == "I need to think about this..."


# ============================================================================
# StreamingChatCompletionsExecutor — basic construction
# ============================================================================

class TestStreamingExecutorConstruction:
    """StreamingChatCompletionsExecutor should be constructable with deps."""

    def test_executor_creation_with_all_deps(self):
        """Executor should accept client factory, request_config, and callbacks."""
        client_factory = MagicMock()
        close_client_fn = MagicMock()
        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        callbacks = StreamCallbacks(
            stream_delta=MagicMock(),
            reasoning_delta=MagicMock(),
            tool_gen_started=MagicMock(),
            first_delta=MagicMock(),
            activity_touch=MagicMock(),
        )
        executor = StreamingChatCompletionsExecutor(
            client_factory=client_factory,
            close_client_fn=close_client_fn,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )
        assert executor._client_factory is client_factory
        assert executor._close_client_fn is close_client_fn
        assert executor._request_config is rc
        assert executor._base_url == "https://api.openai.com/v1"

    def test_executor_from_agent_factory(self):
        """StreamingChatCompletionsExecutor.from_agent() should create from AIAgent."""
        mock_agent = MagicMock()
        mock_agent.base_url = "http://localhost:11434/v1"
        mock_agent.model = "glm-4:latest"
        mock_agent.provider = "ollama"
        mock_agent.api_mode = "chat_completions"
        mock_agent.max_tokens = 4096
        mock_agent.tools = []
        mock_agent.reasoning_config = None
        mock_agent.request_overrides = None
        mock_agent.providers_allowed = None
        mock_agent.providers_ignored = None
        mock_agent.providers_order = None
        mock_agent.provider_sort = None
        mock_agent.provider_require_parameters = None
        mock_agent.provider_data_collection = None
        mock_agent._is_openrouter_url = MagicMock(return_value=False)

        callbacks = StreamCallbacks()
        rc = RequestConfig(
            model="glm-4:latest",
            base_url="http://localhost:11434/v1",
        )

        # We can't fully test from_agent without a real AIAgent,
        # but we can verify the constructor works
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: MagicMock(),
            close_client_fn=lambda client: None,
            request_config=rc,
            base_url="http://localhost:11434/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=True,
            model="glm-4:latest",
        )
        assert executor._is_local is True


# ============================================================================
# StreamingChatCompletionsExecutor — streaming execution
# ============================================================================

class TestStreamingExecutorStreamIteration:
    """Test that the executor iterates SSE chunks and accumulates correctly."""

    def test_text_only_stream(self):
        """A simple text-only stream should produce content with no tool_calls."""
        # Build mock chunks: "Hello" then " world"
        chunks = [
            make_chunk(make_delta(content="Hello"), model="gpt-4o"),
            make_chunk(make_delta(content=" world"), model="gpt-4o"),
            make_chunk(make_delta(content=None), finish_reason="stop", model="gpt-4o"),
        ]
        # usage in separate final chunk (empty choices)
        usage_chunk = SimpleNamespace(
            choices=[],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks + [usage_chunk])

        captured_deltas = []
        callbacks = StreamCallbacks(
            stream_delta=lambda text: captured_deltas.append(text),
        )

        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        api_kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        assert result.content == "Hello world"
        assert result.tool_calls is None
        assert result.finish_reason == "stop"
        assert result.model_name == "gpt-4o"
        assert captured_deltas == ["Hello", " world"]

    def test_tool_call_stream(self):
        """A stream with tool calls should accumulate them correctly."""
        tc_id = "call_abc123"
        chunks = [
            # First chunk: tool call name
            make_chunk(
                make_delta(
                    tool_calls=[make_tool_call_delta(index=0, id=tc_id, name="read_file", arguments="")]
                ),
                model="gpt-4o",
            ),
            # Second chunk: tool call arguments
            make_chunk(
                make_delta(
                    tool_calls=[make_tool_call_delta(index=0, id="", name="", arguments='{"path": "/tmp/x"}')]
                ),
                model="gpt-4o",
            ),
            # Final chunk
            make_chunk(make_delta(content=None), finish_reason="tool_calls", model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        tool_gen_names = []
        callbacks = StreamCallbacks(
            tool_gen_started=lambda name: tool_gen_names.append(name),
        )

        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        api_kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "read_file"
        assert result.tool_calls[0].function.arguments == '{"path": "/tmp/x"}'
        assert result.finish_reason == "tool_calls"
        assert tool_gen_names == ["read_file"]

    def test_reasoning_stream(self):
        """A stream with reasoning should accumulate and fire reasoning callback."""
        chunks = [
            make_chunk(make_delta(reasoning_content="Let me think"), model="deepseek-r1"),
            make_chunk(make_delta(content="The answer is 42"), model="deepseek-r1"),
            make_chunk(make_delta(content=None), finish_reason="stop", model="deepseek-r1"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        reasoning_deltas = []
        text_deltas = []
        first_fired = [False]
        callbacks = StreamCallbacks(
            stream_delta=lambda t: text_deltas.append(t),
            reasoning_delta=lambda t: reasoning_deltas.append(t),
            first_delta=lambda: setattr(first_fired, '__bool__', lambda: True) or first_fired.__setitem__(0, True),
        )

        rc = RequestConfig(model="deepseek-r1", base_url="https://api.deepseek.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.deepseek.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="deepseek-r1",
        )

        api_kwargs = {"model": "deepseek-r1", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        assert result.content == "The answer is 42"
        assert result.reasoning == "Let me think"
        assert reasoning_deltas == ["Let me think"]
        assert text_deltas == ["The answer is 42"]

    def test_usage_in_final_chunk(self):
        """Usage stats from the final chunk should be captured."""
        chunks = [
            make_chunk(make_delta(content="Hi"), model="gpt-4o"),
            # Final chunk with empty choices but usage
            SimpleNamespace(
                choices=[],
                model="gpt-4o",
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            ),
            make_chunk(make_delta(content=None), finish_reason="stop", model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=StreamCallbacks(),
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        api_kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        assert result.usage is not None
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50

    def test_interrupt_stops_stream(self):
        """If interrupt fires mid-stream, iteration should stop."""
        call_count = [0]

        def interrupt_after_two():
            call_count[0] += 1
            return call_count[0] > 2

        chunks = [
            make_chunk(make_delta(content="Hello"), model="gpt-4o"),
            make_chunk(make_delta(content=" world"), model="gpt-4o"),
            make_chunk(make_delta(content=" more"), model="gpt-4o"),
            make_chunk(make_delta(content=" text"), model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=StreamCallbacks(),
            interrupt_check=interrupt_after_two,
            is_local=False,
            model="gpt-4o",
        )

        api_kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        # Should have stopped after ~2 chunks (got partial)
        assert result.content is not None
        # "Hello" + " world" should be captured before interrupt
        assert "Hello" in result.content

    def test_ollama_tool_call_index_reuse(self):
        """Ollama reuses tool call index 0 for parallel calls — executor should remap."""
        # First tool call at index 0
        tc1_chunks = [
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="call_1", name="tool_a", arguments="")]),
                model="ollama-model",
            ),
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="", name="", arguments='{"a": 1}')]),
                model="ollama-model",
            ),
        ]
        # Second tool call REUSES index 0 but with different id
        tc2_chunks = [
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="call_2", name="tool_b", arguments="")]),
                model="ollama-model",
            ),
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="", name="", arguments='{"b": 2}')]),
                model="ollama-model",
            ),
            make_chunk(make_delta(content=None), finish_reason="tool_calls", model="ollama-model"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(tc1_chunks + tc2_chunks)

        rc = RequestConfig(model="ollama-model", base_url="http://localhost:11434/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="http://localhost:11434/v1",
            callbacks=StreamCallbacks(),
            interrupt_check=lambda: False,
            is_local=True,
            model="ollama-model",
        )

        api_kwargs = {"model": "ollama-model", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        # Should have 2 separate tool calls, not 1 merged
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        names = [tc.function.name for tc in result.tool_calls]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_tool_call_arguments_assignment_not_concat_for_name(self):
        """Tool call names should use assignment (not +=), matching OpenAI SDK pattern.

        MiniMax M2.7 via NVIDIA NIM sends the full function name in every chunk.
        Concatenation would produce 'read_fileread_file'. Assignment is correct.
        """
        chunks = [
            # Two chunks both with the full name
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="call_1", name="read_file", arguments="")]),
                model="minimax-model",
            ),
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="", name="read_file", arguments='{"path": "/x"}')]),
                model="minimax-model",
            ),
            make_chunk(make_delta(content=None), finish_reason="tool_calls", model="minimax-model"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        rc = RequestConfig(model="minimax-model", base_url="https://integrate.api.nvidia.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://integrate.api.nvidia.com/v1",
            callbacks=StreamCallbacks(),
            interrupt_check=lambda: False,
            is_local=False,
            model="minimax-model",
        )

        api_kwargs = {"model": "minimax-model", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        # Should be "read_file", NOT "read_fileread_file"
        assert result.tool_calls[0].function.name == "read_file"

    def test_truncated_tool_args_sets_length_finish(self):
        """If tool call arguments can't be repaired, finish_reason should be 'length'."""
        # Tool call with broken JSON args
        broken_args = '{"path": "/tmp/x'  # truncated JSON
        chunks = [
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="call_1", name="read_file", arguments=broken_args)]),
                model="gpt-4o",
            ),
            make_chunk(make_delta(content=None), finish_reason="length", model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=StreamCallbacks(),
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        api_kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        # Unrepairable args should set finish_reason to "length"
        assert result.finish_reason == "length"

    def test_content_suppressed_during_tool_calls(self):
        """When tool calls are active, content deltas still fire stream_delta
        (for reasoning extraction in the CLI), matching the original behavior
        where tool calls suppress _fire_first_delta but NOT stream_delta_callback.
        Both text and tool content are accumulated in the result.
        """
        captured_text = []

        chunks = [
            # Text content before tool call — fires stream_delta normally
            make_chunk(make_delta(content="I'll help."), model="gpt-4o"),
            # Tool call start — fires tool_gen_started
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="call_1", name="read_file", arguments="")]),
                model="gpt-4o",
            ),
            # More text after tool call started — still fires stream_delta
            # (original code does this too, for reasoning extraction)
            make_chunk(make_delta(content=" checking file"), model="gpt-4o"),
            # Tool args
            make_chunk(
                make_delta(tool_calls=[make_tool_call_delta(index=0, id="", name="", arguments='{"path": "/x"}')]),
                model="gpt-4o",
            ),
            make_chunk(make_delta(content=None), finish_reason="tool_calls", model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        callbacks = StreamCallbacks(
            stream_delta=lambda t: captured_text.append(t),
        )

        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        api_kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        # All content fires stream_delta — matching original _call_chat_completions
        assert "I'll help." in captured_text
        assert " checking file" in captured_text
        # Both content and tool_calls are accumulated
        assert result.content == "I'll help. checking file"
        assert result.tool_calls is not None

    def test_stream_options_included(self):
        """stream=True and stream_options should be included in the API call kwargs."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream([
            make_chunk(make_delta(content="ok"), finish_reason="stop", model="gpt-4o"),
        ])

        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=StreamCallbacks(),
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
            capture_rate_limits_fn=lambda resp: None,
        )

        api_kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        # Verify the call was made
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert "stream_options" in call_kwargs
        assert call_kwargs["stream_options"] == {"include_usage": True}

    def test_local_endpoint_timeout_adjustment(self):
        """Local endpoints should use the base timeout for stream read, not 120s."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream([
            make_chunk(make_delta(content="ok"), finish_reason="stop", model="glm-4"),
        ])

        rc = RequestConfig(
            model="glm-4:latest",
            base_url="http://localhost:11434/v1",
        )
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="http://localhost:11434/v1",
            callbacks=StreamCallbacks(),
            interrupt_check=lambda: False,
            is_local=True,  # Local endpoint
            model="glm-4:latest",
            capture_rate_limits_fn=lambda resp: None,
        )

        api_kwargs = {"model": "glm-4:latest", "messages": [{"role": "user", "content": "hi"}]}
        import httpx
        result = executor.execute_streaming(api_kwargs)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # Local endpoint: stream read timeout should match base timeout (1800s default)
        # not the 120s cloud default
        timeout = call_kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        # For local, read timeout should be >= 1800 (or the base timeout)
        assert timeout.read >= 1800.0


# ============================================================================
# first_delta callback — fires once on first meaningful delta
# ============================================================================

class TestFirstDeltaCallback:
    """Test that StreamCallbacks.first_delta fires exactly once on the first
    meaningful streaming delta (text, reasoning, or tool call name).

    This matches the _fire_first_delta closure in _interruptible_streaming_api_call.
    """

    def test_first_delta_fires_on_text_content(self):
        """first_delta should fire once when the first text content arrives."""
        chunks = [
            make_chunk(make_delta(content="Hello"), model="gpt-4o"),
            make_chunk(make_delta(content=" world"), model="gpt-4o"),
            make_chunk(make_delta(content=None), finish_reason="stop", model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        first_delta_calls = []
        callbacks = StreamCallbacks(
            first_delta=lambda: first_delta_calls.append(True),
            stream_delta=lambda t: None,
        )
        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        executor.execute_streaming({"model": "gpt-4o", "messages": []})
        assert first_delta_calls == [True], f"Expected first_delta to fire once, got {len(first_delta_calls)} calls"

    def test_first_delta_fires_on_reasoning_content(self):
        """first_delta should fire once when reasoning content arrives first."""
        chunks = [
            make_chunk(make_delta(reasoning_content="Hmm..."), model="deepseek-r1"),
            make_chunk(make_delta(content="Answer"), model="deepseek-r1"),
            make_chunk(make_delta(content=None), finish_reason="stop", model="deepseek-r1"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        first_delta_calls = []
        callbacks = StreamCallbacks(
            first_delta=lambda: first_delta_calls.append(True),
            reasoning_delta=lambda t: None,
            stream_delta=lambda t: None,
        )
        rc = RequestConfig(model="deepseek-r1", base_url="https://api.deepseek.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.deepseek.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="deepseek-r1",
        )

        executor.execute_streaming({"model": "deepseek-r1", "messages": []})
        # Should fire on reasoning first, NOT again on text content
        assert len(first_delta_calls) == 1, f"Expected first_delta to fire once (on reasoning), got {len(first_delta_calls)}"

    def test_first_delta_fires_on_tool_call_name(self):
        """first_delta should fire once when a tool call name arrives first."""
        tc_delta = SimpleNamespace(
            index=0,
            id="tc_1",
            type="function",
            function=SimpleNamespace(name="read_file", arguments=None),
        )
        chunks = [
            make_chunk(make_delta(tool_calls=[tc_delta]), model="gpt-4o"),
            make_chunk(make_delta(content=None), finish_reason="tool_calls", model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        first_delta_calls = []
        callbacks = StreamCallbacks(
            first_delta=lambda: first_delta_calls.append(True),
            tool_gen_started=lambda n: None,
        )
        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        executor.execute_streaming({"model": "gpt-4o", "messages": []})
        assert len(first_delta_calls) == 1, f"Expected first_delta to fire once (on tool name), got {len(first_delta_calls)}"

    def test_first_delta_does_not_fire_on_empty_stream(self):
        """first_delta should NOT fire on an empty stream (no content deltas)."""
        chunks = [
            SimpleNamespace(choices=[], model="gpt-4o", usage=None),
            make_chunk(make_delta(content=None), finish_reason="stop", model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        first_delta_calls = []
        callbacks = StreamCallbacks(
            first_delta=lambda: first_delta_calls.append(True),
        )
        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        executor.execute_streaming({"model": "gpt-4o", "messages": []})
        assert first_delta_calls == [], f"Expected no first_delta calls, got {len(first_delta_calls)}"

    def test_first_delta_exception_does_not_break_stream(self):
        """Exceptions in first_delta callback should be swallowed."""
        chunks = [
            make_chunk(make_delta(content="Hello"), model="gpt-4o"),
            make_chunk(make_delta(content=None), finish_reason="stop", model="gpt-4o"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream(chunks)

        def bad_first_delta():
            raise RuntimeError("first_delta callback error")

        callbacks = StreamCallbacks(
            first_delta=bad_first_delta,
            stream_delta=lambda t: None,
        )
        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=rc,
            base_url="https://api.openai.com/v1",
            callbacks=callbacks,
            interrupt_check=lambda: False,
            is_local=False,
            model="gpt-4o",
        )

        # Should NOT raise
        result = executor.execute_streaming({"model": "gpt-4o", "messages": []})
        assert result.content == "Hello"
class TestOnClientCreatedCallback:
    """Verify that on_client_created is called when the executor creates a client."""

    def test_on_client_created_called_with_client(self):
        """on_client_created should receive the per-request OpenAI client."""
        from agent.orchestrator.provider_adapters import (
            StreamingChatCompletionsExecutor,
            StreamCallbacks,
            RequestConfig,
        )
        captured_client = []
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream([])
        callbacks = StreamCallbacks(
            on_client_created=lambda client: captured_client.append(client),
        )
        executor = StreamingChatCompletionsExecutor(
            client_factory=lambda: mock_client,
            close_client_fn=lambda c: None,
            request_config=RequestConfig(model="test-model"),
            base_url="https://api.openai.com",
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        result = executor.execute_streaming({"model": "test-model"})
        assert len(captured_client) == 1
        assert captured_client[0] is mock_client

    def test_request_client_available_during_streaming(self):
        """request_client should be set during streaming and cleared after."""
        from agent.orchestrator.provider_adapters import (
            StreamingChatCompletionsExecutor,
            StreamCallbacks,
            RequestConfig,
        )
        clients_during_stream = []
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MockStream([])
        def client_factory():
            return mock_client
        callbacks = StreamCallbacks(
            on_client_created=lambda client: clients_during_stream.append(client),
        )
        executor = StreamingChatCompletionsExecutor(
            client_factory=client_factory,
            close_client_fn=lambda c: None,
            request_config=RequestConfig(model="test-model"),
            base_url="https://api.openai.com",
            callbacks=callbacks,
            interrupt_check=lambda: False,
        )
        result = executor.execute_streaming({"model": "test-model"})
        # Client should have been captured during streaming
        assert len(clients_during_stream) == 1
        # After streaming completes, request_client is cleared
        assert executor.request_client is None
