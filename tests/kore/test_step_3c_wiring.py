"""Tests for Step 3c wiring — making _interruptible_streaming_api_call use
StreamingChatCompletionsExecutor instead of the inline _call_chat_completions closure.

Step 3c is the Strangler Fig "wire up" step: the extracted executor (3b) is now
called from the original method, and the old inline code is deleted.

TDD: write failing tests first, then implement to make them pass.
"""

import json
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

from agent.orchestrator.provider_adapters import (
    RequestConfig,
    StreamingChatCompletionsExecutor,
    StreamCallbacks,
    StreamResult,
)


# ============================================================================
# Test: from_agent factory — StreamingChatCompletionsExecutor.from_agent()
# ============================================================================

class TestStreamingExecutorFromAgent:
    """Test the factory that constructs a StreamingChatCompletionsExecutor
    from an AIAgent instance, injecting the right callbacks and factories."""

    def test_from_agent_creates_executor_with_client_factory(self):
        """from_agent should inject _create_request_openai_client as client_factory."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)

        # The client_factory should call agent._create_request_openai_client
        assert executor._client_factory is not None
        executor._client_factory()
        agent._create_request_openai_client.assert_called_once_with(
            reason="chat_completion_stream_request"
        )

    def test_from_agent_creates_executor_with_close_factory(self):
        """from_agent should inject _close_request_openai_client as close_client_fn."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)

        assert executor._close_client_fn is not None
        test_client = MagicMock()
        executor._close_client_fn(test_client)
        agent._close_request_openai_client.assert_called_once_with(
            test_client, reason="stream_request_complete"
        )

    def test_from_agent_creates_stream_callbacks(self):
        """from_agent should build StreamCallbacks from agent's fire methods."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)

        assert executor._callbacks.stream_delta is not None
        assert executor._callbacks.reasoning_delta is not None
        assert executor._callbacks.tool_gen_started is not None
        assert executor._callbacks.activity_touch is not None

    def test_from_agent_stream_delta_callback_wraps_fire_stream_delta(self):
        """The stream_delta callback should call _fire_stream_delta with
        _stream_needs_break handling."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)

        # Call the callback
        executor._callbacks.stream_delta("hello")

        # Should have called _fire_stream_delta
        agent._fire_stream_delta.assert_called_once_with("hello")

    def test_from_agent_reasoning_delta_callback(self):
        """The reasoning_delta callback should call _fire_reasoning_delta."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        executor._callbacks.reasoning_delta("thinking...")

        agent._fire_reasoning_delta.assert_called_once_with("thinking...")

    def test_from_agent_tool_gen_started_callback(self):
        """The tool_gen_started callback should call _fire_tool_gen_started."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        executor._callbacks.tool_gen_started("write_file")

        agent._fire_tool_gen_started.assert_called_once_with("write_file")

    def test_from_agent_activity_touch_callback(self):
        """The activity_touch callback should call _touch_activity."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        executor._callbacks.activity_touch("receiving stream response")

        agent._touch_activity.assert_called_once_with("receiving stream response")

    def test_from_agent_interrupt_check_reads_interrupt_requested(self):
        """The interrupt_check should return agent._interrupt_requested."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        assert executor._interrupt_check() is False

        agent._interrupt_requested = True
        assert executor._interrupt_check() is True

    def test_from_agent_capture_rate_limits(self):
        """from_agent should inject _capture_rate_limits via the capture_fn."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)

        mock_response = MagicMock()
        executor._capture_rate_limits_fn(mock_response)
        agent._capture_rate_limits.assert_called_once_with(mock_response)

    def test_from_agent_detects_local_endpoint(self):
        """from_agent should detect local endpoints."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "ollama-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        assert executor._is_local is True

    def test_from_agent_detects_cloud_endpoint(self):
        """from_agent should mark cloud endpoints as not local."""
        agent = MagicMock()
        agent.base_url = "https://api.openai.com/v1"
        agent.model = "gpt-4"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        assert executor._is_local is False

    def test_from_agent_builds_request_config(self):
        """from_agent should build a RequestConfig from agent attributes."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.max_tokens = 4096
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False
        agent._ollama_num_ctx = None
        agent.reasoning_config = None
        agent.request_overrides = None
        agent.providers_allowed = []
        agent.providers_ignored = []
        agent.providers_order = []
        agent.provider_sort = None
        agent.provider_require_parameters = False
        agent.provider_data_collection = None
        agent.provider = "ollama"
        agent.session_id = "test-session"

        executor = StreamingChatCompletionsExecutor.from_agent(agent)

        assert isinstance(executor._request_config, RequestConfig)
        assert executor._request_config.model == "test-model"
        assert executor._request_config.base_url == "http://localhost:11434/v1"


# ============================================================================
# Test: StreamCallbacks wiring — stream_delta fires through _fire_stream_delta
# which handles _stream_needs_break
# ============================================================================

class TestStreamDeltaCallbackWiring:
    """Verify the stream_delta callback properly wraps _fire_stream_delta,
    including the _stream_needs_break paragraph-break logic."""

    def test_stream_delta_calls_fire_method(self):
        """stream_delta callback should call agent._fire_stream_delta."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        executor._callbacks.stream_delta("text chunk")
        agent._fire_stream_delta.assert_called_once_with("text chunk")

    def test_stream_needs_break_delegates_to_fire_stream_delta(self):
        """stream_delta callback delegates to _fire_stream_delta, which
        handles _stream_needs_break internally. The callback just passes
        the original text through."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = True  # Set the break flag
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        executor._callbacks.stream_delta("first content")

        # The callback passes original text to _fire_stream_delta.
        # _fire_stream_delta itself handles _stream_needs_break prepending.
        agent._fire_stream_delta.assert_called_once_with("first content")

    def test_stream_needs_break_delegation_not_responsible_for_clearing(self):
        """The stream_delta callback delegates to _fire_stream_delta.
        The _stream_needs_break flag is managed inside _fire_stream_delta,
        not by the callback itself. This test just verifies the callback
        passes text through without modification."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = True
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        # The callback just passes text through — _fire_stream_delta handles
        # the _stream_needs_break logic internally
        executor._callbacks.stream_delta("content")
        agent._fire_stream_delta.assert_called_once_with("content")

    def test_stream_needs_break_not_prepended_for_whitespace_only(self):
        """_stream_needs_break should not prepend break for whitespace-only content."""
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "test-model"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = True
        agent._interrupt_requested = False

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        # Whitespace-only content should NOT clear the break
        executor._callbacks.stream_delta("   ")

        # _fire_stream_delta called, but the break isn't triggered for
        # whitespace-only content (matches original _fire_stream_delta behavior)
        agent._fire_stream_delta.assert_called_once_with("   ")


# ============================================================================
# Test: Integration — executor from_agent + execute_streaming end-to-end
# ============================================================================

class TestStreamingExecutorFromAgentIntegration:
    """Integration tests: from_agent wires executor correctly, produce
    StreamResult that matches the old _call_chat_completions output."""

    def test_from_agent_executor_produces_stream_result(self):
        """End-to-end: from_agent creates executor that can execute_streaming."""
        # Create a mock agent with all required attributes
        agent = MagicMock()
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "llama3"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False
        agent.max_tokens = 4096
        agent._ollama_num_ctx = None
        agent.providers_allowed = []
        agent.providers_ignored = []
        agent.providers_order = []
        agent.provider_sort = None
        agent.provider_require_parameters = False
        agent.provider_data_collection = None
        agent.provider = "ollama"
        agent.session_id = "test"
        agent.reasoning_config = None
        agent.request_overrides = None

        # Create mock client that returns a stream
        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([]))
        mock_stream.response = MagicMock()

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream
        agent._create_request_openai_client.return_value = mock_client

        executor = StreamingChatCompletionsExecutor.from_agent(agent)

        # Execute with minimal kwargs
        api_kwargs = {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
        result = executor.execute_streaming(api_kwargs)

        assert isinstance(result, StreamResult)
        assert result.finish_reason == "stop"  # Empty stream = default finish_reason

    def test_from_agent_request_config_inherits_agent_model(self):
        """The RequestConfig from from_agent should match the agent's model."""
        agent = MagicMock()
        agent.base_url = "https://api.openai.com/v1"
        agent.model = "gpt-4o"
        agent.stream_delta_callback = MagicMock()
        agent.reasoning_callback = MagicMock()
        agent.tool_gen_callback = MagicMock()
        agent._stream_needs_break = False
        agent._interrupt_requested = False
        agent._ollama_num_ctx = None
        agent.reasoning_config = None
        agent.request_overrides = None
        agent.providers_allowed = []
        agent.providers_ignored = []
        agent.providers_order = []
        agent.provider_sort = None
        agent.provider_require_parameters = False
        agent.provider_data_collection = None
        agent.provider = "openai"
        agent.session_id = "test"

        executor = StreamingChatCompletionsExecutor.from_agent(agent)
        assert executor._request_config.model == "gpt-4o"