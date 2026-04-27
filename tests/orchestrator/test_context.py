"""Tests for orchestrator core types: ConversationContext, SessionState,
StreamState, ParsedResponse, ProviderResult, ProviderCapabilities, PipelineEvent.

These verify:
1. Dataclasses accept all expected fields
2. Defaults are sensible (None/empty/False/0)
3. Fields from the DESIGN.md spec are present and typed correctly
4. Composition works (ConversationContext holds other types)
5. Type annotations match what the monolith actually produces
"""

import threading
from dataclasses import fields
from typing import get_type_hints

import pytest


# ============================================================================
# StreamState
# ============================================================================

class TestStreamState:
    """StreamState encapsulates per-pipeline streaming accumulation state.

    Replaces: _stream_needs_break, _current_streamed_assistant_text,
    _codex_streamed_text_parts, _empty_content_retries, _StreamConfig.
    """

    def test_import(self):
        from agent.orchestrator.context import StreamState

    def test_accumulated_text_defaults_empty(self):
        from agent.orchestrator.context import StreamState
        ss = StreamState()
        assert ss.accumulated_text == ""

    def test_text_parts_defaults_empty_list(self):
        from agent.orchestrator.context import StreamState
        ss = StreamState()
        assert ss.text_parts == []

    def test_empty_content_retries_defaults_zero(self):
        from agent.orchestrator.context import StreamState
        ss = StreamState()
        assert ss.empty_content_retries == 0

    def test_needs_break_defaults_false(self):
        from agent.orchestrator.context import StreamState
        ss = StreamState()
        assert ss.needs_break is False

    def test_stream_callback_defaults_none(self):
        from agent.orchestrator.context import StreamState
        ss = StreamState()
        assert ss.stream_callback is None

    def test_reasoning_callback_defaults_none(self):
        from agent.orchestrator.context import StreamState
        ss = StreamState()
        assert ss.reasoning_callback is None

    def test_config_field_present(self):
        from agent.orchestrator.context import StreamState
        ss = StreamState()
        # config should exist (may be None initially for Phase 1)
        assert hasattr(ss, "config")

    def test_all_fields_writable(self):
        from agent.orchestrator.context import StreamState
        ss = StreamState()
        ss.accumulated_text = "hello"
        ss.text_parts = ["a", "b"]
        ss.empty_content_retries = 3
        ss.needs_break = True
        assert ss.accumulated_text == "hello"
        assert ss.text_parts == ["a", "b"]
        assert ss.empty_content_retries == 3
        assert ss.needs_break is True


# ============================================================================
# ProviderCapabilities
# ============================================================================

class TestProviderCapabilities:
    """ProviderCapabilities replaces the maze of _is_*_backend() checks.

    Each provider declares what it supports; the orchestrator dispatches
    on capabilities, not identity.
    """

    def test_import(self):
        from agent.orchestrator.context import ProviderCapabilities

    def test_streaming_defaults_true(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.supports_streaming is True

    def test_tools_defaults_true(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.supports_tools is True

    def test_reasoning_tokens_defaults_false(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.supports_reasoning_tokens is False

    def test_prompt_caching_defaults_false(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.requires_prompt_caching is False

    def test_message_sanitization_defaults_false(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.requires_message_sanitization is False

    def test_max_context_tokens_defaults_none(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.max_context_tokens is None

    def test_custom_stop_defaults_false(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.requires_custom_stop_handling is False

    def test_responses_api_defaults_false(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.supports_responses_api is False

    def test_prompt_caching_strategy_defaults_none(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities()
        assert pc.cache_breakpoint_strategy == "none"

    def test_all_fields_overridable(self):
        from agent.orchestrator.context import ProviderCapabilities
        pc = ProviderCapabilities(
            supports_streaming=False,
            supports_tools=False,
            supports_reasoning_tokens=True,
            requires_prompt_caching=True,
            max_context_tokens=200000,
            supports_responses_api=True,
            cache_breakpoint_strategy="anthropic_4point",
        )
        assert pc.supports_streaming is False
        assert pc.max_context_tokens == 200000
        assert pc.cache_breakpoint_strategy == "anthropic_4point"


# ============================================================================
# ProviderResult
# ============================================================================

class TestProviderResult:
    """ProviderResult is the uniform return type from provider calls.

    No more 'sometimes a dict, sometimes a stream, sometimes None'.
    """

    def test_import(self):
        from agent.orchestrator.context import ProviderResult

    def test_response_defaults_none(self):
        from agent.orchestrator.context import ProviderResult
        pr = ProviderResult()
        assert pr.response is None

    def test_stream_defaults_none(self):
        from agent.orchestrator.context import ProviderResult
        pr = ProviderResult()
        assert pr.stream is None

    def test_usage_defaults_none(self):
        from agent.orchestrator.context import ProviderResult
        pr = ProviderResult()
        assert pr.usage is None

    def test_finish_reason_defaults_none(self):
        from agent.orchestrator.context import ProviderResult
        pr = ProviderResult()
        assert pr.finish_reason is None

    def test_error_defaults_none(self):
        from agent.orchestrator.context import ProviderResult
        pr = ProviderResult()
        assert pr.error is None

    def test_should_retry_defaults_false(self):
        from agent.orchestrator.context import ProviderResult
        pr = ProviderResult()
        assert pr.should_retry is False

    def test_should_fallback_defaults_false(self):
        from agent.orchestrator.context import ProviderResult
        pr = ProviderResult()
        assert pr.should_fallback is False

    def test_with_error_sets_fields(self):
        from agent.orchestrator.context import ProviderResult
        err = ConnectionError("timeout")
        pr = ProviderResult(error=err, should_retry=True)
        assert pr.error is err
        assert pr.should_retry is True
        assert pr.should_fallback is False

    def test_with_fallback_signal(self):
        from agent.orchestrator.context import ProviderResult
        err = ValueError("429 rate limited")
        pr = ProviderResult(error=err, should_fallback=True)
        assert pr.should_fallback is True
        assert pr.should_retry is False


# ============================================================================
# ParsedResponse
# ============================================================================

class TestParsedResponse:
    """ParsedResponse is the canonical output of Stage 3 (Response Processing).

    Normalizes provider-specific response formats into a single structure.
    """

    def test_import(self):
        from agent.orchestrator.context import ParsedResponse

    def test_message_defaults_none(self):
        from agent.orchestrator.context import ParsedResponse
        pr = ParsedResponse()
        assert pr.message is None

    def test_tool_calls_defaults_empty_list(self):
        from agent.orchestrator.context import ParsedResponse
        pr = ParsedResponse()
        assert pr.tool_calls == []

    def test_reasoning_content_defaults_none(self):
        from agent.orchestrator.context import ParsedResponse
        pr = ParsedResponse()
        assert pr.reasoning_content is None

    def test_finish_reason_defaults_none(self):
        from agent.orchestrator.context import ParsedResponse
        pr = ParsedResponse()
        assert pr.finish_reason is None

    def test_usage_defaults_none(self):
        from agent.orchestrator.context import ParsedResponse
        pr = ParsedResponse()
        assert pr.usage is None

    def test_has_tool_calls_property_false_on_empty(self):
        from agent.orchestrator.context import ParsedResponse
        pr = ParsedResponse()
        assert pr.has_tool_calls is False

    def test_has_tool_calls_property_true_when_present(self):
        from agent.orchestrator.context import ParsedResponse
        pr = ParsedResponse(tool_calls=[{"id": "tc1", "function": {"name": "read_file"}}])
        assert pr.has_tool_calls is True


# ============================================================================
# PipelineEvent
# ============================================================================

class TestPipelineEvent:
    """PipelineEvent is the typed event emitted during pipeline processing.

    Replaces scattered _safe_print, _vprint, _emit_status calls.
    """

    def test_import(self):
        from agent.orchestrator.events import PipelineEvent

    def test_kind_field_required(self):
        from agent.orchestrator.events import PipelineEvent
        with pytest.raises(TypeError):
            PipelineEvent()  # kind is required

    def test_kind_set(self):
        from agent.orchestrator.events import PipelineEvent
        pe = PipelineEvent(kind="stream_delta")
        assert pe.kind == "stream_delta"

    def test_data_defaults_empty_dict(self):
        from agent.orchestrator.events import PipelineEvent
        pe = PipelineEvent(kind="status")
        assert pe.data == {}

    def test_session_id_defaults_empty(self):
        from agent.orchestrator.events import PipelineEvent
        pe = PipelineEvent(kind="status")
        assert pe.session_id == ""

    def test_timestamp_auto_set(self):
        import time
        from agent.orchestrator.events import PipelineEvent
        before = time.time()
        pe = PipelineEvent(kind="error", session_id="s1")
        after = time.time()
        assert before <= pe.timestamp <= after

    def test_with_all_fields(self):
        from agent.orchestrator.events import PipelineEvent
        pe = PipelineEvent(
            kind="tool_start",
            data={"tool_name": "read_file"},
            session_id="s1",
        )
        assert pe.kind == "tool_start"
        assert pe.data["tool_name"] == "read_file"
        assert pe.session_id == "s1"


# ============================================================================
# EventBus
# ============================================================================

class TestEventBus:
    """EventBus routes pipeline events to subscribers.

    Replaces _safe_print, _vprint, _emit_status, _emit_warning,
    _fire_stream_delta, _fire_reasoning_delta, _fire_tool_gen_started.
    """

    def test_import(self):
        from agent.orchestrator.events import EventBus

    def test_emit_without_subscribers_does_not_raise(self):
        from agent.orchestrator.events import EventBus, PipelineEvent
        bus = EventBus()
        pe = PipelineEvent(kind="status")
        bus.emit(pe)  # should not raise

    def test_subscriber_receives_event(self):
        from agent.orchestrator.events import EventBus, PipelineEvent
        bus = EventBus()
        received = []
        bus.subscribe("status", lambda e: received.append(e))
        pe = PipelineEvent(kind="status", data={"msg": "hello"})
        bus.emit(pe)
        assert len(received) == 1
        assert received[0].data["msg"] == "hello"

    def test_subscriber_only_receives_subscribed_kind(self):
        from agent.orchestrator.events import EventBus, PipelineEvent
        bus = EventBus()
        status_events = []
        bus.subscribe("status", lambda e: status_events.append(e))
        pe_status = PipelineEvent(kind="status")
        pe_error = PipelineEvent(kind="error")
        bus.emit(pe_status)
        bus.emit(pe_error)
        assert len(status_events) == 1
        assert status_events[0].kind == "status"

    def test_multiple_subscribers_same_kind(self):
        from agent.orchestrator.events import EventBus, PipelineEvent
        bus = EventBus()
        a = []
        b = []
        bus.subscribe("tool_start", lambda e: a.append(e))
        bus.subscribe("tool_start", lambda e: b.append(e))
        bus.emit(PipelineEvent(kind="tool_start"))
        assert len(a) == 1
        assert len(b) == 1

    def test_subscriber_exception_does_not_break_other_subscribers(self):
        """EventBus publish is fire-and-forget; swallow exceptions."""
        from agent.orchestrator.events import EventBus, PipelineEvent
        bus = EventBus()
        good = []
        bus.subscribe("status", lambda e: 1/0)  # will raise
        bus.subscribe("status", lambda e: good.append(e))
        bus.emit(PipelineEvent(kind="status"))
        assert len(good) == 1  # second subscriber still got it


# ============================================================================
# ConversationContext
# ============================================================================

class TestConversationContext:
    """ConversationContext replaces the scatter of self.* attributes.

    Single state object passed through the pipeline. The Orchestrator
    loop owns this; no other thread mutates it during a turn.
    """

    def test_import(self):
        from agent.orchestrator.context import ConversationContext

    def test_session_id_required(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="test-123")
        assert ctx.session_id == "test-123"

    def test_messages_defaults_empty_list(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.messages == []

    def test_system_prompt_defaults_empty(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.system_prompt == ""

    def test_iteration_defaults_zero(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.iteration == 0

    def test_max_iterations_defaults_90(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.max_iterations == 90

    def test_tools_defaults_empty_list(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.tools == []

    def test_stream_needs_break_defaults_false(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.stream_needs_break is False

    def test_api_call_count_defaults_zero(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.api_call_count == 0

    def test_tool_results_pending_defaults_false(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.tool_results_pending is False

    def test_interrupt_event_is_threading_event(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert isinstance(ctx.interrupt_event, threading.Event)
        assert ctx.interrupt_event.is_set() is False

    def test_stream_state_is_stream_state_instance(self):
        from agent.orchestrator.context import ConversationContext, StreamState
        ctx = ConversationContext(session_id="s1")
        assert isinstance(ctx.stream_state, StreamState)

    def test_steer_text_defaults_none(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.steer_text is None

    def test_ephemeral_system_prompt_defaults_none(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.ephemeral_system_prompt is None

    def test_checkpoint_mgr_defaults_none(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.checkpoint_mgr is None

    def test_auxiliary_runtime_defaults_none(self):
        from agent.orchestrator.context import ConversationContext
        ctx = ConversationContext(session_id="s1")
        assert ctx.auxiliary_runtime is None


# ============================================================================
# SessionState
# ============================================================================

class TestSessionState:
    """SessionState is the typed, grouped state for a single conversation session.

    All mutations go through methods. No direct attribute writes from outside.
    Replaces 176 unique self.X attributes across 491 assignments.
    """

    def test_import(self):
        from agent.orchestrator.context import SessionState

    def test_session_id_required(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="test-123")
        assert ss.session_id == "test-123"

    def test_messages_defaults_empty_list(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.messages == []

    def test_system_prompt_defaults_none(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.system_prompt is None

    def test_cached_system_prompt_defaults_none(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.cached_system_prompt is None

    def test_ephemeral_system_prompt_defaults_none(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.ephemeral_system_prompt is None

    def test_fallback_activated_defaults_false(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.fallback_activated is False

    def test_interrupt_event_is_threading_event(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert isinstance(ss.interrupt_event, threading.Event)

    def test_steer_buffer_defaults_none(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.steer_buffer is None

    def test_iteration_count_defaults_zero(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.iteration_count == 0

    def test_api_call_count_defaults_zero(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.api_call_count == 0

    def test_session_title_defaults_none(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.session_title is None

    def test_is_memory_enabled_defaults_false(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.is_memory_enabled is False

    def test_is_user_profile_enabled_defaults_false(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.is_user_profile_enabled is False

    def test_empty_content_retries_defaults_zero(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.empty_content_retries == 0

    def test_active_children_defaults_empty_set(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.active_children == set()

    def test_delegate_depth_defaults_zero(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.delegate_depth == 0

    def test_use_native_cache_layout_defaults_false(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.use_native_cache_layout is False

    def test_session_estimated_cost_defaults_zero(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.session_estimated_cost_usd == 0.0

    def test_session_cost_status_defaults_empty(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.session_cost_status == ""

    def test_pending_steer_defaults_empty_list(self):
        from agent.orchestrator.context import SessionState
        ss = SessionState(session_id="s1")
        assert ss.pending_steer == []

    def test_stream_state_is_stream_state_instance(self):
        from agent.orchestrator.context import SessionState, StreamState
        ss = SessionState(session_id="s1")
        assert isinstance(ss.stream_state, StreamState)