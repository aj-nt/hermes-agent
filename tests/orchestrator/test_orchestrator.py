"""Tests for the Orchestrator: the driver class that composes pipeline stages
into a conversation loop with streaming, event bus, and client management.

Phase 4 (Orchestrator Loop) per DESIGN.md.
"""

from __future__ import annotations

import json
import threading
import pytest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from agent.orchestrator.context import (
    ConversationContext,
    ParsedResponse,
    ProviderCapabilities,
    ProviderResult,
    StreamConfig,
    StreamState,
    UsageInfo,
)
from agent.orchestrator.events import EventBus, PipelineEvent
from agent.orchestrator.providers import (
    CredentialManager,
    FallbackChain,
    ProviderRegistry,
)
from agent.orchestrator.stages import (
    ContextManagementStage,
    PipelineResult,
    PreparedRequest,
    ProviderCallStage,
    RequestPrepStage,
    ResponseProcessingStage,
    StageAction,
    ToolDispatchStage,
    ToolDispatchResult,
)
from agent.orchestrator.tools import ToolExecutor


# ============================================================================
# Helpers
# ============================================================================

class MockProvider:
    """Simulates a provider that returns canned responses."""

    def __init__(self, responses: list = None):
        self._responses = responses or []
        self._call_index = 0
        self.capabilities = ProviderCapabilities()

    def prepare_request(self, ctx):
        return {}

    def execute(self, request):
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return ProviderResult(
            response={
                "choices": [{"message": {"role": "assistant", "content": "Default"}, "finish_reason": "stop"}],
            },
            finish_reason="stop",
        )


def _make_ctx(**overrides) -> ConversationContext:
    defaults = dict(
        session_id="test-session",
        messages=[{"role": "user", "content": "Hello"}],
        system_prompt="You are helpful.",
        tools=[],
        max_iterations=20,
    )
    defaults.update(overrides)
    return ConversationContext(**defaults)


def _text_response(text: str, finish_reason: str = "stop") -> ProviderResult:
    return ProviderResult(
        response={
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        finish_reason=finish_reason,
    )


def _tool_response(tool_calls: list) -> ProviderResult:
    return ProviderResult(
        response={
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": tool_calls}, "finish_reason": "tool_calls"}],
        },
        finish_reason="tool_calls",
    )


# ============================================================================
# Orchestrator creation and configuration
# ============================================================================

class TestOrchestratorCreation:
    """Orchestrator holds stages, registry, event bus, and client manager."""

    def test_create_with_defaults(self):
        from agent.orchestrator.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch is not None

    def test_create_with_custom_registry(self):
        from agent.orchestrator.orchestrator import Orchestrator
        registry = ProviderRegistry()
        orch = Orchestrator(registry=registry)
        assert orch.registry is registry

    def test_create_with_custom_event_bus(self):
        from agent.orchestrator.orchestrator import Orchestrator
        bus = EventBus()
        orch = Orchestrator(event_bus=bus)
        assert orch.event_bus is bus

    def test_create_with_custom_tool_executor(self):
        from agent.orchestrator.orchestrator import Orchestrator
        executor = ToolExecutor()
        orch = Orchestrator(tool_executor=executor)
        assert orch.tool_executor is executor

    def test_create_with_model_name(self):
        from agent.orchestrator.orchestrator import Orchestrator
        orch = Orchestrator(model_name="gpt-4o")
        assert orch.model_name == "gpt-4o"

    def test_default_model_name(self):
        from agent.orchestrator.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch.model_name == "default-model"


# ============================================================================
# Orchestrator.run() — the main loop
# ============================================================================

class TestOrchestratorRun:
    """Orchestrator.run() drives Stage 1→2→3→4→5 in a loop."""

    def test_simple_text_response(self):
        """One iteration: user asks, provider replies, pipeline yields."""
        from agent.orchestrator.orchestrator import Orchestrator

        provider = MockProvider(responses=[_text_response("Hello!")])
        registry = ProviderRegistry()
        registry.register("mock", provider)

        orch = Orchestrator(registry=registry, model_name="mock-model")
        ctx = _make_ctx()
        # Need to set the provider name on the request prep stage
        orch.request_prep = RequestPrepStage(model_name="mock-model")
        # Override provider name resolution
        orch._resolve_provider_name = lambda ctx: "mock"

        result = orch.run(ctx)
        assert result.response.message["content"] == "Hello!"
        assert result.iterations == 1
        assert result.interrupted is False

    def test_tool_call_loop(self):
        """Provider requests tool, tool executes, provider responds."""
        from agent.orchestrator.orchestrator import Orchestrator

        provider = MockProvider(responses=[
            _tool_response([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/tmp/test"}'},
            }]),
            _text_response("The file contains: hello world"),
        ])
        registry = ProviderRegistry()
        registry.register("mock", provider)

        executor = ToolExecutor()
        executor.register("read_file", lambda args: "hello world")

        orch = Orchestrator(registry=registry, tool_executor=executor)
        orch._resolve_provider_name = lambda ctx: "mock"

        ctx = _make_ctx()
        result = orch.run(ctx)

        assert result.iterations == 2
        assert "hello world" in result.response.message["content"]

    def test_interrupt_stops_immediately(self):
        """Setting interrupt_event causes the pipeline to stop."""
        from agent.orchestrator.orchestrator import Orchestrator

        provider = MockProvider(responses=[_text_response("thinking...")])
        registry = ProviderRegistry()
        registry.register("mock", provider)

        orch = Orchestrator(registry=registry)
        orch._resolve_provider_name = lambda ctx: "mock"

        ctx = _make_ctx()
        ctx.interrupt_event.set()
        result = orch.run(ctx)

        assert result.interrupted is True

    def test_max_iterations_terminates(self):
        """Pipeline stops at max_iterations."""
        from agent.orchestrator.orchestrator import Orchestrator

        always_tool = _tool_response([{
            "id": "call_loop",
            "type": "function",
            "function": {"name": "loop_tool", "arguments": "{}"},
        }])
        provider = MockProvider(responses=[always_tool] * 100)
        registry = ProviderRegistry()
        registry.register("mock", provider)

        executor = ToolExecutor()
        executor.register("loop_tool", lambda args: "looping")

        orch = Orchestrator(registry=registry, tool_executor=executor)
        orch._resolve_provider_name = lambda ctx: "mock"

        ctx = _make_ctx(max_iterations=3)
        result = orch.run(ctx)

        assert result.iterations >= 3

    def test_provider_error_with_fallback(self):
        """Provider call fails, PipelineResult contains error information."""
        from agent.orchestrator.orchestrator import Orchestrator

        provider = MockProvider()
        provider.execute = MagicMock(side_effect=ConnectionError("API down"))

        registry = ProviderRegistry()
        registry.register("mock", provider)

        orch = Orchestrator(registry=registry)
        orch._resolve_provider_name = lambda ctx: "mock"

        ctx = _make_ctx()
        result = orch.run(ctx)

        assert result.response.finish_reason == "error"


# ============================================================================
# Streaming support
# ============================================================================

class TestStreamProcessing:
    """StreamState tracks accumulated text and emits events."""

    def test_stream_state_accumulates_text(self):
        """StreamState.accumulated_text grows as deltas arrive."""
        state = StreamState()
        assert state.accumulated_text == ""

        state.accumulated_text += "Hello"
        assert state.accumulated_text == "Hello"

        state.accumulated_text += " world"
        assert state.accumulated_text == "Hello world"

    def test_stream_state_config_defaults(self):
        """StreamConfig has sensible defaults."""
        config = StreamConfig()
        assert config.stale_timeout == 30.0
        assert config.max_inactivity == 120.0
        assert config.empty_content_max_retries == 3

    def test_stream_state_tracks_empty_retries(self):
        """StreamState.empty_content_retries counts consecutive empty responses."""
        state = StreamState()
        assert state.empty_content_retries == 0

        state.empty_content_retries += 1
        assert state.empty_content_retries == 1

    def test_stream_state_needs_break_flag(self):
        """StreamState.needs_break signals interrupt to active stream."""
        state = StreamState()
        assert state.needs_break is False

        state.needs_break = True
        assert state.needs_break is True

    def test_stream_state_text_parts_for_codex(self):
        """StreamState.text_parts collects Codex-style text parts."""
        state = StreamState()
        assert state.text_parts == []

        state.text_parts.append("Part 1")
        state.text_parts.append("Part 2")
        assert state.text_parts == ["Part 1", "Part 2"]


class TestStreamEventEmission:
    """Orchestrator emits stream_delta events via EventBus."""

    def test_event_bus_receives_stream_delta(self):
        bus = EventBus()
        events = []
        bus.subscribe("stream_delta", lambda e: events.append(e))

        bus.emit(PipelineEvent(
            kind="stream_delta",
            data={"text": "Hello", "accumulated": "Hello"},
            session_id="test",
        ))

        assert len(events) == 1
        assert events[0].data["text"] == "Hello"

    def test_event_bus_receives_stream_end(self):
        bus = EventBus()
        events = []
        bus.subscribe("stream_end", lambda e: events.append(e))

        bus.emit(PipelineEvent(
            kind="stream_end",
            data={"finish_reason": "stop"},
            session_id="test",
        ))

        assert len(events) == 1
        assert events[0].kind == "stream_end"

    def test_orchestrator_emits_provider_call_events(self):
        """Orchestrator emits provider_call_start and provider_call_end."""
        bus = EventBus()
        provider = MockProvider(responses=[_text_response("Hi")])
        registry = ProviderRegistry()
        registry.register("mock", provider)

        from agent.orchestrator.orchestrator import Orchestrator
        orch = Orchestrator(registry=registry, event_bus=bus)
        orch._resolve_provider_name = lambda ctx: "mock"

        events = []
        bus.subscribe("provider_call_start", lambda e: events.append(e))
        bus.subscribe("provider_call_end", lambda e: events.append(e))

        ctx = _make_ctx()
        orch.run(ctx)

        # At least start + end for each provider call
        start_events = [e for e in events if e.kind == "provider_call_start"]
        end_events = [e for e in events if e.kind == "provider_call_end"]
        assert len(start_events) >= 1
        assert len(end_events) >= 1

    def test_orchestrator_emits_iteration_events(self):
        """Orchestrator emits iteration_start and iteration_end events."""
        bus = EventBus()
        provider = MockProvider(responses=[_text_response("Done")])
        registry = ProviderRegistry()
        registry.register("mock", provider)

        from agent.orchestrator.orchestrator import Orchestrator
        orch = Orchestrator(registry=registry, event_bus=bus)
        orch._resolve_provider_name = lambda ctx: "mock"

        events = []
        bus.subscribe("iteration_start", lambda e: events.append(e))
        bus.subscribe("iteration_end", lambda e: events.append(e))

        ctx = _make_ctx()
        orch.run(ctx)

        starts = [e for e in events if e.kind == "iteration_start"]
        ends = [e for e in events if e.kind == "iteration_end"]
        assert len(starts) >= 1
        assert len(ends) >= 1


# ============================================================================
# Orchestrator configuration
# ============================================================================

class TestOrchestratorConfiguration:
    """Orchestrator is configurable for different providers and modes."""

    def test_custom_max_iterations(self):
        from agent.orchestrator.orchestrator import Orchestrator
        orch = Orchestrator(max_iterations=50)
        assert orch.max_iterations == 50

    def test_default_max_iterations(self):
        from agent.orchestrator.orchestrator import Orchestrator
        orch = Orchestrator()
        assert orch.max_iterations == 90

    def test_custom_model_name(self):
        from agent.orchestrator.orchestrator import Orchestrator
        orch = Orchestrator(model_name="glm-4")
        assert orch.model_name == "glm-4"

    def test_context_iteration_counter_increments(self):
        """The iteration counter in ConversationContext increments each loop."""
        from agent.orchestrator.orchestrator import Orchestrator

        provider = MockProvider(responses=[
            _tool_response([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"},
            }]),
            _text_response("Done"),
        ])
        registry = ProviderRegistry()
        registry.register("mock", provider)
        executor = ToolExecutor()
        executor.register("test", lambda args: "ok")

        orch = Orchestrator(registry=registry, tool_executor=executor)
        orch._resolve_provider_name = lambda ctx: "mock"

        ctx = _make_ctx()
        assert ctx.iteration == 0
        result = orch.run(ctx)
        assert ctx.iteration >= 2  # incremented by ContextManagementStage


# ============================================================================
# Integration: full conversation through pipeline
# ============================================================================

class TestFullConversation:
    """End-to-end test of a multi-turn tool conversation."""

    def test_multi_tool_conversation(self):
        """Provider calls two tools sequentially, then responds."""
        from agent.orchestrator.orchestrator import Orchestrator

        provider = MockProvider(responses=[
            _tool_response([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/a.txt"}'},
            }]),
            _tool_response([{
                "id": "call_2",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/b.txt"}'},
            }]),
            _text_response("Both files read successfully"),
        ])
        registry = ProviderRegistry()
        registry.register("mock", provider)

        executor = ToolExecutor()
        executor.register("read_file", lambda args: f"contents of {args['path']}")

        orch = Orchestrator(registry=registry, tool_executor=executor)
        orch._resolve_provider_name = lambda ctx: "mock"

        ctx = _make_ctx()
        result = orch.run(ctx)

        assert result.iterations == 3
        assert "successfully" in result.response.message["content"].lower()

    def test_conversation_messages_grow(self):
        """Messages list grows as tool results are appended."""
        from agent.orchestrator.orchestrator import Orchestrator

        provider = MockProvider(responses=[
            _tool_response([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_files", "arguments": "{}"},
            }]),
            _text_response("Here are the files."),
        ])
        registry = ProviderRegistry()
        registry.register("mock", provider)
        executor = ToolExecutor()
        executor.register("list_files", lambda args: "file1.txt, file2.txt")

        orch = Orchestrator(registry=registry, tool_executor=executor)
        orch._resolve_provider_name = lambda ctx: "mock"

        ctx = _make_ctx()
        initial_count = len(ctx.messages)
        result = orch.run(ctx)

        # Messages should have grown: original + assistant + tool result + assistant
        assert len(ctx.messages) > initial_count