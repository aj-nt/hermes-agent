"""Tests for pipeline stages: Request Preparation, Provider Call,
Response Processing, Tool Dispatch, and Context Management.

Each stage is tested in isolation with a mock ConversationContext.
"""

from __future__ import annotations

import pytest
import threading
from dataclasses import dataclass, field
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
    FailoverReason,
    FallbackChain,
    ProviderRegistry,
)
from agent.orchestrator.provider_adapters import (
    OpenAICompatibleProvider,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_ctx(**overrides) -> ConversationContext:
    """Build a minimal ConversationContext for testing."""
    defaults = dict(
        session_id="test-session",
        messages=[{"role": "user", "content": "Hello"}],
        system_prompt="You are a helpful assistant.",
        tools=[],
    )
    defaults.update(overrides)
    return ConversationContext(**defaults)


def _make_parsed_response(
    content: str = "Hi there",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
) -> ParsedResponse:
    """Build a minimal ParsedResponse for testing."""
    return ParsedResponse(
        message={"role": "assistant", "content": content},
        tool_calls=tool_calls or [],
        reasoning_content=None,
        finish_reason=finish_reason,
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_provider_result(
    response: dict | None = None,
    finish_reason: str = "stop",
    error: Exception | None = None,
) -> ProviderResult:
    """Build a minimal ProviderResult for testing."""
    if response is None:
        response = {
            "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    return ProviderResult(
        response=response,
        finish_reason=finish_reason,
        error=error,
    )


# ============================================================================
# Stage protocol tests
# ============================================================================

class TestStageProtocol:
    """Every pipeline stage must conform to the Stage protocol."""

    def test_all_stages_have_process_method(self):
        from agent.orchestrator.stages import (
            RequestPrepStage,
            ProviderCallStage,
            ResponseProcessingStage,
            ToolDispatchStage,
            ContextManagementStage,
        )
        stages = [
            RequestPrepStage(),
            ProviderCallStage(),
            ResponseProcessingStage(),
            ToolDispatchStage(),
            ContextManagementStage(),
        ]
        for stage in stages:
            assert hasattr(stage, 'process') and callable(stage.process), \
                f"{stage.__class__.__name__} has no callable process method"

    def test_each_stage_has_process_method(self):
        from agent.orchestrator.stages import (
            RequestPrepStage,
            ProviderCallStage,
            ResponseProcessingStage,
            ToolDispatchStage,
            ContextManagementStage,
        )
        for cls in [RequestPrepStage, ProviderCallStage, ResponseProcessingStage,
                     ToolDispatchStage, ContextManagementStage]:
            assert hasattr(cls, 'process'), f"{cls.__name__} missing process method"

    def test_each_stage_has_name(self):
        from agent.orchestrator.stages import (
            RequestPrepStage,
            ProviderCallStage,
            ResponseProcessingStage,
            ToolDispatchStage,
            ContextManagementStage,
        )
        for cls in [RequestPrepStage, ProviderCallStage, ResponseProcessingStage,
                     ToolDispatchStage, ContextManagementStage]:
            instance = cls()
            assert hasattr(instance, 'name'), f"{cls.__name__} missing name attribute"
            assert isinstance(instance.name, str)


# ============================================================================
# PipelineResult
# ============================================================================

class TestPipelineResult:
    """PipelineResult carries the output of the full pipeline loop."""

    def test_pipeline_result_has_response(self):
        from agent.orchestrator.stages import PipelineResult
        ctx = _make_ctx()
        parsed = _make_parsed_response()
        result = PipelineResult(context=ctx, response=parsed)
        assert result.response is parsed
        assert result.context is ctx

    def test_pipeline_result_has_iterations(self):
        from agent.orchestrator.stages import PipelineResult
        result = PipelineResult(
            context=_make_ctx(),
            response=_make_parsed_response(),
            iterations=3,
        )
        assert result.iterations == 3

    def test_pipeline_result_has_interrupted_flag(self):
        from agent.orchestrator.stages import PipelineResult
        result = PipelineResult(
            context=_make_ctx(),
            response=_make_parsed_response(),
            interrupted=True,
        )
        assert result.interrupted is True

    def test_pipeline_result_interrupted_defaults_false(self):
        from agent.orchestrator.stages import PipelineResult
        result = PipelineResult(
            context=_make_ctx(),
            response=_make_parsed_response(),
        )
        assert result.interrupted is False

    def test_pipeline_result_has_tool_results(self):
        from agent.orchestrator.stages import PipelineResult
        tool_results = [{"role": "tool", "content": "ok"}]
        result = PipelineResult(
            context=_make_ctx(),
            response=_make_parsed_response(tool_calls=[{"id": "call_1"}]),
            tool_results=tool_results,
        )
        assert result.tool_results == tool_results

    def test_pipeline_result_default_tool_results_empty(self):
        from agent.orchestrator.stages import PipelineResult
        result = PipelineResult(
            context=_make_ctx(),
            response=_make_parsed_response(),
        )
        assert result.tool_results == []


# ============================================================================
# StageAction (loop decision)
# ============================================================================

class TestStageAction:
    """StageAction drives the loop: CONTINUE, YIELD, TERMINATE."""

    def test_continue_means_loop(self):
        from agent.orchestrator.stages import StageAction
        assert StageAction.CONTINUE.value == "continue"

    def test_yield_means_break_with_tool_results(self):
        from agent.orchestrator.stages import StageAction
        assert StageAction.YIELD.value == "yield"

    def test_terminate_means_max_iterations_or_fatal(self):
        from agent.orchestrator.stages import StageAction
        assert StageAction.TERMINATE.value == "terminate"

    def test_all_actions_exist(self):
        from agent.orchestrator.stages import StageAction
        actions = {a.value for a in StageAction}
        assert actions == {"continue", "yield", "terminate"}


# ============================================================================
# Stage 1: Request Preparation
# ============================================================================

class TestRequestPrepStage:
    """Stage 1: Build system prompt, prepare messages, select provider,
    check nudge status, and produce a PreparedRequest."""

    def test_process_returns_prepared_request(self):
        from agent.orchestrator.stages import RequestPrepStage, PreparedRequest
        stage = RequestPrepStage()
        ctx = _make_ctx()
        result = stage.process(ctx)
        assert isinstance(result, PreparedRequest)

    def test_prepared_request_has_api_kwargs(self):
        from agent.orchestrator.stages import RequestPrepStage, PreparedRequest
        stage = RequestPrepStage()
        ctx = _make_ctx()
        result = stage.process(ctx)
        assert isinstance(result, PreparedRequest)
        assert isinstance(result.api_kwargs, dict)

    def test_prepared_request_has_messages(self):
        from agent.orchestrator.stages import RequestPrepStage, PreparedRequest
        stage = RequestPrepStage()
        ctx = _make_ctx(messages=[{"role": "user", "content": "Hello"}])
        result = stage.process(ctx)
        assert len(result.messages) >= 1

    def test_prepared_request_includes_system_prompt(self):
        from agent.orchestrator.stages import RequestPrepStage, PreparedRequest
        stage = RequestPrepStage()
        ctx = _make_ctx(system_prompt="You are helpful.")
        result = stage.process(ctx)
        # System prompt should be injected as first message
        system_msgs = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1
        assert "helpful" in system_msgs[0].get("content", "")

    def test_prepared_request_inherits_tools(self):
        from agent.orchestrator.stages import RequestPrepStage, PreparedRequest
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        stage = RequestPrepStage()
        ctx = _make_ctx(tools=tools)
        result = stage.process(ctx)
        assert result.api_kwargs.get("tools") == tools

    def test_prepared_request_model_set(self):
        from agent.orchestrator.stages import RequestPrepStage, PreparedRequest
        stage = RequestPrepStage()
        ctx = _make_ctx()
        result = stage.process(ctx)
        assert "model" in result.api_kwargs

    def test_prepared_request_preserves_user_messages(self):
        from agent.orchestrator.stages import RequestPrepStage, PreparedRequest
        stage = RequestPrepStage()
        ctx = _make_ctx(messages=[
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ])
        result = stage.process(ctx)
        user_msgs = [m for m in result.messages if m.get("role") == "user"]
        assert len(user_msgs) == 2


# ============================================================================
# Stage 2: Provider Call
# ============================================================================

class TestProviderCallStage:
    """Stage 2: Execute the API call via ProviderProtocol.

    This stage takes a PreparedRequest and calls the provider.
    Returns a ProviderResult.
    """

    def test_process_returns_provider_result(self):
        from agent.orchestrator.stages import ProviderCallStage, PreparedRequest
        stage = ProviderCallStage()
        ctx = _make_ctx()
        prepared = PreparedRequest(
            messages=[{"role": "system", "content": "You are helpful."},
                      {"role": "user", "content": "Hello"}],
            api_kwargs={"model": "test-model"},
            provider_name="openai_compatible",
        )
        # With no provider registry, should error gracefully
        result = stage.process(ctx, prepared)
        assert isinstance(result, ProviderResult)

    def test_process_with_mock_provider(self):
        from agent.orchestrator.stages import ProviderCallStage, PreparedRequest
        mock_provider = MagicMock()
        mock_provider.execute.return_value = ProviderResult(
            response={"choices": [{"message": {"role": "assistant", "content": "Hi"}}]},
            finish_reason="stop",
        )
        registry = ProviderRegistry()
        registry.register("test_provider", mock_provider)
        stage = ProviderCallStage(registry=registry)
        ctx = _make_ctx()
        prepared = PreparedRequest(
            messages=[{"role": "user", "content": "Hello"}],
            api_kwargs={"model": "test-model"},
            provider_name="test_provider",
        )
        result = stage.process(ctx, prepared)
        assert result.finish_reason == "stop"
        mock_provider.execute.assert_called_once()

    def test_process_emits_events(self):
        from agent.orchestrator.stages import ProviderCallStage, PreparedRequest
        bus = EventBus()
        events = []
        bus.subscribe("provider_call_start", lambda e: events.append(e))
        bus.subscribe("provider_call_end", lambda e: events.append(e))

        mock_provider = MagicMock()
        mock_provider.execute.return_value = ProviderResult(finish_reason="stop")
        registry = ProviderRegistry()
        registry.register("test_provider", mock_provider)

        stage = ProviderCallStage(registry=registry, event_bus=bus)
        ctx = _make_ctx()
        prepared = PreparedRequest(
            messages=[{"role": "user", "content": "Hello"}],
            api_kwargs={"model": "test"},
            provider_name="test_provider",
        )
        stage.process(ctx, prepared)
        assert len(events) == 2
        assert events[0].kind == "provider_call_start"

    def test_process_handles_provider_error(self):
        from agent.orchestrator.stages import ProviderCallStage, PreparedRequest
        mock_provider = MagicMock()
        mock_provider.execute.side_effect = ConnectionError("API down")
        registry = ProviderRegistry()
        registry.register("broken", mock_provider)

        stage = ProviderCallStage(registry=registry)
        ctx = _make_ctx()
        prepared = PreparedRequest(
            messages=[{"role": "user", "content": "Hello"}],
            api_kwargs={"model": "test"},
            provider_name="broken",
        )
        result = stage.process(ctx, prepared)
        assert result.error is not None
        assert isinstance(result.error, ConnectionError)
        assert result.should_fallback is True


# ============================================================================
# Stage 3: Response Processing
# ============================================================================

class TestResponseProcessingStage:
    """Stage 3: Parse ProviderResult into ParsedResponse.

    Extracts reasoning content, tool calls, finish reason.
    Handles empty responses, truncation detection.
    """

    def test_basic_text_response(self):
        from agent.orchestrator.stages import ResponseProcessingStage
        stage = ResponseProcessingStage()
        ctx = _make_ctx()
        provider_result = ProviderResult(
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            finish_reason="stop",
        )
        parsed = stage.process(ctx, provider_result)
        assert isinstance(parsed, ParsedResponse)
        assert parsed.message["content"] == "Hello!"
        assert parsed.finish_reason == "stop"

    def test_tool_call_response(self):
        from agent.orchestrator.stages import ResponseProcessingStage
        stage = ResponseProcessingStage()
        ctx = _make_ctx()
        provider_result = ProviderResult(
            response={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "/tmp/test"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            finish_reason="tool_calls",
        )
        parsed = stage.process(ctx, provider_result)
        assert parsed.has_tool_calls is True
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["function"]["name"] == "read_file"

    def test_reasoning_content_extraction(self):
        from agent.orchestrator.stages import ResponseProcessingStage
        stage = ResponseProcessingStage()
        ctx = _make_ctx()
        provider_result = ProviderResult(
            response={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                    },
                    "finish_reason": "stop",
                }],
                "usage": {},
            },
            finish_reason="stop",
        )
        parsed = stage.process(ctx, provider_result)
        # reasoning_content may be None if not present
        assert parsed.message["content"] == "The answer is 42."

    def test_empty_response_handling(self):
        from agent.orchestrator.stages import ResponseProcessingStage
        stage = ResponseProcessingStage()
        ctx = _make_ctx()
        provider_result = ProviderResult(
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }],
            },
            finish_reason="stop",
        )
        parsed = stage.process(ctx, provider_result)
        assert parsed.message["content"] == ""

    def test_usage_extracted(self):
        from agent.orchestrator.stages import ResponseProcessingStage
        stage = ResponseProcessingStage()
        ctx = _make_ctx()
        provider_result = ProviderResult(
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
            finish_reason="stop",
        )
        parsed = stage.process(ctx, provider_result)
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 100
        assert parsed.usage.completion_tokens == 50

    def test_provider_error_returns_error_parsed_response(self):
        from agent.orchestrator.stages import ResponseProcessingStage
        stage = ResponseProcessingStage()
        ctx = _make_ctx()
        provider_result = ProviderResult(
            error=ConnectionError("timeout"),
            should_fallback=True,
        )
        parsed = stage.process(ctx, provider_result)
        assert parsed.finish_reason == "error"
        assert parsed.message is not None  # error message populated

    def test_finish_reason_length_truncation(self):
        from agent.orchestrator.stages import ResponseProcessingStage
        stage = ResponseProcessingStage()
        ctx = _make_ctx()
        provider_result = ProviderResult(
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "I was saying..."},
                    "finish_reason": "length",
                }],
            },
            finish_reason="length",
        )
        parsed = stage.process(ctx, provider_result)
        assert parsed.finish_reason == "length"


# ============================================================================
# Stage 4: Tool Dispatch
# ============================================================================

class TestToolDispatchStage:
    """Stage 4: Route tool calls to handlers and collect results.

    Tool dispatch is the most complex stage. It:
    - Detects tool calls in ParsedResponse
    - Routes to tool handlers (sequential or concurrent)
    - Returns tool result messages to append to conversation
    - Skips when no tool calls present
    """

    def test_no_tool_calls_yields(self):
        from agent.orchestrator.stages import ToolDispatchStage, ToolDispatchResult
        stage = ToolDispatchStage()
        ctx = _make_ctx()
        parsed = _make_parsed_response()  # no tool calls
        result = stage.process(ctx, parsed)
        assert isinstance(result, ToolDispatchResult)
        assert result.tool_results == []
        assert result.action.value == "yield"

    def test_tool_call_produces_tool_result(self):
        from agent.orchestrator.stages import ToolDispatchStage, ToolDispatchResult
        mock_handler = MagicMock(return_value={"result": "file contents"})
        stage = ToolDispatchStage(tool_handler=mock_handler)
        ctx = _make_ctx()
        parsed = _make_parsed_response(
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/tmp/test"}'},
            }],
            finish_reason="tool_calls",
        )
        result = stage.process(ctx, parsed)
        assert len(result.tool_results) == 1
        assert result.action.value == "continue"

    def test_tool_result_has_correct_role(self):
        from agent.orchestrator.stages import ToolDispatchStage
        mock_handler = MagicMock(return_value="ok")
        stage = ToolDispatchStage(tool_handler=mock_handler)
        ctx = _make_ctx()
        parsed = _make_parsed_response(
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }],
        )
        result = stage.process(ctx, parsed)
        assert result.tool_results[0]["role"] == "tool"

    def test_multiple_tool_calls_dispatched(self):
        from agent.orchestrator.stages import ToolDispatchStage
        call_count = 0

        def handler(name, args):
            return f"result_{name}"

        stage = ToolDispatchStage(tool_handler=handler)
        ctx = _make_ctx()
        parsed = _make_parsed_response(
            tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
            ],
        )
        result = stage.process(ctx, parsed)
        assert len(result.tool_results) == 2

    def test_tool_error_produces_error_result(self):
        from agent.orchestrator.stages import ToolDispatchStage
        def failing_handler(name, args):
            raise RuntimeError("Tool failed")
        stage = ToolDispatchStage(tool_handler=failing_handler)
        ctx = _make_ctx()
        parsed = _make_parsed_response(
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "failing_tool", "arguments": "{}"},
            }],
        )
        result = stage.process(ctx, parsed)
        # Should still produce a result, just with error content
        assert len(result.tool_results) == 1
        assert "error" in result.tool_results[0].get("content", "").lower() or "failed" in result.tool_results[0].get("content", "").lower()


# ============================================================================
# Stage 5: Context Management
# ============================================================================

class TestContextManagementStage:
    """Stage 5: Append tool results, check context limits, persist state.

    This is where we decide: loop back or stop.
    """

    def test_append_tool_results_to_context(self):
        from agent.orchestrator.stages import ContextManagementStage
        stage = ContextManagementStage()
        ctx = _make_ctx(messages=[{"role": "user", "content": "hello"}])
        tool_results = [
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
        ]
        action = stage.process(ctx, tool_results=tool_results, parsed=_make_parsed_response())
        assert len(ctx.messages) == 3  # original + assistant + tool result
        assert ctx.messages[-1]["role"] == "tool"

    def test_no_tool_results_no_append(self):
        from agent.orchestrator.stages import ContextManagementStage, StageAction
        stage = ContextManagementStage()
        ctx = _make_ctx(messages=[{"role": "user", "content": "hello"}])
        action = stage.process(ctx, tool_results=[], parsed=_make_parsed_response())
        action_enum = StageAction(action) if isinstance(action, str) else action
        # No tool results means we're done — YIELD
        assert action_enum == StageAction.YIELD or action_enum.value == "yield"

    def test_with_tool_results_continue(self):
        from agent.orchestrator.stages import ContextManagementStage, StageAction
        stage = ContextManagementStage()
        ctx = _make_ctx(messages=[{"role": "user", "content": "hello"}])
        tool_results = [
            {"role": "tool", "tool_call_id": "call_1", "content": "data"},
        ]
        action = stage.process(ctx, tool_results=tool_results, parsed=_make_parsed_response(tool_calls=[{"id": "call_1"}]))
        # Tool results mean we loop back — CONTINUE
        assert action == StageAction.CONTINUE or action.value == "continue"

    def test_max_iterations_terminate(self):
        from agent.orchestrator.stages import ContextManagementStage, StageAction
        stage = ContextManagementStage()
        ctx = _make_ctx(
            messages=[{"role": "user", "content": "hello"}],
            iteration=90,
            max_iterations=90,
        )
        action = stage.process(ctx, tool_results=[], parsed=_make_parsed_response())
        assert action == StageAction.TERMINATE or action.value == "terminate"

    def test_interrupted_yields(self):
        from agent.orchestrator.stages import ContextManagementStage, StageAction
        stage = ContextManagementStage()
        ctx = _make_ctx(messages=[{"role": "user", "content": "hello"}])
        ctx.interrupt_event.set()
        action = stage.process(ctx, tool_results=[], parsed=_make_parsed_response())
        assert action == StageAction.YIELD or action.value == "yield"

    def test_iteration_incremented(self):
        from agent.orchestrator.stages import ContextManagementStage
        stage = ContextManagementStage()
        ctx = _make_ctx(messages=[{"role": "user", "content": "hello"}])
        ctx.iteration = 3
        stage.process(ctx, tool_results=[], parsed=_make_parsed_response())
        assert ctx.iteration == 4


# ============================================================================
# PreparedRequest (Stage 1 output / Stage 2 input)
# ============================================================================

class TestPreparedRequest:
    """PreparedRequest carries the API-ready request between Stage 1 and 2."""

    def test_prepared_request_fields(self):
        from agent.orchestrator.stages import PreparedRequest
        req = PreparedRequest(
            messages=[{"role": "user", "content": "hi"}],
            api_kwargs={"model": "gpt-4", "temperature": 0.7},
            provider_name="openai_compatible",
        )
        assert req.messages == [{"role": "user", "content": "hi"}]
        assert req.api_kwargs["model"] == "gpt-4"
        assert req.provider_name == "openai_compatible"

    def test_prepared_request_has_cache_markers(self):
        from agent.orchestrator.stages import PreparedRequest
        req = PreparedRequest(
            messages=[],
            api_kwargs={},
            provider_name="anthropic",
            cache_markers=[0, 2],  # breakpoint positions
        )
        assert req.cache_markers == [0, 2]

    def test_prepared_request_cache_markers_default_empty(self):
        from agent.orchestrator.stages import PreparedRequest
        req = PreparedRequest(messages=[], api_kwargs={}, provider_name="test")
        assert req.cache_markers == []


# ============================================================================
# ToolDispatchResult (Stage 4 output)
# ============================================================================

class TestToolDispatchResult:
    """ToolDispatchResult carries tool results and the loop action."""

    def test_tool_dispatch_result_fields(self):
        from agent.orchestrator.stages import ToolDispatchResult, StageAction
        result = ToolDispatchResult(
            tool_results=[{"role": "tool", "content": "ok"}],
            action=StageAction.CONTINUE,
        )
        assert len(result.tool_results) == 1
        assert result.action == StageAction.CONTINUE

    def test_empty_tool_dispatch_result(self):
        from agent.orchestrator.stages import ToolDispatchResult, StageAction
        result = ToolDispatchResult(
            tool_results=[],
            action=StageAction.YIELD,
        )
        assert result.tool_results == []
        assert result.action == StageAction.YIELD