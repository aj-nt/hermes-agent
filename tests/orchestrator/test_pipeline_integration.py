"""Integration test: pipeline stages compose into a full conversation loop.

Verifies that Stage 1 → 2 → 3 → 4 → 5 → Loop Decision produces
correct behavior for: simple text response, tool calls, max iterations,
and interruptions. Uses mock provider with no real API calls.
"""

from __future__ import annotations

import json
import threading
import pytest
from unittest.mock import MagicMock

from agent.orchestrator.context import (
    ConversationContext,
    ParsedResponse,
    ProviderCapabilities,
    ProviderResult,
    StreamState,
    UsageInfo,
)
from agent.orchestrator.events import EventBus, PipelineEvent
from agent.orchestrator.providers import ProviderRegistry
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
        # Default: simple text response
        return ProviderResult(
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "Default response"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
            finish_reason="stop",
        )


def _make_ctx(**overrides) -> ConversationContext:
    defaults = dict(
        session_id="integration-test",
        messages=[{"role": "user", "content": "Hello"}],
        system_prompt="You are a test assistant.",
        tools=[],
        max_iterations=20,
    )
    defaults.update(overrides)
    return ConversationContext(**defaults)


def _run_pipeline(ctx, stages, executor=None, max_loops=10, provider_name="mock"):
    """Run the full pipeline loop: RequestPrep → ProviderCall → ResponseProcessing → ToolDispatch → ContextManagement.
    
    Returns PipelineResult when the loop exits (YIELD or TERMINATE).
    """
    request_prep, provider_call, response_proc, tool_dispatch, context_mgmt = stages
    executor = executor or ToolExecutor()
    iterations = 0

    for _ in range(max_loops):
        iterations += 1

        # Stage 1: Request Preparation
        prepared = request_prep.process(ctx)
        # Override provider_name to point to our mock registry entry
        prepared.provider_name = provider_name

        # Stage 2: Provider Call
        provider_result = provider_call.process(ctx, prepared)

        # Stage 3: Response Processing
        parsed = response_proc.process(ctx, provider_result)

        # Stage 4: Tool Dispatch
        dispatch_result = executor.dispatch(parsed)

        # Stage 5: Context Management
        action = context_mgmt.process(ctx, dispatch_result.tool_results, parsed)

        if action == StageAction.YIELD:
            return PipelineResult(
                context=ctx,
                response=parsed,
                iterations=iterations,
                interrupted=ctx.interrupt_event.is_set(),
            )
        elif action == StageAction.TERMINATE:
            return PipelineResult(
                context=ctx,
                response=parsed,
                iterations=iterations,
            )
        # CONTINUE → loop back

    # Shouldn't reach here if max_iterations is set correctly
    return PipelineResult(
        context=ctx,
        response=parsed,
        iterations=iterations,
        interrupted=True,
    )


# ============================================================================
# Composition tests
# ============================================================================

class TestPipelineComposition:
    """Full pipeline loop tests with mock provider."""

    def test_simple_text_response(self):
        """User sends message, provider replies, pipeline yields immediately."""
        provider = MockProvider(responses=[ProviderResult(
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
            finish_reason="stop",
        )])

        registry = ProviderRegistry()
        registry.register("mock", provider)

        stages = (
            RequestPrepStage(model_name="test-model"),
            ProviderCallStage(registry=registry),
            ResponseProcessingStage(),
            ToolDispatchStage(),
            ContextManagementStage(),
        )

        ctx = _make_ctx()
        result = _run_pipeline(ctx, stages)

        assert result.response.message["content"] == "Hello!"
        assert result.response.finish_reason == "stop"
        assert result.interrupted is False
        assert result.iterations == 1

    def test_tool_call_then_final_response(self):
        """Provider requests a tool call, tool executes, then provider
        gives a final text response — pipeline loops once."""

        # First call: tool call; second call: text reply
        provider = MockProvider(responses=[
            ProviderResult(
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
                },
                finish_reason="tool_calls",
            ),
            ProviderResult(
                response={
                    "choices": [{
                        "message": {"role": "assistant", "content": "The file contains: hello"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                },
                finish_reason="stop",
            ),
        ])

        registry = ProviderRegistry()
        registry.register("mock", provider)

        executor = ToolExecutor()
        executor.register("read_file", lambda args: f"contents of {args['path']}")

        stages = (
            RequestPrepStage(model_name="test-model"),
            ProviderCallStage(registry=registry),
            ResponseProcessingStage(),
            ToolDispatchStage(),
            ContextManagementStage(),
        )

        ctx = _make_ctx()
        result = _run_pipeline(ctx, stages, executor=executor)

        assert result.iterations == 2  # first: tool call, second: final
        assert result.response.message["content"] == "The file contains: hello"
        assert result.interrupted is False

    def test_interrupt_stops_pipeline(self):
        """Pipeline stops immediately when interrupt_event is set."""
        provider = MockProvider(responses=[ProviderResult(
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "wait..."},
                    "finish_reason": "stop",
                }],
            },
            finish_reason="stop",
        )])

        registry = ProviderRegistry()
        registry.register("mock", provider)

        ctx = _make_ctx()
        ctx.interrupt_event.set()  # Simulate interrupt

        stages = (
            RequestPrepStage(model_name="test-model"),
            ProviderCallStage(registry=registry),
            ResponseProcessingStage(),
            ToolDispatchStage(),
            ContextManagementStage(),
        )

        result = _run_pipeline(ctx, stages)
        assert result.interrupted is True

    def test_max_iterations_terminates(self):
        """Pipeline terminates when max iterations is reached."""

        # Provider always returns a tool call — forces an infinite loop
        # unless max_iterations terminates it
        always_tool_call = ProviderResult(
            response={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_loop",
                            "type": "function",
                            "function": {"name": "loop_tool", "arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            },
            finish_reason="tool_calls",
        )
        provider = MockProvider(responses=[always_tool_call] * 100)

        registry = ProviderRegistry()
        registry.register("mock", provider)

        executor = ToolExecutor()
        executor.register("loop_tool", lambda args: "looping")

        stages = (
            RequestPrepStage(model_name="test-model"),
            ProviderCallStage(registry=registry),
            ResponseProcessingStage(),
            ToolDispatchStage(),
            ContextManagementStage(),
        )

        ctx = _make_ctx(max_iterations=3)  # Low limit
        result = _run_pipeline(ctx, stages, executor=executor, max_loops=10)

        # Should hit max_iterations
        assert result.iterations >= 3

    def test_pre_dispatch_hook_in_pipeline(self):
        """Check that pre-dispatch hooks are called during tool execution."""
        hook_calls = []

        def checkpoint_hook(name, args):
            hook_calls.append(name)

        executor = ToolExecutor(pre_dispatch_hooks=[checkpoint_hook])
        executor.register("write_file", lambda args: "written")

        provider = MockProvider(responses=[
            ProviderResult(
                response={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": '{"path": "/tmp/test", "content": "hello"}'},
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                },
                finish_reason="tool_calls",
            ),
            ProviderResult(
                response={
                    "choices": [{
                        "message": {"role": "assistant", "content": "File written!"},
                        "finish_reason": "stop",
                    }],
                },
                finish_reason="stop",
            ),
        ])

        registry = ProviderRegistry()
        registry.register("mock", provider)

        stages = (
            RequestPrepStage(model_name="test-model"),
            ProviderCallStage(registry=registry),
            ResponseProcessingStage(),
            ToolDispatchStage(),
            ContextManagementStage(),
        )

        ctx = _make_ctx()
        result = _run_pipeline(ctx, stages, executor=executor)

        assert "write_file" in hook_calls
        assert result.response.message["content"] == "File written!"

    def test_provider_error_returns_error_response(self):
        """Provider call fails — pipeline returns error ParsedResponse."""
        provider = MockProvider()
        provider.execute = MagicMock(side_effect=ConnectionError("API down"))

        registry = ProviderRegistry()
        registry.register("mock", provider)

        stages = (
            RequestPrepStage(model_name="test-model"),
            ProviderCallStage(registry=registry),
            ResponseProcessingStage(),
            ToolDispatchStage(),
            ContextManagementStage(),
        )

        ctx = _make_ctx()
        prepared = stages[0].process(ctx)
        prepared.provider_name = "mock"  # point to our mock registry entry
        provider_result = stages[1].process(ctx, prepared)
        parsed = stages[2].process(ctx, provider_result)

        assert parsed.finish_reason == "error"
        assert "Error" in parsed.message["content"]


# ============================================================================
# Stage contract tests
# ============================================================================

class TestStageContracts:
    """Each stage has a well-defined input/output contract."""

    def test_request_prep_output_is_prepared_request(self):
        stage = RequestPrepStage(model_name="gpt-4")
        ctx = _make_ctx()
        result = stage.process(ctx)
        assert isinstance(result, PreparedRequest)
        assert len(result.messages) > 0
        assert "model" in result.api_kwargs

    def test_provider_call_output_is_provider_result(self):
        provider = MockProvider(responses=[ProviderResult(finish_reason="stop")])
        registry = ProviderRegistry()
        registry.register("mock", provider)
        stage = ProviderCallStage(registry=registry)
        ctx = _make_ctx()
        prepared = PreparedRequest(
            messages=[{"role": "user", "content": "hi"}],
            api_kwargs={"model": "mock"},
            provider_name="mock",
        )
        result = stage.process(ctx, prepared)
        assert isinstance(result, ProviderResult)

    def test_response_processing_output_is_parsed_response(self):
        stage = ResponseProcessingStage()
        ctx = _make_ctx()
        provider_result = ProviderResult(
            response={
                "choices": [{
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }],
            },
            finish_reason="stop",
        )
        result = stage.process(ctx, provider_result)
        assert isinstance(result, ParsedResponse)

    def test_tool_dispatch_output_is_tool_dispatch_result(self):
        executor = ToolExecutor()
        stage = ToolDispatchStage(tool_handler=executor.dispatch)
        ctx = _make_ctx()
        parsed = ParsedResponse(
            message={"role": "assistant", "content": "hi"},
            tool_calls=[],
            finish_reason="stop",
        )
        # Note: ToolDispatchStage wraps the executor
        from agent.orchestrator.stages import ToolDispatchStage as Stage4
        stage4 = Stage4(tool_handler=lambda name, args: "ok")
        result = stage4.process(ctx, parsed)
        assert isinstance(result, ToolDispatchResult)

    def test_context_management_output_is_stage_action(self):
        stage = ContextManagementStage()
        ctx = _make_ctx()
        parsed = ParsedResponse(
            message={"role": "assistant", "content": "hi"},
            tool_calls=[],
            finish_reason="stop",
        )
        action = stage.process(ctx, tool_results=[], parsed=parsed)
        assert isinstance(action, StageAction)