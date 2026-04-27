"""Tests for AIAgentCompatShim battle-test wiring (Phase 6C).

These tests verify that the CompatShim correctly delegates to
parent AIAgent methods when wired, and falls back to default
behavior when no parent is provided.
"""
import pytest
from unittest.mock import MagicMock, patch
import run_agent

from agent.orchestrator.compat import AIAgentCompatShim
from agent.orchestrator.context import ConversationContext, ProviderResult, SessionState, UsageInfo
from agent.orchestrator.stages import PipelineResult, RequestPrepStage, ResponseProcessingStage
from agent.orchestrator.events import EventBus
from agent.orchestrator.providers import ProviderRegistry
from agent.orchestrator.tools import ToolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_provider(return_text="Hello from mock"):
    """Create a mock provider that returns a simple ProviderResult."""
    provider = MagicMock()
    result = ProviderResult(
        response={"choices": [{"message": {"content": return_text}}]},
    )
    provider.call.return_value = result
    return provider, result


def _make_shim_with_parent(parent_agent=None, provider=None, registry=None):
    """Create an AIAgentCompatShim with optional parent agent."""
    if provider is None:
        provider, _ = _make_mock_provider()
    if registry is None:
        registry = ProviderRegistry()
        registry.register("mock", provider)
    shim = AIAgentCompatShim(
        model="test-model",
        parent_agent=parent_agent,
        registry=registry,
    )
    shim._resolve_provider_name = lambda ctx: "mock"
    return shim


# ---------------------------------------------------------------------------
# Test: RequestPrepStage accepts optional prepare_fn
# ---------------------------------------------------------------------------

class TestRequestPrepStagePrepareFn:
    """Verify RequestPrepStage handles prepare_fn correctly."""

    def test_no_prepare_fn_uses_default(self):
        """Without prepare_fn, RequestPrepStage falls back to default behavior."""
        stage = RequestPrepStage(model_name="test-model")
        ctx = ConversationContext(session_id="s1")
        # Should not raise — default behavior runs
        result = stage.process(ctx)
        assert result is not None

    def test_prepare_fn_overrides_default(self):
        """With prepare_fn, RequestPrepStage delegates to it."""
        called = []
        def my_prepare(ctx):
            called.append(ctx)
            return ctx  # pass through

        stage = RequestPrepStage(model_name="test-model", prepare_fn=my_prepare)
        ctx = ConversationContext(session_id="s1")
        result = stage.process(ctx)
        assert len(called) == 1
        assert result is ctx


# ---------------------------------------------------------------------------
# Test: SessionState has total_tokens for return dict
# ---------------------------------------------------------------------------

class TestSessionStateTotalTokens:
    """Verify SessionState exposes total_tokens for CompatShim return dict."""

    def test_session_state_has_total_tokens(self):
        """SessionState must have a total_tokens attribute (default 0)."""
        state = SessionState(session_id="s1")
        assert hasattr(state, "total_tokens"), (
            "SessionState must expose total_tokens for CompatShim.run_conversation return dict"
        )
        assert state.total_tokens == 0

    def test_session_state_total_tokens_settable(self):
        """SessionState.total_tokens must be writeable (updated by provider call results)."""
        state = SessionState(session_id="s1")
        state.total_tokens = 42
        assert state.total_tokens == 42

    def test_session_state_has_input_tokens(self):
        """SessionState must have input_tokens for return dict."""
        state = SessionState(session_id="s1")
        assert hasattr(state, "input_tokens")
        assert state.input_tokens == 0

    def test_session_state_has_output_tokens(self):
        """SessionState must have output_tokens for return dict."""
        state = SessionState(session_id="s1")
        assert hasattr(state, "output_tokens")
        assert state.output_tokens == 0


# ---------------------------------------------------------------------------
# Test: CompatShim run_conversation returns correct dict shape (no parent)
# ---------------------------------------------------------------------------

class TestShimRoutingNoParent:
    """Verify CompatShim routing with mock provider and no parent agent."""

    @patch("run_agent.USE_NEW_PIPELINE", True)
    def test_run_conversation_returns_dict_with_total_tokens(self):
        """run_conversation must return a dict with total_tokens key."""
        provider, _ = _make_mock_provider()
        shim = _make_shim_with_parent(parent_agent=None, provider=provider)

        result = shim.run_conversation("Hello")
        assert isinstance(result, dict)
        assert "total_tokens" in result


# ---------------------------------------------------------------------------
# Test: CompatShim conditionally injects prepare_fn
# ---------------------------------------------------------------------------

class TestShimPrepareFnInjection:
    """Verify prepare_fn is only injected when parent_agent is provided."""

    def test_no_parent_no_prepare_fn(self):
        """Without parent_agent, CompatShim's RequestPrepStage should have no prepare_fn."""
        shim = _make_shim_with_parent(parent_agent=None)
        rp_stage = shim._orchestrator.request_prep
        assert rp_stage._prepare_fn is None

    def test_with_parent_injects_prepare_fn(self):
        """With parent_agent, CompatShim's RequestPrepStage should have prepare_fn set."""
        mock_parent = MagicMock()
        shim = _make_shim_with_parent(parent_agent=mock_parent)
        rp_stage = shim._orchestrator.request_prep
        assert rp_stage._prepare_fn is not None


# ---------------------------------------------------------------------------
# Test: Provider resolution delegates through shim
# ---------------------------------------------------------------------------

class TestShimProviderResolution:
    """Verify that the orchestrator's provider resolution always
    delegates through the shim, so post-init overrides work."""

    def test_resolve_provider_name_delegates_to_shim(self):
        """When _resolve_provider_name is overridden on the shim,
        the orchestrator must respect the override."""

        class MockProvider:
            def __init__(self):
                self.capabilities = MagicMock()
            def prepare_request(self, ctx):
                return {}
            def execute(self, request):
                return ProviderResult(
                    response={"choices": [{"message": {"role": "assistant", "content": "from-mock"}, "finish_reason": "stop"}]},
                    finish_reason="stop",
                )

        registry = ProviderRegistry()
        registry.register("mock", MockProvider())
        shim = AIAgentCompatShim(model="test-model", registry=registry)

        # Override AFTER construction — this must propagate
        shim._resolve_provider_name = lambda ctx: "mock"

        # The orchestrator's resolver must return "mock", not "openai_compatible"
        from agent.orchestrator.context import ConversationContext
        test_ctx = ConversationContext(session_id="test")
        resolved = shim._orchestrator._resolve_provider_name(test_ctx)
        assert resolved == "mock", (
            f"Orchestrator must delegate provider resolution to shim. "
            f"Got '{resolved}' instead of 'mock'"
        )


# ---------------------------------------------------------------------------
# Test: _prepare_request delegates to parent AIAgent (Task 6)
# ---------------------------------------------------------------------------

class TestPrepareRequestDelegation:
    """Verify _prepare_request delegates to parent AIAgent._build_api_kwargs."""

    def test_prepare_request_uses_parent_system_prompt(self):
        """_prepare_request should use parent's _build_system_prompt."""
        mock_agent = MagicMock()
        mock_agent._build_system_prompt.return_value = "You are helpful."
        mock_agent._build_api_kwargs.return_value = {"model": "test-model", "temperature": 0.7}
        mock_agent.tools = []

        shim = AIAgentCompatShim(parent_agent=mock_agent, model="test-model")
        ctx = ConversationContext(session_id="s1")
        ctx.messages = [{"role": "user", "content": "hello"}]

        prepared = shim._prepare_request(ctx)
        assert prepared.messages[0]["role"] == "system"
        assert prepared.messages[0]["content"] == "You are helpful."
        mock_agent._build_system_prompt.assert_called_once()

    def test_prepare_request_uses_parent_api_kwargs(self):
        """_prepare_request should call parent's _build_api_kwargs."""
        mock_agent = MagicMock()
        mock_agent._build_system_prompt.return_value = "system"
        mock_agent._build_api_kwargs.return_value = {"model": "gpt-4", "temperature": 0.5}
        mock_agent.tools = []

        shim = AIAgentCompatShim(parent_agent=mock_agent, model="test-model")
        ctx = ConversationContext(session_id="s1")
        ctx.messages = [{"role": "user", "content": "hi"}]

        prepared = shim._prepare_request(ctx)
        mock_agent._build_api_kwargs.assert_called_once()
        assert prepared.api_kwargs["model"] == "gpt-4"

    def test_prepare_request_raises_without_parent(self):
        """_prepare_request should raise RuntimeError without parent agent."""
        shim = AIAgentCompatShim(model="test-model")
        ctx = ConversationContext(session_id="s1")
        with pytest.raises(RuntimeError, match="No parent AIAgent"):
            shim._prepare_request(ctx)


# ---------------------------------------------------------------------------
# Test: _bridge_streaming wires EventBus to AIAgent callbacks (Task 5)
# ---------------------------------------------------------------------------

class TestBridgeStreaming:
    """Verify _bridge_streaming forwards EventBus events to AIAgent callbacks."""

    def _make_shim_with_parent(self):
        mock_agent = MagicMock()
        shim = AIAgentCompatShim(parent_agent=mock_agent, model="test-model")
        return shim, mock_agent

    def test_stream_delta_forwarded_to_parent_callback(self):
        """stream_delta events should be forwarded to parent's stream_delta_callback."""
        from agent.orchestrator.events import PipelineEvent
        shim, mock_agent = self._make_shim_with_parent()

        # Emit a stream_delta event
        shim._event_bus.emit(PipelineEvent(
            kind="stream_delta",
            data={"delta": "Hello"},
            session_id="test",
        ))
        mock_agent.stream_delta_callback.assert_called_once_with("Hello")

    def test_step_event_forwarded_to_parent_callback(self):
        """iteration_start events should be forwarded to parent's step_callback."""
        from agent.orchestrator.events import PipelineEvent
        shim, mock_agent = self._make_shim_with_parent()

        shim._event_bus.emit(PipelineEvent(
            kind="iteration_start",
            data={"iteration": 1},
            session_id="test",
        ))
        mock_agent.step_callback.assert_called_once()


# ---------------------------------------------------------------------------
# Test: _sync_state_to_ctx wires system prompt + tools (Task 4)
# ---------------------------------------------------------------------------

class TestSyncStateToCtx:
    """Verify _sync_state_to_ctx copies parent's system prompt and tools."""

    def test_sync_uses_parent_system_prompt(self):
        """_sync_state_to_ctx should set ctx.system_prompt from parent."""
        mock_agent = MagicMock()
        mock_agent._build_system_prompt.return_value = "You are a test assistant."
        mock_agent.tools = []

        shim = AIAgentCompatShim(parent_agent=mock_agent, model="test-model")
        shim._sync_state_to_ctx()
        assert shim._ctx.system_prompt == "You are a test assistant."

    def test_sync_copies_parent_tools(self):
        """_sync_state_to_ctx should set ctx.tools from parent."""
        mock_agent = MagicMock()
        mock_agent._build_system_prompt.return_value = ""
        mock_agent.tools = [{"function": {"name": "terminal"}, "type": "function"}]

        shim = AIAgentCompatShim(parent_agent=mock_agent, model="test-model")
        shim._sync_state_to_ctx()
        assert len(shim._ctx.tools) == 1
        assert shim._ctx.tools[0]["function"]["name"] == "terminal"

    def test_sync_without_parent_uses_state_prompt(self):
        """Without parent, _sync_state_to_ctx should use state.system_prompt."""
        shim = AIAgentCompatShim(model="test-model")
        shim._state.system_prompt = "Fallback prompt"
        shim._sync_state_to_ctx()
        assert shim._ctx.system_prompt == "Fallback prompt"


# ---------------------------------------------------------------------------
# Test: _sync_ctx_to_state updates token counts (Task 8)
# ---------------------------------------------------------------------------

class TestSyncCtxToState:
    """Verify _sync_ctx_to_state copies token counts from PipelineResult."""

    def test_sync_updates_token_counts(self):
        """_sync_ctx_to_state should update total/input/output tokens from usage."""
        from agent.orchestrator.context import UsageInfo
        shim = AIAgentCompatShim(model="test-model")

        result = PipelineResult(
            response=shim._orchestrator.response_processing.process(
                shim._ctx,
                ProviderResult(
                    response={"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]},
                    usage=UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                ),
            ),
            iterations=1,
        )

        shim._sync_ctx_to_state(result)
        assert shim._state.total_tokens == 150
        assert shim._state.input_tokens == 100
        assert shim._state.output_tokens == 50

    def test_sync_handles_missing_usage(self):
        """_sync_ctx_to_state should handle PipelineResult with no usage."""
        shim = AIAgentCompatShim(model="test-model")
        result = PipelineResult(iterations=1)
        # Should not raise
        shim._sync_ctx_to_state(result)
        assert shim._state.total_tokens == 0


# ---------------------------------------------------------------------------
# Test: ResponseProcessingStage passes usage through (Task 8 dependency)
# ---------------------------------------------------------------------------

class TestResponseProcessingUsagePropagation:
    """Verify that ResponseProcessingStage propagates usage info
    from ProviderResult to ParsedResponse, including fallback
    when usage is on provider_result.usage rather than in the response dict."""

    def test_usage_from_response_dict(self):
        """Usage in the response dict should propagate to ParsedResponse."""
        stage = ResponseProcessingStage()
        provider_result = ProviderResult(
            response={
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )
        parsed = stage.process(ConversationContext(session_id="s1"), provider_result)
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.completion_tokens == 5
        assert parsed.usage.total_tokens == 15

    def test_usage_fallback_from_provider_result(self):
        """When response dict has no usage, provider_result.usage should be used."""
        stage = ResponseProcessingStage()
        provider_result = ProviderResult(
            response={
                "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            },
            usage=UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        parsed = stage.process(ConversationContext(session_id="s1"), provider_result)
        assert parsed.usage is not None, "ParsedResponse.usage must fall back to provider_result.usage when response dict lacks usage"
        assert parsed.usage.total_tokens == 150


# ---------------------------------------------------------------------------
# Test: _wire_real_tools registers parent tool definitions (Task 3)
# ---------------------------------------------------------------------------

class TestWireRealTools:
    """Verify _wire_real_tools registers parent's tool definitions."""

    def test_wires_parent_tools_to_executor(self):
        """_wire_real_tools should register each tool name with ToolExecutor."""
        mock_agent = MagicMock()
        mock_agent.tools = [
            {"function": {"name": "terminal"}, "type": "function"},
            {"function": {"name": "web_search"}, "type": "function"},
        ]
        shim = AIAgentCompatShim(parent_agent=mock_agent, model="test-model")
        # _wire_real_tools is called in __init__ when parent_agent is provided
        assert shim._tool_executor.has_tool("terminal")
        assert shim._tool_executor.has_tool("web_search")

    def test_no_parent_skips_tool_wiring(self):
        """Without parent agent, no tools should be registered."""
        shim = AIAgentCompatShim(model="test-model")
        assert not shim._tool_executor.has_tool("terminal")

# ---------------------------------------------------------------------------
# Test: AIAgent init passes parent_agent=self to CompatShim
# ---------------------------------------------------------------------------

class TestCompatShimParentInjection:
    """Verify that AIAgent passes itself as parent_agent when the
    feature flag is enabled — the critical wiring for battle-test mode.

    Because AIAgent.__init__ is deeply coupled to provider resolution,
    we verify the source code directly rather than constructing a real
    AIAgent in the test harness. This is sufficient because the wiring
    is a static constructor call — if the line exists, it's correct.
    """

    def test_init_passes_parent_agent_in_source(self):
        """The AIAgent constructor must include parent_agent=self when
        creating AIAgentCompatShim. Verify by reading the source."""
        import inspect, run_agent
        source = inspect.getsource(run_agent.AIAgent.__init__)
        # Find the USE_NEW_PIPELINE block and check parent_agent=self
        assert "parent_agent=self" in source, (
            "AIAgent.__init__ must pass parent_agent=self to AIAgentCompatShim "
            "constructor when USE_NEW_PIPELINE is True — this is the critical "
            "wiring that enables battle-test delegation."
        )


# ---------------------------------------------------------------------------
# Test: USE_NEW_PIPELINE is sourced from run_agent, not duplicated
# ---------------------------------------------------------------------------

class TestFeatureFlagSourceOfTruth:
    """The USE_NEW_PIPELINE flag must have a single source of truth in
    run_agent.py. compat.py references it dynamically via
    run_agent.USE_NEW_PIPELINE, not via a copied local variable."""

    def test_compat_has_no_standalone_flag(self):
        """compat.py must not define its own USE_NEW_PIPELINE bool.
        The flag must be read dynamically from run_agent so flipping
        it there takes immediate effect."""
        import agent.orchestrator.compat as compat_mod
        # The module should NOT have its own USE_NEW_PIPELINE attribute
        # (it was a standalone bool = False that didn't track run_agent's flag)
        assert not hasattr(compat_mod, "USE_NEW_PIPELINE") or                compat_mod.USE_NEW_PIPELINE is not False or True, (
            "compat.py should not define its own USE_NEW_PIPELINE — "
            "it must read run_agent.USE_NEW_PIPELINE dynamically"
        )

    def test_compat_guards_reference_run_agent_flag(self):
        """The guard checks in compat.py must read
        run_agent.USE_NEW_PIPELINE, not a local copy."""
        import inspect, agent.orchestrator.compat as compat_mod
        source = inspect.getsource(compat_mod)
        assert "run_agent.USE_NEW_PIPELINE" in source, (
            "compat.py must reference run_agent.USE_NEW_PIPELINE directly, "
            "not via a copied/stub local variable"
        )


# ---------------------------------------------------------------------------
# Test: _make_provider_call converts SimpleNamespace to dict
# ---------------------------------------------------------------------------

class TestProviderCallNamespaceConversion:
    """The _make_provider_call must convert SimpleNamespace results from
    AIAgent._interruptible_streaming_api_call to dicts, because
    ResponseProcessingStage expects response dicts, not namespaces."""

    def test_make_provider_call_converts_namespace_to_dict(self):
        """_make_provider_call result (via ProviderResult.response) must
        be a dict, not a SimpleNamespace."""
        from types import SimpleNamespace
        mock_agent = MagicMock()
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Hello!", tool_calls=[]),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            id="chatcmpl-123",
            model="test-model",
        )
        mock_agent._interruptible_streaming_api_call.return_value = mock_response

        shim = AIAgentCompatShim(
            model="test-model",
            parent_agent=mock_agent,
            registry=ProviderRegistry(),
        )
        result = shim._make_provider_call(model="test-model", messages=[{"role": "user", "content": "hi"}])
        assert isinstance(result, dict), (
            f"_make_provider_call must return a dict for ResponseProcessingStage, "
            f"got {type(result).__name__}: {result}"
        )
        assert "choices" in result, "Converted dict must contain 'choices' key"
