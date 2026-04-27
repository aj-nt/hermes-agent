"""Tests for AIAgentCompatShim battle-test wiring (Phase 6C).

These tests verify that the CompatShim correctly delegates to
parent AIAgent methods when wired, and falls back to default
behavior when no parent is provided.
"""
import pytest
from unittest.mock import MagicMock, patch

from agent.orchestrator.compat import AIAgentCompatShim
from agent.orchestrator.context import ConversationContext, ProviderResult, SessionState
from agent.orchestrator.stages import PipelineResult, RequestPrepStage
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

    @patch("agent.orchestrator.compat.USE_NEW_PIPELINE", True)
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