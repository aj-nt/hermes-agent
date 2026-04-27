"""Tests for the AIAgent compatibility shim (agent/orchestrator/compat.py).

The compat shim maps the 7 external method calls and 13 gateway-set
attributes to the new Orchestrator/SessionState API. This allows the
gateway, delegate_tool, and cron scheduler to continue working unchanged
during cutover.

Per DESIGN.md Phase 5:
- Feature flag: USE_NEW_PIPELINE=true
- Route chat() and run_conversation() through new Orchestrator
- Run both paths in parallel for validation
- AIAgent compatibility shim maps old attribute names to new fields
"""

from __future__ import annotations

import json
import pytest
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from agent.orchestrator.context import (
    ConversationContext,
    SessionState,
    StreamConfig,
    StreamState,
)
from agent.orchestrator.events import EventBus
from agent.orchestrator.orchestrator import Orchestrator
from agent.orchestrator.providers import ProviderRegistry
from agent.orchestrator.tools import ToolExecutor


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_provider(responses=None):
    """Create a mock provider with canned responses."""
    from agent.orchestrator.context import ProviderResult, ProviderCapabilities

    class MockProvider:
        def __init__(self):
            self._responses = responses or [
                ProviderResult(
                    response={
                        "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                    },
                    finish_reason="stop",
                ),
            ]
            self._call_index = 0
            self.capabilities = ProviderCapabilities()

        def prepare_request(self, ctx):
            return {}

        def execute(self, request):
            if self._call_index < len(self._responses):
                r = self._responses[self._call_index]
                self._call_index += 1
                return r
            return ProviderResult(
                response={"choices": [{"message": {"role": "assistant", "content": "Default"}, "finish_reason": "stop"}]},
                finish_reason="stop",
            )

    return MockProvider()


# ============================================================================
# Compat shim creation and interface
# ============================================================================

class TestCompatShimCreation:
    """AIAgentCompatShim wraps Orchestrator and provides the AIAgent API surface."""

    def test_create_shim_with_defaults(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert shim is not None

    def test_shim_has_chat_method(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert hasattr(shim, 'chat')
        assert callable(shim.chat)

    def test_shim_has_run_conversation_method(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert hasattr(shim, 'run_conversation')
        assert callable(shim.run_conversation)

    def test_shim_has_interrupt_method(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert hasattr(shim, 'interrupt')
        assert callable(shim.interrupt)

    def test_shim_has_release_clients_method(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert hasattr(shim, 'release_clients')

    def test_shim_has_close_method(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert hasattr(shim, 'close')

    def test_shim_has_shutdown_memory_provider_method(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert hasattr(shim, 'shutdown_memory_provider')

    def test_shim_has_commit_memory_session_method(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert hasattr(shim, 'commit_memory_session')

    def test_shim_has_get_activity_summary_method(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert hasattr(shim, 'get_activity_summary')


# ============================================================================
# Gateway-set attributes
# ============================================================================

class TestCompatShimGatewayAttributes:
    """The 13 attributes set by the gateway must be mutable on the shim."""

    def test_tool_progress_callback(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        callback = lambda x: None
        shim.tool_progress_callback = callback
        assert shim.tool_progress_callback is callback

    def test_step_callback(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        callback = lambda x: None
        shim.step_callback = callback
        assert shim.step_callback is callback

    def test_stream_delta_callback(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        callback = lambda x: None
        shim.stream_delta_callback = callback
        assert shim.stream_delta_callback is callback

    def test_interim_assistant_callback(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        callback = lambda x: None
        shim.interim_assistant_callback = callback
        assert shim.interim_assistant_callback is callback

    def test_status_callback(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        callback = lambda x: None
        shim.status_callback = callback
        assert shim.status_callback is callback

    def test_reasoning_config(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        shim.reasoning_config = {"effort": "high"}
        assert shim.reasoning_config == {"effort": "high"}

    def test_service_tier(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        shim.service_tier = "auto"
        assert shim.service_tier == "auto"

    def test_request_overrides(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        shim.request_overrides = {"temperature": 0.5}
        assert shim.request_overrides == {"temperature": 0.5}

    def test_background_review_callback(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        callback = lambda x: None
        shim.background_review_callback = callback
        assert shim.background_review_callback is callback

    def test_api_call_count(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        shim._api_call_count = 5
        assert shim._api_call_count == 5

    def test_last_activity_ts(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        import time
        ts = time.time()
        shim._last_activity_ts = ts
        assert shim._last_activity_ts == ts

    def test_last_activity_desc(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        shim._last_activity_desc = "running tool"
        assert shim._last_activity_desc == "running tool"


# ============================================================================
# Delegate-readable attributes
# ============================================================================

class TestCompatShimDelegateAttributes:
    """Attributes read by delegate_tool must be accessible on the shim."""

    def test_model_attribute(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="gpt-4o")
        assert shim.model == "gpt-4o"

    def test_base_url_attribute(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test", base_url="https://api.openai.com/v1")
        assert shim.base_url == "https://api.openai.com/v1"

    def test_provider_attribute(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test", provider="anthropic")
        assert shim.provider == "anthropic"

    def test_session_id_attribute(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test", session_id="sess-123")
        assert shim.session_id == "sess-123"

    def test_max_iterations_attribute(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test", max_iterations=50)
        assert shim.max_iterations == 50

    def test_default_max_iterations(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test")
        # Default is Orchestrator.DEFAULT_MAX_ITERATIONS (90)
        assert shim.max_iterations == 90


# ============================================================================
# Feature flag
# ============================================================================

class TestFeatureFlag:
    """USE_NEW_PIPELINE controls whether the shim routes through Orchestrator."""

    def test_feature_flag_exists(self):
        from agent.orchestrator.compat import USE_NEW_PIPELINE
        assert isinstance(USE_NEW_PIPELINE, bool)

    def test_feature_flag_default_false(self):
        """By default, the new pipeline is OFF — safe cutover."""
        from agent.orchestrator.compat import USE_NEW_PIPELINE
        assert USE_NEW_PIPELINE is False

    def test_feature_flag_can_be_overridden(self):
        """Tests can override the feature flag via monkeypatch."""
        import agent.orchestrator.compat as compat_module
        original = compat_module.USE_NEW_PIPELINE
        try:
            compat_module.USE_NEW_PIPELINE = True
            assert compat_module.USE_NEW_PIPELINE is True
        finally:
            compat_module.USE_NEW_PIPELINE = original


# ============================================================================
# chat() and run_conversation() routing
# ============================================================================

class TestShimRouting:
    """When USE_NEW_PIPELINE=True, chat() and run_conversation() route through Orchestrator."""

    def test_chat_returns_string(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        provider = _make_mock_provider()
        registry = ProviderRegistry()
        registry.register("mock", provider)

        shim = AIAgentCompatShim(model="test-model", registry=registry)
        shim._resolve_provider_name = lambda ctx: "mock"

        # Monkeypatch the feature flag
        import agent.orchestrator.compat as compat_module
        original_flag = compat_module.USE_NEW_PIPELINE
        try:
            compat_module.USE_NEW_PIPELINE = True
            result = shim.chat("Hello")
            assert isinstance(result, str)
        finally:
            compat_module.USE_NEW_PIPELINE = original_flag

    def test_run_conversation_returns_dict(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        from agent.orchestrator.context import ProviderResult

        provider = _make_mock_provider()
        registry = ProviderRegistry()
        registry.register("mock", provider)

        shim = AIAgentCompatShim(model="test-model", registry=registry)
        shim._resolve_provider_name = lambda ctx: "mock"

        import agent.orchestrator.compat as compat_module
        original_flag = compat_module.USE_NEW_PIPELINE
        try:
            compat_module.USE_NEW_PIPELINE = True
            result = shim.run_conversation("Hello")
            assert isinstance(result, dict)
            assert "final_response" in result
        finally:
            compat_module.USE_NEW_PIPELINE = original_flag


# ============================================================================
# switch_model()
# ============================================================================

class TestSwitchModel:
    """switch_model() updates provider, model, base_url on the shim."""

    def test_switch_model_changes_model(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="gpt-4")
        shim.switch_model("claude-sonnet-4", "anthropic", api_key="sk-test")
        assert shim.model == "claude-sonnet-4"

    def test_switch_model_changes_provider(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="gpt-4")
        shim.switch_model("claude-sonnet-4", "anthropic")
        assert shim.provider == "anthropic"

    def test_switch_model_changes_base_url(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="gpt-4")
        shim.switch_model("claude-sonnet-4", "anthropic", base_url="https://api.anthropic.com")
        # base_url should reflect the new value
        assert "anthropic" in shim.base_url or shim.base_url == "https://api.anthropic.com"


# ============================================================================
# interrupt()
# ============================================================================

class TestInterrupt:
    """interrupt() sets the interrupt event on the ConversationContext."""

    def test_interrupt_sets_event(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert shim._ctx.interrupt_event.is_set() is False
        shim.interrupt("user requested stop")
        assert shim._ctx.interrupt_event.is_set() is True

    def test_interrupt_stores_reason(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        shim.interrupt("user requested stop")
        # The reason is accessible somewhere on the shim
        assert shim._interrupt_reason == "user requested stop"


# ============================================================================
# Session state mapping
# ============================================================================

class TestSessionStateMapping:
    """Old AIAgent attributes map to SessionState fields."""

    def test_session_state_accessible(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert isinstance(shim._state, SessionState)

    def test_session_state_messages(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        # Initially empty
        assert shim._state.messages == []

    def test_conversation_context_accessible(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert isinstance(shim._ctx, ConversationContext)

    def test_event_bus_accessible(self):
        from agent.orchestrator.compat import AIAgentCompatShim
        shim = AIAgentCompatShim(model="test-model")
        assert isinstance(shim._event_bus, EventBus)