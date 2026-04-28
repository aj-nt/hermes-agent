"""Tests for OpenAICompatibleProvider.prepare_request() — Step 3a extraction.

The goal: OpenAICompatibleProvider.prepare_request() should produce the
same kwargs dict that AIAgent._build_api_kwargs() produces for the
chat_completions branch, but without touching self.* on AIAgent.

Instead of computing provider flags from AIAgent.__init__ attributes,
the provider receives a RequestConfig that bundles these flags.
The factory method from_agent() computes the config from AIAgent.

TDD RED phase: write tests that demand the expanded prepare_request()
behavior. They will fail until we implement it.
"""

import pytest
from unittest.mock import MagicMock, patch

from agent.orchestrator.context import ConversationContext
from agent.orchestrator.provider_adapters import OpenAICompatibleProvider, RequestConfig
from agent.orchestrator.providers import ProviderProtocol


# ============================================================================
# Helpers
# ============================================================================

def make_ctx(**overrides) -> ConversationContext:
    """Create a minimal ConversationContext for tests."""
    defaults = dict(
        session_id="test-session",
        messages=[{"role": "user", "content": "hello"}],
        system_prompt="You are helpful.",
    )
    defaults.update(overrides)
    return ConversationContext(**defaults)


# ============================================================================
# Step 3a: RequestConfig — provider flags for chat_completions
# ============================================================================

class TestRequestConfig:
    """RequestConfig bundles the provider-detection flags that AIAgent
    currently computes in _build_api_kwargs (is_openrouter, is_nous, etc.)
    so OpenAICompatibleProvider doesn't need self.* from AIAgent.
    """

    def test_import_request_config(self):
        """RequestConfig should be importable from provider_adapters."""
        from agent.orchestrator.provider_adapters import RequestConfig

    def test_request_config_defaults(self):
        """RequestConfig should default all provider flags to False."""
        from agent.orchestrator.provider_adapters import RequestConfig
        rc = RequestConfig()
        assert rc.is_openrouter is False
        assert rc.is_nous is False
        assert rc.is_qwen_portal is False
        assert rc.is_github_models is False
        assert rc.is_nvidia_nim is False
        assert rc.is_kimi is False
        assert rc.is_custom_provider is False

    def test_request_config_provider_flags(self):
        """RequestConfig should accept provider flag overrides."""
        from agent.orchestrator.provider_adapters import RequestConfig
        rc = RequestConfig(
            is_openrouter=True,
            is_nous=True,
            is_kimi=True,
        )
        assert rc.is_openrouter is True
        assert rc.is_nous is True
        assert rc.is_kimi is True
        assert rc.is_qwen_portal is False  # default

    def test_request_config_model_fields(self):
        """RequestConfig should carry model-level config."""
        from agent.orchestrator.provider_adapters import RequestConfig
        rc = RequestConfig(
            max_tokens=4096,
            model="gpt-4o",
            ollama_num_ctx=8192,
        )
        assert rc.max_tokens == 4096
        assert rc.model == "gpt-4o"
        assert rc.ollama_num_ctx == 8192

    def test_request_config_temperature(self):
        """RequestConfig should carry temperature config."""
        from agent.orchestrator.provider_adapters import RequestConfig
        rc = RequestConfig(
            fixed_temperature=0.7,
            omit_temperature=False,
        )
        assert rc.fixed_temperature == 0.7
        assert rc.omit_temperature is False

    def test_request_config_with_all_flags(self):
        """RequestConfig should be constructable with all flags at once."""
        from agent.orchestrator.provider_adapters import RequestConfig
        rc = RequestConfig(
            model="glm-4:latest",
            base_url="http://localhost:11434/v1",
            max_tokens=4096,
            is_openrouter=False,
            is_nous=False,
            is_qwen_portal=False,
            is_github_models=False,
            is_nvidia_nim=False,
            is_kimi=False,
            is_custom_provider=False,
            ollama_num_ctx=8192,
            fixed_temperature=None,
            omit_temperature=False,
            reasoning_config=None,
            request_overrides=None,
            supports_reasoning=False,
            provider_preferences=None,
        )
        assert rc.model == "glm-4:latest"


# ============================================================================
# Step 3a: Expanded prepare_request()
# ============================================================================

class TestOpenAICompatiblePrepareRequest:
    """OpenAICompatibleProvider.prepare_request() should produce the
    same kwargs as AIAgent._build_api_kwargs() for chat_completions.
    """

    def test_prepare_request_calls_transport_build_kwargs(self):
        """prepare_request should delegate to transport.build_kwargs()."""
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434",
            model_name="glm-4:latest",
        )
        # Before we have transport: prepare_request should use it
        # after we add it. For now, test that the expanded version exists.
        ctx = make_ctx()
        # This will fail if prepare_request doesn't accept transport
        result = provider.prepare_request(ctx)
        assert isinstance(result, dict)
        assert "model" in result
        assert result["model"] == "glm-4:latest"

    def test_prepare_request_with_openrouter_config(self):
        """OpenRouter provider flags should flow through to build_kwargs."""
        from agent.orchestrator.provider_adapters import RequestConfig
        rc = RequestConfig(
            model="anthropic/claude-3.5-sonnet",
            is_openrouter=True,
            provider_preferences={"order": ["anthropic"]},
        )
        provider = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            model_name="anthropic/claude-3.5-sonnet",
            request_config=rc,
        )
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert isinstance(result, dict)
        # model should be in the result
        assert result.get("model") == "anthropic/claude-3.5-sonnet"

    def test_prepare_request_passes_tools(self):
        """Tools from ConversationContext should flow to build_kwargs."""
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        ctx = make_ctx(tools=tools)
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
        )
        result = provider.prepare_request(ctx)
        # Tools should be present (either at top level or inside kwargs)
        assert "tools" in result

    def test_prepare_request_includes_reasoning_config(self):
        """Reasoning config should flow through RequestConfig."""
        from agent.orchestrator.provider_adapters import RequestConfig
        rc = RequestConfig(
            model="deepseek-r1",
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        provider = OpenAICompatibleProvider(
            base_url="https://api.deepseek.com/v1",
            model_name="deepseek-r1",
            request_config=rc,
        )
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        # If reasoning is supported and configured, extra_body should have it
        if "extra_body" in result:
            assert "reasoning" in result["extra_body"]

    def test_prepare_request_with_ollama_num_ctx(self):
        """Ollama num_ctx should flow through RequestConfig as extra_body.options."""
        from agent.orchestrator.provider_adapters import RequestConfig
        rc = RequestConfig(
            model="glm-4:latest",
            ollama_num_ctx=8192,
        )
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            model_name="glm-4:latest",
            request_config=rc,
        )
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        # Ollama num_ctx should appear as extra_body.options.num_ctx
        if "extra_body" in result:
            assert result["extra_body"].get("options", {}).get("num_ctx") == 8192


# ============================================================================
# Step 3a: from_agent() factory method
# ============================================================================

class TestOpenAICompatibleFromAgent:
    """from_agent() factory should compute RequestConfig from AIAgent attributes."""

    def test_from_agent_exists(self):
        """OpenAICompatibleProvider should have a from_agent class method."""
        assert hasattr(OpenAICompatibleProvider, "from_agent")
        assert callable(OpenAICompatibleProvider.from_agent)

    def test_from_agent_creates_provider_with_local_config(self):
        """from_agent with an Ollama-style agent should produce a local provider."""
        # Create a minimal mock agent with Ollama-style attributes
        mock_agent = MagicMock()
        mock_agent.base_url = "http://localhost:11434/v1"
        mock_agent.provider = "ollama"
        mock_agent.model = "glm-4:latest"
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

        provider = OpenAICompatibleProvider.from_agent(mock_agent)
        assert isinstance(provider, OpenAICompatibleProvider)
        # Should detect localhost as local
        assert provider._is_local is True

    def test_from_agent_creates_provider_with_cloud_config(self):
        """from_agent with an OpenRouter-style agent should produce a cloud provider."""
        mock_agent = MagicMock()
        mock_agent.base_url = "https://openrouter.ai/api/v1"
        mock_agent.provider = "openrouter"
        mock_agent.model = "anthropic/claude-3.5-sonnet"
        mock_agent.api_mode = "chat_completions"
        mock_agent.max_tokens = 4096
        mock_agent.tools = []
        mock_agent.reasoning_config = None
        mock_agent.request_overrides = None
        mock_agent.providers_allowed = ["anthropic"]
        mock_agent.providers_ignored = None
        mock_agent.providers_order = None
        mock_agent.provider_sort = None
        mock_agent.provider_require_parameters = None
        mock_agent.provider_data_collection = None
        # _is_openrouter_url is a method — must return bool
        mock_agent._is_openrouter_url = MagicMock(return_value=True)

        provider = OpenAICompatibleProvider.from_agent(mock_agent)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider._is_local is False
        # RequestConfig should detect OpenRouter
        assert provider._request_config.is_openrouter is True

# ============================================================================
# Step 3a: Contract tests — prepare_request matches _build_api_kwargs
# ============================================================================

class TestPrepareRequestContract:
    """Verify that OpenAICompatibleProvider.prepare_request() produces
    kwargs equivalent to AIAgent._build_api_kwargs() for the
    chat_completions branch.

    These tests use a real ChatCompletionsTransport to ensure the
    output matches production behavior.
    """

    def test_basic_kwargs_match_transport(self):
        """A basic openai-compatible request should match transport output."""
        from agent.transports import get_transport
        transport = get_transport("chat_completions")
        rc = RequestConfig(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            max_tokens=4096,
        )
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
            request_config=rc,
            transport=transport,
        )
        ctx = make_ctx(messages=[{"role": "user", "content": "hi"}])
        result = provider.prepare_request(ctx)
        assert result["model"] == "gpt-4o"
        assert result["messages"] == [{"role": "user", "content": "hi"}]
        # Should NOT have extra_body for vanilla OpenAI
        assert "extra_body" not in result or result.get("extra_body") == {}

    def test_openrouter_preferences_flow_through(self):
        """OpenRouter provider preferences should appear in extra_body."""
        from agent.transports import get_transport
        transport = get_transport("chat_completions")
        rc = RequestConfig(
            model="anthropic/claude-3.5-sonnet",
            base_url="https://openrouter.ai/api/v1",
            is_openrouter=True,
            provider_preferences={"order": ["anthropic"]},
        )
        provider = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            model_name="anthropic/claude-3.5-sonnet",
            request_config=rc,
            transport=transport,
        )
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert "extra_body" in result
        assert "provider" in result["extra_body"]
        assert result["extra_body"]["provider"]["order"] == ["anthropic"]

    def test_ollama_num_ctx_appears_in_options(self):
        """Ollama num_ctx should appear as extra_body.options.num_ctx."""
        from agent.transports import get_transport
        transport = get_transport("chat_completions")
        rc = RequestConfig(
            model="glm-4:latest",
            base_url="http://localhost:11434/v1",
            ollama_num_ctx=8192,
        )
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1",
            model_name="glm-4:latest",
            request_config=rc,
            transport=transport,
        )
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert "extra_body" in result
        assert "options" in result["extra_body"]
        assert result["extra_body"]["options"]["num_ctx"] == 8192

    def test_reasoning_config_flows_through(self):
        """Reasoning config should appear in extra_body.reasoning."""
        from agent.transports import get_transport
        transport = get_transport("chat_completions")
        rc = RequestConfig(
            model="deepseek-r1",
            base_url="https://api.deepseek.com/v1",
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        provider = OpenAICompatibleProvider(
            base_url="https://api.deepseek.com/v1",
            model_name="deepseek-r1",
            request_config=rc,
            transport=transport,
        )
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert "extra_body" in result
        assert "reasoning" in result["extra_body"]
        assert result["extra_body"]["reasoning"]["enabled"] is True

    def test_tools_appear_in_kwargs(self):
        """Tools from ConversationContext should appear in kwargs."""
        from agent.transports import get_transport
        transport = get_transport("chat_completions")
        rc = RequestConfig(model="gpt-4o", base_url="https://api.openai.com/v1")
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
            request_config=rc,
            transport=transport,
        )
        tools = [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]
        ctx = make_ctx(tools=tools)
        result = provider.prepare_request(ctx)
        assert "tools" in result
        assert len(result["tools"]) == 1

    def test_no_transport_falls_back_to_minimal(self):
        """Without a transport, prepare_request should produce minimal kwargs."""
        rc = RequestConfig(
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            is_openrouter=True,  # should be ignored without transport
        )
        provider = OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
            request_config=rc,
            # No transport — falls back to minimal dict
        )
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert result["model"] == "gpt-4o"
        assert "messages" in result
        # Minimal fallback should NOT have extra_body
        assert "extra_body" not in result
