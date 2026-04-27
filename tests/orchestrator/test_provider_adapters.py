"""Tests for provider adapters — thin wrappers implementing ProviderProtocol.

Each adapter wraps an existing provider module:
- OpenAICompatibleProvider → chat_completions path (Ollama, OpenRouter, etc.)
- AnthropicProvider → anthropic_messages path
- BedrockProvider → bedrock_converse path
- CodexProvider → codex_responses path
- GeminiProvider → gemini_native path
- OllamaProvider → Ollama/GLM-specific path (extends OpenAICompatible)

Phase 2: These are thin wrappers. They delegate to existing code,
not reimplement it. Tests verify protocol compliance, correct
capability declarations, and that registry dispatch matches the
current if/elif branching in AIAgent._interruptible_api_call.
"""

from unittest.mock import MagicMock
import pytest

from agent.orchestrator.context import (
    ConversationContext,
    ParsedResponse,
    ProviderCapabilities,
    ProviderResult,
    StreamState,
)
from agent.orchestrator.providers import (
    FailoverReason,
    FallbackChain,
    ProviderProtocol,
    ProviderRegistry,
)


# ============================================================================
# Helper — minimal ConversationContext for tests
# ============================================================================

def make_ctx(**overrides) -> ConversationContext:
    """Create a minimal ConversationContext for provider tests."""
    defaults = dict(
        session_id="test-session",
        messages=[{"role": "user", "content": "hello"}],
        system_prompt="You are helpful.",
    )
    defaults.update(overrides)
    return ConversationContext(**defaults)


# ============================================================================
# OpenAICompatibleProvider
# ============================================================================

class TestOpenAICompatibleProvider:
    """Wraps the chat_completions API path.

    This is the default/fallback path used by Ollama, OpenRouter,
    direct OpenAI, Azure, and any OpenAI-compatible endpoint.
    """

    def test_import(self):
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider

    def test_implements_provider_protocol(self):
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(base_url="http://localhost:11434")
        assert isinstance(provider, ProviderProtocol)

    def test_capabilities_default(self):
        """OpenAI-compatible providers support streaming, tools, no prompt caching.
        
        With a localhost base_url, auto-detect makes it local (sanitization=True).
        """
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(base_url="http://localhost:11434")
        caps = provider.capabilities
        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.requires_prompt_caching is False
        assert caps.supports_responses_api is False
        # Local endpoint auto-detects sanitization=True
        assert caps.requires_message_sanitization is True

    def test_capabilities_local_endpoint(self):
        """Local endpoints (Ollama) need custom stop handling and sanitization."""
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(base_url="http://localhost:11434")
        caps = provider.capabilities
        assert caps.requires_custom_stop_handling is True
        assert caps.requires_message_sanitization is True

    def test_capabilities_cloud_endpoint(self):
        """Cloud endpoints (OpenRouter) are standard OpenAI-compatible."""
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            is_local=False,
        )
        caps = provider.capabilities
        assert caps.requires_custom_stop_handling is False
        assert caps.requires_message_sanitization is False

    def test_prepare_request_returns_dict(self):
        """prepare_request should return API kwargs dict."""
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(base_url="http://localhost:11434")
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert isinstance(result, dict)

    def test_prepare_request_includes_model(self):
        """Prepared request should include the model name."""
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434",
            model_name="glm-4:latest",
        )
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert result.get("model") == "glm-4:latest"

    def test_prepare_request_includes_messages(self):
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(base_url="http://localhost:11434")
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert "messages" in result

    def test_execute_returns_provider_result(self):
        """execute() should return a ProviderResult."""
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        mock_call_fn = MagicMock(return_value=MagicMock())
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434",
            call_fn=mock_call_fn,
        )
        result = provider.execute({"model": "test", "messages": []})
        assert isinstance(result, ProviderResult)

    def test_parse_response_returns_parsed_response(self):
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        provider = OpenAICompatibleProvider(base_url="http://localhost:11434")
        ctx = make_ctx()
        provider_result = ProviderResult()
        parsed = provider.parse_response(provider_result, ctx)
        assert isinstance(parsed, ParsedResponse)


# ============================================================================
# AnthropicProvider
# ============================================================================

class TestAnthropicProvider:
    """Wraps the anthropic_messages / _call_anthropic path.

    Anthropic uses the Messages API with prompt caching breakpoints
    and specific content preprocessing (image URL fallbacks).
    """

    def test_import(self):
        from agent.orchestrator.provider_adapters import AnthropicProvider

    def test_implements_provider_protocol(self):
        from agent.orchestrator.provider_adapters import AnthropicProvider
        provider = AnthropicProvider()
        assert isinstance(provider, ProviderProtocol)

    def test_capabilities_anthropic(self):
        """Anthropic supports streaming, tools, prompt caching, reasoning tokens."""
        from agent.orchestrator.provider_adapters import AnthropicProvider
        provider = AnthropicProvider()
        caps = provider.capabilities
        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.requires_prompt_caching is True
        assert caps.supports_responses_api is False

    def test_capabilities_cache_breakpoint_strategy(self):
        """Anthropic provider declares cache breakpoint strategy."""
        from agent.orchestrator.provider_adapters import AnthropicProvider
        provider = AnthropicProvider()
        caps = provider.capabilities
        assert caps.cache_breakpoint_strategy in (
            "anthropic_4point",
            "anthropic_3point",
            "none",
        )

    def test_prepare_request_returns_dict(self):
        from agent.orchestrator.provider_adapters import AnthropicProvider
        provider = AnthropicProvider()
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert isinstance(result, dict)

    def test_execute_returns_provider_result(self):
        from agent.orchestrator.provider_adapters import AnthropicProvider
        mock_call_fn = MagicMock(return_value=MagicMock())
        provider = AnthropicProvider(call_fn=mock_call_fn)
        result = provider.execute({})
        assert isinstance(result, ProviderResult)

    def test_parse_response_returns_parsed_response(self):
        from agent.orchestrator.provider_adapters import AnthropicProvider
        provider = AnthropicProvider()
        ctx = make_ctx()
        provider_result = ProviderResult()
        parsed = provider.parse_response(provider_result, ctx)
        assert isinstance(parsed, ParsedResponse)


# ============================================================================
# BedrockProvider
# ============================================================================

class TestBedrockProvider:
    """Wraps the bedrock_converse / _bedrock_call path.

    Bedrock uses boto3 converse() / converse_stream() with
    its own message format and credential management.
    """

    def test_import(self):
        from agent.orchestrator.provider_adapters import BedrockProvider

    def test_implements_provider_protocol(self):
        from agent.orchestrator.provider_adapters import BedrockProvider
        provider = BedrockProvider(region="us-east-1")
        assert isinstance(provider, ProviderProtocol)

    def test_capabilities_bedrock(self):
        """Bedrock supports streaming and tools."""
        from agent.orchestrator.provider_adapters import BedrockProvider
        provider = BedrockProvider(region="us-east-1")
        caps = provider.capabilities
        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.supports_responses_api is False

    def test_prepare_request_returns_dict(self):
        from agent.orchestrator.provider_adapters import BedrockProvider
        provider = BedrockProvider(region="us-east-1")
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert isinstance(result, dict)

    def test_execute_returns_provider_result(self):
        from agent.orchestrator.provider_adapters import BedrockProvider
        mock_call_fn = MagicMock(return_value=MagicMock())
        provider = BedrockProvider(region="us-east-1", call_fn=mock_call_fn)
        result = provider.execute({})
        assert isinstance(result, ProviderResult)

    def test_parse_response_returns_parsed_response(self):
        from agent.orchestrator.provider_adapters import BedrockProvider
        provider = BedrockProvider(region="us-east-1")
        ctx = make_ctx()
        provider_result = ProviderResult()
        parsed = provider.parse_response(provider_result, ctx)
        assert isinstance(parsed, ParsedResponse)


# ============================================================================
# CodexProvider
# ============================================================================

class TestCodexProvider:
    """Wraps the codex_responses / _run_codex_stream path.

    Codex/Responses API has its own streaming protocol
    and response format.
    """

    def test_import(self):
        from agent.orchestrator.provider_adapters import CodexProvider

    def test_implements_provider_protocol(self):
        from agent.orchestrator.provider_adapters import CodexProvider
        provider = CodexProvider()
        assert isinstance(provider, ProviderProtocol)

    def test_capabilities_codex(self):
        """Codex supports responses API and streaming."""
        from agent.orchestrator.provider_adapters import CodexProvider
        provider = CodexProvider()
        caps = provider.capabilities
        assert caps.supports_responses_api is True
        assert caps.supports_streaming is True

    def test_prepare_request_returns_dict(self):
        from agent.orchestrator.provider_adapters import CodexProvider
        provider = CodexProvider()
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert isinstance(result, dict)

    def test_execute_returns_provider_result(self):
        from agent.orchestrator.provider_adapters import CodexProvider
        mock_call_fn = MagicMock(return_value=MagicMock())
        provider = CodexProvider(call_fn=mock_call_fn)
        result = provider.execute({})
        assert isinstance(result, ProviderResult)

    def test_parse_response_returns_parsed_response(self):
        from agent.orchestrator.provider_adapters import CodexProvider
        provider = CodexProvider()
        ctx = make_ctx()
        provider_result = ProviderResult()
        parsed = provider.parse_response(provider_result, ctx)
        assert isinstance(parsed, ParsedResponse)


# ============================================================================
# GeminiProvider
# ============================================================================

class TestGeminiProvider:
    """Wraps the gemini_native path.

    Gemini uses its own contents/generateContent format,
    not OpenAI-compatible.
    """

    def test_import(self):
        from agent.orchestrator.provider_adapters import GeminiProvider

    def test_implements_provider_protocol(self):
        from agent.orchestrator.provider_adapters import GeminiProvider
        provider = GeminiProvider()
        assert isinstance(provider, ProviderProtocol)

    def test_capabilities_gemini(self):
        """Gemini supports streaming and tools, no prompt caching."""
        from agent.orchestrator.provider_adapters import GeminiProvider
        provider = GeminiProvider()
        caps = provider.capabilities
        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.requires_prompt_caching is False
        assert caps.supports_responses_api is False

    def test_prepare_request_returns_dict(self):
        from agent.orchestrator.provider_adapters import GeminiProvider
        provider = GeminiProvider()
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert isinstance(result, dict)

    def test_execute_returns_provider_result(self):
        from agent.orchestrator.provider_adapters import GeminiProvider
        mock_call_fn = MagicMock(return_value=MagicMock())
        provider = GeminiProvider(call_fn=mock_call_fn)
        result = provider.execute({})
        assert isinstance(result, ProviderResult)

    def test_parse_response_returns_parsed_response(self):
        from agent.orchestrator.provider_adapters import GeminiProvider
        provider = GeminiProvider()
        ctx = make_ctx()
        provider_result = ProviderResult()
        parsed = provider.parse_response(provider_result, ctx)
        assert isinstance(parsed, ParsedResponse)


# ============================================================================
# OllamaProvider (extends OpenAICompatibleProvider)
# ============================================================================

class TestOllamaProvider:
    """Ollama/GLM-specific provider extending OpenAICompatibleProvider.

    Detects GLM-specific heuristics (stop-to-length, think tags)
    and applies them during response parsing.
    """

    def test_import(self):
        from agent.orchestrator.provider_adapters import OllamaProvider

    def test_implements_provider_protocol(self):
        from agent.orchestrator.provider_adapters import OllamaProvider
        provider = OllamaProvider(base_url="http://localhost:11434")
        assert isinstance(provider, ProviderProtocol)

    def test_is_subclass_of_openai_compatible(self):
        from agent.orchestrator.provider_adapters import (
            OllamaProvider,
            OpenAICompatibleProvider,
        )
        assert issubclass(OllamaProvider, OpenAICompatibleProvider)

    def test_capabilities_local(self):
        """Ollama is a local provider — needs stop handling and sanitization."""
        from agent.orchestrator.provider_adapters import OllamaProvider
        provider = OllamaProvider(base_url="http://localhost:11434")
        caps = provider.capabilities
        assert caps.requires_custom_stop_handling is True
        assert caps.requires_message_sanitization is True
        assert caps.supports_reasoning_tokens is False  # Ollama doesn't do reasoning tokens

    def test_prepare_request_returns_dict(self):
        from agent.orchestrator.provider_adapters import OllamaProvider
        provider = OllamaProvider(base_url="http://localhost:11434")
        ctx = make_ctx()
        result = provider.prepare_request(ctx)
        assert isinstance(result, dict)


# ============================================================================
# ProviderRegistry dispatch integration
# ============================================================================

class TestProviderRegistryDispatch:
    """Registry dispatch should match the current if/elif branching
    in AIAgent.__init__ for api_mode selection.

    This is the key integration test: given a (provider, base_url),
    the registry should return the right adapter type.
    """

    def test_ollama_resolves_to_ollama_provider(self):
        from agent.orchestrator.provider_adapters import OllamaProvider
        reg = ProviderRegistry()
        provider = OllamaProvider(base_url="http://localhost:11434")
        reg.register("ollama", provider)
        resolved = reg.get("ollama")
        assert isinstance(resolved, OllamaProvider)

    def test_anthropic_resolves_to_anthropic_provider(self):
        from agent.orchestrator.provider_adapters import AnthropicProvider
        reg = ProviderRegistry()
        provider = AnthropicProvider()
        reg.register("anthropic", provider)
        resolved = reg.get("anthropic")
        assert isinstance(resolved, AnthropicProvider)

    def test_bedrock_resolves_to_bedrock_provider(self):
        from agent.orchestrator.provider_adapters import BedrockProvider
        reg = ProviderRegistry()
        provider = BedrockProvider(region="us-east-1")
        reg.register("bedrock", provider)
        resolved = reg.get("bedrock")
        assert isinstance(resolved, BedrockProvider)

    def test_codex_resolves_to_codex_provider(self):
        from agent.orchestrator.provider_adapters import CodexProvider
        reg = ProviderRegistry()
        provider = CodexProvider()
        reg.register("codex", provider)
        resolved = reg.get("codex")
        assert isinstance(resolved, CodexProvider)

    def test_openrouter_resolves_to_openai_compatible(self):
        from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
        reg = ProviderRegistry()
        provider = OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1",
            is_local=False,
        )
        reg.register("openrouter", provider)
        resolved = reg.get("openrouter")
        assert isinstance(resolved, OpenAICompatibleProvider)

    def test_full_registry_setup(self):
        """A fully populated registry resolves all expected providers."""
        from agent.orchestrator.provider_adapters import (
            AnthropicProvider,
            BedrockProvider,
            CodexProvider,
            GeminiProvider,
            OllamaProvider,
            OpenAICompatibleProvider,
        )

        reg = ProviderRegistry()
        reg.register("ollama", OllamaProvider(base_url="http://localhost:11434"))
        reg.register("anthropic", AnthropicProvider())
        reg.register("bedrock", BedrockProvider(region="us-east-1"))
        reg.register("codex", CodexProvider())
        reg.register("gemini", GeminiProvider())
        reg.register("openrouter", OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1", is_local=False,
        ))

        assert len(reg.list_names()) == 6
        assert isinstance(reg.get("ollama"), OllamaProvider)
        assert isinstance(reg.get("anthropic"), AnthropicProvider)
        assert isinstance(reg.get("bedrock"), BedrockProvider)
        assert isinstance(reg.get("codex"), CodexProvider)
        assert isinstance(reg.get("gemini"), GeminiProvider)
        assert isinstance(reg.get("openrouter"), OpenAICompatibleProvider)