"""Integration test: provider resolution matches AIAgent if/elif dispatch.

The current dispatch in AIAgent.__init__ (lines 550-582) uses an
if/elif chain on (provider, base_url) to set api_mode. The new
Orchestrator uses ProviderRegistry lookup. This test verifies that
a resolve_provider_name() function produces the same result for
every (provider, base_url) combination that the current code handles.

This is the behavioral parity test mentioned in the DESIGN.md:
"Integration test: new provider path produces same behavior as old path."
"""

import pytest

from agent.orchestrator.provider_adapters import (
    AnthropicProvider,
    BedrockProvider,
    CodexProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)
from agent.orchestrator.providers import ProviderRegistry


# ============================================================================
# Provider resolver — translates (provider, base_url) to adapter name
# ============================================================================

def resolve_provider_name(provider: str, base_url: str) -> str:
    """Resolve (provider, base_url) to a provider adapter name.

    This mirrors the if/elif dispatch in AIAgent.__init__ (lines 550-582).
    Returns one of: "ollama", "anthropic", "bedrock", "codex", "gemini",
    "openai_compatible".

    This function will eventually live on the Orchestrator or a
    ProviderFactory. For now it's here for testability.
    """
    from agent.orchestrator.provider_adapters import OpenAICompatibleProvider

    url_lower = (base_url or "").lower().rstrip("/")
    # Extract hostname for checks
    try:
        from urllib.parse import urlparse
        hostname = urlparse(base_url or "").hostname or ""
    except Exception:
        hostname = ""

    # Codex / Responses API providers
    if provider in ("openai-codex", "xai"):
        return "codex"
    # Codex auto-detect by URL
    if provider == "" or provider is None:
        if hostname == "chatgpt.com" and "/backend-api/codex" in url_lower:
            return "codex"
        if hostname == "api.x.ai":
            return "codex"

    # Anthropic
    if provider == "anthropic":
        return "anthropic"
    if (provider == "" or provider is None) and hostname == "api.anthropic.com":
        return "anthropic"
    # Third-party Anthropic-compatible (e.g. MiniMax, DashScope)
    if url_lower.endswith("/anthropic"):
        return "anthropic"

    # Bedrock
    if provider == "bedrock":
        return "bedrock"
    if hostname.startswith("bedrock-runtime.") and "amazonaws.com" in hostname:
        return "bedrock"

    # Gemini native
    if provider == "gemini":
        return "gemini"

    # Ollama (local endpoint)
    if provider == "ollama" or OpenAICompatibleProvider._detect_local(base_url or ""):
        return "ollama"

    # Default: OpenAI-compatible (OpenRouter, direct OpenAI, Azure, etc.)
    return "openai_compatible"


# ============================================================================
# Behavioral parity tests
# ============================================================================

class TestProviderResolutionParity:
    """Verifies resolve_provider_name matches AIAgent.__init__ api_mode dispatch.

    Each test case corresponds to a branch in the if/elif chain.
    """

    # --- Explicit api_mode branches ---

    def test_explicit_chat_completions(self):
        """If api_mode is explicitly 'chat_completions', use OpenAI-compatible."""
        # This is the default/fallback — tested implicitly by all non-matches
        assert resolve_provider_name("", "https://api.openai.com/v1") == "openai_compatible"

    def test_explicit_provider_codex(self):
        """provider=='openai-codex' → codex."""
        assert resolve_provider_name("openai-codex", "") == "codex"

    def test_explicit_provider_xai(self):
        """provider=='xai' → codex."""
        assert resolve_provider_name("xai", "") == "codex"

    # --- Anthropic detection ---

    def test_explicit_provider_anthropic(self):
        """provider=='anthropic' → anthropic."""
        assert resolve_provider_name("anthropic", "https://api.anthropic.com") == "anthropic"

    def test_auto_detect_anthropic_by_url(self):
        """No provider, but hostname is api.anthropic.com → anthropic."""
        assert resolve_provider_name("", "https://api.anthropic.com/v1") == "anthropic"

    def test_anthropic_compatible_endpoint(self):
        """URL ending in /anthropic → anthropic (MiniMax, DashScope)."""
        assert resolve_provider_name("", "https://api.minimax.chat/v1/anthropic") == "anthropic"

    # --- Bedrock detection ---

    def test_explicit_provider_bedrock(self):
        """provider=='bedrock' → bedrock."""
        assert resolve_provider_name("bedrock", "") == "bedrock"

    def test_auto_detect_bedrock_by_url(self):
        """base_url is bedrock-runtime.<region>.amazonaws.com → bedrock."""
        assert resolve_provider_name(
            "", "https://bedrock-runtime.us-east-1.amazonaws.com"
        ) == "bedrock"

    # --- Codex auto-detection ---

    def test_codex_chatgpt_url(self):
        """chatgpt.com with /backend-api/codex → codex."""
        assert resolve_provider_name(
            "", "https://chatgpt.com/backend-api/codex"
        ) == "codex"

    def test_codex_xai_url(self):
        """api.x.ai hostname → codex."""
        assert resolve_provider_name("", "https://api.x.ai/v1") == "codex"

    # --- Ollama detection ---

    def test_ollama_localhost(self):
        """localhost:11434 → ollama."""
        assert resolve_provider_name("", "http://localhost:11434/v1") == "ollama"

    def test_ollama_explicit_provider(self):
        """provider=='ollama' → ollama."""
        assert resolve_provider_name("ollama", "http://192.168.86.28:11434") == "ollama"

    def test_ollama_192_168(self):
        """192.168.x.x address → ollama (local endpoint)."""
        assert resolve_provider_name("", "http://192.168.86.28:11434/v1") == "ollama"

    # --- Gemini ---

    def test_gemini_explicit(self):
        """provider=='gemini' → gemini."""
        assert resolve_provider_name("gemini", "") == "gemini"

    # --- Default: OpenAI-compatible ---

    def test_openrouter(self):
        """OpenRouter URL → openai_compatible."""
        assert resolve_provider_name(
            "", "https://openrouter.ai/api/v1"
        ) == "openai_compatible"

    def test_direct_openai(self):
        """api.openai.com (not codex) → openai_compatible."""
        assert resolve_provider_name(
            "", "https://api.openai.com/v1"
        ) == "openai_compatible"

    def test_custom_endpoint(self):
        """Some random cloud URL → openai_compatible."""
        assert resolve_provider_name(
            "", "https://llm.somecloud.com/v1"
        ) == "openai_compatible"

    def test_azure_endpoint(self):
        """Azure OpenAI endpoint → openai_compatible."""
        assert resolve_provider_name(
            "", "https://myresource.openai.azure.com/openai/deployments"
        ) == "openai_compatible"


# ============================================================================
# Registry + resolver integration
# ============================================================================

class TestRegistryWithResolver:
    """Building a registry from resolve_provider_name results gives
    the correct adapter type for every (provider, base_url) pair.
    """

    @staticmethod
    def _build_registry() -> ProviderRegistry:
        """Build a fully-populated registry matching the production setup."""
        reg = ProviderRegistry()
        reg.register("ollama", OllamaProvider(base_url="http://localhost:11434"))
        reg.register("anthropic", AnthropicProvider())
        reg.register("bedrock", BedrockProvider(region="us-east-1"))
        reg.register("codex", CodexProvider())
        reg.register("gemini", GeminiProvider())
        reg.register("openai_compatible", OpenAICompatibleProvider(
            base_url="https://openrouter.ai/api/v1", is_local=False,
        ))
        return reg

    # Maps adapter name → expected type
    ADAPTER_TYPES = {
        "ollama": OllamaProvider,
        "anthropic": AnthropicProvider,
        "bedrock": BedrockProvider,
        "codex": CodexProvider,
        "gemini": GeminiProvider,
        "openai_compatible": OpenAICompatibleProvider,
    }

    def test_resolve_ollama_gets_ollama_provider(self):
        reg = self._build_registry()
        name = resolve_provider_name("", "http://localhost:11434/v1")
        provider = reg.get(name)
        assert isinstance(provider, self.ADAPTER_TYPES[name])

    def test_resolve_anthropic_gets_anthropic_provider(self):
        reg = self._build_registry()
        name = resolve_provider_name("anthropic", "https://api.anthropic.com")
        provider = reg.get(name)
        assert isinstance(provider, AnthropicProvider)
        assert provider.capabilities.requires_prompt_caching is True

    def test_resolve_bedrock_gets_bedrock_provider(self):
        reg = self._build_registry()
        name = resolve_provider_name("bedrock", "")
        provider = reg.get(name)
        assert isinstance(provider, BedrockProvider)

    def test_resolve_codex_gets_codex_provider(self):
        reg = self._build_registry()
        name = resolve_provider_name("openai-codex", "")
        provider = reg.get(name)
        assert isinstance(provider, CodexProvider)
        assert provider.capabilities.supports_responses_api is True

    def test_resolve_openrouter_gets_openai_compatible(self):
        reg = self._build_registry()
        name = resolve_provider_name("", "https://openrouter.ai/api/v1")
        provider = reg.get(name)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.capabilities.requires_custom_stop_handling is False

    def test_resolve_gemini_gets_gemini_provider(self):
        reg = self._build_registry()
        name = resolve_provider_name("gemini", "")
        provider = reg.get(name)
        assert isinstance(provider, GeminiProvider)


# ============================================================================
# Capability-based dispatch (replacing _is_* checks)
# ============================================================================

class TestCapabilityBasedDispatch:
    """The new system dispatches on capabilities, not identity.

    This replaces scattered _is_*_backend() checks with
    ProviderCapabilities checks.
    """

    def test_prompt_caching_only_on_anthropic(self):
        """Only Anthropic provider declares prompt caching."""
        reg = TestRegistryWithResolver._build_registry()
        for name in reg.list_names():
            provider = reg.get(name)
            if isinstance(provider, AnthropicProvider):
                assert provider.capabilities.requires_prompt_caching is True
            else:
                assert provider.capabilities.requires_prompt_caching is False

    def test_responses_api_only_on_codex(self):
        """Only Codex provider supports the Responses API."""
        reg = TestRegistryWithResolver._build_registry()
        for name in reg.list_names():
            provider = reg.get(name)
            if isinstance(provider, CodexProvider):
                assert provider.capabilities.supports_responses_api is True
            else:
                assert provider.capabilities.supports_responses_api is False

    def test_message_sanitization_on_local_only(self):
        """Only local providers (Ollama) require message sanitization."""
        reg = TestRegistryWithResolver._build_registry()
        for name in reg.list_names():
            provider = reg.get(name)
            if isinstance(provider, OllamaProvider):
                assert provider.capabilities.requires_message_sanitization is True
            elif isinstance(provider, OpenAICompatibleProvider):
                # Cloud OpenAI-compatible does NOT need sanitization
                assert provider.capabilities.requires_message_sanitization is False

    def test_custom_stop_handling_on_local_only(self):
        """Only local providers need custom stop handling."""
        reg = TestRegistryWithResolver._build_registry()
        for name in reg.list_names():
            provider = reg.get(name)
            if isinstance(provider, OllamaProvider):
                assert provider.capabilities.requires_custom_stop_handling is True
            elif isinstance(provider, OpenAICompatibleProvider):
                assert provider.capabilities.requires_custom_stop_handling is False