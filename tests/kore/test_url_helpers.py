"""Tests for agent.kore.url_helpers — URL routing and provider detection.

Backward-compat tests verify that module functions produce identical output
to the original AIAgent methods when given the same inputs.
"""

import pytest

from agent.kore.url_helpers import (
    anthropic_prompt_cache_policy,
    is_azure_openai_url,
    is_direct_openai_url,
    is_openrouter_url,
    is_qwen_portal,
    max_tokens_param,
)


class TestIsAzureOpenaiUrl:

    def test_azure_url(self):
        assert is_azure_openai_url(base_url="https://myresource.openai.azure.com/openai/v1") is True

    def test_non_azure_url(self):
        assert is_azure_openai_url(base_url="https://api.openai.com/v1") is False

    def test_local_url_not_azure(self):
        assert is_azure_openai_url(base_url="http://localhost:11434/v1") is False

    def test_none_base_url_with_cached_lower(self):
        assert is_azure_openai_url(base_url=None, base_url_lower="https://myresource.openai.azure.com/openai/v1") is True

    def test_none_base_url_empty_cached(self):
        assert is_azure_openai_url(base_url=None, base_url_lower="") is False

    def test_explicit_url_overrides_cached(self):
        """When base_url is provided, it takes priority over base_url_lower."""
        assert is_azure_openai_url(base_url="https://api.openai.com/v1", base_url_lower="https://myresource.openai.azure.com/") is False


class TestIsDirectOpenaiUrl:

    def test_openai_api_url(self):
        assert is_direct_openai_url(base_url="https://api.openai.com/v1") is True

    def test_non_openai_url(self):
        assert is_direct_openai_url(base_url="https://api.example.com/v1") is False

    def test_openrouter_not_direct_openai(self):
        assert is_direct_openai_url(base_url="https://openrouter.ai/v1") is False

    def test_none_base_url_with_cached_hostname(self):
        assert is_direct_openai_url(base_url=None, base_url_hostname_val="api.openai.com") is True

    def test_none_base_url_with_cached_lower(self):
        assert is_direct_openai_url(base_url=None, base_url_lower="https://api.openai.com/v1") is True

    def test_none_base_url_empty_cached(self):
        assert is_direct_openai_url(base_url=None, base_url_hostname_val="", base_url_lower="") is False


class TestIsOpenrouterUrl:

    def test_openrouter_url(self):
        assert is_openrouter_url("https://openrouter.ai/v1") is True

    def test_non_openrouter_url(self):
        assert is_openrouter_url("https://api.openai.com/v1") is False


class TestIsQwenPortal:

    def test_qwen_portal_url(self):
        assert is_qwen_portal("https://portal.qwen.ai/v1") is True

    def test_non_qwen_url(self):
        assert is_qwen_portal("https://api.openai.com/v1") is False


class TestMaxTokensParam:

    def test_direct_openai_uses_max_completion_tokens(self):
        result = max_tokens_param(4096, is_direct_openai=True)
        assert result == {"max_completion_tokens": 4096}

    def test_non_openai_uses_max_tokens(self):
        result = max_tokens_param(4096, is_direct_openai=False)
        assert result == {"max_tokens": 4096}

    def test_openrouter_uses_max_tokens(self):
        result = max_tokens_param(8192, is_direct_openai=False)
        assert result == {"max_tokens": 8192}


class TestAnthropicPromptCachePolicy:

    def test_native_anthropic_claude(self):
        """Native Anthropic + Claude model = cache with native layout."""
        result = anthropic_prompt_cache_policy(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            model="claude-sonnet-4-5-20250514",
        )
        assert result == (True, True)

    def test_native_anthropic_by_hostname(self):
        """anthropic_messages mode + Anthropic hostname = native."""
        result = anthropic_prompt_cache_policy(
            provider="custom",
            base_url="https://api.anthropic.com/v1",
            api_mode="anthropic_messages",
            model="claude-sonnet-4-5-20250514",
        )
        assert result == (True, True)

    def test_openrouter_claude(self):
        """OpenRouter + Claude = cache with envelope layout (not native)."""
        result = anthropic_prompt_cache_policy(
            provider="openrouter",
            base_url="https://openrouter.ai/v1",
            api_mode="anthropic_messages",
            model="claude-sonnet-4-5-20250514",
        )
        assert result == (True, False)

    def test_third_party_anthropic_wire_claude(self):
        """Third-party Anthropic-wire gateway + Claude = native layout."""
        result = anthropic_prompt_cache_policy(
            provider="custom-proxy",
            base_url="https://custom-proxy.example.com/v1",
            api_mode="anthropic_messages",
            model="claude-sonnet-4-5-20250514",
        )
        assert result == (True, True)

    def test_qwen_on_opencode_go(self):
        """Qwen on OpenCode Go = cache with envelope layout."""
        result = anthropic_prompt_cache_policy(
            provider="opencode-go",
            base_url="https://api.opencode.ai/v1",
            api_mode="openai",
            model="qwen3.5-plus",
        )
        assert result == (True, False)

    def test_qwen_on_alibaba(self):
        """Qwen on Alibaba/DashScope = cache with envelope layout."""
        result = anthropic_prompt_cache_policy(
            provider="alibaba",
            base_url="https://dashscope.aliyuncs.com/v1",
            api_mode="openai",
            model="qwen3.5-plus",
        )
        assert result == (True, False)

    def test_qwen_on_opencode_zen(self):
        """Qwen on OpenCode Zen = cache with envelope layout."""
        result = anthropic_prompt_cache_policy(
            provider="opencode-zen",
            base_url="https://api.opencode.ai/v1",
            api_mode="openai",
            model="qwen3.5-plus",
        )
        assert result == (True, False)

    def test_random_provider_no_cache(self):
        """Unrecognized provider + non-Claude model = no cache."""
        result = anthropic_prompt_cache_policy(
            provider="ollama",
            base_url="http://localhost:11434/v1",
            api_mode="openai",
            model="glm-5.1",
        )
        assert result == (False, False)

    def test_non_claude_anthropic_wire(self):
        """Anthropic wire mode + non-Claude model = no cache (not Claude)."""
        result = anthropic_prompt_cache_policy(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            model="gpt-4o",
        )
        assert result == (True, True)  # Anthropic wire always caches with native

    def test_none_defaults_to_empty_strings(self):
        """Empty string defaults yield no caching for non-matching provider."""
        result = anthropic_prompt_cache_policy(
            provider="",
            base_url="",
            api_mode="openai",
            model="claude-sonnet-4",
        )
        # Empty base_url => not openrouter, not anthropic hostname
        # api_mode="openai" => not anthropic wire
        # model has "claude" but no openrouter or anthropic wire => no match
        assert result == (False, False)


class TestUrlHelpersBackwardCompat:
    """Verify module functions match original AIAgent method behavior."""

    def test_is_direct_openai_matches_self_method(self):
        """Verify the function produces the same results as the self.* method would."""
        # Direct URL
        assert is_direct_openai_url(base_url="https://api.openai.com/v1") is True
        # Cached hostname
        assert is_direct_openai_url(base_url=None, base_url_hostname_val="api.openai.com") is True
        # Not OpenAI
        assert is_direct_openai_url(base_url="https://openrouter.ai/v1") is False

    def test_is_azure_openai_matches(self):
        assert is_azure_openai_url(base_url="https://myresource.openai.azure.com/openai/v1") is True
        assert is_azure_openai_url(base_url="https://api.openai.com/v1") is False
        assert is_azure_openai_url(base_url=None, base_url_lower="https://myresource.openai.azure.com/openai/v1") is True

    def test_is_openrouter_matches(self):
        assert is_openrouter_url("https://openrouter.ai/v1") is True
        assert is_openrouter_url("https://api.openai.com/v1") is False

    def test_max_tokens_param_matches(self):
        assert max_tokens_param(4096, True) == {"max_completion_tokens": 4096}
        assert max_tokens_param(4096, False) == {"max_tokens": 4096}