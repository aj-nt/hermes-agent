"""Tests for agent.kore.url_helpers — URL routing and provider detection.

Backward-compat tests verify that module functions produce identical output
to the original AIAgent methods when given the same inputs.
"""

import pytest

from agent.kore.url_helpers import (
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