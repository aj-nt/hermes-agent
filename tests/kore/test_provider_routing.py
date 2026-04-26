"""Tests for provider routing/detection functions in agent.kore.url_helpers.

Backward-compat tests verify that the extracted kore functions produce
the same results as the AIAgent methods that delegate to them.
"""

import pytest
from unittest.mock import MagicMock

from agent.kore.url_helpers import (
    is_direct_openai_url,
    is_openrouter_url,
    is_qwen_portal,
    is_azure_openai_url,
    max_tokens_param,
    anthropic_preserve_dots,
    should_sanitize_tool_calls,
)


class TestAnthropicPreserveDots:
    """Pure function tests for anthropic_preserve_dots."""

    def test_alibaba_provider(self):
        assert anthropic_preserve_dots(provider="alibaba", base_url="") is True

    def test_minimax_provider(self):
        assert anthropic_preserve_dots(provider="minimax", base_url="") is True

    def test_minimax_cn_provider(self):
        assert anthropic_preserve_dots(provider="minimax-cn", base_url="") is True

    def test_opencode_go_provider(self):
        assert anthropic_preserve_dots(provider="opencode-go", base_url="") is True

    def test_opencode_zen_provider(self):
        assert anthropic_preserve_dots(provider="opencode-zen", base_url="") is True

    def test_zai_provider(self):
        assert anthropic_preserve_dots(provider="zai", base_url="") is True

    def test_bedrock_provider(self):
        assert anthropic_preserve_dots(provider="bedrock", base_url="") is True

    def test_dashscope_base_url(self):
        assert anthropic_preserve_dots(provider="", base_url="https://dashscope.aliyuncs.com/v1") is True

    def test_aliyuncs_base_url(self):
        assert anthropic_preserve_dots(provider="", base_url="https://dashscope.aliyuncs.com/v1") is True

    def test_minimax_base_url(self):
        assert anthropic_preserve_dots(provider="", base_url="https://api.minimax.chat/v1") is True

    def test_opencode_zen_base_url(self):
        assert anthropic_preserve_dots(provider="", base_url="https://opencode.ai/zen/models") is True

    def test_bigmodel_base_url(self):
        assert anthropic_preserve_dots(provider="", base_url="https://open.bigmodel.cn/api/paas/v4") is True

    def test_bedrock_runtime_base_url(self):
        assert anthropic_preserve_dots(provider="", base_url="https://bedrock-runtime.us-east-1.amazonaws.com") is True

    def test_openai_no_preserve(self):
        assert anthropic_preserve_dots(provider="openai", base_url="https://api.openai.com/v1") is False

    def test_anthropic_no_preserve(self):
        assert anthropic_preserve_dots(provider="anthropic", base_url="https://api.anthropic.com") is False

    def test_ollama_no_preserve(self):
        assert anthropic_preserve_dots(provider="ollama", base_url="http://localhost:11434") is False

    def test_case_insensitive_provider(self):
        assert anthropic_preserve_dots(provider="Alibaba", base_url="") is True
        assert anthropic_preserve_dots(provider="MiniMax", base_url="") is True

    def test_none_provider(self):
        assert anthropic_preserve_dots(provider=None, base_url="") is False

    def test_none_base_url(self):
        assert anthropic_preserve_dots(provider="", base_url=None) is False


class TestShouldSanitizeToolCalls:
    """Pure function tests for should_sanitize_tool_calls."""

    def test_codex_responses_no_sanitize(self):
        assert should_sanitize_tool_calls(api_mode="codex_responses") is False

    def test_chat_completions_sanitize(self):
        assert should_sanitize_tool_calls(api_mode="chat_completions") is True

    def test_anthropic_sanitize(self):
        assert should_sanitize_tool_calls(api_mode="anthropic_messages") is True

    def test_bedrock_sanitize(self):
        assert should_sanitize_tool_calls(api_mode="bedrock_converse") is True


class TestAnthropicPreserveDotsBackwardCompat:
    """Backward-compat: verify AIAgent method matches kore function."""

    def test_matches_kore_function(self):
        import run_agent
        agent = run_agent.AIAgent.__new__(run_agent.AIAgent)

        # Test with various provider/base_url combos that the method matches
        test_cases = [
            ("alibaba", "", True),
            ("minimax", "", True),
            ("zai", "", True),
            ("bedrock", "", True),
            ("", "https://dashscope.aliyuncs.com/v1", True),
            ("openai", "https://api.openai.com/v1", False),
            ("anthropic", "https://api.anthropic.com", False),
            ("ollama", "http://localhost:11434", False),
        ]
        for provider, base_url, expected in test_cases:
            agent.provider = provider
            agent.base_url = base_url
            result = agent._anthropic_preserve_dots()
            assert result == expected, f"provider={provider!r}, base_url={base_url!r}: expected {expected}, got {result}"
            # Also verify it matches the kore function
            assert result == anthropic_preserve_dots(provider=provider, base_url=base_url)


class TestShouldSanitizeToolCallsBackwardCompat:
    """Backward-compat: verify AIAgent method matches kore function."""

    def test_matches_kore_function(self):
        import run_agent
        agent = run_agent.AIAgent.__new__(run_agent.AIAgent)

        test_cases = [
            ("codex_responses", False),
            ("chat_completions", True),
            ("anthropic_messages", True),
        ]
        for api_mode, expected in test_cases:
            agent.api_mode = api_mode
            result = agent._should_sanitize_tool_calls()
            assert result == expected == should_sanitize_tool_calls(api_mode=api_mode)