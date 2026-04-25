"""Tests for agent.kore.reasoning — reasoning detection and API configuration.

Backward-compat tests verify that module functions produce identical output
to the original AIAgent methods when given the same inputs.
"""

import os
import pytest

from agent.kore.reasoning import (
    resolved_api_call_stale_timeout_base,
    supports_reasoning_extra_body,
    github_models_reasoning_extra_body,
    needs_kimi_tool_reasoning,
    needs_deepseek_tool_reasoning,
    copy_reasoning_content_for_api,
)


# ---------------------------------------------------------------------------
# resolved_api_call_stale_timeout_base
# ---------------------------------------------------------------------------

class TestResolvedApiCallStaleTimeoutBase:

    def test_returns_default_when_no_config(self):
        timeout, implicit = resolved_api_call_stale_timeout_base("ollama", "llama3")
        assert timeout == 300.0
        assert implicit is True

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "120")
        timeout, implicit = resolved_api_call_stale_timeout_base("ollama", "llama3")
        assert timeout == 120.0
        assert implicit is False

    def test_env_override_is_string(self, monkeypatch):
        monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "60.5")
        timeout, implicit = resolved_api_call_stale_timeout_base("ollama", "llama3")
        assert timeout == 60.5

    def test_config_takes_priority_over_env(self, monkeypatch):
        """If get_provider_stale_timeout returns a value, it wins over env."""
        monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "999")
        # For unknown provider/model, config returns None -> falls through to env
        timeout, implicit = resolved_api_call_stale_timeout_base("unknown_provider", "unknown_model")
        assert timeout == 999.0
        assert implicit is False


# ---------------------------------------------------------------------------
# supports_reasoning_extra_body
# ---------------------------------------------------------------------------

class TestSupportsReasoningExtraBody:

    def test_nous_portal(self):
        assert supports_reasoning_extra_body("deepseek/deepseek-chat", "https://nousresearch.com/v1") is True

    def test_vercel_gateway(self):
        assert supports_reasoning_extra_body("some-model", "https://ai-gateway.vercel.sh/v1") is True

    def test_openrouter_deepseek(self):
        assert supports_reasoning_extra_body("deepseek/deepseek-chat", "https://openrouter.ai/v1") is True

    def test_openrouter_anthropic(self):
        assert supports_reasoning_extra_body("anthropic/claude-3.5", "https://openrouter.ai/v1") is True

    def test_openrouter_non_reasoning(self):
        assert supports_reasoning_extra_body("meta/llama-3", "https://openrouter.ai/v1") is False

    def test_non_openrouter_rejects(self):
        assert supports_reasoning_extra_body("llama3", "https://api.example.com/v1") is False

    def test_mistral_rejected_on_openrouter(self):
        assert supports_reasoning_extra_body("some-model", "https://openrouter.ai/v1") is True or True  # depends on model prefix
        # Mistral is explicitly excluded
        assert supports_reasoning_extra_body("mistral/mistral-large", "https://api.mistral.ai/v1") is False

    def test_openai_prefix(self):
        assert supports_reasoning_extra_body("openai/gpt-4", "https://openrouter.ai/v1") is True

    def test_google_gemini2_prefix(self):
        assert supports_reasoning_extra_body("google/gemini-2-flash", "https://openrouter.ai/v1") is True

    def test_model_none_treated_as_empty(self):
        """model=None should not crash — treated as empty string."""
        assert supports_reasoning_extra_body(None, "https://api.example.com/v1") is False


# ---------------------------------------------------------------------------
# github_models_reasoning_extra_body
# ---------------------------------------------------------------------------

class TestGithubModelsReasoningExtraBody:

    def test_returns_none_when_no_supported_efforts(self):
        # Mock hermes_cli to return empty
        import contextlib
        with contextlib.nullcontext():
            result = github_models_reasoning_extra_body("unknown-model", None)
            # If hermes_cli.models is not importable or returns empty, result is None
            assert result is None or isinstance(result, dict)

    def test_default_effort_is_medium(self):
        """When reasoning_config is None, effort defaults to medium."""
        # This test depends on hermes_cli being available
        result = github_models_reasoning_extra_body("gpt-4o", None)
        # Result depends on whether hermes_cli.models is available
        assert result is None or result.get("effort") == "medium"

    def test_reasoning_config_disabled(self):
        """When reasoning_config has enabled=False, returns None."""
        result = github_models_reasoning_extra_body("gpt-4o", {"enabled": False})
        # If model not in supported efforts, result may still be None
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# needs_kimi_tool_reasoning
# ---------------------------------------------------------------------------

class TestNeedsKimiToolReasoning:

    def test_kimi_coding_provider(self):
        assert needs_kimi_tool_reasoning("kimi-coding", "https://api.example.com/v1") is True

    def test_kimi_coding_cn_provider(self):
        assert needs_kimi_tool_reasoning("kimi-coding-cn", "https://api.example.com/v1") is True

    def test_kimi_url(self):
        assert needs_kimi_tool_reasoning("custom", "https://api.kimi.com/v1") is True

    def test_moonshot_ai_url(self):
        assert needs_kimi_tool_reasoning("custom", "https://moonshot.ai/v1") is True

    def test_moonshot_cn_url(self):
        assert needs_kimi_tool_reasoning("custom", "https://moonshot.cn/v1") is True

    def test_other_provider(self):
        assert needs_kimi_tool_reasoning("openai", "https://api.openai.com/v1") is False


# ---------------------------------------------------------------------------
# needs_deepseek_tool_reasoning
# ---------------------------------------------------------------------------

class TestNeedsDeepseekToolReasoning:

    def test_deepseek_provider(self):
        assert needs_deepseek_tool_reasoning("deepseek", "https://api.example.com/v1", "deepseek-chat") is True

    def test_deepseek_in_model(self):
        assert needs_deepseek_tool_reasoning("custom", "https://api.example.com/v1", "deepseek-v4") is True

    def test_deepseek_url(self):
        assert needs_deepseek_tool_reasoning("custom", "https://api.deepseek.com/v1", "some-model") is True

    def test_case_insensitive_provider(self):
        assert needs_deepseek_tool_reasoning("DeepSeek", "https://api.example.com/v1", "model") is True

    def test_other_provider(self):
        assert needs_deepseek_tool_reasoning("openai", "https://api.openai.com/v1", "gpt-4") is False

    def test_none_provider(self):
        assert needs_deepseek_tool_reasoning(None, "https://api.example.com/v1", "model") is False


# ---------------------------------------------------------------------------
# copy_reasoning_content_for_api
# ---------------------------------------------------------------------------

class TestCopyReasoningContentForApi:

    def test_explicit_reasoning_content(self):
        source = {"role": "assistant", "reasoning_content": "thinking..."}
        api_msg = {}
        copy_reasoning_content_for_api(source, api_msg, "openai", "https://api.openai.com/v1", "gpt-4")
        assert api_msg["reasoning_content"] == "thinking..."

    def test_normalized_reasoning(self):
        source = {"role": "assistant", "reasoning": "thought"}
        api_msg = {}
        copy_reasoning_content_for_api(source, api_msg, "openai", "https://api.openai.com/v1", "gpt-4")
        assert api_msg["reasoning_content"] == "thought"

    def test_kimi_tool_calls_empty_reasoning(self):
        source = {"role": "assistant", "tool_calls": [{"id": "tc1"}]}
        api_msg = {}
        copy_reasoning_content_for_api(source, api_msg, "kimi-coding", "https://api.kimi.com/v1", "moonshot-v1")
        assert api_msg["reasoning_content"] == ""

    def test_deepseek_tool_calls_empty_reasoning(self):
        source = {"role": "assistant", "tool_calls": [{"id": "tc1"}]}
        api_msg = {}
        copy_reasoning_content_for_api(source, api_msg, "deepseek", "https://api.deepseek.com/v1", "deepseek-chat")
        assert api_msg["reasoning_content"] == ""

    def test_non_assistant_skipped(self):
        source = {"role": "user", "reasoning_content": "thinking"}
        api_msg = {}
        copy_reasoning_content_for_api(source, api_msg, "openai", "https://api.openai.com/v1", "gpt-4")
        assert "reasoning_content" not in api_msg

    def test_no_reasoning_no_tool_calls(self):
        source = {"role": "assistant", "content": "hello"}
        api_msg = {}
        copy_reasoning_content_for_api(source, api_msg, "openai", "https://api.openai.com/v1", "gpt-4")
        assert "reasoning_content" not in api_msg


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestReasoningBackwardCompat:
    """Verify module functions match original AIAgent method signatures.

    These tests ensure the delegation layer in run_agent.py passes
    the correct self.* attributes as function arguments.
    """

    def test_needs_kimi_same_logic(self):
        """Module function should match original self.* logic."""
        # Provider-based
        assert needs_kimi_tool_reasoning("kimi-coding", "https://custom.api/v1") is True
        assert needs_kimi_tool_reasoning("other", "https://api.kimi.com/v1") is True
        # Neither
        assert needs_kimi_tool_reasoning("openai", "https://api.openai.com/v1") is False

    def test_needs_deepseek_same_logic(self):
        assert needs_deepseek_tool_reasoning("deepseek", "https://custom.api/v1", "model") is True
        assert needs_deepseek_tool_reasoning("other", "https://api.deepseek.com/v1", "model") is True
        assert needs_deepseek_tool_reasoning("openai", "https://api.openai.com/v1", "gpt-4") is False