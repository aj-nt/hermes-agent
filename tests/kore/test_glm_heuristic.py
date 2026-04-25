"""
Tests for glm_heuristic module.

Verify the extracted Ollama/GLM stop-to-length heuristic functions
produce identical results to the original AIAgent methods.
"""

import pytest

from agent.kore.config import ProviderConfig
from agent.kore.glm_heuristic import (
    is_ollama_glm_backend,
    should_treat_stop_as_truncated,
)


def _make_config(**overrides):
    """Create a minimal ProviderConfig for testing."""
    defaults = dict(
        model="glm-4",
        provider="ollama",
        base_url="http://localhost:11434",
        api_mode="chat_completions",
    )
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class TestIsOllamaGlmBackend:
    """Tests for is_ollama_glm_backend()."""

    def test_glm_on_ollama(self):
        config = _make_config(model="glm-4", base_url="http://localhost:11434")
        assert is_ollama_glm_backend(config) is True

    def test_glm_on_port_11434(self):
        config = _make_config(model="glm-4", base_url="http://192.168.1.28:11434")
        assert is_ollama_glm_backend(config) is True

    def test_glm_on_localhost_no_port(self):
        config = _make_config(model="glm-4", base_url="http://localhost:8000")
        assert is_ollama_glm_backend(config) is True

    def test_zai_provider_local(self):
        config = _make_config(model="not-glm", provider="zai", base_url="http://localhost:11434")
        assert is_ollama_glm_backend(config) is True

    def test_non_glm_non_ollama(self):
        config = _make_config(model="gpt-4", provider="openai", base_url="https://api.openai.com")
        assert is_ollama_glm_backend(config) is False

    def test_glm_on_remote_api(self):
        config = _make_config(model="glm-4", provider="openrouter", base_url="https://openrouter.ai")
        assert is_ollama_glm_backend(config) is False

    def test_zai_on_remote(self):
        config = _make_config(model="not-glm", provider="zai", base_url="https://api.example.com")
        assert is_ollama_glm_backend(config) is False

    def test_empty_strings(self):
        config = _make_config(model="", provider="", base_url="")
        assert is_ollama_glm_backend(config) is False

    def test_none_values(self):
        config = ProviderConfig()  # all None
        assert is_ollama_glm_backend(config) is False


class TestShouldTreatStopAsTruncated:
    """Tests for should_treat_stop_as_truncated()."""

    def test_non_stop_finish_reason(self):
        config = _make_config()
        assert should_treat_stop_as_truncated(config, "length", None) is False

    def test_non_chat_completions_mode(self):
        config = _make_config(api_mode="responses")
        assert should_treat_stop_as_truncated(config, "stop", None) is False

    def test_non_ollama_backend(self):
        config = _make_config(model="gpt-4", provider="openai", base_url="https://api.openai.com")
        assert should_treat_stop_as_truncated(config, "stop", None) is False

    def test_no_tool_messages(self):
        config = _make_config()
        msg = type("Msg", (), {"content": "a test response here", "tool_calls": None})()
        assert should_treat_stop_as_truncated(config, "stop", msg, []) is False

    def test_with_tool_messages_short_response(self):
        config = _make_config()
        msg = type("Msg", (), {"content": "ok", "tool_calls": None})()
        messages = [{"role": "tool", "content": "result"}]
        # Too short (<20 chars), no natural ending
        assert should_treat_stop_as_truncated(config, "stop", msg, messages) is False

    def test_with_tool_messages_truncated_response(self):
        config = _make_config()
        content = "This is a truncated response that keeps going without"
        msg = type("Msg", (), {"content": content, "tool_calls": None})()
        messages = [{"role": "tool", "content": "result"}]
        # No natural ending + tool messages + GLM backend = truncated
        assert should_treat_stop_as_truncated(config, "stop", msg, messages) is True

    def test_with_tool_messages_complete_response(self):
        config = _make_config()
        content = "This is a complete response with proper ending."
        msg = type("Msg", (), {"content": content, "tool_calls": None})()
        messages = [{"role": "tool", "content": "result"}]
        # Has natural ending (period) = not truncated
        assert should_treat_stop_as_truncated(config, "stop", msg, messages) is False

    def test_with_tool_calls_is_not_truncated(self):
        config = _make_config()
        msg = type("Msg", (), {
            "content": "incomplete response",
            "tool_calls": [{"id": "1"}],
        })()
        messages = [{"role": "tool", "content": "result"}]
        # Tool calls present = not a truncated text response
        assert should_treat_stop_as_truncated(config, "stop", msg, messages) is False

    def test_none_assistant_message(self):
        config = _make_config()
        assert should_treat_stop_as_truncated(config, "stop", None) is False


class TestGlmHeuristicBackwardCompat:
    """Verify extracted functions match AIAgent method behavior."""

    @pytest.fixture
    def agent(self):
        from run_agent import AIAgent
        from agent.kore.config import ProviderConfig
        a = AIAgent.__new__(AIAgent)
        # Set attributes needed by _is_ollama_glm_backend
        a.model = "glm-4"
        a.provider = "ollama"
        a.base_url = "http://localhost:11434"
        a._base_url_lower = "http://localhost:11434"
        a.api_mode = "chat_completions"
        a._provider_config = ProviderConfig(
            model="glm-4",
            provider="ollama",
            base_url="http://localhost:11434",
            api_mode="chat_completions",
        )
        return a

    def test_is_ollama_glm_backend_matches_agent(self, agent):
        config = _make_config()
        extracted = is_ollama_glm_backend(config)
        original = agent._is_ollama_glm_backend()
        assert extracted == original

    def test_is_ollama_glm_backend_non_glm_matches(self, agent):
        from agent.kore.config import ProviderConfig
        config = _make_config(model="gpt-4", provider="openai", base_url="https://api.openai.com")
        extracted = is_ollama_glm_backend(config)
        agent.model = "gpt-4"
        agent.provider = "openai"
        agent.base_url = "https://api.openai.com"
        agent._base_url_lower = "https://api.openai.com"
        agent._provider_config = ProviderConfig(
            model="gpt-4", provider="openai", base_url="https://api.openai.com",
        )
        original = agent._is_ollama_glm_backend()
        assert extracted == original
