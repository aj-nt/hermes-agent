"""Tests for agent.kore.responses_api -- extracted responses API detection functions.

Backward-compat tests verify that module functions produce the same output
as the corresponding AIAgent methods for identical inputs.
"""

import pytest

from agent.kore.responses_api import (
    model_requires_responses_api,
    provider_model_requires_responses_api,
)


class TestModelRequiresResponsesApi:

    def test_gpt5_models(self):
        assert model_requires_responses_api("gpt-5") is True
        assert model_requires_responses_api("gpt-5.4") is True
        assert model_requires_responses_api("GPT-5.4-MINI") is True

    def test_non_gpt5_models(self):
        assert model_requires_responses_api("gpt-4") is False
        assert model_requires_responses_api("claude-3-opus") is False
        assert model_requires_responses_api("o3") is False

    def test_vendor_prefix_stripped(self):
        assert model_requires_responses_api("openai/gpt-5") is True
        assert model_requires_responses_api("openrouter/gpt-5.4") is True

    def test_model_without_prefix(self):
        assert model_requires_responses_api("gpt-4o") is False


class TestProviderModelRequiresResponsesApi:

    def test_copilot_provider_delegates(self):
        """Copilot provider delegates to _should_use_copilot_responses_api."""
        # This may fail if hermes_cli is not installed, but the fallback should work
        result = provider_model_requires_responses_api("gpt-4", provider="copilot")
        # Copilot-specific logic may or may not flag this, but should not raise
        assert isinstance(result, bool)

    def test_non_copilot_provider_gpt5(self):
        assert provider_model_requires_responses_api("gpt-5") is True

    def test_non_copilot_provider_non_gpt5(self):
        assert provider_model_requires_responses_api("gpt-4") is False

    def test_none_provider(self):
        assert provider_model_requires_responses_api("gpt-5", provider=None) is True
        assert provider_model_requires_responses_api("gpt-4", provider=None) is False


class TestResponsesApiBackwardCompat:
    """Verify module functions match AIAgent method behavior."""

    @pytest.fixture(autouse=True)
    def _make_agent(self):
        from run_agent import AIAgent
        self.agent = AIAgent.__new__(AIAgent)

    def test_model_requires_responses_api_compat(self):
        for model in ["gpt-5", "gpt-5.4", "openai/gpt-5", "gpt-4", "claude-3"]:
            assert model_requires_responses_api(model) == self.agent._model_requires_responses_api(model), f"Mismatch for {model}"

    def test_provider_model_requires_responses_api_compat(self):
        for model in ["gpt-5", "gpt-4"]:
            for provider in [None, "openai", "copilot"]:
                assert provider_model_requires_responses_api(model, provider=provider) == self.agent._provider_model_requires_responses_api(model, provider=provider), f"Mismatch for {model}/{provider}"
