"""Tests that _set_builtin_provider_config clears stale api_key.

Regression test for #14134: when switching from one provider to another,
the old provider's api_key was left in model.api_key, causing credential
drift — the new provider would try to use the old provider's key and get
401 errors.

The helper _set_builtin_provider_config is the single source of truth
for built-in provider config updates. All model-flow functions that
set provider/base_url/api_mode should route through it.
"""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with config that has a stale api_key."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear env vars that could interfere
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    monkeypatch.delenv("STEPFUN_BASE_URL", raising=False)
    return home


def _write_config(home, model_dict):
    """Write a config.yaml with the given model dict."""
    import yaml
    config_yaml = home / "config.yaml"
    config_yaml.write_text(yaml.dump({"model": model_dict}))


def _read_model_config(home):
    """Read the model section from config.yaml."""
    import yaml
    config = yaml.safe_load((home / "config.yaml").read_text()) or {}
    return config.get("model", {})


# ── Helper-level tests ──────────────────────────────────────────────


class TestSetBuiltinProviderConfig:
    """Direct tests for _set_builtin_provider_config."""

    def test_api_key_cleared(self, config_home):
        """api_key from a previous provider must be popped."""
        _write_config(config_home, {
            "default": "old-model",
            "provider": "custom",
            "api_key": "sk-stale-key",
        })

        from hermes_cli.config import load_config
        from hermes_cli.main import _set_builtin_provider_config

        cfg = load_config()
        _set_builtin_provider_config(cfg, "openrouter", "https://openrouter.ai/v1", "chat_completions")

        model = cfg["model"]
        assert model["provider"] == "openrouter"
        assert "api_key" not in model, f"api_key should be popped, found: {model.get('api_key')}"

    def test_base_url_set_when_provided(self, config_home):
        """base_url is set when explicitly provided."""
        _write_config(config_home, {"default": "m", "provider": "old"})

        from hermes_cli.config import load_config
        from hermes_cli.main import _set_builtin_provider_config

        cfg = load_config()
        _set_builtin_provider_config(cfg, "openrouter", "https://openrouter.ai/v1", "chat_completions")

        assert cfg["model"]["base_url"] == "https://openrouter.ai/v1"

    def test_base_url_cleared_when_empty(self, config_home):
        """base_url is popped when empty string is passed (e.g. anthropic)."""
        _write_config(config_home, {
            "default": "m",
            "provider": "custom",
            "base_url": "https://stale.example.com/v1",
            "api_key": "sk-old",
        })

        from hermes_cli.config import load_config
        from hermes_cli.main import _set_builtin_provider_config

        cfg = load_config()
        _set_builtin_provider_config(cfg, "anthropic")

        assert "base_url" not in cfg["model"], "base_url should be cleared for anthropic"

    def test_api_mode_set_when_provided(self, config_home):
        """api_mode is set when explicitly provided."""
        _write_config(config_home, {"default": "m", "provider": "old"})

        from hermes_cli.config import load_config
        from hermes_cli.main import _set_builtin_provider_config

        cfg = load_config()
        _set_builtin_provider_config(cfg, "openrouter", "https://openrouter.ai/v1", "chat_completions")

        assert cfg["model"]["api_mode"] == "chat_completions"

    def test_api_mode_cleared_when_empty(self, config_home):
        """api_mode is popped when empty string is passed (e.g. kimi, stepfun)."""
        _write_config(config_home, {
            "default": "m",
            "provider": "custom",
            "api_mode": "anthropic_messages",
        })

        from hermes_cli.config import load_config
        from hermes_cli.main import _set_builtin_provider_config

        cfg = load_config()
        _set_builtin_provider_config(cfg, "stepfun", "https://api.stepfun.com/v1")

        assert "api_mode" not in cfg["model"], "api_mode should be cleared when empty"

    def test_string_model_normalized_to_dict(self, config_home):
        """A bare-string model value is normalized to a dict."""
        _write_config(config_home, "old-model-string")

        from hermes_cli.config import load_config
        from hermes_cli.main import _set_builtin_provider_config

        cfg = load_config()
        _set_builtin_provider_config(cfg, "ai-gateway", "https://gateway.ai/v1", "chat_completions")

        model = cfg["model"]
        assert isinstance(model, dict)
        assert model.get("provider") == "ai-gateway"

    def test_no_api_key_no_error(self, config_home):
        """Pop on a config without api_key should not raise."""
        _write_config(config_home, {"default": "m", "provider": "old"})

        from hermes_cli.config import load_config
        from hermes_cli.main import _set_builtin_provider_config

        cfg = load_config()
        _set_builtin_provider_config(cfg, "nous", "https://api.nous.com/v1")

        assert cfg["model"]["provider"] == "nous"
        assert "api_key" not in cfg["model"]

    def test_returns_model_dict(self, config_home):
        """The helper returns the normalized model dict for further modification."""
        _write_config(config_home, {"default": "m"})

        from hermes_cli.config import load_config
        from hermes_cli.main import _set_builtin_provider_config

        cfg = load_config()
        result = _set_builtin_provider_config(cfg, "bedrock", "https://bedrock.us-east-1.amazonaws.com")

        assert isinstance(result, dict)
        assert result["provider"] == "bedrock"
        # Can still modify before saving
        result["bedrock_region"] = "us-east-1"
        assert cfg["model"]["bedrock_region"] == "us-east-1"


# ── Integration: api_key_provider flow ──────────────────────────────


class TestApiKeyDriftOnProviderSwitch:
    """Switching from one api-key provider to another must clear the
    stale api_key from the model config dict."""

    def test_api_key_cleared_on_provider_switch(self, config_home, monkeypatch):
        """Start with model.api_key from provider A,
        switch to provider B — api_key must be popped."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        _write_config(config_home, {
            "default": "some-old-model",
            "provider": "ollama-cloud",
            "base_url": "https://api.ola.cloud/v1",
            "api_key": "sk-stale",
        })

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "some-old-model")

        model = _read_model_config(config_home)
        assert isinstance(model, dict)
        assert model.get("provider") == "zai"
        assert "api_key" not in model, (
            f"api_key should be cleared on provider switch, found: {model.get('api_key')}"
        )

    def test_api_mode_also_cleared_on_non_opencode_switch(self, config_home, monkeypatch):
        """A stale api_mode from a previous custom provider must also
        be cleared when switching to a non-opencode provider."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        _write_config(config_home, {
            "default": "custom-model",
            "provider": "custom",
            "base_url": "https://custom.old/v1",
            "api_key": "sk-stale",
            "api_mode": "anthropic_messages",
        })

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "custom-model")

        model = _read_model_config(config_home)
        assert isinstance(model, dict)
        assert "api_key" not in model
        assert "api_mode" not in model, (
            f"api_mode should be cleared for non-opencode, got: {model.get('api_mode')}"
        )

    def test_switch_preserves_default_model(self, config_home, monkeypatch):
        """The model.default should be updated to the new selection even
        when there was a stale api_key."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        _write_config(config_home, {
            "default": "old-model-from-previous-provider",
            "provider": "ollama-cloud",
            "api_key": "sk-stale",
        })

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "old-model-from-previous-provider")

        model = _read_model_config(config_home)
        assert model.get("default") == "glm-5"
        assert "api_key" not in model

    def test_no_api_key_no_error(self, config_home, monkeypatch):
        """If config has no stale api_key, switching should still work fine."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        pconfig = PROVIDER_REGISTRY.get("zai")
        if not pconfig:
            pytest.skip("zai not in PROVIDER_REGISTRY")

        _write_config(config_home, {
            "default": "old-model",
            "provider": "ollama-cloud",
        })

        monkeypatch.setenv("GLM_API_KEY", "test-key")

        from hermes_cli.main import _model_flow_api_key_provider
        from hermes_cli.config import load_config

        with patch("hermes_cli.auth._prompt_model_selection", return_value="glm-5"), \
             patch("hermes_cli.auth.deactivate_provider"), \
             patch("builtins.input", return_value=""):
            _model_flow_api_key_provider(load_config(), "zai", "old-model")

        model = _read_model_config(config_home)
        assert model.get("provider") == "zai"
        assert model.get("default") == "glm-5"
        assert "api_key" not in model