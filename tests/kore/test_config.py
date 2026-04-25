"""
Tests for Kore configuration dataclasses.

These verify:
1. Dataclasses accept all expected fields
2. Defaults are sensible (None/empty/False)
3. AIAgent.__init__ params can be mapped 1:1 into config objects
4. ProviderConfig convenience properties work
"""

import pytest
from dataclasses import fields, asdict

from agent.kore.config import (
    ProviderConfig,
    StreamConfig,
    SessionConfig,
    AgentConfig,
)


class TestProviderConfig:
    """ProviderConfig groups connection, model, and routing params."""

    def test_all_connection_fields_default_none(self):
        cfg = ProviderConfig()
        assert cfg.base_url is None
        assert cfg.api_key is None
        assert cfg.provider is None
        assert cfg.api_mode is None

    def test_model_defaults_empty_string(self):
        cfg = ProviderConfig()
        assert cfg.model == ""

    def test_generation_params_default_none(self):
        cfg = ProviderConfig()
        assert cfg.max_tokens is None
        assert cfg.reasoning_config is None
        assert cfg.service_tier is None
        assert cfg.request_overrides is None

    def test_openrouter_fields_default_none(self):
        cfg = ProviderConfig()
        assert cfg.providers_allowed is None
        assert cfg.providers_ignored is None
        assert cfg.providers_order is None
        assert cfg.provider_sort is None
        assert cfg.provider_require_parameters is False
        assert cfg.provider_data_collection is None

    def test_fallback_defaults_none(self):
        cfg = ProviderConfig()
        assert cfg.fallback_model is None
        assert cfg.credential_pool is None

    def test_derived_state_defaults(self):
        cfg = ProviderConfig()
        assert cfg.client is None
        assert cfg.anthropic_client is None
        assert cfg.client_kwargs is None
        assert cfg.is_anthropic_oauth is False

    def test_construction_from_typical_kwargs(self):
        """Simulate AIAgent.__init__ mapping its params to ProviderConfig."""
        cfg = ProviderConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test123",
            provider="openai",
            api_mode="chat_completions",
            model="gpt-4o",
            max_tokens=4096,
            reasoning_config={"effort": "medium"},
            providers_allowed=["anthropic"],
            providers_ignored=["mistral"],
            fallback_model={"provider": "anthropic", "model": "claude-3-opus"},
        )
        assert cfg.provider == "openai"
        assert cfg.api_mode == "chat_completions"
        assert cfg.max_tokens == 4096
        assert cfg.providers_allowed == ["anthropic"]

    def test_is_openrouter_by_provider(self):
        assert ProviderConfig(provider="openrouter").is_openrouter is True
        assert ProviderConfig(provider="openai").is_openrouter is False

    def test_is_openrouter_by_base_url(self):
        assert ProviderConfig(base_url="https://openrouter.ai/api/v1").is_openrouter is True
        assert ProviderConfig(base_url="https://api.openai.com/v1").is_openrouter is False

    def test_is_anthropic_mode(self):
        assert ProviderConfig(api_mode="anthropic_messages").is_anthropic_mode is True
        assert ProviderConfig(api_mode="chat_completions").is_anthropic_mode is False

    def test_is_bedrock_mode(self):
        assert ProviderConfig(api_mode="bedrock_converse").is_bedrock_mode is True

    def test_is_codex_mode(self):
        assert ProviderConfig(api_mode="codex_responses").is_codex_mode is True

    def test_is_chat_completions_mode(self):
        assert ProviderConfig(api_mode="chat_completions").is_chat_completions_mode is True
        assert ProviderConfig(api_mode="anthropic_messages").is_chat_completions_mode is False

    def test_all_61_init_params_covered(self):
        """Verify every AIAgent param that belongs in ProviderConfig is here.

        These are the params that AIAgent.__init__ currently accepts and that
        should map to ProviderConfig:
        base_url, api_key, provider, api_mode, model, max_tokens,
        reasoning_config, service_tier, request_overrides,
        providers_allowed, providers_ignored, providers_order,
        provider_sort, provider_require_parameters,
        provider_data_collection, credential_pool, fallback_model
        """
        field_names = {f.name for f in fields(ProviderConfig)}
        expected = {
            "base_url", "api_key", "provider", "api_mode", "model",
            "max_tokens", "reasoning_config", "service_tier",
            "request_overrides",
            "providers_allowed", "providers_ignored", "providers_order",
            "provider_sort", "provider_require_parameters",
            "provider_data_collection", "credential_pool", "fallback_model",
            # Derived state
            "client", "anthropic_client", "client_kwargs",
            "is_anthropic_oauth",
        }
        assert expected == field_names, f"Missing: {expected - field_names}, Extra: {field_names - expected}"


class TestStreamConfig:
    """StreamConfig groups all callback params."""

    def test_all_callbacks_default_none(self):
        cfg = StreamConfig()
        assert cfg.tool_progress_callback is None
        assert cfg.tool_start_callback is None
        assert cfg.tool_complete_callback is None
        assert cfg.stream_delta_callback is None
        assert cfg.interim_assistant_callback is None
        assert cfg.thinking_callback is None
        assert cfg.reasoning_callback is None
        assert cfg.tool_gen_callback is None
        assert cfg.step_callback is None
        assert cfg.clarify_callback is None
        assert cfg.status_callback is None

    def test_construction_with_callbacks(self):
        cb = lambda: None
        cfg = StreamConfig(
            stream_delta_callback=cb,
            clarify_callback=cb,
            status_callback=cb,
        )
        assert cfg.stream_delta_callback is cb
        assert cfg.clarify_callback is cb

    def test_field_count_matches_init_params(self):
        """StreamConfig should have exactly the 11 callback params from __init__."""
        assert len(fields(StreamConfig)) == 11


class TestSessionConfig:
    """SessionConfig groups identity, platform, and persistence params."""

    def test_identity_defaults(self):
        cfg = SessionConfig()
        assert cfg.session_id is None
        assert cfg.parent_session_id is None

    def test_platform_defaults(self):
        cfg = SessionConfig()
        assert cfg.platform is None
        assert cfg.user_id is None
        assert cfg.chat_id is None

    def test_persistence_defaults(self):
        cfg = SessionConfig()
        assert cfg.persist_session is True
        assert cfg.skip_context_files is False
        assert cfg.skip_memory is False
        assert cfg.pass_session_id is False
        assert cfg.checkpoints_enabled is False
        assert cfg.checkpoint_max_snapshots == 50

    def test_construction_from_gateway_kwargs(self):
        """Simulate gateway creating AIAgent with session params."""
        cfg = SessionConfig(
            session_id="abc123",
            platform="telegram",
            user_id="1234567890",
            chat_id="-1001234567890",
            chat_type="group",
            gateway_session_key="agent:main:telegram:group:-1001234567890",
        )
        assert cfg.platform == "telegram"
        assert cfg.gateway_session_key.startswith("agent:main:")


class TestAgentConfig:
    """AgentConfig groups behavioral/display params."""

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_iterations == 90
        assert cfg.tool_delay == 1.0
        assert cfg.quiet_mode is False
        assert cfg.verbose_logging is False
        assert cfg.save_trajectories is False
        assert cfg.enabled_toolsets is None
        assert cfg.disabled_toolsets is None
        assert cfg.log_prefix == ""

    def test_construction_from_cli_kwargs(self):
        """Simulate CLI creating AIAgent with behavioral params."""
        cfg = AgentConfig(
            max_iterations=120,
            quiet_mode=True,
            enabled_toolsets=["web", "terminal"],
        )
        assert cfg.max_iterations == 120
        assert cfg.quiet_mode is True
        assert cfg.enabled_toolsets == ["web", "terminal"]


class TestConfigMapping:
    """Verify that all 61 AIAgent.__init__ params map to exactly one config."""

    # The complete set of 61 AIAgent.__init__ params, grouped by which
    # config dataclass owns them. If a param appears here but not in any
    # config, or appears in two configs, the test fails.

    PARAM_ASSIGNMENTS = {
        "ProviderConfig": [
            "base_url", "api_key", "provider", "api_mode", "model",
            "max_tokens", "reasoning_config", "service_tier",
            "request_overrides",
            "providers_allowed", "providers_ignored", "providers_order",
            "provider_sort", "provider_require_parameters",
            "provider_data_collection", "credential_pool", "fallback_model",
        ],
        "StreamConfig": [
            "tool_progress_callback", "tool_start_callback",
            "tool_complete_callback", "thinking_callback",
            "reasoning_callback", "clarify_callback",
            "step_callback", "stream_delta_callback",
            "interim_assistant_callback", "tool_gen_callback",
            "status_callback",
        ],
        "SessionConfig": [
            "session_id", "platform", "user_id", "user_name", "chat_id",
            "chat_name", "chat_type", "thread_id", "gateway_session_key",
            "session_db", "parent_session_id", "pass_session_id",
            "persist_session", "skip_context_files", "skip_memory",
            "checkpoints_enabled", "checkpoint_max_snapshots",
        ],
        "AgentConfig": [
            "max_iterations", "iteration_budget", "tool_delay",
            "enabled_toolsets", "disabled_toolsets",
            "save_trajectories", "verbose_logging", "quiet_mode",
            "ephemeral_system_prompt", "log_prefix_chars", "log_prefix",
            "prefill_messages",
            "acp_command", "acp_args",
        ],
    }

    # Params intentionally NOT in any config (handled differently):
    # - command, args: deprecated aliases, handled in __init__ body
    EXCLUDED = {"command", "args"}

    def test_every_init_param_is_assigned(self):
        """All 61 __init__ params must map to exactly one config (or be excluded)."""
        all_assigned = set()
        for params in self.PARAM_ASSIGNMENTS.values():
            all_assigned.update(params)
        all_assigned |= self.EXCLUDED

        # The canonical 61 params from AIAgent.__init__
        init_params = {
            "base_url", "api_key", "provider", "api_mode", "acp_command",
            "acp_args", "command", "args", "model", "max_iterations",
            "tool_delay", "enabled_toolsets", "disabled_toolsets",
            "save_trajectories", "verbose_logging", "quiet_mode",
            "ephemeral_system_prompt", "log_prefix_chars", "log_prefix",
            "providers_allowed", "providers_ignored", "providers_order",
            "provider_sort", "provider_require_parameters",
            "provider_data_collection", "session_id",
            "tool_progress_callback", "tool_start_callback",
            "tool_complete_callback", "thinking_callback",
            "reasoning_callback", "clarify_callback", "step_callback",
            "stream_delta_callback", "interim_assistant_callback",
            "tool_gen_callback", "status_callback", "max_tokens",
            "reasoning_config", "service_tier", "request_overrides",
            "prefill_messages", "platform", "user_id", "user_name",
            "chat_id", "chat_name", "chat_type", "thread_id",
            "gateway_session_key", "skip_context_files", "skip_memory",
            "session_db", "parent_session_id", "iteration_budget",
            "fallback_model", "credential_pool", "checkpoints_enabled",
            "checkpoint_max_snapshots", "pass_session_id", "persist_session",
        }

        unassigned = init_params - all_assigned
        double_assigned = all_assigned - init_params - {"command", "args"}

        assert unassigned == set(), f"Params not in any config: {unassigned}"
        assert double_assigned == set(), f"Unknown params in assignments: {double_assigned}"

    def test_no_param_in_two_configs(self):
        """No param should appear in more than one config."""
        seen = {}
        for config_name, params in self.PARAM_ASSIGNMENTS.items():
            for p in params:
                if p in seen:
                    assert False, f"{p} in both {seen[p]} and {config_name}"
                seen[p] = config_name

    def test_config_dataclasses_match_assignments(self):
        """Each config class has exactly the fields listed in PARAM_ASSIGNMENTS
        (plus any derived fields that are set during init, not passed as params)."""
        config_classes = {
            "ProviderConfig": ProviderConfig,
            "StreamConfig": StreamConfig,
            "SessionConfig": SessionConfig,
            "AgentConfig": AgentConfig,
        }

        for name, cls in config_classes.items():
            field_names = {f.name for f in fields(cls)}
            assigned = set(self.PARAM_ASSIGNMENTS[name])

            # Derived/computed fields that aren't __init__ params are OK
            derived = field_names - assigned

            # But assigned fields that aren't in the dataclass are NOT OK
            missing = assigned - field_names
            assert missing == set(), f"{name}: assigned params missing from dataclass: {missing}"