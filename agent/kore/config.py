"""
Kore configuration dataclasses.

These group AIAgent's 61 __init__ parameters into coherent objects.
AIAgent still accepts all 61 params for backward compatibility, but
internally it assembles these config objects and passes them to Kore
modules. This is the Fowler "Introduce Parameter Object" pattern —
the signature stays the same, but we gain typed, structured access.

Design notes:
- All fields default to None/empty so AIAgent can construct from its
  legacy params without breaking. Missing values flow through as None.
- Frozen=False because AIAgent mutates these during its init (e.g.
  api_mode auto-detection). Once we've fully migrated, we can freeze.
- These are pure data — no behavior. Logic lives in the modules that
  consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProviderConfig:
    """Everything a provider adapter needs to construct and call an LLM.

    Grouped from AIAgent.__init__ params:
        base_url, api_key, provider, api_mode, model, max_tokens,
        reasoning_config, service_tier, request_overrides,
        providers_allowed, providers_ignored, providers_order,
        provider_sort, provider_require_parameters,
        provider_data_collection, credential_pool, fallback_model

    Plus derived state set during __init__:
        api_mode (auto-detected), _is_anthropic_oauth,
        _anthropic_client, client, _client_kwargs
    """

    # Connection
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None
    api_mode: Optional[str] = None  # "chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"

    # Model
    model: str = ""

    # Generation parameters
    max_tokens: Optional[int] = None
    reasoning_config: Optional[Dict[str, Any]] = None
    service_tier: Optional[str] = None
    request_overrides: Optional[Dict[str, Any]] = None

    # OpenRouter routing preferences
    providers_allowed: Optional[List[str]] = None
    providers_ignored: Optional[List[str]] = None
    providers_order: Optional[List[str]] = None
    provider_sort: Optional[str] = None
    provider_require_parameters: bool = False
    provider_data_collection: Optional[str] = None

    # Fallback / credential rotation
    fallback_model: Optional[Any] = None  # Dict or List[Dict] — legacy format
    credential_pool: Optional[Any] = None

    # Derived state (set during AIAgent.__init__ after auto-detection)
    # These start as None and get populated by AIAgent._resolve_provider()
    client: Optional[Any] = None  # OpenAI client instance
    anthropic_client: Optional[Any] = None  # Anthropic client instance
    client_kwargs: Optional[Dict[str, Any]] = None  # Stored for client rebuild
    is_anthropic_oauth: bool = False

    @property
    def is_openrouter(self) -> bool:
        """True if the provider or base_url points to OpenRouter."""
        if self.provider == "openrouter":
            return True
        host = (self.base_url or "").lower()
        return "openrouter.ai" in host

    @property
    def is_anthropic_mode(self) -> bool:
        """True if using Anthropic Messages API."""
        return self.api_mode == "anthropic_messages"

    @property
    def is_bedrock_mode(self) -> bool:
        """True if using AWS Bedrock Converse API."""
        return self.api_mode == "bedrock_converse"

    @property
    def is_codex_mode(self) -> bool:
        """True if using OpenAI Responses API."""
        return self.api_mode == "codex_responses"

    @property
    def is_chat_completions_mode(self) -> bool:
        """True if using OpenAI Chat Completions API."""
        return self.api_mode == "chat_completions"


@dataclass
class StreamConfig:
    """Callback functions for streaming, tool progress, and status updates.

    Grouped from AIAgent.__init__ params:
        tool_progress_callback, tool_start_callback, tool_complete_callback,
        thinking_callback, reasoning_callback, clarify_callback,
        step_callback, stream_delta_callback, interim_assistant_callback,
        tool_gen_callback, status_callback

    These are all callables that the platform layer (CLI, TUI, gateway)
    registers to receive events during a conversation. The Kore loop and
    providers will receive this object instead of 11 separate callback
    params.
    """

    # Tool lifecycle
    tool_progress_callback: Optional[Any] = None  # callable(tool_name, args_preview)
    tool_start_callback: Optional[Any] = None     # callable(tool_name, tool_input)
    tool_complete_callback: Optional[Any] = None   # callable(tool_name, result)

    # Streaming
    stream_delta_callback: Optional[Any] = None    # callable(delta_text)
    interim_assistant_callback: Optional[Any] = None  # callable(text)
    thinking_callback: Optional[Any] = None        # callable(thinking_text)
    reasoning_callback: Optional[Any] = None      # callable(reasoning_text)
    tool_gen_callback: Optional[Any] = None        # callable(chunk)

    # Agent lifecycle
    step_callback: Optional[Any] = None           # callable(step_info)
    clarify_callback: Optional[Any] = None        # callable(question, choices) -> str
    status_callback: Optional[Any] = None          # callable(status_msg)


@dataclass
class SessionConfig:
    """Session identity and persistence configuration.

    Grouped from AIAgent.__init__ params:
        session_id, platform, user_id, user_name, chat_id, chat_name,
        chat_type, thread_id, gateway_session_key, session_db,
        parent_session_id, pass_session_id, persist_session,
        skip_context_files, skip_memory, checkpoints_enabled,
        checkpoint_max_snapshots
    """

    # Identity
    session_id: Optional[str] = None
    parent_session_id: Optional[str] = None

    # Platform context
    platform: Optional[str] = None  # "cli", "telegram", "discord", "whatsapp"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    chat_id: Optional[str] = None
    chat_name: Optional[str] = None
    chat_type: Optional[str] = None
    thread_id: Optional[str] = None
    gateway_session_key: Optional[str] = None

    # Persistence
    session_db: Optional[Any] = None  # SessionDB instance
    persist_session: bool = True
    pass_session_id: bool = False
    skip_context_files: bool = False
    skip_memory: bool = False

    # Checkpoints
    checkpoints_enabled: bool = False
    checkpoint_max_snapshots: int = 50


@dataclass
class AgentConfig:
    """Agent behavior and display configuration.

    Grouped from AIAgent.__init__ params:
        max_iterations, tool_delay, enabled_toolsets, disabled_toolsets,
        save_trajectories, verbose_logging, quiet_mode,
        ephemeral_system_prompt, log_prefix_chars, log_prefix,
        prefill_messages

    This is the "everything else" bucket — behavioral knobs that don't
    belong to provider, stream, or session config.
    """

    # Iteration / budget
    max_iterations: int = 90
    iteration_budget: Optional[Any] = None  # IterationBudget instance

    # Tool configuration
    tool_delay: float = 1.0
    enabled_toolsets: Optional[List[str]] = None
    disabled_toolsets: Optional[List[str]] = None

    # Display
    quiet_mode: bool = False
    verbose_logging: bool = False
    log_prefix_chars: int = 100
    log_prefix: str = ""

    # Trajectory / debugging
    save_trajectories: bool = False
    ephemeral_system_prompt: Optional[str] = None
    prefill_messages: Optional[List[Dict[str, Any]]] = None

    # ACP (sub-agent spawning)
    acp_command: Optional[str] = None
    acp_args: Optional[List[str]] = None