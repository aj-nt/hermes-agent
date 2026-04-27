"""Core types for the Kore Orchestrator pipeline.

Phase 1: Data structures and type definitions. No behavioral change
to existing code — these types will be adopted incrementally during
cutover (Phase 5).

All types are dataclasses with explicit defaults. The goal is to
replace the 176 self.* attributes scattered across AIAgent with
typed, grouped state objects.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional


# ============================================================================
# StreamState
# ============================================================================

@dataclass
class StreamConfig:
    """Timeout and stale-detection parameters for streaming.

    Extracted from the inline constants in _interruptible_streaming_api_call.
    """

    stale_timeout: float = 30.0        # seconds with no delta before stale
    stale_char_rate: float = 0.5       # chars/sec below this is stale
    max_inactivity: float = 120.0      # hard timeout for no activity
    empty_content_max_retries: int = 3 # consecutive empty responses allowed


@dataclass
class StreamState:
    """Per-pipeline streaming accumulation state.

    Replaces: _stream_needs_break, _current_streamed_assistant_text,
    _codex_streamed_text_parts, _empty_content_retries, _StreamConfig.

    Not shared across threads — each ConversationContext owns one.
    """

    config: Optional[StreamConfig] = None
    accumulated_text: str = ""
    text_parts: list[str] = field(default_factory=list)
    empty_content_retries: int = 0
    needs_break: bool = False
    stream_callback: Optional[Callable] = None
    reasoning_callback: Optional[Callable] = None


# ============================================================================
# ProviderCapabilities
# ============================================================================

@dataclass
class ProviderCapabilities:
    """Declares what a provider supports. The orchestrator dispatches
    on capabilities, not provider identity.

    Replaces: _is_ollama_glm_backend(), _needs_kimi_tool_reasoning(),
    _is_anthropic_backend(), and other scattered _is_*() checks.
    """

    supports_streaming: bool = True
    supports_tools: bool = True
    supports_reasoning_tokens: bool = False
    requires_prompt_caching: bool = False
    requires_message_sanitization: bool = False
    max_context_tokens: Optional[int] = None
    requires_custom_stop_handling: bool = False
    supports_responses_api: bool = False
    cache_breakpoint_strategy: str = "none"  # "anthropic_4point", "anthropic_3point", "none"


# ============================================================================
# UsageInfo
# ============================================================================

@dataclass
class UsageInfo:
    """Token usage from a provider response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


# ============================================================================
# ProviderResult
# ============================================================================

@dataclass
class ProviderResult:
    """Uniform return type from provider calls.

    No more 'sometimes a dict, sometimes a stream, sometimes None'.
    Every provider call returns one of these; the pipeline stages
    can inspect it without isinstance checks.
    """

    response: Optional[dict] = None
    stream: Optional[Iterator] = None
    usage: Optional[UsageInfo] = None
    finish_reason: Optional[str] = None
    error: Optional[Exception] = None
    should_retry: bool = False
    should_fallback: bool = False


# ============================================================================
# ParsedResponse
# ============================================================================

@dataclass
class ParsedResponse:
    """Canonical output of Stage 3 (Response Processing).

    Normalizes provider-specific response formats into a single structure.
    Built from the raw ProviderResult by Stage 3 logic.
    """

    message: Optional[dict] = None
    tool_calls: list[dict] = field(default_factory=list)
    reasoning_content: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[UsageInfo] = None

    @property
    def has_tool_calls(self) -> bool:
        """True if the response contains tool calls to dispatch."""
        return len(self.tool_calls) > 0


# ============================================================================
# ConversationContext
# ============================================================================

@dataclass
class ConversationContext:
    """Per-turn state passed through the pipeline.

    The Orchestrator loop creates one ConversationContext per turn.
    Single-writer — only the pipeline stages mutate it, not other threads.

    Replaces the scatter of self.* attributes that were read and written
    across 146 methods of AIAgent.
    """

    session_id: str = ""

    # --- Conversation ---
    messages: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    iteration: int = 0
    max_iterations: int = 90
    tools: list[dict] = field(default_factory=list)

    # --- Streaming ---
    stream_state: StreamState = field(default_factory=StreamState)
    stream_needs_break: bool = False

    # --- Interruption ---
    interrupt_event: threading.Event = field(default_factory=threading.Event)

    # --- Steering ---
    steer_text: Optional[str] = None

    # --- Session bookkeeping ---
    ephemeral_system_prompt: Optional[str] = None
    api_call_count: int = 0
    tool_results_pending: bool = False

    # --- Checkpointing ---
    checkpoint_mgr: Any = None  # CheckpointManager instance

    # --- Auxiliary routing ---
    auxiliary_runtime: Any = None  # AuxiliaryRuntime instance

    # --- Rate limits ---
    rate_limits: Any = None  # RateLimitState instance

    # --- Activity tracking ---
    activity: Any = None  # ActivityTracker instance

    # --- Callbacks (set by gateway per-turn) ---
    stream_callback: Optional[Callable] = None


# ============================================================================
# SessionState
# ============================================================================

@dataclass
class SessionState:
    """Typed, grouped state for a single conversation session.

    Lives across multiple turns. Grouped by domain; all mutations go
    through methods (to be added in later phases as we cut over).

    Replaces 176 unique self.X attributes across 491 assignments
    in AIAgent.__init__ and run_conversation.
    """

    session_id: str = ""

    # --- Conversation ---
    messages: list[dict] = field(default_factory=list)
    system_prompt: Optional[str] = None
    cached_system_prompt: Optional[str] = None
    ephemeral_system_prompt: Optional[str] = None

    # --- Model/Provider ---
    active_model: Any = None           # ModelConfig (Phase 2)
    credential_pool: Any = None        # CredentialPool (Phase 2)
    fallback_chain: Any = None         # FallbackChain (Phase 2)
    fallback_activated: bool = False

    # --- Provider Clients ---
    primary_client: Any = None
    anthropic_client: Any = None
    codex_client: Any = None
    auxiliary_runtime: Any = None      # AuxiliaryRuntime instance

    # --- Lifecycle ---
    interrupt_event: threading.Event = field(default_factory=threading.Event)
    steer_buffer: Optional[str] = None
    iteration_count: int = 0
    api_call_count: int = 0

    # --- Rate Limits ---
    rate_limits: Any = None            # RateLimitState

    # --- Streaming ---
    stream_state: StreamState = field(default_factory=StreamState)

    # --- Session Persistence ---
    session_db: Any = None             # SessionDB instance
    session_title: Optional[str] = None

    # --- Checkpointing ---
    checkpoint_mgr: Any = None        # CheckpointManager instance

    # --- Memory ---
    memory_coord: Any = None           # MemoryCoordinator (Phase 1+)
    is_memory_enabled: bool = False
    is_user_profile_enabled: bool = False

    # --- Activity Tracking ---
    last_activity_ts: float = 0.0
    last_activity_desc: str = ""
    tool_progress_callback: Optional[Callable] = None
    step_callback: Optional[Callable] = None
    status_callback: Optional[Callable] = None
    background_review_callback: Optional[Callable] = None

    # --- Delegate Tracking ---
    active_children: set = field(default_factory=set)
    delegate_depth: int = 0

    # --- Prompt Caching ---
    use_native_cache_layout: bool = False

    # --- Usage Accounting ---
    session_estimated_cost_usd: float = 0.0
    session_cost_status: str = ""

    # --- Provider-Specific ---
    pending_steer: list[dict] = field(default_factory=list)
    empty_content_retries: int = 0