# Kore Orchestrator: Design Specification

## Problem Statement

`AIAgent` in `run_agent.py` is a 146-method, 10,680-line god object. `run_conversation()` alone is 3,426 lines. The architectural decisions baked into it are 3 years old. Provider-specific hacks (GLM, Codex, Anthropic, Qwen, Bedrock, Ollama) are scattered through the main loop. State management is ad-hoc dicts and side effects. Error recovery is a maze of try/except blocks. Streaming was bolted on, not designed in.

We've extracted 17 kore modules (2,780 lines, 500+ tests) covering pure functions and stateless utilities. What remains is the orchestration core — a tangled, stateful pipeline that resists further surgical extraction.

**Decision: Stop extracting. Build new.**

The extracted kore modules are solid, well-tested, and form the foundation. The remaining monolith will be replaced, not refactored.

---

## Design Principles

1. **Pipeline, not spaghetti.** Request processing flows through explicit stages. Each stage has a single responsibility and a clear contract.
2. **State is explicit, never implicit.** No mutating `self.*` across 146 methods. State lives in a typed context object passed through the pipeline.
3. **Providers are plugins, not conditionals.** No `if self._is_ollama_glm_backend()` scattered through the loop. Providers declare capabilities; the orchestrator dispatches on capabilities, not identity.
4. **Streaming is first-class.** Built into the pipeline from day one, not retrofitted.
5. **Observable by default.** Every stage emits events. Logging, metrics, and debugging are structural, not after-the-fact print statements.
6. **Testable in isolation.** Each pipeline stage has a pure-function entry point testable without mocking the entire agent.
7. **Incrementally cut-over.** We build the new orchestrator alongside the old one, swap module by module, not big-bang.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Chat Interface                     │
│                 (thin CLI adapter)                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                  Orchestrator                        │
│  Receives request, drives pipeline, returns response │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
   ┌──────────┐ ┌──────────┐  ┌──────────────┐
   │ Pipeline  │ │ Pipeline  │  │  Pipeline    │
   │ Stages   │ │ Stages   │  │  Stages      │
   │ (core)   │ │ (provider)│ │  (tool exec) │
   └──────────┘ └──────────┘  └──────────────┘
          │            │                │
          ▼            ▼                ▼
   ┌──────────────────────────────────────────┐
   │            Kore Modules                  │
   │  (existing, tested, reusable)            │
   │  config, think_blocks, tool_calls,       │
   │  error_utils, display_utils, etc.        │
   └──────────────────────────────────────────┘
```

---

## Core Types

### ConversationContext

Replaces the scatter of `self.*` attributes. Single immutable-ish state object passed through the pipeline.

```python
@dataclass
class ConversationContext:
    """Single-writer, per-pipeline state. The Orchestrator loop owns this;
    no other thread mutates it during a turn."""
    session_id: str
    messages: list[dict]           # conversation history
    system_prompt: str
    model_config: ModelConfig      # active model + provider info
    iteration: int                 # current loop iteration
    max_iterations: int
    tools: list[dict]              # available tool schemas
    stream_callback: Callable | None
    interrupt_event: threading.Event
    steer_text: str | None
    rate_limits: RateLimitState
    activity: ActivityTracker

    # --- Streaming ---
    stream_state: StreamState      # deltas, partial text, stream config
    stream_needs_break: bool       # interrupt signal for active stream

    # --- Checkpointing ---
    checkpoint_mgr: CheckpointManager | None  # filesystem snapshots

    # --- Memory integration ---
    memory_coord: MemoryCoordinator  # unified memory access point

    # --- Session bookkeeping ---
    ephemeral_system_prompt: str | None  # not saved to trajectories
    api_call_count: int            # per-turn API call counter
    tool_results_pending: bool     # tool calls awaiting dispatch

    # --- Auxiliary routing ---
    auxiliary_runtime: AuxiliaryRuntime | None  # side-task client router
```

### PipelineEvent

Typed events emitted during processing. Replaces scattered `_safe_print`, `_vprint`, `_emit_status` calls.

```python
@dataclass
class PipelineEvent:
    kind: str          # "stream_delta", "tool_start", "tool_end", "error", "status", ...
    data: dict
    timestamp: float
    session_id: str
```

### ProviderResult

Uniform return type from provider calls. No more "sometimes a dict, sometimes a stream, sometimes None".

```python
@dataclass
class ProviderResult:
    response: dict | None           # parsed API response
    stream: Iterator | None         # for streaming calls
    usage: UsageInfo | None
    finish_reason: str | None
    error: Exception | None
    should_retry: bool
    should_fallback: bool
```

### StreamState

Encapsulates all streaming accumulation state. Currently scattered across `_stream_needs_break`, `_current_streamed_assistant_text`, `_codex_streamed_text_parts`, `_empty_content_retries`, `_StreamConfig`.

```python
@dataclass
class StreamState:
    """Per-pipeline streaming state. Not shared across threads."""
    config: StreamConfig             # timeout, stale detection params
    accumulated_text: str            # text built from streaming deltas
    text_parts: list[str]            # Codex-style parts (may differ from deltas)
    empty_content_retries: int       # consecutive empty responses seen
    needs_break: bool                # interrupt signal for active stream
    stream_callback: Callable | None # gateway-provided delta emitter
    reasoning_callback: Callable | None  # gateway-provided reasoning delta emitter
```

### AuxiliaryRuntime

Replaces `auxiliary_client.py`'s ad-hoc resolution logic. Each `ConversationContext` holds an `AuxiliaryRuntime` that routes side-task API calls.

```python
class AuxiliaryRuntime:
    """Routes side-task API calls (compression, title generation,
    session search, vision) through the best available backend.
    
    Resolution order (configurable):
    1. OpenRouter  (OPENROUTER_API_KEY)
    2. Nous Portal (~/.hermes/auth.json active provider)
    3. Custom endpoint (config.yaml model.base_url + OPENAI_API_KEY)
    4. Codex OAuth (Responses API via access token)
    
    Each task type can override the resolution order in config.
    """
    
    def get_client(self, task: AuxiliaryTask) -> tuple[Any, str]:
        """Return (client, model_name) for the given task type."""
        ...
    
    def get_vision_client(self) -> tuple[Any, str]:
        """Return client for image analysis tasks."""
        ...
```

```python
class AuxiliaryTask(Enum):
    COMPRESSION = "compression"
    TITLE_GENERATION = "title_generation"
    SESSION_SEARCH = "session_search"
    VISION_ANALYSIS = "vision_analysis"
    BROWSER_VISION = "browser_vision"
    WEB_EXTRACTION = "web_extraction"
```

---

## Pipeline Stages

The orchestrator drives a conversation through repeated application of this pipeline:

### Stage 1: Request Preparation
- Build system prompt via `prompt_builder` — includes memory blocks from `MemoryCoordinator.build_prompt_blocks()`, skill blocks, vault injection, checkpoint injection, ephemeral prompt
- Apply prompt caching markers if provider supports it (`ProviderCapabilities.requires_prompt_caching` → inject Anthropic-style `cache_control` breakpoints)
- Prepare messages for API (kore: `message_prep`, `sanitization`) — strip surrogates, enforce content policy, handle vision URLs
- Hydrate persistent state from session DB (todo store, checkpoint tracking)
- Select provider and build API kwargs
- Check nudge status (`MemoryCoordinator.should_nudge()`) — inject memory/skill nudge text if due
- Input: `ConversationContext`
- Output: Prepared API request + cache markers

### Stage 2: Provider Call
- Execute the API call (streaming or non-streaming)
- Handle provider-specific transport:
  - **OpenAI-compatible** (GLM, Qwen, OpenRouter, Azure): `_call_chat_completions` path
  - **Anthropic**: `_call_anthropic` path with prompt cache breakpoints and content preprocessing
  - **Bedrock**: `_bedrock_call` path with AWS credential refresh
  - **Codex/Responses API**: `_run_codex_stream` path with its own response format
- Emit streaming deltas via `EventBus` — each delta updates `StreamState.accumulated_text` and fires `stream_callback`
- Error classification (kore: `error_utils` + `agent/error_classifier.py`) → produces `FallbackDecision`
- Retry logic, credential rotation (via `CredentialManager`), rate limit capture
- Codex streaming: handle `_codex_streamed_text_parts` and Codex-specific retry patterns
- Input: API request
- Output: `ProviderResult`

### Stage 3: Response Processing
- Parse response into canonical assistant message dict (`_build_assistant_message`)
- Extract reasoning content (kore: `reasoning`, `think_blocks`)
- Detect truncation / incomplete responses (kore: `glm_heuristic`, `think_blocks`)
- Handle empty responses (`_empty_content_retries` counter)
- Detect and parse tool calls from response (`kore:tool_calls` parse + `_sanitize_tool_call_arguments`)
- Build `ParsedResponse` with normalized message dict, tool calls, reasoning content, finish reason
- Input: `ProviderResult`
- Output: `ParsedResponse`

### Stage 4: Tool Dispatch
- Detect tool calls in response
- Pre-dispatch hooks:
  - **Checkpoint**: if tool is `write_file` or `patch`, ensure checkpoint via `CheckpointManager`
  - **Steer drain**: apply pending steer messages to tool results
  - **Resource tracking**: set `_current_tool` for activity tracking
- Route to tool handlers (sequential or concurrent, via kore: `tool_scheduling`)
- Handle delegate_task spawning — create child `Orchestrator` with restricted config, shared `MemoryBus`
- Post-dispatch:
  - Reset nudge counters on memory/skill tool use (`MemoryCoordinator.record_tool_use()`)
  - Attach write metadata to memory tool calls (`WriteMetadataTracker.build_metadata()`)
  - Clean up task resources (VM/browser cleanup) for completed delegates
- Collect tool results, apply interrupt checks
- Input: `ParsedResponse` (with tool calls)
- Output: Tool results (messages to append)

### Stage 5: Context Management
- Append tool results to conversation
- Check context window limits
- Pre-compression hook: `MemoryCoordinator.on_pre_compress()` — notify external providers before compression
- Compress if needed — `context_compressor` called by `MemoryCoordinator`, uses `AuxiliaryRuntime` for the compression LLM call
- Check iteration budget (kore: `tdd_gate`)
- If max iterations reached → `_handle_max_iterations`: request summary from model, return final response
- Hydrate todo store state back from compressed context
- Persist session state to SQLite
- Input: Updated conversation
- Output: `ConversationContext` ready for next iteration OR termination signal

### Loop Decision
- If tool calls were present → loop back to Stage 1
- If assistant message is final → break, deliver response
- If max iterations → handle gracefully (kore: `tdd_gate` style termination)
- If interrupted → break with interrupt message

### Pre-Pipeline: Session Lifecycle Hooks
The Orchestrator also handles events outside the per-turn loop:
- **Session start**: `MemoryCoordinator.initialize()`, create/retrieve session from DB
- **Session rotation** (`/new`): `MemoryCoordinator.on_session_end()` — commit memory without teardown, reset `SessionState`
- **Session teardown** (CLI exit, gateway eviction): `MemoryCoordinator.shutdown()`, `ClientManager.cleanup()`, `CheckpointManager.flush()`
- **Model switch** (`/model`): Update `SessionState.active_model`, `CredentialManager`, and `ClientManager`; clear cached system prompt; prompt caching markers may change
- **Compress** (`/compress`): Force context compression, commit memory
- **Slash commands**: See Slash Commands section below

---

## Provider Protocol

Providers are no longer special-cased in the orchestrator. Each provider implements a protocol:

```python
class ProviderProtocol(Protocol):
    """Any LLM provider the orchestrator can call."""
    
    def prepare_request(self, ctx: ConversationContext) -> PreparedRequest:
        """Build provider-specific API kwargs from context."""
        ...
    
    def execute(self, request: PreparedRequest) -> ProviderResult:
        """Make the API call. Return streaming or non-streaming result."""
        ...
    
    def parse_response(self, result: ProviderResult, ctx: ConversationContext) -> ParsedResponse:
        """Convert provider response into canonical format."""
        ...
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        """Declare what this provider supports."""
        ...
```

### ProviderCapabilities

Replaces the maze of `_is_ollama_glm_backend()`, `_needs_kimi_tool_reasoning()`, etc.

```python
@dataclass
class ProviderCapabilities:
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_reasoning_tokens: bool = False
    requires_prompt_caching: bool = False
    requires_message_sanitization: bool = False
    max_context_tokens: int | None = None
    requires_custom_stop_handling: bool = False
    supports_responses_api: bool = False
```

Existing provider-specific code moves into provider implementations:
- `agent/anthropic_adapter.py` → implements `ProviderProtocol`
- `agent/bedrock_adapter.py` → implements `ProviderProtocol`
- `agent/gemini_native_adapter.py` → implements `ProviderProtocol`
- `agent/codex_responses_adapter.py` → implements `ProviderProtocol`
- New `OllamaProvider` → implements `ProviderProtocol` (absorbs GLM-specific logic from kore/glm_heuristic)
- New `OpenAICompatibleProvider` → implements `ProviderProtocol` (OpenRouter, direct OpenAI, Azure)

### Prompt Caching

Anthropic's prompt caching requires cache breakpoints to be injected into specific message positions. Currently handled by `agent/prompt_caching.py` (73 lines) with 84 references in `run_agent.py` — the breakpoints are computed during `_build_system_prompt` and `_build_api_kwargs`, then applied as `cache_control` markers on content blocks.

This is a cross-cutting concern that touches Stage 1 (system prompt assembly) and Stage 2 (API request construction). In the new design:

```python
@dataclass
class ProviderCapabilities:
    # ... existing fields ...
    supports_prompt_caching: bool = False
    cache_breakpoint_strategy: str = "none"  # "anthropic_4point", "anthropic_3point", "none"
```

**Stage 1** (`RequestPrep.prepare()`): After assembling the system prompt, call `apply_cache_markers(system_prompt_parts, provider.capabilities)` which injects `cache_control: {"type": "ephemeral"}` markers at the right breakpoints for the active provider's strategy. The number and position of breakpoints depends on the strategy (4-point for Claude 3.5+, 3-point for older models).

**Stage 2** (`ProviderCall.execute()`): The prepared request already contains the markers. No special handling needed in the provider adapter — markers are part of the content blocks.

This means `prompt_caching.py` stays as a kore module, called during Stage 1 based on `ProviderCapabilities`. No more `if self._is_anthropic_backend()` checks in the main loop.

---

## Event/Display System

Replaces `_safe_print`, `_vprint`, `_emit_status`, `_emit_warning`, `_fire_stream_delta`, `_fire_reasoning_delta`, `_fire_tool_gen_started`.

```python
class EventBus:
    """Central event bus for pipeline observability."""
    
    def emit(self, event: PipelineEvent) -> None: ...
    def subscribe(self, kind: str, handler: Callable) -> None: ...
    
    # Built-in subscribers route events to:
    # - Terminal display (existing display module)
    # - Session logging
    # - Rate limit tracking
    # - Usage accounting
    # - Debug/diagnostic output
```

This makes the display layer a *subscriber* to events rather than mixed into the pipeline logic.

---

## State Management

### SessionState (replaces scattered self.* attrs)

The current AIAgent has 176 unique `self.X` attributes set across 491 assignments. This dataclass groups them by domain. Fields are added during migration phases; initial version is sparse and grows as we cut over.

```python
class SessionState:
    """Typed, grouped state for a single conversation session.
    
    All mutations go through methods. No direct attribute writes
    from outside this class.
    """
    
    # --- Conversation ---
    messages: list[dict]
    system_prompt: str | None
    cached_system_prompt: str | None      # prompt cache, invalidated on change
    ephemeral_system_prompt: str | None    # not saved to trajectories
    
    # --- Model/Provider ---
    active_model: ModelConfig
    credential_pool: CredentialPool
    fallback_chain: FallbackChain
    fallback_activated: bool              # currently on fallback provider?
    
    # --- Provider Clients ---
    primary_client: Any                    # active OpenAI-compatible client
    anthropic_client: Any | None           # Anthropic-specific client
    codex_client: Any | None              # Codex/Responses API client
    auxiliary_runtime: AuxiliaryRuntime    # side-task client router
    
    # --- Lifecycle ---
    interrupt_event: threading.Event
    steer_buffer: str | None
    iteration_count: int
    api_call_count: int                   # per-turn, reset by gateway on cache reuse
    
    # --- Rate Limits ---
    rate_limits: RateLimitState
    
    # --- Streaming ---
    stream_state: StreamState
    
    # --- Session Persistence ---
    session_id: str
    session_db: SessionDB
    session_title: str | None             # generated title for session
    
    # --- Checkpointing ---
    checkpoint_mgr: CheckpointManager | None
    
    # --- Memory (owned by MemoryCoordinator, referenced here) ---
    memory_coord: MemoryCoordinator
    is_memory_enabled: bool
    is_user_profile_enabled: bool
    
    # --- Activity Tracking ---
    last_activity_ts: float               # for timeout detection
    last_activity_desc: str               # human-readable description
    tool_progress_callback: Callable | None
    step_callback: Callable | None
    status_callback: Callable | None
    background_review_callback: Callable | None
    
    # --- Delegate Tracking ---
    active_children: set                   # child agent task IDs
    delegate_depth: int                    # nesting level limit
    
    # --- Prompt Caching ---
    use_native_cache_layout: bool          # Anthropic prompt cache markers
    
    # --- Usage Accounting ---
    session_estimated_cost_usd: float
    session_cost_status: str
    
    # --- Provider-Specific ---
    pending_steer: list[dict]               # queued steer messages
    empty_content_retries: int             # consecutive empty-response retries
```

### Key invariant: State mutations go through methods, not direct attribute writes.

No more `self.base_url = new_url` scattered across 20 methods. State transitions are explicit and traceable.

---

## Fallback & Credential Rotation

Currently spread across `_try_activate_fallback`, `_restore_primary_runtime`, `_recover_with_credential_pool`, `_swap_credential`, and credential refresh methods.

Replaced by:

```python
class CredentialManager:
    """Owns credential lifecycle, rotation, and fallback decisions."""
    
    def get_credential(self, ctx: ConversationContext) -> Credential:
        """Get best available credential, rotating if needed."""
        ...
    
    def report_failure(self, error: Exception) -> FallbackDecision:
        """Classify error and decide: retry, rotate, or fallback provider."""
        ...
    
    def restore_primary(self) -> bool:
        """Attempt to restore primary provider after fallback."""
        ...
```

```python
class FallbackChain:
    """Ordered list of fallback providers with retry budgets."""
    
    def __init__(self, providers: list[ProviderProtocol]): ...
    
    def next(self, current: ProviderProtocol, reason: FailoverReason) -> ProviderProtocol | None:
        """Select next provider in chain based on failure reason."""
        ...
```

---

## Runtime Model Switching

Currently `switch_model()` (180 lines) performs in-place model/provider/credential switching mid-session, called by `/model` commands in CLI and gateway. This is a major state mutation — it changes the provider, client, API key, base URL, and clears cached system prompt, all while a conversation is in progress.

In the new design, model switching is a `SessionState` transition, not a method on the agent:

```python
class Orchestrator:
    def switch_model(self, new_model: str, new_provider: str, 
                     api_key: str = "", base_url: str = "", 
                     api_mode: str = "") -> None:
        """Switch model/provider in-place for a live session.
        
        Called by /model command handlers after model_switch.resolve()
        has validated and resolved credentials.
        """
        # 1. Update SessionState
        self.state.active_model = ModelConfig(
            model=new_model, provider=new_provider, ...
        )
        # 2. Rebuild CredentialManager and FallbackChain
        self.credentials = CredentialManager(self.state.active_model, ...)
        self.fallback_chain = FallbackChain(self.credentials.resolve_chain())
        # 3. Rebuild ClientManager for new provider
        self.client_mgr.release(self.state.primary_client)
        self.state.primary_client = self.client_mgr.get(new_provider, self.state)
        # 4. Clear cached system prompt (prompt structure may change)
        self.state.cached_system_prompt = None
        # 5. Rebuild AuxiliaryRuntime for new model context window
        self.state.auxiliary_runtime = AuxiliaryRuntime(self.state.active_model)
        # 6. Emit event
        self.events.emit(PipelineEvent(
            kind="model_switch",
            data={"old_model": old, "new_model": new_model},
            ...
        ))
```

Key point: model switching is the one case where `SessionState` is mutated by an external command during a live session. The `ConversationContext` for the current turn is not affected (it's already been prepared), but the next turn picks up the new model.

---

## Tool Execution

Currently `_execute_tool_calls`, `_execute_tool_calls_concurrent`, `_execute_tool_calls_sequential`, `_invoke_tool`, `_dispatch_delegate_task` — all on AIAgent with deep coupling to state.

Replaced by:

```python
class ToolExecutor:
    """Dispatches tool calls and collects results."""
    
    def execute(self, tool_calls: list[ToolCall], ctx: ConversationContext) -> list[ToolResult]:
        """Execute tool calls, respecting concurrency config."""
        ...
    
    # Internal routing:
    # - concurrent vs sequential (from ctx config)
    # - delegate_task spawning
    # - interrupt/steer handling
    # - timeout management
```

Tool definitions and schemas stay in existing kore modules. The executor just orchestrates their dispatch.

---

## Checkpointing

The current `CheckpointManager` (654 lines, 36 references in `run_agent.py`) is not just a prompt injection concern — it's an active participant in tool dispatch. It:

1. **Intercepts file-write tools** (`write_file`, `patch`) before execution, creating filesystem snapshots
2. **Tracks working directories** per tool (different dirs for terminal, write_file, etc.)
3. **Restores checkpoints** on failure or user request (`/checkpoint restore`)
4. **Flushes snapshots** on session end or graceful shutdown

The spec originally listed `checkpoint_injection.py` (86 lines) as a prompt block concern. That's only half the story. `CheckpointManager` is a stateful service that must participate in Stage 4 (Tool Dispatch):

```python
class CheckpointManager:
    """Manages filesystem snapshots for write operations.
    
    Not just prompt injection — it wraps tool execution to ensure
    safe file mutations with rollback capability.
    """
    
    enabled: bool
    max_snapshots: int
    
    def ensure_checkpoint(self, work_dir: str, label: str) -> None:
        """Create snapshot before a write operation."""
        ...
    
    def get_working_dir_for_path(self, file_path: str) -> str:
        """Resolve the working directory for a given file path."""
        ...
    
    def flush(self) -> None:
        """Write all pending snapshots to disk."""
        ...
```

**Integration with pipeline:**

- **Stage 1**: `checkpoint_injection.py` adds checkpoint guidance to system prompt (if enabled)
- **Stage 4**: Before executing `write_file` or `patch`, `ToolExecutor` calls `CheckpointManager.ensure_checkpoint()`
- **Session lifecycle**: `Orchestrator.shutdown()` calls `CheckpointManager.flush()`

The `CheckpointManager` instance lives on `SessionState` and is created during Orchestrator initialization based on config (`checkpoints_enabled`, `checkpoint_max_snapshots`).

---

## Client Lifecycle

Currently ~500 lines of `_create_openai_client`, `_close_openai_client`, `_replace_primary_openai_client`, `_ensure_primary_openai_client`, `_cleanup_dead_connections`, `_build_keepalive_http_client` plus credential refresh.

Replaced by:

```python
class ClientManager:
    """Owns HTTP client lifecycle, connection pooling, and cleanup."""
    
    def get_client(self, provider: str, ctx: ConversationContext) -> Any:
        """Get or create a client for the given provider."""
        ...
    
    def release(self, client: Any) -> None:
        """Release a client back to the pool or close it."""
        ...
    
    def cleanup(self) -> None:
        """Close all managed clients, cleanup dead connections."""
        ...
```

Most of this logic already exists in `agent/kore/client_lifecycle.py` (247 lines). We promote it to a proper class with the state it needs, instead of methods on AIAgent.

---

## Auxiliary Client Router

`agent/auxiliary_client.py` (3,445 lines) is not just a client wrapper — it's a **side-task routing system** that resolves the best available backend for each auxiliary task type. The current implementation has:

- `CodexAuxiliaryClient`, `AnthropicAuxiliaryClient`, `AsyncCodexAuxiliaryClient`, `AsyncAnthropicAuxiliaryClient` — provider-specific adapters
- Resolution chain: OpenRouter → Nous Portal → custom endpoint → Codex OAuth
- Task-specific overrides: compression, title generation, session search, vision, browser vision, web extraction
- Credential management: OAuth refresh for Codex, Nous Portal auth
- `_emit_auxiliary_failure` — surfaces side-task failures to the user

The spec said "redesigned alongside orchestrator" — here's the design:

```python
class AuxiliaryRuntime:
    """Routes side-task API calls through the best available backend.
    
    Replaces auxiliary_client.py's ad-hoc resolution logic with a
    clean protocol-based system. Each task type can override the
    default resolution order in config.
    """
    
    def __init__(self, config: AuxiliaryConfig):
        self._resolvers: list[AuxiliaryResolver] = self._build_chain(config)
        self._cache: dict[AuxiliaryTask, tuple] = {}  # task → (client, model)
        self._failures: dict[str, float] = {}           # provider → last_failure_ts
    
    def get_client(self, task: AuxiliaryTask) -> tuple[Any, str]:
        """Return (httpx_client, model_name) for the given task type.
        
        Tries each resolver in the chain until one succeeds.
        Caches the result for reuse within the session.
        """
        ...
    
    def get_vision_client(self) -> tuple[Any, str]:
        """Return client for image analysis tasks."""
        return self.get_client(AuxiliaryTask.VISION_ANALYSIS)
    
    def get_compression_client(self) -> tuple[Any, str]:
        """Return client for context compression tasks."""
        return self.get_client(AuxiliaryTask.COMPRESSION)
```

**Key design decisions:**
- The `AuxiliaryRuntime` lives on `SessionState`, not on `MemoryCoordinator`. It's a session-scoped resource.
- Compression calls still go through `context_compressor.py`, which uses `AuxiliaryRuntime` to get a client.
- Title generation, session search, and vision analysis all use `AuxiliaryRuntime.get_client()`.
- Failure reporting goes through `EventBus` (`PipelineEvent(kind="auxiliary_failure")`) instead of `_emit_auxiliary_failure`.
- The current async variants (`AsyncCodexAuxiliaryClient`, `AsyncAnthropicAuxiliaryClient`) are not replicated — Hermes uses a threaded gateway, not asyncio. Async calls are handled via `threading.Thread`.

---

## Slash Commands

Gateway currently intercepts `/new`, `/compress`, `/usage`, `/model`, `/clear`, `/checkpoint`, `/skills`, `/help`, `/quit`, `/bye` commands before they reach the agent. Some of these trigger agent-side logic:

| Command | Gateway Action | Agent Impact |
|---------|---------------|--------------|
| `/new` | Create new session, reset agent | `commit_memory_session()`, `reset_session_state()` |
| `/compress` | Force context compression | `_compress_context()`, re-summarize |
| `/model` | Switch model/provider | `switch_model()` — rebuilds clients, credentials |
| `/usage` | Display token/cost summary | Read `session_estimated_cost_usd` |
| `/clear` | Clear conversation history | Reset `messages` in SessionState |
| `/checkpoint` | Manage checkpoints | `CheckpointManager` operations |

The new design adds a **Command Layer** between gateway message receipt and the Orchestrator loop:

```python
class CommandLayer:
    """Pre-processes slash commands before they reach the Orchestrator.
    
    Some commands are gateway-local (no agent interaction needed).
    Others require Orchestrator state changes.
    """
    
    def handle(self, command: str, args: str, state: SessionState, 
               memory: MemoryCoordinator, client_mgr: ClientManager) -> CommandResult:
        """Route slash command. Returns result for gateway display + 
        optional state mutations to apply before next turn."""
        ...
```

**CommandResult** indicates whether the command was handled locally, requires a state mutation (model switch, session reset), or should be forwarded to the model as a regular message.

This replaces the scattered `if command.startswith("/new")` checks in `gateway/run.py` (11,418 lines) with a single dispatch point. Gateway calls `CommandLayer.handle()` first; if the command returns `FORWARD_TO_MODEL`, it becomes a regular user message.

---

## TodoStore

The `TodoStore` (278 lines, 42 references in `run_agent.py`) is per-session state that must survive context compression and rotation. Currently it's initialized in `AIAgent.__init__` and hydrated from conversation history via `_hydrate_todo_store()`.

The spec didn't mention TodoStore at all. It lives on `SessionState`:

```python
class SessionState:
    # ...
    todo_store: TodoStore  # in-memory task planner, hydrated from history
```

**Key concern: compression survival.** When context compression trims the message history, todo items embedded in earlier messages would be lost. `_hydrate_todo_store()` restores them by scanning the compressed messages for todo blocks. In the new design:

- Stage 5 (Context Management) calls `todo_store.snapshot()` before compression to preserve state
- After compression, Stage 1 (Request Preparation) can call `todo_store.rehydrate(compressed_messages)` if needed
- `TodoStore` is already a standalone module (`tools/todo_tool.py`) — it stays as-is, just lives on `SessionState`

---

## Session Persistence

The current system has a dual persistence model: JSON session files and a SQLite sessions table. The spec says "JSON files eliminated, SQLite-only with WAL mode." Here's what that entails:

### What Goes
- `sessions/{session_id}.json` — one file per session, redundant with SQLite
- `_persist_session()` — JSON file write
- `_flush_messages_to_session_db()` — duplicate write to SQLite

### What Stays (moved to new locations)
- `_save_session_log()` — trajectory logging (different from session persistence)
- `_clean_session_content()` — sanitize messages before persistence
- Session title generation — triggered by Orchestrator, uses `AuxiliaryRuntime`
- `_hydrate_todo_store()` — restore todo state from session DB

### Session Database Schema (new)

```sql
-- Single source of truth
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    platform TEXT,
    model TEXT,
    provider TEXT,
    created_at REAL,
    updated_at REAL,
    message_count INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0,
    metadata TEXT  -- JSON blob for extensible session metadata
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    role TEXT NOT NULL,
    content TEXT,
    metadata TEXT,  -- JSON: tool_calls, reasoning, usage, etc.
    created_at REAL,
    sequence INTEGER
);

CREATE INDEX idx_messages_session ON messages(session_id, sequence);
```

WAL mode enabled at connection creation. The `SessionDB` class (already exists) gains write methods that replace the current dual-path persistence. Migration from JSON files to SQLite-only is part of Phase 5 (Cutover).

---

## Migration Strategy

We don't big-bang. We build alongside the old code and swap in pieces.

### Phase 1: Types and Context (Week 1-2)
- Define `ConversationContext`, `PipelineEvent`, `ProviderResult`, `ProviderCapabilities`
- Define `ProviderProtocol`
- Define `SessionState` dataclass (initially sparse, grows as we migrate)
- Build `EventBus`
- **All existing code continues to work unchanged.**

### Phase 2: Provider Plugins (Week 2-4)
- Wrap `anthropic_adapter`, `bedrock_adapter`, `gemini_native_adapter`, `codex_responses_adapter` in `ProviderProtocol` adapters
- Build `OllamaProvider` and `OpenAICompatibleProvider`
- Wire providers through a `ProviderRegistry` lookup
- Build `CredentialManager` and `FallbackChain`
- **Integration test: new provider path produces same behavior as old path.**

### Phase 3: Pipeline Stages (Week 3-5)
- Implement Stage 1 (Request Prep) — draws from existing `message_prep`, `prompt_builder`, `config`
- Implement Stage 2 (Provider Call) — uses `ProviderProtocol`
- Implement Stage 3 (Response Processing) — draws from `reasoning`, `think_blocks`, `tool_calls`
- Implement Stage 4 (Tool Dispatch) — new `ToolExecutor`, uses `tool_calls` kore module
- Implement Stage 5 (Context Management) — uses `context_compressor`
- **Integration test: each stage tested in isolation, then composed.**

### Phase 4: Orchestrator Loop (Week 4-6)
- Build the `Orchestrator` class that drives Stage 1→2→3→4→5 in a loop
- Wire up `EventBus` for display/logging
- Wire up `ClientManager`
- **Integration test: full conversation through new pipeline.**

### Phase 5: Cutover (Week 6-7)
- Feature flag: `USE_NEW_PIPELINE=true`
- Route `chat()` and `run_conversation()` through new `Orchestrator`
- Run both paths in parallel for validation
- Remove old code once new path is battle-tested on QMM

### Phase 6: Cleanup (Week 7-8)
- Remove dead code from `run_agent.py`
- Remove `AIAgent` class entirely
- Rename `Orchestrator` → `AIAgent` if backward compat needed, or keep new name
- Update imports across codebase

---

## What Stays, What Goes, What's New

### Stays (kore modules, already extracted and tested)
- `agent/kore/config.py` — configuration and model metadata
- `agent/kore/think_blocks.py` — think block parsing and stripping
- `agent/kore/reasoning.py` — reasoning content extraction
- `agent/kore/tool_calls.py` — tool call parsing and sanitization
- `agent/kore/error_utils.py` — error classification and formatting
- `agent/kore/display_utils.py` — display formatting helpers
- `agent/kore/glm_heuristic.py` — GLM stop-to-length detection
- `agent/kore/url_helpers.py` — URL classification and manipulation
- `agent/kore/provider_headers.py` — provider-specific HTTP headers
- `agent/kore/vision_utils.py` — image/vision content handling
- `agent/kore/message_prep.py` — message sanitization for API calls
- `agent/kore/client_lifecycle.py` — client creation/cleanup logic (promoted to class)
- `agent/kore/tdd_gate.py` — iteration budget and termination
- `agent/kore/steer.py` — steerable conversation injection
- `agent/kore/tool_scheduling.py` — tool concurrency and scheduling
- `agent/kore/responses_api.py` — Responses API helpers
- `agent/sanitization.py` — content sanitization

### Stays (top-level modules, already separate)
- `tools/memory_tool.py` (MemoryStore) — stays, wrapped by MemoryCoordinator
- `agent/memory_manager.py` — stays, wrapped by MemoryCoordinator
- `agent/memory_provider.py` — stays, MemoryProvider protocol unchanged
- `plugins/memory.py` — stays, plugin loading path unchanged
- `agent/display.py` — terminal display (will subscribe to EventBus)
- `agent/error_classifier.py` — error classification
- `agent/context_compressor.py` — context compression (called by MemoryCoordinator)
- `agent/vault_injection.py` — vault prompt injection (called by MemoryCoordinator.build_prompt_blocks)
- `agent/checkpoint_injection.py` — checkpoint injection (called by MemoryCoordinator.build_prompt_blocks)
- `agent/credential_pool.py` / `agent/credential_sources.py` — credential management
- `agent/rate_limit_tracker.py` / `agent/nous_rate_guard.py` — rate limiting
- `agent/prompt_builder.py` — system prompt construction
- `agent/model_metadata.py` — model capability metadata
- `agent/usage_pricing.py` — usage tracking and pricing
- `agent/title_generator.py` — session title generation
- `agent/shell_hooks.py` — shell hook execution
- `agent/skill_commands.py` / `agent/skill_utils.py` / `agent/skill_preprocessing.py` — skills
- `agent/trajectory.py` — trajectory logging
- `agent/redact.py` — content redaction
- `agent/checkpoint_store.py` / `agent/checkpoint_injection.py` — checkpointing
- All provider adapters (wrapped in ProviderProtocol)
- All existing test suites

### Goes (replaced by new pipeline)
- `AIAgent.__init__` (294 lines, 176 unique attrs, 491 assignments) → `SessionState` + `Orchestrator.__init__`
- `AIAgent.run_conversation` (3,426 lines) → Pipeline stages + Loop Decision
- `AIAgent._interruptible_streaming_api_call` (841 lines) → Stage 2 (Provider Call) + `StreamState`
- `AIAgent._build_api_kwargs` (164 lines) → Stage 1 (Request Prep) + Provider implementations
- `AIAgent._build_system_prompt` (227 lines) → Stage 1 (Request Prep)
- `AIAgent._execute_tool_calls_*` (860+ lines combined) → Stage 4 (Tool Dispatch) + `ToolExecutor`
- `AIAgent._handle_max_iterations` (164 lines) → Stage 5 (Context Management) + graceful termination
- `AIAgent._build_assistant_message` (157 lines) → Stage 3 (Response Processing) + `ParsedResponse`
- `AIAgent.switch_model` (180 lines) → `Orchestrator.switch_model()` with `SessionState` transition
- `AIAgent._call` (257 lines) and `_call_chat_completions` (252 lines) → Provider adapters
- `AIAgent._run_codex_stream` (125 lines) → Codex provider adapter
- All `_create_openai_client*`, `_close_*`, `_ensure_*` (200+ lines) → `ClientManager`
- All streaming emission methods (`_safe_print`, `_vprint`, `_emit_status`, `_emit_warning`, `_fire_stream_delta`, `_fire_reasoning_delta`, `_fire_tool_gen_started`) → `EventBus`
- All `_try_activate_fallback`, `_restore_primary_runtime`, `_recover_with_credential_pool` (366 lines) → `FallbackChain` + `CredentialManager`
- All `_sanitize_*` methods (63 references) → kore `sanitization` + provider adapters
- All provider-specific conditionals scattered through the loop → `ProviderCapabilities` declarations
- `_compress_context` (117 lines) → Stage 5 via `MemoryCoordinator`
- All `_memory_*` attributes on AIAgent (18+ instances) → `MemoryCoordinator`
- `_build_memory_write_metadata()` → `WriteMetadataTracker`
- `_sync_external_memory_for_turn()` → `MemoryCoordinator.on_turn_end()`
- `shutdown_memory_provider()` → `MemoryCoordinator.shutdown()`
- `commit_memory_session()` → `MemoryCoordinator.on_session_end()`
- `_spawn_background_review()` attribute copying → `MemoryCoordinator.build_review_context()` + lightweight Orchestrator run
- Nudge counter logic scattered in `run_conversation()` → `NudgeTracker`
- System prompt memory block assembly in `_build_system_prompt()` → `MemoryCoordinator.build_prompt_blocks()`
- Prompt cache breakpoint computation in `_build_api_kwargs` → `ProviderCapabilities.cache_breakpoint_strategy` + kore `prompt_caching`
- Checkpoint interception logic in `_invoke_tool` → `ToolExecutor` pre-dispatch hook via `CheckpointManager`
- `_hydrate_todo_store` → `TodoStore.rehydrate()` on compression recovery
- `_emit_auxiliary_failure` → `EventBus` event (kind="auxiliary_failure")
- `_reset_stream_delivery_tracking`, `_record_streamed_assistant_text`, `_has_stream_consumers` → `StreamState` methods
- Slash command handling in gateway → `CommandLayer` dispatch

### New (doesn't exist yet)
- `agent/orchestrator.py` — the Orchestrator class and pipeline driver
- `agent/orchestrator/stages.py` — pipeline stage implementations
- `agent/orchestrator/context.py` — ConversationContext, SessionState, ParsedResponse, StreamState, AuxiliaryConfig, AuxiliaryTask
- `agent/orchestrator/events.py` — PipelineEvent, EventBus
- `agent/orchestrator/providers.py` — ProviderProtocol, ProviderRegistry, ProviderCapabilities
- `agent/orchestrator/fallback.py` — FallbackChain, FailoverReason, CredentialManager
- `agent/orchestrator/tools.py` — ToolExecutor (with checkpoint pre-dispatch hook)
- `agent/orchestrator/clients.py` — ClientManager
- `agent/orchestrator/memory.py` — MemoryCoordinator, NudgeTracker, WriteMetadataTracker, ReviewContext, MemoryBus, MemoryEvent
- `agent/orchestrator/auxiliary.py` — AuxiliaryRuntime, AuxiliaryResolver chain
- `agent/orchestrator/commands.py` — CommandLayer (slash command dispatch)
- `agent/orchestrator/session.py` — SessionDB extended with write methods, session persistence
- `agent/orchestrator/compat.py` — AIAgent compatibility shim (maps old attribute names to new SessionState fields)

---

## Testing Strategy

1. **Unit tests per pipeline stage.** Each stage has clear input/output types. Test in isolation with mock context.
2. **Provider adapter tests.** Each provider adapter tested against its real API contract (where possible) and against recorded fixtures.
3. **Integration tests.** Full conversation loop with recorded API responses.
4. **Behavioral parity tests.** For each cutover phase, run old and new paths with identical inputs, assert output equivalence.
5. **Kore tests remain untouched.** They're our safety net.

---

## Memory System

Memory in hermes-agent is not a "call this API" afterthought — it has 131 touchpoints in `run_agent.py` alone and participates in every pipeline stage. The new design must make memory a first-class pipeline concern.

### Current Memory Architecture

Two parallel memory systems exist today, both wired directly into `AIAgent.__init__` and scattered throughout `run_conversation`:

**Built-in MemoryStore** (`tools/memory_tool`):
- SQLite-backed, four-layer architecture (hot/warm/cold/recent-context)
- Injected into system prompt at `_build_system_prompt()` time
- Three sections: `memory` (agent notes), `user` (user profile), `recent_context` (session probe)
- Tool-based writes: agent calls `memory` tool → `_invoke_tool` routes to `MemoryStore`
- Nudge system: `_turns_since_memory` counter increments each loop, resets on tool use, injects a reminder if threshold exceeded

**External MemoryManager** (`agent/memory_manager` + `agent/memory_provider`):
- Plugin architecture: one provider at a time (e.g., Honcho)
- Lifecycle: `initialize()` → `prefetch()` / `sync_turn()` per turn → `on_session_end()` at close
- Adds tool schemas to the tool surface (e.g., Honcho-specific tools)
- System prompt block injected alongside built-in memory
- Background review: `_spawn_background_review()` forks a child `AIAgent` instance to review conversation for memory saves
- Writes carry provenance metadata: `_build_memory_write_metadata()` tracks origin, context, task_id, tool_call_id

### Problems with Current Integration

1. **Two systems, no abstraction.** Built-in `_memory_store` and external `_memory_manager` are checked independently throughout the codebase. No unified interface — every call site must remember to check both and handle their different APIs.

2. **Scattered lifecycle management.** Init in `__init__` (lines 1260-1365), shutdown in `shutdown_memory_provider()`, commit in `commit_memory_session()`, turn sync in `_sync_external_memory_for_turn()`, system prompt injection in `_build_system_prompt()`, nudge tracking in `run_conversation()`, tool dispatch in `_invoke_tool()` — memory lifecycle events are spread across 6+ methods with no central coordination.

3. **State duplication.** `_memory_write_origin`, `_memory_write_context`, `_memory_enabled`, `_user_profile_enabled`, `_memory_nudge_interval`, `_turns_since_memory` — all attributes on AIAgent that should belong to the memory subsystem.

4. **Background review coupling.** `_spawn_background_review()` creates a full child AIAgent just to review the conversation. It copies `_memory_store`, `_memory_enabled`, `_user_profile_enabled`, sets `_memory_write_origin = "background_review"`. This is deep coupling — the background review knows internal attribute names.

5. **Nudge system is ad-hoc.** `_turns_since_memory` increments on each iteration, resets on tool use, and injects a nudge text when threshold is exceeded. This logic is mixed into the tool execution path with no clear contract.

6. **System prompt construction is order-dependent.** Memory blocks, user profile, recent context, vault injection, checkpoint injection, external memory — all appended to `prompt_parts` in a specific sequence in `_build_system_prompt()`. Any reordering risks breaking behavior.

### New Memory Architecture

```python
class MemoryCoordinator:
    """Unified interface for all memory systems.
    
    Owns the lifecycle of both built-in MemoryStore and external 
    MemoryManager providers. All memory operations route through here.
    """
    
    def __init__(self, config: MemoryConfig, session_id: str):
        self.store: MemoryStore | None = None       # built-in
        self.manager: MemoryManager | None = None   # external providers
        self.nudge_tracker: NudgeTracker = NudgeTracker()
        self.write_metadata: WriteMetadataTracker = WriteMetadataTracker()
        self._config = config
    
    # --- Lifecycle ---
    
    def initialize(self, session_id: str, **kwargs) -> None:
        """Load built-in store, initialize external providers."""
        ...
    
    def on_turn_start(self, turn: int, message: str) -> None:
        """Notify all providers of turn start. Returns nudge text if due."""
        ...
    
    def on_turn_end(self, user_msg: str, assistant_msg: str, *, interrupted: bool) -> None:
        """Sync turn to external providers, prefetch for next turn."""
        ...
    
    def on_session_end(self, messages: list) -> None:
        """Final extraction pass. Does NOT tear down providers."""
        ...
    
    def shutdown(self, messages: list | None = None) -> None:
        """End-of-session extraction + full provider teardown."""
        ...
    
    # --- System Prompt ---
    
    def build_prompt_blocks(self) -> list[str]:
        """Return ordered prompt blocks for memory, user profile, recent context, external."""
        # Replaces the scattered _memory_store.format_for_system_prompt() calls
        # and _memory_manager.build_system_prompt() call in _build_system_prompt()
        ...
    
    # --- Tool Integration ---
    
    def get_tool_schemas(self) -> list[dict]:
        """Return all memory tool schemas (built-in + external)."""
        ...
    
    def handle_tool_call(self, tool_name: str, args: dict, *, metadata: dict) -> str:
        """Route tool calls to built-in or external handler.
        
        Automatically attaches write metadata (origin, context, session_id).
        """
        ...
    
    def should_nudge(self) -> str | None:
        """Return nudge text if memory/skill nudge is due, else None."""
        ...
    
    def record_tool_use(self, tool_name: str) -> None:
        """Reset nudge counters when memory/skill tools are actually used."""
        ...
    
    # --- Background Review ---
    
    def build_review_context(self) -> ReviewContext:
        """Extract just what a background review agent needs, 
        without deep-coupling to internal attribute names."""
        ...
    
    # --- Write Provenance ---
    
    def set_write_origin(self, origin: str, context: str) -> None:
        """Set provenance for subsequent memory writes."""
        ...
```

### NudgeTracker

Extracts the ad-hoc counter logic into a testable unit:

```python
class NudgeTracker:
    """Track turns since last memory/skill usage and produce nudge text."""
    
    def __init__(self, nudge_interval: int = 10, skill_interval: int = 10):
        self._turns_since_memory = 0
        self._turns_since_skill = 0
        self._memory_interval = nudge_interval
        self._skill_interval = skill_interval
    
    def on_turn(self) -> None:
        self._turns_since_memory += 1
        self._turns_since_skill += 1
    
    def on_memory_use(self) -> None:
        self._turns_since_memory = 0
    
    def on_skill_use(self) -> None:
        self._turns_since_skill = 0
    
    def memory_nudge(self) -> str | None:
        """Return nudge text if memory interval exceeded, else None."""
        if self._turns_since_memory >= self._memory_interval:
            return MEMORY_NUDGE_TEXT
        return None
    
    def skill_nudge(self) -> str | None:
        """Return nudge text if skill interval exceeded, else None."""
        if self._turns_since_skill >= self._skill_interval:
            return SKILL_NUDGE_TEXT
        return None
```

### WriteMetadataTracker

Extracts provenance tracking:

```python
class WriteMetadataTracker:
    """Track and attach write provenance to memory operations."""
    
    def __init__(self, session_id: str, parent_session_id: str = "", platform: str = "cli"):
        self._origin = "assistant_tool"
        self._context = "foreground"
        self._session_id = session_id
        self._parent_session_id = parent_session_id
        self._platform = platform
    
    def set_origin(self, origin: str, context: str) -> None:
        self._origin = origin
        self._context = context
    
    def build_metadata(self, *, task_id: str = None, tool_call_id: str = None) -> dict:
        """Build provenance dict for external memory writes."""
        metadata = {
            "write_origin": self._origin,
            "execution_context": self._context,
            "session_id": self._session_id,
            "parent_session_id": self._parent_session_id,
            "platform": self._platform,
            "tool_name": "memory",
        }
        if task_id:
            metadata["task_id"] = task_id
        if tool_call_id:
            metadata["tool_call_id"] = tool_call_id
        return {k: v for k, v in metadata.items() if v not in (None, "")}
```

### ReviewContext

Decouples background review from AIAgent internals:

```python
@dataclass
class ReviewContext:
    """Snapshot of memory state for a background review agent.
    
    Instead of copying private AIAgent attributes, extract just the
    config needed to spawn a review agent.
    """
    memory_enabled: bool
    user_profile_enabled: bool
    store_config: dict | None      # enough to recreate MemoryStore
    manager_config: dict | None    # enough to recreate MemoryManager
    session_id: str
    write_origin: str
    write_context: str
```

### Integration with Pipeline Stages

Memory participates in every stage:

**Stage 1 (Request Preparation):**
- `MemoryCoordinator.build_prompt_blocks()` → injected into system prompt
- `MemoryCoordinator.should_nudge()` → append nudge text if due
- `NudgeTracker.on_turn()` → increment counters

**Stage 2 (Provider Call):**
- No direct memory involvement (memory is not in the API call path)

**Stage 3 (Response Processing):**
- No direct memory involvement

**Stage 4 (Tool Dispatch):**
- `MemoryCoordinator.handle_tool_call()` → route memory tool calls to built-in or external handler
- `WriteMetadataTracker.build_metadata()` → attach provenance to writes
- `NudgeTracker.on_memory_use()` / `on_skill_use()` → reset counters on tool use

**Stage 5 (Context Management):**
- `MemoryCoordinator.on_turn_end()` → sync turn to external providers, prefetch next turn

**Session lifecycle:**
- `MemoryCoordinator.initialize()` → at session start
- `MemoryCoordinator.on_session_end()` → on `/new`, context rotation (commit without teardown)
- `MemoryCoordinator.shutdown()` → at CLI exit, gateway eviction (full teardown)

### What Stays from Current Code

- `tools/memory_tool.py` (MemoryStore) — stays, wrapped by MemoryCoordinator
- `agent/memory_manager.py` — stays, wrapped by MemoryCoordinator
- `agent/memory_provider.py` — stays, MemoryProvider protocol unchanged
- `plugins/memory.py` — stays, plugin loading path unchanged
- `agent/vault_injection.py` — stays, injected via `build_prompt_blocks()`
- `agent/checkpoint_injection.py` — stays, injected via `build_prompt_blocks()`

### What Goes

- All `_memory_*` attributes on AIAgent (18+ instances) → `MemoryCoordinator` owns them
- `_build_memory_write_metadata()` → `WriteMetadataTracker.build_metadata()`
- `_sync_external_memory_for_turn()` → `MemoryCoordinator.on_turn_end()`
- `shutdown_memory_provider()` → `MemoryCoordinator.shutdown()`
- `commit_memory_session()` → `MemoryCoordinator.on_session_end()`
- `_spawn_background_review()` memory attribute copying → `MemoryCoordinator.build_review_context()`
- Nudge counter logic scattered in `run_conversation()` → `NudgeTracker.on_turn()` / `should_nudge()`
- System prompt memory block assembly in `_build_system_prompt()` → `MemoryCoordinator.build_prompt_blocks()`

---

## Original Open Questions (all resolved, kept for reference)

1. **Thread safety model.** ✅ Resolved — see Thread Safety table in Open Questions (Remaining).

2. **Backward compatibility surface.** ✅ Audited — see Backward Compatibility section in Open Questions (Remaining).

3. **Session persistence format.** ✅ Redesigned — see Session Persistence section.

4. **Auxiliary client integration.** ✅ Redesigned — see Auxiliary Client Router section.

5. **Provider adapter timeline.** ✅ Wrap first, redesign later — see Open Questions (Remaining) #5.

6. **Context compression is owned by MemoryCoordinator.** ✅ Resolved — MemoryCoordinator gets a pre-compression hook (`on_pre_compress`) and owns the decision of when to compress. Not a standalone pipeline stage.

7. **Background review: lightweight pipeline run, not full agent spawn.** ✅ Resolved — see Background Review (Simplified) in Shared Memory Architecture.

---

## Shared Memory Architecture

### The Problem

Currently, every `AIAgent` instance gets its own `MemoryStore` initialized from the same SQLite database, but there is no coordination between instances. The gateway caches agents per-session in `_agent_cache`, background review forks a full child agent, and `delegate_task` spawns subagents with their own sessions. None of these share memory in real-time.

This means:
- A delegate subagent can't read the parent's mid-conversation memory updates until they're flushed to disk
- Two gateway sessions for the same user can't see each other's memory writes until both reload
- The background review agent shares the `_memory_store` object reference but has no coordination protocol for concurrent writes
- External memory providers (e.g., Honcho) have `on_delegation()` notifications but they're bolted on after the fact, not structural

### Design Principle: Memory is a shared service, not per-agent state.

### MemoryBus

A process-level singleton that coordinates all memory operations across Orchestrator instances.

```python
class MemoryBus:
    """Process-level singleton coordinating memory across all Orchestrator instances.
    
    All MemoryCoordinator instances in the same process share a MemoryBus.
    This enables:
    - Delegate subagents to read parent's mid-session memory
    - Background review to write directly to the shared store
    - Multiple gateway sessions for the same user to see each other's updates
    - Cross-instance coordination for collaborative project work
    """
    
    _instance: "MemoryBus | None" = None
    _lock = threading.Lock()
    
    def __init__(self, config: MemoryConfig):
        self._store: MemoryStore = MemoryStore(...)   # single shared store
        self._coordinators: dict[str, MemoryCoordinator] = {}  # session_id -> coordinator
        self._event_subscribers: list[Callable[MemoryEvent, None]] = []
        self._config = config
    
    @classmethod
    def get(cls, config: MemoryConfig | None = None) -> "MemoryBus":
        """Get or create the process-level singleton."""
        ...
    
    def register(self, coordinator: MemoryCoordinator) -> None:
        """Register a coordinator for event routing."""
        ...
    
    def unregister(self, coordinator: MemoryCoordinator) -> None:
        """Unregister a coordinator (session end, gateway eviction)."""
        ...
    
    def publish(self, event: MemoryEvent) -> None:
        """Broadcast a memory event to all registered coordinators.
        
        Events include:
        - Memory written (any coordinator can react)
        - Memory consolidated
        - Context compressed
        - Session started/ended
        """
        for subscriber in self._event_subscribers:
            try:
                subscriber(event)
            except Exception:
                pass
    
    def get_shared_store(self) -> MemoryStore:
        """Return the shared MemoryStore instance.
        
        Used by background review and delegate subagents to write directly
        to the same store — no attribute copying needed.
        """
        return self._store
    
    def refresh_snapshot(self, session_id: str) -> None:
        """Notify a coordinator that its system prompt snapshot is stale.
        
        Called when another instance writes to memory that would appear
        in the target's prompt. The coordinator re-fetches from the store
        on next prompt build.
        """
        coordinator = self._coordinators.get(session_id)
        if coordinator:
            coordinator.invalidate_snapshot()
```

### MemoryEvent

```python
@dataclass
class MemoryEvent:
    """An event published to the MemoryBus."""
    kind: str              # "write", "replace", "remove", "consolidate", "compress", "session_start", "session_end"
    source_session_id: str  # which instance triggered this
    target: str            # "memory" or "user"
    key: str | None        # key affected (for targeted invalidation)
    content_preview: str | None  # first 200 chars (for logging/debugging)
    timestamp: float
```

### How Shared Memory Works in Practice

**Same process (gateway, delegation, background review):**
- All `MemoryCoordinator` instances share the same `MemoryStore` via `MemoryBus.get_shared_store()`
- Writes via `MemoryCoordinator.handle_tool_call()` go through the shared store
- `MemoryBus.publish()` notifies other coordinators to invalidate their cached prompt blocks
- Background review: gets a `MemoryCoordinator` that shares the same `MemoryBus`. No attribute copying.
- Delegate subagents: get a `MemoryCoordinator` from the same `MemoryBus`. They can read everything the parent has written, and their writes are visible to the parent on next prompt build.

**Cross-process (future: multiple Hermes instances):**
- The `MemoryStore` SQLite database is the source of truth on disk
- `MemoryBus` can watch for changes (WAL mode + polling, or file system watcher)
- External memory providers (Honcho) already have their own synchronization
- This is a future extension — the `MemoryBus` interface is designed for it but process-local coordination is the immediate win

### MemoryCoordinator Updates for Shared Mode

```python
class MemoryCoordinator:
    def __init__(self, config: MemoryConfig, session_id: str, bus: MemoryBus | None = None):
        self.store: MemoryStore | None = None
        self.manager: MemoryManager | None = None
        self.nudge_tracker: NudgeTracker = NudgeTracker()
        self.write_metadata: WriteMetadataTracker = WriteMetadataTracker(...)
        self._bus = bus or MemoryBus.get()
        self._snapshot_valid = False
        self._cached_blocks: list[str] | None = None
    
    def initialize(self, session_id: str, **kwargs) -> None:
        # Store is shared from the bus, not created per-instance
        if self._bus:
            self.store = self._bus.get_shared_store()
        else:
            self.store = MemoryStore(db=self._db)
        self._bus.register(self)
        ...
    
    def build_prompt_blocks(self) -> list[str]:
        """Return cached blocks if snapshot is valid, else rebuild."""
        if self._snapshot_valid and self._cached_blocks is not None:
            return self._cached_blocks
        blocks = self._rebuild_prompt_blocks()
        self._cached_blocks = blocks
        self._snapshot_valid = True
        return blocks
    
    def invalidate_snapshot(self) -> None:
        """Called by MemoryBus when another instance writes memory."""
        self._snapshot_valid = False
        self._cached_blocks = None
    
    def handle_tool_call(self, tool_name: str, args: dict, *, metadata: dict) -> str:
        result = ... # dispatch to store or manager
        # Invalidate other coordinators' snapshots
        self._bus.publish(MemoryEvent(
            kind="write",
            source_session_id=self._session_id,
            target=args.get("target", "memory"),
            key=args.get("key"),
            content_preview=(args.get("content") or "")[:200],
            timestamp=time.time(),
        ))
        return result
```

### Background Review (Simplified)

Before (50+ lines of attribute copying):
```python
review_agent = AIAgent(...)
review_agent._memory_write_origin = "background_review"
review_agent._memory_write_context = "background_review"
review_agent._memory_store = self._memory_store
review_agent._memory_enabled = self._memory_enabled
review_agent._user_profile_enabled = self._user_profile_enabled
review_agent._memory_nudge_interval = 0
review_agent._skill_nudge_interval = 0
review_agent.run_conversation(user_message=prompt, conversation_history=messages_snapshot)
```

After (lightweight pipeline run):
```python
# Background review is just an Orchestrator with restricted config
review_coordinator = MemoryCoordinator(
    config=self._memory._config,
    session_id=f"{self._session_id}:review",
    bus=self._memory._bus,  # shared bus, shared store
)
review_coordinator.set_write_origin("background_review", "background_review")

review_pipeline = Orchestrator(
    config=PipelineConfig(
        system_prompt=MEMORY_REVIEW_PROMPT,
        tool_whitelist={"memory", "memory_add", "memory_replace", "skill_manage", "skill_view"},
        max_iterations=8,
        streaming=False,
        interrupts=False,
    ),
    memory=review_coordinator,
    # No streaming callback, no interrupt event, minimal display
)
review_pipeline.run(user_message=prompt, history=messages_snapshot)
# Writes went directly to the shared MemoryStore via the bus.
# Next prompt build by the primary coordinator will pick them up.
```

### Delegate Subagent Memory Sharing

Before (no sharing — subagent gets its own MemoryStore):
```python
# In delegate_task: subagent creates its own AIAgent with its own session
# No memory visibility between parent and child
# on_delegation() is a post-hoc notification to external providers only
```

After (shared via MemoryBus):
```python
# Parent coordinator creates a child coordinator sharing the bus
child_coordinator = MemoryCoordinator(
    config=parent_coordinator._config,
    session_id=f"{parent_session_id}:delegate:{task_id}",
    bus=parent_coordinator._bus,  # same bus, shares store
)
# Child can now:
# - Read everything parent has in memory
# - Write new memories visible to parent on next prompt build
# - Use same external memory provider (on_delegation becomes automatic)
```

### What This Unlocks

1. **Multi-agent project work.** Multiple Orchestrator instances (primary + delegates) working on the same project share memory writes in real-time. A delegate that saves a key insight makes it immediately visible to the primary's next turn.

2. **Gateway session coordination.** Two gateway sessions for the same user on different platforms see each other's memory writes via snapshot invalidation. No more "why doesn't it remember what I said on Telegram?"

3. **Cleaner background review.** No more forking a full AIAgent. A restricted pipeline run with the same `MemoryCoordinator`/`MemoryBus`. Writes are structural, not attribute hacks.

4. **Future: cross-process coordination.** The `MemoryBus` interface supports `publish()`/`subscribe()` which can be extended to inter-process communication (WebSocket, Redis pub/sub, etc.) when we want to coordinate across multiple Hermes instances.

5. **Event-driven memory updates.** Other subsystems can subscribe to `MemoryBus` events. Context compression can trigger memory consolidation. Rate limit status can trigger memory nudge adjustments. The bus becomes the nervous system.

---

## Open Questions (Remaining)

1. **Thread safety model.** Resolved. Strategy:

    | Component | Thread Safety Strategy |
    |---|---|
    | ConversationContext | Single-writer per pipeline (no lock needed) |
    | MemoryBus registry | `threading.Lock` |
    | MemoryStore (SQLite) | WAL mode, `check_same_thread=False` |
    | EventBus subscribers | `threading.Lock` (subscribe at init only) |
    | EventBus publish | Fire-and-forget, swallow exceptions |
    | ClientManager | `threading.Lock` on create/teardown only, httpx handles request concurrency |
    | StreamState | Per-pipeline, not shared across threads |
    | FallbackChain | `threading.Lock` on failover transitions |
    | Interrupt signaling | `threading.Event` (stays as-is) |

    Key insight: each Orchestrator pipeline owns its own ConversationContext, so the main loop is lock-free. The only shared mutable state is the MemoryBus (registry + shared SQLite), which has clear lock boundaries. No asyncio, no multiprocessing, no custom lock hierarchies.

2. **Backward compatibility surface.** Audited. The external API surface that the new Orchestrator must expose:

    **Methods called externally:**
    - `interrupt(reason: str)` — gateway, delegate_tool, cron scheduler
    - `run_conversation(...)` — gateway (primary entry), cron scheduler
    - `release_clients()` — gateway (cache eviction)
    - `close()` — gateway (full teardown)
    - `shutdown_memory_provider(messages)` — gateway (session teardown)
    - `commit_memory_session(messages)` — gateway (session rotation)
    - `get_activity_summary()` — cron scheduler (timeout detection)

    **Attributes set by gateway (per-turn configuration):**
    - `tool_progress_callback` — gateway
    - `step_callback` — gateway
    - `stream_delta_callback` — gateway
    - `interim_assistant_callback` — gateway
    - `status_callback` — gateway
    - `reasoning_config` — gateway
    - `service_tier` — gateway
    - `request_overrides` — gateway
    - `background_review_callback` — gateway
    - `_api_call_count` — gateway (reset to 0 on cached reuse)
    - `_last_activity_ts` — gateway (set on cached reuse)
    - `_last_activity_desc` — gateway (set on cached reuse)

    **Attributes read by delegate_tool:**
    - `model`, `base_url`, `platform` — inherited by child agents
    - `providers_allowed`, `providers_ignored`, `providers_order`, `provider_sort` — inherited
    - `_client_kwargs.get("api_key")` — credential inheritance
    - `valid_tool_names` — tool surface filtering
    - `_memory_manager` — delegation notifications
    - `max_tokens`, `prefill_messages`, `_session_db`, `session_id` — inherited by child agents
    - `_active_children` — lifecycle tracking
    - `_delegate_depth` — depth limiting
    - `_print_fn` — output routing

    **Constructor parameters (AIAgent.__init__):**
    - Core: `model`, `provider`, `base_url`, `api_key`, `api_mode`
    - Limits: `max_iterations`, `max_tokens`, `reasoning_config`, `iteration_budget`
    - Behavior: `quiet_mode`, `skip_memory`, `skip_context_files`, `platform`
    - Delegation: `acp_command`, `acp_args`, `parent_session_id`, `clarify_callback`, `thinking_callback`
    - Tools: `enabled_toolsets`, `ephemeral_system_prompt`
    - Gateway: `tool_progress_callback`, `log_prefix`
    - Credentials: `providers_allowed`, `providers_ignored`, `providers_order`, `provider_sort`

    **Migration strategy:** Provide an `AIAgent` compatibility shim that delegates to `Orchestrator`. The shim maps old attribute names to new `ConversationContext` / `SessionState` fields. This allows the gateway, delegate_tool, and cron scheduler to continue working unchanged during cutover.

3. **Session persistence format.** Redesigned. The cutover is the opportunity to clean up persistence. JSON session files are redundant with SQLite. MemoryStore moves to WAL mode for concurrent access. Format evolution is part of the redesign, not deferred.

4. **Auxiliary client.** Redesigned. See Auxiliary Client Router section.

5. **Provider adapter timeline.** Wrap first, redesign later. Thin `ProviderProtocol` adapters that delegate to existing code. Works immediately, carries legacy code but no risk of rediscovering edge cases. Each adapter gets a redesign pass on its own schedule.

---

## Noted Minor Gaps (deferred to implementation)

These are real concerns but small enough to address during implementation rather than in the design spec:

1. **Ephemeral system prompts** (38 refs) — Prompts that aren't saved to trajectories. Handled by a flag on `SessionState.ephemeral_system_prompt`. No architectural impact.

2. **Vision handling** — `_describe_image_for_anthropic_fallback` and `vision_utils.py` are provider-specific content preprocessing paths. They become methods on the Anthropic provider adapter. No new components needed.

3. **Task resource cleanup** — `_cleanup_task_resources` (33 lines) handles VM and browser cleanup for completed delegate tasks. This is a post-dispatch hook in `ToolExecutor`, not a new subsystem.

4. **Error classifier integration** — `agent/error_classifier.py` (949 lines) is listed as "stays" but isn't shown feeding into `FallbackChain` decisions. During Phase 2 (Provider Plugins), each provider adapter will call `classify_error()` and return a `FallbackDecision` that the `FallbackChain` consumes. The wiring is straightforward.

5. **Codex Responses API** — `codex_responses_adapter.py` (1,000 lines) is a full alternative API path with its own streaming protocol. The `ProviderProtocol` needs `supports_responses_api: bool` in `ProviderCapabilities`, and the Codex adapter needs a separate `execute_responses_api()` method. This is captured in the provider adapter design.

6. **`_save_session_log`** (66 lines) — Trajectory logging (JSONL of API calls). Moves to `EventBus` subscriber: `SessionLogger` subscribes to `api_call` events. No new component needed.

7. **`_describe_image_for_anthropic_fallback`** — Vision fallback for Anthropic when image URLs fail. Belongs in the Anthropic provider adapter. No architectural impact.

---

*This spec represents the current design intent. It will evolve as we build and learn. The migration is incremental — we can stop at any phase and have a working system.*