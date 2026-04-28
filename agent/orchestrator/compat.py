"""AIAgent compatibility shim — maps old AIAgent API to new Orchestrator.

Phase 6C (Battle-Test Wiring):
- When True: run_conversation() delegates to Orchestrator, which delegates
  hard work (API calls, tool dispatch, streaming) back to the parent AIAgent
- When False: pass-through to existing AIAgent (not implemented here)
- Maps 7 external methods and 13 gateway-set attributes
- Maps delegate-readable attributes to SessionState fields

The shim allows the gateway, delegate_tool, and cron scheduler to
continue working unchanged during cutover.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from agent.orchestrator.memory import MemoryCoordinator
from agent.orchestrator.stages import PreparedRequest


from agent.orchestrator.context import (
    ConversationContext,
    ParsedResponse,
    ProviderResult,
    SessionState,
    StreamState,
    UsageInfo,
)
from agent.orchestrator.events import EventBus, PipelineEvent
from agent.orchestrator.orchestrator import Orchestrator
from agent.orchestrator.provider_adapters import OpenAICompatibleProvider
from agent.orchestrator.providers import ProviderRegistry
from agent.orchestrator.stages import PipelineResult, RequestPrepStage
from agent.orchestrator.tools import ToolExecutor

logger = logging.getLogger(__name__)

# ============================================================================
# Feature flag — controls whether new pipeline is used
# ============================================================================

# ============================================================================
# Helpers for converting SimpleNamespace results to dicts
# ============================================================================

def _namespace_to_dict(obj: Any) -> Any:
    """Recursively convert SimpleNamespace/dotted objects to dicts.

    AIAgent._interruptible_streaming_api_call returns SimpleNamespace objects
    that mimic the OpenAI SDK response shape. This converts them to plain dicts
    so ProviderResult and ResponseProcessingStage can handle them uniformly.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_namespace_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    # Fallback for other objects with __dict__
    if hasattr(obj, "__dict__"):
        return {k: _namespace_to_dict(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    return obj

def _get_finish_reason(result: Any) -> str:
    """Extract finish_reason from a SimpleNamespace or dict response."""
    try:
        if isinstance(result, dict):
            return result.get("choices", [{}])[0].get("finish_reason", "stop")
        return getattr(result.choices[0], "finish_reason", "stop") or "stop"
    except (AttributeError, IndexError, TypeError):
        return "stop"

class AIAgentCompatShim:
    """Compatibility shim that maps the AIAgent external API to Orchestrator.

    Provides the 7 external methods:
    - chat(message) → str
    - run_conversation(message, **kwargs) → dict
    - interrupt(reason) → None
    - release_clients() → None
    - close() → None
    - shutdown_memory_provider(messages) → None
    - commit_memory_session(messages) → None
    - get_activity_summary() → dict

    And the 13 gateway-set attributes + 13 delegate-readable attributes.

        When False, this shim is not used (AIAgent runs directly).
    """

    def __init__(
        self,
        parent_agent: Any = None,  # AIAgent instance — injected for delegation
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_key: str = "",
        max_iterations: Optional[int] = None,
        api_mode: str = "",
        session_id: str = "",
        registry: Optional[ProviderRegistry] = None,
        event_bus: Optional[EventBus] = None,
        tool_executor: Optional[ToolExecutor] = None,
        # Remaining constructor params map to SessionState fields
        **kwargs,
    ) -> None:
        # --- Parent agent reference (for delegation) ---
        self._agent = parent_agent
        # Memory coordinator (must exist before parent wiring below)
        self.memory = MemoryCoordinator(session_id=session_id or f"session-{id(self)}")
        if parent_agent is not None:
            self.memory.store = getattr(parent_agent, "_memory_store", None)
            self.memory.manager = getattr(parent_agent, "_memory_manager", None)
            self.memory._memory_enabled = getattr(parent_agent, "_memory_enabled", False)
            self.memory._user_profile_enabled = getattr(parent_agent, "_user_profile_enabled", False)

        # --- Core identity (delegate-readable) ---
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.api_mode = api_mode
        self.session_id = session_id or f"session-{id(self)}"
        self.max_iterations = max_iterations or Orchestrator.DEFAULT_MAX_ITERATIONS

        # --- Internal state ---
        self._state = SessionState(
            session_id=self.session_id,
            active_model=model,
        )
        self._ctx = ConversationContext(
            session_id=self.session_id,
            max_iterations=self.max_iterations,
        )
        self._event_bus = event_bus or EventBus()
        self._registry = registry or ProviderRegistry()
        self._tool_executor = tool_executor or ToolExecutor()

        # --- Wire real provider call through parent AIAgent ---
        self._wire_real_provider()

        # --- Build Orchestrator with real delegations ---
        self._orchestrator = Orchestrator(
            registry=self._registry,
            event_bus=self._event_bus,
            tool_executor=self._tool_executor,
            model_name=model,
            max_iterations=self.max_iterations,
            # Inject RequestPrepStage that uses parent's _build_api_kwargs (only when wired)
            request_prep=RequestPrepStage(
                model_name=model,
                prepare_fn=self._prepare_request if self._agent else None,
            ),
        )
        # Wire provider name resolution through the shim (deferred — always
        # calls self._resolve_provider_name so post-init overrides work)
        self._orchestrator._resolve_provider_name = lambda ctx: self._resolve_provider_name(ctx)

        # --- Wire real tool dispatch through parent AIAgent ---
        self._wire_real_tools()

        # --- Wire EventBus → AIAgent callbacks for streaming ---
        self._bridge_streaming()

        self._interrupt_reason: Optional[str] = None

        # --- Gateway-set attributes (initially None/empty) ---
        self.tool_progress_callback: Optional[Callable] = None
        self.step_callback: Optional[Callable] = None
        self.stream_delta_callback: Optional[Callable] = None
        self.interim_assistant_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        self.reasoning_config: Optional[dict] = None
        self.service_tier: Optional[str] = None
        self.request_overrides: Optional[dict] = None
        self.background_review_callback: Optional[Callable] = None
        self._api_call_count: int = 0
        self._last_activity_ts: float = 0.0
        self._last_activity_desc: str = ""

    # ========================================================================
    # Delegation wiring — private methods
    # ========================================================================

    def _wire_real_provider(self) -> None:
        """Register an OpenAICompatibleProvider that delegates API calls to
        the parent AIAgent's proven _interruptible_streaming_api_call method.

        This is the core battle-test wiring: the new pipeline coordinates,
        but the actual HTTP call rides on thousands of hours of production
        testing in the old code path.
        """
        call_fn = self._make_provider_call if self._agent is not None else None
        provider = OpenAICompatibleProvider(
            base_url=self.base_url,
            model_name=self.model,
            call_fn=call_fn,
        )
        self._registry.register("openai_compatible", provider)
        logger.info(f"[pipeline] Registered openai_compatible provider (delegation={'wired' if call_fn else 'stub'})")

    def _wire_real_tools(self) -> None:
        """Register parent AIAgent's tool definitions with the ToolExecutor
        and set the dispatch handler to model_tools.handle_function_call.

        The ToolExecutor stores name→handler mappings. We register each
        tool name from the parent's tool definitions and set the handler
        to _dispatch_tool which bridges to model_tools.handle_function_call.
        """
        if self._agent is None or not hasattr(self._agent, "tools"):
            logger.info("[pipeline] No parent agent or tools — tool dispatch uses stubs")
            return

        parent_tools = self._agent.tools
        if not parent_tools:
            logger.info("[pipeline] Parent agent has no tools — skipping tool wiring")
            return

        registered = 0
        for tool_def in parent_tools:
            func = tool_def.get("function", {})
            name = func.get("name", "")
            if name:
                # Wrap in lambda to capture tool name — ToolExecutor calls handler(args_dict)
                self._tool_executor.register(name, handler=lambda a, n=name: self._dispatch_tool(n, a))
                registered += 1

        logger.info(f"[pipeline] Registered {registered} tools with ToolExecutor")

    def _bridge_streaming(self) -> None:
        """Subscribe to EventBus stream events and forward to AIAgent callbacks.

        When the ProviderCallStage emits stream deltas (or when the
        _make_provider_call wrapper fires them via the event bus), these
        subscribers forward them to the parent AIAgent's callback attributes:
        - stream_delta_callback → gateway stream consumer
        - step_callback → gateway progress display
        """
        logger.info("[pipeline] Streaming bridge set up (delta + reasoning events)")
        shim = self  # closure reference

        def on_stream_delta(event: PipelineEvent) -> None:
            """Forward stream text deltas to the parent AIAgent."""
            if shim._agent is None:
                return
            callback = getattr(shim._agent, "stream_delta_callback", None)
            if callback and callable(callback):
                try:
                    callback(event.data.get("delta", ""))
                except Exception:
                    pass  # Don't let callback errors break the pipeline

        def on_stream_reasoning(event: PipelineEvent) -> None:
            """Forward reasoning deltas to the parent AIAgent."""
            if shim._agent is None:
                return
            # _fire_reasoning_delta is a method on AIAgent
            fire_fn = getattr(shim._agent, "_fire_reasoning_delta", None)
            if fire_fn and callable(fire_fn):
                try:
                    fire_fn(event.data.get("delta", ""))
                except Exception:
                    pass

        def on_step_event(event: PipelineEvent) -> None:
            """Forward iteration/tool events to step_callback for progress."""
            if shim._agent is None:
                return
            callback = getattr(shim._agent, "step_callback", None)
            if callback and callable(callback):
                try:
                    callback(event.data)
                except Exception:
                    pass

        self._event_bus.subscribe("stream_delta", on_stream_delta)
        self._event_bus.subscribe("stream_reasoning", on_stream_reasoning)
        self._event_bus.subscribe("iteration_start", on_step_event)
        self._event_bus.subscribe("iteration_end", on_step_event)
        self._event_bus.subscribe("tool_start", on_step_event)
        self._event_bus.subscribe("tool_end", on_step_event)

    def _make_provider_call(self, **kwargs) -> Any:
        """Delegate the API call to the parent AIAgent's proven path.

        Bridges OpenAICompatibleProvider.execute(**request) →
        AIAgent._interruptible_streaming_api_call().

        The provider's execute() unpacks the request dict as **kwargs,
        so we receive them as individual keyword args and repack into
        a single dict for _interruptible_streaming_api_call().

        Returns a dict (not SimpleNamespace) so ResponseProcessingStage
        can use .get() on the response.
        """
        if self._agent is None:
            raise RuntimeError("[pipeline] No parent AIAgent for provider call delegation")

        logger.info("[pipeline] Provider call delegated to AIAgent._interruptible_streaming_api_call")
        raw_result = self._agent._interruptible_streaming_api_call(
            kwargs,
            on_first_delta=None,  # Streaming delta wiring comes through EventBus
        )
        # Convert SimpleNamespace response to dict for ResponseProcessingStage
        if isinstance(raw_result, dict):
            return raw_result
        return _namespace_to_dict(raw_result)

    # Agent-level tools that need access to parent AIAgent state stores
    # and cannot be dispatched through model_tools.handle_function_call.
    # Must mirror the set in model_tools._AGENT_LOOP_TOOLS.
    _AGENT_LOOP_TOOLS = frozenset({"todo", "memory", "session_search", "delegate_task", "checkpoint", "clarify"})

    def _dispatch_tool(self, name: str, args: dict) -> str:
        """Dispatch a tool call through the appropriate handler.

        Agent-loop tools (memory, session_search, todo, etc.) need access
        to parent AIAgent state stores and are handled inline. All other
        tools delegate to model_tools.handle_function_call which provides
        sandboxing, argument coercion, and plugin hooks.
        """
        logger.info(f"[pipeline] Tool dispatch: {name}({list(args.keys())})")

        # Agent-loop tools: these need parent state (_memory_store,
        # _session_db, etc.) and cannot go through handle_function_call
        # (which rejects them with "must be handled by the agent loop").
        if name in self._AGENT_LOOP_TOOLS:
            result = self._dispatch_agent_loop_tool(name, args)
        else:
            from model_tools import handle_function_call

            task_id = getattr(self._agent, "task_id", None) if self._agent else None
            result = handle_function_call(
                function_name=name,
                function_args=args,
                task_id=task_id,
                session_id=self.session_id,
            )
        # Record tool use for nudge tracking
        self.memory.record_tool_use(name)
        return result

    def _dispatch_agent_loop_tool(self, name: str, args: dict) -> str:
        """Handle agent-level tools that need parent AIAgent state.

        Mirrors the if/elif dispatch in run_agent._invoke_tool and the
        sequential path's tool handling, but from the pipeline context.
        Falls back to an error message when the parent agent or required
        state is not available (e.g. delegate_task subagents).
        """
        import json as _json

        if name == "memory":
            if self._agent is None:
                return _json.dumps({"error": "Memory not available: no parent agent."})
            _memory_store = getattr(self._agent, "_memory_store", None)
            if _memory_store is None:
                return _json.dumps({"error": "Memory not available: memory store not initialized."})
            from tools.memory_tool import memory_tool as _memory_tool
            target = args.get("target", "memory")
            result = _memory_tool(
                action=args.get("action"),
                target=target,
                content=args.get("content"),
                old_text=args.get("old_text"),
                category=args.get("category"),
                key=args.get("key"),
                priority=args.get("priority"),
                query=args.get("query"),
                store=_memory_store,
            )
            # Bridge: notify external memory provider of writes
            _memory_manager = getattr(self._agent, "_memory_manager", None)
            if _memory_manager and args.get("action") in ("add", "replace"):
                try:
                    _memory_manager.on_memory_write(
                        args.get("action", ""),
                        target,
                        args.get("content", ""),
                        metadata=self._build_memory_write_metadata(),
                    )
                except Exception:
                    pass
            return result

        elif name == "session_search":
            if self._agent is None:
                return _json.dumps({"success": False, "error": "Session search not available: no parent agent."})
            _session_db = getattr(self._agent, "_session_db", None)
            if not _session_db:
                return _json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=args.get("query", ""),
                role_filter=args.get("role_filter"),
                limit=args.get("limit", 3),
                fast=args.get("fast", True),
                db=_session_db,
                current_session_id=self.session_id,
            )

        elif name == "todo":
            if self._agent is None:
                return _json.dumps({"error": "Todo not available: no parent agent."})
            _todo_store = getattr(self._agent, "_todo_store", None)
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=args.get("todos"),
                merge=args.get("merge", False),
                store=_todo_store,
            )

        elif name == "clarify":
            if self._agent is None:
                return _json.dumps({"error": "Clarify not available: no parent agent."})
            from tools.clarify_tool import clarify_tool as _clarify_tool
            _clarify_callback = getattr(self._agent, "clarify_callback", None)
            return _clarify_tool(
                question=args.get("question", ""),
                choices=args.get("choices"),
                callback=_clarify_callback,
            )

        elif name == "delegate_task":
            if self._agent is None:
                return _json.dumps({"error": "delegate_task not available: no parent agent."})
            return self._agent._dispatch_delegate_task(args)

        elif name == "checkpoint":
            if self._agent is None:
                return _json.dumps({"error": "Checkpoint not available: no parent agent."})
            from tools.checkpoint_tool import checkpoint_tool as _checkpoint_tool
            _checkpoint_store = getattr(self._agent, "_checkpoint_store", None)
            return _checkpoint_tool(
                action=args.get("action"),
                task=args.get("task"),
                progress=args.get("progress"),
                state=args.get("state"),
                decisions=args.get("decisions"),
                blocked=args.get("blocked"),
                unresolved=args.get("unresolved"),
                store=_checkpoint_store,
                agent=self._agent,
            )

        return _json.dumps({"error": f"Unknown agent-loop tool: {name}"})

    def _build_memory_write_metadata(self) -> dict:
        """Build metadata dict for memory write notifications."""
        return {
            "session_id": self.session_id,
            "pipeline": True,
        }

    def _prepare_request(self, ctx: ConversationContext) -> PreparedRequest:
        """Build API kwargs using AIAgent._build_api_kwargs.

        Called by RequestPrepStage when _prepare_fn is set. This injects
        the parent's system prompt, tools schema, and API parameters into
        the pipeline — everything that _build_api_kwargs produces.
        """

        if self._agent is None:
            raise RuntimeError("[pipeline] No parent AIAgent for request preparation")

        # Use the parent's system prompt builder (includes memory, vault, etc.)
        system_prompt = ""
        if hasattr(self._agent, "_build_system_prompt"):
            system_prompt = self._agent._build_system_prompt() or ""

        # Build messages with system prompt prepended
        messages = list(ctx.messages)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # Call the parent's _build_api_kwargs for model params, tools, etc.
        api_kwargs = {}
        if hasattr(self._agent, "_build_api_kwargs"):
            api_kwargs = self._agent._build_api_kwargs(messages)

        # Ensure messages are in the right place — api_kwargs may already contain messages
        if "messages" not in api_kwargs:
            api_kwargs["messages"] = messages

        # Determine provider name
        provider_name = self._resolve_provider_name(ctx)

        logger.info(f"[pipeline] Request prep: {len(messages)} messages, "
                     f"system_prompt={len(system_prompt)} chars, provider={provider_name}")

        return PreparedRequest(
            messages=messages,
            api_kwargs=api_kwargs,
            provider_name=provider_name,
        )

    # ========================================================================
    # External method 1: chat()
    # ========================================================================

    def chat(self, message: str, stream_callback: Optional[Callable] = None) -> str:
        """Simple chat interface that returns just the final response.

                """
        if message is None:
            raise TypeError("chat() message must be a string, got None")
        if not isinstance(message, str):
            raise TypeError(f"chat() message must be a string, got {type(message).__name__}")
        if not message.strip():
            raise ValueError("chat() message must not be empty")

        # Wire stream callback if provided
        if stream_callback is not None:
            self.stream_delta_callback = stream_callback

        logger.info(f"[pipeline] chat called: message length={len(message)}")
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result.get("final_response", "")

    # ========================================================================
    # External method 2: run_conversation()
    # ========================================================================

    def run_conversation(self, message: str, **kwargs) -> dict:
        """Full conversation interface returning a result dict.

        Delegates to Orchestrator.run() with a ConversationContext
        built from the current session state + the new user message.

        Returns a dict matching the AIAgent.run_conversation() return format:
        {
            "final_response": str,
            "messages": list[dict],
            "iterations": int,
            ...
        }
        """

        logger.info(f"[pipeline] run_conversation called: message length={len(message)}")

        # Build context from current session state
        self._sync_state_to_ctx()
        self._ctx.messages.append({"role": "user", "content": message})

        # Set stream callback if provided via kwargs
        stream_callback = kwargs.get("stream_callback") or self.stream_delta_callback
        if stream_callback:
            self._ctx.stream_callback = stream_callback

        # Run the pipeline
        result: PipelineResult = self._orchestrator.run(self._ctx)

        # Sync context back to state
        self._sync_ctx_to_state(result)

        # Build return dict matching AIAgent.run_conversation() format
        final_response = ""
        if result.response and result.response.message:
            final_response = result.response.message.get("content", "")

        logger.info(
            f"[pipeline] run_conversation completed: {result.iterations} iterations, "
            f"finish_reason={result.response.finish_reason if result.response else None}, "
            f"tokens={self._state.total_tokens}"
        )

        return {
            "final_response": final_response,
            "messages": list(self._ctx.messages),
            "iterations": result.iterations,
            "interrupted": result.interrupted,
            "finish_reason": result.response.finish_reason if result.response else None,
            "total_tokens": self._state.total_tokens,
            "input_tokens": self._state.input_tokens,
            "output_tokens": self._state.output_tokens,
        }

    # ========================================================================
    # External method 3: interrupt()
    # ========================================================================

    def interrupt(self, message: Optional[str] = None) -> None:
        """Interrupt the current conversation. Sets the interrupt event."""
        self._interrupt_reason = message or "interrupted"
        self._ctx.interrupt_event.set()
        logger.info(f"[pipeline] Interrupt requested: {self._interrupt_reason}")

    # ========================================================================
    # External method 4: release_clients()
    # ========================================================================

    def release_clients(self) -> None:
        """Release HTTP clients. Delegates to parent AIAgent if available."""
        if self._agent is not None and hasattr(self._agent, "release_clients"):
            self._agent.release_clients()
            logger.info("[pipeline] release_clients delegated to parent AIAgent")
        else:
            logger.info("[pipeline] release_clients — no parent agent, skipping")

    # ========================================================================
    # External method 5: close()
    # ========================================================================

    def close(self) -> None:
        """Full teardown. Releases clients, flushes memory."""
        logger.info("[pipeline] close called — full teardown")
        self.release_clients()
        self.shutdown_memory_provider()

    # ========================================================================
    # External method 6: shutdown_memory_provider()
    # ========================================================================

    def shutdown_memory_provider(self, messages: Optional[list] = None) -> None:
        """Shut down memory provider at session end."""
        if self._agent is not None and hasattr(self._agent, "shutdown_memory_provider"):
            self._agent.shutdown_memory_provider(messages)
            logger.info("[pipeline] shutdown_memory_provider delegated to parent AIAgent")
        else:
            logger.info("[pipeline] shutdown_memory_provider — no parent agent, skipping")

    # ========================================================================
    # External method 7: commit_memory_session()
    # ========================================================================

    def commit_memory_session(self, messages: Optional[list] = None) -> None:
        """Commit memory session (e.g., on /new command)."""
        if self._agent is not None and hasattr(self._agent, "commit_memory_session"):
            self._agent.commit_memory_session(messages)
            logger.info("[pipeline] commit_memory_session delegated to parent AIAgent")
        else:
            logger.info("[pipeline] commit_memory_session — no parent agent, skipping")

    # ========================================================================
    # External method 8: get_activity_summary()
    # ========================================================================

    def get_activity_summary(self) -> dict:
        """Return activity summary for timeout detection."""
        return {
            "last_activity_ts": self._last_activity_ts,
            "last_activity_desc": self._last_activity_desc,
            "api_call_count": self._api_call_count,
            "session_id": self.session_id,
        }

    # ========================================================================
    # switch_model()
    # ========================================================================

    def switch_model(
        self,
        new_model: str,
        new_provider: str,
        api_key: str = "",
        base_url: str = "",
        api_mode: str = "",
    ) -> None:
        """Switch the model/provider in-place for a live session.

        Mirrors AIAgent.switch_model() but updates SessionState instead
        of scattered self.* attributes.
        """
        self.model = new_model
        self.provider = new_provider
        if base_url:
            self.base_url = base_url
        if api_key:
            self.api_key = api_key
        if api_mode:
            self.api_mode = api_mode

        # Update inner state
        self._state.active_model = new_model

        # Reset system prompt cache (model change invalidates it)
        self._state.cached_system_prompt = None

        # Re-wire the provider with new model/base_url
        if self._registry:
            try:
                self._registry.register("openai_compatible", OpenAICompatibleProvider(
                    base_url=self.base_url or "",
                    model_name=new_model,
                    call_fn=self._make_provider_call if self._agent else None,
                ))
            except Exception:
                pass  # Registry.register may raise on duplicate

        logger.info(f"[pipeline] switch_model: {new_model} ({new_provider})")

    # ========================================================================
    # Internal: state synchronization
    # ========================================================================

    def _sync_state_to_ctx(self) -> None:
        """Copy SessionState fields + parent's system prompt into ConversationContext."""
        logger.info(f"[pipeline] Syncing state to context: {len(self._state.messages)} messages")
        self._ctx.messages = self._state.messages
        self._ctx.session_id = self._state.session_id
        self._ctx.max_iterations = self.max_iterations
        # Reset iteration counter for new turn
        self._ctx.iteration = 0

        # Build system prompt from the parent AIAgent's proven path
        if self._agent is not None and hasattr(self._agent, "_build_system_prompt"):
            system_prompt = self._agent._build_system_prompt() or ""
            self._ctx.system_prompt = system_prompt
        elif self._state.system_prompt:
            self._ctx.system_prompt = self._state.system_prompt
        else:
            self._ctx.system_prompt = ""

        # Wire tools from parent
        if self._agent is not None and hasattr(self._agent, "tools"):
            self._ctx.tools = self._agent.tools

    def _sync_ctx_to_state(self, result: PipelineResult) -> None:
        """Copy ConversationContext fields back to SessionState and self."""
        self._state.messages = list(self._ctx.messages)
        self._state.iteration_count = self._ctx.iteration
        self._api_call_count += result.iterations
        self._last_activity_ts = time.time()
        self._last_activity_desc = f"pipeline completed {result.iterations} iterations"

        # Sync token counts from usage
        if result.response and result.response.usage:
            usage = result.response.usage
            self._state.total_tokens = getattr(usage, "total_tokens", 0)
            self._state.input_tokens = getattr(usage, "prompt_tokens", 0)
            self._state.output_tokens = getattr(usage, "completion_tokens", 0)

    # ========================================================================
    # Provider name resolution wiring
    # ========================================================================

    def _resolve_provider_name(self, ctx: ConversationContext) -> str:
        """Resolve provider name for the current context.

        Uses the parent AIAgent's api_mode to determine the correct
        provider, falling back to openai_compatible for local endpoints.
        """
        if self._agent is not None:
            api_mode = getattr(self._agent, "api_mode", "chat_completions")
            # Map api_mode to provider registry name
            if api_mode == "anthropic_messages":
                return "anthropic"
            elif api_mode == "codex_responses":
                return "codex"
            elif api_mode == "bedrock_converse":
                return "bedrock"
        # Default to openai_compatible (covers Ollama, OpenRouter, direct OpenAI)
        return "openai_compatible"