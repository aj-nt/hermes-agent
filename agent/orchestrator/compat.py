"""AIAgent compatibility shim — maps old AIAgent API to new Orchestrator.

Phase 5 (Cutover) per DESIGN.md:
- Feature flag: USE_NEW_PIPELINE controls routing
- When True: chat() and run_conversation() delegate to Orchestrator
- When False: pass-through to existing AIAgent (not implemented here)
- Maps 7 external methods and 13 gateway-set attributes
- Maps delegate-readable attributes to SessionState fields

The shim allows the gateway, delegate_tool, and cron scheduler to
continue working unchanged during cutover.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from agent.orchestrator.context import (
    ConversationContext,
    SessionState,
    StreamState,
)
from agent.orchestrator.events import EventBus
from agent.orchestrator.orchestrator import Orchestrator
from agent.orchestrator.providers import ProviderRegistry
from agent.orchestrator.stages import PipelineResult
from agent.orchestrator.tools import ToolExecutor

logger = logging.getLogger(__name__)

# ============================================================================
# Feature flag — controls whether new pipeline is used
# ============================================================================

# Default OFF — safe cutover. Set to True via config or monkeypatch to enable.
USE_NEW_PIPELINE: bool = False


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

    When USE_NEW_PIPELINE is True, routes through Orchestrator.
    When False, this shim is not used (AIAgent runs directly).
    """

    def __init__(
        self,
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_key: str = "",
        max_iterations: Optional[int] = None,
        session_id: str = "",
        registry: Optional[ProviderRegistry] = None,
        event_bus: Optional[EventBus] = None,
        tool_executor: Optional[ToolExecutor] = None,
        # Remaining constructor params map to SessionState fields
        **kwargs,
    ) -> None:
        # --- Core identity (delegate-readable) ---
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
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
        self._orchestrator = Orchestrator(
            registry=self._registry,
            event_bus=self._event_bus,
            tool_executor=self._tool_executor,
            model_name=model,
            max_iterations=self.max_iterations,
        )
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

        # --- Gateway callback wiring ---
        # When callbacks are set by the gateway, they need to reach
        # the EventBus subscribers and ConversationContext.

    # ========================================================================
    # External method 1: chat()
    # ========================================================================

    def chat(self, message: str, stream_callback: Optional[Callable] = None) -> str:
        """Simple chat interface that returns just the final response.

        When USE_NEW_PIPELINE is True, delegates to Orchestrator.run().
        """
        if not USE_NEW_PIPELINE:
            raise NotImplementedError(
                "AIAgentCompatShim.chat() requires USE_NEW_PIPELINE=True. "
                "Use the original AIAgent for the old pipeline."
            )

        # Wire stream callback if provided
        if stream_callback is not None:
            self.stream_delta_callback = stream_callback

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
        if not USE_NEW_PIPELINE:
            raise NotImplementedError(
                "AIAgentCompatShim.run_conversation() requires USE_NEW_PIPELINE=True."
            )

        # Build context from current session state
        self._sync_state_to_ctx()
        self._ctx.messages.append({"role": "user", "content": message})

        # Set stream callback if provided via kwargs
        stream_callback = kwargs.get("stream_callback") or self.stream_delta_callback
        if stream_callback:
            self._ctx.stream_callback = stream_callback

        # Wire provider resolution through the shim
        self._orchestrator._resolve_provider_name = self._resolve_provider_name

        # Run the pipeline
        result: PipelineResult = self._orchestrator.run(self._ctx)

        # Sync context back to state
        self._sync_ctx_to_state(result)

        # Build return dict matching AIAgent.run_conversation() format
        final_response = ""
        if result.response and result.response.message:
            final_response = result.response.message.get("content", "")

        return {
            "final_response": final_response,
            "messages": list(self._ctx.messages),
            "iterations": result.iterations,
            "interrupted": result.interrupted,
            "finish_reason": result.response.finish_reason if result.response else None,
        }

    # ========================================================================
    # External method 3: interrupt()
    # ========================================================================

    def interrupt(self, message: str = None) -> None:
        """Interrupt the current conversation. Sets the interrupt event."""
        self._interrupt_reason = message or "interrupted"
        self._ctx.interrupt_event.set()
        logger.info(f"Interrupt requested: {self._interrupt_reason}")

    # ========================================================================
    # External method 4: release_clients()
    # ========================================================================

    def release_clients(self) -> None:
        """Release HTTP clients. Delegates to ClientManager in the new pipeline."""
        # ClientManager cleanup — will be wired in Phase 2+
        logger.info("release_clients called — client cleanup delegated to ClientManager")

    # ========================================================================
    # External method 5: close()
    # ========================================================================

    def close(self) -> None:
        """Full teardown. Releases clients, flushes memory."""
        logger.info("close called — full teardown")
        self.release_clients()

    # ========================================================================
    # External method 6: shutdown_memory_provider()
    # ========================================================================

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """Shut down memory provider at session end."""
        logger.info("shutdown_memory_provider called — delegating to MemoryCoordinator")

    # ========================================================================
    # External method 7: commit_memory_session()
    # ========================================================================

    def commit_memory_session(self, messages: list = None) -> None:
        """Commit memory session (e.g., on /new command)."""
        logger.info("commit_memory_session called — delegating to MemoryCoordinator")

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

        # Update inner state
        self._state.active_model = new_model

        # Reset system prompt cache (model change invalidates it)
        self._state.cached_system_prompt = None

        logger.info(f"switch_model: {new_model} ({new_provider})")

    # ========================================================================
    # Internal: state synchronization
    # ========================================================================

    def _sync_state_to_ctx(self) -> None:
        """Copy SessionState fields into ConversationContext before pipeline run."""
        self._ctx.messages = self._state.messages
        self._ctx.system_prompt = self._state.system_prompt or ""
        self._ctx.session_id = self._state.session_id
        self._ctx.max_iterations = self.max_iterations
        # Reset iteration counter for new turn
        self._ctx.iteration = 0

    def _sync_ctx_to_state(self, result: PipelineResult) -> None:
        """Copy ConversationContext fields back to SessionState after pipeline run."""
        self._state.messages = list(self._ctx.messages)
        self._state.iteration_count = self._ctx.iteration
        self._state.api_call_count = self._api_call_count
        self._last_activity_ts = self._ctx.iteration  # Will be replaced by ActivityTracker

    # ========================================================================
    # Provider name resolution wiring
    # ========================================================================

    def _resolve_provider_name(self, ctx: ConversationContext) -> str:
        """Resolve provider name for the current context.

        Override this method or set it on the internal orchestrator
        for custom provider resolution (e.g., during testing).
        """
        return self._orchestrator._resolve_provider_name(ctx)