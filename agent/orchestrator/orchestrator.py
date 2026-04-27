"""Orchestrator: the driver class that composes pipeline stages into a
conversation loop with streaming, event bus, and client management.

Phase 4 (Orchestrator Loop) per DESIGN.md:
- Drives Stage 1→2→3→4→5 in a loop
- Wires up EventBus for display/logging
- Wires up ClientManager and ProviderRegistry
- Emits iteration_start/end and provider_call_start/end events
- Handles streaming via StreamState
- Determines loop decision (CONTINUE / YIELD / TERMINATE)

The Orchestrator replaces the 3,426-line run_conversation() method.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Sequence

from agent.orchestrator.context import (
    ConversationContext,
    ParsedResponse,
    ProviderResult,
    StreamState,
)
from agent.orchestrator.events import EventBus, PipelineEvent
from agent.orchestrator.providers import ProviderRegistry
from agent.orchestrator.stages import (
    ContextManagementStage,
    PipelineResult,
    PreparedRequest,
    ProviderCallStage,
    RequestPrepStage,
    ResponseProcessingStage,
    StageAction,
    ToolDispatchStage,
    ToolDispatchResult,
)
from agent.orchestrator.tools import ToolExecutor

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the pipeline loop: RequestPrep → ProviderCall → ResponseProcesing
    → ToolDispatch → ContextManagement → loop decision.

    The Orchestrator is the top-level entry point. Call run(ctx) with a
    ConversationContext and get a PipelineResult back.

    Attributes:
        registry: ProviderRegistry for provider lookup
        event_bus: EventBus for observability
        tool_executor: ToolExecutor for tool dispatch
        model_name: default model name for API calls
        max_iterations: maximum loop iterations (default 90, matching AIAgent)
    """

    # Default iteration budget — same as AIAgent._max_iterations
    DEFAULT_MAX_ITERATIONS = 90
    DEFAULT_MODEL_NAME = "default-model"

    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        event_bus: Optional[EventBus] = None,
        tool_executor: Optional[ToolExecutor] = None,
        model_name: Optional[str] = None,
        max_iterations: Optional[int] = None,
        # Stage overrides for testing
        request_prep: Optional[RequestPrepStage] = None,
        provider_call: Optional[ProviderCallStage] = None,
        response_processing: Optional[ResponseProcessingStage] = None,
        tool_dispatch: Optional[ToolDispatchStage] = None,
        context_management: Optional[ContextManagementStage] = None,
    ) -> None:
        self.registry = registry or ProviderRegistry()
        self.event_bus = event_bus or EventBus()
        self.tool_executor = tool_executor or ToolExecutor()
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.max_iterations = max_iterations or self.DEFAULT_MAX_ITERATIONS

        # Pipeline stages — can be overridden for testing
        self.request_prep = request_prep or RequestPrepStage(model_name=self.model_name)
        self.provider_call = provider_call or ProviderCallStage(
            registry=self.registry,
            event_bus=self.event_bus,
        )
        self.response_processing = response_processing or ResponseProcessingStage()
        # ToolDispatchStage is a thin adapter — ToolExecutor does the real work
        self.tool_dispatch = tool_dispatch or ToolDispatchStage()
        self.context_management = context_management or ContextManagementStage()

        # Provider name resolution — override for testing
        self._resolve_provider_name = self._default_resolve_provider_name

    def _default_resolve_provider_name(self, ctx: ConversationContext) -> str:
        """Default provider resolution. Override for testing."""
        return "openai_compatible"

    def run(self, ctx: ConversationContext) -> PipelineResult:
        """Execute the full pipeline loop.

        Args:
            ctx: ConversationContext with messages, system prompt, etc.

        Returns:
            PipelineResult with the final response, iteration count,
            and interrupted flag.
        """
        iterations = 0

        # Respect max_iterations from context if set
        effective_max = ctx.max_iterations or self.max_iterations

        while True:
            iterations += 1

            # Check termination conditions before starting iteration
            if ctx.interrupt_event.is_set():
                # Build a minimal result for the interrupted state
                return PipelineResult(
                    context=ctx,
                    response=ParsedResponse(
                        message={"role": "assistant", "content": "[interrupted]"},
                        finish_reason="interrupted",
                    ),
                    iterations=iterations,
                    interrupted=True,
                )

            if iterations > effective_max:
                return PipelineResult(
                    context=ctx,
                    response=ParsedResponse(
                        message={"role": "assistant", "content": "[max_iterations_reached]"},
                        finish_reason="max_iterations",
                    ),
                    iterations=iterations,
                )

            # Emit iteration_start event
            self.event_bus.emit(PipelineEvent(
                kind="iteration_start",
                data={"iteration": iterations, "session_id": ctx.session_id},
                session_id=ctx.session_id,
            ))

            # Stage 1: Request Preparation
            prepared = self.request_prep.process(ctx)
            prepared.provider_name = self._resolve_provider_name(ctx)

            # Stage 2: Provider Call
            provider_result = self.provider_call.process(ctx, prepared)

            # Stage 3: Response Processing
            parsed = self.response_processing.process(ctx, provider_result)

            # Stage 4: Tool Dispatch (use ToolExecutor for actual dispatch)
            dispatch_result = self.tool_executor.dispatch(parsed)

            # Stage 5: Context Management
            action = self.context_management.process(ctx, dispatch_result.tool_results, parsed)

            # Emit iteration_end event
            self.event_bus.emit(PipelineEvent(
                kind="iteration_end",
                data={
                    "iteration": iterations,
                    "action": action.value,
                    "finish_reason": parsed.finish_reason,
                    "has_tool_calls": parsed.has_tool_calls,
                    "session_id": ctx.session_id,
                },
                session_id=ctx.session_id,
            ))

            # Loop decision
            if action == StageAction.YIELD:
                return PipelineResult(
                    context=ctx,
                    response=parsed,
                    iterations=iterations,
                    interrupted=ctx.interrupt_event.is_set(),
                    tool_results=dispatch_result.tool_results,
                )
            elif action == StageAction.TERMINATE:
                return PipelineResult(
                    context=ctx,
                    response=parsed,
                    iterations=iterations,
                )
            # CONTINUE → loop back for next iteration