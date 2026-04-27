"""Pipeline stages for the Kore Orchestrator.

Each stage has a single responsibility and a clear contract:
  - Input: a typed context object
  - Output: a typed result object
  - Side effects: only via injected callbacks or EventBus

Stages are callable classes with a process() method.
The Orchestrator loop drives Stage1 → Stage2 → Stage3 → Stage4 → Stage5,
then checks the loop decision and either continues or yields.

Design reference: DESIGN.md, "Pipeline Stages" section.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent.orchestrator.context import (
    ConversationContext,
    ParsedResponse,
    ProviderCapabilities,
    ProviderResult,
    StreamState,
    UsageInfo,
)
from agent.orchestrator.events import EventBus, PipelineEvent
from agent.orchestrator.providers import ProviderRegistry


# ============================================================================
# StageAction — loop decision enum
# ============================================================================

class StageAction(enum.Enum):
    """Drives the Orchestrator loop decision after each iteration.

    CONTINUE: tool results appended, loop back to Stage 1
    YIELD: final response ready, break out of loop
    TERMINATE: max iterations or fatal error, end conversation
    """

    CONTINUE = "continue"
    YIELD = "yield"
    TERMINATE = "terminate"


# ============================================================================
# PreparedRequest — Stage 1 output / Stage 2 input
# ============================================================================

@dataclass
class PreparedRequest:
    """The API-ready request produced by Stage 1 (Request Preparation).

    Contains everything needed to make an API call:
    - messages list with system prompt injected
    - api_kwargs (model, temperature, tools, etc.)
    - provider_name for registry lookup
    - cache_markers for prompt caching providers
    """

    messages: list[dict] = field(default_factory=list)
    api_kwargs: dict = field(default_factory=dict)
    provider_name: str = ""
    cache_markers: list[int] = field(default_factory=list)


# ============================================================================
# ToolDispatchResult — Stage 4 output
# ============================================================================

@dataclass
class ToolDispatchResult:
    """Result of Stage 4 (Tool Dispatch).

    tool_results: messages to append to conversation (role="tool")
    action: what the loop should do next
    """

    tool_results: list[dict] = field(default_factory=list)
    action: StageAction = StageAction.YIELD


# ============================================================================
# PipelineResult — final output of the pipeline loop
# ============================================================================

@dataclass
class PipelineResult:
    """Final output of the Orchestrator pipeline.

    Carries the response, updated context, and metadata about the run.
    """

    context: ConversationContext = field(default_factory=ConversationContext)
    response: Optional[ParsedResponse] = None
    iterations: int = 0
    interrupted: bool = False
    tool_results: list[dict] = field(default_factory=list)


# ============================================================================
# Stage 1: Request Preparation
# ============================================================================

class RequestPrepStage:
    """Build system prompt, prepare messages, select provider.

    Input: ConversationContext
    Output: PreparedRequest (messages + api_kwargs + provider_name)

    This stage:
    - Injects system prompt as first message
    - Prepares messages for API (sanitization handled by provider)
    - Sets up API kwargs (model, temperature, tools)
    - Marks cache breakpoints for prompt-caching providers
    """

    name: str = "request_prep"

    def __init__(self, model_name: str = "default-model", prepare_fn: Optional[Callable] = None) -> None:
        self._model_name = model_name
        self._prepare_fn = prepare_fn

    def process(self, ctx: ConversationContext) -> PreparedRequest:
        """Build a PreparedRequest from the conversation context.

        If a prepare_fn was injected, delegate to it (battle-test wiring).
        Otherwise use the default logic.
        """
        if self._prepare_fn is not None:
            return self._prepare_fn(ctx)

        messages = list(ctx.messages)

        # Prepend system prompt
        if ctx.system_prompt:
            messages = [{"role": "system", "content": ctx.system_prompt}] + messages

        # Build api_kwargs
        api_kwargs: dict = {
            "model": self._model_name,
        }

        if ctx.tools:
            api_kwargs["tools"] = ctx.tools

        return PreparedRequest(
            messages=messages,
            api_kwargs=api_kwargs,
            provider_name="openai_compatible",  # default, overridden by Orchestrator
            cache_markers=[],
        )


# ============================================================================
# Stage 2: Provider Call
# ============================================================================

class ProviderCallStage:
    """Execute the API call via ProviderProtocol.

    Input: ConversationContext + PreparedRequest
    Output: ProviderResult

    This stage:
    - Looks up provider from registry by provider_name
    - Calls provider.execute()
    - Emits provider_call_start/end events on EventBus
    - Catches errors and wraps them in ProviderResult with should_fallback
    """

    name: str = "provider_call"

    def __init__(
        self,
        registry: Optional[ProviderRegistry] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus

    def process(
        self,
        ctx: ConversationContext,
        prepared: PreparedRequest,
    ) -> ProviderResult:
        """Execute the provider call."""
        provider_name = prepared.provider_name

        # Emit start event
        if self._event_bus:
            self._event_bus.emit(PipelineEvent(
                kind="provider_call_start",
                data={"provider": provider_name},
                session_id=ctx.session_id,
            ))

        # Get provider from registry
        if self._registry is None:
            return ProviderResult(
                error=RuntimeError(f"No provider registry configured"),
                should_fallback=True,
            )

        provider = self._registry.get(provider_name)
        if provider is None:
            result = ProviderResult(
                error=RuntimeError(f"No provider registered as '{provider_name}'"),
                should_fallback=True,
            )
            if self._event_bus:
                self._event_bus.emit(PipelineEvent(
                    kind="provider_call_end",
                    data={"provider": provider_name, "error": str(result.error)},
                    session_id=ctx.session_id,
                ))
            return result

        # Execute the provider call
        try:
            request = provider.prepare_request(ctx)
            result = provider.execute(request)
        except Exception as exc:
            result = ProviderResult(
                error=exc,
                should_fallback=True,
            )

        # Emit end event
        if self._event_bus:
            self._event_bus.emit(PipelineEvent(
                kind="provider_call_end",
                data={
                    "provider": provider_name,
                    "finish_reason": result.finish_reason,
                    "error": str(result.error) if result.error else None,
                },
                session_id=ctx.session_id,
            ))

        return result


# ============================================================================
# Stage 3: Response Processing
# ============================================================================

class ResponseProcessingStage:
    """Parse ProviderResult into ParsedResponse.

    Input: ConversationContext + ProviderResult
    Output: ParsedResponse

    This stage:
    - Extracts assistant message from response
    - Extracts tool calls
    - Extracts reasoning content
    - Handles empty responses
    - Handles provider errors (wraps into error ParsedResponse)
    """

    name: str = "response_processing"

    def process(
        self,
        ctx: ConversationContext,
        provider_result: ProviderResult,
    ) -> ParsedResponse:
        """Parse the provider response into a canonical ParsedResponse."""
        # Handle provider errors
        if provider_result.error is not None:
            return ParsedResponse(
                message={"role": "assistant", "content": f"Error: {provider_result.error}"},
                tool_calls=[],
                finish_reason="error",
                usage=provider_result.usage,
            )

        response = provider_result.response or {}
        choices = response.get("choices", [])

        if not choices:
            # Empty response
            return ParsedResponse(
                message={"role": "assistant", "content": ""},
                tool_calls=[],
                finish_reason=provider_result.finish_reason or "stop",
                usage=provider_result.usage,
            )

        choice = choices[0]
        message = choice.get("message", {})

        # Extract tool calls
        tool_calls = message.get("tool_calls", [])

        # Extract reasoning content (may be None)
        reasoning_content = message.get("reasoning_content")

        # Build canonical message dict
        canonical_message = {
            "role": "assistant",
            "content": message.get("content", ""),
        }
        if tool_calls:
            canonical_message["tool_calls"] = tool_calls

        # Extract finish reason
        finish_reason = choice.get("finish_reason", provider_result.finish_reason or "stop")

        # Extract usage (from response dict, falling back to provider_result.usage)
        usage_raw = response.get("usage", {})
        usage = None
        if usage_raw:
            usage = UsageInfo(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
                cache_read_tokens=usage_raw.get("cache_read_tokens", 0),
                cache_creation_tokens=usage_raw.get("cache_creation_tokens", 0),
            )
        elif provider_result.usage is not None:
            usage = provider_result.usage

        return ParsedResponse(
            message=canonical_message,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
            usage=usage,
        )


# ============================================================================
# Stage 4: Tool Dispatch
# ============================================================================

class ToolDispatchStage:
    """Route tool calls to handlers and collect results.

    Input: ConversationContext + ParsedResponse
    Output: ToolDispatchResult (tool result messages + action)

    This stage:
    - Detects tool calls in ParsedResponse
    - Routes each to the tool handler
    - Collects tool result messages
    - Determines loop action: CONTINUE (has tool calls) or YIELD (no tool calls)
    """

    name: str = "tool_dispatch"

    def __init__(
        self,
        tool_handler: Optional[Callable] = None,
    ) -> None:
        """tool_handler: callable(name, args) -> result string or dict.
        If None, a default handler that returns "ok" is used.
        """
        self._tool_handler = tool_handler or self._default_handler

    @staticmethod
    def _default_handler(name: str, args: dict) -> str:
        return "ok"

    def process(
        self,
        ctx: ConversationContext,
        parsed: ParsedResponse,
    ) -> ToolDispatchResult:
        """Dispatch tool calls and return results."""
        if not parsed.has_tool_calls:
            return ToolDispatchResult(
                tool_results=[],
                action=StageAction.YIELD,
            )

        tool_results: list[dict] = []
        for tc in parsed.tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            args_str = func.get("arguments", "{}")
            call_id = tc.get("id", "")

            # Parse args if string
            import json
            try:
                if isinstance(args_str, str):
                    args = json.loads(args_str) if args_str else {}
                else:
                    args = args_str
            except json.JSONDecodeError:
                args = {}

            # Call the handler
            try:
                result = self._tool_handler(name, args)
                if isinstance(result, dict):
                    content = json.dumps(result)
                else:
                    content = str(result)
            except Exception as exc:
                content = f"Error: {exc}"

            tool_results.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": content,
            })

        return ToolDispatchResult(
            tool_results=tool_results,
            action=StageAction.CONTINUE,
        )


# ============================================================================
# Stage 5: Context Management
# ============================================================================

class ContextManagementStage:
    """Append tool results, check context limits, persist state.

    Input: ConversationContext + ParsedResponse + tool_results
    Output: StageAction (CONTINUE / YIELD / TERMINATE)

    This stage:
    - Appends assistant message and tool results to conversation
    - Increments iteration counter
    - Checks max iterations
    - Checks interrupt event
    - Determines whether to loop or break
    """

    name: str = "context_management"

    def process(
        self,
        ctx: ConversationContext,
        tool_results: list[dict],
        parsed: ParsedResponse,
    ) -> StageAction:
        """Append results and decide whether to continue, yield, or terminate."""
        # Always increment iteration
        ctx.iteration += 1

        # Check max iterations
        if ctx.iteration >= ctx.max_iterations:
            return StageAction.TERMINATE

        # Check interrupt
        if ctx.interrupt_event.is_set():
            return StageAction.YIELD

        # If no tool results, we're done — yield the response
        if not tool_results:
            return StageAction.YIELD

        # Append assistant message to conversation
        if parsed.message:
            ctx.messages.append(parsed.message)

        # Append tool results
        ctx.messages.extend(tool_results)

        # Tool results present — loop back
        return StageAction.CONTINUE