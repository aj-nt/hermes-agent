"""Thin ProviderProtocol adapters wrapping existing provider code.

Each adapter implements the ProviderProtocol interface:
- capabilities: ProviderCapabilities for this provider type
- prepare_request(ctx): Build provider-specific API kwargs
- execute(request): Make the API call, return ProviderResult
- parse_response(result, ctx): Convert to ParsedResponse

Phase 2 strategy: "Wrap first, redesign later." Adapters delegate to
existing adapter modules and AIAgent methods. They do NOT reimplement
provider logic. The actual delegation functions are injected for
testability (same DI pattern as ClientManager).

Production wiring uses from_defaults() factories.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from agent.orchestrator.context import (
    ConversationContext,
    ParsedResponse,
    ProviderCapabilities,
    ProviderResult,
    UsageInfo,
)
from agent.orchestrator.providers import ProviderProtocol

logger = logging.getLogger(__name__)

# Type aliases for injected functions
CallFn = Callable[..., Any]
ParseFn = Callable[..., ParsedResponse]
PrepareFn = Callable[..., dict]


# ============================================================================
# RequestConfig — provider flags for chat_completions
# ============================================================================

@dataclass
class RequestConfig:
    """Bundles all the provider-detection flags and model-level config
    that AIAgent._build_api_kwargs currently computes from self.* attributes.

    This decouples OpenAICompatibleProvider from AIAgent — the provider
    receives a RequestConfig instead of reading self.* directly.

    Step 3a: initially carries the same flags that _build_api_kwargs
    computes inline. Future steps may simplify this as more logic
    moves into the transport layer.
    """

    # --- Model/endpoint identity ---
    model: str = ""
    base_url: str = ""

    # --- Token limits ---
    max_tokens: Optional[int] = None
    ephemeral_max_output_tokens: Optional[int] = None
    ollama_num_ctx: Optional[int] = None

    # --- Provider detection flags ---
    is_openrouter: bool = False
    is_nous: bool = False
    is_qwen_portal: bool = False
    is_github_models: bool = False
    is_nvidia_nim: bool = False
    is_kimi: bool = False
    is_custom_provider: bool = False

    # --- Temperature ---
    fixed_temperature: Optional[Any] = None
    omit_temperature: bool = False

    # --- Reasoning ---
    reasoning_config: Optional[Dict[str, Any]] = None
    supports_reasoning: bool = False
    github_reasoning_extra: Optional[Dict[str, Any]] = None

    # --- Request overrides ---
    request_overrides: Optional[Dict[str, Any]] = None

    # --- Provider preferences (OpenRouter) ---
    provider_preferences: Optional[Dict[str, Any]] = None

    # --- Anthropic on OpenRouter/Nous ---
    anthropic_max_output: Optional[int] = None

    # --- Qwen session metadata ---
    qwen_session_metadata: Optional[Dict[str, Any]] = None

    # --- Callbacks (injected from AIAgent) ---
    max_tokens_param_fn: Optional[Callable] = None
    qwen_prepare_fn: Optional[Callable] = None
    qwen_prepare_inplace_fn: Optional[Callable] = None


# ============================================================================
# OpenAICompatibleProvider
# ============================================================================

class OpenAICompatibleProvider(ProviderProtocol):
    """Wraps the chat_completions API path.

    This is the default/fallback path used by:
    - Ollama (local GLM/Qwen/etc.)
    - OpenRouter
    - Direct OpenAI
    - Azure OpenAI
    - Any OpenAI-compatible endpoint

    Capabilities vary by whether the endpoint is local or cloud.
    """

    def __init__(
        self,
        base_url: str = "",
        model_name: str = "",
        is_local: Optional[bool] = None,
        call_fn: Optional[CallFn] = None,
        parse_fn: Optional[ParseFn] = None,
        request_config: Optional[RequestConfig] = None,
        transport: Any = None,
    ) -> None:
        self._base_url = base_url
        self._model_name = model_name
        # Auto-detect local endpoint if not specified
        if is_local is not None:
            self._is_local = is_local
        else:
            self._is_local = self._detect_local(base_url)
        # Injected delegation functions
        self._call_fn = call_fn
        self._parse_fn = parse_fn or self._default_parse_response
        # RequestConfig — provider flags for build_kwargs
        self._request_config = request_config or RequestConfig(
            model=model_name,
            base_url=base_url,
        )
        # Transport — ChatCompletionsTransport instance for build_kwargs
        self._transport = transport

    @classmethod
    def from_agent(cls, agent: Any) -> "OpenAICompatibleProvider":
        """Factory: create an OpenAICompatibleProvider from an AIAgent instance.

        Computes all provider-detection flags from agent attributes,
        matching the logic currently in AIAgent._build_api_kwargs()
        for the chat_completions branch.
        """
        from utils import base_url_host_matches
        from agent.kore.url_helpers import is_qwen_portal

        base_url = getattr(agent, "base_url", "") or ""
        base_url_lower = base_url.lower()

        # Provider detection flags (mirrors _build_api_kwargs lines 5742-5798)
        is_qwen = is_qwen_portal(base_url_lower)
        is_or = getattr(agent, "_is_openrouter_url", lambda: False)()
        is_gh = (
            base_url_host_matches(base_url_lower, "models.github.ai")
            or base_url_host_matches(base_url_lower, "api.githubcopilot.com")
        )
        is_nous = "nousresearch" in base_url_lower
        is_nvidia = "integrate.api.nvidia.com" in base_url_lower
        is_kimi = (
            base_url_host_matches(base_url, "api.kimi.com")
            or base_url_host_matches(base_url, "moonshot.ai")
            or base_url_host_matches(base_url, "moonshot.cn")
        )

        # Provider preferences (OpenRouter-specific)
        prefs: Dict[str, Any] = {}
        if getattr(agent, "providers_allowed", None):
            prefs["only"] = agent.providers_allowed
        if getattr(agent, "providers_ignored", None):
            prefs["ignore"] = agent.providers_ignored
        if getattr(agent, "providers_order", None):
            prefs["order"] = agent.providers_order
        if getattr(agent, "provider_sort", None):
            prefs["sort"] = agent.provider_sort
        if getattr(agent, "provider_require_parameters", None):
            prefs["require_parameters"] = True
        if getattr(agent, "provider_data_collection", None):
            prefs["data_collection"] = agent.provider_data_collection

        # Anthropic max output for Claude on OpenRouter/Nous
        ant_max = None
        if (is_or or is_nous) and "claude" in (getattr(agent, "model", "") or "").lower():
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                ant_max = _get_anthropic_max_output(agent.model)
            except Exception:
                pass

        # Qwen session metadata
        import uuid
        qwen_meta = None
        if is_qwen:
            qwen_meta = {
                "sessionId": getattr(agent, "session_id", None) or "hermes",
                "promptId": str(uuid.uuid4()),
            }

        # Temperature
        fixed_temp = None
        omit_temp = False
        try:
            from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
            ft = _fixed_temperature_for_model(agent.model, base_url)
            omit_temp = ft is OMIT_TEMPERATURE
            fixed_temp = ft if not omit_temp else None
        except Exception:
            pass

        # Reasoning
        supports_reasoning = False
        github_reasoning_extra = None
        try:
            from agent.kore.reasoning import supports_reasoning_extra_body
            supports_reasoning = supports_reasoning_extra_body(agent.model, base_url_lower)
        except Exception:
            pass
        if is_gh:
            try:
                from agent.kore.reasoning import github_models_reasoning_extra_body
                github_reasoning_extra = github_models_reasoning_extra_body(
                    agent.model, agent.reasoning_config
                )
            except Exception:
                pass

        # Ephemeral max output override
        ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)

        rc = RequestConfig(
            model=agent.model,
            base_url=base_url,
            max_tokens=getattr(agent, "max_tokens", None),
            ephemeral_max_output_tokens=ephemeral_out,
            ollama_num_ctx=getattr(agent, "_ollama_num_ctx", None),
            is_openrouter=is_or,
            is_nous=is_nous,
            is_qwen_portal=is_qwen,
            is_github_models=is_gh,
            is_nvidia_nim=is_nvidia,
            is_kimi=is_kimi,
            is_custom_provider=getattr(agent, "provider", "") == "custom",
            fixed_temperature=fixed_temp,
            omit_temperature=omit_temp,
            reasoning_config=getattr(agent, "reasoning_config", None),
            supports_reasoning=supports_reasoning,
            github_reasoning_extra=github_reasoning_extra,
            request_overrides=getattr(agent, "request_overrides", None),
            provider_preferences=prefs or None,
            anthropic_max_output=ant_max,
            qwen_session_metadata=qwen_meta,
            max_tokens_param_fn=getattr(agent, "_max_tokens_param", None),
            qwen_prepare_fn=(
                agent._qwen_prepare_chat_messages if is_qwen else None
            ),
            qwen_prepare_inplace_fn=(
                agent._qwen_prepare_chat_messages_inplace if is_qwen else None
            ),
        )

        # Get transport from agent's cache
        transport = None
        try:
            transport = agent._get_transport()
        except Exception:
            pass

        return cls(
            base_url=base_url,
            model_name=agent.model,
            is_local=cls._detect_local(base_url) if base_url else False,
            request_config=rc,
            transport=transport,
        )

    @staticmethod
    def _detect_local(base_url: str) -> bool:
        """Heuristic: localhost/127.0.0.1/192.168.x.x is a local endpoint."""
        if not base_url:
            return False
        lower = base_url.lower()
        return any(pattern in lower for pattern in (
            "localhost", "127.0.0.1", "0.0.0.0", "192.168.", "::1",
        ))

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            requires_prompt_caching=False,
            requires_message_sanitization=self._is_local,
            requires_custom_stop_handling=self._is_local,
            supports_responses_api=False,
            cache_breakpoint_strategy="none",
        )

    def prepare_request(self, ctx: ConversationContext) -> dict:
        """Build chat completions API kwargs from context and RequestConfig.

        If a transport (ChatCompletionsTransport) is available, delegates
        to transport.build_kwargs() with all the provider flags from
        RequestConfig — matching the output of AIAgent._build_api_kwargs().

        Falls back to a minimal dict if no transport is injected.
        """
        if self._transport is not None and hasattr(self._transport, "build_kwargs"):
            rc = self._request_config
            return self._transport.build_kwargs(
                model=rc.model or self._model_name,
                messages=ctx.messages,
                tools=ctx.tools or None,
                timeout=None,
                max_tokens=rc.max_tokens,
                ephemeral_max_output_tokens=rc.ephemeral_max_output_tokens,
                max_tokens_param_fn=rc.max_tokens_param_fn,
                reasoning_config=rc.reasoning_config,
                request_overrides=rc.request_overrides,
                session_id=ctx.session_id,
                model_lower=(rc.model or self._model_name or "").lower(),
                is_openrouter=rc.is_openrouter,
                is_nous=rc.is_nous,
                is_qwen_portal=rc.is_qwen_portal,
                is_github_models=rc.is_github_models,
                is_nvidia_nim=rc.is_nvidia_nim,
                is_kimi=rc.is_kimi,
                is_custom_provider=rc.is_custom_provider,
                ollama_num_ctx=rc.ollama_num_ctx,
                provider_preferences=rc.provider_preferences,
                qwen_prepare_fn=rc.qwen_prepare_fn,
                qwen_prepare_inplace_fn=rc.qwen_prepare_inplace_fn,
                qwen_session_metadata=rc.qwen_session_metadata,
                fixed_temperature=rc.fixed_temperature,
                omit_temperature=rc.omit_temperature,
                supports_reasoning=rc.supports_reasoning,
                github_reasoning_extra=rc.github_reasoning_extra,
                anthropic_max_output=rc.anthropic_max_output,
            )

        # Minimal fallback — matches old behavior
        kwargs: Dict[str, Any] = {
            "model": self._model_name,
            "messages": ctx.messages,
        }
        if ctx.tools:
            kwargs["tools"] = ctx.tools
        return kwargs

    def execute(self, request: dict) -> ProviderResult:
        """Execute the API call via injected call_fn.

        If no call_fn is set, returns a stub ProviderResult.
        Production wiring injects the real client.chat.completions.create.
        """
        if self._call_fn is not None:
            try:
                response = self._call_fn(**request)
                return ProviderResult(
                    response=response,
                    should_retry=False,
                    should_fallback=False,
                )
            except Exception as exc:
                return ProviderResult(
                    error=exc,
                    should_retry=False,
                    should_fallback=False,
                )
        # Stub — no call_fn injected
        return ProviderResult()

    def parse_response(
        self, result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        """Parse the response via injected or default parse logic."""
        return self._parse_fn(result, ctx)

    @staticmethod
    def _default_parse_response(
        result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        """Default parse: extract from ProviderResult fields.

        Used when no custom parse_fn is injected. Will be enhanced
        when wired to real response processing in Phase 3.
        """
        return ParsedResponse()


# ============================================================================
# Streaming Support — Step 3b
# ============================================================================

@dataclass
class StreamCallbacks:
    """Callbacks for streaming execution events.

    Mirrors the callbacks currently on AIAgent (stream_delta_callback,
    reasoning_callback, tool_gen_callback) plus the on_first_delta
    closure used in _interruptible_streaming_api_call.
    """

    stream_delta: Optional[Callable[[str], None]] = None
    reasoning_delta: Optional[Callable[[str], None]] = None
    tool_gen_started: Optional[Callable[[str], None]] = None
    first_delta: Optional[Callable[[], None]] = None
    activity_touch: Optional[Callable[[str], None]] = None
    on_client_created: Optional[Callable[[Any], None]] = None


@dataclass
class StreamResult:
    """Accumulated result from a streaming chat completions call.

    Holds all the data extracted from SSE chunks, then converts to a
    SimpleNamespace matching the non-streaming response shape that the
    rest of the agent loop expects.
    """

    content: Optional[str] = None
    tool_calls: Optional[List[SimpleNamespace]] = None
    reasoning: Optional[str] = None
    finish_reason: str = "stop"
    model_name: Optional[str] = None
    usage: Any = None
    partial_tool_names: List[str] = field(default_factory=list)

    def to_response(self) -> SimpleNamespace:
        """Convert to a SimpleNamespace matching the non-streaming OpenAI response shape.

        This is what _call_chat_completions returns — the agent loop
        processes it identically regardless of streaming vs non-streaming.
        """
        # Build tool_calls list (or None)
        mock_tool_calls = None
        has_truncated_tool_args = False
        if self.tool_calls:
            mock_tool_calls = []
            for tc in self.tool_calls:
                arguments = tc.function.arguments
                tool_name = tc.function.name or "?"
                if arguments and arguments.strip():
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        # Attempt repair before flagging as truncated.
                        # Models like GLM-5.1 produce trailing commas,
                        # unclosed brackets, Python None, etc.
                        from agent.sanitization import _repair_tool_call_arguments
                        repaired = _repair_tool_call_arguments(arguments, tool_name)
                        if repaired != "{}":
                            arguments = repaired
                        else:
                            has_truncated_tool_args = True

                mock_tool_calls.append(SimpleNamespace(
                    id=tc.id,
                    type=tc.type,
                    extra_content=tc.extra_content if hasattr(tc, "extra_content") else None,
                    function=SimpleNamespace(
                        name=tc.function.name,
                        arguments=arguments,
                    ),
                ))

        effective_finish_reason = self.finish_reason or "stop"
        if has_truncated_tool_args:
            effective_finish_reason = "length"

        mock_message = SimpleNamespace(
            role="assistant",
            content=self.content,
            tool_calls=mock_tool_calls,
            reasoning_content=self.reasoning,
        )
        mock_choice = SimpleNamespace(
            index=0,
            message=mock_message,
            finish_reason=effective_finish_reason,
        )
        return SimpleNamespace(
            id="stream-" + str(uuid.uuid4()),
            model=self.model_name,
            choices=[mock_choice],
            usage=self.usage,
        )


class StreamingChatCompletionsExecutor:
    """Encapsulates the streaming call logic from _call_chat_completions.

    This is Step 3b of the Kore redesign: extracting the ~250-line
    _call_chat_completions closure from run_agent.py into a testable,
    injectable class.

    The executor:
    1. Builds stream kwargs (stream=True, stream_options, timeouts)
    2. Creates a per-request OpenAI client via injected factory
    3. Iterates SSE chunks, accumulating content/tool_calls/reasoning/usage
    4. Fires callbacks (stream_delta, reasoning, tool_gen, first_delta, activity)
    5. Handles Ollama index-reuse for parallel tool calls
    6. Repairs truncated tool-call JSON arguments
    7. Returns a StreamResult that can be converted to a SimpleNamespace
       matching the non-streaming response shape

    The outer stale-stream detector, retry loop, and thread management
    remain in run_agent.py for now — they'll be extracted in later steps.
    """

    def __init__(
        self,
        client_factory: Callable[[], Any],
        close_client_fn: Callable[[Any], None],
        request_config: RequestConfig,
        base_url: str,
        callbacks: StreamCallbacks,
        interrupt_check: Callable[[], bool],
        is_local: bool = False,
        model: str = "",
        capture_rate_limits_fn: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._client_factory = client_factory
        self._close_client_fn = close_client_fn
        self._request_config = request_config
        self._base_url = base_url
        self._callbacks = callbacks
        self._interrupt_check = interrupt_check
        self._is_local = is_local
        self._model = model
        self._capture_rate_limits_fn = capture_rate_limits_fn
        # The per-request OpenAI client, set during execute_streaming()
        # and cleared in the finally block.  External code (e.g. the
        # stale-stream detector) can read this during streaming.
        self.request_client: Any = None

    @classmethod
    def from_agent(cls, agent: Any) -> "StreamingChatCompletionsExecutor":
        """Factory: create a StreamingChatCompletionsExecutor from an AIAgent instance.

        Injects the agent's client factory, close function, fire methods,
        and interrupt check as callbacks. The RequestConfig is built from
        agent attributes to match _build_api_kwargs.
        """
        from agent.model_metadata import is_local_endpoint as _is_local_fn

        base_url = getattr(agent, "base_url", "") or ""
        is_local = _is_local_fn(base_url) if base_url else False

        # Build RequestConfig from agent (mirrors OpenAICompatibleProvider.from_agent)
        rc = RequestConfig(
            model=getattr(agent, "model", ""),
            base_url=base_url,
            max_tokens=getattr(agent, "max_tokens", None),
            ephemeral_max_output_tokens=getattr(agent, "_ephemeral_max_output_tokens", None),
            ollama_num_ctx=getattr(agent, "_ollama_num_ctx", None),
            reasoning_config=getattr(agent, "reasoning_config", None),
            request_overrides=getattr(agent, "request_overrides", None),
        )

        # Wire the stream_delta callback through _fire_stream_delta
        # so it handles _stream_needs_break paragraph-break logic
        def _stream_delta_wrapper(text: str) -> None:
            agent._fire_stream_delta(text)

        callbacks = StreamCallbacks(
            stream_delta=_stream_delta_wrapper,
            reasoning_delta=lambda text: agent._fire_reasoning_delta(text),
            tool_gen_started=lambda name: agent._fire_tool_gen_started(name),
            activity_touch=lambda msg: agent._touch_activity(msg),
        )

        return cls(
            client_factory=lambda: agent._create_request_openai_client(
                reason="chat_completion_stream_request"
            ),
            close_client_fn=lambda client: agent._close_request_openai_client(
                client, reason="stream_request_complete"
            ),
            request_config=rc,
            base_url=base_url,
            callbacks=callbacks,
            interrupt_check=lambda: agent._interrupt_requested,
            is_local=is_local,
            model=getattr(agent, "model", ""),
            capture_rate_limits_fn=lambda resp: agent._capture_rate_limits(resp),
        )

    def execute_streaming(self, api_kwargs: dict) -> StreamResult:
        """Execute a streaming chat completions call and return accumulated result.

        This is the core logic from _call_chat_completions (run_agent.py:4435-4686).
        It does NOT include the outer thread/retry/stale-detector — those
        remain in _interruptible_streaming_api_call for now.

        Args:
            api_kwargs: The kwargs dict from prepare_request (or _build_api_kwargs).
                        stream=True and stream_options will be added by this method.

        Returns:
            StreamResult with accumulated content, tool_calls, reasoning, etc.
        """
        import httpx as _httpx
        from agent.model_metadata import is_local_endpoint

        # Per-provider / per-model request_timeout_seconds
        _provider_timeout_cfg = get_provider_request_timeout(
            self._request_config.model, self._base_url
        )
        _base_timeout = (
            _provider_timeout_cfg
            if _provider_timeout_cfg is not None
            else float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
        )

        # Read timeout: config wins. Otherwise use HERMES_STREAM_READ_TIMEOUT
        # (default 120s) for cloud providers.
        if _provider_timeout_cfg is not None:
            _stream_read_timeout = _provider_timeout_cfg
        else:
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            # Local providers (Ollama, llama.cpp) can take minutes for
            # prefill on large contexts. Auto-increase read timeout.
            if _stream_read_timeout == 120.0 and self._base_url and self._is_local:
                _stream_read_timeout = _base_timeout
                logger.debug(
                    "Local provider detected (%s) — stream read timeout raised to %.0fs",
                    self._base_url, _stream_read_timeout,
                )

        stream_kwargs = {
            **api_kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": _httpx.Timeout(
                connect=30.0,
                read=_stream_read_timeout,
                write=_base_timeout,
                pool=30.0,
            ),
        }

        # Create per-request client and expose for stale-stream detector
        self.request_client = self._client_factory()
        request_client = self.request_client
        # Notify outer scope (e.g. stale-stream detector) of the new client
        if self._callbacks.on_client_created is not None:
            self._callbacks.on_client_created(self.request_client)
        try:
            stream = request_client.chat.completions.create(**stream_kwargs)

            # Capture rate limit headers from the initial HTTP response
            if self._capture_rate_limits_fn is not None:
                self._capture_rate_limits_fn(getattr(stream, "response", None))

            # Touch activity
            if self._callbacks.activity_touch is not None:
                self._callbacks.activity_touch("waiting for provider response (streaming)")

            # --- Accumulate chunks ---
            content_parts: list = []
            tool_calls_acc: dict = {}
            tool_gen_notified: set = set()
            # Ollama reuses index 0 for parallel tool calls
            _last_id_at_idx: dict = {}
            _active_slot_by_idx: dict = {}
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None
            first_delta_fired = False

            for chunk in stream:
                if self._callbacks.activity_touch is not None:
                    self._callbacks.activity_touch("receiving stream response")

                if self._interrupt_check():
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # Usage in final chunk with empty choices
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # --- Reasoning content ---
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    if not first_delta_fired:
                        first_delta_fired = True
                        if self._callbacks.first_delta is not None:
                            try:
                                self._callbacks.first_delta()
                            except Exception:
                                pass
                    if self._callbacks.reasoning_delta is not None:
                        try:
                            self._callbacks.reasoning_delta(reasoning_text)
                        except Exception:
                            pass

                # --- Text content ---
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        # No tool calls active — stream text normally
                        if not first_delta_fired:
                            first_delta_fired = True
                            if self._callbacks.first_delta is not None:
                                try:
                                    self._callbacks.first_delta()
                                except Exception:
                                    pass
                        if self._callbacks.stream_delta is not None:
                            try:
                                self._callbacks.stream_delta(delta.content)
                            except Exception:
                                pass
                    else:
                        # Tool calls suppress regular content streaming.
                        # But reasoning tags in suppressed content should still
                        # reach the display via the stream delta callback.
                        if self._callbacks.stream_delta is not None:
                            try:
                                self._callbacks.stream_delta(delta.content)
                            except Exception:
                                pass

                # --- Tool call deltas ---
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        raw_idx = tc_delta.index if tc_delta.index is not None else 0
                        delta_id = tc_delta.id or ""

                        # Ollama fix: detect new tool call reusing same index
                        if raw_idx not in _active_slot_by_idx:
                            _active_slot_by_idx[raw_idx] = raw_idx
                        if (
                            delta_id
                            and raw_idx in _last_id_at_idx
                            and delta_id != _last_id_at_idx[raw_idx]
                        ):
                            new_slot = max(tool_calls_acc, default=-1) + 1
                            _active_slot_by_idx[raw_idx] = new_slot
                        if delta_id:
                            _last_id_at_idx[raw_idx] = delta_id
                        idx = _active_slot_by_idx[raw_idx]

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "extra_content": None,
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                # Assignment, not +=. Function names are atomic
                                # identifiers delivered complete in the first
                                # chunk (OpenAI spec). Some providers (MiniMax)
                                # resend the full name in every chunk.
                                entry["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                        extra = getattr(tc_delta, "extra_content", None)
                        if extra is None and hasattr(tc_delta, "model_extra"):
                            extra = (tc_delta.model_extra or {}).get("extra_content")
                        if extra is not None:
                            if hasattr(extra, "model_dump"):
                                extra = extra.model_dump()
                            entry["extra_content"] = extra

                        # Fire once per tool when the full name is available
                        name = entry["function"]["name"]
                        if name and idx not in tool_gen_notified:
                            tool_gen_notified.add(idx)
                            if not first_delta_fired:
                                first_delta_fired = True
                                if self._callbacks.first_delta is not None:
                                    try:
                                        self._callbacks.first_delta()
                                    except Exception:
                                        pass
                            if self._callbacks.tool_gen_started is not None:
                                try:
                                    self._callbacks.tool_gen_started(name)
                                except Exception:
                                    pass

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Usage in the final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

        finally:
            # Clear the exposed client reference and close it
            self.request_client = None
            self._close_client_fn(request_client)

        # Build the StreamResult
        full_content = "".join(content_parts) or None
        full_reasoning = "".join(reasoning_parts) or None

        # Convert tool_calls_acc to list of SimpleNamespace
        result_tool_calls = None
        if tool_calls_acc:
            result_tool_calls = []
            for idx in sorted(tool_calls_acc):
                tc = tool_calls_acc[idx]
                result_tool_calls.append(SimpleNamespace(
                    id=tc["id"],
                    type=tc["type"],
                    extra_content=tc.get("extra_content"),
                    function=SimpleNamespace(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                ))

        return StreamResult(
            content=full_content,
            tool_calls=result_tool_calls,
            reasoning=full_reasoning,
            finish_reason=finish_reason or "stop",
            model_name=model_name,
            usage=usage_obj,
            partial_tool_names=list(tool_gen_notified) if tool_gen_notified else [],
        )


def get_provider_request_timeout(model: str, base_url: str) -> Optional[float]:
    """Get per-provider request timeout from config.

    Delegates to AIAgent._get_provider_request_timeout via import
    if available, otherwise returns None (use default).
    """
    try:
        # Late import to avoid circular dependency
        from run_agent import _get_provider_request_timeout
        return _get_provider_request_timeout(model, base_url)
    except ImportError:
        return None


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama/GLM-specific provider extending OpenAICompatibleProvider.

    Inherits the chat_completions path but adds GLM-specific capabilities:
    - Custom stop handling (GLM stop-to-length heuristic)
    - Message sanitization for local endpoints
    - No reasoning tokens
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "",
        call_fn: Optional[CallFn] = None,
        parse_fn: Optional[ParseFn] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            is_local=True,  # Ollama is always local
            call_fn=call_fn,
            parse_fn=parse_fn,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            requires_prompt_caching=False,
            requires_message_sanitization=True,
            requires_custom_stop_handling=True,
            supports_reasoning_tokens=False,
            supports_responses_api=False,
            cache_breakpoint_strategy="none",
        )


# ============================================================================
# AnthropicProvider
# ============================================================================

class AnthropicProvider(ProviderProtocol):
    """Wraps the anthropic_messages API path.

    Anthropic uses the Messages API with:
    - Prompt caching breakpoints (4-point or 3-point strategy)
    - Reasoning tokens (extended thinking)
    - Unique content preprocessing (image URL fallbacks)
    """

    def __init__(
        self,
        cache_strategy: str = "anthropic_4point",
        call_fn: Optional[CallFn] = None,
        parse_fn: Optional[ParseFn] = None,
    ) -> None:
        self._cache_strategy = cache_strategy
        self._call_fn = call_fn
        self._parse_fn = parse_fn or self._default_parse_response

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            requires_prompt_caching=True,
            supports_reasoning_tokens=True,
            supports_responses_api=False,
            requires_message_sanitization=False,
            requires_custom_stop_handling=False,
            cache_breakpoint_strategy=self._cache_strategy,
        )

    def prepare_request(self, ctx: ConversationContext) -> dict:
        """Build Anthropic Messages API kwargs."""
        kwargs = {
            "messages": ctx.messages,
            "system": ctx.system_prompt,
            "max_tokens": 8192,
        }
        if ctx.tools:
            kwargs["tools"] = ctx.tools
        return kwargs

    def execute(self, request: dict) -> ProviderResult:
        if self._call_fn is not None:
            try:
                response = self._call_fn(**request)
                return ProviderResult(response=response)
            except Exception as exc:
                return ProviderResult(error=exc)
        return ProviderResult()

    def parse_response(
        self, result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        return self._parse_fn(result, ctx)

    @staticmethod
    def _default_parse_response(
        result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        return ParsedResponse()


# ============================================================================
# BedrockProvider
# ============================================================================

class BedrockProvider(ProviderProtocol):
    """Wraps the bedrock_converse API path.

    AWS Bedrock uses boto3 converse() / converse_stream()
    with its own message format and AWS credential management.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        call_fn: Optional[CallFn] = None,
        parse_fn: Optional[ParseFn] = None,
    ) -> None:
        self._region = region
        self._call_fn = call_fn
        self._parse_fn = parse_fn or self._default_parse_response

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            requires_prompt_caching=False,
            supports_responses_api=False,
            requires_message_sanitization=False,
            requires_custom_stop_handling=False,
            cache_breakpoint_strategy="none",
        )

    def prepare_request(self, ctx: ConversationContext) -> dict:
        """Build Bedrock Converse API kwargs.

        Bedrock passes region as a special arg __bedrock_region__
        (matching current AIAgent convention).
        """
        kwargs = {
            "messages": ctx.messages,
            "__bedrock_region__": self._region,
            "__bedrock_converse__": True,
        }
        if ctx.tools:
            kwargs["toolConfig"] = {"tools": ctx.tools}
        return kwargs

    def execute(self, request: dict) -> ProviderResult:
        if self._call_fn is not None:
            try:
                response = self._call_fn(**request)
                return ProviderResult(response=response)
            except Exception as exc:
                return ProviderResult(error=exc)
        return ProviderResult()

    def parse_response(
        self, result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        return self._parse_fn(result, ctx)

    @staticmethod
    def _default_parse_response(
        result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        return ParsedResponse()


# ============================================================================
# CodexProvider
# ============================================================================

class CodexProvider(ProviderProtocol):
    """Wraps the codex_responses / _run_codex_stream path.

    The Codex/Responses API uses a different request/response
    format with its own streaming protocol.
    """

    def __init__(
        self,
        call_fn: Optional[CallFn] = None,
        parse_fn: Optional[ParseFn] = None,
    ) -> None:
        self._call_fn = call_fn
        self._parse_fn = parse_fn or self._default_parse_response

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_responses_api=True,
            requires_prompt_caching=False,
            requires_message_sanitization=False,
            requires_custom_stop_handling=False,
            cache_breakpoint_strategy="none",
        )

    def prepare_request(self, ctx: ConversationContext) -> dict:
        """Build Codex Responses API kwargs."""
        kwargs = {
            "input": ctx.messages,
        }
        if ctx.tools:
            kwargs["tools"] = ctx.tools
        return kwargs

    def execute(self, request: dict) -> ProviderResult:
        if self._call_fn is not None:
            try:
                response = self._call_fn(**request)
                return ProviderResult(response=response)
            except Exception as exc:
                return ProviderResult(error=exc)
        return ProviderResult()

    def parse_response(
        self, result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        return self._parse_fn(result, ctx)

    @staticmethod
    def _default_parse_response(
        result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        return ParsedResponse()


# ============================================================================
# GeminiProvider
# ============================================================================

class GeminiProvider(ProviderProtocol):
    """Wraps the gemini_native API path.

    Google Gemini uses its own contents/generateContent format,
    not OpenAI-compatible. This adapter wraps the gemini_native_adapter
    module.
    """

    def __init__(
        self,
        call_fn: Optional[CallFn] = None,
        parse_fn: Optional[ParseFn] = None,
    ) -> None:
        self._call_fn = call_fn
        self._parse_fn = parse_fn or self._default_parse_response

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            requires_prompt_caching=False,
            supports_responses_api=False,
            requires_message_sanitization=False,
            requires_custom_stop_handling=False,
            cache_breakpoint_strategy="none",
        )

    def prepare_request(self, ctx: ConversationContext) -> dict:
        """Build Gemini generateContent API kwargs."""
        kwargs = {
            "contents": ctx.messages,
        }
        if ctx.tools:
            kwargs["tools"] = ctx.tools
        return kwargs

    def execute(self, request: dict) -> ProviderResult:
        if self._call_fn is not None:
            try:
                response = self._call_fn(**request)
                return ProviderResult(response=response)
            except Exception as exc:
                return ProviderResult(error=exc)
        return ProviderResult()

    def parse_response(
        self, result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        return self._parse_fn(result, ctx)

    @staticmethod
    def _default_parse_response(
        result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        return ParsedResponse()