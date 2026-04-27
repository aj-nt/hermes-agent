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

import logging
from typing import Any, Callable, Optional

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
        """Build chat completions API kwargs from context."""
        kwargs = {
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
# OllamaProvider
# ============================================================================

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