"""Provider protocol, registry, and fallback chain for the Kore Orchestrator.

These types replace the maze of _is_*_backend() checks in run_agent.py
with capability-based dispatch. Each provider implements ProviderProtocol
and declares what it supports via ProviderCapabilities.

Phase 1: Type definitions and registry. Provider adapters (Phase 2)
will wrap existing provider code in ProviderProtocol implementations.
"""

from __future__ import annotations

import enum
from typing import Any, Optional

from agent.orchestrator.context import (
    ConversationContext,
    ParsedResponse,
    ProviderCapabilities,
    ProviderResult,
)


# ============================================================================
# ProviderProtocol
# ============================================================================

class ProviderProtocol:
    """Any LLM provider the orchestrator can call.

    Concrete implementations wrap existing adapter code:
    - OllamaProvider → wraps GLM/Ollama path
    - OpenAICompatibleProvider → wraps OpenRouter, direct OpenAI, Azure
    - AnthropicProvider → wraps _call_anthropic path
    - BedrockProvider → wraps _bedrock_call path
    - CodexProvider → wraps _run_codex_stream path

    Phase 1: Defines the protocol shape as a base class.
    Phase 2: Will migrate to typing.Protocol once we drop Python 3.9.
    """

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Declare what this provider supports."""
        return ProviderCapabilities()

    def prepare_request(self, ctx: ConversationContext) -> dict:
        """Build provider-specific API kwargs from context."""
        raise NotImplementedError

    def execute(self, request: dict) -> ProviderResult:
        """Make the API call. Return streaming or non-streaming result."""
        raise NotImplementedError

    def parse_response(
        self, result: ProviderResult, ctx: ConversationContext
    ) -> ParsedResponse:
        """Convert provider response into canonical format."""
        raise NotImplementedError


# ============================================================================
# FailoverReason
# ============================================================================

class FailoverReason(enum.Enum):
    """Why a provider was abandoned and fallback activated."""

    RATE_LIMITED = "rate_limited"
    AUTH_FAILED = "auth_failed"
    CONNECTION_ERROR = "connection_error"
    MODEL_OVERLOADED = "model_overloaded"
    CONTEXT_TOO_LONG = "context_too_long"
    UNKNOWN = "unknown"


# ============================================================================
# ProviderRegistry
# ============================================================================

class ProviderRegistry:
    """Maps provider names to ProviderProtocol implementations.

    The orchestrator looks up providers by name instead of branching
    on provider identity. This replaces the scattered _is_*_backend()
    conditional checks.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderProtocol] = {}

    def register(self, name: str, provider: ProviderProtocol) -> None:
        """Register a provider under the given name."""
        self._providers[name] = provider

    def get(self, name: str) -> ProviderProtocol:
        """Look up a provider by name. Raises KeyError if not found."""
        if name not in self._providers:
            raise KeyError(f"No provider registered as '{name}'. "
                           f"Available: {list(self._providers.keys())}")
        return self._providers[name]

    def list_names(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._providers.keys())


# ============================================================================
# FallbackChain
# ============================================================================

class FallbackChain:
    """Ordered list of fallback providers with retry budgets.

    On failure, the chain selects the next provider based on the
    failure reason. Rate-limited → try next; auth failure → try next
    with different credentials; connection error → try next; etc.
    """

    def __init__(self, providers: list[ProviderProtocol]) -> None:
        self._providers: list[ProviderProtocol] = list(providers)

    def next(
        self,
        current: ProviderProtocol,
        reason: FailoverReason,
    ) -> Optional[ProviderProtocol]:
        """Select next provider in chain based on failure reason.

        Returns None if the chain is exhausted (current is last).
        For now, simply returns the next provider regardless of reason.
        Reason-based filtering (e.g., skip providers that don't support
        streaming on stream errors) will be added in Phase 2 when we
        have concrete provider implementations with real capabilities.
        """
        try:
            idx = self._providers.index(current)
        except ValueError:
            # Current not in chain — return first if available
            return self._providers[0] if self._providers else None

        next_idx = idx + 1
        if next_idx < len(self._providers):
            return self._providers[next_idx]
        return None


# ============================================================================
# CredentialManager
# ============================================================================

class CredentialManager:
    """Owns credential lifecycle, rotation, and fallback decisions.

    Phase 1: Shell with interface. Phase 2: Wire up to existing
    credential_pool.py and credential_sources.py.
    """

    def __init__(self) -> None:
        self._credentials: dict[str, Any] = {}

    def get_credential(self, provider: str) -> Optional[str]:
        """Get best available credential for the given provider."""
        return self._credentials.get(provider)

    def report_failure(self, provider: str, error: Exception) -> FailoverReason:
        """Classify error and decide: retry, rotate, or fallback."""
        # Phase 1: Simple classification. Phase 2: Full error_classifier integration.
        error_msg = str(error).lower()
        if "429" in error_msg or "rate" in error_msg:
            return FailoverReason.RATE_LIMITED
        if "401" in error_msg or "403" in error_msg or "auth" in error_msg:
            return FailoverReason.AUTH_FAILED
        if "connect" in error_msg or "timeout" in error_msg:
            return FailoverReason.CONNECTION_ERROR
        return FailoverReason.UNKNOWN