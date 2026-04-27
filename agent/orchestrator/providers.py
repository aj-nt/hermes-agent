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

    Wraps the existing CredentialPool (agent/credential_pool.py) for
    multi-credential failover, and provides error classification that
    feeds into FallbackChain decisions.

    Phase 2: Enhanced with pool delegation, restore_primary, and
    richer error classification (503, 529, context overflow).
    Full error_classifier integration comes in Phase 5.
    """

    def __init__(self, credential_pool: Any = None) -> None:
        self._credentials: dict[str, Any] = {}
        self._primary_api_key: Optional[str] = None
        self.active_pool: Any = credential_pool

    @property
    def primary_api_key(self) -> Optional[str]:
        return self._primary_api_key

    @primary_api_key.setter
    def primary_api_key(self, value: Optional[str]) -> None:
        self._primary_api_key = value

    def get_credential(self, provider: str) -> Optional[str]:
        """Get best available credential for the given provider.

        If a CredentialPool is active, delegates to pool.current().
        Otherwise returns the primary API key or a stored credential.
        """
        if self.active_pool is not None:
            current = self.active_pool.current()
            if current is not None:
                return getattr(current, "access_token", None)
        if self._primary_api_key:
            return self._primary_api_key
        return self._credentials.get(provider)

    def report_failure(self, provider: str, error: Exception) -> FailoverReason:
        """Classify error and decide: retry, rotate, or fallback.

        Returns a FailoverReason that the FallbackChain uses to select
        the next provider. This is a simplified classifier — Phase 5
        will wire to the full agent/error_classifier.py classify_api_error().
        """
        error_msg = str(error).lower()

        # Rate limiting
        if "429" in error_msg or "rate" in error_msg:
            return FailoverReason.RATE_LIMITED

        # Auth failures
        if "401" in error_msg or "403" in error_msg or "auth" in error_msg:
            return FailoverReason.AUTH_FAILED

        # Server overloaded
        if "503" in error_msg or "529" in error_msg or "overload" in error_msg:
            return FailoverReason.MODEL_OVERLOADED

        # Context too long
        if "context_length" in error_msg or "context_overflow" in error_msg:
            return FailoverReason.CONTEXT_TOO_LONG

        # Connection / transport errors
        if "connect" in error_msg or "timeout" in error_msg:
            return FailoverReason.CONNECTION_ERROR

        return FailoverReason.UNKNOWN

    def restore_primary(self) -> bool:
        """Attempt to restore primary provider after fallback.

        Without a pool, there's nothing to restore — always succeeds.
        With a pool, checks if a primary credential is available.
        """
        if self.active_pool is None:
            return True
        # With a pool, check if the current entry is available
        current = self.active_pool.current()
        return current is not None