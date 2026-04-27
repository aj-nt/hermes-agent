"""ClientManager — HTTP client lifecycle management for the Kore Orchestrator.

Owns creation, pooling, cleanup, dead-connection detection, and teardown
for all HTTP clients used by provider adapters. Delegates to lifecycle
functions injected at construction time — this keeps ClientManager testable
without loading the full agent runtime (which requires Python 3.10+).

Phase 2: Thin wrapper with dependency injection.
Production wiring: ClientManager.from_defaults() injects real functions.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Type aliases for injected lifecycle functions
CloseFn = Callable[[Any], None]
CheckDeadFn = Callable[[Any], bool]
ForceCloseSocketsFn = Callable[[Any], int]
BuildKeepaliveFn = Callable[[str], Optional[Any]]
IsClosedFn = Callable[[Any], bool]


class ClientManager:
    """Owns HTTP client lifecycle, connection pooling, and cleanup.

    Each provider needs an HTTP client. Rather than scattering client
    creation across the agent (currently 200+ lines of _create_openai_client,
    _close_openai_client, etc.), ClientManager centralizes it.

    Lifecycle functions are injected so the class is testable without
    loading the full runtime. Use ClientManager.from_defaults() for
    production wiring.

    Thread safety: create/teardown are guarded by _lock. Request-time
    client access is single-writer per pipeline (no lock needed).
    """

    def __init__(
        self,
        *,
        close_fn: CloseFn = None,
        is_closed_fn: IsClosedFn = None,
        check_dead_fn: CheckDeadFn = None,
        force_close_fn: ForceCloseSocketsFn = None,
        build_keepalive_fn: BuildKeepaliveFn = None,
    ) -> None:
        self._lock = threading.Lock()
        self._primary_client: Optional[Any] = None
        self._anthropic_client: Optional[Any] = None
        self._codex_client: Optional[Any] = None

        # Injected lifecycle functions (default to no-ops for testing)
        self._close_fn: CloseFn = close_fn or self._default_close
        self._is_closed_fn: IsClosedFn = is_closed_fn or self._default_is_closed
        self._check_dead_fn: CheckDeadFn = check_dead_fn or (lambda c: False)
        self._force_close_fn: ForceCloseSocketsFn = force_close_fn or (lambda c: 0)
        self._build_keepalive_fn: BuildKeepaliveFn = build_keepalive_fn or (lambda base_url="": None)

    @classmethod
    def from_defaults(cls) -> "ClientManager":
        """Create a ClientManager wired to the real client_lifecycle functions.

        This method performs runtime imports so it's only called in
        production, not in tests (avoids Python 3.9 compatibility
        issues with the broader agent package).
        """
        from agent.kore.client_lifecycle import (
            build_keepalive_http_client,
            cleanup_dead_connections,
            force_close_tcp_sockets,
            is_openai_client_closed,
        )

        def close_client(client: Any) -> None:
            """Close a client, handling both property and method is_closed."""
            try:
                if is_openai_client_closed(client):
                    return  # already closed
                client.close()
            except Exception:
                logger.debug("Error closing client", exc_info=True)

        return cls(
            close_fn=close_client,
            is_closed_fn=is_openai_client_closed,
            check_dead_fn=cleanup_dead_connections,
            force_close_fn=force_close_tcp_sockets,
            build_keepalive_fn=build_keepalive_http_client,
        )

    # --- Default implementations (for testing without injection) ---

    @staticmethod
    def _default_close(client: Any) -> None:
        """Default close: try client.close(), swallow errors."""
        try:
            if hasattr(client, 'close'):
                client.close()
        except Exception:
            pass

    @staticmethod
    def _default_is_closed(client: Any) -> bool:
        """Default is_closed check: check is_closed attribute."""
        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            if callable(is_closed_attr):
                return bool(is_closed_attr())
            return bool(is_closed_attr)
        return False

    # --- Primary client (OpenAI-compatible) ---

    @property
    def primary_client(self) -> Optional[Any]:
        return self._primary_client

    @primary_client.setter
    def primary_client(self, client: Any) -> None:
        with self._lock:
            old = self._primary_client
            self._primary_client = client
        if old is not None and old is not client:
            self._safe_close(old)

    def release_primary(self) -> None:
        """Close and release the primary client."""
        with self._lock:
            client = self._primary_client
            self._primary_client = None
        if client is not None:
            self._safe_close(client)

    # --- Anthropic client ---

    @property
    def anthropic_client(self) -> Optional[Any]:
        return self._anthropic_client

    @anthropic_client.setter
    def anthropic_client(self, client: Any) -> None:
        with self._lock:
            old = self._anthropic_client
            self._anthropic_client = client
        if old is not None and old is not client:
            self._safe_close(old)

    def release_anthropic(self) -> None:
        """Close and release the Anthropic client."""
        with self._lock:
            client = self._anthropic_client
            self._anthropic_client = None
        if client is not None:
            self._safe_close(client)

    # --- Codex/Responses API client ---

    @property
    def codex_client(self) -> Optional[Any]:
        return self._codex_client

    @codex_client.setter
    def codex_client(self, client: Any) -> None:
        with self._lock:
            self._codex_client = client

    def release_codex(self) -> None:
        """Release the Codex client."""
        with self._lock:
            client = self._codex_client
            self._codex_client = None
        if client is not None:
            self._safe_close(client)

    # --- Cleanup all ---

    def cleanup(self) -> None:
        """Close all managed clients and reset state.

        Safe to call multiple times. Handles close() errors gracefully.
        """
        with self._lock:
            clients = [
                self._primary_client,
                self._anthropic_client,
                self._codex_client,
            ]
            self._primary_client = None
            self._anthropic_client = None
            self._codex_client = None
        # Close outside lock
        for client in clients:
            if client is not None:
                self._safe_close(client)

    # --- Dead connection detection ---

    def check_dead_connections(self) -> bool:
        """Check if the primary client has dead TCP connections.

        Delegates to injected check_dead_fn.
        Returns True if dead connections were found.
        """
        if self._primary_client is None:
            return False
        return self._check_dead_fn(self._primary_client)

    # --- Force-close TCP sockets ---

    def force_close_sockets(self) -> int:
        """Force-close stale TCP sockets on the primary client.

        Delegates to injected force_close_fn.
        Returns the number of sockets closed.
        """
        if self._primary_client is None:
            return 0
        return self._force_close_fn(self._primary_client)

    # --- Build keepalive client ---

    def build_keepalive_client(self, base_url: str = "") -> Optional[Any]:
        """Build an httpx.Client with TCP keepalive and proxy support.

        Delegates to injected build_keepalive_fn.
        Returns the client, or None if creation failed.
        """
        return self._build_keepalive_fn(base_url=base_url)

    # --- Internal helpers ---

    def _safe_close(self, client: Any) -> None:
        """Close a client, swallowing errors."""
        try:
            self._close_fn(client)
        except Exception:
            logger.debug("Error closing client", exc_info=True)