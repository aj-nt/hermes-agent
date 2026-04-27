"""Tests for ClientManager — HTTP client lifecycle management.

ClientManager owns:
- Client creation (primary OpenAI, Anthropic, per-request)
- Client pooling and reuse
- Dead connection detection and cleanup
- Force-close of stale TCP sockets
- Proper teardown on session end

Uses dependency injection so tests don't need the full agent runtime.
"""

import threading
from unittest.mock import MagicMock, patch
import pytest

from agent.orchestrator.clients import ClientManager


# ============================================================================
# Construction and defaults
# ============================================================================

class TestClientManagerInit:
    """ClientManager can be instantiated with no active clients."""

    def test_import(self):
        from agent.orchestrator.clients import ClientManager
        assert ClientManager is not None

    def test_default_state_has_no_clients(self):
        mgr = ClientManager()
        assert mgr.primary_client is None
        assert mgr.anthropic_client is None
        assert mgr.codex_client is None

    def test_has_lock_for_thread_safety(self):
        mgr = ClientManager()
        assert hasattr(mgr, '_lock')
        assert isinstance(mgr._lock, type(threading.Lock()))

    def test_injected_functions_can_be_overridden(self):
        """Constructor accepts lifecycle functions for DI."""
        close_fn = MagicMock()
        mgr = ClientManager(close_fn=close_fn)
        assert mgr._close_fn is close_fn

    def test_default_close_fn_works(self):
        """Default close_fn calls client.close() and swallows errors."""
        mgr = ClientManager()
        mock_client = MagicMock()
        mgr._default_close(mock_client)
        mock_client.close.assert_called()

    def test_default_close_fn_swallows_errors(self):
        """Default close_fn should not raise even if close() fails."""
        mgr = ClientManager()
        bad_client = MagicMock()
        bad_client.close.side_effect = RuntimeError("already closed")
        mgr._default_close(bad_client)  # should not raise

    def test_default_is_closed_fn(self):
        """Default is_closed checks property and method forms."""
        mgr = ClientManager()
        # Property form (httpx.Client.is_closed)
        prop_client = MagicMock()
        prop_client.is_closed = True
        assert mgr._default_is_closed(prop_client) is True

        # Method form (openai.OpenAI.is_closed)
        method_client = MagicMock()
        method_client.is_closed = MagicMock(return_value=True)
        assert mgr._default_is_closed(method_client) is True

        # Not closed
        open_client = MagicMock()
        open_client.is_closed = False
        assert mgr._default_is_closed(open_client) is False


# ============================================================================
# Primary client management
# ============================================================================

class TestClientManagerPrimaryClient:
    """Manager creates, stores, and releases primary OpenAI-compatible clients."""

    def test_set_and_get_primary_client(self):
        mgr = ClientManager()
        mock_client = MagicMock()
        mgr.primary_client = mock_client
        assert mgr.primary_client is mock_client

    def test_set_primary_client_replaces_previous(self):
        mgr = ClientManager()
        mock1 = MagicMock()
        mock2 = MagicMock()
        mgr.primary_client = mock1
        mgr.primary_client = mock2
        assert mgr.primary_client is mock2

    def test_releasing_old_client_on_replace(self):
        """Setting a new primary client should close the old one."""
        close_fn = MagicMock()
        mgr = ClientManager(close_fn=close_fn)
        old_client = MagicMock()
        new_client = MagicMock()
        mgr.primary_client = old_client
        mgr.primary_client = new_client
        close_fn.assert_called_with(old_client)

    def test_release_primary_client_sets_none(self):
        mgr = ClientManager()
        mock_client = MagicMock()
        mgr.primary_client = mock_client
        mgr.release_primary()
        assert mgr.primary_client is None

    def test_release_primary_calls_close_fn(self):
        """Releasing a client should close it via injected close_fn."""
        close_fn = MagicMock()
        mgr = ClientManager(close_fn=close_fn)
        mock_client = MagicMock()
        mgr.primary_client = mock_client
        mgr.release_primary()
        close_fn.assert_called_with(mock_client)

    def test_no_double_close_on_same_reference(self):
        """Setting the same client again should NOT close it."""
        close_fn = MagicMock()
        mgr = ClientManager(close_fn=close_fn)
        mock_client = MagicMock()
        mgr.primary_client = mock_client
        mgr.primary_client = mock_client  # set same again
        close_fn.assert_not_called()


# ============================================================================
# Anthropic client management
# ============================================================================

class TestClientManagerAnthropicClient:
    """Manager creates, stores, and releases Anthropic-specific clients."""

    def test_set_and_get_anthropic_client(self):
        mgr = ClientManager()
        mock_client = MagicMock()
        mgr.anthropic_client = mock_client
        assert mgr.anthropic_client is mock_client

    def test_release_anthropic_client_sets_none(self):
        mgr = ClientManager()
        mock_client = MagicMock()
        mgr.anthropic_client = mock_client
        mgr.release_anthropic()
        assert mgr.anthropic_client is None

    def test_release_anthropic_calls_close_fn(self):
        close_fn = MagicMock()
        mgr = ClientManager(close_fn=close_fn)
        mock_client = MagicMock()
        mgr.anthropic_client = mock_client
        mgr.release_anthropic()
        close_fn.assert_called_with(mock_client)

    def test_replacing_anthropic_closes_old(self):
        close_fn = MagicMock()
        mgr = ClientManager(close_fn=close_fn)
        old = MagicMock()
        new = MagicMock()
        mgr.anthropic_client = old
        mgr.anthropic_client = new
        close_fn.assert_called_with(old)


# ============================================================================
# Codex client management
# ============================================================================

class TestClientManagerCodexClient:
    """Manager stores a codex/Responses API client for per-request use."""

    def test_set_and_get_codex_client(self):
        mgr = ClientManager()
        mock_client = MagicMock()
        mgr.codex_client = mock_client
        assert mgr.codex_client is mock_client

    def test_release_codex_client_sets_none(self):
        mgr = ClientManager()
        mock_client = MagicMock()
        mgr.codex_client = mock_client
        mgr.release_codex()
        assert mgr.codex_client is None

    def test_release_codex_calls_close_fn(self):
        close_fn = MagicMock()
        mgr = ClientManager(close_fn=close_fn)
        mock_client = MagicMock()
        mgr.codex_client = mock_client
        mgr.release_codex()
        close_fn.assert_called_with(mock_client)


# ============================================================================
# Cleanup — close all clients
# ============================================================================

class TestClientManagerCleanup:
    """cleanup() closes all managed clients and resets state."""

    def test_cleanup_closes_all_clients(self):
        close_fn = MagicMock()
        mgr = ClientManager(close_fn=close_fn)
        primary = MagicMock()
        anthropic = MagicMock()
        codex = MagicMock()
        mgr._primary_client = primary
        mgr._anthropic_client = anthropic
        mgr._codex_client = codex
        mgr.cleanup()
        assert mgr.primary_client is None
        assert mgr.anthropic_client is None
        assert mgr.codex_client is None
        assert close_fn.call_count == 3

    def test_cleanup_with_no_clients(self):
        """Cleanup should be safe with no active clients."""
        mgr = ClientManager()
        mgr.cleanup()  # should not raise

    def test_cleanup_handles_close_error(self):
        """If close_fn raises, cleanup should continue with remaining clients."""
        call_log = []

        def flaky_close(client):
            call_log.append(client)
            if client == "bad":
                raise RuntimeError("connection already closed")

        mgr = ClientManager(close_fn=flaky_close)
        mgr._primary_client = "bad"
        mgr._anthropic_client = "good1"
        mgr._codex_client = "good2"
        mgr.cleanup()
        # All three should be attempted
        assert call_log == ["bad", "good1", "good2"]
        assert mgr.primary_client is None


# ============================================================================
# Dead connection detection
# ============================================================================

class TestClientManagerDeadConnections:
    """ClientManager delegates to injected check_dead_fn."""

    def test_check_returns_true_when_dead(self):
        check_fn = MagicMock(return_value=True)
        mgr = ClientManager(check_dead_fn=check_fn)
        mock_client = MagicMock()
        mgr._primary_client = mock_client
        assert mgr.check_dead_connections() is True
        check_fn.assert_called_once_with(mock_client)

    def test_check_returns_false_when_clean(self):
        check_fn = MagicMock(return_value=False)
        mgr = ClientManager(check_dead_fn=check_fn)
        mock_client = MagicMock()
        mgr._primary_client = mock_client
        assert mgr.check_dead_connections() is False
        check_fn.assert_called_once_with(mock_client)

    def test_check_returns_false_when_no_primary(self):
        check_fn = MagicMock()
        mgr = ClientManager(check_dead_fn=check_fn)
        assert mgr.check_dead_connections() is False
        check_fn.assert_not_called()


# ============================================================================
# Force-close TCP sockets
# ============================================================================

class TestClientManagerForceCloseSockets:
    """ClientManager delegates force-close to injected force_close_fn."""

    def test_force_close_primary_sockets(self):
        force_fn = MagicMock(return_value=3)
        mgr = ClientManager(force_close_fn=force_fn)
        mock_client = MagicMock()
        mgr._primary_client = mock_client
        result = mgr.force_close_sockets()
        assert result == 3
        force_fn.assert_called_once_with(mock_client)

    def test_force_close_returns_zero_when_no_primary(self):
        force_fn = MagicMock()
        mgr = ClientManager(force_close_fn=force_fn)
        result = mgr.force_close_sockets()
        assert result == 0
        force_fn.assert_not_called()


# ============================================================================
# Build keepalive HTTP client
# ============================================================================

class TestClientManagerBuildKeepaliveClient:
    """ClientManager delegates keepalive client building to injected fn."""

    def test_build_keepalive_client(self):
        mock_httpx = MagicMock()
        build_fn = MagicMock(return_value=mock_httpx)
        mgr = ClientManager(build_keepalive_fn=build_fn)
        result = mgr.build_keepalive_client(base_url="http://localhost:11434")
        assert result is mock_httpx
        build_fn.assert_called_once_with(base_url="http://localhost:11434")

    def test_build_keepalive_client_returns_none_on_failure(self):
        build_fn = MagicMock(return_value=None)
        mgr = ClientManager(build_keepalive_fn=build_fn)
        result = mgr.build_keepalive_client(base_url="http://bad-url")
        assert result is None


# ============================================================================
# Thread safety
# ============================================================================

class TestClientManagerThreadSafety:
    """ClientManager operations are safe under concurrent access."""

    def test_concurrent_set_primary(self):
        """Multiple threads setting primary_client should not corrupt state."""
        mgr = ClientManager()
        clients = [MagicMock() for _ in range(10)]
        errors = []

        def set_client(i):
            try:
                mgr.primary_client = clients[i]
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=set_client, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # One of the clients should be set
        assert mgr.primary_client in clients

    def test_concurrent_cleanup(self):
        """Multiple threads calling cleanup should not raise."""
        mgr = ClientManager()
        mgr._primary_client = MagicMock()
        errors = []

        def cleanup():
            try:
                mgr.cleanup()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cleanup) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mgr.primary_client is None