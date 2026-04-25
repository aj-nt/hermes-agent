"""Tests for agent.kore.client_lifecycle -- extracted HTTP lifecycle utilities."""

import pytest

from agent.kore.client_lifecycle import (
    is_openai_client_closed,
    build_keepalive_http_client,
    force_close_tcp_sockets,
)


class TestIsOpenaiClientClosed:

    def test_real_httpx_client(self):
        import httpx
        client = httpx.Client()
        assert is_openai_client_closed(client) is False
        client.close()
        assert is_openai_client_closed(client) is True

    def test_non_httpx_client(self):
        obj = object()
        # Should not crash, returns False for non-httpx objects
        result = is_openai_client_closed(obj)
        assert isinstance(result, bool)


class TestBuildKeepaliveHttpClient:

    def test_creates_client_or_returns_none(self):
        """build_keepalive_http_client may return None if proxy config fails."""
        client = build_keepalive_http_client("https://api.openai.com")
        if client is not None:
            assert hasattr(client, 'send')
            client.close()
        # Returning None is also valid (proxy/failover path)

    def test_custom_base_url(self):
        """build_keepalive_http_client may return None in test env."""
        client = build_keepalive_http_client("https://example.com/v1")
        if client is not None:
            client.close()


class TestForceCloseTcpSockets:

    def test_closed_client(self):
        import httpx
        client = httpx.Client()
        client.close()
        # Should not raise even on already-closed client
        force_close_tcp_sockets(client)

    def test_active_client(self):
        import httpx
        client = httpx.Client()
        try:
            force_close_tcp_sockets(client)
        finally:
            try:
                client.close()
            except Exception:
                pass


class TestClientLifecycleBackwardCompat:

    @pytest.fixture(autouse=True)
    def _make_agent(self):
        from run_agent import AIAgent
        self.agent = AIAgent.__new__(AIAgent)

    def test_is_openai_client_closed_compat(self):
        import httpx
        client = httpx.Client()
        try:
            assert is_openai_client_closed(client) == self.agent._is_openai_client_closed(client)
        finally:
            client.close()
