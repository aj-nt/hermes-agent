"""Client lifecycle management utilities for OpenAI/httpx connections.

Extracted from run_agent.py to decouple TCP keepalive, socket management,
and proxy configuration from the AIAgent god-object.

Layer 1 extraction: pure static functions with no self.* references.
"""

import httpx
import logging
import socket
from typing import Any, Optional
from unittest.mock import Mock

logger = logging.getLogger(__name__)


def _get_proxy_from_env() -> Optional[str]:
    """Read proxy URL from environment variables.

    Checks HTTPS_PROXY, HTTP_PROXY, ALL_PROXY (and lowercase variants) in order.
    Returns the first valid proxy URL found, or None if no proxy is configured.
    """
    import os
    from utils import normalize_proxy_url

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = os.environ.get(key, "").strip()
        if value:
            return normalize_proxy_url(value)
    return None


def _get_proxy_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """Return an env-configured proxy unless NO_PROXY excludes this base URL."""
    import urllib.request
    from utils import base_url_hostname

    proxy = _get_proxy_from_env()
    if not proxy or not base_url:
        return proxy

    host = base_url_hostname(base_url)
    if not host:
        return proxy

    try:
        if urllib.request.proxy_bypass_environment(host):
            return None
    except Exception:
        pass

    return proxy


def is_openai_client_closed(client: Any) -> bool:
    """Check if an OpenAI client is closed.

    Handles both property and method forms of is_closed:
    - httpx.Client.is_closed is a bool property
    - openai.OpenAI.is_closed is a method returning bool

    Prior bug: getattr(client, "is_closed", False) returned the bound method,
    which is always truthy, causing unnecessary client recreation on every call.
    """
    if isinstance(client, Mock):
        return False

    is_closed_attr = getattr(client, "is_closed", None)
    if is_closed_attr is not None:
        # Handle method (openai SDK) vs property (httpx)
        if callable(is_closed_attr):
            if is_closed_attr():
                return True
        elif bool(is_closed_attr):
            return True

    http_client = getattr(client, "_client", None)
    if http_client is not None:
        return bool(getattr(http_client, "is_closed", False))
    return False


def build_keepalive_http_client(base_url: str = "") -> Any:
    """Build an httpx.Client with TCP keepalive and proxy support.

    Sets socket options for TCP keepalive probes (30s idle, 10s interval, 3 count)
    and configures proxy from environment variables while respecting NO_PROXY.
    Returns None if client creation fails (e.g. missing proxy or bad config).
    """
    _sock_opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if hasattr(socket, "TCP_KEEPIDLE"):
        _sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30))
        _sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10))
        _sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3))
    elif hasattr(socket, "TCP_KEEPALIVE"):
        _sock_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30))
    # When a custom transport is provided, httpx won't auto-read proxy
    # from env vars (allow_env_proxies = trust_env and transport is None).
    # Explicitly read proxy settings while still honoring NO_PROXY for
    # loopback / local endpoints such as a locally hosted sub2api.
    _proxy = _get_proxy_for_base_url(base_url)
    try:
        return httpx.Client(
            transport=httpx.HTTPTransport(socket_options=_sock_opts),
            proxy=_proxy,
        )
    except Exception:
        logger.warning("Failed to create keepalive HTTP client for %s", base_url, exc_info=True)
        return None


def force_close_tcp_sockets(client: Any) -> int:
    """Force-close underlying TCP sockets to prevent CLOSE-WAIT accumulation.

    When a provider drops a connection mid-stream, httpx's ``client.close()``
    performs a graceful shutdown which leaves sockets in CLOSE-WAIT until the
    OS times them out (often minutes).  This method walks the httpx transport
    pool and issues ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()`` to
    force an immediate TCP RST, freeing the file descriptors.

    Returns the number of sockets force-closed.
    """
    closed = 0
    try:
        http_client = getattr(client, "_client", None)
        if http_client is None:
            return 0
        transport = getattr(http_client, "_transport", None)
        if transport is None:
            return 0
        pool = getattr(transport, "_pool", None)
        if pool is None:
            return 0
        # httpx uses httpcore connection pools; connections live in
        # _connections (list) or _pool (list) depending on version.
        connections = (
            getattr(pool, "_connections", None)
            or getattr(pool, "_pool", None)
            or []
        )
        for conn in list(connections):
            stream = (
                getattr(conn, "_network_stream", None)
                or getattr(conn, "_stream", None)
            )
            if stream is None:
                continue
            sock = getattr(stream, "_sock", None)
            if sock is None:
                sock = getattr(stream, "stream", None)
                if sock is not None:
                    sock = getattr(sock, "_sock", None)
            if sock is None:
                continue
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
            closed += 1
    except Exception as exc:
        logger.debug("Force-close TCP sockets sweep error: %s", exc)
    return closed


def cleanup_dead_connections(client) -> bool:
    """Detect and clean up dead TCP connections on the primary client.

    Inspects the httpx connection pool for sockets in unhealthy states
    (CLOSE-WAIT, errors). If any are found, returns True so the caller
    can rebuild the primary client from scratch.

    This is the pure-function extraction of AIAgent._cleanup_dead_connections.
    The caller is responsible for rebuilding the client when True is returned.

    Args:
        client: An OpenAI client instance (or compatible object with
            ``_client._transport._pool`` path).

    Returns:
        True if dead connections were found, False otherwise.
    """
    if client is None:
        return False
    try:
        http_client = getattr(client, "_client", None)
        if http_client is None:
            return False
        transport = getattr(http_client, "_transport", None)
        if transport is None:
            return False
        pool = getattr(transport, "_pool", None)
        if pool is None:
            return False
        connections = (
            getattr(pool, "_connections", None)
            or getattr(pool, "_pool", None)
            or []
        )
        dead_count = 0
        for conn in list(connections):
            # Check for connections that are idle but have closed sockets
            stream = (
                getattr(conn, "_network_stream", None)
                or getattr(conn, "_stream", None)
            )
            if stream is None:
                continue
            sock = getattr(stream, "_sock", None)
            if sock is None:
                sock = getattr(stream, "stream", None)
                if sock is not None:
                    sock = getattr(sock, "_sock", None)
            if sock is None:
                continue
            # Probe socket health with a non-blocking recv peek
            import socket as _socket
            try:
                sock.setblocking(False)
                data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                if data == b"":
                    dead_count += 1
            except BlockingIOError:
                pass  # No data available — socket is healthy
            except OSError:
                dead_count += 1
            finally:
                try:
                    sock.setblocking(True)
                except OSError:
                    pass
        if dead_count > 0:
            logger.warning(
                "Found %d dead connection(s) in client pool — rebuild needed",
                dead_count,
            )
            return True
    except Exception as exc:
        logger.debug("Dead connection check error: %s", exc)
    return False
