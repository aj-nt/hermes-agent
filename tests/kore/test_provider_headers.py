"""Tests for provider-specific HTTP header factories."""

import platform

from agent.kore.provider_headers import (
    routermint_headers,
    qwen_portal_headers,
    _QWEN_CODE_VERSION,
)


class TestRoutermintHeaders:
    def test_contains_user_agent_key(self):
        h = routermint_headers()
        assert "User-Agent" in h

    def test_user_agent_starts_with_hermes(self):
        h = routermint_headers()
        assert h["User-Agent"].startswith("HermesAgent/")

    def test_user_agent_contains_version(self):
        h = routermint_headers()
        # Version is dynamic (from hermes_cli.__version__), just check format
        parts = h["User-Agent"].split("/")
        assert len(parts) == 2
        assert len(parts[1]) > 0  # non-empty version string


class TestQwenPortalHeaders:
    def test_contains_required_keys(self):
        h = qwen_portal_headers()
        assert "User-Agent" in h
        assert "X-DashScope-CacheControl" in h
        assert "X-DashScope-UserAgent" in h
        assert "X-DashScope-AuthType" in h

    def test_cache_control_enabled(self):
        h = qwen_portal_headers()
        assert h["X-DashScope-CacheControl"] == "enable"

    def test_auth_type_is_oauth(self):
        h = qwen_portal_headers()
        assert h["X-DashScope-AuthType"] == "qwen-oauth"

    def test_user_agent_format(self):
        h = qwen_portal_headers()
        ua = h["User-Agent"]
        assert ua.startswith(f"QwenCode/{_QWEN_CODE_VERSION}")
        # Should contain platform info: (os; arch)
        assert "(" in ua
        assert ")" in ua

    def test_user_agent_matches_ua_header(self):
        h = qwen_portal_headers()
        assert h["User-Agent"] == h["X-DashScope-UserAgent"]

    def test_version_is_expected(self):
        assert _QWEN_CODE_VERSION == "0.14.1"