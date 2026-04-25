"""Tests for agent.kore.error_utils — extracted error handling functions.

Backward-compat tests verify that module functions produce the same output
as the corresponding AIAgent methods for identical inputs.
"""

import re
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.kore.error_utils import (
    summarize_api_error,
    mask_api_key_for_logs,
    clean_error_message,
    extract_api_error_context,
    clean_session_content,
    dump_api_request_debug,
    usage_summary_for_api_request_hook,
)


# ---------------------------------------------------------------------------
# summarize_api_error
# ---------------------------------------------------------------------------

class TestSummarizeApiError:

    def test_plain_string_error(self):
        err = RuntimeError("something went wrong")
        result = summarize_api_error(err)
        assert "something went wrong" in result

    def test_html_error_extracts_title(self):
        err = Exception("<!DOCTYPE html><html><head><title>502 Bad Gateway</title></head></html>")
        result = summarize_api_error(err)
        assert "502 Bad Gateway" in result
        assert "<html>" not in result

    def test_html_error_with_status_code(self):
        err = Exception("<html><title>Error</title></html>")
        err.status_code = 503
        result = summarize_api_error(err)
        assert "HTTP 503" in result
        assert "Error" in result

    def test_html_error_with_ray_id(self):
        html = '<!DOCTYPE html><html><title>502</title>Cloudflare Ray ID: <strong>abc123</strong></html>'
        err = Exception(html)
        result = summarize_api_error(err)
        assert "abc123" in result

    def test_json_body_error(self):
        err = Exception("Bad request")
        err.body = {"error": {"message": "invalid model"}}
        result = summarize_api_error(err)
        assert "invalid model" in result

    def test_json_body_with_status(self):
        err = Exception("Bad request")
        err.status_code = 400
        err.body = {"error": {"message": "invalid param"}}
        result = summarize_api_error(err)
        assert "HTTP 400" in result
        assert "invalid param" in result

    def test_fallback_truncation(self):
        err = RuntimeError("x" * 1000)
        result = summarize_api_error(err)
        assert len(result) < 600  # truncated to ~500 + prefix


# ---------------------------------------------------------------------------
# mask_api_key_for_logs
# ---------------------------------------------------------------------------

class TestMaskApiKeyForLogs:

    def test_none_key(self):
        assert mask_api_key_for_logs(None) is None

    def test_empty_key(self):
        assert mask_api_key_for_logs("") is None

    def test_short_key(self):
        assert mask_api_key_for_logs("abc") == "***"

    def test_long_key(self):
        result = mask_api_key_for_logs("sk-1234567890abcdef")
        assert result.startswith("sk-12345")
        assert result.endswith("cdef")
        assert "..." in result

    def test_exactly_12_chars(self):
        assert mask_api_key_for_logs("a" * 12) == "***"

    def test_13_chars(self):
        result = mask_api_key_for_logs("a" * 13)
        assert "..." in result
        # Format: first8 + "..." + last4 = longer than input for short keys
        assert result.startswith("a" * 8)


# ---------------------------------------------------------------------------
# clean_error_message
# ---------------------------------------------------------------------------

class TestCleanErrorMessage:

    def test_empty(self):
        assert clean_error_message("") == "Unknown error"

    def test_html_content(self):
        result = clean_error_message("<!DOCTYPE html><html><body>Error</body></html>")
        assert "HTML error page" in result

    def test_whitespace_collapsing(self):
        result = clean_error_message("line1\nline2\nline3")
        assert "\n" not in result
        assert "line1 line2 line3" == result

    def test_truncation(self):
        result = clean_error_message("x" * 200)
        assert len(result) <= 153  # 150 + "..."
        assert result.endswith("...")

    def test_short_message_unchanged(self):
        result = clean_error_message("short error")
        assert result == "short error"


# ---------------------------------------------------------------------------
# extract_api_error_context
# ---------------------------------------------------------------------------

class TestExtractApiErrorContext:

    def test_empty_error(self):
        err = Exception("")
        result = extract_api_error_context(err)
        # Should have a message key even from empty string
        assert isinstance(result, dict)

    def test_body_with_code(self):
        err = Exception("test")
        err.body = {"error": {"code": "rate_limit_exceeded", "message": "Too many requests"}}
        result = extract_api_error_context(err)
        assert result["reason"] == "rate_limit_exceeded"
        assert "Too many requests" in result["message"]

    def test_retry_after_header(self):
        err = Exception("test")
        err.response = SimpleNamespace(headers={"retry-after": "30"})
        result = extract_api_error_context(err)
        assert "reset_at" in result
        assert isinstance(result["reset_at"], float)

    def test_x_ratelimit_reset_header(self):
        err = Exception("test")
        err.response = SimpleNamespace(headers={"x-ratelimit-reset": "1234567890"})
        result = extract_api_error_context(err)
        assert result["reset_at"] == "1234567890"

    def test_body_retry_after(self):
        err = Exception("test")
        err.body = {"retry_after": "60"}
        result = extract_api_error_context(err)
        assert "reset_at" in result


# ---------------------------------------------------------------------------
# clean_session_content
# ---------------------------------------------------------------------------

class TestCleanSessionContent:

    def test_empty(self):
        assert clean_session_content("") == ""

    def test_none_like(self):
        # Should handle falsy
        assert clean_session_content(None) is None

    def test_whitespace_collapse(self):
        result = clean_session_content("hello\n\n\nworld")
        assert result.strip() == result  # strip called

    def test_preserves_regular_text(self):
        result = clean_session_content("Hello world")
        assert "Hello world" in result


# ---------------------------------------------------------------------------
# dump_api_request_debug
# ---------------------------------------------------------------------------

class TestDumpApiRequestDebug:

    def test_basic_dump(self, tmp_path):
        result = dump_api_request_debug(
            {"model": "test", "messages": [], "timeout": 30},
            reason="test_dump",
            session_id="sess123",
            base_url="https://api.openai.com/v1",
            api_mode="chat",
            logs_dir=tmp_path,
        )
        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "test_dump" in content
        assert "sess123" in content
        # timeout should be stripped
        assert "timeout" not in content

    def test_with_error(self, tmp_path):
        err = RuntimeError("test error")
        err.status_code = 500
        result = dump_api_request_debug(
            {"model": "test"},
            reason="error_dump",
            error=err,
            logs_dir=tmp_path,
        )
        assert result is not None
        content = result.read_text()
        assert "RuntimeError" in content
        assert "500" in content

    def test_mask_key_in_dump(self, tmp_path):
        result = dump_api_request_debug(
            {"model": "test"},
            reason="key_test",
            api_key="sk-1234567890abcdef",
            logs_dir=tmp_path,
        )
        content = result.read_text()
        assert "1234567890abcdef" not in content
        assert "..." in content

    def test_exception_returns_none(self, tmp_path):
        # Pass invalid logs_dir to trigger exception
        result = dump_api_request_debug(
            {"model": "test"},
            reason="test",
            logs_dir="/nonexistent/path/that/cannot/exist",
            verbose_logging=True,
        )
        # Should not raise, returns None
        assert result is None


# ---------------------------------------------------------------------------
# usage_summary_for_api_request_hook
# ---------------------------------------------------------------------------

class TestUsageSummaryForApiRequestHook:

    def test_none_response(self):
        assert usage_summary_for_api_request_hook(None) is None

    def test_no_usage_attribute(self):
        assert usage_summary_for_api_request_hook(SimpleNamespace()) is None

    def test_with_usage(self):
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        resp = SimpleNamespace(usage=usage)
        with patch("agent.usage_pricing.normalize_usage") as mock_norm:
            from dataclasses import dataclass
            @dataclass
            class FakeUsage:
                prompt_tokens: int = 10
                completion_tokens: int = 20
                total_tokens: int = 30
                raw_usage: object = None
            
            mock_norm.return_value = FakeUsage()
            result = usage_summary_for_api_request_hook(resp, provider="openai", api_mode="chat")
            assert result is not None
            assert result["prompt_tokens"] == 10
            assert result["total_tokens"] == 30
            assert "raw_usage" not in result


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------

class TestErrorUtilsBackwardCompat:
    """Verify module functions match AIAgent method behavior."""

    @pytest.fixture(autouse=True)
    def _make_agent(self):
        """Lightweight AIAgent fixture (no __init__)."""
        from run_agent import AIAgent
        self.agent = AIAgent.__new__(AIAgent)

    def test_summarize_api_error_compat(self):
        err = RuntimeError("test error with details")
        assert summarize_api_error(err) == self.agent._summarize_api_error(err)

    def test_mask_api_key_compat(self):
        key = "sk-abcdefghijklmnop"
        assert mask_api_key_for_logs(key) == self.agent._mask_api_key_for_logs(key)

    def test_clean_error_message_compat(self):
        msg = "Something went <html>badly</html> wrong"
        assert clean_error_message(msg) == self.agent._clean_error_message(msg)

    def test_clean_session_content_compat(self):
        content = "Hello\n\n\nworld"
        assert clean_session_content(content) == self.agent._clean_session_content(content)

    def test_extract_api_error_context_compat(self):
        err = Exception("test")
        err.body = {"error": {"code": "rate_limited", "message": "slow down"}}
        assert extract_api_error_context(err) == self.agent._extract_api_error_context(err)
