"""Tests for agent.kore.display_utils -- extracted display/formatting utilities."""

import pytest

from agent.kore.display_utils import (
    summarize_background_review_actions,
    wrap_verbose,
    normalize_interim_visible_text,
)


class TestSummarizeBackgroundReviewActions:

    def test_empty_messages(self):
        result = summarize_background_review_actions([], None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_review_messages_with_tool_calls(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\": \"/tmp/test\"}"}}]},
            {"role": "tool", "content": "file contents here"},
        ]
        result = summarize_background_review_actions(messages, None)
        assert isinstance(result, list)

    def test_review_messages_without_tool_calls(self):
        messages = [
            {"role": "assistant", "content": "I will check the file."},
        ]
        result = summarize_background_review_actions(messages, None)
        assert isinstance(result, list)


class TestWrapVerbose:

    def test_short_text(self):
        result = wrap_verbose("LABEL", "short text", indent="  ")
        assert "LABEL" in result

    def test_long_text_wraps(self):
        result = wrap_verbose("LABEL", "a" * 200, indent="  ")
        assert "LABEL" in result

    def test_custom_indent(self):
        result = wrap_verbose("LABEL", "text", indent="  ")
        assert result.startswith("  LABEL")


class TestNormalizeInterimVisibleText:

    def test_strips_whitespace(self):
        result = normalize_interim_visible_text("  hello  ")
        assert result == "hello"

    def test_none_returns_empty(self):
        result = normalize_interim_visible_text(None)
        assert result == ""

    def test_empty_string(self):
        result = normalize_interim_visible_text("")
        assert result == ""


class TestDisplayUtilsBackwardCompat:

    @pytest.fixture(autouse=True)
    def _make_agent(self):
        from run_agent import AIAgent
        self.agent = AIAgent.__new__(AIAgent)

    def test_summarize_background_review_actions_compat(self):
        messages = [
            {"role": "assistant", "content": "Checking the file."},
        ]
        result_mod = summarize_background_review_actions(messages, None)
        result_agent = self.agent._summarize_background_review_actions(messages, None)
        assert result_mod == result_agent

    def test_wrap_verbose_compat(self):
        result_mod = wrap_verbose("TEST", "some text content")
        result_agent = self.agent._wrap_verbose("TEST", "some text content")
        assert result_mod == result_agent

    def test_normalize_interim_visible_text_compat(self):
        for text in ["  hello  ", None, "", "normal text"]:
            assert normalize_interim_visible_text(text) == self.agent._normalize_interim_visible_text(text)
