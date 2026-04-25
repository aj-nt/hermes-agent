
"""Tests for think_blocks module.

These verify the extracted think-block stripping and content analysis
functions produce identical results to the original AIAgent methods.
"""

import pytest

from agent.kore.think_blocks import (
    strip_think_blocks,
    has_natural_response_ending,
    has_content_after_think_block,
)


class TestStripThinkBlocks:
    """Tests for strip_think_blocks()."""

    def test_empty_string(self):
        assert strip_think_blocks("") == ""

    def test_no_think_tags_passthrough(self):
        assert strip_think_blocks("Hello world") == "Hello world"

    def test_think_tags_stripped(self):
        content = "before" + chr(60) + "think" + chr(62) + "reasoning" + chr(60) + "/think" + chr(62) + "after"
        result = strip_think_blocks(content)
        assert "reasoning" not in result
        assert "before" in result
        assert "after" in result

    def test_thinking_tags_stripped(self):
        content = "before" + chr(60) + "thinking" + chr(62) + "deep" + chr(60) + "/thinking" + chr(62) + "after"
        result = strip_think_blocks(content)
        assert "deep" not in result

    def test_reasoning_tags_stripped(self):
        content = "a" + chr(60) + "reasoning" + chr(62) + "logic" + chr(60) + "/reasoning" + chr(62) + "b"
        result = strip_think_blocks(content)
        assert result == "ab"

    def test_case_insensitive_tags(self):
        content = "before" + chr(60) + "THINK" + chr(62) + "secret" + chr(60) + "/THINK" + chr(62) + "after"
        result = strip_think_blocks(content)
        assert "secret" not in result

    def test_unterminated_think_block(self):
        tag = chr(60) + "think"
        content = "visible text" + chr(10) + tag + chr(62) + "rest is reasoning"
        result = strip_think_blocks(content)
        assert tag not in result

    def test_tool_call_tags_stripped(self):
        content = "before" + chr(60) + "tool_calls" + chr(62) + "call()" + chr(60) + "/tool_calls" + chr(62) + "after"
        result = strip_think_blocks(content)
        assert "call" not in result

    def test_multiple_think_blocks(self):
        open_tag = chr(60) + "think" + chr(62)
        close_tag = chr(60) + "/think" + chr(62)
        content = "start" + open_tag + "one" + close_tag + "mid" + open_tag + "two" + close_tag + "end"
        assert strip_think_blocks(content) == "startmidend"

    def test_whitespace_only_passthrough(self):
        assert strip_think_blocks("Hello world") == "Hello world"


class TestHasNaturalResponseEnding:
    """Tests for has_natural_response_ending()."""

    def test_empty_string(self):
        assert has_natural_response_ending("") is False

    def test_period(self):
        assert has_natural_response_ending("Hello.") is True

    def test_exclamation(self):
        assert has_natural_response_ending("Hello!") is True

    def test_question_mark(self):
        assert has_natural_response_ending("Hello?") is True

    def test_no_ending_punct(self):
        assert has_natural_response_ending("Hello") is False

    def test_cjk_period(self):
        assert has_natural_response_ending("\u3002") is True

    def test_triple_backtick(self):
        assert has_natural_response_ending("code()\n```") is True

    def test_closing_bracket(self):
        assert has_natural_response_ending("result]") is True

    def test_whitespace_only(self):
        assert has_natural_response_ending("   ") is False

    def test_trailing_whitespace_stripped(self):
        assert has_natural_response_ending("Hello.  ") is True


class TestHasContentAfterThinkBlock:
    """Tests for has_content_after_think_block()."""

    def test_empty(self):
        assert has_content_after_think_block("") is False

    def test_only_think_block(self):
        content = chr(60) + "think" + chr(62) + "reasoning" + chr(60) + "/think" + chr(62)
        assert has_content_after_think_block(content) is False

    def test_think_block_with_visible_text(self):
        open_tag = chr(60) + "think" + chr(62)
        close_tag = chr(60) + "/think" + chr(62)
        content = open_tag + "reasoning" + close_tag + "visible text"
        assert has_content_after_think_block(content) is True

    def test_no_think_block(self):
        assert has_content_after_think_block("just text") is True

    def test_whitespace_after_think_block(self):
        open_tag = chr(60) + "think" + chr(62)
        close_tag = chr(60) + "/think" + chr(62)
        content = open_tag + "reasoning" + close_tag + "   "
        assert has_content_after_think_block(content) is False


class TestThinkBlocksBackwardCompat:
    """Verify extracted functions produce identical results to AIAgent methods."""

    @pytest.fixture
    def agent(self):
        from run_agent import AIAgent
        return AIAgent.__new__(AIAgent)

    def test_strip_think_blocks_matches_agent(self, agent):
        open_tag = chr(60) + "think" + chr(62)
        close_tag = chr(60) + "/think" + chr(62)
        cases = [
            "",
            "plain text",
            "before" + open_tag + "reasoning" + close_tag + "after",
            "text" + chr(60) + "thinking" + chr(62) + "deep" + chr(60) + "/thinking" + chr(62) + "more",
        ]
        for case in cases:
            extracted = strip_think_blocks(case)
            original = agent._strip_think_blocks(case)
            assert extracted == original, f"Mismatch for: {case!r}"

    def test_has_natural_ending_matches_agent(self, agent):
        from run_agent import AIAgent as _AIAgent
        cases = ["", "Hello.", "Hello", "text"]
        for case in cases:
            extracted = has_natural_response_ending(case)
            original = _AIAgent._has_natural_response_ending(case)
            assert extracted == original, f"Mismatch for: {case!r}"

    def test_has_content_after_think_matches_agent(self, agent):
        open_tag = chr(60) + "think" + chr(62)
        close_tag = chr(60) + "/think" + chr(62)
        cases = [
            "",
            open_tag + "reasoning" + close_tag,
            open_tag + "reasoning" + close_tag + "visible",
            "just text",
        ]
        for case in cases:
            extracted = has_content_after_think_block(case)
            original = agent._has_content_after_think_block(case)
            assert extracted == original, f"Mismatch for: {case!r}"
