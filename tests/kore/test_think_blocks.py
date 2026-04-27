
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


    @pytest.mark.parametrize("emoji", [
        "\U0001f49b",  # yellow heart
        "\u2728",       # sparkles
        "\U0001f64c",  # raised hands
        "\U0001f919",  # call me hand
        "\U0001f525",  # fire
        "\U0001f4aa",  # flexed bicep
        "\U0001f680",  # rocket
        "\U0001f60e",  # sunglasses
        "\U0001f60a",  # smiling face
        "\U0001f44b",  # waving hand
        "\u2764\ufe0f", # red heart
        "\U0001f64f",  # folded hands
        "\U0001f44d",  # thumbs up
        "\U0001f4af",  # 100
        "\U0001f389",  # party popper
        "\U0001fae1",  # salute
    ])
    def test_emoji_are_natural_endings(self, emoji):
        """Emoji at end of response should count as natural ending (#14572)."""
        assert has_natural_response_ending(
            f"Here's your answer. Let me know if you need more {emoji}"
        ) is True

    @pytest.mark.parametrize("symbol", [
        "\u2192",  # arrow right (Sm)
        "\u2190",  # arrow left (Sm)
        "\u2794",  # broad arrow (So)
        "\u221e",  # infinity (Sm)
        "\u2248",  # approximately equal (Sm)
        "\u2713",  # check mark (So)
        "\u2764",  # heavy black heart (So)
        "\u2605",  # black star (So)
    ])
    def test_math_symbols_and_sign_off_glyphs_are_natural(self, symbol):
        """Math symbols (Sm) and sign-off glyphs (So) are natural endings."""
        assert has_natural_response_ending(
            f"Here's the result {symbol}"
        ) is True

    def test_variation_selector_emoji_is_natural(self):
        """Heart with variation selector (VS16) should be natural."""
        # Red heart with VS16: e with variation selector
        assert has_natural_response_ending("Love this! \u2764\ufe0f") is True

    def test_triple_backtick_fence_ending(self):
        """Triple backtick fence is a natural ending for code blocks."""
        assert has_natural_response_ending("code()\n```") is True


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


class TestLooksLikeCodexIntermediateAck:
    """Tests for looks_like_codex_intermediate_ack()."""

    def test_returns_false_with_tool_messages(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        messages = [{"role": "tool", "content": "result"}]
        assert looks_like_codex_intermediate_ack("check the repo", "I'll look into it", messages) is False

    def test_returns_false_for_empty_assistant(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        assert looks_like_codex_intermediate_ack("check the repo", "", []) is False

    def test_returns_false_for_long_assistant(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        long_text = "x" * 1201
        assert looks_like_codex_intermediate_ack("check the repo", long_text, []) is False

    def test_returns_false_without_future_ack(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        assert looks_like_codex_intermediate_ack("check the repo", "The repo has 5 files.", []) is False

    def test_returns_true_with_workspace_ack(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        assert looks_like_codex_intermediate_ack(
            "check the repo",
            "I'll look into the codebase and report back.",
            [],
        ) is True

    def test_returns_true_with_user_workspace_target(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        assert looks_like_codex_intermediate_ack(
            "what's in the ~/project directory?",
            "I'll check that for you.",
            [],
        ) is True

    def test_returns_false_without_action_marker(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        # Has future ack but no action marker targeting workspace
        assert looks_like_codex_intermediate_ack(
            "tell me about weather",
            "I'll tell you about the weather.",
            [],
        ) is False

    def test_returns_true_with_curly_apostrophe(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        # Curly apostrophe (U+2019) in "I'll"
        assert looks_like_codex_intermediate_ack(
            "check the repo",
            "I\u2019ll review the codebase.",
            [],
        ) is True

    def test_think_block_stripped_before_check(self):
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        open_tag = chr(60) + "think" + chr(62)
        close_tag = chr(60) + "/think" + chr(62)
        content = open_tag + "reasoning" + close_tag + "I'll look into the project and report back."
        assert looks_like_codex_intermediate_ack("check the repo", content, []) is True

    def test_backward_compat_matches_agent(self):
        """Verify extracted function matches the AIAgent method."""
        from agent.kore.think_blocks import looks_like_codex_intermediate_ack
        from run_agent import AIAgent
        agent = AIAgent.__new__(AIAgent)
        cases = [
            ("check the repo", "I'll look into the codebase.", []),
            ("tell me about weather", "I'll tell you about the weather.", []),
            ("", "I'll check that.", []),
            ("check ~/src", "Let me review the project.", []),
            ("what's up", "The repo has 5 files.", []),
        ]
        for user_msg, assistant, msgs in cases:
            extracted = looks_like_codex_intermediate_ack(user_msg, assistant, msgs)
            original = agent._looks_like_codex_intermediate_ack(user_msg, assistant, msgs)
            assert extracted == original, f"Mismatch for: {user_msg!r}, {assistant!r}"
