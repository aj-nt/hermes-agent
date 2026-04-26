"""Tests for looks_like_codex_intermediate_ack extraction to agent.kore.think_blocks.

Backward-compat tests verify that the extracted kore function produces
the same results as the AIAgent method that delegates to it.
"""

import pytest
from agent.kore.think_blocks import looks_like_codex_intermediate_ack


class TestLooksLikeCodexIntermediateAck:
    """Pure function tests for looks_like_codex_intermediate_ack."""

    def test_tool_messages_present_returns_false(self):
        """If any tool messages exist, it's not an intermediate ack."""
        messages = [{"role": "tool", "content": "result"}]
        result = looks_like_codex_intermediate_ack(
            user_message="check the repo",
            assistant_content="I'll look into that.",
            messages=messages,
        )
        assert result is False

    def test_empty_assistant_content_returns_false(self):
        result = looks_like_codex_intermediate_ack(
            user_message="check the repo",
            assistant_content="",
            messages=[],
        )
        assert result is False

    def test_think_block_only_returns_false(self):
        """Content that's only think blocks (strips to empty) returns False."""
        result = looks_like_codex_intermediate_ack(
            user_message="check the repo",
            assistant_content="<think>planning...</think>",
            messages=[],
        )
        assert result is False

    def test_long_assistant_content_returns_false(self):
        """Content over 1200 chars returns False even with future ack."""
        result = looks_like_codex_intermediate_ack(
            user_message="check the repo",
            assistant_content="I'll look into that. " + "x" * 1200,
            messages=[],
        )
        assert result is False

    def test_no_future_ack_returns_false(self):
        """Without future-ack language, returns False."""
        result = looks_like_codex_intermediate_ack(
            user_message="check the repo",
            assistant_content="The codebase has 50 files.",
            messages=[],
        )
        assert result is False

    def test_future_ack_with_action_and_workspace(self):
        """Classic pattern: user asks about workspace, assistant says I'll look into it."""
        result = looks_like_codex_intermediate_ack(
            user_message="check the repo",
            assistant_content="I'll look into the codebase and report back.",
            messages=[],
        )
        assert result is True

    def test_future_ack_with_workspace_in_user(self):
        result = looks_like_codex_intermediate_ack(
            user_message="look at the directory structure",
            assistant_content="I'll inspect the files and analyze the structure.",
            messages=[],
        )
        assert result is True

    def test_future_ack_with_path_in_user(self):
        result = looks_like_codex_intermediate_ack(
            user_message="check ~/projects/myapp",
            assistant_content="I'll check the project and review the code.",
            messages=[],
        )
        assert result is True

    def test_future_ack_without_workspace_returns_false(self):
        """Future ack + action but no workspace mention = False."""
        result = looks_like_codex_intermediate_ack(
            user_message="tell me a joke",
            assistant_content="I'll help with that.",
            messages=[],
        )
        assert result is False

    def test_codex_style_ack(self):
        """Codex-style planning message with workspace reference."""
        result = looks_like_codex_intermediate_ack(
            user_message="scan the repository for bugs",
            assistant_content="I'll scan the codebase and explore the project structure.",
            messages=[],
        )
        assert result is True

    def test_case_insensitive_future_ack(self):
        """Future ack patterns match case-insensitively."""
        result = looks_like_codex_intermediate_ack(
            user_message="check the repo",
            assistant_content="I Will look into the repository.",
            messages=[],
        )
        assert result is True


class TestLooksLikeCodexIntermediateAckBackwardCompat:
    """Backward-compat: verify AIAgent method matches kore function."""

    def test_matches_kore_function(self):
        import run_agent
        agent = run_agent.AIAgent.__new__(run_agent.AIAgent)

        test_cases = [
            # (user, assistant, messages, expected)
            ("check the repo", "I'll look into the codebase and report back.", [], True),
            ("tell me a joke", "I'll help with that.", [], False),
            ("check the repo", "", [], False),
            ("check the repo", "I'll look into that.", [{"role": "tool", "content": "x"}], False),
        ]
        for user_msg, assistant, messages, expected in test_cases:
            result = agent._looks_like_codex_intermediate_ack(user_msg, assistant, messages)
            assert result == expected, f"user={user_msg!r}, assistant={assistant!r}: expected {expected}, got {result}"
            # Also verify matches kore function
            kore_result = looks_like_codex_intermediate_ack(user_msg, assistant, messages)
            assert result == kore_result