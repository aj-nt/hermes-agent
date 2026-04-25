"""Tests for agent.kore.message_prep — Qwen message formatting functions."""

import copy
import pytest
from agent.kore.message_prep import (
    qwen_prepare_chat_messages,
    qwen_prepare_chat_messages_inplace,
)


class TestQwenPrepareChatMessages:
    """Unit tests for qwen_prepare_chat_messages (pure function)."""

    def test_empty_list_returns_empty(self):
        assert qwen_prepare_chat_messages([]) == []

    def test_string_content_wrapped(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = qwen_prepare_chat_messages(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_already_structured_content_kept(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = qwen_prepare_chat_messages(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_does_not_mutate_original(self):
        original = [{"role": "user", "content": "hello"}]
        result = qwen_prepare_chat_messages(original)
        # Original should still have string content
        assert original[0]["content"] == "hello"
        # Result should have list content
        assert isinstance(result[0]["content"], list)

    def test_system_message_cache_control_injected(self):
        msgs = [{"role": "system", "content": [{"type": "text", "text": "You are helpful."}]}]
        result = qwen_prepare_chat_messages(msgs)
        # Last content part of system message should have cache_control
        last_part = result[0]["content"][-1]
        assert "cache_control" in last_part
        assert last_part["cache_control"] == {"type": "ephemeral"}

    def test_no_cache_control_on_user_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = qwen_prepare_chat_messages(msgs)
        # No system message means no cache_control
        user_parts = result[0]["content"]
        for part in user_parts:
            assert "cache_control" not in part


class TestQwenPrepareChatMessagesInplace:
    """Unit tests for qwen_prepare_chat_messages_inplace (mutating)."""

    def test_empty_list_noop(self):
        msgs = []
        qwen_prepare_chat_messages_inplace(msgs)
        assert msgs == []

    def test_string_content_mutated(self):
        msgs = [{"role": "user", "content": "hello"}]
        qwen_prepare_chat_messages_inplace(msgs)
        assert msgs[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_returns_none(self):
        result = qwen_prepare_chat_messages_inplace([{"role": "user", "content": "hi"}])
        assert result is None

    def test_system_message_cache_control(self):
        msgs = [{"role": "system", "content": [{"type": "text", "text": "You are helpful."}]}]
        qwen_prepare_chat_messages_inplace(msgs)
        last_part = msgs[0]["content"][-1]
        assert "cache_control" in last_part