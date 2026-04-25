"""Tests for agent.kore.reasoning — extract_reasoning and get_messages_up_to_last_assistant."""

import pytest
from agent.kore.reasoning import extract_reasoning, get_messages_up_to_last_assistant


class TestExtractReasoning:
    """Unit tests for extract_reasoning (pure function)."""

    def test_no_reasoning_returns_none(self):
        msg = type('M', (), {'content': 'hello'})()
        assert extract_reasoning(msg) is None

    def test_reasoning_field(self):
        msg = type('M', (), {'reasoning': 'I think therefore...', 'content': 'response'})()
        assert extract_reasoning(msg) == 'I think therefore...'

    def test_reasoning_content_field(self):
        msg = type('M', (), {'reasoning_content': 'deep thoughts', 'content': 'response'})()
        assert extract_reasoning(msg) == 'deep thoughts'

    def test_no_duplicate_reasoning(self):
        msg = type('M', (), {'reasoning': 'same', 'reasoning_content': 'same', 'content': 'response'})()
        result = extract_reasoning(msg)
        assert result == 'same'

    def test_combined_reasoning_parts(self):
        msg = type('M', (), {'reasoning': 'part1', 'reasoning_content': 'part2', 'content': 'response'})()
        result = extract_reasoning(msg)
        assert 'part1' in result
        assert 'part2' in result

    def test_reasoning_details(self):
        msg = type('M', (), {
            'reasoning_details': [
                {'summary': 'thought1'},
                {'thinking': 'thought2'},
            ],
            'content': 'response'
        })()
        result = extract_reasoning(msg)
        assert 'thought1' in result
        assert 'thought2' in result

    def test_inline_thinking_tags(self):
        msg = type('M', (), {'content': '<thinking>inner thought</thinking>response'})()
        result = extract_reasoning(msg)
        assert result == 'inner thought'


class TestGetMessagesUpToLastAssistant:
    """Unit tests for get_messages_up_to_last_assistant (pure function)."""

    def test_empty_messages(self):
        assert get_messages_up_to_last_assistant([]) == []

    def test_no_assistant_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = get_messages_up_to_last_assistant(msgs)
        assert len(result) == 2

    def test_truncates_at_last_assistant(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = get_messages_up_to_last_assistant(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_returns_copy_not_reference(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = get_messages_up_to_last_assistant(msgs)
        assert result is not msgs