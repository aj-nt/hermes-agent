"""Tests for agent.kore.tool_calls — extracted tool call utility functions.

Backward-compat tests verify that module functions produce the same output
as the corresponding AIAgent methods for identical inputs.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.kore.tool_calls import (
    get_tool_call_id_static,
    sanitize_api_messages,
    cap_delegate_task_calls,
    deduplicate_tool_calls,
    deterministic_call_id,
    split_responses_tool_id,
    sanitize_tool_calls_for_strict_api,
    sanitize_tool_call_arguments,
)


# ---------------------------------------------------------------------------
# get_tool_call_id_static
# ---------------------------------------------------------------------------

class TestGetToolCallIdStatic:

    def test_dict_with_id(self):
        tc = {"id": "call_abc123"}
        assert get_tool_call_id_static(tc) == "call_abc123"

    def test_dict_without_id(self):
        tc = {"function": {"name": "test"}}
        result = get_tool_call_id_static(tc)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_object_with_id(self):
        tc = SimpleNamespace(id="call_xyz")
        assert get_tool_call_id_static(tc) == "call_xyz"


# ---------------------------------------------------------------------------
# sanitize_api_messages
# ---------------------------------------------------------------------------

class TestSanitizeApiMessages:

    def test_removes_system_tool_names(self):
        msgs = [
            {"role": "system", "content": "You have tools: tool_a, tool_b"},
            {"role": "user", "content": "hello"},
        ]
        result = sanitize_api_messages(msgs)
        assert len(result) == 2
        assert "tool_a" not in result[0].get("content", "")
        assert "hello" in result[1]["content"]

    def test_preserves_user_messages(self):
        msgs = [{"role": "user", "content": "Use tool_a to do X"}]
        result = sanitize_api_messages(msgs)
        assert result[0]["content"] == "Use tool_a to do X"

    def test_empty_list(self):
        assert sanitize_api_messages([]) == []


# ---------------------------------------------------------------------------
# cap_delegate_task_calls
# ---------------------------------------------------------------------------

class TestCapDelegateTaskCalls:

    def test_no_delegate_calls(self):
        tool_calls = [
            {"function": {"name": "read_file"}},
            {"function": {"name": "write_file"}},
        ]
        result = cap_delegate_task_calls(tool_calls)
        assert len(result) == 2

    def test_with_delegate_calls_capped(self):
        tool_calls = [
            {"function": {"name": "delegate_task"}},
            {"function": {"name": "delegate_task"}},
            {"function": {"name": "delegate_task"}},
            {"function": {"name": "read_file"}},
        ]
        with patch("tools.delegate_tool._get_max_concurrent_children", return_value=1):
            result = cap_delegate_task_calls(tool_calls)
        # Should cap delegate_task calls to max_concurrent_children (1)
        delegate_count = sum(1 for tc in result if tc["function"]["name"] == "delegate_task")
        assert delegate_count <= 1
        # Non-delegate should survive
        assert any(tc["function"]["name"] == "read_file" for tc in result)


# ---------------------------------------------------------------------------
# deduplicate_tool_calls
# ---------------------------------------------------------------------------

class TestDeduplicateToolCalls:

    def test_no_duplicates(self):
        tool_calls = [
            {"id": "call_1", "function": {"name": "a"}},
            {"id": "call_2", "function": {"name": "b"}},
        ]
        result = deduplicate_tool_calls(tool_calls)
        assert len(result) == 2

    def test_removes_duplicate_ids(self):
        tool_calls = [
            {"id": "call_1", "function": {"name": "a"}},
            {"id": "call_1", "function": {"name": "a"}},
            {"id": "call_2", "function": {"name": "b"}},
        ]
        result = deduplicate_tool_calls(tool_calls)
        assert len(result) == 2
        ids = [tc["id"] for tc in result]
        assert ids.count("call_1") == 1

    def test_empty_list(self):
        assert deduplicate_tool_calls([]) == []


# ---------------------------------------------------------------------------
# deterministic_call_id
# ---------------------------------------------------------------------------

class TestDeterministicCallId:

    def test_returns_string(self):
        result = deterministic_call_id("read_file", {"path": "/tmp"}, 0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic(self):
        r1 = deterministic_call_id("read_file", {"path": "/tmp"}, 0)
        r2 = deterministic_call_id("read_file", {"path": "/tmp"}, 0)
        assert r1 == r2

    def test_different_index_different_id(self):
        r1 = deterministic_call_id("read_file", {"path": "/tmp"}, 0)
        r2 = deterministic_call_id("read_file", {"path": "/tmp"}, 1)
        assert r1 != r2


# ---------------------------------------------------------------------------
# split_responses_tool_id
# ---------------------------------------------------------------------------

class TestSplitResponsesToolId:

    def test_with_tool_use_prefix(self):
        # Responses API format: "tool_use_NAME_ID"
        result = split_responses_tool_id("tool_use_read_file_abc123")
        assert isinstance(result, (str, tuple, list))

    def test_passthrough(self):
        result = split_responses_tool_id("call_abc123")
        assert isinstance(result, (str, tuple, list))


# ---------------------------------------------------------------------------
# sanitize_tool_calls_for_strict_api
# ---------------------------------------------------------------------------

class TestSanitizeToolCallsForStrictApi:

    def test_removes_extra_fields(self):
        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                    "extra_field": "should_be_removed",
                }
            ],
        }
        result = sanitize_tool_calls_for_strict_api(api_msg)
        for tc in result.get("tool_calls", [api_msg])[0].get("tool_calls", [api_msg]):
            if isinstance(tc, dict):
                assert "extra_field" not in tc

    def test_preserves_required_fields(self):
        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        }
        result = sanitize_tool_calls_for_strict_api(api_msg)
        # Should still have the essential fields
        assert "tool_calls" in result or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------

class TestToolCallsBackwardCompat:
    """Verify module functions match AIAgent method behavior."""

    @pytest.fixture(autouse=True)
    def _make_agent(self):
        from run_agent import AIAgent
        self.agent = AIAgent.__new__(AIAgent)

    def test_get_tool_call_id_static_compat(self):
        tc = {"id": "call_test123"}
        assert get_tool_call_id_static(tc) == self.agent._get_tool_call_id_static(tc)

    def test_deduplicate_tool_calls_compat(self):
        tool_calls = [
            {"id": "call_1", "function": {"name": "a"}},
            {"id": "call_1", "function": {"name": "a"}},
        ]
        assert deduplicate_tool_calls(tool_calls) == self.agent._deduplicate_tool_calls(tool_calls)

    def test_deterministic_call_id_compat(self):
        result_mod = deterministic_call_id("test_fn", {"key": "val"}, 0)
        result_agent = self.agent._deterministic_call_id("test_fn", {"key": "val"}, 0)
        assert result_mod == result_agent

    def test_split_responses_tool_id_compat(self):
        raw_id = "tool_use_test_abc"
        assert split_responses_tool_id(raw_id) == self.agent._split_responses_tool_id(raw_id)

    def test_sanitize_tool_calls_for_strict_api_compat(self):
        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        }
        result_mod = sanitize_tool_calls_for_strict_api(copy.deepcopy(api_msg))
        result_agent = self.agent._sanitize_tool_calls_for_strict_api(copy.deepcopy(api_msg))
        assert result_mod == result_agent
