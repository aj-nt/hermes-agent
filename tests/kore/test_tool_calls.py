"""Tests for agent.kore.tool_calls — extracted tool call utility functions.

Backward-compat tests verify that module functions produce the same output
as the corresponding AIAgent methods for identical inputs.
"""

import copy
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
    VALID_API_ROLES,
    TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER,
)


# ---------------------------------------------------------------------------
# get_tool_call_id_static
# ---------------------------------------------------------------------------

class TestGetToolCallIdStatic:

    def test_dict_with_id(self):
        tc = {"id": "call_abc123"}
        assert get_tool_call_id_static(tc) == "call_abc123"

    def test_dict_without_id(self):
        """When dict has no 'id', return empty string."""
        tc = {"function": {"name": "test"}}
        assert get_tool_call_id_static(tc) == ""

    def test_object_with_id(self):
        tc = SimpleNamespace(id="call_xyz")
        assert get_tool_call_id_static(tc) == "call_xyz"


# ---------------------------------------------------------------------------
# sanitize_api_messages
# ---------------------------------------------------------------------------

class TestSanitizeApiMessages:

    def test_drops_invalid_roles(self):
        """Messages with roles not in VALID_API_ROLES are dropped."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "invalid_role", "content": "drop me"},
            {"role": "user", "content": "hello"},
        ]
        result = sanitize_api_messages(msgs)
        assert len(result) == 2
        assert all(m["role"] in VALID_API_ROLES for m in result)

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
            SimpleNamespace(function=SimpleNamespace(name="read_file")),
            SimpleNamespace(function=SimpleNamespace(name="write_file")),
        ]
        result = cap_delegate_task_calls(tool_calls)
        assert len(result) == 2

    def test_with_delegate_calls_capped(self):
        tool_calls = [
            SimpleNamespace(function=SimpleNamespace(name="delegate_task")),
            SimpleNamespace(function=SimpleNamespace(name="delegate_task")),
            SimpleNamespace(function=SimpleNamespace(name="delegate_task")),
            SimpleNamespace(function=SimpleNamespace(name="read_file")),
        ]
        with patch("tools.delegate_tool._get_max_concurrent_children", return_value=1):
            result = cap_delegate_task_calls(tool_calls)
        delegate_count = sum(1 for tc in result if tc.function.name == "delegate_task")
        assert delegate_count <= 1
        assert any(tc.function.name == "read_file" for tc in result)


# ---------------------------------------------------------------------------
# deduplicate_tool_calls
# ---------------------------------------------------------------------------

class TestDeduplicateToolCalls:

    def test_no_duplicates(self):
        tool_calls = [
            SimpleNamespace(id="call_1", function=SimpleNamespace(name="a", arguments="{}")),
            SimpleNamespace(id="call_2", function=SimpleNamespace(name="b", arguments="{}")),
        ]
        result = deduplicate_tool_calls(tool_calls)
        assert len(result) == 2

    def test_removes_duplicate_ids(self):
        tool_calls = [
            SimpleNamespace(id="call_1", function=SimpleNamespace(name="a", arguments="{}")),
            SimpleNamespace(id="call_1", function=SimpleNamespace(name="a", arguments="{}")),
            SimpleNamespace(id="call_2", function=SimpleNamespace(name="b", arguments="{}")),
        ]
        result = deduplicate_tool_calls(tool_calls)
        assert len(result) == 2
        names = [tc.function.name for tc in result]
        assert names.count("a") == 1

    def test_empty_list(self):
        assert deduplicate_tool_calls([]) == []


# ---------------------------------------------------------------------------
# deterministic_call_id
# ---------------------------------------------------------------------------

class TestDeterministicCallId:

    def test_returns_string(self):
        result = deterministic_call_id("read_file", '{"path": "/tmp"}', 0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic(self):
        r1 = deterministic_call_id("read_file", '{"path": "/tmp"}', 0)
        r2 = deterministic_call_id("read_file", '{"path": "/tmp"}', 0)
        assert r1 == r2

    def test_different_index_different_id(self):
        r1 = deterministic_call_id("read_file", '{"path": "/tmp"}', 0)
        r2 = deterministic_call_id("read_file", '{"path": "/tmp"}', 1)
        assert r1 != r2


# ---------------------------------------------------------------------------
# split_responses_tool_id
# ---------------------------------------------------------------------------

class TestSplitResponsesToolId:

    def test_with_tool_use_prefix(self):
        result = split_responses_tool_id("tool_use_read_file_abc123")
        assert isinstance(result, tuple)

    def test_passthrough(self):
        result = split_responses_tool_id("call_abc123")
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# sanitize_tool_calls_for_strict_api
# ---------------------------------------------------------------------------

class TestSanitizeToolCallsForStrictApi:

    def test_strips_codex_fields(self):
        """Strips call_id and response_item_id (Codex Responses API fields)."""
        api_msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                    "call_id": "chatcmpl-abc",
                    "response_item_id": "resp-xyz",
                }
            ],
        }
        result = sanitize_tool_calls_for_strict_api(copy.deepcopy(api_msg))
        tc = result["tool_calls"][0]
        assert "call_id" not in tc
        assert "response_item_id" not in tc
        # Required fields remain
        assert "id" in tc
        assert "function" in tc

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
        result = sanitize_tool_calls_for_strict_api(copy.deepcopy(api_msg))
        tc = result["tool_calls"][0]
        assert "id" in tc
        assert "type" in tc
        assert "function" in tc


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
        # Use SimpleNamespace to match what the agent actually passes
        tool_calls = [
            SimpleNamespace(id="call_1", function=SimpleNamespace(name="a", arguments="{}")),
            SimpleNamespace(id="call_1", function=SimpleNamespace(name="a", arguments="{}")),
        ]
        result_mod = deduplicate_tool_calls(tool_calls)
        result_agent = self.agent._deduplicate_tool_calls(tool_calls)
        assert len(result_mod) == len(result_agent)

    def test_deterministic_call_id_compat(self):
        result_mod = deterministic_call_id("test_fn", '{"key": "val"}', 0)
        result_agent = self.agent._deterministic_call_id("test_fn", '{"key": "val"}', 0)
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

    def test_valid_api_roles_compat(self):
        """Module constant matches AIAgent class constant."""
        assert VALID_API_ROLES == self.agent._VALID_API_ROLES

    def test_corruption_marker_compat(self):
        """Module constant matches AIAgent class constant."""
        assert TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER == self.agent._TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER
