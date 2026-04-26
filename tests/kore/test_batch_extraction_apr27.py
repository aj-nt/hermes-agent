"""Tests for batch extraction of low-coupling methods from run_agent.py.

Covers:
  - anthropic_preserve_dots (url_helpers)
  - repair_tool_call (tool_calls)
  - format_tools_for_trajectory (trajectory)
  - convert_to_trajectory_format (trajectory)
  - cleanup_dead_connections (client_lifecycle)
  - apply_pending_steer_to_tool_results (steer)
"""

import json
import pytest


# ============================================================
# 1. anthropic_preserve_dots → url_helpers.anthropic_preserve_dots
# ============================================================

class TestAnthropicPreserveDots:
    """Pure function: provider/base_url → bool."""

    def test_alibaba_provider(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="alibaba") is True

    def test_minimax_provider(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="minimax") is True

    def test_bedrock_provider(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="bedrock") is True

    def test_opencode_go_provider(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="opencode-go") is True

    def test_zai_provider(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="zai") is True

    def test_openai_provider_false(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="openai") is False

    def test_dashscope_in_base_url(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(
            provider="custom", base_url="https://dashscope.aliyuncs.com/v1"
        ) is True

    def test_aliyuncs_in_base_url(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(
            provider="custom", base_url="https://api.aliyuncs.com/v1"
        ) is True

    def test_minimax_in_base_url(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(
            provider="custom", base_url="https://minimax.example.com/v1"
        ) is True

    def test_bedrock_runtime_in_base_url(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(
            provider="custom",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        ) is True

    def test_opencode_zen_in_base_url(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(
            provider="custom", base_url="https://opencode.ai/zen/v1"
        ) is True

    def test_bigmodel_in_base_url(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(
            provider="custom", base_url="https://open.bigmodel.cn/api/v1"
        ) is True

    def test_no_match_returns_false(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="openai", base_url="https://api.openai.com/v1") is False

    def test_none_provider_and_base_url(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider=None, base_url=None) is False

    def test_empty_provider_and_base_url(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="", base_url="") is False

    def test_provider_case_insensitive(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(provider="Alibaba") is True
        assert anthropic_preserve_dots(provider="MINIMAX") is True

    def test_base_url_case_insensitive(self):
        from agent.kore.url_helpers import anthropic_preserve_dots
        assert anthropic_preserve_dots(
            provider="", base_url="HTTPS://DASHSCOPE.ALICLOUD.COM/V1"
        ) is True


# ============================================================
# 2. repair_tool_call → tool_calls.repair_tool_call
# ============================================================

class TestRepairToolCall:
    """Fuzzy tool name matching via repair_tool_call."""

    def test_exact_match(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("read_file", {"read_file", "write_file"}) == "read_file"

    def test_lowercase_direct_match(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("Read_File", {"read_file", "write_file"}) == "read_file"

    def test_hyphen_to_underscore(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("read-file", {"read_file", "write_file"}) == "read_file"

    def test_space_to_underscore(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("read file", {"read_file", "write_file"}) == "read_file"

    def test_camel_case_to_snake(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("TodoTool", {"todo_tool", "write_file"}) == "todo_tool"

    def test_strip_tool_suffix(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("Patch_tool", {"patch", "write_file"}) == "patch"

    def test_double_tool_suffix(self):
        from agent.kore.tool_calls import repair_tool_call
        # TodoTool_tool → TodoTool → todo_tool
        assert repair_tool_call("TodoTool_tool", {"todo_tool", "write_file"}) == "todo_tool"

    def test_fuzzy_match(self):
        from agent.kore.tool_calls import repair_tool_call
        # "serch_files" fuzzy-matches "search_files"
        assert repair_tool_call("serch_files", {"search_files", "write_file"}) == "search_files"

    def test_no_match_returns_none(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("xyz_abc", {"read_file", "write_file"}) is None

    def test_empty_tool_name(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("", {"read_file"}) is None

    def test_empty_valid_names(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("read_file", set()) is None

    def test_browser_click_tool_suffix(self):
        from agent.kore.tool_calls import repair_tool_call
        assert repair_tool_call("BrowserClick_tool", {"browser_click"}) == "browser_click"


# ============================================================
# 3. format_tools_for_trajectory → trajectory.format_tools_for_trajectory
# ============================================================

class TestFormatToolsForTrajectory:
    """Format tool definitions for trajectory system message."""

    def test_empty_tools(self):
        from agent.trajectory import format_tools_for_trajectory
        assert format_tools_for_trajectory([]) == "[]"

    def test_single_tool(self):
        from agent.trajectory import format_tools_for_trajectory
        tools = [{"function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object"}}}]
        result = json.loads(format_tools_for_trajectory(tools))
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert result[0]["description"] == "Read a file"
        assert result[0]["required"] is None

    def test_multiple_tools(self):
        from agent.trajectory import format_tools_for_trajectory
        tools = [
            {"function": {"name": "read_file", "description": "Read", "parameters": {"type": "object"}}},
            {"function": {"name": "write_file", "description": "Write", "parameters": {"type": "object"}}},
        ]
        result = json.loads(format_tools_for_trajectory(tools))
        assert len(result) == 2
        assert result[0]["name"] == "read_file"
        assert result[1]["name"] == "write_file"

    def test_missing_description(self):
        from agent.trajectory import format_tools_for_trajectory
        tools = [{"function": {"name": "search", "parameters": {"type": "object"}}}]
        result = json.loads(format_tools_for_trajectory(tools))
        assert result[0]["description"] == ""

    def test_missing_parameters(self):
        from agent.trajectory import format_tools_for_trajectory
        tools = [{"function": {"name": "search", "description": "Search"}}]
        result = json.loads(format_tools_for_trajectory(tools))
        assert result[0]["parameters"] == {}


# ============================================================
# 4. convert_to_trajectory_format → trajectory.convert_to_trajectory_format
# ============================================================

class TestConvertToTrajectoryFormat:
    """Convert internal messages to ShareGPT trajectory format."""

    def _make_msg(self, role, content=None, tool_calls=None, reasoning=None, tool_call_id=None):
        msg = {"role": role}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        if reasoning is not None:
            msg["reasoning"] = reasoning
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
        return msg

    def test_simple_user_assistant(self):
        from agent.trajectory import convert_to_trajectory_format
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        trajectory = convert_to_trajectory_format(messages, "Hello", completed=True, tools_json="[]")
        # First entry is system, second is human prompt
        assert trajectory[0]["from"] == "system"
        assert trajectory[1]["from"] == "human"
        assert trajectory[1]["value"] == "Hello"
        # Assistant response
        assert trajectory[2]["from"] == "gpt"

    def test_tool_call_and_response(self):
        from agent.trajectory import convert_to_trajectory_format
        messages = [
            {"role": "user", "content": "Read file"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "test.txt"}'}, "id": "call_1"}
            ]},
            {"role": "tool", "content": "file contents", "tool_call_id": "call_1"},
        ]
        trajectory = convert_to_trajectory_format(messages, "Read file", completed=True, tools_json="[]")
        # Should have system, human, gpt (with tool call), tool response
        roles = [entry["from"] for entry in trajectory]
        assert "gpt" in roles
        assert "tool" in roles

    def test_completed_flag(self):
        from agent.trajectory import convert_to_trajectory_format
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        # Should not crash with completed=False
        trajectory = convert_to_trajectory_format(messages, "Hi", completed=False, tools_json="[]")
        assert len(trajectory) >= 2

    def test_reasoning_in_assistant_message(self):
        from agent.trajectory import convert_to_trajectory_format
        messages = [
            {"role": "user", "content": "Think"},
            {"role": "assistant", "content": "Answer", "reasoning": "Let me think..."},
        ]
        trajectory = convert_to_trajectory_format(messages, "Think", completed=True, tools_json="[]")
        gpt_entries = [e for e in trajectory if e["from"] == "gpt"]
        assert len(gpt_entries) >= 1
        assert "Let me think" in gpt_entries[0]["value"]

    def test_scratchpad_conversion(self):
        from agent.trajectory import convert_to_trajectory_format
        messages = [
            {"role": "user", "content": "Think"},
            {"role": "assistant", "content": "<REASONING_SCRATCHPAD>I reason</REASONING_SCRATCHPAD>Answer"},
        ]
        trajectory = convert_to_trajectory_format(messages, "Think", completed=True, tools_json="[]")
        gpt_entries = [e for e in trajectory if e["from"] == "gpt"]
        # Should convert REASONING_SCRATCHPAD tags to think tags
        assert "<REASONING_SCRATCHPAD>" not in gpt_entries[0]["value"]


# ============================================================
# 5. cleanup_dead_connections → client_lifecycle.cleanup_dead_connections
# ============================================================

class TestCleanupDeadConnections:
    """Pure-function detection of dead sockets in httpx connection pool."""

    def test_none_client_returns_false(self):
        from agent.kore.client_lifecycle import cleanup_dead_connections
        assert cleanup_dead_connections(None) is False

    def test_client_without_http_client_returns_false(self):
        from agent.kore.client_lifecycle import cleanup_dead_connections

        class FakeClient:
            pass

        assert cleanup_dead_connections(FakeClient()) is False

    def test_client_without_transport_returns_false(self):
        from agent.kore.client_lifecycle import cleanup_dead_connections

        class FakeClient:
            _client = type("HC", (), {})()

        assert cleanup_dead_connections(FakeClient()) is False

    def test_client_without_pool_returns_false(self):
        from agent.kore.client_lifecycle import cleanup_dead_connections

        class FakeClient:
            class _inner:
                pass
            _client = type("HC", (), {"_transport": type("T", (), {"_pool": None})()})()

        assert cleanup_dead_connections(FakeClient()) is False


# ============================================================
# 6. apply_pending_steer_to_tool_results → steer.apply_steer_to_tool_results
# ============================================================

class TestApplySteerToToolResults:
    """Append /steer text to the last tool result in a batch."""

    def test_no_tool_messages_no_op(self):
        from agent.kore.steer import apply_steer_to_tool_results
        messages = [{"role": "user", "content": "hi"}]
        # num_tool_msgs=0 → no changes, returns False
        result = apply_steer_to_tool_results(messages, num_tool_msgs=0, steer_text="guide")
        assert result is False

    def test_no_steer_text_no_op(self):
        from agent.kore.steer import apply_steer_to_tool_results
        messages = [
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        result = apply_steer_to_tool_results(messages, num_tool_msgs=1, steer_text="")
        assert result is False

    def test_appends_steer_to_last_tool_result(self):
        from agent.kore.steer import apply_steer_to_tool_results
        messages = [
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "file contents", "tool_call_id": "1"},
        ]
        result = apply_steer_to_tool_results(messages, num_tool_msgs=1, steer_text="be concise")
        assert result is True
        assert "be concise" in messages[-1]["content"]
        assert "file contents" in messages[-1]["content"]

    def test_appends_steer_to_multimodal_content(self):
        from agent.kore.steer import apply_steer_to_tool_results
        messages = [
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": [
                {"type": "text", "text": "image result"},
            ], "tool_call_id": "1"},
        ]
        result = apply_steer_to_tool_results(messages, num_tool_msgs=1, steer_text="focus")
        assert result is True
        # Should append a text block
        content = messages[-1]["content"]
        assert isinstance(content, list)
        assert any("focus" in str(b) for b in content)

    def test_no_tool_role_in_range_returns_false(self):
        from agent.kore.steer import apply_steer_to_tool_results
        messages = [
            {"role": "assistant", "content": "no tools here"},
        ]
        result = apply_steer_to_tool_results(messages, num_tool_msgs=1, steer_text="guide")
        assert result is False

    def test_multiple_tool_results_steer_on_last(self):
        from agent.kore.steer import apply_steer_to_tool_results
        messages = [
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "result1", "tool_call_id": "1"},
            {"role": "tool", "content": "result2", "tool_call_id": "2"},
        ]
        result = apply_steer_to_tool_results(messages, num_tool_msgs=2, steer_text="summarize")
        assert result is True
        assert "summarize" in messages[-1]["content"]
        assert "summarize" not in messages[-2]["content"]

    def test_none_steer_text_no_op(self):
        from agent.kore.steer import apply_steer_to_tool_results
        messages = [
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        result = apply_steer_to_tool_results(messages, num_tool_msgs=1, steer_text=None)
        assert result is False