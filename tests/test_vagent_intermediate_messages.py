"""Tests for vagent intermediate message accumulation and session persistence.

The ToolExecutorService records every tool call and result as OpenAI-format
message dicts.  After the gRPC stream completes, vagent_grpc drains these
intermediate messages and builds a full conversation history that includes
all assistant(tool_calls) and tool(result) entries — enabling session search
to show tool usage, not just the final user/assistant pair.
"""
import json
import threading

import pytest


# ── Unit tests: message accumulator (no gRPC server needed) ───────────


def test_tool_call_accumulation():
    """Tool call messages are recorded as assistant messages with tool_calls."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    batch = agent_pb2.ToolBatch(
        calls=[agent_pb2.ToolCall(
            call_id="call_1",
            name="terminal",
            arguments_json='{"command":"ls"}',
        )]
    )
    _record_tool_call_messages(batch)

    msgs = drain_intermediate_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] is None
    assert len(msgs[0]["tool_calls"]) == 1
    assert msgs[0]["tool_calls"][0]["id"] == "call_1"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "terminal"
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"command":"ls"}'


def test_tool_result_accumulation():
    """Tool result messages are recorded as role=tool with tool_call_id."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_result_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    _record_tool_result_messages([
        agent_pb2.ToolResult(
            call_id="call_1",
            output_json='{"files": ["a.txt"]}',
            is_error=False,
        ),
    ])

    msgs = drain_intermediate_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == '{"files": ["a.txt"]}'


def test_tool_call_then_result_ordering():
    """Tool call message appears before tool result message in the accumulator."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        _record_tool_result_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    batch = agent_pb2.ToolBatch(
        calls=[agent_pb2.ToolCall(
            call_id="call_1",
            name="terminal",
            arguments_json='{"command":"ls"}',
        )]
    )
    _record_tool_call_messages(batch)
    _record_tool_result_messages([
        agent_pb2.ToolResult(
            call_id="call_1",
            output_json='"file.txt"',
            is_error=False,
        ),
    ])

    msgs = drain_intermediate_messages()
    # assistant(tool_calls) first, then tool(result)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"] is not None
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "call_1"


def test_multiple_batches():
    """Multiple tool batches produce alternating assistant/tool groups."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        _record_tool_result_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    # First batch: terminal call
    batch1 = agent_pb2.ToolBatch(
        calls=[agent_pb2.ToolCall(
            call_id="call_1",
            name="terminal",
            arguments_json='{"command":"ls"}',
        )]
    )
    _record_tool_call_messages(batch1)
    _record_tool_result_messages([
        agent_pb2.ToolResult(call_id="call_1", output_json='"file.txt"', is_error=False),
    ])

    # Second batch: web search
    batch2 = agent_pb2.ToolBatch(
        calls=[agent_pb2.ToolCall(
            call_id="call_2",
            name="web_search",
            arguments_json='{"query":"test"}',
        )]
    )
    _record_tool_call_messages(batch2)
    _record_tool_result_messages([
        agent_pb2.ToolResult(call_id="call_2", output_json='"results"', is_error=False),
    ])

    msgs = drain_intermediate_messages()
    # 4 messages: assistant(tool_calls_1), tool(result_1), assistant(tool_calls_2), tool(result_2)
    assert len(msgs) == 4
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["id"] == "call_1"
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "call_1"
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["id"] == "call_2"
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == "call_2"


def test_reset_clears_accumulator():
    """reset_intermediate_messages clears all accumulated messages."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    batch = agent_pb2.ToolBatch(
        calls=[agent_pb2.ToolCall(call_id="c1", name="terminal", arguments_json="{}")]
    )
    _record_tool_call_messages(batch)
    assert len(drain_intermediate_messages()) == 1

    reset_intermediate_messages()
    assert len(drain_intermediate_messages()) == 0


def test_message_list_built_correctly():
    """The full messages list includes conversation_history + user + intermediates + final.

    This simulates the composition logic in vagent_grpc.run_vagent_turn:
      messages = conversation_history + [user] + drain_intermediate_messages() + [final_assistant]
    """
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        _record_tool_result_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    # Simulate vagent turn: one tool batch executed
    batch = agent_pb2.ToolBatch(
        calls=[agent_pb2.ToolCall(call_id="call_1", name="terminal", arguments_json='{"command":"ls"}')]
    )
    _record_tool_call_messages(batch)
    _record_tool_result_messages([
        agent_pb2.ToolResult(call_id="call_1", output_json='"file.txt"', is_error=False),
    ])

    # Simulate what vagent_grpc.py does to build the messages list
    conversation_history = [
        {"role": "user", "content": "previous message"},
        {"role": "assistant", "content": "previous response"},
    ]
    user_message = "list files"
    final_response = "I found the following files: file.txt"

    intermediate_messages = drain_intermediate_messages()
    messages = list(conversation_history)
    messages.append({"role": "user", "content": user_message})
    messages.extend(intermediate_messages)
    messages.append({"role": "assistant", "content": final_response})

    # Expected sequence (6 messages):
    #   0: user (history)
    #   1: assistant (history)
    #   2: user (current turn)
    #   3: assistant with tool_calls
    #   4: tool result
    #   5: assistant (final response)
    assert len(messages) == 6
    assert messages[0] == {"role": "user", "content": "previous message"}
    assert messages[1] == {"role": "assistant", "content": "previous response"}
    assert messages[2] == {"role": "user", "content": "list files"}
    assert messages[3]["role"] == "assistant"
    assert messages[3]["content"] is None
    assert messages[3]["tool_calls"][0]["id"] == "call_1"
    assert messages[4]["role"] == "tool"
    assert messages[4]["tool_call_id"] == "call_1"
    assert messages[5]["role"] == "assistant"
    assert messages[5]["content"] == "I found the following files: file.txt"


def test_thread_safety():
    """Concurrent record + drain should not lose or corrupt data."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        _record_tool_result_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    errors = []

    def writer(thread_id):
        try:
            for i in range(50):
                batch = agent_pb2.ToolBatch(
                    calls=[agent_pb2.ToolCall(
                        call_id=f"t{thread_id}_c{i}",
                        name="terminal",
                        arguments_json='{}',
                    )]
                )
                _record_tool_call_messages(batch)
                _record_tool_result_messages([
                    agent_pb2.ToolResult(
                        call_id=f"t{thread_id}_c{i}",
                        output_json='"ok"',
                        is_error=False,
                    )
                ])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised: {errors}"

    msgs = drain_intermediate_messages()
    # 4 threads x 50 iterations x 2 messages (call + result) = 400
    assert len(msgs) == 400


def test_empty_batch_is_noop():
    """An empty ToolBatch records nothing."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    batch = agent_pb2.ToolBatch(calls=[])
    _record_tool_call_messages(batch)
    assert len(drain_intermediate_messages()) == 0


def test_no_intermediates_when_turn_has_no_tools():
    """A turn with no tool calls produces an empty intermediate list."""
    from agent.transports.vagent_tool_executor import (
        drain_intermediate_messages,
        reset_intermediate_messages,
    )

    reset_intermediate_messages()

    # No calls to _record_* at all — simulate a turn where the LLM
    # responds directly without invoking any tools.
    intermediate_messages = drain_intermediate_messages()

    conversation_history = [{"role": "user", "content": "hello"}]
    messages = list(conversation_history)
    messages.append({"role": "user", "content": "hi again"})
    messages.extend(intermediate_messages)
    messages.append({"role": "assistant", "content": "Hello!"})

    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_multi_tool_batch():
    """A single batch with multiple tool calls produces one assistant message with N tool_calls."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        _record_tool_result_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    # LLM requests 3 tools in a single batch (parallel tool calls)
    batch = agent_pb2.ToolBatch(
        calls=[
            agent_pb2.ToolCall(call_id="c1", name="terminal", arguments_json='{"command":"ls"}'),
            agent_pb2.ToolCall(call_id="c2", name="web_search", arguments_json='{"query":"test"}'),
            agent_pb2.ToolCall(call_id="c3", name="read_file", arguments_json='{"path":"/tmp/x"}'),
        ]
    )
    _record_tool_call_messages(batch)
    _record_tool_result_messages([
        agent_pb2.ToolResult(call_id="c1", output_json='"file.txt"', is_error=False),
        agent_pb2.ToolResult(call_id="c2", output_json='"results"', is_error=False),
        agent_pb2.ToolResult(call_id="c3", output_json='"contents"', is_error=False),
    ])

    msgs = drain_intermediate_messages()
    # 1 assistant message with 3 tool_calls + 3 tool result messages = 4
    assert len(msgs) == 4
    assert msgs[0]["role"] == "assistant"
    assert len(msgs[0]["tool_calls"]) == 3
    assert msgs[1]["role"] == "tool"
    assert msgs[2]["role"] == "tool"
    assert msgs[3]["role"] == "tool"

    # Verify tool result messages are keyed correctly
    call_ids = {msgs[0]["tool_calls"][i]["id"] for i in range(3)}
    result_ids = {msgs[i]["tool_call_id"] for i in range(1, 4)}
    assert call_ids == result_ids


def test_error_tool_results():
    """Error tool results are captured with is_error flag."""
    from agent.transports.vagent_tool_executor import (
        _record_tool_call_messages,
        _record_tool_result_messages,
        drain_intermediate_messages,
        reset_intermediate_messages,
    )
    import agent_pb2

    reset_intermediate_messages()

    batch = agent_pb2.ToolBatch(
        calls=[agent_pb2.ToolCall(call_id="e1", name="terminal", arguments_json='{"command":"rm -rf /"}')]
    )
    _record_tool_call_messages(batch)
    _record_tool_result_messages([
        agent_pb2.ToolResult(
            call_id="e1",
            output_json='{"error": "permission denied"}',
            is_error=True,
        ),
    ])

    msgs = drain_intermediate_messages()
    assert len(msgs) == 2
    # The tool result content is recorded as-is
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "e1"
    result_content = json.loads(msgs[1]["content"])
    assert "error" in result_content