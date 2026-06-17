"""vagent tool executor — standalone gRPC server that receives tool calls from
vagent and dispatches them using hermes-agent's handle_function_call.

Launched as a sidecar process by vagent_grpc.py's run_vagent_turn().
Vagent calls back to this server to execute tools during the agent loop.

Message accumulation: ToolExecutorService records every tool call and result
as intermediate messages (OpenAI-format dicts).  The caller drains these via
drain_intermediate_messages() to build a full conversation history for
session persistence.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional

import grpc

import agent_pb2
import agent_pb2_grpc

logger = logging.getLogger(__name__)

DEFAULT_PORT = 50053

# ── Tool dispatch — plugs into hermes-agent's handle_function_call ──

_tool_dispatcher: Optional[Callable] = None
_dispatch_lock = threading.Lock()


def set_tool_dispatcher(fn: Callable[[str, Dict[str, Any], str], str]) -> None:
    """Set the tool dispatch function from the hermes-agent side.

    fn signature: handle_function_call(tool_name: str, arguments: dict, task_id: str) -> str
    Returns a JSON string — the tool result.
    """
    global _tool_dispatcher
    with _dispatch_lock:
        _tool_dispatcher = fn


# ── Intermediate message accumulation ────────────────────────────────
# Thread-safe list of message dicts in OpenAI format that vagent_grpc
# drains after processing each tool_batch event.  This allows the
# Python side to reconstruct the full conversation (including tool
# calls and results) for session persistence — without any proto changes.

_accum_lock = threading.Lock()
_accum_messages: List[Dict[str, Any]] = []
# Maps call_id → tool_name so result messages can carry tool_name for
# session DB indexing (FTS on the tool_name column).  Populated by
# _record_tool_call_messages, consumed by _record_tool_result_messages.
_call_id_to_name: Dict[str, str] = {}

def _record_tool_call_messages(batch: agent_pb2.ToolBatch) -> None:
    """Record an assistant message with tool_calls from a ToolBatch event.

    Called by the executor *before* dispatching, so the tool_call message
    appears before the result messages in the accumulated list.

    Also builds a call_id → tool_name mapping so that tool result messages
    can include the ``tool_name`` field for session DB indexing.
    """
    if not batch.calls:
        return

    tool_calls = []
    name_map: Dict[str, str] = {}
    for call in batch.calls:
        call_id = call.call_id
        name = call.name
        tool_calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": call.arguments_json,
            },
        })
        name_map[call_id] = name

    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
    with _accum_lock:
        _accum_messages.append(msg)
        _call_id_to_name.update(name_map)

def _record_tool_result_messages(results: List[agent_pb2.ToolResult]) -> None:
    """Record tool result messages after execution.

    Each result becomes a role="tool" message keyed by tool_call_id.
    Includes ``tool_name`` (resolved from the earlier tool call) so that
    _flush_messages_to_session_db can populate the session DB's tool_name
    column for FTS indexing.
    """
    for result in results:
        # Resolve tool_name from the earlier tool call mapping, and append
        # inside a single lock acquisition to avoid races with drain/clear.
        with _accum_lock:
            name = _call_id_to_name.pop(result.call_id, None)
            msg: Dict[str, Any] = {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": result.output_json,
            }
            if name:
                msg["tool_name"] = name
            _accum_messages.append(msg)


def drain_intermediate_messages() -> List[Dict[str, Any]]:
    """Return and clear all accumulated intermediate messages.

    Called by vagent_grpc after the stream completes to collect
    tool call + result messages for the conversation history.
    Also clears the call_id→name mapping (entries should have been
    consumed by _record_tool_result_messages, but clear any stragglers).
    """
    with _accum_lock:
        msgs = list(_accum_messages)
        _accum_messages.clear()
        _call_id_to_name.clear()
    return msgs


def reset_intermediate_messages() -> None:
    """Clear accumulated messages and call-id mapping (called at start of turn)."""
    with _accum_lock:
        _accum_messages.clear()
        _call_id_to_name.clear()


# ── gRPC server ────────────────────────────────────────────────────

class ToolExecutorService(agent_pb2_grpc.ToolExecutorServicer):
    def Execute(self, request: agent_pb2.ToolBatch, context: grpc.ServicerContext) -> agent_pb2.ToolResults:
        # Record the assistant tool_call message BEFORE execution
        _record_tool_call_messages(request)

        results = []

        for call in request.calls:
            tool_name = call.name
            call_id = call.call_id

            try:
                arguments = json.loads(call.arguments_json)
            except json.JSONDecodeError:
                results.append(agent_pb2.ToolResult(
                    call_id=call_id,
                    output_json=json.dumps({"error": "invalid JSON arguments"}),
                    is_error=True,
                ))
                continue

            with _dispatch_lock:
                dispatcher = _tool_dispatcher

            if dispatcher is None:
                results.append(agent_pb2.ToolResult(
                    call_id=call_id,
                    output_json=json.dumps({"error": "no tool dispatcher configured"}),
                    is_error=True,
                ))
                continue

            try:
                output = dispatcher(tool_name, arguments, task_id=None)
                results.append(agent_pb2.ToolResult(
                    call_id=call_id,
                    output_json=output,
                    is_error=False,
                ))
            except Exception as e:
                logger.exception("Tool %s failed", tool_name)
                results.append(agent_pb2.ToolResult(
                    call_id=call_id,
                    output_json=json.dumps({"error": str(e)}),
                    is_error=True,
                ))

        # Record the tool result messages AFTER execution
        _record_tool_result_messages(results)

        return agent_pb2.ToolResults(results=results)


def start_tool_executor_server(port: int = DEFAULT_PORT) -> int:
    """Start the tool executor gRPC server on the given port.

    Returns the actual port used (same as input if successful).
    Raises RuntimeError if the server fails to start.
    """
    # Reset message accumulator for a fresh turn
    reset_intermediate_messages()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    agent_pb2_grpc.add_ToolExecutorServicer_to_server(ToolExecutorService(), server)

    try:
        bound = server.add_insecure_port(f"0.0.0.0:{port}")
        if bound == 0:
            raise RuntimeError(f"Failed to bind to port {port}")
    except Exception:
        # Port in use — let OS pick
        bound = server.add_insecure_port("0.0.0.0:0")
        if bound == 0:
            raise RuntimeError("Failed to bind tool executor to any port")

    server.start()
    logger.info("vagent tool executor listening on port %d", bound)

    # Store for cleanup
    import atexit
    atexit.register(lambda: server.stop(0))

    return bound