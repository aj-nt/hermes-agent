"""vagent tool executor — standalone gRPC server that receives tool calls from
vagant and dispatches them using hermes-agent's handle_function_call.

Launched as a sidecar process by vagent_grpc.py's run_vagent_turn().
Vagent calls back to this server to execute tools during the agent loop.
"""

from __future__ import annotations

import json
import logging
import signal
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


# ── gRPC server ────────────────────────────────────────────────────

class ToolExecutorService(agent_pb2_grpc.ToolExecutorServicer):
    def Execute(self, request: agent_pb2.ToolBatch, context: grpc.ServicerContext) -> agent_pb2.ToolResults:
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

        return agent_pb2.ToolResults(results=results)


def start_tool_executor_server(port: int = DEFAULT_PORT) -> int:
    """Start the tool executor gRPC server on the given port.

    Returns the actual port used (same as input if successful).
    Raises RuntimeError if the server fails to start.
    """
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
