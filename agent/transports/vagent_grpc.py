"""vagent gRPC transport — hermes-agent's client for the vagent agent loop.

Architecture:
  hermes (Python)                     vagent (Go)
      │                                   │
      │── 1. Start ToolExecutor server ───┤
      │── 2. Chat(msg, tools, addr) ─────>│  agent loop starts
      │                                   │  calls LLM
      │<── 3. stream: text_delta ─────────┤
      │                                   │  LLM returns tool calls
      │<── 4. stream: tool_batch ─────────┤  (observability)
      │<── 5. ToolExecutor.Execute() ─────┤  calls back for execution
      │── 6. return results ─────────────>│  continues loop
      │<── 7. stream: text_delta ─────────┤
      │<── 8. stream: done ───────────────┤  final response
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

import grpc

import agent_pb2
import agent_pb2_grpc

from agent.transports.vagent_tool_executor import (
    start_tool_executor_server,
    set_tool_dispatcher,
)

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────

DEFAULT_ADDRESS = "localhost:50052"
DEFAULT_TIMEOUT = 300  # seconds


# ── Public API ────────────────────────────────────────────────────

def run_vagent_turn(
    user_message: str,
    *,
    system_prompt: str = "",
    provider: str = "",
    model: str = "",
    api_key: str = "",
    base_url: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    max_iterations: int = 90,
    session_id: str = "",
    stream_callback: Optional[Callable[[str], None]] = None,
    handle_function_call: Optional[Callable] = None,
    address: str = DEFAULT_ADDRESS,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Run a conversation turn via the vagent gRPC backend.

    Args:
        user_message: The user's input.
        system_prompt: System prompt to inject.
        provider: LLM provider name (e.g., "custom:ollama").
        model: Model name (e.g., "glm-5.1:cloud").
        api_key: API key for the provider.
        base_url: Base URL for the provider.
        tools: List of tool schemas (OpenAI format).
        max_iterations: Max LLM-calling iterations.
        session_id: Unique session identifier.
        stream_callback: Called with each text delta during streaming.
        handle_function_call: Tool dispatch function (same as model_tools.handle_function_call).
            Signature: (tool_name, arguments, task_id) -> str (JSON output).
            If None, tools are passed to vagent but execution will fail.
        address: vagent gRPC server address (host:port).
        timeout: Max seconds for the entire turn.

    Returns:
        Dict with keys: final_response, messages, iterations, exit_reason.
        Same shape as conversation_loop.run_conversation().
    """
    # Start the tool executor server that vagent will call back to
    executor_port = start_tool_executor_server(0)  # 0 = OS picks port
    executor_address = f"localhost:{executor_port}"

    # Set up the tool dispatcher
    if handle_function_call is not None:
        set_tool_dispatcher(handle_function_call)

    channel = grpc.insecure_channel(address)
    stub = agent_pb2_grpc.AgentStub(channel)

    # Build the ChatRequest
    proto_tools = [_tool_to_proto(t) for t in (tools or [])]

    request = agent_pb2.ChatRequest(
        session_id=session_id,
        user_message=user_message,
        system_prompt=system_prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        tools=proto_tools,
        max_iterations=max_iterations,
        tool_executor_address=executor_address,
    )

    # Accumulate state
    final_response = ""
    exit_reason = "unknown"
    iterations = 0
    all_text = ""

    try:
        stream = stub.Chat(request, timeout=timeout)

        for event in stream:
            which = event.WhichOneof("event")
            if which is None:
                continue

            if which == "text_delta":
                delta = event.text_delta
                all_text += delta
                if stream_callback:
                    stream_callback(delta)

            elif which == "tool_batch":
                batch = event.tool_batch
                logger.info(
                    "vagent tool batch: %d calls — %s",
                    len(batch.calls),
                    [c.name for c in batch.calls],
                )
                # Results are handled by the ToolExecutor server callback —
                # vagent calls it directly. This event is for observability.

            elif which == "done":
                final_response = event.done.final_response
                exit_reason = event.done.exit_reason
                iterations = event.done.iterations
                break

            elif which == "error":
                raise RuntimeError(
                    f"vagent error: {event.error.code} — {event.error.message}"
                )

    except grpc.RpcError as e:
        code = e.code()
        details = e.details()
        logger.error("vagent gRPC error: %s — %s", code, details)
        raise RuntimeError(f"vagent gRPC error ({code}): {details}") from e
    finally:
        channel.close()

    return {
        "final_response": final_response or all_text,
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_response or all_text},
        ],
        "iterations": iterations,
        "exit_reason": exit_reason,
    }


# ── Helpers ───────────────────────────────────────────────────────

def _tool_to_proto(tool: Dict[str, Any]) -> agent_pb2.ToolDefinition:
    """Convert an OpenAI-format tool schema to the proto ToolDefinition."""
    func = tool.get("function", tool)
    return agent_pb2.ToolDefinition(
        name=func.get("name", ""),
        description=func.get("description", ""),
        parameters_json=json.dumps(func.get("parameters", {})),
    )
