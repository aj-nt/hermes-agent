"""vagent gRPC transport — hermes-agent's client for the vagent agent loop.

Architecture:
  hermes (Python)                     vagent (Go)
      |                                   |
      |-- 1. Start ToolExecutor server ---|
      |-- 2. Chat(msg, tools, addr) ---->|  agent loop starts
      |                                   |  calls LLM
      |<-- 3. stream: text_delta ---------|  (observability)
      |<-- 4. stream: tool_batch ---------|  (observability)
      |<-- 5. ToolExecutor.Execute() -----|  calls back for execution
      |                                   |    executor records call + result
      |-- 6. return results ------------>|  continues loop
      |<-- 7. stream: text_delta ---------|
      |<-- 8. stream: done --------------|  final response

Intermediate message capture:
  The ToolExecutor records every tool call and result as OpenAI-format
  message dicts in a thread-safe accumulator.  After the gRPC stream
  completes, run_vagent_turn drains the accumulator to build a full
  conversation history — including all assistant tool_calls and tool
  result messages — for session persistence in state.db.
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
    drain_intermediate_messages,
    reset_intermediate_messages,
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
    conversation_history: Optional[List[Dict[str, Any]]] = None,
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
        conversation_history: Prior messages (role/content dicts) to prepend.
        address: vagent gRPC server address (host:port).
        timeout: Max seconds for the entire turn.

    Returns:
        Dict with keys: final_response, messages, iterations, exit_reason,
        completed, failed, api_calls.  The messages list includes all
        intermediate tool-call and tool-result messages, not just the
        user request and final assistant response.
    """
    # Start the tool executor server that vagent will call back to
    executor_port = start_tool_executor_server(0)  # 0 = OS picks port
    executor_address = f"localhost:{executor_port}"

    # Set up the tool dispatcher
    if handle_function_call is not None:
        set_tool_dispatcher(handle_function_call)

    # Reset the intermediate message accumulator for this turn
    reset_intermediate_messages()

    channel = grpc.insecure_channel(address)
    stub = agent_pb2_grpc.AgentStub(channel)

    # Build the ChatRequest
    proto_tools = [_tool_to_proto(t) for t in (tools or [])]

    # Build proto messages from conversation history
    proto_msgs = []
    if conversation_history:
        for m in conversation_history:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                # content can be a list of content blocks (vision etc) —
                # extract text parts only for the proto
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = " ".join(text_parts)
            proto_msgs.append(agent_pb2.Message(
                role=str(role),
                content=str(content),
                tool_call_id=str(m.get("tool_call_id", "")),
                name=str(m.get("name", "")),
            ))

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
        messages=proto_msgs,
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

            elif which == "thinking_delta":
                # Reasoning/CoT chunks — currently not persisted
                pass

            elif which == "tool_batch":
                batch = event.tool_batch
                logger.info(
                    "vagent tool batch: %d calls — %s",
                    len(batch.calls),
                    [c.name for c in batch.calls],
                )
                # The tool_batch stream event is for observability only.
                # Intermediate messages (assistant tool_calls + tool results)
                # are recorded by the ToolExecutor during Execute() and
                # drained after the stream completes (see drain below).

            elif which == "done":
                final_response = event.done.final_response
                exit_reason = event.done.exit_reason
                iterations = event.done.iterations
                break

            elif which == "error":
                raise RuntimeError(
                    f"vagent error: {event.error.code} — {event.error.message}"
                )

        # After the stream completes, drain all intermediate messages that
        # the ToolExecutor accumulated during the turn.  This gives us the
        # full chain: assistant (tool_calls) → tool (result) for each batch,
        # in correct execution order.
        intermediate_messages = drain_intermediate_messages()

    except grpc.RpcError as e:
        code = e.code()
        details = e.details()
        logger.error("vagent gRPC error: %s — %s", code, details)
        raise RuntimeError(f"vagent gRPC error ({code}): {details}") from e
    finally:
        channel.close()

    # Derive completed/failed from exit_reason for contract parity with
    # the Python agent loop's return shape (callers check these keys).
    completed = exit_reason == "completed"
    failed = exit_reason in ("error", "unknown")

    # Build the full messages list: conversation_history + user + intermediates + final assistant
    messages = list(conversation_history) if conversation_history else []
    messages.append({"role": "user", "content": user_message})
    messages.extend(intermediate_messages)
    messages.append({"role": "assistant", "content": final_response or all_text})

    return {
        "final_response": final_response or all_text,
        "messages": messages,
        "iterations": iterations,
        "exit_reason": exit_reason,
        "completed": completed,
        "failed": failed,
        "api_calls": iterations,  # vagent iterations ≈ API calls
    }


# ── Private helpers ───────────────────────────────────────────────

def _tool_to_proto(tool: Dict[str, Any]) -> agent_pb2.ToolDefinition:
    """Convert an OpenAI-format tool schema to the proto ToolDefinition."""
    func = tool.get("function", tool)
    return agent_pb2.ToolDefinition(
        name=func.get("name", ""),
        description=func.get("description", ""),
        parameters_json=json.dumps(func.get("parameters", {})),
    )