"""Trajectory format conversion and saving utilities.

Extracted from run_agent.py to decouple trajectory handling from the AIAgent god-object.
Originally, _convert_to_trajectory_format was an AIAgent method that called
self._format_tools_for_system_message(). The extraction makes both functions pure:
  - format_tools_for_trajectory(tools) replaces self._format_tools_for_system_message()
  - convert_to_trajectory_format(messages, user_query, completed, tools_json) is the
    full conversion pipeline, now calling format_tools_for_trajectory internally.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to think tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening REASONING_SCRATCHPAD without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def format_tools_for_trajectory(tools: list) -> str:
    """Format tool definitions for the system message in trajectory format.

    This is the pure-function extraction of AIAgent._format_tools_for_system_message.
    Takes the tools list directly rather than accessing self.tools.

    Args:
        tools: List of tool dicts, each with a "function" key containing
               "name", "description", and "parameters".

    Returns:
        JSON string of formatted tool definitions.
    """
    if not tools:
        return "[]"

    formatted_tools = []
    for tool in tools:
        func = tool["function"]
        formatted_tool = {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
            "required": None,
        }
        formatted_tools.append(formatted_tool)

    return json.dumps(formatted_tools, ensure_ascii=False)


def convert_to_trajectory_format(
    messages: List[Dict[str, Any]],
    user_query: str,
    completed: bool,
    tools_json: str = "[]",
) -> List[Dict[str, Any]]:
    """Convert internal message format to trajectory format for saving.

    This is the pure-function extraction of AIAgent._convert_to_trajectory_format.
    The caller passes the pre-formatted tools_json string (from
    format_tools_for_trajectory) instead of calling self._format_tools_for_system_message().

    Args:
        messages: Internal message history.
        user_query: Original user query.
        completed: Whether the conversation completed successfully.
        tools_json: Pre-formatted JSON string of tool definitions.

    Returns:
        Messages in ShareGPT trajectory format.
    """
    trajectory = []

    # Add system message with tool definitions
    system_msg = (
        "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "the user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_results> XML tags. Here are the available tools:\n"
        f"<tools>\n{tools_json}\n</tools>\n"
        "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
        "{\'title\': \'FunctionCall\', \'type\': \'object\', \'properties\': {\'name\': {\'title\': \'Name\', \'type\': \'string\'}, "
        "\'arguments\': {\'title\': \'Arguments\', \'type\': \'object\'}}, \'required\': [\'name\', \'arguments\']}\n"
        "Each function call should be enclosed within <invoke> XML tags.\n"
        "Example:\n<invoke>\n{\'name\': <function-name>,\'arguments\': <args-dict>}\n</invoke>"
    )

    trajectory.append({
        "from": "system",
        "value": system_msg
    })

    trajectory.append({
        "from": "human",
        "value": user_query
    })

    i = 1

    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "assistant":
            if "tool_calls" in msg and msg["tool_calls"]:
                content = ""

                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f" Crescent\n{msg['reasoning']}\n Crescent\n"

                if msg.get("content") and msg["content"].strip():
                    content += convert_scratchpad_to_think(msg["content"]) + "\n"

                for tool_call in msg["tool_calls"]:
                    if not tool_call or not isinstance(tool_call, dict):
                        continue
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                    except json.JSONDecodeError:
                        logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                        arguments = {}

                    tool_call_json = {
                        "name": tool_call["function"]["name"],
                        "arguments": arguments
                    }
                    content += f"<invoke>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</invoke>\n"

                if " Crescent" not in content:
                    content = " Crescent\n Crescent\n" + content

                trajectory.append({
                    "from": "gpt",
                    "value": content.rstrip()
                })

                tool_responses = []
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    tool_msg = messages[j]
                    tool_response = "<tool_results>\n"

                    tool_content = tool_msg["content"]
                    try:
                        if tool_content.strip().startswith(("{", "[")):
                            tool_content = json.loads(tool_content)
                    except (json.JSONDecodeError, AttributeError):
                        pass

                    tool_index = len(tool_responses)
                    tool_name = (
                        msg["tool_calls"][tool_index]["function"]["name"]
                        if tool_index < len(msg["tool_calls"])
                        else "unknown"
                    )
                    tool_response += json.dumps({
                        "tool_call_id": tool_msg.get("tool_call_id", ""),
                        "name": tool_name,
                        "content": tool_content
                    }, ensure_ascii=False)
                    tool_response += "\n</tool_results>"
                    tool_responses.append(tool_response)
                    j += 1

                if tool_responses:
                    trajectory.append({
                        "from": "tool",
                        "value": "\n".join(tool_responses)
                    })
                    i = j - 1

            else:
                content = ""

                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f" Crescent\n{msg['reasoning']}\n Crescent\n"

                raw_content = msg["content"] or ""
                content += convert_scratchpad_to_think(raw_content)

                if " Crescent" not in content:
                    content = " Crescent\n Crescent\n" + content

                trajectory.append({
                    "from": "gpt",
                    "value": content.strip()
                })

        elif msg["role"] == "user":
            trajectory.append({
                "from": "human",
                "value": msg["content"]
            })

        i += 1

    return trajectory


def save_trajectory(trajectory: List[Dict[str, Any]], model: str,
                    completed: bool, filename: str = None):
    """Append a trajectory entry to a JSONL file.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to trajectory_samples.jsonl
                  or failed_trajectories.jsonl based on ``completed``.
    """
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"

    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
