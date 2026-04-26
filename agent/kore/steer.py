"""Steer injection utilities for appending /steer text to tool results.

Extracted from run_agent.py to decouple steer handling from the AIAgent god-object.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def apply_steer_to_tool_results(
    messages: list,
    num_tool_msgs: int,
    steer_text: Optional[str],
) -> bool:
    """Append /steer text to the last tool result in this turn.

    Called at the end of a tool-call batch, before the next API call.
    The steer is appended to the last ``role:"tool"`` message's content
    with a clear marker so the model understands it came from the user
    and NOT from the tool itself. Role alternation is preserved —
    nothing new is inserted, we only modify existing content.

    This is the pure-function extraction of AIAgent._apply_pending_steer_to_tool_results.
    The caller is responsible for draining pending_steer and passing it here.

    Args:
        messages: The running messages list (modified in place).
        num_tool_msgs: Number of tool results appended in this batch.
        steer_text: The steer text to inject. Empty/None means no-op.

    Returns:
        True if steer was applied, False if no-op (no tool messages,
        no steer text, or no tool-role message found in range).
    """
    if num_tool_msgs <= 0 or not messages:
        return False
    if not steer_text:
        return False

    # Find the last tool-role message in the recent tail.
    target_idx = None
    for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
        msg = messages[j]
        if isinstance(msg, dict) and msg.get("role") == "tool":
            target_idx = j
            break

    if target_idx is None:
        # No tool result in this batch (e.g. all skipped by interrupt).
        # The caller's fallback path should deliver it as a normal next-turn
        # user message; we cannot inject here.
        return False

    marker = f"\n\nUser guidance: {steer_text}"
    existing_content = messages[target_idx].get("content", "")
    if not isinstance(existing_content, str):
        # Anthropic multimodal content blocks — preserve them and append
        # a text block at the end.
        try:
            blocks = list(existing_content) if existing_content else []
            blocks.append({"type": "text", "text": marker.lstrip()})
            messages[target_idx]["content"] = blocks
        except Exception:
            # Fall back to string replacement if content shape is unexpected.
            messages[target_idx]["content"] = f"{existing_content}{marker}"
    else:
        messages[target_idx]["content"] = existing_content + marker

    logger.info(
        "Delivered /steer to agent after tool batch (%d chars): %s",
        len(steer_text),
        steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
    )
    return True