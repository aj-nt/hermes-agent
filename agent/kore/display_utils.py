"""Display and text formatting utilities.

Extracted from run_agent.py to decouple display logic from the AIAgent god-object.
"""

import json
import logging
import re
import shutil
import textwrap
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def summarize_background_review_actions(
    review_messages: List[Dict],
    prior_snapshot: List[Dict],
) -> List[str]:
    """Build the human-facing action summary for a background review pass.

    Walks the review agent's session messages and collects "successful tool
    action" descriptions to surface to the user (e.g. "Memory updated").
    Tool messages already present in ``prior_snapshot`` are skipped so we
    don't re-surface stale results from the prior conversation that the
    review agent inherited via ``conversation_history`` (issue #14944).

    Matching is by ``tool_call_id`` when available, with a content-equality
    fallback for tool messages that lack one.
    """
    existing_tool_call_ids = set()
    existing_tool_contents = set()
    for prior in prior_snapshot or []:
        if not isinstance(prior, dict) or prior.get("role") != "tool":
            continue
        tcid = prior.get("tool_call_id")
        if tcid:
            existing_tool_call_ids.add(tcid)
        else:
            content = prior.get("content")
            if isinstance(content, str):
                existing_tool_contents.add(content)

    actions: List[str] = []
    for msg in review_messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid and tcid in existing_tool_call_ids:
            continue
        if not tcid:
            content_str = msg.get("content")
            if isinstance(content_str, str) and content_str in existing_tool_contents:
                continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("success"):
            continue
        message = data.get("message", "")
        target = data.get("target", "")
        if "created" in message.lower():
            actions.append(message)
        elif "updated" in message.lower():
            actions.append(message)
        elif "added" in message.lower() or (target and "add" in message.lower()):
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")
        elif "Entry added" in message:
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")
        elif "removed" in message.lower() or "replaced" in message.lower():
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")
    return actions


def wrap_verbose(label: str, text: str, indent: str = "     ") -> str:
    """Word-wrap verbose tool output to fit the terminal width.

    Splits *text* on existing newlines and wraps each line individually,
    preserving intentional line breaks (e.g. pretty-printed JSON).
    Returns a ready-to-print string with *label* on the first line and
    continuation lines indented.
    """
    cols = shutil.get_terminal_size((120, 24)).columns
    wrap_width = max(40, cols - len(indent))
    out_lines: list[str] = []
    for raw_line in text.split("\n"):
        if len(raw_line) <= wrap_width:
            out_lines.append(raw_line)
        else:
            wrapped = textwrap.wrap(raw_line, width=wrap_width,
                               break_long_words=True,
                               break_on_hyphens=False)
            out_lines.extend(wrapped or [raw_line])
    body = ("\n" + indent).join(out_lines)
    return f"{indent}{label}{body}"


def normalize_interim_visible_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()
