"""Checkpoint injection -- load checkpoint from parent session into system prompt.

When a new session starts (or a compression creates a continuation session),
this module finds the most recent checkpoint from the session's lineage
and injects it into the system prompt.

This is Layer 4 in the memory architecture. Only in_progress checkpoints
are injected -- completed ones are skipped.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SEPARATOR = "\u2550" * 46  # same as vault injection
CHAR_LIMIT = 4000


def build_checkpoint_system_prompt(store, parent_session_id: str = None) -> str:
    """Build the checkpoint injection block for the system prompt.

    Returns empty string if no applicable checkpoint exists.
    """
    if not store or not parent_session_id:
        return ""

    data = store.read(parent_session_id)

    # Walk lineage if direct parent has no checkpoint
    if data is None and store._dir.exists():
        all_sessions = store.list_sessions()
        for sid in reversed(all_sessions):
            attempt = store.read(sid)
            if attempt and attempt.get("status") == "in_progress":
                data = attempt
                break

    if not data or not isinstance(data, dict):
        return ""

    if data.get("status") == "completed":
        return ""

    lines = []
    lines.append(f"Task: {data.get('task', 'Unknown')}")
    lines.append(f"Status: {data.get('status', 'in_progress')}")
    lines.append("")

    progress = data.get("progress", [])
    if progress:
        lines.append("Progress:")
        for step in progress:
            status_icon = {
                "completed": "[x]", "in_progress": "[~]",
                "pending": "[ ]", "cancelled": "[-]",
            }.get(step.get("status", "pending"), "[ ]")
            line = f"  {status_icon} {step.get('step', '')}"
            if step.get("result"):
                line += f" -- {step['result']}"
            lines.append(line)
        lines.append("")

    state = data.get("state", {})
    if state:
        lines.append("State:")
        for key, value in state.items():
            if value and str(value) not in ("[]", ""):
                lines.append(f"  {key}: {value}")
        lines.append("")

    decisions = data.get("decisions", [])
    if decisions:
        lines.append("Decisions:")
        for d in decisions:
            lines.append(f"  - {d}")
        lines.append("")

    blocked = data.get("blocked", [])
    if blocked:
        lines.append(f"Blocked: {', '.join(blocked)}")

    unresolved = data.get("unresolved", [])
    if unresolved:
        lines.append(f"Unresolved: {', '.join(unresolved)}")

    content = "\n".join(lines)

    if len(content) > CHAR_LIMIT:
        content = content[:CHAR_LIMIT] + "\n[... truncated at char limit ...]"

    header = "CHECKPOINT: RESUME FROM HERE (saved before compaction, injected on session start)"
    return f"{SEPARATOR}\n{header}\n{SEPARATOR}\n{content}"