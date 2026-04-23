"""Checkpoint tool -- save and restore mid-task state across context compaction.

The checkpoint tool lets the agent save its current task progress, state, and
decisions to disk so it can resume after context compaction or session restart.
This is Layer 4 in the memory architecture (after Layer 1 memory, Layer 2
personality, Layer 3 vault).

Checkpoint files are stored as YAML in ~/.hermes/checkpoints/<session_id>.yaml.
"""

import json
import logging
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_TOOL_SCHEMA = {
    "name": "checkpoint",
    "description": (
        "Save or restore mid-task state (checkpoint). "
        "Write a checkpoint before risky operations or when you have made significant progress. "
        "The system also auto-writes a checkpoint before context compaction. "
        "On session start, any existing checkpoint from the parent session is auto-injected "
        "so you can resume where you left off. "
        "Use 'write' to create/overwrite, 'update' to merge into existing, "
        "'read' to check current checkpoint, 'clear' to delete."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "update", "read", "clear"],
                "description": (
                    "write: Create or overwrite checkpoint for current session. "
                    "update: Merge progress/state/decisions into existing checkpoint. "
                    "read: Read current checkpoint. "
                    "clear: Delete current checkpoint (task done or abandoned)."
                ),
            },
            "task": {
                "type": "string",
                "description": "One-line description of what you are working on. Required for 'write'.",
            },
            "progress": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string", "description": "What this step does"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                        },
                        "result": {
                            "type": "string",
                            "description": "Outcome (optional, for completed steps)",
                        },
                    },
                    "required": ["step", "status"],
                },
                "description": "Ordered list of task steps with status. Each step: {step, status, result?}.",
            },
            "state": {
                "type": "object",
                "properties": {
                    "active_branch": {"type": "string"},
                    "files_changed": {"type": "array", "items": {"type": "string"}},
                    "tests_status": {"type": "string"},
                    "last_commit": {"type": "string"},
                    "pushed": {"type": "boolean"},
                    "working_directory": {"type": "string"},
                },
                "description": "Machine-readable facts about the current state: branch, files, test results, commits.",
            },
            "decisions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Non-obvious choices made during this task (the 'why', not the 'what').",
            },
            "blocked": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Things blocked on external input. Empty if unblocked.",
            },
            "unresolved": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Open questions or unknowns. Empty if none.",
            },
        },
        "required": ["action"],
    },
}


def _git_state(workdir: str = None) -> Dict[str, Any]:
    """Best-effort capture of git state from the working directory."""
    state = {}
    if not workdir:
        return state
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=workdir,
        )
        if branch.returncode == 0:
            state["active_branch"] = branch.stdout.strip()

        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=workdir,
        )
        if commit.returncode == 0:
            state["last_commit"] = commit.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return state


def checkpoint_tool(
    action: str,
    task: str = None,
    progress: List[Dict] = None,
    state: Dict[str, Any] = None,
    decisions: List[str] = None,
    blocked: List[str] = None,
    unresolved: List[str] = None,
    store=None,
    agent=None,
) -> str:
    """Execute a checkpoint action. Returns JSON result string."""
    if store is None:
        from agent.checkpoint_store import CheckpointStore
        store = CheckpointStore()

    session_id = getattr(agent, "session_id", "unknown") if agent else "unknown"

    if action == "write":
        if not task:
            return json.dumps({"success": False, "error": "'task' is required for write action"})

        # Auto-populate git state if not provided
        effective_state = dict(state or {})
        workdir = effective_state.get("working_directory")
        if workdir and "active_branch" not in effective_state:
            git = _git_state(workdir)
            effective_state.update(git)
        # Auto-populate from todo store if available
        todo_snapshot = None
        if agent and hasattr(agent, "_todo_store") and agent._todo_store:
            try:
                todo_snapshot = agent._todo_store.format_for_injection()
            except Exception:
                pass
        if todo_snapshot:
            effective_state["todo_snapshot"] = todo_snapshot

        data = {
            "session_id": session_id,
            "task": task,
            "status": "in_progress",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "progress": progress or [],
            "state": effective_state,
            "decisions": decisions or [],
            "blocked": blocked or [],
            "unresolved": unresolved or [],
        }
        store.write(session_id, data)
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "message": f"Checkpoint saved for session {session_id}",
        })

    elif action == "update":
        existing = store.read(session_id)
        if not existing:
            return json.dumps({
                "success": False,
                "error": f"No checkpoint exists for session {session_id}. Use 'write' first.",
            })

        # Merge progress (append new steps)
        if progress:
            existing["progress"] = existing.get("progress", []) + progress

        # Merge state (overwrite keys)
        if state:
            existing["state"] = {**existing.get("state", {}), **state}

        # Append decisions
        if decisions:
            existing["decisions"] = existing.get("decisions", []) + decisions

        # Replace blocked/unresolved
        if blocked is not None:
            existing["blocked"] = blocked
        if unresolved is not None:
            existing["unresolved"] = unresolved

        # Update git state if workdir provided
        workdir = (state or {}).get("working_directory") if state else None
        if workdir:
            git = _git_state(workdir)
            existing["state"] = {**existing.get("state", {}), **git}

        store.write(session_id, existing)
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "message": f"Checkpoint updated for session {session_id}",
        })

    elif action == "read":
        data = store.read(session_id)
        if not data:
            return json.dumps({
                "success": False,
                "error": f"No checkpoint exists for session {session_id}",
            })
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "checkpoint": data,
        })

    elif action == "clear":
        store.delete(session_id)
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "message": f"Checkpoint cleared for session {session_id}",
        })

    else:
        return json.dumps({"success": False, "error": f"Unknown action: {action}"})


# --- Registry ---
from tools.registry import registry

registry.register(
    name="checkpoint",
    toolset="todo",
    schema=CHECKPOINT_TOOL_SCHEMA,
    handler=lambda args, **kw: checkpoint_tool(
        action=args.get("action"),
        task=args.get("task"),
        progress=args.get("progress"),
        state=args.get("state"),
        decisions=args.get("decisions"),
        blocked=args.get("blocked"),
        unresolved=args.get("unresolved"),
        store=kw.get("store"),
        agent=kw.get("agent"),
    ),
    check_fn=lambda: True,  # always available
    emoji="🔖",
)