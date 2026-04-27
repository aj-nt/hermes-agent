"""Tool scheduling and safety predicates for parallel/dispatch decisions.

Pure functions and constants that determine whether tool calls can run
concurrently, and whether terminal commands are destructive.

These were module-level helpers in run_agent.py; extracted as part of the
Kore refactor so they can be tested and reused independently.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

# Tools that must never run concurrently (interactive / user-facing).
# When any of these appear in a batch, we fall back to sequential execution.
NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# Read-only tools with no shared mutable session state.
PARALLEL_SAFE_TOOLS = frozenset({
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
})

# File tools can run concurrently when they target independent paths.
PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

# Maximum number of concurrent worker threads for parallel tool execution.
MAX_TOOL_WORKERS = 8

# Patterns that indicate a terminal command may modify/delete files.
DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        cp\s|install\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
# Output redirects that overwrite files (> but not >>)
REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def is_destructive_command(cmd: str) -> bool:
    """Heuristic: does this terminal command look like it modifies/deletes files?"""
    if not cmd:
        return False
    if DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


def should_parallelize_tool_batch(tool_calls) -> bool:
    """Return True when a tool-call batch is safe to run concurrently."""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in PATH_SCOPED_TOOLS:
            scoped_path = extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in PARALLEL_SAFE_TOOLS:
            return False

    return True


def extract_parallel_scope_path(tool_name: str, function_args: dict) -> Optional[Path]:
    """Return the normalized file target for path-scoped tools."""
    if tool_name not in PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # Avoid resolve(); the file may not exist yet.
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def paths_overlap(left: Path, right: Path) -> bool:
    """Return True when two paths may refer to the same subtree."""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        # Empty paths shouldn't reach here (guarded upstream), but be safe.
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]


def pool_may_recover_from_rate_limit(pool) -> bool:
    """Decide whether to wait for credential-pool rotation instead of falling back.

    The existing pool-rotation path requires the pool to (1) exist and (2) have
    at least one entry not currently in exhaustion cooldown.  But rotation is
    only meaningful when the pool has more than one entry.

    With a single-credential pool (common for Gemini OAuth, Vertex service
    accounts, and any "one personal key" configuration), the primary entry
    just 429'd and there is nothing to rotate to.  Waiting for the pool
    cooldown to expire means retrying against the same exhausted quota — the
    daily-quota 429 will recur immediately, and the retry budget is burned.

    In that case we must fall back to the configured ``fallback_model``
    instead.  Returns True only when rotation has somewhere to go.

    See issue #11314.
    """
    if pool is None:
        return False
    if not pool.has_available():
        return False
    return len(pool.entries()) > 1