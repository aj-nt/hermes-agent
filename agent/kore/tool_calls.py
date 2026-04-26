"""Tool call utility functions for ID generation, sanitization, and dedup.

Extracted from run_agent.py to decouple tool call handling from the AIAgent god-object.

All functions are pure (no class state, no side effects beyond logging).
"""

import copy
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.codex_responses_adapter import (
    _deterministic_call_id as _codex_deterministic_call_id,
    _split_responses_tool_id as _codex_split_responses_tool_id,
)



# ── Constants (moved from AIAgent class) ─────────────────────────────────

VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER = (
    "[hermes-agent: tool call arguments were corrupted in this session and "
    "have been dropped to keep the conversation alive. See issue #15236.]"
)

logger = logging.getLogger(__name__)

def get_tool_call_id_static(tc) -> str:
    """Extract call ID from a tool_call entry (dict or object)."""
    if isinstance(tc, dict):
        return tc.get("id", "") or ""
    return getattr(tc, "id", "") or ""


def sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs before every LLM call.

    Runs unconditionally — not gated on whether the context compressor
    is present — so orphans from session loading or manual message
    manipulation are always caught.
    """
    # --- Role allowlist: drop messages with roles the API won't accept ---
    filtered = []
    for msg in messages:
        role = msg.get("role")
        if role not in VALID_API_ROLES:
            logger.debug(
                "Pre-call sanitizer: dropping message with invalid role %r",
                role,
            )
            continue
        filtered.append(msg)
    messages = filtered

    surviving_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid = get_tool_call_id_static(tc)
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    # 1. Drop tool results with no matching assistant call
    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]
        logger.debug(
            "Pre-call sanitizer: removed %d orphaned tool result(s)",
            len(orphaned_results),
        )

    # 2. Inject stub results for calls whose result was dropped
    missing_results = surviving_call_ids - result_call_ids
    if missing_results:
        patched: List[Dict[str, Any]] = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = get_tool_call_id_static(tc)
                    if cid in missing_results:
                        patched.append({
                            "role": "tool",
                            "content": "[Result unavailable — see context summary above]",
                            "tool_call_id": cid,
                        })
        messages = patched
        logger.debug(
            "Pre-call sanitizer: added %d stub tool result(s)",
            len(missing_results),
        )
    return messages


def cap_delegate_task_calls(tool_calls: list) -> list:
    """Truncate excess delegate_task calls to max_concurrent_children.

    The delegate_tool caps the task list inside a single call, but the
    model can emit multiple separate delegate_task tool_calls in one
    turn.  This truncates the excess, preserving all non-delegate calls.

    Returns the original list if no truncation was needed.
    """
    from tools.delegate_tool import _get_max_concurrent_children
    max_children = _get_max_concurrent_children()
    delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
    if delegate_count <= max_children:
        return tool_calls
    kept_delegates = 0
    truncated = []
    for tc in tool_calls:
        if tc.function.name == "delegate_task":
            if kept_delegates < max_children:
                truncated.append(tc)
                kept_delegates += 1
        else:
            truncated.append(tc)
    logger.warning(
        "Truncated %d excess delegate_task call(s) to enforce "
        "max_concurrent_children=%d limit",
        delegate_count - max_children, max_children,
    )
    return truncated


def deduplicate_tool_calls(tool_calls: list) -> list:
    """Remove duplicate (tool_name, arguments) pairs within a single turn.

    Only the first occurrence of each unique pair is kept.
    Returns the original list if no duplicates were found.
    """
    seen: set = set()
    unique: list = []
    for tc in tool_calls:
        key = (tc.function.name, tc.function.arguments)
        if key not in seen:
            seen.add(key)
            unique.append(tc)
        else:
            logger.warning("Removed duplicate tool call: %s", tc.function.name)
    return unique if len(unique) < len(tool_calls) else tool_calls


def deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """Generate a deterministic call_id from tool call content.

    Used as a fallback when the API doesn't provide a call_id.
    Deterministic IDs prevent cache invalidation — random UUIDs would
    make every API call's prefix unique, breaking OpenAI's prompt cache.
    """
    return _codex_deterministic_call_id(fn_name, arguments, index)


def split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
    """Split a stored tool id into (call_id, response_item_id)."""
    return _codex_split_responses_tool_id(raw_id)


def sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
    """Strip Codex Responses API fields from tool_calls for strict providers.

    Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
    validate the Chat Completions schema and reject unknown fields (call_id,
    response_item_id) with 400 or 422 errors. These fields are preserved in
    the internal message history — this method only modifies the outgoing
    API copy.

    Creates new tool_call dicts rather than mutating in-place, so the
    original messages list retains call_id/response_item_id for Codex
    Responses API compatibility (e.g. if the session falls back to a
    Codex provider later).

    Fields stripped: call_id, response_item_id
    """
    tool_calls = api_msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return api_msg
    _STRIP_KEYS = {"call_id", "response_item_id"}
    api_msg["tool_calls"] = [
        {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
        if isinstance(tc, dict) else tc
        for tc in tool_calls
    ]
    return api_msg


def sanitize_tool_call_arguments(
    messages: list,
    *,
    logger=None,
    session_id: str = None,
) -> int:
    """Repair corrupted assistant tool-call argument JSON in-place."""
    log = logger or logging.getLogger(__name__)
    if not isinstance(messages, list):
        return 0

    repaired = 0
    marker = TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER

    def _prepend_marker(tool_msg: dict) -> None:
        existing = tool_msg.get("content")
        if isinstance(existing, str):
            if not existing:
                tool_msg["content"] = marker
            elif not existing.startswith(marker):
                tool_msg["content"] = f"{marker}\n{existing}"
            return
        if existing is None:
            tool_msg["content"] = marker
            return
        try:
            existing_text = json.dumps(existing)
        except TypeError:
            existing_text = str(existing)
        tool_msg["content"] = f"{marker}\n{existing_text}"

    message_index = 0
    while message_index < len(messages):
        msg = messages[message_index]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            message_index += 1
            continue

        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            message_index += 1
            continue

        insert_at = message_index + 1
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            arguments = function.get("arguments")
            if arguments is None or arguments == "":
                function["arguments"] = "{}"
                continue
            if isinstance(arguments, str) and not arguments.strip():
                function["arguments"] = "{}"
                continue
            if not isinstance(arguments, str):
                continue

            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                tool_call_id = tool_call.get("id")
                function_name = function.get("name", "?")
                preview = arguments[:80]
                log.warning(
                    "Corrupted tool_call arguments repaired before request "
                    "(session=%s, message_index=%s, tool_call_id=%s, function=%s, preview=%r)",
                    session_id or "-",
                    message_index,
                    tool_call_id or "-",
                    function_name,
                    preview,
                )
                function["arguments"] = "{}"

                existing_tool_msg = None
                scan_index = message_index + 1
                while scan_index < len(messages):
                    candidate = messages[scan_index]
                    if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                        break
                    if candidate.get("tool_call_id") == tool_call_id:
                        existing_tool_msg = candidate
                        break
                    scan_index += 1

                if existing_tool_msg is None:
                    messages.insert(
                        insert_at,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": marker,
                        },
                    )
                    insert_at += 1
                else:
                    _prepend_marker(existing_tool_msg)

                repaired += 1

        message_index += 1

    return repaired


def repair_tool_call(tool_name: str, valid_tool_names: set) -> str | None:
    """Attempt to repair a mismatched tool name before aborting.

    Models sometimes emit variants of a tool name that differ only
    in casing, separators, or class-like suffixes. Normalize
    aggressively before falling back to fuzzy match:

    1. Lowercase direct match.
    2. Lowercase + hyphens/spaces -> underscores.
    3. CamelCase -> snake_case (TodoTool -> todo_tool).
    4. Strip trailing ``_tool`` / ``-tool`` / ``tool`` suffix that
       Claude-style models sometimes tack on (TodoTool_tool ->
       TodoTool -> Todo -> todo). Applied twice so double-tacked
       suffixes like ``TodoTool_tool`` reduce all the way.
    5. Fuzzy match (difflib, cutoff=0.7).

    See #14784 for the original reports (TodoTool_tool, Patch_tool,
    BrowserClick_tool were all returning "Unknown tool" before).

    Returns the repaired name if found in valid_tool_names, else None.
    """
    import re as _re
    from difflib import get_close_matches

    if not tool_name:
        return None

    def _norm(s: str) -> str:
        return s.lower().replace("-", "_").replace(" ", "_")

    def _camel_snake(s: str) -> str:
        return _re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    def _strip_tool_suffix(s: str) -> str | None:
        lc = s.lower()
        for suffix in ("_tool", "-tool", "tool"):
            if lc.endswith(suffix):
                return s[: -len(suffix)].rstrip("_-")
        return None

    # Cheap fast-paths first
    lowered = tool_name.lower()
    if lowered in valid_tool_names:
        return lowered
    normalized = _norm(tool_name)
    if normalized in valid_tool_names:
        return normalized

    # Build the full candidate set for class-like emissions.
    cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
    # Strip trailing tool-suffix up to twice
    for _ in range(2):
        extra: set[str] = set()
        for c in cands:
            stripped = _strip_tool_suffix(c)
            if stripped:
                extra.add(stripped)
                extra.add(_norm(stripped))
                extra.add(_camel_snake(stripped))
        cands |= extra

    for c in cands:
        if c and c in valid_tool_names:
            return c

    # Fuzzy match as last resort.
    matches = get_close_matches(lowered, valid_tool_names, n=1, cutoff=0.7)
    if matches:
        return matches[0]

    return None
