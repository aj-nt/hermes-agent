"""TDD Gate: loop detection for file-mutating tool operations.

When an agent repeatedly edits the same Python source file without
verification, it is likely stuck in a tool-mediated edit loop (e.g.,
Unicode corruption, regex escaping, indentation misalignment).  The
TDD gate tracks per-file edit counts and injects a nudge message
requiring a failing test before further edits.

Scope: `.py` and `.pyi` files only — shipping code.
Exempt: config files, data files, prototyping notebooks, etc.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

# ── Configuration ──────────────────────────────────────────────────────

# File extensions covered by the TDD gate.
_TDD_GATED_EXTENSIONS = frozenset({".py", ".pyi"})

# Edit count thresholds.
_WARN_AFTER = 3          #Inject a soft warning after this many edits.
_REQUIRE_TEST_AFTER = 5  # Require a failing test before more edits.

# ── Normalisation ─────────────────────────────────────────────────────


def _normalise_path(path: str) -> str:
    """Resolve and lowercase a file path for deduplication."""
    try:
        return os.path.realpath(path).lower()
    except Exception:
        return path.lower()


def is_tdd_gated_path(path: str) -> bool:
    """True if *path* should be tracked by the TDD gate.

    Only Python source files (``.py``, ``.pyi``) are gated.  Config,
    data, documentation, and TEST files are exempt (editing a test
    IS verification, not the thing being verified).
    """
    _, ext = os.path.splitext(path)
    if ext.lower() not in _TDD_GATED_EXTENSIONS:
        return False
    # Test files are exempt: editing a test IS the verification step.
    normalized = path.replace(os.sep, "/").lower()
    basename = os.path.basename(normalized)
    if basename.startswith("test_") or basename.endswith("_test.py"):
        return False
    if "/tests/" in normalized or "/test/" in normalized:
        return False
    return True


# ── Nudge messages ────────────────────────────────────────────────────

_WARN_MSG = (
    "TDD GATE: You have edited {path} {n} times without verification. "
    "If this next edit doesn't resolve the issue, write a failing test "
    "first and then make the edit to pass it. Loops burn your iteration budget."
)

_REQUIRE_MSG = (
    "TDD GATE REQUIRES TEST-FIRST: You have edited {path} {n} times. "
    "Before editing this file again, you MUST:\n"
    "1. Write a failing test that exercises the exact behavior you want.\n"
    "2. Run the test to confirm it fails (RED).\n"
    "3. Then make the minimal edit to pass it (GREEN).\n"
    "This breaks tool-mediated edit loops. If you cannot write a test, "
    "use execute_code with direct Python I/O (open rb/wb) instead of "
    "write_file or patch."
)


def build_nudge(path: str, edit_count: int) -> Optional[str]:
    """Return a nudge message if the edit count crosses a threshold.

    Returns ``None`` if no nudge is needed (first or second edit).
    """
    if edit_count < _WARN_AFTER:
        return None
    if edit_count < _REQUIRE_TEST_AFTER:
        return _WARN_MSG.format(path=path, n=edit_count)
    return _REQUIRE_MSG.format(path=path, n=edit_count)


# ── Tracker ───────────────────────────────────────────────────────────


class TddGateTracker:
    """Per-session tracker for file edit counts.

    Thread-safety: this class is NOT thread-safe.  The caller must
    synchronise if used from multiple threads (the concurrent tool
    executor does this by collecting pre-flight data before spawning
    workers).
    """

    def __init__(self) -> None:
        # normalised_path → list of tool_call_ids
        self._edits: Dict[str, List[str]] = {}
        # normalised_path → nudge message already injected
        # (avoid injecting the same nudge twice)
        self._nudged: Dict[str, int] = {}

    def record(self, path: str, tool_call_id: str = "") -> Tuple[int, Optional[str]]:
        """Record an edit and return ``(edit_count, nudge_msg_or_None)``.

        *edit_count* is the total number of edits to *path* in this
        session (including the one just recorded).  *nudge_msg* is a
        non-None string when a nudge should be injected.
        """
        key = _normalise_path(path)
        self._edits.setdefault(key, []).append(tool_call_id)
        count = len(self._edits[key])

        # Only nudge if we haven't already nudged at this count.
        already_nudged_at = self._nudged.get(key, 0)
        nudge = build_nudge(path, count)
        if nudge is not None and count > already_nudged_at:
            self._nudged[key] = count
            return count, nudge
        return count, None

    def get_edit_count(self, path: str) -> int:
        """Return the number of edits recorded for *path*."""
        key = _normalise_path(path)
        return len(self._edits.get(key, []))

    def reset(self) -> None:
        """Clear all tracked edits."""
        self._edits.clear()
        self._nudged.clear()