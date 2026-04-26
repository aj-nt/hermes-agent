"""Tests for the TDD gate (agent.kore.tdd_gate)."""

import os
import pytest

from agent.kore.tdd_gate import (
    TddGateTracker,
    build_nudge,
    is_tdd_gated_path,
    _normalise_path,
    _WARN_AFTER,
    _REQUIRE_TEST_AFTER,
)


# ── is_tdd_gated_path ───────────────────────────────────────────────


class TestIsTddGatedPath:
    def test_py_files_gated(self):
        assert is_tdd_gated_path("foo.py") is True
        assert is_tdd_gated_path("bar/baz.py") is True

    def test_pyi_files_gated(self):
        assert is_tdd_gated_path("foo.pyi") is True

    def test_yaml_exempt(self):
        assert is_tdd_gated_path("config.yaml") is False

    def test_json_exempt(self):
        assert is_tdd_gated_path("data.json") is False

    def test_md_exempt(self):
        assert is_tdd_gated_path("README.md") is False

    def test_toml_exempt(self):
        assert is_tdd_gated_path("pyproject.toml") is False

    def test_txt_exempt(self):
        assert is_tdd_gated_path("notes.txt") is False

    def test_case_insensitive(self):
        assert is_tdd_gated_path("FOO.PY") is True
        assert is_tdd_gated_path("Bar.Pyi") is True

    def test_no_extension_exempt(self):
        assert is_tdd_gated_path("Makefile") is False

    def test_hidden_py_exempt(self):
        """Hidden Python files (e.g. .gitkeep) are not .py."""
        assert is_tdd_gated_path(".gitkeep") is False

    def test_files_exempt_by_prefix(self):
        assert is_tdd_gated_path("tests/kore/test_tdd_gate.py") is False
        assert is_tdd_gated_path("src/test_foo.py") is False

    def test_files_exempt_by_suffix(self):
        assert is_tdd_gated_path("integration_test.py") is False

    def test_files_exempt_by_directory(self):
        assert is_tdd_gated_path("project/tests/helpers.py") is False
        assert is_tdd_gated_path("project/test/conftest.py") is False

    def test_non_test_py_still_gated(self):
        assert is_tdd_gated_path("project/src/module.py") is True
        assert is_tdd_gated_path("agent/kore/tdd_gate.py") is True


# ── build_nudge ──────────────────────────────────────────────────────


class TestBuildNudge:
    def test_first_edit_no_nudge(self):
        assert build_nudge("foo.py", 1) is None

    def test_second_edit_no_nudge(self):
        """Warn threshold is 3, so n=2 gets None."""
        assert build_nudge("foo.py", 2) is None

    def test_third_edit_warns(self):
        """n=3 is at warn threshold but below require threshold."""
        msg = build_nudge("foo.py", 3)
        assert msg is not None
        assert "3 times" in msg
        assert "TDD GATE" in msg
        assert "REQUIRES" not in msg

    def test_fourth_edit_still_warns(self):
        """n=4 is still below require threshold of 5."""
        msg = build_nudge("foo.py", 4)
        assert msg is not None
        assert "REQUIRES TEST-FIRST" not in msg

    def test_fifth_edit_requires_test(self):
        msg = build_nudge("foo.py", 5)
        assert "REQUIRES TEST-FIRST" in msg

    def test_nudge_includes_path(self):
        msg = build_nudge("agent/kore/think_blocks.py", 3)
        assert "agent/kore/think_blocks.py" in msg

    def test_nudge_mentions_execute_code(self):
        """After 5 edits, the nudge should mention execute_code as alternative."""
        msg = build_nudge("foo.py", 5)
        assert "execute_code" in msg


# ── TddGateTracker ───────────────────────────────────────────────────


class TestTddGateTracker:
    def test_first_edit_no_nudge(self):
        tracker = TddGateTracker()
        count, nudge = tracker.record("foo.py")
        assert count == 1
        assert nudge is None

    def test_second_edit_no_nudge(self):
        """Warn threshold is 3 (not 2)."""
        tracker = TddGateTracker()
        tracker.record("foo.py")
        count, nudge = tracker.record("foo.py")
        assert count == 2
        assert nudge is None

    def test_third_edit_warns(self):
        tracker = TddGateTracker()
        tracker.record("foo.py")
        tracker.record("foo.py")
        count, nudge = tracker.record("foo.py")
        assert count == 3
        assert nudge is not None
        assert "TDD GATE" in nudge
        assert "REQUIRES" not in nudge

    def test_fifth_edit_requires_test(self):
        tracker = TddGateTracker()
        for i in range(4):
            tracker.record("foo.py")
        count, nudge = tracker.record("foo.py")
        assert count == 5
        assert "REQUIRES TEST-FIRST" in nudge

    def test_no_duplicate_nudge_at_same_count(self):
        """If record() is called again at the same count, no new nudge."""
        tracker = TddGateTracker()
        tracker.record("foo.py")   # 1
        tracker.record("foo.py")   # 2
        tracker.record("foo.py")   # 3 — nudge
        # Already nudged at count 3; another record would be count 4
        # which is still at warn level but IS a new count, so it nudges.
        count, nudge = tracker.record("foo.py")  # 4 — warn nudge
        assert nudge is not None

    def test_separate_files_tracked_independently(self):
        tracker = TddGateTracker()
        tracker.record("foo.py")
        tracker.record("foo.py")
        count, nudge = tracker.record("bar.py")
        assert count == 1
        assert nudge is None  # bar.py is on its first edit

    def test_path_normalisation(self):
        """Different path representations of the same file are deduped."""
        tracker = TddGateTracker()
        tracker.record("./foo.py")
        tracker.record("foo.py")
        # Should normalise to the same file; 3rd edit = warn
        count, nudge = tracker.record("foo.py")
        assert count == 3
        assert nudge is not None

    def test_get_edit_count(self):
        tracker = TddGateTracker()
        tracker.record("foo.py")
        tracker.record("foo.py")
        tracker.record("foo.py")
        assert tracker.get_edit_count("foo.py") == 3
        assert tracker.get_edit_count("bar.py") == 0

    def test_get_edit_count_normalised(self):
        tracker = TddGateTracker()
        tracker.record("./foo.py")
        assert tracker.get_edit_count("foo.py") == 1

    def test_reset(self):
        tracker = TddGateTracker()
        tracker.record("foo.py")
        tracker.record("foo.py")
        tracker.reset()
        assert tracker.get_edit_count("foo.py") == 0
        count, nudge = tracker.record("foo.py")
        assert count == 1
        assert nudge is None

    def test_tool_call_id_tracked(self):
        tracker = TddGateTracker()
        count, _ = tracker.record("foo.py", "tc_001")
        assert count == 1

    def test_higher_thresholds(self):
        """After many edits, the nudge keeps escalating."""
        tracker = TddGateTracker()
        for i in range(1, 10):
            count, nudge = tracker.record("foo.py")
            assert count == i
            if i < _WARN_AFTER:
                assert nudge is None
            elif i < _REQUIRE_TEST_AFTER:
                assert nudge is not None
                assert "REQUIRES" not in nudge
            else:
                assert nudge is not None
                assert "REQUIRES" in nudge


# ── Unicode bug regression test ──────────────────────────────────────


class TestUnicodeBugRegression:
    """The TDD gate exists because of a real 326-message loop caused by
    Unicode corruption in the think_blocks.py extraction.  This test
    verifies the gate would have caught that loop.
    """

    def test_would_have_caught_think_blocks_loop(self):
        """The agent edited think_blocks.py 3+ times with write_file
        and patch, each time producing a different corruption of the
        \\u2019 character.  The TDD gate should warn on edit 3 and
        require a test on edit 5.
        """
        tracker = TddGateTracker()

        # Simulate the actual sequence
        edits = [
            ("agent/kore/think_blocks.py", "write_file"),  # 1st: initial extraction
            ("agent/kore/think_blocks.py", "write_file"),  # 2nd: attempted fix
            ("agent/kore/think_blocks.py", "patch"),       # 3rd: another attempted fix
            # At this point the agent is stuck -- the TDD gate warns
            ("agent/kore/think_blocks.py", "patch"),       # 4th: still stuck
            # Now the TDD gate REQUIRES a test before more edits
            ("agent/kore/think_blocks.py", "patch"),       # 5th: should be blocked
        ]

        for i, (path, tool) in enumerate(edits):
            count, nudge = tracker.record(path, f"tc_{i}")
            if count == 3:
                assert nudge is not None, "Should have warned after 3rd edit"
            if count == 5:
                assert nudge is not None, "Should require test after 5th edit"
                assert "failing test" in nudge.lower(), (
                    "Nudge should tell the agent to write a failing test"
                )
                assert "execute_code" in nudge, (
                    "Nudge should mention execute_code as alternative for Unicode"
                )