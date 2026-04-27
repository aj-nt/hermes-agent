"""Tests for tool_scheduling module.

Tests for the extracted pure functions that govern tool parallelization
and destructive command detection.
"""

import os
from pathlib import Path

import pytest

from agent.kore.tool_scheduling import (
    NEVER_PARALLEL_TOOLS,
    PARALLEL_SAFE_TOOLS,
    PATH_SCOPED_TOOLS,
    MAX_TOOL_WORKERS,
    is_destructive_command,
    should_parallelize_tool_batch,
    extract_parallel_scope_path,
    paths_overlap,
    pool_may_recover_from_rate_limit,
)
from run_agent import (
    _is_destructive_command,
    _should_parallelize_tool_batch,
    _extract_parallel_scope_path,
    _paths_overlap,
    _pool_may_recover_from_rate_limit,
    _NEVER_PARALLEL_TOOLS,
    _PARALLEL_SAFE_TOOLS,
    _PATH_SCOPED_TOOLS,
    _MAX_TOOL_WORKERS,
)


# ============================================================================
# Backward compatibility: re-exports match kore originals
# ============================================================================

class TestToolSchedulingReExports:
    """Re-exported names from run_agent must match kore originals."""

    def test_never_parallel_tools_match(self):
        assert _NEVER_PARALLEL_TOOLS is NEVER_PARALLEL_TOOLS

    def test_parallel_safe_tools_match(self):
        assert _PARALLEL_SAFE_TOOLS is PARALLEL_SAFE_TOOLS

    def test_path_scoped_tools_match(self):
        assert _PATH_SCOPED_TOOLS is PATH_SCOPED_TOOLS

    def test_max_tool_workers_match(self):
        assert _MAX_TOOL_WORKERS == MAX_TOOL_WORKERS

    def test_is_destructive_command_match(self):
        for cmd in ["rm -rf /", "cp a b", "", "ls -la"]:
            assert _is_destructive_command(cmd) == is_destructive_command(cmd), cmd

    def test_should_parallelize_match(self):
        # Can't easily construct tool_call objects here; just verify the function exists
        assert callable(_should_parallelize_tool_batch)
        assert callable(should_parallelize_tool_batch)

    def test_paths_overlap_match(self):
        for left, right in [
            (Path("src/a.py"), Path("src/a.py")),
            (Path("src"), Path("src/sub/a.py")),
            (Path("src/a.py"), Path("src/b.py")),
        ]:
            assert _paths_overlap(left, right) == paths_overlap(left, right), f"{left} vs {right}"

    def test_pool_may_recover_match(self):
        assert _pool_may_recover_from_rate_limit(None) == pool_may_recover_from_rate_limit(None)
        assert _pool_may_recover_from_rate_limit(None) is False


# ============================================================================
# is_destructive_command
# ============================================================================

class TestIsDestructiveCommand:

    def test_empty_is_not_destructive(self):
        assert is_destructive_command("") is False

    def test_rm_is_destructive(self):
        assert is_destructive_command("rm -rf /tmp/test") is True

    def test_rmdir_is_destructive(self):
        assert is_destructive_command("rmdir old_dir") is True

    def test_cp_is_destructive(self):
        assert is_destructive_command("cp .env.local .env") is True

    def test_install_is_destructive(self):
        assert is_destructive_command("install template.env .env") is True

    def test_mv_is_destructive(self):
        assert is_destructive_command("mv old.txt new.txt") is True

    def test_sed_inplace_is_destructive(self):
        assert is_destructive_command("sed -i 's/old/new/g' file.txt") is True

    def test_ls_is_not_destructive(self):
        assert is_destructive_command("ls -la") is False

    def test_echo_is_not_destructive(self):
        assert is_destructive_command("echo hello") is False

    def test_redirect_overwrite_is_destructive(self):
        assert is_destructive_command("echo hello > file.txt") is True

    def test_redirect_append_is_not_destructive(self):
        assert is_destructive_command("echo hello >> file.txt") is False

    def test_git_reset_is_destructive(self):
        assert is_destructive_command("git reset --hard HEAD~1") is True

    def test_git_clean_is_destructive(self):
        assert is_destructive_command("git clean -fdx") is True

    def test_dd_is_destructive(self):
        assert is_destructive_command("dd if=/dev/zero of=/dev/sda") is True

    def test_truncate_is_destructive(self):
        assert is_destructive_command("truncate -s 0 logfile") is True

    def test_shred_is_destructive(self):
        assert is_destructive_command("shred secret.txt") is True

    def test_combined_with_and(self):
        assert is_destructive_command("ls && rm file") is True

    def test_combined_with_pipe(self):
        assert is_destructive_command("cat file | rm other") is True

    def test_combined_with_semicolon(self):
        assert is_destructive_command("echo done; rm file") is True


# ============================================================================
# paths_overlap
# ============================================================================

class TestPathsOverlap:

    def test_same_path(self):
        assert paths_overlap(Path("src/a.py"), Path("src/a.py")) is True

    def test_different_paths_same_dir(self):
        assert paths_overlap(Path("src/a.py"), Path("src/b.py")) is False

    def test_parent_child(self):
        assert paths_overlap(Path("src"), Path("src/sub/a.py")) is True

    def test_unrelated_paths(self):
        assert paths_overlap(Path("src/a.py"), Path("other/a.py")) is False

    def test_child_does_not_overlap_parent(self):
        # src/sub/a.py does NOT overlap src/a.py
        # (the child is more specific, not a prefix match)
        assert paths_overlap(Path("src/sub/a.py"), Path("src/a.py")) is False

    def test_empty_paths(self):
        assert paths_overlap(Path(""), Path("")) is False

    def test_one_empty(self):
        assert paths_overlap(Path(""), Path("src/a.py")) is False
        assert paths_overlap(Path("src/a.py"), Path("")) is False


# ============================================================================
# extract_parallel_scope_path
# ============================================================================

class TestExtractParallelScopePath:

    def test_path_scoped_tool_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = extract_parallel_scope_path("write_file", {"path": "notes.txt"})
        assert result is not None
        assert str(tmp_path) in str(result)

    def test_non_path_tool_returns_none(self):
        assert extract_parallel_scope_path("search_files", {"path": "/tmp"}) is None

    def test_absolute_path(self):
        result = extract_parallel_scope_path("read_file", {"path": "/tmp/test.txt"})
        assert result == Path("/tmp/test.txt")

    def test_tilde_expansion(self, monkeypatch):
        result = extract_parallel_scope_path("write_file", {"path": "~/notes.txt"})
        assert result is not None
        # Should expand ~ to home directory
        assert "~" not in str(result)

    def test_missing_path_returns_none(self):
        assert extract_parallel_scope_path("write_file", {}) is None

    def test_empty_path_returns_none(self):
        assert extract_parallel_scope_path("write_file", {"path": ""}) is None

    def test_relative_vs_absolute_same_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        abs_path = tmp_path / "notes.txt"
        abs_path.touch()
        rel = extract_parallel_scope_path("write_file", {"path": "notes.txt"})
        absolute = extract_parallel_scope_path("write_file", {"path": str(abs_path)})
        assert paths_overlap(rel, absolute)


# ============================================================================
# pool_may_recover_from_rate_limit
# ============================================================================

class TestPoolMayRecoverFromRateLimit:

    def test_none_pool(self):
        assert pool_may_recover_from_rate_limit(None) is False

    def _pool(self, n, has_available=True):
        """Minimal pool mock."""
        class Pool:
            def __init__(self, count, avail):
                self._count = count
                self._avail = avail
            def has_available(self):
                return self._avail
            def entries(self):
                return list(range(self._count))
        return Pool(n, has_available)

    def test_single_entry_cannot_recover(self):
        assert pool_may_recover_from_rate_limit(self._pool(1)) is False

    def test_single_entry_no_available(self):
        assert pool_may_recover_from_rate_limit(self._pool(1, has_available=False)) is False

    def test_two_entries_can_recover(self):
        assert pool_may_recover_from_rate_limit(self._pool(2)) is True

    def test_many_entries_no_available(self):
        assert pool_may_recover_from_rate_limit(self._pool(3, has_available=False)) is False

    def test_ten_entries_can_recover(self):
        assert pool_may_recover_from_rate_limit(self._pool(10)) is True


# ============================================================================
# Constants smoke tests
# ============================================================================

class TestConstants:
    def test_never_parallel_contains_clarify(self):
        assert "clarify" in NEVER_PARALLEL_TOOLS

    def test_parallel_safe_contains_read_file(self):
        assert "read_file" in PARALLEL_SAFE_TOOLS

    def test_path_scoped_contains_write_file(self):
        assert "write_file" in PATH_SCOPED_TOOLS

    def test_max_tool_workers_is_positive(self):
        assert MAX_TOOL_WORKERS > 0
        assert MAX_TOOL_WORKERS <= 32  # reasonable upper bound

    def test_path_scoped_is_subset_of_union(self):
        # PATH_SCOPED can overlap with PARALLEL_SAFE (read_file)
        # and with tools not in either (write_file, patch)
        assert "read_file" in (PARALLEL_SAFE_TOOLS | PATH_SCOPED_TOOLS)
        assert "write_file" in PATH_SCOPED_TOOLS
        assert "patch" in PATH_SCOPED_TOOLS
