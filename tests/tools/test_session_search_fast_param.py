"""Tests for the 'fast' parameter being passed through to session_search.

BUG: Both dispatch call sites in run_agent.py (CLI and gateway) omitted
the `fast` parameter, causing fast=False calls to always use the fast
(term-index) path and never reach FTS5 + LLM summarization.

These tests verify:
1. session_search(fast=False) routes to _full_search (FTS5+LLM path)
2. session_search(fast=True) routes to _fast_search (term index path)
3. The registry handler passes fast through
4. The dispatch sites in run_agent.py pass fast through
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call


class TestFastParamRouter:
    """Verify that the `fast` parameter routes to the correct internal function."""

    def test_fast_true_calls_fast_search(self):
        """fast=True should call _fast_search, not _full_search."""
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db._contains_cjk.return_value = False
        mock_db.search_by_terms.return_value = [
            {"session_id": "s1", "source": "cli", "session_started": 1709500000,
             "model": "test", "title": "test session", "match_count": 5},
        ]
        mock_db.get_session.return_value = {"parent_session_id": None}
        mock_db.get_child_session_ids.return_value = []

        with patch("tools.session_search_tool._fast_search") as mock_fast, \
             patch("tools.session_search_tool._full_search") as mock_full:
            mock_fast.return_value = json.dumps({"success": True, "mode": "fast"})
            session_search(query="docker", db=mock_db, fast=True)
            mock_fast.assert_called_once()
            mock_full.assert_not_called()

    def test_fast_false_calls_full_search(self):
        """fast=False should call _full_search, not _fast_search."""
        from tools.session_search_tool import session_search

        mock_db = MagicMock()

        with patch("tools.session_search_tool._full_search") as mock_full, \
             patch("tools.session_search_tool._fast_search") as mock_fast:
            mock_full.return_value = json.dumps({"success": True, "mode": "full"})
            session_search(query="docker", db=mock_db, fast=False)
            mock_full.assert_called_once()
            mock_fast.assert_not_called()

    def test_default_is_fast_true(self):
        """Omitting fast should default to True (fast path)."""
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db._contains_cjk.return_value = False
        mock_db.search_by_terms.return_value = []

        with patch("tools.session_search_tool._fast_search") as mock_fast, \
             patch("tools.session_search_tool._full_search") as mock_full:
            mock_fast.return_value = json.dumps({"success": True, "mode": "fast", "results": []})
            session_search(query="docker", db=mock_db)
            mock_fast.assert_called_once()
            mock_full.assert_not_called()

    def test_cjk_forces_full_search(self):
        """CJK queries should fall through to _full_search even with fast=True."""
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db._contains_cjk.return_value = True

        with patch("tools.session_search_tool._full_search") as mock_full, \
             patch("tools.session_search_tool._fast_search") as mock_fast:
            mock_full.return_value = json.dumps({"success": True, "mode": "full", "results": []})
            session_search(query="docker", db=mock_db, fast=True)
            # CJK should force fast=False internally
            mock_full.assert_called_once()
            mock_fast.assert_not_called()


class TestRegistryHandlerPassesFast:
    """Verify the registry handler lambda forwards the 'fast' parameter."""

    def test_handler_includes_fast_kwarg(self):
        """The registry handler should pass fast=args.get('fast', True)."""
        from tools.registry import registry
        # Trigger module import so session_search registers itself
        import tools.session_search_tool  # noqa: F401

        entry = registry._tools.get("session_search")
        assert entry is not None, "session_search not found in tool registry"

        handler = entry.handler
        import inspect
        source = inspect.getsource(handler)
        assert "fast" in source, (
            f"Registry handler for session_search does not reference 'fast'. "
            f"Source:\n{source}"
        )


class TestRunAgentDispatchPassesFast:
    """Verify that both dispatch call sites in run_agent.py pass fast= through."""

    def test_cli_dispatch_passes_fast_param(self):
        """CLI dispatch at ~line 6957 should include fast=function_args.get('fast', True)."""
        import run_agent
        import inspect
        source = inspect.getsource(run_agent.AIAgent)
        lines = source.split('\n')

        # Find all session_search dispatch blocks
        found_fast_in_dispatch = False
        for i, line in enumerate(lines):
            if 'function_name == "session_search"' in line:
                # Check the next ~12 lines for _session_search call
                block = '\n'.join(lines[i:i + 12])
                if "_session_search(" in block:
                    if "fast=" in block:
                        found_fast_in_dispatch = True
                    # If we found a dispatch block without fast=, fail immediately
                    else:
                        pytest.fail(
                            f"session_search dispatch block at source line ~{i} "
                            f"missing 'fast' parameter.\nBlock:\n{block}"
                        )

        assert found_fast_in_dispatch, (
            "No session_search dispatch block with _session_search call found in AIAgent source"
        )

    def test_gateway_dispatch_passes_fast_param(self):
        """Gateway dispatch at ~line 7465 should also include fast=."""
        import run_agent
        import inspect
        source = inspect.getsource(run_agent.AIAgent)
        lines = source.split('\n')

        # There should be exactly 2 session_search dispatch blocks
        dispatch_blocks = []
        for i, line in enumerate(lines):
            if 'function_name == "session_search"' in line:
                block = '\n'.join(lines[i:i + 12])
                dispatch_blocks.append((i, block))

        # Both blocks must have fast=
        blocks_with_fast = 0
        for line_num, block in dispatch_blocks:
            if "_session_search(" in block:
                assert "fast=" in block, (
                    f"session_search dispatch at source line ~{line_num} "
                    f"missing 'fast' parameter.\nBlock:\n{block}"
                )
                blocks_with_fast += 1

        # Should find at least 2 dispatch blocks (CLI + gateway), both with fast=
        assert blocks_with_fast >= 2, (
            f"Expected >= 2 session_search dispatch blocks with fast=, "
            f"found {blocks_with_fast}"
        )