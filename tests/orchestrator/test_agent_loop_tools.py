"""Tests for agent-loop tool dispatch through CompatShim._dispatch_tool.

Agent-loop tools (memory, session_search, todo, clarify, delegate_task,
checkpoint) need access to parent AIAgent state stores and cannot be
dispatched through model_tools.handle_function_call.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from agent.orchestrator.compat import AIAgentCompatShim


class TestAgentLoopToolRouting:
    """Agent-loop tools are handled in _dispatch_agent_loop_tool, not handle_function_call."""

    def test_memory_dispatched_to_agent_loop_handler(self):
        """memory should NOT go through handle_function_call."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        mock_parent._memory_store = MagicMock()
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-al")

        with patch("model_tools.handle_function_call") as mock_hfc:
            with patch.object(shim, "_dispatch_agent_loop_tool", return_value='{"success": true}') as mock_al:
                shim._dispatch_tool("memory", {"action": "search", "query": "test"})

        mock_al.assert_called_once_with("memory", {"action": "search", "query": "test"})
        mock_hfc.assert_not_called()

    def test_session_search_dispatched_to_agent_loop_handler(self):
        """session_search should NOT go through handle_function_call."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-al")

        with patch("model_tools.handle_function_call") as mock_hfc:
            with patch.object(shim, "_dispatch_agent_loop_tool", return_value='{"results": []}') as mock_al:
                shim._dispatch_tool("session_search", {"query": "test"})

        mock_al.assert_called_once_with("session_search", {"query": "test"})
        mock_hfc.assert_not_called()

    def test_todo_dispatched_to_agent_loop_handler(self):
        """todo should NOT go through handle_function_call."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-al")

        with patch("model_tools.handle_function_call") as mock_hfc:
            with patch.object(shim, "_dispatch_agent_loop_tool", return_value='{}') as mock_al:
                shim._dispatch_tool("todo", {"todos": []})

        mock_al.assert_called_once_with("todo", {"todos": []})
        mock_hfc.assert_not_called()

    def test_regular_tool_goes_to_handle_function_call(self):
        """Non-agent-loop tools still go through handle_function_call."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-al")

        with patch("model_tools.handle_function_call", return_value='{"results": []}') as mock_hfc:
            result = shim._dispatch_tool("web_search", {"query": "test"})

        mock_hfc.assert_called_once()

    def test_agent_loop_tools_set_includes_all_required(self):
        """The _AGENT_LOOP_TOOLS set must include all agent-loop tools."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-al")
        # Must be a superset of model_tools._AGENT_LOOP_TOOLS
        from model_tools import _AGENT_LOOP_TOOLS as MODEL_TOOLS_SET
        assert MODEL_TOOLS_SET <= shim._AGENT_LOOP_TOOLS, (
            f"CompatShim._AGENT_LOOP_TOOLS missing: {MODEL_TOOLS_SET - shim._AGENT_LOOP_TOOLS}"
        )


class TestAgentLoopToolNoParent:
    """Agent-loop tools return a clear error when no parent agent is available."""

    def test_memory_no_parent(self):
        shim = AIAgentCompatShim(parent_agent=None, session_id="test-np")
        result = shim._dispatch_agent_loop_tool("memory", {"action": "search", "query": "test"})
        data = json.loads(result)
        assert "error" in data
        assert "no parent agent" in data["error"].lower()

    def test_session_search_no_parent(self):
        shim = AIAgentCompatShim(parent_agent=None, session_id="test-np")
        result = shim._dispatch_agent_loop_tool("session_search", {"query": "test"})
        data = json.loads(result)
        assert "error" in data

    def test_todo_no_parent(self):
        shim = AIAgentCompatShim(parent_agent=None, session_id="test-np")
        result = shim._dispatch_agent_loop_tool("todo", {"todos": []})
        data = json.loads(result)
        assert "error" in data

    def test_clarify_no_parent(self):
        shim = AIAgentCompatShim(parent_agent=None, session_id="test-np")
        result = shim._dispatch_agent_loop_tool("clarify", {"question": "test?"})
        data = json.loads(result)
        assert "error" in data

    def test_delegate_task_no_parent(self):
        shim = AIAgentCompatShim(parent_agent=None, session_id="test-np")
        result = shim._dispatch_agent_loop_tool("delegate_task", {"goal": "test"})
        data = json.loads(result)
        assert "error" in data

    def test_checkpoint_no_parent(self):
        shim = AIAgentCompatShim(parent_agent=None, session_id="test-np")
        result = shim._dispatch_agent_loop_tool("checkpoint", {"action": "read"})
        data = json.loads(result)
        assert "error" in data


class TestAgentLoopToolMemoryDispatch:
    """Memory tool dispatch with mock state stores."""

    def test_memory_no_store(self):
        """memory returns error when _memory_store is not initialized."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        mock_parent._memory_store = None
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-ms")

        result = shim._dispatch_agent_loop_tool("memory", {"action": "search", "query": "test"})
        data = json.loads(result)
        assert "error" in data

    def test_session_search_no_db(self):
        """session_search returns error when _session_db is not available."""
        mock_parent = MagicMock()
        mock_parent.tools = []
        mock_parent._session_db = None
        shim = AIAgentCompatShim(parent_agent=mock_parent, session_id="test-db")

        result = shim._dispatch_agent_loop_tool("session_search", {"query": "test"})
        data = json.loads(result)
        assert data.get("success") is False
        assert "error" in data