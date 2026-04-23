"""Test that flush_checkpoint writes a checkpoint via checkpoint_tool."""
import json
import pytest
from unittest.mock import MagicMock, patch


def test_flush_checkpoint_method_exists():
    """AIAgent must have a flush_checkpoint method."""
    from run_agent import AIAgent
    assert hasattr(AIAgent, "flush_checkpoint")


def test_flush_checkpoint_writes_to_store(tmp_path):
    """flush_checkpoint should write a checkpoint with current session state."""
    from agent.checkpoint_store import CheckpointStore

    store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")

    mock_todo = MagicMock()
    mock_todo._items = [
        {"content": "Step 1", "status": "completed"},
        {"content": "Step 2", "status": "in_progress"},
    ]
    mock_todo.format_for_injection.return_value = "- [x] Step 1\n- [~] Step 2"

    mock_session_db = MagicMock()
    mock_session_db.get_session_title.return_value = "Test session title"

    mock_agent = MagicMock()
    mock_agent.session_id = "flush_test_session"
    mock_agent._checkpoint_store = store
    mock_agent._todo_store = mock_todo
    mock_agent._session_db = mock_session_db

    # Call the real flush_checkpoint method
    from run_agent import AIAgent
    AIAgent.flush_checkpoint(mock_agent)

    # Verify a checkpoint was written for this session
    saved = store.read("flush_test_session")
    assert saved is not None
    assert saved["task"] == "Test session title"
    assert saved["status"] == "in_progress"


def test_flush_checkpoint_noops_without_store(tmp_path):
    """flush_checkpoint should silently return if _checkpoint_store is None."""
    mock_agent = MagicMock()
    mock_agent._checkpoint_store = None

    from run_agent import AIAgent
    # Should not raise
    AIAgent.flush_checkpoint(mock_agent)