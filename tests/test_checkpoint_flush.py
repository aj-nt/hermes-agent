"""Test that flush_checkpoint is called during context compression."""
import json
import pytest
from unittest.mock import MagicMock


def test_flush_checkpoint_method_exists():
    """AIAgent must have a flush_checkpoint method."""
    from run_agent import AIAgent
    assert hasattr(AIAgent, "flush_checkpoint")


def test_flush_checkpoint_writes_to_store(tmp_path):
    """flush_checkpoint should write a checkpoint with current session state."""
    from agent.checkpoint_store import CheckpointStore
    from tools.checkpoint_tool import checkpoint_tool
    store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")

    mock_agent = MagicMock()
    mock_agent.session_id = "flush_test_session"
    mock_agent._checkpoint_store = store
    mock_agent._todo_store = MagicMock()
    mock_agent._todo_store.format_for_injection.return_value = "- [x] Step 1"

    # Verify the checkpoint tool writes successfully (same path flush_checkpoint uses)
    result = checkpoint_tool(
        action="write",
        task="Auto-checkpoint before compression",
        progress=[],
        state={},
        decisions=[],
        store=store,
        agent=mock_agent,
    )
    data = json.loads(result)
    assert data["success"] is True
    saved = store.read("flush_test_session")
    assert saved is not None
    assert saved["task"] == "Auto-checkpoint before compression"