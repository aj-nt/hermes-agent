import json
import pytest
from unittest.mock import MagicMock
from agent.checkpoint_store import CheckpointStore
from tools.checkpoint_tool import checkpoint_tool


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")


@pytest.fixture
def agent_context(store):
    """Simulate the fields the tool handler pulls from the AIAgent."""
    ctx = MagicMock()
    ctx.session_id = "test_session_001"
    ctx._todo_store = MagicMock()
    ctx._todo_store.format_for_injection.return_value = "- [x] Step 1\n- [ ] Step 2"
    return ctx


class TestCheckpointWrite:
    def test_write_basic_checkpoint(self, store, agent_context):
        result = checkpoint_tool(
            action="write",
            task="Build feature X",
            progress=[{"step": "Setup", "status": "completed"}],
            state={"active_branch": "main"},
            decisions=["Use SQLite"],
            store=store,
            agent=agent_context,
        )
        data = json.loads(result)
        assert data["success"] is True
        assert data["session_id"] == "test_session_001"
        saved = store.read("test_session_001")
        assert saved["task"] == "Build feature X"
        assert saved["status"] == "in_progress"

    def test_write_includes_auto_fields(self, store, agent_context):
        checkpoint_tool(
            action="write",
            task="Test task",
            progress=[],
            state={},
            decisions=[],
            store=store,
            agent=agent_context,
        )
        saved = store.read("test_session_001")
        assert "created" in saved
        assert "updated" in saved
        assert saved["session_id"] == "test_session_001"


class TestCheckpointUpdate:
    def test_update_merges_progress(self, store, agent_context):
        checkpoint_tool(
            action="write",
            task="Build feature X",
            progress=[{"step": "Setup", "status": "completed"}],
            state={"active_branch": "main"},
            decisions=[],
            store=store,
            agent=agent_context,
        )
        result = checkpoint_tool(
            action="update",
            progress=[{"step": "Implement", "status": "in_progress"}],
            state={"active_branch": "feat/x"},
            store=store,
            agent=agent_context,
        )
        data = json.loads(result)
        assert data["success"] is True
        saved = store.read("test_session_001")
        assert saved["task"] == "Build feature X"  # preserved
        assert len(saved["progress"]) == 2  # merged

    def test_update_nonexistent_returns_error(self, store, agent_context):
        result = checkpoint_tool(
            action="update",
            progress=[],
            state={},
            store=store,
            agent=agent_context,
        )
        data = json.loads(result)
        assert data["success"] is False


class TestCheckpointRead:
    def test_read_existing(self, store, agent_context):
        checkpoint_tool(
            action="write",
            task="Test task",
            progress=[],
            state={},
            decisions=[],
            store=store,
            agent=agent_context,
        )
        result = checkpoint_tool(action="read", store=store, agent=agent_context)
        data = json.loads(result)
        assert data["success"] is True
        assert data["checkpoint"]["task"] == "Test task"

    def test_read_nonexistent(self, store, agent_context):
        result = checkpoint_tool(action="read", store=store, agent=agent_context)
        data = json.loads(result)
        assert data["success"] is False
        assert "no checkpoint" in data["error"].lower()


class TestCheckpointClear:
    def test_clear_removes_checkpoint(self, store, agent_context):
        checkpoint_tool(
            action="write",
            task="Test task",
            progress=[],
            state={},
            decisions=[],
            store=store,
            agent=agent_context,
        )
        result = checkpoint_tool(action="clear", store=store, agent=agent_context)
        data = json.loads(result)
        assert data["success"] is True
        assert store.read("test_session_001") is None