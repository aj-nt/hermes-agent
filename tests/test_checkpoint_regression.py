"""Regression tests for bugs found during red-team QA review.

Bug #1: get_parent_session_id() didn't exist on SessionDB
Bug #2: Path traversal via session_id
Bug #3: "checkpoint" not in any toolset definition
Bug #4: flush_checkpoint test didn't actually call flush_checkpoint
Bug #5: Injection fallback grabbed ANY in_progress checkpoint from unrelated sessions
"""
import pytest
from pathlib import Path


class TestBug1SessionDBLineageWalk:
    """Bug #1: _build_system_prompt must use get_session()["parent_session_id"]
    instead of the nonexistent get_parent_session_id() method.
    """

    def test_session_db_has_no_get_parent_session_id_method(self):
        """SessionDB must NOT have a get_parent_session_id method (it never did)."""
        from hermes_state import SessionDB
        assert not hasattr(SessionDB, "get_parent_session_id"), (
            "SessionDB should not have get_parent_session_id -- "
            "use get_session(sid)['parent_session_id'] instead"
        )

    def test_get_session_returns_parent_session_id_key(self, tmp_path):
        """get_session() must return a dict with 'parent_session_id' key."""
        from hermes_state import SessionDB
        from pathlib import Path
        db = SessionDB(db_path=Path(tmp_path) / "test.db")
        # Create a parent session
        parent_id = db.create_session("parent_session", source="test", model="test")
        # Create a child session
        child_id = db.create_session("child_session", source="test", model="test", parent_session_id=parent_id)
        sess = db.get_session(child_id)
        assert sess is not None
        assert "parent_session_id" in sess
        assert sess["parent_session_id"] == parent_id


class TestBug2PathTraversal:
    """Bug #2: session_id must be validated to prevent path traversal."""

    def test_reject_dotdot_in_session_id(self, tmp_path):
        from agent.checkpoint_store import CheckpointStore
        store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")
        with pytest.raises(ValueError, match="Invalid session_id"):
            store.write("../../../etc/passwd", {"task": "evil"})

    def test_reject_slash_in_session_id(self, tmp_path):
        from agent.checkpoint_store import CheckpointStore
        store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")
        with pytest.raises(ValueError, match="Invalid session_id"):
            store.write("sub/dir", {"task": "evil"})

    def test_reject_backslash_in_session_id(self, tmp_path):
        from agent.checkpoint_store import CheckpointStore
        store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")
        with pytest.raises(ValueError, match="Invalid session_id"):
            store.write("sub\\dir", {"task": "evil"})

    def test_accept_normal_session_id(self, tmp_path):
        from agent.checkpoint_store import CheckpointStore
        store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")
        # Should not raise
        store.write("20260422_225800_abc123", {"task": "normal"})

    def test_read_rejects_path_traversal(self, tmp_path):
        from agent.checkpoint_store import CheckpointStore
        store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")
        with pytest.raises(ValueError, match="Invalid session_id"):
            store.read("../../../etc/passwd")

    def test_delete_rejects_path_traversal(self, tmp_path):
        from agent.checkpoint_store import CheckpointStore
        store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")
        with pytest.raises(ValueError, match="Invalid session_id"):
            store.delete("../../../etc/passwd")


class TestBug3CheckpointInToolset:
    """Bug #3: 'checkpoint' must be in a toolset definition so the model gets the schema."""

    def test_checkpoint_in_todo_toolset(self):
        from toolsets import TOOLSETS
        todo_tools = TOOLSETS.get("todo", {}).get("tools", [])
        assert "checkpoint" in todo_tools, (
            f"'checkpoint' must be in the 'todo' toolset tools list. "
            f"Current todo tools: {todo_tools}"
        )

    def test_resolve_toolset_todo_includes_checkpoint(self):
        from toolsets import resolve_toolset
        resolved = resolve_toolset("todo")
        assert "checkpoint" in resolved, (
            f"resolve_toolset('todo') must return ['todo', 'checkpoint']. Got: {resolved}"
        )


class TestBug5InjectionNoCrossSession:
    """Bug #5: build_checkpoint_system_prompt must NOT fall through to grabbing
    any random in_progress checkpoint from an unrelated session.
    """

    def test_no_injection_when_parent_has_no_checkpoint(self, tmp_path):
        from agent.checkpoint_store import CheckpointStore
        from agent.checkpoint_injection import build_checkpoint_system_prompt

        store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")
        # Write a checkpoint for an unrelated session
        store.write("unrelated_session", {
            "task": "Unrelated task",
            "status": "in_progress",
            "progress": [],
        })

        # Ask for a parent_session_id that has no checkpoint
        prompt = build_checkpoint_system_prompt(store=store, parent_session_id="missing_session")
        assert prompt == "", (
            "Must not inject a checkpoint from an unrelated session when "
            "the requested parent_session_id has no checkpoint"
        )

    def test_only_injects_specific_parent(self, tmp_path):
        from agent.checkpoint_store import CheckpointStore
        from agent.checkpoint_injection import build_checkpoint_system_prompt

        store = CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")
        # Write checkpoints for two sessions
        store.write("parent_A", {
            "task": "Parent A task",
            "status": "in_progress",
            "progress": [],
        })
        store.write("parent_B", {
            "task": "Parent B task",
            "status": "in_progress",
            "progress": [],
        })

        # Request parent_A specifically
        prompt = build_checkpoint_system_prompt(store=store, parent_session_id="parent_A")
        assert "Parent A task" in prompt
        assert "Parent B task" not in prompt