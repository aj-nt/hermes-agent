import pytest
from agent.checkpoint_store import CheckpointStore
from agent.checkpoint_injection import build_checkpoint_system_prompt


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")


@pytest.fixture
def parent_checkpoint():
    return {
        "session_id": "old_session_001",
        "task": "Red team QA on PR #14303",
        "status": "in_progress",
        "created": "2026-04-22T22:58:00",
        "updated": "2026-04-22T23:15:00",
        "progress": [
            {"step": "Read code paths", "status": "completed"},
            {"step": "Add edge-case tests", "status": "in_progress"},
        ],
        "state": {"active_branch": "fix/delegation", "tests_status": "22/22 passing"},
        "decisions": ["SSRF is low-severity"],
        "blocked": [],
        "unresolved": [],
    }


class TestBuildCheckpointPrompt:
    def test_injects_in_progress_checkpoint(self, store, parent_checkpoint):
        store.write("old_session_001", parent_checkpoint)
        prompt = build_checkpoint_system_prompt(store=store, parent_session_id="old_session_001")
        assert "Red team QA on PR #14303" in prompt
        assert "CHECKPOINT" in prompt

    def test_skips_completed_checkpoints(self, store, parent_checkpoint):
        parent_checkpoint["status"] = "completed"
        store.write("old_session_001", parent_checkpoint)
        prompt = build_checkpoint_system_prompt(store=store, parent_session_id="old_session_001")
        assert prompt == ""

    def test_returns_empty_when_no_parent(self, store):
        prompt = build_checkpoint_system_prompt(store=store, parent_session_id="nonexistent")
        assert prompt == ""

    def test_includes_progress_and_decisions(self, store, parent_checkpoint):
        store.write("old_session_001", parent_checkpoint)
        prompt = build_checkpoint_system_prompt(store=store, parent_session_id="old_session_001")
        assert "Read code paths" in prompt
        assert "SSRF is low-severity" in prompt

    def test_truncates_oversized_checkpoints(self, store, parent_checkpoint):
        parent_checkpoint["task"] = "x" * 5000
        parent_checkpoint["decisions"] = ["d" * 2000] * 10
        store.write("old_session_001", parent_checkpoint)
        prompt = build_checkpoint_system_prompt(store=store, parent_session_id="old_session_001")
        assert len(prompt) < 6000