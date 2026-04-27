"""Tests for orchestrator memory coordinator types.

These verify:
1. NudgeTracker increments and resets correctly
2. WriteMetadataTracker builds provenance dicts
3. MemoryEvent carries the right fields
4. MemoryCoordinator interface exists and is instantiable
5. ReviewContext captures memory state for background review
"""

import pytest
import time


# ============================================================================
# NudgeTracker
# ============================================================================

class TestNudgeTracker:
    """NudgeTracker replaces the ad-hoc _turns_since_memory counter."""

    def test_import(self):
        from agent.orchestrator.memory import NudgeTracker

    def test_memory_nudge_defaults_none(self):
        from agent.orchestrator.memory import NudgeTracker
        nt = NudgeTracker()
        assert nt.memory_nudge() is None

    def test_skill_nudge_defaults_none(self):
        from agent.orchestrator.memory import NudgeTracker
        nt = NudgeTracker()
        assert nt.skill_nudge() is None

    def test_on_turn_increments_counter(self):
        from agent.orchestrator.memory import NudgeTracker
        nt = NudgeTracker(nudge_interval=3)
        nt.on_turn()
        nt.on_turn()
        nt.on_turn()
        # After 3 turns, nudge should fire
        nudge = nt.memory_nudge()
        assert nudge is not None

    def test_memory_use_resets_counter(self):
        from agent.orchestrator.memory import NudgeTracker
        nt = NudgeTracker(nudge_interval=3)
        nt.on_turn()
        nt.on_turn()
        nt.on_memory_use()  # resets
        nt.on_turn()
        # Only 1 turn since reset, not enough for nudge
        assert nt.memory_nudge() is None

    def test_skill_use_resets_counter(self):
        from agent.orchestrator.memory import NudgeTracker
        nt = NudgeTracker(skill_interval=3)
        nt.on_turn()
        nt.on_turn()
        nt.on_skill_use()  # resets
        nt.on_turn()
        assert nt.skill_nudge() is None

    def test_custom_interval(self):
        from agent.orchestrator.memory import NudgeTracker
        nt = NudgeTracker(nudge_interval=5)
        for _ in range(4):
            nt.on_turn()
        assert nt.memory_nudge() is None  # 4 < 5
        nt.on_turn()
        assert nt.memory_nudge() is not None  # 5 >= 5


# ============================================================================
# WriteMetadataTracker
# ============================================================================

class TestWriteMetadataTracker:
    """WriteMetadataTracker builds provenance dicts for memory writes."""

    def test_import(self):
        from agent.orchestrator.memory import WriteMetadataTracker

    def test_build_metadata_includes_origin(self):
        from agent.orchestrator.memory import WriteMetadataTracker
        wmt = WriteMetadataTracker(session_id="s1")
        meta = wmt.build_metadata()
        assert "write_origin" in meta

    def test_build_metadata_includes_session_id(self):
        from agent.orchestrator.memory import WriteMetadataTracker
        wmt = WriteMetadataTracker(session_id="s1")
        meta = wmt.build_metadata()
        assert meta["session_id"] == "s1"

    def test_build_metadata_includes_platform(self):
        from agent.orchestrator.memory import WriteMetadataTracker
        wmt = WriteMetadataTracker(session_id="s1", platform="telegram")
        meta = wmt.build_metadata()
        assert meta["platform"] == "telegram"

    def test_set_origin_updates_metadata(self):
        from agent.orchestrator.memory import WriteMetadataTracker
        wmt = WriteMetadataTracker(session_id="s1")
        wmt.set_origin("background_review", "background_review")
        meta = wmt.build_metadata()
        assert meta["write_origin"] == "background_review"
        assert meta["execution_context"] == "background_review"

    def test_optional_fields_excluded_when_empty(self):
        from agent.orchestrator.memory import WriteMetadataTracker
        wmt = WriteMetadataTracker(session_id="s1", parent_session_id="")
        meta = wmt.build_metadata()
        # Empty strings should be excluded
        assert "parent_session_id" not in meta

    def test_task_id_included_when_provided(self):
        from agent.orchestrator.memory import WriteMetadataTracker
        wmt = WriteMetadataTracker(session_id="s1")
        meta = wmt.build_metadata(task_id="task-42")
        assert meta["task_id"] == "task-42"


# ============================================================================
# MemoryEvent
# ============================================================================

class TestMemoryEvent:
    """MemoryEvent is published to the MemoryBus."""

    def test_import(self):
        from agent.orchestrator.memory import MemoryEvent

    def test_kind_field(self):
        from agent.orchestrator.memory import MemoryEvent
        me = MemoryEvent(kind="write", source_session_id="s1", target="memory")
        assert me.kind == "write"

    def test_key_defaults_none(self):
        from agent.orchestrator.memory import MemoryEvent
        me = MemoryEvent(kind="write", source_session_id="s1", target="memory")
        assert me.key is None

    def test_timestamp_auto_set(self):
        from agent.orchestrator.memory import MemoryEvent
        before = time.time()
        me = MemoryEvent(kind="write", source_session_id="s1", target="memory")
        after = time.time()
        assert before <= me.timestamp <= after


# ============================================================================
# MemoryCoordinator
# ============================================================================

class TestMemoryCoordinator:
    """MemoryCoordinator is the unified interface for all memory systems."""

    def test_import(self):
        from agent.orchestrator.memory import MemoryCoordinator

    def test_can_instantiate(self):
        from agent.orchestrator.memory import MemoryCoordinator
        mc = MemoryCoordinator(session_id="s1")
        assert mc is not None

    def test_nudge_tracker_instance(self):
        from agent.orchestrator.memory import MemoryCoordinator, NudgeTracker
        mc = MemoryCoordinator(session_id="s1")
        assert isinstance(mc.nudge_tracker, NudgeTracker)

    def test_write_metadata_instance(self):
        from agent.orchestrator.memory import MemoryCoordinator, WriteMetadataTracker
        mc = MemoryCoordinator(session_id="s1")
        assert isinstance(mc.write_metadata, WriteMetadataTracker)


# ============================================================================
# ReviewContext
# ============================================================================

class TestReviewContext:
    """ReviewContext decouples background review from AIAgent internals."""

    def test_import(self):
        from agent.orchestrator.memory import ReviewContext

    def test_fields_present(self):
        from agent.orchestrator.memory import ReviewContext
        rc = ReviewContext(
            memory_enabled=True,
            user_profile_enabled=False,
            session_id="s1",
            write_origin="background_review",
            write_context="background_review",
        )
        assert rc.memory_enabled is True
        assert rc.session_id == "s1"
        assert rc.write_origin == "background_review"
