"""Regression guard for session search coverage: backfill JSON sessions into SQLite.

Session JSON files can exist on disk without corresponding rows in the
``messages`` table. This happens when sessions were created by code that
didn't call ``append_message()`` — for example, the old ``main`` before
term_index support was added, or sessions created via ``/new`` which
resets state before flushing.

``SessionDB.backfill_messages_from_json()`` scans the sessions directory
for JSON files whose session_id is missing from the ``messages`` table,
parses each file, inserts both the session row (if absent) and its
messages, then reindexes the term index to pick up the new data.
"""

import json
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def sessions_dir(tmp_path):
    """Create a sessions/ directory alongside the DB."""
    d = tmp_path / "sessions"
    d.mkdir()
    return d


def _write_session(path, session_id, messages, model="test:model",
                   session_start="2026-04-20T12:00:00", platform="cli"):
    """Helper: write a session JSON file."""
    data = {
        "session_id": session_id,
        "model": model,
        "platform": platform,
        "session_start": session_start,
        "last_updated": session_start,
        "system_prompt": "",
        "tools": [],
        "message_count": len(messages),
        "messages": messages,
    }
    Path(path).write_text(json.dumps(data))
    return data


# ── Session creation ──────────────────────────────────────────────────────

def test_creates_missing_session_row(db, sessions_dir):
    """Session in JSON but not in DB at all should be created."""
    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [{"role": "user", "content": "Hello"}],
        session_start="2026-04-20T12:00:00",
    )

    inserted = db.backfill_messages_from_json(sessions_dir)
    assert inserted >= 1

    # Session row should now exist
    row = db._conn.execute(
        "SELECT id, model, source FROM sessions WHERE id = ?", ("abc123",)
    ).fetchone()
    assert row is not None
    assert row["model"] == "test:model"
    assert row["source"] == "cli"


def test_does_not_duplicate_existing_session(db, sessions_dir):
    """Session already in DB should not be re-created (INSERT OR IGNORE)."""
    db.create_session("abc123", source="cli", model="existing:model")

    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [{"role": "user", "content": "Hello"}],
    )

    db.backfill_messages_from_json(sessions_dir)

    # Should still have only one session row, with the original model
    rows = db._conn.execute(
        "SELECT id, model FROM sessions WHERE id = ?", ("abc123",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["model"] == "existing:model"


# ── Message insertion ─────────────────────────────────────────────────────

def test_inserts_messages_for_empty_session(db, sessions_dir):
    """Session in DB with 0 messages gets its JSON messages inserted."""
    db.create_session("abc123", source="cli")

    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "tool", "content": '{"success": true}'},
        ],
    )

    db.backfill_messages_from_json(sessions_dir)

    msgs = db._conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
        ("abc123",),
    ).fetchall()

    # All 3 messages inserted
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello"
    assert msgs[1]["role"] == "assistant"
    # Tool messages ARE inserted (they have search value in backfill context)
    assert msgs[2]["role"] == "tool"


def test_skips_session_with_existing_messages(db, sessions_dir):
    """Session already having messages in DB should not be re-inserted."""
    db.create_session("abc123", source="cli")
    db.append_message("abc123", role="user", content="Existing")

    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [{"role": "user", "content": "From JSON"}],
    )

    db.backfill_messages_from_json(sessions_dir)

    msgs = db._conn.execute(
        "SELECT content FROM messages WHERE session_id = ? ORDER BY id",
        ("abc123",),
    ).fetchall()
    # Should only have the original message, not the JSON one
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Existing"


def test_synthesizes_timestamps_from_session_start(db, sessions_dir):
    """Messages without timestamps get monotonically increasing timestamps
    derived from session_start."""
    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ],
        session_start="2026-04-20T12:00:00",
    )

    db.backfill_messages_from_json(sessions_dir)

    msgs = db._conn.execute(
        "SELECT timestamp FROM messages WHERE session_id = ? ORDER BY id",
        ("abc123",),
    ).fetchall()

    # Timestamps should be monotonically increasing
    assert msgs[0]["timestamp"] < msgs[1]["timestamp"]
    # First timestamp should be close to the session start epoch
    assert msgs[0]["timestamp"] > 1745000000  # ~2025


def test_handles_message_without_content(db, sessions_dir):
    """Messages with None/empty content should not crash backfill."""
    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},  # e.g. function_call only
        ],
    )

    db.backfill_messages_from_json(sessions_dir)

    msgs = db._conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
        ("abc123",),
    ).fetchall()
    assert len(msgs) == 2


def test_preserves_tool_calls_and_reasoning(db, sessions_dir):
    """Tool calls, finish_reason, and reasoning should be preserved."""
    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [
            {"role": "user", "content": "Run it"},
            {
                "role": "assistant",
                "content": "Done",
                "finish_reason": "stop",
                "reasoning": "I thought about it",
                "tool_calls": [
                    {"name": "terminal", "arguments": '{"command": "ls"}'},
                ],
            },
        ],
    )

    db.backfill_messages_from_json(sessions_dir)

    msgs = db._conn.execute(
        "SELECT role, finish_reason, reasoning FROM messages WHERE session_id = ? ORDER BY id",
        ("abc123",),
    ).fetchall()
    assert msgs[1]["finish_reason"] == "stop"
    assert msgs[1]["reasoning"] == "I thought about it"


# ── Term index integration ────────────────────────────────────────────────

def test_backfill_populates_term_index(db, sessions_dir):
    """After backfill, term_index should contain terms from inserted messages."""
    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [{"role": "user", "content": "proton mail himalaya bridge setup"}],
    )

    db.backfill_messages_from_json(sessions_dir)

    # Term index should have entries for this session
    terms = db._conn.execute(
        "SELECT DISTINCT term FROM term_index WHERE session_id = ?",
        ("abc123",),
    ).fetchall()
    term_set = {r["term"] for r in terms}
    # Should contain at least some terms from the content
    assert len(term_set) > 0


def test_backfill_skips_tool_messages_in_term_index(db, sessions_dir):
    """Tool messages should be inserted into messages table but NOT indexed."""
    _write_session(
        sessions_dir / "session_abc123.json",
        "abc123",
        [
            {"role": "user", "content": "search for proton"},
            {"role": "tool", "content": '{"success": true, "results": ["noise_term_xyz"]}'},
        ],
    )

    db.backfill_messages_from_json(sessions_dir)

    # Both messages should be in messages table
    msgs = db._conn.execute(
        "SELECT role FROM messages WHERE session_id = ?", ("abc123",)
    ).fetchall()
    roles = [r["role"] for r in msgs]
    assert "user" in roles
    assert "tool" in roles

    # But tool message terms should NOT be in term_index
    terms = db._conn.execute(
        "SELECT term FROM term_index WHERE session_id = ?", ("abc123",)
    ).fetchall()
    term_set = {r["term"] for r in terms}
    assert "noise_term_xyz" not in term_set


# ── Edge cases ─────────────────────────────────────────────────────────────

def test_handles_malformed_json_gracefully(db, sessions_dir):
    """Malformed JSON files should be skipped, not crash the backfill."""
    # Write a valid session
    _write_session(
        sessions_dir / "session_good.json",
        "good",
        [{"role": "user", "content": "Hello"}],
    )

    # Write a corrupted JSON file
    (sessions_dir / "session_bad.json").write_text("{broken json")

    # Write a non-session JSON file (different name pattern)
    (sessions_dir / "request_dump_xyz.json").write_text('{"foo": "bar"}')

    inserted = db.backfill_messages_from_json(sessions_dir)
    # Should have inserted the good session's messages and moved on
    assert inserted >= 1


def test_handles_empty_session_json(db, sessions_dir):
    """Session JSON with empty messages list should create session row only."""
    _write_session(
        sessions_dir / "session_empty.json",
        "empty",
        [],  # No messages
    )

    db.backfill_messages_from_json(sessions_dir)

    # Session should exist
    row = db._conn.execute(
        "SELECT id FROM sessions WHERE id = ?", ("empty",)
    ).fetchone()
    assert row is not None

    # But no messages
    msgs = db._conn.execute(
        "SELECT id FROM messages WHERE session_id = ?", ("empty",)
    ).fetchall()
    assert len(msgs) == 0


def test_only_processes_session_prefix_files(db, sessions_dir):
    """Files not matching session_*.json pattern should be ignored."""
    _write_session(
        sessions_dir / "session_real.json",
        "real",
        [{"role": "user", "content": "Hello"}],
    )

    # Create a file that doesn't match session_*.json pattern
    (sessions_dir / "request_dump_2026.json").write_text(json.dumps({
        "session_id": "dump",
        "messages": [{"role": "user", "content": "Should not be processed"}],
    }))

    db.backfill_messages_from_json(sessions_dir)

    # Only the session_real should be in the DB
    sessions = db._conn.execute("SELECT id FROM sessions").fetchall()
    session_ids = {r["id"] for r in sessions}
    assert "real" in session_ids
    assert "dump" not in session_ids


def test_returns_count_of_inserted_messages(db, sessions_dir):
    """backfill_messages_from_json should return total messages inserted."""
    _write_session(
        sessions_dir / "session_s1.json",
        "s1",
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )
    _write_session(
        sessions_dir / "session_s2.json",
        "s2",
        [
            {"role": "user", "content": "Hey"},
            {"role": "assistant", "content": "Yo"},
            {"role": "user", "content": "What's up"},
        ],
    )

    count = db.backfill_messages_from_json(sessions_dir)
    assert count == 5  # 2 + 3


def test_idempotent(db, sessions_dir):
    """Running backfill twice should not duplicate messages."""
    _write_session(
        sessions_dir / "session_abc.json",
        "abc",
        [{"role": "user", "content": "Hello"}],
    )

    count1 = db.backfill_messages_from_json(sessions_dir)
    count2 = db.backfill_messages_from_json(sessions_dir)

    assert count1 == 1
    assert count2 == 0  # Second run finds nothing new to insert

    msgs = db._conn.execute(
        "SELECT content FROM messages WHERE session_id = ?", ("abc",)
    ).fetchall()
    assert len(msgs) == 1  # Not duplicated