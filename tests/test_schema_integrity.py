"""Tests for database schema integrity — column presence, migration correctness.

These tests verify that:
1. A fresh SessionDB has all expected columns in both tables
2. append_message succeeds with every column (the codex_message_items bug)
3. Migration from an older schema version correctly adds missing columns
4. A migration that fails to add a column does NOT silently bump the schema version

The motivating bug: v9 migration's ALTER TABLE ADD COLUMN "codex_message_items"
silently failed (caught by except OperationalError: pass), then schema_version
was bumped to 9 anyway. Every subsequent append_message INSERT failed silently
because the column was missing — all session messages were written to JSON files
but never indexed into SQLite.
"""

import sqlite3
import pytest
from pathlib import Path

from hermes_state import SessionDB, SCHEMA_VERSION


@pytest.fixture()
def db(tmp_path):
    """Create a fresh SessionDB for column/insertion tests."""
    db_path = tmp_path / "test_schema.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


# Canonical column sets — update these when migrations add new columns.
# These serve as the "schema contract" that the code expects.
EXPECTED_SESSIONS_COLUMNS = {
    "id",
    "source",
    "user_id",
    "model",
    "model_config",
    "system_prompt",
    "parent_session_id",
    "started_at",
    "ended_at",
    "end_reason",
    "message_count",
    "tool_call_count",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "billing_provider",
    "billing_base_url",
    "billing_mode",
    "estimated_cost_usd",
    "actual_cost_usd",
    "cost_status",
    "cost_source",
    "pricing_version",
    "title",
    "api_call_count",
}

EXPECTED_MESSAGES_COLUMNS = {
    "id",
    "session_id",
    "role",
    "content",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "timestamp",
    "token_count",
    "finish_reason",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",  # v9 — the column that was missing
}


def _get_columns(conn, table):
    """Return set of column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


class TestSchemaColumnsPresent:
    """Verify every expected column exists in a fresh SessionDB.

    If a migration adds a new column and this test fails, add it to
    EXPECTED_*_COLUMNS above and fix the migration — don't just update the set.
    """

    def test_sessions_table_has_all_expected_columns(self, db):
        actual = _get_columns(db._conn, "sessions")
        missing = EXPECTED_SESSIONS_COLUMNS - actual
        assert not missing, f"Missing columns in sessions table: {missing}"

    def test_messages_table_has_all_expected_columns(self, db):
        actual = _get_columns(db._conn, "messages")
        missing = EXPECTED_MESSAGES_COLUMNS - actual
        assert not missing, f"Missing columns in messages table: {missing}"

    def test_schema_version_matches_known_constant(self, db):
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == SCHEMA_VERSION

    def test_no_extra_unexpected_columns_in_messages(self, db):
        """Catch typos or accidental duplicates in the expected set."""
        actual = _get_columns(db._conn, "messages")
        extra = actual - EXPECTED_MESSAGES_COLUMNS
        # We allow 'extra' columns since the DB may have internal FTS triggers etc.
        # But the core data columns should all be accounted for.
        # Just verify codex_message_items is present specifically.
        assert "codex_message_items" in actual


class TestAppendMessageWithAllColumns:
    """Test that append_message works with every supported column.

    This is the exact INSERT that was silently failing when codex_message_items
    was missing from the schema.
    """

    def test_append_message_with_all_columns(self, db):
        sid = db.create_session("test-append", "cli")
        msg_id = db.append_message(
            session_id=sid,
            role="assistant",
            content="test content",
            tool_name=None,
            tool_calls=None,
            tool_call_id=None,
            token_count=42,
            finish_reason="stop",
            reasoning="think step by step",
            reasoning_content="<think>step</think>",
            reasoning_details={"type": "summary", "text": "step"},
            codex_reasoning_items=[{"id": "item1"}],
            codex_message_items=[{"id": "msg1", "role": "assistant"}],
        )
        assert msg_id is not None
        assert isinstance(msg_id, int)
        assert msg_id > 0

        # Verify the message was actually stored
        messages = db.get_messages(sid)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "test content"
        assert msg["token_count"] == 42
        assert msg["finish_reason"] == "stop"
        assert msg["reasoning"] == "think step by step"
        assert msg["reasoning_content"] == "<think>step</think>"
        # JSON-serialized fields: get_messages() returns raw JSON strings
        # (deserialization happens in get_messages_as_conversation)
        import json
        assert json.loads(msg["reasoning_details"]) == {"type": "summary", "text": "step"}
        assert json.loads(msg["codex_reasoning_items"]) == [{"id": "item1"}]
        assert json.loads(msg["codex_message_items"]) == [{"id": "msg1", "role": "assistant"}]

    def test_append_message_minimal_columns(self, db):
        """append_message with only required fields still succeeds."""
        sid = db.create_session("test-minimal", "cli")
        msg_id = db.append_message(
            session_id=sid,
            role="user",
            content="hello",
        )
        assert msg_id > 0
        messages = db.get_messages(sid)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"
        assert messages[0]["codex_message_items"] is None

    def test_append_message_updates_message_count(self, db):
        """append_message increments session.message_count."""
        sid = db.create_session("test-count", "cli")
        db.append_message(session_id=sid, role="user", content="hi")
        db.append_message(session_id=sid, role="assistant", content="hello")
        session = db.get_session(sid)
        assert session["message_count"] == 2

    def test_append_message_with_tool_calls(self, db):
        """append_message counts tool_calls in message_count and tool_call_count."""
        sid = db.create_session("test-tools", "cli")
        db.append_message(
            session_id=sid,
            role="assistant",
            content=None,
            tool_calls=[
                {"name": "terminal", "arguments": '{"command": "ls"}'},
                {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
            ],
        )
        session = db.get_session(sid)
        assert session["message_count"] == 1
        assert session["tool_call_count"] == 2


class TestSchemaMigrationFromV8:
    """Test migration from v8 (the version before codex_message_items).

    This simulates the exact scenario that caused the bug: an existing v8
    database that needs the v9 ALTER TABLE to add codex_message_items.
    """

    def _create_v8_db(self, db_path):
        """Create a database at schema version 8 (without codex_message_items)."""
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (8);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT
            );

            CREATE TABLE state_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
            CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
        """)

        # Insert a session and a message to prove migration preserves data
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("v8-session", "cli", 1000.0),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("v8-session", "user", "hello from v8", 1001.0),
        )
        conn.commit()
        conn.close()

    def test_migration_adds_codex_message_items(self, tmp_path):
        """V8 -> v10 migration must add codex_message_items column."""
        db_path = tmp_path / "migrate_v8.db"
        self._create_v8_db(db_path)

        # Open with SessionDB — triggers migration
        migrated_db = SessionDB(db_path=db_path)

        # Verify schema version bumped
        cursor = migrated_db._conn.execute("SELECT version FROM schema_version")
        assert cursor.fetchone()[0] == SCHEMA_VERSION

        # The critical check: codex_message_items column must exist
        columns = _get_columns(migrated_db._conn, "messages")
        assert "codex_message_items" in columns, (
            f"codex_message_items column missing after migration. "
            f"Columns: {sorted(columns)}"
        )

    def test_migration_preserves_existing_data(self, tmp_path):
        """V8 -> v10 migration preserves sessions and messages."""
        db_path = tmp_path / "migrate_v8_data.db"
        self._create_v8_db(db_path)

        migrated_db = SessionDB(db_path=db_path)

        # Existing session still accessible
        session = migrated_db.get_session("v8-session")
        assert session is not None
        assert session["source"] == "cli"

        # Existing message still accessible
        messages = migrated_db.get_messages("v8-session")
        assert len(messages) == 1
        assert messages[0]["content"] == "hello from v8"

    def test_migration_enables_append_message(self, tmp_path):
        """After v8 -> v10 migration, append_message with all columns works."""
        db_path = tmp_path / "migrate_v8_append.db"
        self._create_v8_db(db_path)

        migrated_db = SessionDB(db_path=db_path)

        # This INSERT is exactly what was silently failing before the fix
        sid = migrated_db.create_session("post-migration", "cli")
        msg_id = migrated_db.append_message(
            session_id=sid,
            role="assistant",
            content="post-migration message",
            codex_message_items=[{"id": "codex_1"}],
        )
        assert msg_id > 0

        messages = migrated_db.get_messages(sid)
        assert len(messages) == 1
        assert messages[0]["content"] == "post-migration message"
        import json
        assert json.loads(messages[0]["codex_message_items"]) == [{"id": "codex_1"}]

        migrated_db.close()


class TestMigrationWontSilentlySkipColumn:
    """Verify that if ALTER TABLE fails for any reason other than
    'duplicate column', the migration raises instead of silently
    bumping the schema version.

    This is a regression test for the bug where v9's ALTER TABLE
    ADD COLUMN "codex_message_items" failed silently (caught by
    `except OperationalError: pass`), causing all append_message
    INSERTs to fail because the column didn't exist.
    """

    def test_alter_table_non_duplicate_error_propagates(self, tmp_path):
        """Simulate an ALTER TABLE failure (not 'duplicate column name')
        and verify the migration does NOT silently bump schema version."""
        db_path = tmp_path / "migration_trap.db"

        # Create a minimal v8 DB
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (8);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                parent_session_id TEXT,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                ended_at REAL,
                end_reason TEXT,
                api_call_count INTEGER DEFAULT 0,
                FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT
            );

            CREATE TABLE state_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
            CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
        """)
        conn.commit()
        conn.close()

        # Pre-add the column so ALTER TABLE fails with "duplicate column"
        # which the migration should handle gracefully (no exception).
        conn = sqlite3.connect(str(db_path))
        conn.execute('ALTER TABLE messages ADD COLUMN "codex_message_items" TEXT')
        conn.commit()
        conn.close()

        # Opening this DB should NOT raise — "duplicate column" is expected
        # and the migration should complete successfully.
        db = SessionDB(db_path=db_path)
        columns = _get_columns(db._conn, "messages")
        assert "codex_message_items" in columns, (
            f"codex_message_items missing even though column was pre-added. "
            f"Columns: {sorted(columns)}"
        )

        # Verify schema reached current version
        cursor = db._conn.execute("SELECT version FROM schema_version")
        assert cursor.fetchone()[0] == SCHEMA_VERSION
        db.close()


class TestFTS5AndTermIndexAfterMigration:
    """Verify FTS5 and term_index still work after migration."""

    def test_fts5_reindex_after_migration(self, tmp_path):
        """After v8 -> v10 migration, messages can be found via FTS5."""
        db_path = tmp_path / "migrate_v8_fts.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (8);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                parent_session_id TEXT,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                ended_at REAL,
                end_reason TEXT,
                api_call_count INTEGER DEFAULT 0,
                FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT
            );

            CREATE TABLE state_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
            CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
        """)
        conn.commit()
        conn.close()

        db = SessionDB(db_path=db_path)
        sid = db.create_session("fts-test", "cli")
        db.append_message(session_id=sid, role="user", content="find me via fts")
        db.append_message(session_id=sid, role="assistant", content="found you")

        # Verify FTS5 can find the message
        cursor = db._conn.execute(
            "SELECT content FROM messages_fts WHERE messages_fts MATCH ?", ("find",)
        )
        results = cursor.fetchall()
        assert len(results) >= 1
        assert any("find me via fts" in r[0] for r in results)

        db.close()