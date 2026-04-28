"""Tests for recent context injection — session-start probe.

When a new session starts, the agent should automatically know about
recent sessions without having to call session_search manually.

MemoryStore.build_recent_context_block() queries the session database
for the most recent sessions (excluding the current lineage) and
formats a compact block for system prompt injection.
"""

import sqlite3
import time
from pathlib import Path

import pytest

from tools.memory_tool import MemoryStore, DEFAULT_MEMORY_CHAR_LIMIT, DEFAULT_USER_CHAR_LIMIT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SESSIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        source TEXT DEFAULT '',
        model TEXT,
        title TEXT,
        started_at REAL,
        ended_at REAL,
        end_reason TEXT,
        parent_session_id TEXT,
        system_prompt TEXT,
        token_usage TEXT
    );

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT,
        timestamp REAL,
        tokens INTEGER DEFAULT 0,
        model TEXT,
        metadata TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
"""


@pytest.fixture
def full_db(tmp_path):
    """Create a temporary state.db with both memories AND sessions tables.

    This mirrors production where both tables live in the same database.
    Returns an open sqlite3.Connection.
    """
    db_file = tmp_path / "test_state.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (10);

        CREATE TABLE IF NOT EXISTS state_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target TEXT NOT NULL CHECK(target IN ('memory', 'user')),
            category TEXT NOT NULL CHECK(category IN (
                'user', 'environment', 'quirk', 'project', 'observation'
            )),
            key TEXT NOT NULL,
            content TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 3 CHECK(priority BETWEEN 1 AND 5),
            created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now')),
            last_accessed REAL NOT NULL DEFAULT (strftime('%s','now')),
            source_session TEXT,
            access_count INTEGER NOT NULL DEFAULT 0,
            expires_at REAL,
            UNIQUE(target, key)
        );

        CREATE INDEX idx_memories_target_cat ON memories(target, category);
        CREATE INDEX idx_memories_priority ON memories(priority DESC);

        CREATE VIRTUAL TABLE memories_fts USING fts5(
            content, content='memories', content_rowid='id'
        );

        CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
        END;

        CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
        END;

        CREATE TRIGGER memories_fts_update AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
            INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
        END;
    """)
    conn.executescript(SESSIONS_SCHEMA)
    conn.commit()
    return conn


@pytest.fixture
def store(full_db):
    """Create a MemoryStore backed by a full test database."""
    s = MemoryStore(db=full_db)
    s._migrated = True  # Skip migration in tests
    return s


def _insert_session(conn, session_id, title=None, started_at=None,
                     ended_at=None, end_reason=None, parent_session_id=None,
                     source="cli", model="test-model"):
    """Helper to insert a session row."""
    if started_at is None:
        started_at = time.time()
    conn.execute(
        "INSERT INTO sessions (id, source, model, title, started_at, ended_at, end_reason, parent_session_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, source, model, title, started_at, ended_at, end_reason, parent_session_id)
    )
    conn.commit()


def _insert_message(conn, session_id, role, content, timestamp=None):
    """Helper to insert a message row."""
    if timestamp is None:
        timestamp = time.time()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, timestamp)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Test: build_recent_context_block
# ---------------------------------------------------------------------------

class TestRecentContextBlock:
    """Test that MemoryStore can build a recent-context block from session data.

    The block should:
    1. Include the 1-2 most recent sessions (excluding current lineage)
    2. Show title (or preview if untitled), timestamp, message count
    3. Stay within a reasonable byte budget (~500 chars)
    4. Return empty string when no sessions exist
    5. Return empty string when sessions table is missing
    6. Exclude sessions with no messages (empty sessions)
    """

    def test_returns_empty_when_no_session_table(self, tmp_path):
        """When the sessions table doesn't exist, return empty string (graceful degradation)."""
        # Create a MemoryStore with only the memories table — no sessions table
        db_file = tmp_path / "minimal.db"
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 3,
                created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
                updated_at REAL NOT NULL DEFAULT (strftime('%s','now')),
                last_accessed REAL NOT NULL DEFAULT (strftime('%s','now')),
                source_session TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                expires_at REAL,
                UNIQUE(target, key)
            );
        """)
        conn.commit()
        store = MemoryStore(db=conn)
        store._migrated = True
        result = store.build_recent_context_block(current_session_id="test_123")
        assert result == ""

    def test_returns_empty_when_no_sessions(self, store, full_db):
        """When there are no sessions at all, return empty string."""
        result = store.build_recent_context_block(current_session_id="test_123")
        assert result == ""

    def test_returns_recent_session_with_title(self, store, full_db):
        """A titled recent session appears in the context block."""
        _insert_session(full_db, "sess_old1", title="Memory system bugs fixed",
                        started_at=time.time() - 300)
        _insert_message(full_db, "sess_old1", "user", "Let's fix the migration bugs",
                       timestamp=time.time() - 290)

        result = store.build_recent_context_block(current_session_id="sess_new")
        assert "Memory system bugs fixed" in result
        # Session IDs are internal — they should NOT be in the output
        assert "sess_old1" not in result

    def test_returns_recent_session_with_preview_when_untitled(self, store, full_db):
        """An untitled session shows the first user message as preview."""
        _insert_session(full_db, "sess_old1", title=None,
                        started_at=time.time() - 300)
        _insert_message(full_db, "sess_old1", "user", "check your session logs",
                       timestamp=time.time() - 290)

        result = store.build_recent_context_block(current_session_id="sess_new")
        assert "check your session logs" in result

    def test_excludes_current_session(self, store, full_db):
        """The current session should never appear in recent context."""
        _insert_session(full_db, "sess_new", title="Current session",
                        started_at=time.time())
        _insert_message(full_db, "sess_new", "user", "what's up",
                       timestamp=time.time())

        result = store.build_recent_context_block(current_session_id="sess_new")
        assert "Current session" not in result
        assert "what's up" not in result

    def test_excludes_empty_sessions(self, store, full_db):
        """Sessions with zero messages should not appear in recent context."""
        _insert_session(full_db, "sess_empty", title="Empty session",
                        started_at=time.time() - 300)
        # No messages inserted for sess_empty

        result = store.build_recent_context_block(current_session_id="sess_new")
        assert result == ""

    def test_limits_to_two_recent_sessions(self, store, full_db):
        """Only the 2 most recent sessions appear, even if more exist."""
        for i in range(5):
            ts = time.time() - (5 - i) * 300
            _insert_session(full_db, f"sess_{i}", title=f"Session {i}",
                            started_at=ts)
            _insert_message(full_db, f"sess_{i}", "user", f"msg {i}",
                           timestamp=ts + 10)

        result = store.build_recent_context_block(current_session_id="sess_new")
        # Should contain sessions 4 and 3 (most recent), but not 2, 1, 0
        assert "Session 4" in result
        assert "Session 3" in result
        assert "Session 2" not in result

    def test_stays_within_char_budget(self, store, full_db):
        """Output should stay within ~500 char budget."""
        _insert_session(full_db, "sess_1", title="A" * 300,
                        started_at=time.time() - 300)
        _insert_message(full_db, "sess_1", "user", "msg",
                       timestamp=time.time() - 290)

        result = store.build_recent_context_block(current_session_id="sess_new")
        # Even with a 300-char title, total should be capped
        assert len(result) <= 600  # 500 budget + some overhead for formatting

    def test_formats_with_header(self, store, full_db):
        """Output should have a header identifying it as recent context."""
        _insert_session(full_db, "sess_1", title="Recent work",
                        started_at=time.time() - 300)
        _insert_message(full_db, "sess_1", "user", "working on things",
                       timestamp=time.time() - 290)

        result = store.build_recent_context_block(current_session_id="sess_new")
        assert "RECENT CONTEXT" in result

    def test_shows_message_count(self, store, full_db):
        """Each session entry should show its message count."""
        _insert_session(full_db, "sess_1", title="Long session",
                        started_at=time.time() - 300)
        for i in range(5):
            _insert_message(full_db, "sess_1", "user", f"message {i}",
                           timestamp=time.time() - 290 + i)

        result = store.build_recent_context_block(current_session_id="sess_new")
        assert "5" in result  # message count displayed

    def test_excludes_cron_sessions(self, store, full_db):
        """Cron sessions should not appear in recent context — they're noise."""
        # Insert a cron session that's the most recent
        _insert_session(full_db, "cron_recent", title=None,
                        source="cron", started_at=time.time() - 60)
        _insert_message(full_db, "cron_recent", "user",
                        "scheduled health check", timestamp=time.time() - 55)

        # Insert a real CLI session that's older
        _insert_session(full_db, "cli_older", title="Kore refactor session",
                        source="cli", started_at=time.time() - 300)
        _insert_message(full_db, "cli_older", "user", "let's work on the pipeline",
                        timestamp=time.time() - 290)

        result = store.build_recent_context_block(current_session_id="sess_new")
        # The cron session should NOT appear
        assert "health check" not in result
        assert "scheduled" not in result
        # The real session SHOULD appear
        assert "Kore refactor session" in result

    def test_cron_sessions_never_appear_even_when_only_recent(self, store, full_db):
        """If only cron sessions exist, recent context should be empty, not show crons."""
        # Only cron sessions — no real user sessions
        _insert_session(full_db, "cron_1", title=None,
                        source="cron", started_at=time.time() - 60)
        _insert_message(full_db, "cron_1", "user", "cron task output",
                        timestamp=time.time() - 55)
        _insert_session(full_db, "cron_2", title=None,
                        source="cron", started_at=time.time() - 120)
        _insert_message(full_db, "cron_2", "user", "another cron job",
                        timestamp=time.time() - 115)

        result = store.build_recent_context_block(current_session_id="sess_new")
        assert result == ""