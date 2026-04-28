#!/usr/bin/env python3
"""
Memory Tool Module — SQLite-Backed Four-Layer Persistent Memory

Architecture:
  Layer 1 (HOT):  ~3KB injected into system prompt every turn.
                    Selected from warm store by priority and recency.
  Layer 2 (WARM): SQLite `memories` table in state.db — unlimited, structured,
                    searchable via FTS5. All reads and writes hit this layer.
  Layer 3 (COLD):  Session transcripts in state.db (sessions + messages tables).
                    Searched via session_search tool. No changes needed.
  Layer 4 (RECENT CONTEXT): Auto-injected probe from recent session metadata.
                    Built at system-prompt-build time from sessions table.
                    Crash-safe — requires no agent action, always present
                    at session start so the agent knows what it was just
                    working on.

Categories:
  user        → preferences, identity, boundaries (never auto-evicted)
  environment → OS, paths, versions (evicted when replaced by newer value)
  quirk       → tool bugs, workarounds (evicted after 60d inactivity, low access)
  project     → project state, test counts (upsert by key — one entry per project)
  observation → transient findings (TTL-based, default 30-day expiry)

Priority 1-5:
  5 = sacred (always in hot), 4 = high, 3 = normal (default), 2 = low, 1 = archival

Key-based upserts:
  Same (target, key) pair replaces the old entry. No duplicate accumulation.

Backward compatibility:
  - format_for_system_prompt() returns same ══/§ format as before
  - Tool API still accepts action=add/replace/remove with old params
  - New actions: search, consolidate
  - New params: category, key, priority (optional, sensible defaults)
  - Auto-migrates MEMORY.md/USER.md entries on first load if table is empty
"""

import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTRY_DELIMITER = "\n§\n"

# Default char budgets for hot memory injection
DEFAULT_MEMORY_CHAR_LIMIT = 3000
DEFAULT_USER_CHAR_LIMIT = 1500


def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

# Category defaults for auto-classification and new entries
DEFAULT_CATEGORY = "environment"
DEFAULT_PRIORITY = 3

# TTL for observation category (seconds) — 30 days
OBSERVATION_TTL = 30 * 24 * 3600

# Inactivity threshold for quirk eviction (seconds) — 60 days
QUIRK_INACTIVITY_THRESHOLD = 60 * 24 * 3600
QUIRK_MIN_ACCESS_FOR_KEEP = 2

# Hot memory: minimum priority to be considered for injection
HOT_MIN_PRIORITY = 3

# ---------------------------------------------------------------------------
# Security scanning — injection/exfiltration detection
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    (r'curl\s+[^\n]*\${?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\${?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|~/\.hermes/\.env', "hermes_env"),
]

_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfiltration patterns. Returns error string if blocked."""
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."
    return None


# ---------------------------------------------------------------------------
# Heuristic classification for migrating flat-file entries
# ---------------------------------------------------------------------------

# Keywords that suggest a category
_CATEGORY_KEYWORDS = {
    "user": [
        "prefers", "goes by", "born", "wife", "husband", "partner", "phone",
        "approves", "values", "style", "stance", "sober",
    ],
    "environment": [
        "runs ollama", "always-on", "mac mini", "mac studio", "python",
        "venv", "homebrew", "launchd", "hostname", "ssh key", "pat at",
        "gh_token", "macbook", "brew", "/usr/bin", "/opt/homebrew",
    ],
    "quirk": [
        "corrupts", "bug", "broken", "workaround", "gotcha", "doesn't work",
        "redact", "rabbit hole", "hangs", "fails", "permission denied",
        "ci gotchas", "dead end", "truncation", "heuristic",
    ],
    "project": [
        "kore", "dogfood", "fork", "upstream", "test count", "tests green",
        "cherry-pick", "pr #", "glassknife", "hermes-agent",
    ],
}

# Keywords that suggest priority
_PRIORITY_BOOSTERS = {
    5: ["sober", "never mention", "boundary"],
    4: ["prefers", "goes by", "born", "phone", "wife", "approves", "values"],
}


def _classify_entry(content: str) -> Tuple[str, str, int]:
    """Classify a flat-file memory entry into (key, category, priority).
    
    Uses keyword heuristics. Falls back to environment/3 for unrecognized entries.
    """
    lower = content.lower()
    
    # Determine category
    scores = {cat: 0 for cat in _CATEGORY_KEYWORDS}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[cat] += 1
    
    # Special: if content mentions a person's identity/preferences strongly → user
    user_identity_markers = ["goes by", "born", "sober since", "prefers python"]
    if any(m in lower for m in user_identity_markers):
        scores["user"] += 3
    
    best_cat = max(scores, key=lambda k: scores[k])
    if scores[best_cat] == 0:
        best_cat = DEFAULT_CATEGORY
    
    # Determine priority
    priority = DEFAULT_PRIORITY
    for pri, keywords in _PRIORITY_BOOSTERS.items():
        if any(kw in lower for kw in keywords):
            priority = max(priority, pri)
    
    # Generate a key from the first meaningful words
    # Take first 3-5 words, lowercase, joined by underscores
    words = re.sub(r'[^a-zA-Z0-9\s]', '', content.split('.')[0].split(',')[0]).split()[:4]
    key = "_".join(w.lower() for w in words) if words else f"migrated_{int(time.time())}"
    
    return key, best_cat, priority


# ---------------------------------------------------------------------------
# MemoryStore — SQLite-backed persistent memory
# ---------------------------------------------------------------------------

class MemoryStore:
    """
    SQLite-backed persistent memory with hot/warm/cold layers.
    
    Layer 1 (HOT): ~3KB injected into system prompt, selected from warm store.
    Layer 2 (WARM): All entries in SQLite `memories` table — search, add, replace, remove.
    Layer 3 (COLD): Session transcripts (already exist, no changes).
    
    Maintains backward compatibility with the flat-file MemoryStore API:
    - format_for_system_prompt() returns same format
    - load_from_disk() triggers migration if needed
    - add/replace/remove still work with old params (target, content, old_text)
    """
    
    def __init__(self, db=None, memory_char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
                 user_char_limit: int = DEFAULT_USER_CHAR_LIMIT):
        """Initialize with a HermesState/SessionDB instance or sqlite3.Connection.
        
        Args:
            db: A SessionDB instance (preferred), sqlite3.Connection, or None.
                If None, creates an in-memory SQLite database for testing/isolation.
            memory_char_limit: Max chars for hot memory block.
            user_char_limit: Max chars for hot user block.
        """
        if db is None:
            # Create an in-memory SQLite database for testing/isolation
            self._db = sqlite3.connect(":memory:")
            self._db.row_factory = sqlite3.Row
            # Create the memories schema
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
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
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content, content='memories', content_rowid='id'
                );
                CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
                    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
                END;
            """)
            self._db.commit()
        else:
            self._db = db
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._migrated = False
        
        # Frozen snapshot for system prompt — set at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": "", "recent_context": ""}
    
    # -- Private: database access --
    
    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL write statement. Handles SessionDB or raw connection."""
        if hasattr(self._db, '_execute_write'):
            # SessionDB with retry logic — wrap in a callable
            def _do(conn):
                cursor = conn.cursor()
                cursor.execute(sql, params)
                return cursor
            return self._db._execute_write(_do)
        # Direct sqlite3.Connection (including in-memory)
        conn = self._db
        if hasattr(self._db, '_conn'):
            conn = self._db._conn
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        return cursor
    
    def _query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT and return all rows."""
        if hasattr(self._db, '_conn'):
            conn = self._db._conn
        else:
            conn = self._db
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute a SELECT and return one row."""
        rows = self._query(sql, params)
        return rows[0] if rows else None
    
    # -- Migration from flat files --
    
    def _migrate_from_files(self):
        """Migrate entries from MEMORY.md/USER.md to SQLite memories table.
        
        Uses INSERT OR IGNORE so partially-migrated tables are handled correctly —
        existing rows (by unique key) are skipped, not duplicated.
        Only runs once: after successful migration, a flag is written to state_meta
        so deleted entries don't reappear on future sessions.
        """
        if self._migrated:
            return
        
        # If the table doesn't exist yet (pre-v10 DB), skip silently
        try:
            self._query_one("SELECT id FROM memories LIMIT 1")
        except Exception:
            self._migrated = True
            return
        
        # Check if migration already ran (persisted in state_meta)
        try:
            row = self._query_one(
                "SELECT value FROM state_meta WHERE key = 'memories_migrated'"
            )
            if row and row[0] == "1":
                self._migrated = True
                return
        except Exception:
            pass  # state_meta might not exist yet — proceed with migration
        
        mem_dir = get_memory_dir()
        migrated_count = 0
        
        for target, filename in [("memory", "MEMORY.md"), ("user", "USER.md")]:
            filepath = mem_dir / filename
            if not filepath.exists():
                continue
            entries = self._read_file(filepath)
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
                key, category, priority = _classify_entry(entry)
                try:
                    self._execute(
                        "INSERT OR IGNORE INTO memories (target, category, key, content, priority) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (target, category, key, entry, priority)
                    )
                    migrated_count += 1
                except Exception as e:
                    logger.warning(f"Migration: failed to insert entry '{entry[:50]}...': {e}")
        
        # Mark migration as complete in state_meta so it never re-runs
        try:
            self._execute(
                "INSERT OR REPLACE INTO state_meta (key, value) VALUES ('memories_migrated', '1')"
            )
        except Exception:
            pass  # state_meta table might not exist — not critical
        
        self._migrated = True
        logger.info(f"Migrated {migrated_count} flat-file memory entries to SQLite")
    
    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a flat memory file and split into entries (backward compat)."""
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []
        if not raw.strip():
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]
    
    # -- Public API --
    
    def load_from_disk(self, current_session_id: str = ""):
        """Load memories from SQLite and capture hot memory snapshot.
        
        If memories table is empty, migrates from flat files first.
        Then builds the system prompt snapshot from hot entries.
        Also builds the recent-context block from previous sessions.
        
        Args:
            current_session_id: Current session ID, used to exclude it from
                the recent-context block. Defaults to "" (no exclusion).
        """
        self._migrate_from_files()
        self._system_prompt_snapshot = {
            "memory": self._build_hot_block("memory"),
            "user": self._build_hot_block("user"),
            "recent_context": self.build_recent_context_block(current_session_id=current_session_id),
        }
    
    def save_to_disk(self, target: str):
        """No-op for backward compatibility. SQLite auto-persists.
        
        The flat-file MemoryStore required explicit saves; SQLite commits
        after every operation. Keeping this method so run_agent.py calls
        don't break.
        """
        pass
    
    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """Return the frozen hot memory snapshot for system prompt injection.
        
        Returns None if the snapshot is empty for that target.
        Mid-session writes do NOT affect this — preserves prefix cache.
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None
    
    # ---- Recent context probe (session-start injection) ----
    
    # Maximum chars for the recent-context block (excluding header/separator)
    RECENT_CONTEXT_CHAR_LIMIT = 500
    
    def build_recent_context_block(self, current_session_id: str = "") -> str:
        """Build a compact recent-context block from previous sessions.
        
        Queries the sessions/messages tables for the 2 most recent sessions
        that (a) are not the current session and (b) have at least one message.
        Returns a formatted block suitable for system prompt injection,
        or empty string if no suitable sessions exist or the tables
        are unavailable.
        
        This is called at system prompt build time (session start) so the
        agent automatically knows what it was just working on, without
        needing to call session_search manually.
        
        Args:
            current_session_id: The current session's ID to exclude from results.
        """
        try:
            conn = self._db._conn if hasattr(self._db, '_conn') else self._db
            cursor = conn.cursor()
            
            # Check that the sessions table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            if not cursor.fetchone():
                return ""
            
            # Check that the messages table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            )
            if not cursor.fetchone():
                return ""
            
            # Query up to 4 recent sessions (fetch extra so we can filter
            # the current session and empty sessions), then take top 2.
            # Exclude child sessions (parent_session_id IS NULL).
            # Exclude the current session.
            # Exclude cron sessions — they're noise in recent context.
            # Require at least 1 message (via subquery).
            query = """
                SELECT s.id, s.title, s.started_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS msg_count,
                       COALESCE(
                           (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 60)
                            FROM messages m
                            WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                            ORDER BY m.timestamp, m.id LIMIT 1),
                           ''
                       ) AS preview
                FROM sessions s
                WHERE s.parent_session_id IS NULL
                  AND s.id != ?
                  AND s.source != 'cron'
                  AND (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) > 0
                ORDER BY s.started_at DESC
                LIMIT 2
            """
            cursor.execute(query, (current_session_id,))
            rows = cursor.fetchall()
            
            if not rows:
                return ""
            
            entries = []
            for row in rows:
                session_id, title, started_at, msg_count, preview = (
                    row[0], row[1], row[2], row[3], row[4]
                )
                
                # Format timestamp
                if started_at:
                    try:
                        ts = datetime.fromtimestamp(started_at, tz=timezone.utc)
                        time_str = ts.strftime("%b %d %H:%M")
                    except (OSError, ValueError):
                        time_str = ""
                else:
                    time_str = ""
                
                # Use title if available, otherwise preview
                label = title.strip() if title else (preview.strip() if preview else "Untitled")
                
                entry = f"- [{time_str}] {label} ({msg_count} msgs)"
                entries.append(entry)
            
            if not entries:
                return ""
            
            content = "\n".join(entries)
            
            # Truncate to budget
            if len(content) > self.RECENT_CONTEXT_CHAR_LIMIT:
                content = content[:self.RECENT_CONTEXT_CHAR_LIMIT - 3] + "..."
            
            header = "RECENT CONTEXT (auto-injected from last sessions)"
            separator = "═" * 46
            return f"{separator}\n{header}\n{separator}\n{content}"
        
        except Exception:
            # Graceful degradation — never break the agent
            return ""
    
    # ---- Core operations ----
    
    def add(self, target: str, content: str, category: str = None,
            key: str = None, priority: int = None,
            source_session: str = None, expires_at: float = None) -> Dict[str, Any]:
        """Add or upsert a memory entry.
        
        If key is provided and (target, key) already exists, replaces the entry.
        If key is None, auto-generates one from content.
        
        Args:
            target: 'memory' or 'user'
            content: The memory content text
            category: One of user/environment/quirk/project/observation (default: environment)
            key: Unique identifier within target. If matches existing, replaces.
            priority: 1-5 (default: 3)
            source_session: Session ID that created this entry
            expires_at: Unix timestamp when this entry should be auto-evicted (None = never)
        """
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        
        if target not in ("memory", "user"):
            return {"success": False, "error": f"Invalid target '{target}'. Use 'memory' or 'user'."}
        
        if category and category not in ("user", "environment", "quirk", "project", "observation"):
            return {"success": False, "error": f"Invalid category '{category}'. Use: user, environment, quirk, project, observation."}
        
        # Security scan
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}
        
        if category is None:
            category = DEFAULT_CATEGORY
        if priority is None:
            priority = DEFAULT_PRIORITY
        if not (1 <= priority <= 5):
            return {"success": False, "error": f"Priority must be 1-5, got {priority}."}
        
        # Auto-generate key from content if not provided
        if key is None:
            words = re.sub(r'[^a-zA-Z0-9\s]', '', content.split('.')[0].split(',')[0]).split()[:4]
            key = "_".join(w.lower() for w in words)
            if not key:
                key = f"entry_{int(time.time())}"
        
        now = time.time()
        
        try:
            # UPSERT: update if key exists, insert otherwise
            self._execute(
                """INSERT INTO memories (target, category, key, content, priority, 
                   created_at, updated_at, last_accessed, source_session, access_count, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                   ON CONFLICT(target, key) DO UPDATE SET
                   content = excluded.content,
                   category = excluded.category,
                   priority = excluded.priority,
                   updated_at = excluded.updated_at,
                   last_accessed = excluded.updated_at,
                   source_session = excluded.source_session,
                   expires_at = excluded.expires_at
                """,
                (target, category, key, content, priority, now, now, now, source_session, expires_at)
            )
        except Exception as e:
            logger.error(f"Memory add failed: {e}")
            return {"success": False, "error": f"Database error: {e}"}
        
        return self._success_response(target, "Entry added.")
    
    def replace(self, target: str, old_text: str = None, content: str = None,
                key: str = None, category: str = None,
                priority: int = None) -> Dict[str, Any]:
        """Replace an entry by substring match or by (target, key).
        
        Backward-compatible: old_text still works. New: key-based replacement.
        At least one of old_text or key must be provided.
        """
        if not old_text and not key:
            return {"success": False, "error": "old_text or key is required for replace."}
        if not content:
            return {"success": False, "error": "content is required for replace."}
        
        content = content.strip()
        
        if target not in ("memory", "user"):
            return {"success": False, "error": f"Invalid target '{target}'. Use 'memory' or 'user'."}
        
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}
        
        now = time.time()
        
        if key:
            # Key-based replacement: find by (target, key) and update
            existing = self._query_one(
                "SELECT id FROM memories WHERE target = ? AND key = ?",
                (target, key)
            )
            if existing:
                # Update existing entry
                updates = ["content = ?", "updated_at = ?", "last_accessed = ?"]
                params = [content, now, now]
                if category:
                    updates.append("category = ?")
                    params.append(category)
                if priority is not None:
                    updates.append("priority = ?")
                    params.append(priority)
                params.extend([target, key])
                self._execute(
                    f"UPDATE memories SET {', '.join(updates)} WHERE target = ? AND key = ?",
                    tuple(params)
                )
                return self._success_response(target, "Entry replaced via key.")
            else:
                # Key doesn't exist — insert instead
                cat = category or DEFAULT_CATEGORY
                pri = priority or DEFAULT_PRIORITY
                return self.add(target, content, category=cat, key=key, priority=pri)
        
        # old_text-based replacement (backward compat)
        old_text = old_text.strip()
        rows = self._query(
            "SELECT id, content FROM memories WHERE target = ?",
            (target,)
        )
        matches = [(r[0], r[1]) for r in rows if old_text in r[1]]
        
        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}
        
        if len(matches) > 1:
            unique_texts = set(r[1] for r in matches)
            if len(unique_texts) > 1:
                previews = [r[1][:80] + ("..." if len(r[1]) > 80 else "") for r in matches]
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific or use key-based replace.",
                    "matches": previews,
                }
        
        entry_id = matches[0][0]
        updates = ["content = ?", "updated_at = ?", "last_accessed = ?"]
        params = [content, now, now]
        if category:
            updates.append("category = ?")
            params.append(category)
        if priority is not None:
            updates.append("priority = ?")
            params.append(priority)
        params.append(entry_id)
        
        self._execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
            tuple(params)
        )
        return self._success_response(target, "Entry replaced.")
    
    def remove(self, target: str, old_text: str = None, key: str = None) -> Dict[str, Any]:
        """Remove an entry by substring match or by (target, key)."""
        if not old_text and not key:
            return {"success": False, "error": "old_text or key is required for remove."}
        
        if target not in ("memory", "user"):
            return {"success": False, "error": f"Invalid target '{target}'. Use 'memory' or 'user'."}
        
        if key:
            self._execute(
                "DELETE FROM memories WHERE target = ? AND key = ?",
                (target, key)
            )
            return self._success_response(target, "Entry removed via key.")
        
        # old_text-based removal (backward compat)
        old_text = old_text.strip()
        rows = self._query(
            "SELECT id, content FROM memories WHERE target = ?",
            (target,)
        )
        matches = [(r[0], r[1]) for r in rows if old_text in r[1]]
        
        if not matches:
            return {"success": False, "error": f"No entry matched '{old_text}'."}
        
        if len(matches) > 1:
            unique_texts = set(r[1] for r in matches)
            if len(unique_texts) > 1:
                previews = [r[1][:80] + ("..." if len(r[1]) > 80 else "") for r in matches]
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific or use key-based remove.",
                    "matches": previews,
                }
        
        entry_id = matches[0][0]
        self._execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        return self._success_response(target, "Entry removed.")
    
    def search(self, query: str, target: str = None, category: str = None,
               limit: int = 10) -> Dict[str, Any]:
        """FTS5 search across memory entries.
        
        Args:
            query: Search term(s)
            target: Filter by 'memory' or 'user' (None = both)
            category: Filter by category (None = all)
            limit: Max results to return
        """
        try:
            # Use FTS5 for content search
            # Sanitize query: FTS5 interprets 'word:...' as column filters.
            # Since our FTS table only has the 'content' column, any colonated
            # term (layer:4, agent:primary, http://...) triggers "no such column".
            # Wrapping in double quotes forces FTS5 phrase-match mode.
            # Escape any existing double quotes in the query first.
            safe_query = '"' + query.replace('"', '""') + '"'
            sql = """
                SELECT m.id, m.target, m.category, m.key, m.content, m.priority,
                       m.created_at, m.updated_at, m.last_accessed, m.access_count
                FROM memories m
                JOIN memories_fts fts ON m.id = fts.rowid
                WHERE memories_fts MATCH ?
            """
            params: list = [safe_query]
            
            if target:
                sql += " AND m.target = ?"
                params.append(target)
            if category:
                sql += " AND m.category = ?"
                params.append(category)
            
            sql += " ORDER BY m.priority DESC, m.last_accessed DESC LIMIT ?"
            params.append(limit)
            
            rows = self._query(sql, tuple(params))
            
            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "target": r[1],
                    "category": r[2],
                    "key": r[3],
                    "content": r[4],
                    "priority": r[5],
                    "last_accessed": r[8],
                    "access_count": r[9],
                })
            
            # FTS5 phrase matching is exact — "layer:4" won't match "Layer 4".
            # If FTS5 returns nothing, fall back to LIKE for partial matching.
            if not results and query:
                like_sql = """
                    SELECT id, target, category, key, content, priority,
                           created_at, updated_at, last_accessed, access_count
                    FROM memories
                    WHERE content LIKE ?
                """
                like_params: list = [f"%{query}%"]
                if target:
                    like_sql += " AND target = ?"
                    like_params.append(target)
                if category:
                    like_sql += " AND category = ?"
                    like_params.append(category)
                like_sql += " ORDER BY priority DESC, last_accessed DESC LIMIT ?"
                like_params.append(limit)
                
                like_rows = self._query(like_sql, tuple(like_params))
                for r in like_rows:
                    results.append({
                        "id": r[0],
                        "target": r[1],
                        "category": r[2],
                        "key": r[3],
                        "content": r[4],
                        "priority": r[5],
                        "last_accessed": r[8],
                        "access_count": r[9],
                    })
            
            # Touch access counts for returned entries
            if results:
                ids = ",".join(str(r["id"]) for r in results)
                now = time.time()
                self._execute(
                    f"UPDATE memories SET access_count = access_count + 1, "
                    f"last_accessed = ? WHERE id IN ({ids})",
                    (now,)
                )
            
            return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {"success": False, "error": f"Search error: {e}"}
    
    def consolidate(self, force: bool = False) -> Dict[str, Any]:
        """Run consolidation: expire observations, evict stale quirks.
        
        Args:
            force: If True, run even if not due (normally runs at session end)
        
        Returns dict with stats about what was done.
        """
        now = time.time()
        stats = {"expired": 0, "evicted_quirks": 0}
        
        # 1. Expire observations past their TTL
        try:
            self._execute(
                "DELETE FROM memories WHERE category = 'observation' "
                "AND expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
            stats["expired"] = 1  # SQLite doesn't return rowcount easily
        except Exception as e:
            logger.warning(f"Consolidation: expire failed: {e}")
        
        # 2. Evict stale quirks (inactive > 60 days, accessed < 2 times)
        #    But only if we have more than 50 entries total (don't evict when sparse)
        try:
            count_row = self._query_one("SELECT COUNT(*) as cnt FROM memories")
            total = count_row[0] if count_row else 0
            
            if total > 50:
                threshold = now - QUIRK_INACTIVITY_THRESHOLD
                self._execute(
                    "DELETE FROM memories WHERE category = 'quirk' "
                    "AND last_accessed < ? AND access_count < ?",
                    (threshold, QUIRK_MIN_ACCESS_FOR_KEEP)
                )
                stats["evicted_quirks"] = 1
        except Exception as e:
            logger.warning(f"Consolidation: quirk eviction failed: {e}")
        
        return {"success": True, "stats": stats}
    
    # ---- Hot memory selection ----
    
    def _build_hot_block(self, target: str) -> str:
        """Select entries for hot injection and render as a text block.
        
        Priority >= 3, ordered by priority DESC then last_accessed DESC,
        truncated to char limit.
        """
        try:
            rows = self._query(
                """SELECT category, key, content, priority FROM memories
                   WHERE target = ? AND priority >= ?
                   ORDER BY priority DESC, last_accessed DESC""",
                (target, HOT_MIN_PRIORITY)
            )
        except Exception:
            # Table might not exist yet
            return ""
        
        if not rows:
            return ""
        
        limit = self.memory_char_limit if target == "memory" else self.user_char_limit
        
        entries = []
        running = 0
        for r in rows:
            category, key, content, priority = r[0], r[1], r[2], r[3]
            # Render with category label
            rendered = f"[{category}] {content}" if category != DEFAULT_CATEGORY else content
            entry_len = len(rendered) + 1  # +1 for delimiter
            
            if running + entry_len > limit and entries:
                break  # Would exceed budget, stop
            
            entries.append(rendered)
            running += entry_len
        
        if not entries:
            return ""
        
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        
        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"
        
        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"
    
    # ---- Backward compat helpers ----
    
    def _entries_for(self, target: str) -> List[str]:
        """Get content list for backward compat (e.g., _success_response)."""
        try:
            rows = self._query(
                "SELECT content FROM memories WHERE target = ? ORDER BY priority DESC, last_accessed DESC",
                (target,)
            )
            return [r[0] for r in rows]
        except Exception:
            return []
    
    def _char_count(self, target: str) -> int:
        """Count total chars for a target (for backward compat)."""
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))
    
    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit
    
    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        
        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        return resp


# ---------------------------------------------------------------------------
# Tool dispatcher — backward-compatible with existing LLM calls
# ---------------------------------------------------------------------------

def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    category: str = None,
    key: str = None,
    priority: int = None,
    query: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Actions:
      add       — Add a new entry (use key for upsert, category/priority optional)
      replace   — Replace entry by old_text or key
      remove    — Remove entry by old_text or key
      search    — FTS5 search across memories (query param)
      consolidate — Run eviction/expiry (called at session end)

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in ("memory", "user"):
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content, category=category, key=key,
                          priority=priority)

    elif action == "replace":
        if not old_text and not key:
            return tool_error("old_text or key is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text=old_text, content=content,
                             key=key, category=category, priority=priority)

    elif action == "remove":
        if not old_text and not key:
            return tool_error("old_text or key is required for 'remove' action.", success=False)
        result = store.remove(target, old_text=old_text, key=key)

    elif action == "search":
        if not query:
            return tool_error("query is required for 'search' action.", success=False)
        result = store.search(query, target=target if target else None,
                            category=category)

    elif action == "consolidate":
        result = store.consolidate()

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove, search, consolidate", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema — updated with new params
# ---------------------------------------------------------------------------

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- You learn something about the environment (OS, paths, versions, quirks)\n"
        "- You discover a bug, workaround, or non-obvious behavior worth remembering\n"
        "- User shares a preference, habit, or personal detail\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts. "
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "CATEGORIES (use 'category' param):\n"
        "  user         → who the user is (name, preferences, boundaries). Priority 4, never auto-evicted.\n"
        "  environment  → OS, paths, versions, tool configs. Priority 3, replaced by newer values.\n"
        "  quirk        → tool bugs, workarounds, gotchas. Priority 3, auto-evicted after 60d inactivity.\n"
        "  project      → project state, test counts, status. Priority 3, upsert by key.\n"
        "  observation  → transient findings. Priority 2, auto-expires after 30 days.\n\n"
        "KEYS: Use keys for entries you'll update later (e.g., key='kore_test_count' for test stats). "
        "Same key replaces the old value (upsert).\n\n"
        "ACTIONS:\n"
        "  add         — Save a new memory entry. Use key for updatable entries.\n"
        "  replace     — Update an existing entry by old_text or key.\n"
        "  remove      — Delete an entry by old_text or key.\n"
        "  search      — FTS5 search across memories (query required).\n"
        "  consolidate — Run eviction/expiry (auto-called at session end).\n\n"
        "TWO TARGETS:\n"
        "  'user'    → who the user is — name, role, preferences, communication style, pet peeves\n"
        "  'memory'  → your notes — environment facts, project conventions, tool quirks, lessons learned\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "search", "consolidate"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove. Use 'key' for precise matching."
            },
            "key": {
                "type": "string",
                "description": "Unique identifier for the entry within its target. Same key replaces existing entries (upsert)."
            },
            "category": {
                "type": "string",
                "enum": ["user", "environment", "quirk", "project", "observation"],
                "description": "Entry category. Defaults to 'environment'. See description for semantics."
            },
            "priority": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Priority 1-5. Higher = more likely in system prompt. Default: 3 for environment/quirk/project, 4 for user."
            },
            "query": {
                "type": "string",
                "description": "Search query for 'search' action. FTS5 full-text search."
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        category=args.get("category"),
        key=args.get("key"),
        priority=args.get("priority"),
        query=args.get("query"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)