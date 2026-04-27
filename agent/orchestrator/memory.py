"""Memory coordinator and supporting types for the Kore Orchestrator.

Unifies built-in MemoryStore and external MemoryManager into a single
interface. All memory operations route through MemoryCoordinator.

Phase 1: Type definitions and shell implementations.
Phase 2: Wire up to existing MemoryStore and MemoryManager.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# NudgeTracker
# ============================================================================

class NudgeTracker:
    """Track turns since last memory/skill usage and produce nudge text.

    Extracts the ad-hoc _turns_since_memory and skill nudge counters
    from run_conversation() into a testable unit.
    """

    MEMORY_NUDGE_TEXT = (
        "Consider using the memory tool to save important information "
        "for future sessions."
    )
    SKILL_NUDGE_TEXT = (
        "Consider loading and using relevant skills for this task."
    )

    def __init__(
        self,
        nudge_interval: int = 10,
        skill_interval: int = 10,
    ) -> None:
        self._turns_since_memory = 0
        self._turns_since_skill = 0
        self._memory_interval = nudge_interval
        self._skill_interval = skill_interval

    def on_turn(self) -> None:
        """Increment counters each pipeline iteration."""
        self._turns_since_memory += 1
        self._turns_since_skill += 1

    def on_memory_use(self) -> None:
        """Reset memory nudge counter when memory tools are used."""
        self._turns_since_memory = 0

    def on_skill_use(self) -> None:
        """Reset skill nudge counter when skills are loaded/used."""
        self._turns_since_skill = 0

    def memory_nudge(self) -> Optional[str]:
        """Return nudge text if memory interval exceeded, else None."""
        if self._turns_since_memory >= self._memory_interval:
            return self.MEMORY_NUDGE_TEXT
        return None

    def skill_nudge(self) -> Optional[str]:
        """Return nudge text if skill interval exceeded, else None."""
        if self._turns_since_skill >= self._skill_interval:
            return self.SKILL_NUDGE_TEXT
        return None


# ============================================================================
# WriteMetadataTracker
# ============================================================================

class WriteMetadataTracker:
    """Track and attach write provenance to memory operations.

    Replaces _build_memory_write_metadata() scattered across run_agent.py.
    """

    def __init__(
        self,
        session_id: str = "",
        parent_session_id: str = "",
        platform: str = "cli",
    ) -> None:
        self._origin = "assistant_tool"
        self._context = "foreground"
        self._session_id = session_id
        self._parent_session_id = parent_session_id
        self._platform = platform

    def set_origin(self, origin: str, context: str) -> None:
        """Set provenance for subsequent memory writes."""
        self._origin = origin
        self._context = context

    def build_metadata(
        self,
        *,
        task_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Build provenance dict for memory writes.

        Empty strings are excluded to keep metadata clean.
        """
        metadata: dict[str, str] = {
            "write_origin": self._origin,
            "execution_context": self._context,
            "session_id": self._session_id,
            "platform": self._platform,
            "tool_name": "memory",
        }
        if self._parent_session_id:
            metadata["parent_session_id"] = self._parent_session_id
        if task_id:
            metadata["task_id"] = task_id
        if tool_call_id:
            metadata["tool_call_id"] = tool_call_id
        return metadata


# ============================================================================
# MemoryEvent
# ============================================================================

@dataclass
class MemoryEvent:
    """An event published to the MemoryBus.

    Events include writes, replacements, deletions, consolidation,
    compression, and session lifecycle.
    """

    kind: str
    source_session_id: str
    target: str  # "memory" or "user"
    key: Optional[str] = None
    content_preview: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# ReviewContext
# ============================================================================

@dataclass
class ReviewContext:
    """Snapshot of memory state for a background review agent.

    Decouples background review from AIAgent internals — no more
    copying _memory_store, _memory_enabled, etc.
    """

    memory_enabled: bool = False
    user_profile_enabled: bool = False
    store_config: Optional[dict] = None
    manager_config: Optional[dict] = None
    session_id: str = ""
    write_origin: str = "background_review"
    write_context: str = "background_review"


# ============================================================================
# MemoryCoordinator
# ============================================================================

class MemoryCoordinator:
    """Unified interface for all memory systems.

    Owns the lifecycle of both built-in MemoryStore and external
    MemoryManager providers. All memory operations route through here.

    Phase 1: Shell with NudgeTracker and WriteMetadataTracker wired.
    Phase 2: Connect to existing MemoryStore and MemoryManager.
    """

    def __init__(self, session_id: str = "", *, nudge_interval: int = 10, skill_interval: int = 10) -> None:
        self._session_id = session_id
        self.nudge_tracker: NudgeTracker = NudgeTracker(
            nudge_interval=nudge_interval, skill_interval=skill_interval
        )
        self.write_metadata: WriteMetadataTracker = WriteMetadataTracker(
            session_id=session_id
        )
        # Store and Manager wired in Phase 2
        self.store: Any = None
        self.manager: Any = None

    # --- Lifecycle (Phase 2 shells) ---

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Load built-in store, initialize external providers."""
        self._session_id = session_id

    def on_turn_start(self, turn: int, message: str) -> Optional[str]:
        """Notify all providers of turn start. Returns nudge text if due."""
        self.nudge_tracker.on_turn()
        memory_nudge = self.nudge_tracker.memory_nudge()
        skill_nudge = self.nudge_tracker.skill_nudge()
        # Return whichever nudge fires first (memory takes priority)
        if memory_nudge:
            return memory_nudge
        if skill_nudge:
            return skill_nudge
        return None

    def on_turn_end(
        self, user_msg: str, assistant_msg: str, *, interrupted: bool = False
    ) -> None:
        """Sync turn to external providers, prefetch for next turn."""
        if self.manager is not None and not interrupted and user_msg and assistant_msg:
            self.manager.sync_all(user_msg, assistant_msg)
            self.manager.queue_prefetch_all(user_msg)

    def on_session_end(self, messages: list) -> None:
        """Final extraction pass. Does NOT tear down providers."""
        if self.manager is not None:
            self.manager.on_session_end(messages or [])

    def shutdown(self, messages: Optional[list] = None) -> None:
        """End-of-session extraction + full provider teardown."""
        if self.manager is not None:
            self.manager.on_session_end(messages or [])
            self.manager.shutdown_all()

    # --- System Prompt ---

    def build_prompt_blocks(self) -> list[str]:
        """Return ordered prompt blocks for memory, user profile, external."""
        blocks: list[str] = []
        if self.store:
            if self._memory_enabled:
                mem_block = self.store.format_for_system_prompt("memory")
                if mem_block:
                    blocks.append(mem_block)
            if self._user_profile_enabled:
                user_block = self.store.format_for_system_prompt("user")
                if user_block:
                    blocks.append(user_block)
            # Recent context is always included when store exists
            recent_block = self.store.format_for_system_prompt("recent_context")
            if recent_block:
                blocks.append(recent_block)
        if self.manager:
            ext_block = self.manager.build_system_prompt()
            if ext_block:
                blocks.append(ext_block)
        return blocks

    # --- Tool Integration ---

    def get_tool_schemas(self) -> list[dict]:
        """Return all memory tool schemas (built-in + external)."""
        schemas: list[dict] = []
        if self.store is not None and self._memory_enabled:
            store_schema = self.store.get_tool_schema()
            if store_schema:
                schemas.append(store_schema)
        if self.manager is not None:
            for schema in self.manager.get_all_tool_schemas():
                schemas.append(schema)
        return schemas

    def handle_tool_call(
        self, tool_name: str, args: dict, *, metadata: dict
    ) -> str:
        """Route tool calls to built-in or external handler.
        
        Memory tool (add/replace/delete/search/consolidate) goes to store.
        External memory tools go to manager if manager.has_tool() returns True.
        """
        if tool_name == "memory" and self.store is not None:
            return self.store.handle_tool_call(tool_name, args, metadata)
        if self.manager is not None and self.manager.has_tool(tool_name):
            return self.manager.handle_tool_call(tool_name, args)
        raise ValueError(f"Unknown memory tool: {tool_name}")

    def should_nudge(self) -> Optional[str]:
        """Return nudge text if memory/skill nudge is due, else None."""
        return self.nudge_tracker.memory_nudge() or self.nudge_tracker.skill_nudge()

    def record_tool_use(self, tool_name: str) -> None:
        """Reset nudge counters when memory/skill tools are actually used."""
        if "memory" in tool_name:
            self.nudge_tracker.on_memory_use()
        if "skill" in tool_name:
            self.nudge_tracker.on_skill_use()

    # --- Background Review ---

    def build_review_context(self) -> ReviewContext:
        """Extract just what a background review agent needs."""
        return ReviewContext(
            memory_enabled=self.store is not None,
            user_profile_enabled=self.store is not None,
            session_id=self._session_id,
            write_origin="background_review",
            write_context="background_review",
        )

    # --- Write Provenance ---

    def set_write_origin(self, origin: str, context: str) -> None:
        """Set provenance for subsequent memory writes."""
        self.write_metadata.set_origin(origin, context)