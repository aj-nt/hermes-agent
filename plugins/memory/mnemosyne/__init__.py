"""Mnemosyne memory provider for Hermes Agent.

Connects Hermes to a Mnemosyne memory daemon via gRPC.
Activate by setting ``memory.provider: mnemosyne`` in config.

Requires: ``mnemosyne-sdk`` installed in Hermes' venv.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# --- Tool schemas exposed to the model ---

_MEMORY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "Persistent memory for facts, preferences, and lessons. "
            "Two stores: 'memory' (your notes) and 'user' (what you know about the user). "
            "Connected to Mnemosyne memory daemon — memories persist across all agents and sessions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove", "search", "consolidate"],
                    "description": "What to do with memory."
                },
                "target": {
                    "type": "string",
                    "enum": ["memory", "user"],
                    "description": "Which store: 'memory' (your notes) or 'user' (user profile)."
                },
                "category": {
                    "type": "string",
                    "enum": ["user", "environment", "quirk", "project", "observation"],
                    "description": "Category for the memory entry."
                },
                "key": {
                    "type": "string",
                    "description": "Unique key for upsert. Use descriptive, reusable keys."
                },
                "content": {
                    "type": "string",
                    "description": "The memory content."
                },
                "priority": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Priority 1-5. Higher = more likely in context window."
                },
                "query": {
                    "type": "string",
                    "description": "Search query for 'search' action."
                },
                "old_text": {
                    "type": "string",
                    "description": "Unique substring of the entry to replace/remove."
                },
            },
            "required": ["action", "target"],
        },
    },
}


class MnemosyneProvider(MemoryProvider):
    """Memory provider backed by Mnemosyne daemon."""

    @property
    def name(self) -> str:
        return "mnemosyne"

    def __init__(self):
        self._client = None
        self._agent_id = "hermes"
        self._session_id = None

    def is_available(self) -> bool:
        """Check if mnemosyne SDK is importable."""
        try:
            import mnemosyne  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Connect to the Mnemosyne daemon and register this agent."""
        self._session_id = session_id
        self._agent_id = kwargs.get("agent_identity", "hermes")

        try:
            from mnemosyne import MnemosyneClient

            daemon_addr = kwargs.get("mnemosyne_address", "localhost:50051")
            self._client = MnemosyneClient(daemon_addr)

            # Register agent (synchronous init — connect happens lazily)
            import asyncio
            async def _register():
                await self._client.connect()
                await self._client.register_agent(self._agent_id, "Hermes", "assistant")
                await self._client.create_session(
                    self._session_id or "unknown",
                    source=kwargs.get("platform", "cli"),
                    title=None,
                )

            try:
                asyncio.get_event_loop().run_until_complete(_register())
            except RuntimeError:
                asyncio.run(_register())

            logger.info(f"Mnemosyne: connected as {self._agent_id}, session {self._session_id}")

        except Exception as e:
            logger.warning(f"Mnemosyne: unavailable ({e}). Memory disabled.")
            self._client = None

    def system_prompt_block(self) -> str:
        """Return the hot block from Mnemosyne daemon."""
        if not self._client:
            return ""

        try:
            import asyncio

            async def _fetch():
                mem = await self._client.get_hot_block(target="memory")
                user = await self._client.get_hot_block(target="user")
                parts = []
                if mem and mem.content:
                    parts.append(mem.content)
                if user and user.content:
                    parts.append(user.content)
                return "\n".join(parts)

            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_fetch())
            except RuntimeError:
                return asyncio.run(_fetch())

        except Exception as e:
            logger.debug(f"Mnemosyne hot block: {e}")
            return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Expose memory tool backed by Mnemosyne."""
        if not self._client:
            return []
        return [_MEMORY_TOOL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory tool calls by delegating to Mnemosyne daemon."""
        if tool_name != "memory" or not self._client:
            return {"error": "Mnemosyne memory provider not available"}

        action = args.get("action", "")
        target = args.get("target", "memory")
        category = args.get("category", "observation")
        key = args.get("key", "")
        content = args.get("content", "")
        priority = args.get("priority", 3)
        query = args.get("query", "")
        old_text = args.get("old_text", "")

        try:
            import asyncio

            async def _dispatch():
                if action == "add":
                    mem = await self._client.add_memory(
                        key=key, content=content, target=target,
                        category=category, priority=priority,
                    )
                    return {"result": f"Memory added: {mem.key}", "memory": mem.dict() if hasattr(mem, 'dict') else str(mem)}

                elif action == "replace":
                    # Search for the old entry first, then replace
                    results = await self._client.search_memories(old_text, target=target, limit=5)
                    if not results:
                        return {"error": f"No memory found matching '{old_text}'"}
                    old = results[0]
                    mem = await self._client.replace_memory(
                        memory_id=old.id, content=content, key=key,
                        priority=priority,
                    )
                    return {"result": f"Memory replaced: {mem.key}"}

                elif action == "remove":
                    results = await self._client.search_memories(old_text, target=target, limit=5)
                    if not results:
                        return {"error": f"No memory found matching '{old_text}'"}
                    await self._client.remove_memory(results[0].id)
                    return {"result": f"Memory removed: {results[0].key}"}

                elif action == "search":
                    results = await self._client.search_memories(query, target=target, limit=10)
                    if not results:
                        return {"result": f"No memories found for '{query}'"}
                    lines = []
                    for m in results:
                        lines.append(f"[{m.category}] {m.key}: {m.content[:200]}")
                    return {"result": "\n".join(lines)}

                elif action == "consolidate":
                    result = await self._client.consolidate()
                    return {"result": f"Consolidation complete. Auto-deleted: {result.auto_deleted}"}

                else:
                    return {"error": f"Unknown action: {action}"}

            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_dispatch())
            except RuntimeError:
                return asyncio.run(_dispatch())

        except Exception as e:
            logger.error(f"Mnemosyne tool error: {e}")
            return {"error": str(e)}

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Record the turn in Mnemosyne daemon."""
        sid = session_id or self._session_id
        if not self._client or not sid:
            return

        try:
            import asyncio

            async def _sync():
                await self._client.add_messages(
                    sid,
                    messages=[
                        {"role": "user", "content": user_content[:2000] if user_content else ""},
                        {"role": "assistant", "content": assistant_content[:4000] if assistant_content else ""},
                    ],
                )

            try:
                asyncio.get_event_loop().run_until_complete(_sync())
            except RuntimeError:
                asyncio.run(_sync())

        except Exception as e:
            logger.debug(f"Mnemosyne sync_turn failed: {e}")

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Any = None,
    ) -> None:
        """Mirror built-in memory writes to the Mnemosyne daemon."""
        if not self._client or not content:
            return

        try:
            import asyncio

            async def _mirror():
                if action == "add":
                    # Generate a key from the first ~80 chars of content
                    key = content[:80].strip().replace("\n", " ")
                    category = "user" if target == "user" else "observation"
                    await self._client.add_memory(
                        key=key,
                        content=content,
                        target=target,
                        category=category,
                        priority=3,
                    )
                elif action == "replace":
                    # Search for the old entry first (metadata may help), fall back to content
                    old = metadata.get("old_text", "") if isinstance(metadata, dict) else ""
                    query = old or content[:200]
                    results = await self._client.search_memories(query, target=target, limit=5)
                    if results:
                        await self._client.replace_memory(
                            memory_id=results[0].id,
                            content=content,
                            key=content[:80].strip().replace("\n", " "),
                            priority=3,
                        )

            try:
                asyncio.get_event_loop().run_until_complete(_mirror())
            except RuntimeError:
                asyncio.run(_mirror())

        except Exception as e:
            logger.debug(f"Mnemosyne on_memory_write failed: {e}")

    def shutdown(self) -> None:
        """Close the daemon connection."""
        if self._client:
            try:
                import asyncio
                try:
                    asyncio.get_event_loop().run_until_complete(self._client.close())
                except RuntimeError:
                    asyncio.run(self._client.close())
            except Exception:
                pass
            self._client = None
