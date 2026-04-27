"""ToolExecutor: the Stage 4 tool dispatch engine.

Replaces the scattered _invoke_tool, _execute_tool_calls,
_execute_tool_calls_concurrent, and checkpoint interception logic
in run_agent.py (860+ lines combined).

ToolExecutor manages:
- Tool registration (name → callable)
- Pre-dispatch hooks (e.g., checkpointing for write operations)
- Argument parsing (JSON string or dict)
- Dispatch (sequential for now, concurrent later)
- Error handling (tool errors become error result messages)
- Post-dispatch callbacks (e.g., nudge counter reset, write metadata)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent.orchestrator.context import ParsedResponse
from agent.orchestrator.stages import StageAction, ToolDispatchResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Routes tool calls to registered handlers with hook support.

    Usage:
        executor = ToolExecutor()
        executor.register("read_file", lambda args: "file contents")
        result = executor.dispatch(parsed_response)

    Pre-dispatch hooks run before each tool call. They receive (name, args)
    and can raise to block execution, or just perform side effects (like
    checkpointing).

    Post-dispatch callbacks run after each tool call. They receive
    (name, args, result) and can perform bookkeeping (like nudge counter
    reset or write metadata attachment).
    """

    def __init__(
        self,
        pre_dispatch_hooks: Optional[Sequence[Callable[[str, dict], None]]] = None,
        post_dispatch_callbacks: Optional[Sequence[Callable[[str, dict, Any], None]]] = None,
    ) -> None:
        self._tools: Dict[str, Callable[[dict], Any]] = {}
        self._pre_hooks: List[Callable[[str, dict], None]] = list(pre_dispatch_hooks or [])
        self._post_callbacks: List[Callable[[str, dict, Any], None]] = list(post_dispatch_callbacks or [])

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def register(self, name: str, handler: Callable[[dict], Any]) -> None:
        """Register a tool handler by name. Overwrites existing."""
        self._tools[name] = handler

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def dispatch(self, parsed: ParsedResponse) -> ToolDispatchResult:
        """Route tool calls in ParsedResponse to registered handlers.

        Returns a ToolDispatchResult with:
        - tool_results: list of role="tool" messages
        - action: CONTINUE if tool calls were dispatched, YIELD if none
        """
        if not parsed.has_tool_calls:
            return ToolDispatchResult(
                tool_results=[],
                action=StageAction.YIELD,
            )

        tool_results: list[dict] = []
        for tc in parsed.tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            call_id = tc.get("id", "")
            args_raw = func.get("arguments", "{}")

            # Parse arguments
            args = self._parse_arguments(args_raw)

            # Run pre-dispatch hooks (non-blocking)
            self._run_pre_hooks(name, args)

            # Execute tool
            content = self._execute_tool(name, args)

            tool_results.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": content,
            })

            # Run post-dispatch callbacks
            self._run_post_callbacks(name, args, content)

        return ToolDispatchResult(
            tool_results=tool_results,
            action=StageAction.CONTINUE,
        )

    def _parse_arguments(self, args_raw: Any) -> dict:
        """Parse tool call arguments from JSON string or dict."""
        if isinstance(args_raw, dict):
            return args_raw
        if isinstance(args_raw, str):
            try:
                return json.loads(args_raw) if args_raw.strip() else {}
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Malformed tool arguments: {args_raw!r}, falling back to {{}}")
                return {}
        return {}

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a single tool call, returning the result as a string."""
        handler = self._tools.get(name)
        if handler is None:
            return f"Error: Unknown tool '{name}'"

        try:
            result = handler(args)
            if isinstance(result, dict):
                return json.dumps(result)
            return str(result)
        except Exception as exc:
            logger.warning(f"Tool '{name}' raised {exc}")
            return f"Error: {exc}"

    def _run_pre_hooks(self, name: str, args: dict) -> None:
        """Run pre-dispatch hooks. Exceptions are logged but don't block execution."""
        for hook in self._pre_hooks:
            try:
                hook(name, args)
            except Exception as exc:
                logger.warning(f"Pre-dispatch hook {hook.__name__} raised {exc}")

    def _run_post_callbacks(self, name: str, args: dict, result: str) -> None:
        """Run post-dispatch callbacks. Exceptions are logged but don't affect results."""
        for callback in self._post_callbacks:
            try:
                callback(name, args, result)
            except Exception as exc:
                logger.warning(f"Post-dispatch callback {callback.__name__} raised {exc}")