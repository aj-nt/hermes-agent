"""EventBus and PipelineEvent for the Kore Orchestrator.

Replaces scattered _safe_print, _vprint, _emit_status, _emit_warning,
_fire_stream_delta, _fire_reasoning_delta, _fire_tool_gen_started calls
with a central, typed event system.

Every pipeline stage emits PipelineEvents; the display layer, session
logger, and rate limit tracker subscribe to the kinds they care about.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ============================================================================
# PipelineEvent
# ============================================================================

@dataclass
class PipelineEvent:
    """Typed event emitted during pipeline processing.

    Kind strings are convention-based. Core kinds:
        stream_delta, stream_end, tool_start, tool_end, tool_gen_started,
        error, status, warning, model_switch, auxiliary_failure,
        reasoning_delta, max_iterations, context_compressed,
        memory_nudge, skill_nudge, session_start, session_end
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# EventBus
# ============================================================================

class EventBus:
    """Central event bus for pipeline observability.

    Subscribers register for event kinds. Emitters fire events.
    Exceptions in subscribers are swallowed — the bus never breaks
    the pipeline.

    Thread safety: subscribe/unsubscribe should only happen at init.
    emit() can be called from any thread.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[PipelineEvent], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, kind: str, handler: Callable[[PipelineEvent], None]) -> None:
        """Register a handler for events of the given kind.

        Should only be called during setup, not mid-pipeline.
        """
        with self._lock:
            if kind not in self._subscribers:
                self._subscribers[kind] = []
            self._subscribers[kind].append(handler)

    def emit(self, event: PipelineEvent) -> None:
        """Fire-and-forget event to all subscribers of that kind.

        Exceptions in subscribers are swallowed so the pipeline
        never breaks due to a display or logging error.
        """
        handlers = self._subscribers.get(event.kind, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # Swallow — the bus is structural, not critical path
                pass