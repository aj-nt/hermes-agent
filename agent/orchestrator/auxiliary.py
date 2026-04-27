"""Auxiliary runtime routing for side-task LLM calls.

Replaces auxiliary_client.py's ad-hoc resolution logic with a clean
protocol-based system. Each task type can override the default resolution
order in config.

Phase 1: Type definitions and shell interface. Phase 2: Wire up real
provider resolution (OpenRouter, Nous Portal, custom endpoint, Codex OAuth).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# AuxiliaryTask
# ============================================================================

class AuxiliaryTask(enum.Enum):
    """Side-task types that need LLM calls outside the primary loop."""

    COMPRESSION = "compression"
    TITLE_GENERATION = "title_generation"
    SESSION_SEARCH = "session_search"
    VISION_ANALYSIS = "vision_analysis"
    BROWSER_VISION = "browser_vision"
    WEB_EXTRACTION = "web_extraction"


# ============================================================================
# AuxiliaryConfig
# ============================================================================

@dataclass
class AuxiliaryConfig:
    """Configuration for auxiliary runtime resolution.

    resolution_order: Default list of resolver names to try.
        Phase 1 stub: ["openrouter", "nous_portal", "custom_endpoint", "codex_oauth"]
    task_overrides: Per-task overrides of the default resolution order.
        e.g., {"vision_analysis": ["openrouter", "nous_portal"]}
    """

    resolution_order: list[str] = field(default_factory=lambda: [
        "openrouter", "nous_portal", "custom_endpoint", "codex_oauth"
    ])
    task_overrides: dict[str, list[str]] = field(default_factory=dict)


# ============================================================================
# AuxiliaryRuntime
# ============================================================================

class AuxiliaryRuntime:
    """Routes side-task API calls through the best available backend.

    Tries each resolver in the chain until one succeeds.
    Caches the result for reuse within the session.

    Phase 1: Interface only, no real backends wired.
    Phase 2: Connect to OpenRouter, Nous Portal, custom endpoint, Codex OAuth.
    """

    def __init__(self, config: AuxiliaryConfig) -> None:
        self._config = config
        self._cache: dict[AuxiliaryTask, tuple[Any, str]] = {}
        self._failures: dict[str, float] = {}

    def get_client(self, task: AuxiliaryTask) -> tuple[Any, str]:
        """Return (client, model_name) for the given task type.

        Phase 1: Returns (None, "") since no resolvers are wired yet.
        Phase 2: Try each resolver in the task's resolution order,
        caching successful results.
        """
        if task in self._cache:
            return self._cache[task]

        # Phase 1: no real backends wired
        result = (None, "")
        self._cache[task] = result
        return result

    def get_vision_client(self) -> tuple[Any, str]:
        """Return client for image analysis tasks."""
        return self.get_client(AuxiliaryTask.VISION_ANALYSIS)

    def get_compression_client(self) -> tuple[Any, str]:
        """Return client for context compression tasks."""
        return self.get_client(AuxiliaryTask.COMPRESSION)

    def get_title_generation_client(self) -> tuple[Any, str]:
        """Return client for session title generation."""
        return self.get_client(AuxiliaryTask.TITLE_GENERATION)

    def get_session_search_client(self) -> tuple[Any, str]:
        """Return client for session search."""
        return self.get_client(AuxiliaryTask.SESSION_SEARCH)

    def report_failure(self, provider: str) -> None:
        """Record a provider failure for fallback decisions."""
        import time
        self._failures[provider] = time.time()