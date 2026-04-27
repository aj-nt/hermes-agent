"""Tests for orchestrator auxiliary runtime routing.

These verify:
1. AuxiliaryTask enum values match the spec
2. AuxiliaryRuntime can resolve task types to client/model pairs
3. AuxiliaryConfig groups the resolution order settings
"""

import pytest


# ============================================================================
# AuxiliaryTask
# ============================================================================

class TestAuxiliaryTask:
    """AuxiliaryTask enumerates the side-task types that need LLM calls
    outside the primary conversation loop.
    """

    def test_import(self):
        from agent.orchestrator.auxiliary import AuxiliaryTask

    def test_compression_value(self):
        from agent.orchestrator.auxiliary import AuxiliaryTask
        assert AuxiliaryTask.COMPRESSION.value == "compression"

    def test_title_generation_value(self):
        from agent.orchestrator.auxiliary import AuxiliaryTask
        assert AuxiliaryTask.TITLE_GENERATION.value == "title_generation"

    def test_session_search_value(self):
        from agent.orchestrator.auxiliary import AuxiliaryTask
        assert AuxiliaryTask.SESSION_SEARCH.value == "session_search"

    def test_vision_analysis_value(self):
        from agent.orchestrator.auxiliary import AuxiliaryTask
        assert AuxiliaryTask.VISION_ANALYSIS.value == "vision_analysis"

    def test_browser_vision_value(self):
        from agent.orchestrator.auxiliary import AuxiliaryTask
        assert AuxiliaryTask.BROWSER_VISION.value == "browser_vision"

    def test_web_extraction_value(self):
        from agent.orchestrator.auxiliary import AuxiliaryTask
        assert AuxiliaryTask.WEB_EXTRACTION.value == "web_extraction"


# ============================================================================
# AuxiliaryConfig
# ============================================================================

class TestAuxiliaryConfig:
    """AuxiliaryConfig groups resolution-order settings for auxiliary tasks."""

    def test_import(self):
        from agent.orchestrator.auxiliary import AuxiliaryConfig

    def test_resolution_order_defaults(self):
        from agent.orchestrator.auxiliary import AuxiliaryConfig
        cfg = AuxiliaryConfig()
        # Default resolution order should include common providers
        assert isinstance(cfg.resolution_order, list)
        assert len(cfg.resolution_order) > 0

    def test_task_overrides_defaults_empty(self):
        from agent.orchestrator.auxiliary import AuxiliaryConfig
        cfg = AuxiliaryConfig()
        assert cfg.task_overrides == {}


# ============================================================================
# AuxiliaryRuntime
# ============================================================================

class TestAuxiliaryRuntime:
    """AuxiliaryRuntime routes side-task API calls through the best available backend.

    Replaces auxiliary_client.py's ad-hoc resolution logic.
    """

    def test_import(self):
        from agent.orchestrator.auxiliary import AuxiliaryRuntime

    def test_can_instantiate_with_config(self):
        from agent.orchestrator.auxiliary import AuxiliaryRuntime, AuxiliaryConfig
        config = AuxiliaryConfig()
        runtime = AuxiliaryRuntime(config)
        assert runtime is not None

    def test_get_client_returns_tuple(self):
        from agent.orchestrator.auxiliary import AuxiliaryRuntime, AuxiliaryConfig, AuxiliaryTask
        config = AuxiliaryConfig(resolution_order=[])
        runtime = AuxiliaryRuntime(config)
        # With no resolvers, should return (None, "")
        client, model = runtime.get_client(AuxiliaryTask.COMPRESSION)
        # Phase 1: no real backends wired, returns None
        assert client is None
        assert model == ""

    def test_get_vision_client_delegates_to_task(self):
        from agent.orchestrator.auxiliary import AuxiliaryRuntime, AuxiliaryConfig, AuxiliaryTask
        config = AuxiliaryConfig(resolution_order=[])
        runtime = AuxiliaryRuntime(config)
        client, model = runtime.get_vision_client()
        assert client is None

    def test_get_compression_client_delegates_to_task(self):
        from agent.orchestrator.auxiliary import AuxiliaryRuntime, AuxiliaryConfig, AuxiliaryTask
        config = AuxiliaryConfig(resolution_order=[])
        runtime = AuxiliaryRuntime(config)
        client, model = runtime.get_compression_client()
        assert client is None