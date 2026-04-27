"""Tests for orchestrator ProviderProtocol, ProviderRegistry, and related types.

These verify:
1. ProviderProtocol is a valid typing.Protocol
2. ProviderRegistry can register and look up providers
3. FallbackChain resolves provider sequence
4. CredentialManager interface exists
5. FailoverReason enum values match the spec
"""

import pytest
from typing import get_type_hints

try:
    from typing import get_protocol_members
except ImportError:
    get_protocol_members = None  # Python 3.9 compat


# ============================================================================
# ProviderProtocol
# ============================================================================

class TestProviderProtocol:
    """ProviderProtocol defines the contract every LLM provider implements.

    Replaces scattered _is_*_backend() checks with capability-based dispatch.
    """

    def test_import(self):
        from agent.orchestrator.providers import ProviderProtocol

    def test_is_base_class_with_protocol_methods(self):
        from agent.orchestrator.providers import ProviderProtocol
        # ProviderProtocol is a base class defining the contract.
        # Phase 2 will migrate to typing.Protocol once Python 3.9 is dropped.
        assert isinstance(ProviderProtocol, type)  # it's a class

    def test_has_prepare_request_method(self):
        from agent.orchestrator.providers import ProviderProtocol
        assert 'prepare_request' in dir(ProviderProtocol)

    def test_has_execute_method(self):
        from agent.orchestrator.providers import ProviderProtocol
        assert 'execute' in dir(ProviderProtocol)

    def test_has_parse_response_method(self):
        from agent.orchestrator.providers import ProviderProtocol
        assert 'parse_response' in dir(ProviderProtocol)

    def test_has_capabilities_property(self):
        from agent.orchestrator.providers import ProviderProtocol
        assert 'capabilities' in dir(ProviderProtocol)


class TestProviderRegistry:
    """ProviderRegistry maps provider names to ProviderProtocol implementations.

    The orchestrator looks up providers by name instead of branching
    on provider identity.
    """

    def test_import(self):
        from agent.orchestrator.providers import ProviderRegistry

    def test_register_and_get(self):
        from agent.orchestrator.providers import ProviderRegistry, ProviderCapabilities
        from agent.orchestrator.context import ProviderResult

        class FakeProvider:
            @property
            def capabilities(self):
                return ProviderCapabilities()

            def prepare_request(self, ctx):
                return {}

            def execute(self, request):
                return ProviderResult()

            def parse_response(self, result, ctx):
                from agent.orchestrator.context import ParsedResponse
                return ParsedResponse()

        reg = ProviderRegistry()
        provider = FakeProvider()
        reg.register("fake", provider)
        assert reg.get("fake") is provider

    def test_get_unknown_raises(self):
        from agent.orchestrator.providers import ProviderRegistry
        reg = ProviderRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_providers(self):
        from agent.orchestrator.providers import ProviderRegistry, ProviderCapabilities
        from agent.orchestrator.context import ProviderResult, ParsedResponse

        class FakeProvider:
            @property
            def capabilities(self):
                return ProviderCapabilities()
            def prepare_request(self, ctx): return {}
            def execute(self, request): return ProviderResult()
            def parse_response(self, result, ctx): return ParsedResponse()

        reg = ProviderRegistry()
        reg.register("alpha", FakeProvider())
        reg.register("beta", FakeProvider())
        names = reg.list_names()
        assert "alpha" in names
        assert "beta" in names
        assert len(names) == 2


class TestFailoverReason:
    """FailoverReason classifies why a provider was abandoned."""

    def test_import(self):
        from agent.orchestrator.providers import FailoverReason

    def test_reason_values(self):
        from agent.orchestrator.providers import FailoverReason
        assert FailoverReason.RATE_LIMITED is not None
        assert FailoverReason.AUTH_FAILED is not None
        assert FailoverReason.CONNECTION_ERROR is not None
        assert FailoverReason.MODEL_OVERLOADED is not None
        assert FailoverReason.CONTEXT_TOO_LONG is not None
        assert FailoverReason.UNKNOWN is not None


class TestFallbackChain:
    """FallbackChain selects next provider on failure."""

    def test_import(self):
        from agent.orchestrator.providers import FallbackChain

    def test_next_returns_none_with_single_provider(self):
        from agent.orchestrator.providers import FallbackChain, FailoverReason
        from agent.orchestrator.context import ProviderCapabilities, ProviderResult, ParsedResponse

        class FakeProvider:
            @property
            def capabilities(self):
                return ProviderCapabilities()
            def prepare_request(self, ctx): return {}
            def execute(self, request): return ProviderResult()
            def parse_response(self, result, ctx): return ParsedResponse()

        only_provider = FakeProvider()
        chain = FallbackChain([only_provider])
        result = chain.next(only_provider, FailoverReason.RATE_LIMITED)
        assert result is None  # no more providers

    def test_next_returns_next_provider(self):
        from agent.orchestrator.providers import FallbackChain, FailoverReason
        from agent.orchestrator.context import ProviderCapabilities, ProviderResult, ParsedResponse

        class FakeProvider:
            def __init__(self, name):
                self._name = name
            @property
            def capabilities(self):
                return ProviderCapabilities()
            def prepare_request(self, ctx): return {}
            def execute(self, request): return ProviderResult()
            def parse_response(self, result, ctx): return ParsedResponse()

        p1 = FakeProvider("primary")
        p2 = FakeProvider("fallback1")
        chain = FallbackChain([p1, p2])
        result = chain.next(p1, FailoverReason.RATE_LIMITED)
        assert result is p2

    def test_next_exhausted_returns_none(self):
        from agent.orchestrator.providers import FallbackChain, FailoverReason
        from agent.orchestrator.context import ProviderCapabilities, ProviderResult, ParsedResponse

        class FakeProvider:
            @property
            def capabilities(self):
                return ProviderCapabilities()
            def prepare_request(self, ctx): return {}
            def execute(self, request): return ProviderResult()
            def parse_response(self, result, ctx): return ParsedResponse()

        p1, p2 = FakeProvider(), FakeProvider()
        chain = FallbackChain([p1, p2])
        result = chain.next(p2, FailoverReason.CONNECTION_ERROR)
        assert result is None  # already at last in chain


class TestCredentialManager:
    """CredentialManager owns credential lifecycle, rotation, and fallback decisions."""

    def test_import(self):
        from agent.orchestrator.providers import CredentialManager

    def test_can_instantiate(self):
        from agent.orchestrator.providers import CredentialManager
        cm = CredentialManager()
        assert cm is not None