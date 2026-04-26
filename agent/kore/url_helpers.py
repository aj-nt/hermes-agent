"""URL and routing helper functions for provider detection.

Extracted from run_agent.py to decouple URL matching and routing logic
from the AIAgent god-object.

Layer 2 extraction: these functions read self.* attributes. They are extracted
as pure functions that take the needed URLs/config as parameters; the AIAgent
methods become thin delegations.
"""

from utils import base_url_hostname, base_url_host_matches


def is_direct_openai_url(base_url: str = None, base_url_lower: str = "", base_url_hostname_val: str = "") -> bool:
    """Return True when a base URL targets OpenAI's native API.

    Args:
        base_url: Explicit base URL to check. If None, uses the cached values.
        base_url_lower: Cached lowercase base URL (used when base_url is None).
        base_url_hostname_val: Cached hostname (used when base_url is None).
    """
    if base_url is not None:
        hostname = base_url_hostname(base_url)
    else:
        hostname = base_url_hostname_val or base_url_hostname(base_url_lower)
    return hostname == "api.openai.com"


def is_openrouter_url(base_url_lower: str) -> bool:
    """Return True when the base URL targets OpenRouter."""
    return base_url_host_matches(base_url_lower, "openrouter.ai")


def is_qwen_portal(base_url_lower: str) -> bool:
    """Return True when the base URL targets Qwen Portal."""
    return base_url_host_matches(base_url_lower, "portal.qwen.ai")


def is_azure_openai_url(base_url: str = None, base_url_lower: str = "") -> bool:
    """Return True when a base URL targets Azure OpenAI.

    Azure OpenAI exposes an OpenAI-compatible endpoint at
    ``{resource}.openai.azure.com/openai/v1`` that accepts the
    standard ``openai`` Python client.  Unlike api.openai.com it
    does NOT support the Responses API — gpt-5.x models are served
    on the regular ``/chat/completions`` path — so routing decisions
    must treat Azure separately from direct OpenAI.
    """
    if base_url is not None:
        url = str(base_url).lower()
    else:
        url = base_url_lower or ""
    return "openai.azure.com" in url


def max_tokens_param(value: int, is_direct_openai: bool, is_azure_openai: bool = False) -> dict:
    """Return the correct max tokens kwarg for the current provider.

    OpenAI's newer models (gpt-4o, o-series, gpt-5+) require
    'max_completion_tokens'. Azure OpenAI also requires
    'max_completion_tokens' for gpt-5.x models served via the
    OpenAI-compatible endpoint. OpenRouter, local models, and older
    OpenAI models use 'max_tokens'.

    Args:
        value: The max token count.
        is_direct_openai: Whether the provider is direct OpenAI (not via proxy).
        is_azure_openai: Whether the provider is Azure OpenAI.
    """
    if is_direct_openai or is_azure_openai:
        return {"max_completion_tokens": value}
    return {"max_tokens": value}