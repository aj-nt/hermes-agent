"""Responses API detection utilities.

Extracted from run_agent.py to decouple responses API model detection
from the AIAgent god-object.
"""

import re
from typing import Optional


def model_requires_responses_api(model: str) -> bool:
    """Return True for models that require the Responses API path.

    GPT-5.x models are rejected on /v1/chat/completions by both
    OpenAI and OpenRouter (error: ``unsupported_api_for_model``).
    Detect these so the correct api_mode is set regardless of
    which provider is serving the model.
    """
    m = model.lower()
    # Strip vendor prefix (e.g. "openai/gpt-5.4" → "gpt-5.4")
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m.startswith("gpt-5")


def provider_model_requires_responses_api(
    model: str,
    *,
    provider: Optional[str] = None,
) -> bool:
    """Return True when this provider/model pair should use Responses API."""
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "copilot":
        try:
            from hermes_cli.models import _should_use_copilot_responses_api
            return _should_use_copilot_responses_api(model)
        except Exception:
            # Fall back to the generic GPT-5 rule if Copilot-specific
            # logic is unavailable for any reason.
            pass
    return model_requires_responses_api(model)
