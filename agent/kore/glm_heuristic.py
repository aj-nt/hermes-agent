"""
Ollama/GLM stop-to-length heuristic.

Detects when an Ollama-hosted GLM model reports ``finish_reason: "stop"``
but has actually truncated the response.  This is a known quirk of the
Ollama/GLM combination where the model emits a stop reason even when
the response was cut short.

Ported from AIAgent._is_ollama_glm_backend and
AIAgent._should_treat_stop_as_truncated.
"""

from __future__ import annotations

import re
from typing import List, Optional

from agent.kore.config import ProviderConfig
from agent.kore.think_blocks import (
    has_natural_response_ending,
    strip_think_blocks,
)
from agent.model_metadata import is_local_endpoint


def is_ollama_glm_backend(config: ProviderConfig) -> bool:
    """Detect the narrow backend family affected by Ollama/GLM stop misreports.

    Returns True if the model is a GLM variant (or the provider is ZAI)
    *and* the endpoint is running through Ollama or on a local network
    address (which includes Tailscale CGNAT).

    Args:
        config: Provider configuration with model, provider, and base_url info.
    """
    model_lower = (config.model or "").lower()
    provider_lower = (config.provider or "").lower()
    if "glm" not in model_lower and provider_lower != "zai":
        return False
    base_url_lower = (config.base_url or "").lower()
    if "ollama" in base_url_lower or ":11434" in base_url_lower:
        return True
    return bool(config.base_url and is_local_endpoint(config.base_url))


def should_treat_stop_as_truncated(
    config: ProviderConfig,
    finish_reason: str,
    assistant_message,
    messages: Optional[List] = None,
    heuristic_enabled: bool = True,
) -> bool:
    """Detect conservative stop-to-length misreports for Ollama/GLM models.

    When an Ollama-hosted GLM model reports ``finish_reason: "stop"``
    *without* tool calls and with a short, punctuation-less visible
    response, it is likely a truncated generation that the server
    incorrectly labelled as complete.

    Guard rails (to avoid false positives, cf. #14572):
      - heuristic_enabled flag (default True) disables the heuristic
        entirely when set to False.
      - Short responses (<500 chars with whitespace) are almost certainly
        complete -- a truly truncated response would be long enough to hit
        the token limit.
      - Only applies after tool-use turns (Ollama/GLM is known to
        misreport stop-after-tool continuations).
      - Responses ending with emoji or other Unicode sign-off glyphs are
        treated as naturally complete (see has_natural_response_ending).

    Args:
        config: Provider configuration (must include api_mode).
        finish_reason: The finish_reason from the API response.
        assistant_message: The assistant message object (or dict).
        messages: The full conversation messages list.
        heuristic_enabled: Config opt-out; when False, never trigger.
    """
    # Config opt-out: if the user has disabled the heuristic, never trigger.
    if not heuristic_enabled:
        return False

    if finish_reason != "stop" or config.api_mode != "chat_completions":
        return False
    if not is_ollama_glm_backend(config):
        return False
    if not any(
        isinstance(msg, dict) and msg.get("role") == "tool"
        for msg in (messages or [])
    ):
        return False
    if assistant_message is None or getattr(assistant_message, "tool_calls", None):
        return False

    content = getattr(assistant_message, "content", None)
    if not isinstance(content, str):
        return False

    visible_text = strip_think_blocks(content).strip()
    if not visible_text:
        return False
    # Very short responses with spaces are almost certainly complete --
    # they couldn't have hit a meaningful token limit.
    if len(visible_text) < 20 or not re.search(r"\s", visible_text):
        return False
    # Short-to-medium responses (<500 chars) are very unlikely to be
    # truncated.  Raising this gate from 20 to 500 eliminates the vast
    # majority of false positives from conversational replies that simply
    # lack terminal punctuation.  (See #14572 for the original bug where
    # emoji sign-offs triggered continuation loops on every turn.)
    if len(visible_text) < 500:
        return False

    return not has_natural_response_ending(visible_text)
