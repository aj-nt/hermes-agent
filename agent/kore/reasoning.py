"""Reasoning and API call configuration utilities.

Extracted from run_agent.py to decouple reasoning detection, Github Models
reasoning payload formatting, and stale timeout resolution from the AIAgent
god-object.

Layer 2 extraction: these functions read self.* attributes. They are extracted
as pure functions that take the needed attributes as parameters; the AIAgent
methods become thin delegations passing self.model, self.provider, etc.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from utils import base_url_host_matches
from hermes_cli.timeouts import get_provider_stale_timeout


def resolved_api_call_stale_timeout_base(
    provider: str, model: str
) -> Tuple[float, bool]:
    """Resolve the base non-stream stale timeout and whether it is implicit.

    Priority:
      1. ``providers.<id>.models.<model>.stale_timeout_seconds``
      2. ``providers.<id>.stale_timeout_seconds``
      3. ``HERMES_API_CALL_STALE_TIMEOUT`` env var
      4. 300.0s default

    Returns ``(timeout_seconds, uses_implicit_default)`` so the caller can
    preserve legacy behaviors that only apply when the user has *not*
    explicitly configured a stale timeout, such as auto-disabling the
    detector for local endpoints.
    """
    cfg = get_provider_stale_timeout(provider, model)
    if cfg is not None:
        return cfg, False

    env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT")
    if env_timeout is not None:
        return float(env_timeout), False

    return 300.0, True


def supports_reasoning_extra_body(
    model: str, base_url_lower: str
) -> bool:
    """Return True when reasoning extra_body is safe to send for this route/model.

    OpenRouter forwards unknown extra_body fields to upstream providers.
    Some providers/routes reject `reasoning` with 400s, so gate it to
    known reasoning-capable model families and direct Nous Portal.
    """
    if base_url_host_matches(base_url_lower, "nousresearch.com"):
        return True
    if base_url_host_matches(base_url_lower, "ai-gateway.vercel.sh"):
        return True
    if (
        base_url_host_matches(base_url_lower, "models.github.ai")
        or base_url_host_matches(base_url_lower, "api.githubcopilot.com")
    ):
        try:
            from hermes_cli.models import github_model_reasoning_efforts

            return bool(github_model_reasoning_efforts(model))
        except Exception:
            return False
    if "openrouter" not in base_url_lower:
        return False
    if "api.mistral.ai" in base_url_lower:
        return False

    model_lower = (model or "").lower()
    reasoning_model_prefixes = (
        "deepseek/",
        "anthropic/",
        "openai/",
        "x-ai/",
        "google/gemini-2",
        "qwen/qwen3",
    )
    return any(model_lower.startswith(prefix) for prefix in reasoning_model_prefixes)


def github_models_reasoning_extra_body(
    model: str, reasoning_config: Optional[dict]
) -> Optional[dict]:
    """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
    try:
        from hermes_cli.models import github_model_reasoning_efforts
    except Exception:
        return None

    supported_efforts = github_model_reasoning_efforts(model)
    if not supported_efforts:
        return None

    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            return None
        requested_effort = str(
            reasoning_config.get("effort", "medium")
        ).strip().lower()
    else:
        requested_effort = "medium"

    if requested_effort == "xhigh" and "high" in supported_efforts:
        requested_effort = "high"
    elif requested_effort not in supported_efforts:
        if requested_effort == "minimal" and "low" in supported_efforts:
            requested_effort = "low"
        elif "medium" in supported_efforts:
            requested_effort = "medium"
        else:
            requested_effort = supported_efforts[0]

    return {"effort": requested_effort}


def needs_kimi_tool_reasoning(provider: str, base_url: str) -> bool:
    """Return True when the current provider is Kimi / Moonshot thinking mode.

    Kimi ``/coding`` and Moonshot thinking mode both require
    ``reasoning_content`` on every assistant tool-call message; omitting
    it causes the next replay to fail with HTTP 400.
    """
    return (
        provider in {"kimi-coding", "kimi-coding-cn"}
        or base_url_host_matches(base_url, "api.kimi.com")
        or base_url_host_matches(base_url, "moonshot.ai")
        or base_url_host_matches(base_url, "moonshot.cn")
    )


def needs_deepseek_tool_reasoning(
    provider: str, base_url: str, model: str
) -> bool:
    """Return True when the current provider is DeepSeek thinking mode.

    DeepSeek V4 thinking mode requires ``reasoning_content`` on every
    assistant tool-call turn; omitting it causes HTTP 400 when the
    message is replayed in a subsequent API request (#15250).
    """
    provider_lower = (provider or "").lower()
    model_lower = (model or "").lower()
    return (
        provider_lower == "deepseek"
        or "deepseek" in model_lower
        or base_url_host_matches(base_url, "api.deepseek.com")
    )


def copy_reasoning_content_for_api(
    source_msg: dict,
    api_msg: dict,
    provider: str,
    base_url: str,
    model: str,
) -> None:
    """Copy provider-facing reasoning fields onto an API replay message."""
    if source_msg.get("role") != "assistant":
        return

    explicit_reasoning = source_msg.get("reasoning_content")
    if isinstance(explicit_reasoning, str):
        api_msg["reasoning_content"] = explicit_reasoning
        return

    normalized_reasoning = source_msg.get("reasoning")
    if isinstance(normalized_reasoning, str) and normalized_reasoning:
        api_msg["reasoning_content"] = normalized_reasoning
        return

    # Providers that require an echoed reasoning_content on every
    # assistant tool-call turn. Detection logic lives in the per-provider
    # helpers so both the creation path (_build_assistant_message) and
    # this replay path stay in sync.
    if source_msg.get("tool_calls") and (
        needs_kimi_tool_reasoning(provider, base_url)
        or needs_deepseek_tool_reasoning(provider, base_url, model)
    ):
        api_msg["reasoning_content"] = ""


def extract_reasoning(assistant_message) -> Optional[str]:
    """
    Extract reasoning/thinking content from an assistant message.

    OpenRouter and various providers can return reasoning in multiple formats:
    1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
    2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
    3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)

    Args:
        assistant_message: The assistant message object from the API response

    Returns:
        Combined reasoning text, or None if no reasoning found
    """
    reasoning_parts = []

    # Check direct reasoning field
    if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
        reasoning_parts.append(assistant_message.reasoning)

    # Check reasoning_content field (alternative name used by some providers)
    if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
        # Don't duplicate if same as reasoning
        if assistant_message.reasoning_content not in reasoning_parts:
            reasoning_parts.append(assistant_message.reasoning_content)

    # Check reasoning_details array (OpenRouter unified format)
    # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
        for detail in assistant_message.reasoning_details:
            if isinstance(detail, dict):
                # Extract summary from reasoning detail object
                summary = (
                    detail.get('summary')
                    or detail.get('thinking')
                    or detail.get('content')
                    or detail.get('text')
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary)

    # Some providers embed reasoning directly inside assistant content
    # instead of returning structured reasoning fields.  Only fall back
    # to inline extraction when no structured reasoning was found.
    content = getattr(assistant_message, "content", None)
    if not reasoning_parts and isinstance(content, str) and content:
        inline_patterns = (
            r"<think>(.*?)</think>",
            r"<thinking>(.*?)</thinking>",
            r"<thought>(.*?)</thought>",
            r"<reasoning>(.*?)</reasoning>",
            r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
        )
        for pattern in inline_patterns:
            flags = re.DOTALL | re.IGNORECASE
            for block in re.findall(pattern, content, flags=flags):
                cleaned = block.strip()
                if cleaned and cleaned not in reasoning_parts:
                    reasoning_parts.append(cleaned)

    # Combine all reasoning parts
    if reasoning_parts:
        return "\n\n".join(reasoning_parts)

    return None


def get_messages_up_to_last_assistant(messages: List[Dict]) -> List[Dict]:
    """
    Get messages up to (but not including) the last assistant turn.

    This is used when we need to "roll back" to the last successful point
    in the conversation, typically when the final assistant message is
    incomplete or malformed.

    Args:
        messages: Full message list

    Returns:
        Messages up to the last complete assistant turn (ending with user/tool message)
    """
    if not messages:
        return []

    # Find the index of the last assistant message
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        # No assistant message found, return all messages
        return messages.copy()

    # Return everything up to (not including) the last assistant message
    return messages[:last_assistant_idx]
