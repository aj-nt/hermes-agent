"""Provider-specific message preparation utilities.

Extracted from run_agent.py to decouple Qwen/Anthropic message formatting
from the AIAgent god-object.

Layer 1 extraction: pure functions, no self references.
"""

import copy
from typing import Dict, List, Optional


def qwen_prepare_chat_messages(api_messages: list) -> list:
    prepared = copy.deepcopy(api_messages)
    if not prepared:
        return prepared

    for msg in prepared:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            # Normalize: convert bare strings to text dicts, keep dicts as-is.
            # deepcopy already created independent copies, no need for dict().
            normalized_parts = []
            for part in content:
                if isinstance(part, str):
                    normalized_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    normalized_parts.append(part)
            if normalized_parts:
                msg["content"] = normalized_parts

    # Inject cache_control on the last part of the system message.
    for msg in prepared:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                content[-1]["cache_control"] = {"type": "ephemeral"}
            break

    return prepared


def qwen_prepare_chat_messages_inplace(messages: list) -> None:
    """In-place variant — mutates an already-copied message list."""
    if not messages:
        return

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            normalized_parts = []
            for part in content:
                if isinstance(part, str):
                    normalized_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    normalized_parts.append(part)
            if normalized_parts:
                msg["content"] = normalized_parts

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                content[-1]["cache_control"] = {"type": "ephemeral"}
            break
