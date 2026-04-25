"""Error handling utilities for provider API failures.

Extracted from run_agent.py to decouple error summarization, masking,
context extraction, and session content cleaning from the AIAgent god-object.

All functions are pure (no class state, no side effects beyond logging).
"""

import copy
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agent.trajectory import convert_scratchpad_to_think
from utils import env_var_enabled

logger = logging.getLogger(__name__)

def summarize_api_error(error: Exception) -> str:
    """Extract a human-readable one-liner from an API error.

    Handles Cloudflare HTML error pages (502, 503, etc.) by pulling the
    <title> tag instead of dumping raw HTML.  Falls back to a truncated
    str(error) for everything else.
    """
    raw = str(error)

    # Cloudflare / proxy HTML pages: grab the <title> for a clean summary
    if "<!DOCTYPE" in raw or "<html" in raw:
        m = re.search(r"<title[^>]*>([^<]+)</title>", raw, re.IGNORECASE)
        title = m.group(1).strip() if m else "HTML error page (title not found)"
        # Also grab Cloudflare Ray ID if present
        ray = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
        ray_id = ray.group(1).strip() if ray else None
        status_code = getattr(error, "status_code", None)
        parts = []
        if status_code:
            parts.append(f"HTTP {status_code}")
        parts.append(title)
        if ray_id:
            parts.append(f"Ray {ray_id}")
        return " — ".join(parts)

    # JSON body errors from OpenAI/Anthropic SDKs
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
        if msg:
            status_code = getattr(error, "status_code", None)
            prefix = f"HTTP {status_code}: " if status_code else ""
            return f"{prefix}{msg[:300]}"

    # Fallback: truncate the raw string but give more room than 200 chars
    status_code = getattr(error, "status_code", None)
    prefix = f"HTTP {status_code}: " if status_code else ""
    return f"{prefix}{raw[:500]}"


def mask_api_key_for_logs(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def clean_error_message(error_msg: str) -> str:
    """
    Clean up error messages for user display, removing HTML content and truncating.
    
    Args:
        error_msg: Raw error message from API or exception
        
    Returns:
        Clean, user-friendly error message
    """
    if not error_msg:
        return "Unknown error"
        
    # Remove HTML content (common with CloudFlare and gateway error pages)
    if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
        return "Service temporarily unavailable (HTML error page returned)"
        
    # Remove newlines and excessive whitespace
    cleaned = ' '.join(error_msg.split())
    
    # Truncate if too long
    if len(cleaned) > 150:
        cleaned = cleaned[:150] + "..."
        
    return cleaned


def extract_api_error_context(error: Exception) -> Dict[str, Any]:
    """Extract structured rate-limit details from provider errors."""
    context: Dict[str, Any] = {}

    body = getattr(error, "body", None)
    payload = None
    if isinstance(body, dict):
        payload = body.get("error") if isinstance(body.get("error"), dict) else body
    if isinstance(payload, dict):
        reason = payload.get("code") or payload.get("error")
        if isinstance(reason, str) and reason.strip():
            context["reason"] = reason.strip()
        message = payload.get("message") or payload.get("error_description")
        if isinstance(message, str) and message.strip():
            context["message"] = message.strip()
        for key in ("resets_at", "reset_at"):
            value = payload.get(key)
            if value not in (None, ""):
                context["reset_at"] = value
                break
        retry_after = payload.get("retry_after")
        if retry_after not in (None, "") and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass
        ratelimit_reset = headers.get("x-ratelimit-reset")
        if ratelimit_reset and "reset_at" not in context:
            context["reset_at"] = ratelimit_reset

    if "message" not in context:
        raw_message = str(error).strip()
        if raw_message:
            context["message"] = raw_message[:500]

    if "reset_at" not in context:
        message = context.get("message") or ""
        if isinstance(message, str):
            delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
            if delay_match:
                value = float(delay_match.group(1))
                seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                context["reset_at"] = time.time() + seconds
            else:
                sec_match = re.search(
                    r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                    message,
                    re.IGNORECASE,
                )
                if sec_match:
                    context["reset_at"] = time.time() + float(sec_match.group(1))

    return context


def clean_session_content(content: str) -> str:
    """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
    if not content:
        return content
    content = convert_scratchpad_to_think(content)
    content = re.sub(r'\n+(<think>)', r'\n\1', content)
    content = re.sub(r'(</think>)\n+', r'\1\n', content)
    return content.strip()

def dump_api_request_debug(
    api_kwargs: Dict[str, Any],
    *,
    reason: str,
    error: Optional[Exception] = None,
    api_key: Optional[str] = None,
    base_url: str = "",
    api_mode: str = "",
    session_id: str = "",
    logs_dir: Path = None,
    verbose_logging: bool = False,
    mask_fn=mask_api_key_for_logs,
) -> Optional[Path]:
    """Dump a debug-friendly HTTP request record for the active inference API.

    Captures the request body from api_kwargs (excluding transport-only keys
    like timeout). Intended for debugging provider-side 4xx failures where
    retries are not useful.

    This is a pure-function version. The AIAgent method delegates to this,
    passing its instance attributes as arguments.
    """
    try:
        body = copy.deepcopy(api_kwargs)
        body.pop("timeout", None)
        body = {k: v for k, v in body.items() if v is not None}

        dump_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "reason": reason,
            "request": {
                "method": "POST",
                "url": f"{base_url.rstrip('/')}{'/responses' if api_mode == 'codex_responses' else '/chat/completions'}",
                "headers": {
                    "Authorization": f"Bearer {mask_fn(api_key)}",
                    "Content-Type": "application/json",
                },
                "body": body,
            },
        }

        if error is not None:
            error_info: Dict[str, Any] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            for attr_name in ("status_code", "request_id", "code", "param", "type"):
                attr_value = getattr(error, attr_name, None)
                if attr_value is not None:
                    error_info[attr_name] = attr_value

            body_attr = getattr(error, "body", None)
            if body_attr is not None:
                error_info["body"] = body_attr

            response_obj = getattr(error, "response", None)
            if response_obj is not None:
                try:
                    error_info["response_status"] = getattr(response_obj, "status_code", None)
                    error_info["response_text"] = response_obj.text
                except Exception as e:
                    logger.debug("Could not extract error response details: %s", e)

            dump_payload["error"] = error_info

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dump_file = logs_dir / f"request_dump_{session_id}_{timestamp}.json"
        dump_file.write_text(
            json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
            print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

        return dump_file
    except Exception as dump_error:
        if verbose_logging:
            logging.warning(f"Failed to dump API request debug payload: {dump_error}")
        return None

def usage_summary_for_api_request_hook(
    response: Any, *, provider: str = "", api_mode: str = ""
) -> Optional[Dict[str, Any]]:
    """Token buckets for ``post_api_request`` plugins (no raw ``response`` object)."""
    if response is None:
        return None
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return None
    from dataclasses import asdict
    from agent.usage_pricing import normalize_usage

    cu = normalize_usage(raw_usage, provider=provider, api_mode=api_mode)
    summary = asdict(cu)
    summary.pop("raw_usage", None)
    summary["prompt_tokens"] = cu.prompt_tokens
    summary["total_tokens"] = cu.total_tokens
    return summary
