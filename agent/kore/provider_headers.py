"""
Provider-specific HTTP header factories.

These are pure functions — no self, no state. They produce the HTTP
headers that specific providers require for authentication or
Cloudflare bypass.

Extracted from run_agent.py as part of the Kore refactor (Fowler Step 2).
Each provider adapter will own its own header factory; these are the
ones that were already module-level functions in the god object.
"""

from __future__ import annotations

import platform as _plat


def routermint_headers() -> dict:
    """Return the User-Agent header RouterMint needs to avoid Cloudflare 1010 blocks."""
    from hermes_cli import __version__ as _HERMES_VERSION

    return {
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    }


# Qwen Code version — must match the version string that portal.qwen.ai
# expects in the User-Agent header.  Kept as a module-level constant so
# the Qwen provider adapter can reference it.
_QWEN_CODE_VERSION = "0.14.1"


def qwen_portal_headers() -> dict:
    """Return default HTTP headers required by Qwen Portal API."""
    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }