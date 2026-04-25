"""
Think-block stripping and content analysis utilities.

These functions strip reasoning/thinking blocks from LLM assistant
content and analyze whether visible text remains. They handle multiple
tag variants (think, thinking, reasoning, REASONING_SCRATCHPAD, thought)
and also strip stray tool-call XML that open models sometimes emit
inside content.

Extracted from run_agent.py as part of the Kore refactor (Fowler Step 2c).
These are pure functions -- no self, no state, no side effects.

Ported from openclaw/openclaw#67318 (Gemma-style tool call handling).
"""

from __future__ import annotations

import re


# Tag variants that represent reasoning/thinking blocks.
# Must stay in sync across: strip_think_blocks(), has_content_after_think_block(),
# gateway/stream_consumer.py, and the unterminated-tag regex below.
_THINK_TAG_NAMES = (
    "think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD",
)

# Tool-call XML tag names that some open models emit inline.
_TOOL_CALL_TAG_NAMES = (
    "tool_call", "tool_calls", "tool_result",
    "function_call", "function_calls",
)

# Unicode-aware set of characters that typically end a complete sentence
# or response. Includes CJK punctuation for Chinese/Japanese/Korean.
_NATURAL_ENDINGS = '.!?:)\'\"]]}。！？：）】」』》'


def strip_think_blocks(content: str) -> str:
    """Remove reasoning/thinking blocks from content, returning only visible text.

    Handles four cases:
      1. Closed tag pairs -- the common path when the provider emits
         complete reasoning blocks.
      2. Unterminated open tag at a block boundary (start of text or
         after a newline) -- e.g. MiniMax M2.7 / NIM endpoints where the
         closing tag is dropped.  Everything from the open tag to end
         of string is stripped.  The block-boundary check mirrors
         gateway/stream_consumer.py's filter so models that mention
         think tags in prose aren't over-stripped.
      3. Stray orphan open/close tags that slip through.
      4. Tag variants: think, thinking, reasoning, REASONING_SCRATCHPAD,
         thought (Gemma 4), all case-insensitive.

    Additionally strips standalone tool-call XML blocks that some open
    models (notably Gemma variants on OpenRouter) emit inside assistant
    content instead of via the structured tool_calls field:
      * tool_call, tool_calls, tool_result
      * function_call, function_calls
      * function name="..." (Gemma style)

    The function variant is boundary-gated (only strips when the tag sits
    at start-of-line or after punctuation and carries a name="..."
    attribute) so prose mentions like "Use <function> in JavaScript" are
    preserved.  Ported from openclaw/openclaw#67318.
    """
    if not content:
        return ""
    # 1. Closed tag pairs -- case-insensitive for all variants so
    #    mixed-case tags don't slip through to the unterminated-tag
    #    pass and take trailing content with them.
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
    # 1b. Tool-call XML blocks (openclaw/openclaw#67318). Handle the
    #     generic tag names first -- they have no attribute gating since
    #     a literal tool_call in prose is already vanishingly rare.
    for _tc_name in _TOOL_CALL_TAG_NAMES:
        content = re.sub(
            rf'<{_tc_name}\b[^>]*>.*?</{_tc_name}>',
            '',
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
    # 1c. <function name="...">...</function> -- Gemma-style standalone
    #     tool call. Only strip when the tag sits at a block boundary
    #     (start of text, after a newline, or after sentence-ending
    #     punctuation) AND carries a name="..." attribute. This keeps
    #     prose mentions like "Use <function> to declare" safe.
    content = re.sub(
        r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
        r'<function\b[^>]*\bname\s*=[^>]*>'
        r'(?:(?:(?!</function>).)*)</function>',
        '',
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 2. Unterminated reasoning block -- open tag at a block boundary
    #    (start of text, or after a newline) with no matching close.
    #    Strip from the tag to end of string.
    content = re.sub(
        r'(?:^|\n)[ \t]*<(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\b[^>]*>.*$',
        '',
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 3. Stray orphan open/close tags that slipped through.
    content = re.sub(
        r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*',
        '',
        content,
        flags=re.IGNORECASE,
    )
    # 3b. Stray tool-call closers. (We do NOT strip bare <function> or
    #     unterminated <function name="..."> because a truncated tail
    #     during streaming may still be valuable to the user; matches
    #     OpenClaw's intentional asymmetry.)
    content = re.sub(
        r'</(?:tool_call|tool_calls|tool_result|function_call|function_calls|function)>\s*',
        '',
        content,
        flags=re.IGNORECASE,
    )
    return content


def has_natural_response_ending(content: str) -> bool:
    """Heuristic: does visible assistant text look intentionally finished?

    Recognises ASCII/CJK punctuation, emoji, and other common sign-off
    glyphs as natural endings.  Returns True for characters that are
    unlikely to appear mid-sentence in a truncated response.

    Extended in #14572 to cover emoji sign-offs (e.g. hearts, sparkles,
    raised hands) which were previously false-positive triggers for the
    Ollama/GLM stop-to-length heuristic.
    """
    import unicodedata

    if not content:
        return False
    stripped = content.rstrip()
    if not stripped:
        return False
    if stripped.endswith("```"):
        return True

    # Strip trailing variation selectors (U+FE0F) and zero-width joiners
    # (U+200D) that emoji sequences use, so we check the "real" base glyph.
    i = len(stripped) - 1
    while i >= 0 and unicodedata.category(stripped[i]) in ("Mn", "Me", "Cf"):
        i -= 1
    if i < 0:
        return False
    last_char = stripped[i]

    # ASCII and CJK punctuation that signal a complete thought.
    if last_char in _NATURAL_ENDINGS:
        return True

    # Emoji and other Unicode sign-off glyphs.
    # We use unicodedata categories rather than a hard-coded codepoint
    # list so we automatically cover new emoji as Python's Unicode
    # database grows.
    cat = unicodedata.category(last_char)
    # So (Other_Symbol) covers sparkle, muscle, rocket, check, cross, warning etc.
    # Sk (Modifier_Symbol) covers VS16 (heart variation selector, etc.)
    # Sm (Math_Symbol) covers arrows and similar sign-off glyphs.
    if cat in ("So", "Sk", "Sm"):
        return True
    # Emoji_Presentation property: many emoji are General_Category=So
    # but some are in Lo/Lm/Other.  Check the wide "Extended_Pictographic"
    # property via the Emoji character range heuristic (U+1F000..U+1FAFF).
    cp = ord(last_char)
    if 0x1F000 <= cp <= 0x1FAFF:
        return True

    return False


def has_content_after_think_block(content: str) -> bool:
    """Check if content has actual text after any reasoning/thinking blocks.

    This detects cases where the model only outputs reasoning but no actual
    response, which indicates an incomplete generation that should be retried.
    Must stay in sync with strip_think_blocks() tag variants.

    Args:
        content: The assistant message content to check

    Returns:
        True if there's meaningful content after think blocks, False otherwise
    """
    if not content:
        return False
    # Remove all reasoning tag variants (must match strip_think_blocks)
    cleaned = strip_think_blocks(content)
    # Check if there's any non-whitespace content remaining
    return bool(cleaned.strip())
