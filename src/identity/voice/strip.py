"""
Opener strip — light, surgical regex removal of the most common
opener fluff at the start of a model output.

This module is deliberately narrow.

  - Patterns match only the START of the output (^ anchored).
  - A match removes the matched prefix and any trailing punctuation
    / whitespace, then returns the remainder.
  - Patterns are conservative: misses are fine (lint catches them).
    False positives — stripping something the user wanted to keep —
    are NOT fine.
  - Strip never rewrites mid-text content. Hedging, structural
    issues, padding inside the body — all handled by lint+regenerate,
    not by this module.
"""

from __future__ import annotations

import re


# Each pattern matches a phrase that opens a response and then
# captures whatever comes after the punctuation that ends the
# opener. We rebuild the output as the captured remainder.
#
# Patterns are case-insensitive and tolerant of curly quotes.
OPENER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Great question!" / "That's a great point."
    re.compile(
        r"^\s*(?:great|excellent|wonderful|fantastic|terrific)\s+"
        r"(?:question|point|observation)[!.,]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:that['’]s|that is)\s+(?:a\s+)?"
        r"(?:great|excellent|wonderful|fantastic|terrific|good|interesting)\s+"
        r"(?:question|point|observation|thought)[!.,]?\s*",
        re.IGNORECASE,
    ),
    # "I'm happy to help" / "I'd be happy to help"
    re.compile(
        r"^\s*(?:i['’]m|i['’]d be|i would be|i am)\s+happy\s+to\s+"
        r"(?:help|assist|jump in|dig in)[!.,]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*happy\s+to\s+(?:help|assist|jump in|dig in)[!.,]?\s*",
        re.IGNORECASE,
    ),
    # "Certainly!" / "Absolutely!" / "Of course!"
    re.compile(
        r"^\s*(?:certainly|absolutely|of course|sure thing|definitely)[!.,]\s*",
        re.IGNORECASE,
    ),
    # "Let me think about this for you" / "Let me unpack this"
    re.compile(
        r"^\s*let me\s+"
        r"(?:think about this|unpack this|break this down|walk you through)"
        r"(?:\s+for you)?[!.,]?\s*",
        re.IGNORECASE,
    ),
    # "Thanks for sharing"
    re.compile(
        r"^\s*(?:thanks|thank you)\s+for\s+"
        r"(?:sharing|asking|the question|that)[!.,]?\s*",
        re.IGNORECASE,
    ),
    # "I appreciate you ..."
    re.compile(
        r"^\s*i appreciate\s+(?:you|your)\s+"
        r"(?:sharing|asking|question|trust|reaching out)[!.,]?\s*",
        re.IGNORECASE,
    ),
)


def strip_openers(text: str) -> str:
    """Remove the leading opener phrase from `text` when one of the
    OPENER_PATTERNS matches. Applies up to three passes to catch
    chained openers ("Great question! Happy to help — ..."). The
    body is otherwise unchanged.

    When no pattern matches, the input is returned unmodified."""

    if not text:
        return text

    result = text
    for _ in range(3):
        before = result
        for pat in OPENER_PATTERNS:
            m = pat.match(result)
            if m:
                result = result[m.end():]
        if result == before:
            break

    return result.lstrip()
