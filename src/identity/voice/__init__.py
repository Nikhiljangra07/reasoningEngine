"""
Voice — output-time discipline for the identity layer.

Two utilities:

  strip.py  — light, surgical regex strip of the most common opener
              fluff. Fail-closed: anything strip misses is caught by
              lint. The strip never rewrites — it only removes a
              prefix when an exact pattern matches.

  lint.py   — structured lint over the generated output. Checks
              against the anti-laws and the seven identity criteria.
              When the lint fails, the caller regenerates with a
              stronger directive rather than trying to rewrite the
              output. This avoids the brittleness of regex-based
              hedging-pattern rewrites.
"""

from src.identity.voice.lint import (
    LintContext,
    LintResult,
    LintViolation,
    Severity,
    build_regenerate_directive,
    lint,
    should_regenerate,
)
from src.identity.voice.strip import (
    OPENER_PATTERNS,
    strip_openers,
)

__all__ = [
    # lint
    "LintContext",
    "LintResult",
    "LintViolation",
    "Severity",
    "build_regenerate_directive",
    "lint",
    "should_regenerate",
    # strip
    "OPENER_PATTERNS",
    "strip_openers",
]
