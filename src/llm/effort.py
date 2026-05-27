"""
Effort tier — the user-facing iteration budget control.

Four discrete tiers control how many engine iterations a request gets:

    LOW    → 3 iterations    (fast, cheap, surface-level confidence)
    MEDIUM → 6 iterations    (default; balances depth and cost)
    HIGH   → 10 iterations   (deep; closer to MAX_ITERATIONS=12)
    AUTO   → 12 iterations   (engine cap; user authorizes full discretion —
                              convergence + budget enforce the real stop)

Iterations remain the engine's existing knob — this module is the thin
translation layer between a user-friendly label and the integer the
formation engine already accepts via `max_iterations`.

The engine's own hard cap (src/formation/convergence_protocol.MAX_ITERATIONS=12)
is the absolute ceiling. Effort tiers stay strictly below it so the engine
keeps headroom for convergence checks and forced-stop logic.

Pricing intuition (rough, will vary by model mix):
    LOW    ≈ 5-8 LLM calls   →  $0.02 - $0.06
    MEDIUM ≈ 10-15 LLM calls →  $0.05 - $0.15
    HIGH   ≈ 18-25 LLM calls →  $0.10 - $0.30

These are the numbers to show in the UI's "effort selector" — they map
directly to user expectations of latency and spend.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Effort(str, Enum):
    """User-facing effort tiers. Inherits str so JSON serialization is trivial."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


# Iteration budget per tier. Capped by MAX_ITERATIONS=12 in the engine.
#
# These used to be 3/6/10/12. Halved because each iteration fans out
# to 5 domains × multiple concepts × Ke critics, so even MEDIUM (6 iter)
# routinely overshot the 720s wall-time cap on long DEEP prompts and
# the frontend gave up before the engine returned. AUTO=8 matches the
# benchmark setting that produced responses reliably.
EFFORT_ITERATIONS: dict[Effort, int] = {
    Effort.LOW: 2,
    Effort.MEDIUM: 3,
    Effort.HIGH: 5,
    Effort.AUTO: 8,
}


# Default when no effort is supplied.
DEFAULT_EFFORT: Effort = Effort.MEDIUM


def normalize_effort(value: Any) -> Effort:
    """
    Coerce an arbitrary value into an Effort tier.

    Accepts: Effort enum, lowercase string ("low"/"medium"/"high"),
    or None (returns DEFAULT_EFFORT). Unknown strings fall back to default
    rather than raising — callers downstream stay simple.
    """
    if value is None:
        return DEFAULT_EFFORT
    if isinstance(value, Effort):
        return value
    if isinstance(value, str):
        try:
            return Effort(value.strip().lower())
        except ValueError:
            return DEFAULT_EFFORT
    return DEFAULT_EFFORT


def iterations_for(effort: Any) -> int:
    """Return the iteration budget for the given effort tier."""
    return EFFORT_ITERATIONS[normalize_effort(effort)]
