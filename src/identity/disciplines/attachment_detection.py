"""
Attachment detection — name patterns that may distort the user's
judgment, without diagnosing the user.

The discipline scans the user's brief (or any user-supplied text) for
recognizable distortion patterns and surfaces them as AttachmentFlags
that the engine can fold into prompt context. The synthesizer
references the PATTERN, not the user — "this looks like sunk-cost
weight on the decision," not "you are showing sunk-cost bias."

Five kinds, all named for the pattern rather than the person:

  SUNK_COST            — past investment used as future justification
  IDENTITY_PROTECTION  — refusing a move because it threatens how the
                         user sees themselves or how they think
                         others see them
  URGENCY_AS_FEAR      — apparent urgency that's actually avoidance of
                         a harder, slower decision
  PATIENCE_AS_AVOIDANCE — apparent patience / "more research" that's
                         actually deferral of a hard call
  CONSENSUS_DRIFT      — adopting the position of the room because
                         it's the room, not because it's right

No LLM calls. Patterns are detected via lexicon co-occurrence. False
positives cost more than false negatives — when in doubt, don't flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class AttachmentKind(str, Enum):
    SUNK_COST             = "sunk_cost"
    IDENTITY_PROTECTION   = "identity_protection"
    URGENCY_AS_FEAR       = "urgency_as_fear"
    PATIENCE_AS_AVOIDANCE = "patience_as_avoidance"
    CONSENSUS_DRIFT       = "consensus_drift"


@dataclass(frozen=True)
class AttachmentFlag:
    """A detected distortion pattern.

    `kind` is the pattern category. `evidence` is the substring(s) of
    the user's text that triggered the flag — included so the
    synthesizer can be specific ('the line "I've already spent two
    years on this" reads as sunk-cost weight'). `severity` is on
    0..1 — multiplexed from cue density."""

    kind:     AttachmentKind
    evidence: tuple[str, ...]
    severity: float


# Each pattern definition is a (kind, primary_cues, secondary_cues,
# negation_cues) tuple. A flag fires when at least one primary cue
# matches AND no negation cue matches. Severity scales with the count
# of secondary cues, capped at 1.0.
_PatternDef = tuple[
    AttachmentKind,
    tuple[str, ...],  # primary
    tuple[str, ...],  # secondary (boost severity)
    tuple[str, ...],  # negation (kill the flag)
]

_PATTERNS: tuple[_PatternDef, ...] = (
    # SUNK COST
    (
        AttachmentKind.SUNK_COST,
        (
            "already spent",
            "already invested",
            "put so much",
            "put too much",
            "can't waste",
            "after all this time",
            "after all that work",
            "all those years",
            "we've come this far",
        ),
        ("years", "months", "thousands", "savings", "energy", "work"),
        ("regardless of what i spent", "ignore what i spent"),
    ),
    # IDENTITY PROTECTION
    (
        AttachmentKind.IDENTITY_PROTECTION,
        (
            "who i am",
            "what i stand for",
            "people will think",
            "would look bad",
            "lose face",
            "i'm the kind",
            "i'm not the kind",
            "my whole identity",
            "i built my name",
        ),
        ("reputation", "image", "brand", "persona"),
        ("regardless of how it looks",),
    ),
    # URGENCY AS FEAR
    (
        AttachmentKind.URGENCY_AS_FEAR,
        (
            "right now",
            "have to act now",
            "can't wait",
            "no time",
            "before it's too late",
            "before someone else",
        ),
        ("scared", "afraid", "panic", "rushing", "freaking"),
        ("genuine deadline", "actual deadline", "hard cutoff"),
    ),
    # PATIENCE AS AVOIDANCE
    (
        AttachmentKind.PATIENCE_AS_AVOIDANCE,
        (
            "more research",
            "more data",
            "more time to think",
            "not ready yet",
            "wait and see",
            "let me sleep on it",
            "let's see how it plays out",
        ),
        ("six months", "next quarter", "eventually", "someday"),
        ("research is the deliverable", "data is the deliverable"),
    ),
    # CONSENSUS DRIFT
    (
        AttachmentKind.CONSENSUS_DRIFT,
        (
            "everyone says",
            "everyone is doing",
            "what people do",
            "the standard advice",
            "the playbook says",
            "smart people think",
        ),
        ("conventional", "default", "industry standard"),
        ("regardless of consensus", "ignore the playbook"),
    ),
)


def _find_occurrences(text: str, cues: tuple[str, ...]) -> tuple[str, ...]:
    low = text.lower()
    return tuple(c for c in cues if c in low)


def scan(text: str) -> list[AttachmentFlag]:
    """Scan text for distortion patterns. Returns one AttachmentFlag
    per detected pattern. Empty list means nothing flagged.

    Severity is computed as 0.4 + 0.15 * len(secondary_hits), capped
    at 1.0. The 0.4 floor reflects that any pattern firing at all is
    worth surfacing; the secondary hits scale up severity within the
    pattern."""

    if not text or not text.strip():
        return []

    flags: list[AttachmentFlag] = []

    for kind, primary, secondary, negations in _PATTERNS:
        primary_hits = _find_occurrences(text, primary)
        if not primary_hits:
            continue
        if _find_occurrences(text, negations):
            continue
        secondary_hits = _find_occurrences(text, secondary)
        severity = min(1.0, 0.4 + 0.15 * len(secondary_hits))
        flags.append(AttachmentFlag(
            kind=kind,
            evidence=tuple(primary_hits + secondary_hits),
            severity=severity,
        ))

    return flags
