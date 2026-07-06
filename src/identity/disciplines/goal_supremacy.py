"""
Goal supremacy — the user's real goal outranks every other pressure.

Two operations.

discriminate(claim, goal, context)
    Score how well a candidate claim, finding, or recommendation
    serves the real goal. Used by the dossier builder to weight
    findings, and by the synthesizer to filter response components.

surface_real_goal(stated, signals)
    Detect when the stated goal contradicts other signals in the
    brief and propose an alternative. The engine then injects the
    RECOVER_GOAL_PROBE so the user can confirm one or the other.

No LLM calls. Heuristics here are deliberately conservative — the
discipline produces evidence, the user (and the model with that
evidence in hand) decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from src.identity.singular_path import Goal


# ---------------------------------------------------------------------------
# Score type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServeScore:
    """How well something serves the real goal.

    `score` is on 0..1.
    `verdict` is a 3-bucket reading the engine can branch on.
    `reasons` are the short clauses the synthesizer can fold into its
    prompt context — never rendered verbatim, used to weight prose.
    `serves_attachment_only` is True when the claim looks like it
    serves the user's current attachment but not the real goal —
    important enough to surface so the engine can warn rather than
    silently downrank."""

    score:                  float
    verdict:                str  # one of: "serves", "neutral", "diverts"
    reasons:                tuple[str, ...]
    serves_attachment_only: bool = False


# ---------------------------------------------------------------------------
# discriminate — score a claim against the real goal
# ---------------------------------------------------------------------------

# Words/phrases that, when present in a claim, suggest it is pointed at
# preserving an attachment (sunk cost, identity, comfort) rather than
# advancing the goal. These are SIGNALS, not verdicts — the engine
# combines them with overlap to decide.
_ATTACHMENT_LEXICON = (
    "preserve",
    "protect",
    "defend",
    "vindicate",
    "prove right",
    "save face",
    "stick with",
    "double down",
    "wait it out",
    "not yet",
    "more time",
    "more data",
)

# Words/phrases that suggest the claim is action-oriented and
# goal-directed. Same caveat — signal, not verdict.
_GOAL_DIRECTED_LEXICON = (
    "advance",
    "ship",
    "cut",
    "remove",
    "build",
    "test",
    "convert",
    "decide",
    "commit",
    "move",
)


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _overlap_ratio(a: str, b: str) -> float:
    """Jaccard overlap on token sets. Cheap, conservative, robust
    against short strings (returns 0 rather than NaN on empty input)."""
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _matches(text: str, lexicon: Iterable[str]) -> tuple[str, ...]:
    low = text.lower()
    return tuple(w for w in lexicon if w in low)


def discriminate(claim: str, goal: Goal) -> ServeScore:
    """Score how well a claim serves the user's real goal.

    Algorithm (deliberate transparency — auditable, not a model call):

      1. Compute overlap between the claim and the real goal.
      2. Compute overlap between the claim and the stated goal.
      3. Detect attachment-lexicon hits in the claim.
      4. Detect goal-directed-lexicon hits.
      5. Combine into a 0..1 score with a 3-bucket verdict.

    The 'serves_attachment_only' flag fires when the claim has high
    overlap with the stated goal but low overlap with the real goal
    AND attachment lexicon is present — i.e. the claim looks
    on-topic but is pointed at preserving the user's current
    attachment rather than reaching the goal. The engine surfaces
    this rather than silently downranking it."""

    real_ov   = _overlap_ratio(claim, goal.real)
    stated_ov = _overlap_ratio(claim, goal.stated)
    att_hits  = _matches(claim, _ATTACHMENT_LEXICON)
    dir_hits  = _matches(claim, _GOAL_DIRECTED_LEXICON)

    serves_attachment_only = (
        goal.surfaced
        and stated_ov >= 0.20
        and real_ov < 0.10
        and bool(att_hits)
    )

    # Score combines overlap with directional lexicon — capped at 1.
    raw = real_ov + 0.05 * len(dir_hits) - 0.05 * len(att_hits)
    score = max(0.0, min(1.0, raw))

    if serves_attachment_only:
        verdict = "diverts"
    elif score >= 0.30:
        verdict = "serves"
    elif score >= 0.10:
        verdict = "neutral"
    else:
        verdict = "diverts"

    reasons: list[str] = []
    if real_ov > 0:
        reasons.append(f"overlap-with-real-goal:{real_ov:.2f}")
    if dir_hits:
        reasons.append(f"goal-directed-signal:{','.join(dir_hits)}")
    if att_hits:
        reasons.append(f"attachment-signal:{','.join(att_hits)}")
    if serves_attachment_only:
        reasons.append("serves-stated-but-not-real-goal")

    return ServeScore(
        score=score,
        verdict=verdict,
        reasons=tuple(reasons),
        serves_attachment_only=serves_attachment_only,
    )


# ---------------------------------------------------------------------------
# surface_real_goal — detect contradiction between stated goal & signals
# ---------------------------------------------------------------------------

# Contradiction patterns: pairs of (stated-goal-cue, signal-cue) that,
# when both appear, suggest the stated goal contradicts the user's
# actual position. Conservative — we want false negatives over false
# positives. A spurious probe is more costly than missing one, because
# the probe interrupts the user.
_CONTRADICTION_PATTERNS: tuple[tuple[str, str, str], ...] = (
    # (stated cue, signal cue, alternative)
    ("ship",        "perfect",        "perfecting it before shipping"),
    ("decide",      "more research",  "delaying the decision"),
    ("focus",       "everything",     "doing everything at once"),
    ("simplify",    "feature",        "adding more"),
    ("quit",        "improve",        "improving rather than quitting"),
    ("commit",      "options",        "keeping options open"),
    ("launch",      "polish",         "polishing instead of launching"),
    ("cut scope",   "add",            "expanding rather than cutting"),
)


def surface_real_goal(stated: str, signals: tuple[str, ...]) -> Goal:
    """Detect when the stated goal contradicts the rest of the brief.

    `stated` is what the user said the goal is.
    `signals` is a tuple of other lines from the brief / session that
    might contradict the stated goal.

    When a contradiction pattern matches, returns a Goal with
    `surfaced=True`, `real` set to the inferred underlying goal, and
    `signals` preserved for auditability. The engine then injects
    RECOVER_GOAL_PROBE.

    When no contradiction is detected, returns a consistent Goal
    (stated == real, surfaced=False).

    This function intentionally does not call a model. The signal is
    coarse on purpose — when in doubt, do not surface, because a
    spurious probe wastes a user turn."""

    stated_low = stated.lower()

    for stated_cue, signal_cue, alternative in _CONTRADICTION_PATTERNS:
        if stated_cue not in stated_low:
            continue
        matching = tuple(s for s in signals if signal_cue in s.lower())
        if matching:
            return Goal(
                stated=stated,
                real=alternative,
                surfaced=True,
                signals=matching,
            )

    return Goal(stated=stated, real=stated, surfaced=False, signals=tuple())
