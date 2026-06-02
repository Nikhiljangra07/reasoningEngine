"""
Resource conversion — every constraint, sunk effort, dead-end,
criticism, or wasted asset has a convertible form.

The discipline takes a candidate resource (often something the user
calls a problem, a sunk cost, or a criticism) and surfaces the
convertible forms — what it can be converted INTO if reframed. The
synthesizer uses these convertible forms to shape recommendations and
to prevent the conversation from treating dead-ends as terminal.

Two operations.

evaluate(resource, goal)
    Return a ConvertibilityScore — a 0..1 reading of how convertible
    the resource is relative to the user's real goal.

latent_uses(resource)
    Return a small set of named convertible forms — what this thing
    can become. The synthesizer cites these by their named form, not
    by their internal label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.identity.singular_path import Goal


# Resource categories — each has a small set of canonical conversions
# the discipline knows about. Categories are deliberately coarse;
# anything that doesn't match a category falls back to a general
# evaluator that uses lexicon overlap with the goal.
_CategoryDef = tuple[
    str,                # category label
    tuple[str, ...],    # cues that identify the category
    tuple[str, ...],    # named latent uses
]

_CATEGORIES: tuple[_CategoryDef, ...] = (
    (
        "sunk_effort",
        ("wasted", "thrown away", "sunk", "all for nothing", "didn't work out"),
        (
            "compressed apprenticeship",
            "proof of survivable hard problem",
            "calibrated reference for the next attempt",
            "credible failure story for partners or hires",
        ),
    ),
    (
        "criticism",
        ("told me", "criticism", "feedback", "they said", "they think", "called me out"),
        (
            "free QA on the public position",
            "signal of which arguments need armor",
            "list of unconvinced parties to study or convert",
            "test of conviction under pressure",
        ),
    ),
    (
        "dead_end",
        ("dead end", "doesn't work", "didn't pan out", "hit a wall", "broken approach"),
        (
            "ruling-out of one branch in the search tree",
            "narrowed hypothesis space",
            "evidence for what the next attempt must avoid",
            "freed budget for the next branch",
        ),
    ),
    (
        "constraint",
        ("can't afford", "no time", "no team", "no resources", "limited", "constraint"),
        (
            "forced prioritization of the singular path",
            "credible scope-reduction lever",
            "filter against feature drift",
            "test of whether the goal is real enough to survive scarcity",
        ),
    ),
    (
        "rejection",
        ("rejected", "no thanks", "passed on it", "turned me down", "didn't buy"),
        (
            "calibration of pitch fit to audience",
            "narrowing of the actual buyer universe",
            "stress test on conviction in the offering",
            "data point for the next iteration",
        ),
    ),
)


@dataclass(frozen=True)
class Resource:
    """A candidate resource the user has labeled as a problem,
    constraint, sunk cost, or rejection.

    `text` is what the user said. `goal_relevance` is the engine's
    overlap reading against the real goal — passed in rather than
    re-computed here to keep the discipline pure."""

    text:           str
    goal_relevance: float = 0.0


@dataclass(frozen=True)
class ConvertibilityScore:
    """How convertible the resource is.

    `score` on 0..1. `category` is the matched category label (or
    "general" when no category fired). `uses` are the named latent
    forms. `note` is a short clause for prompt context."""

    score:    float
    category: str
    uses:     tuple[str, ...]
    note:     str


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _overlap(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _match_category(text: str) -> _CategoryDef | None:
    low = text.lower()
    for cat in _CATEGORIES:
        _, cues, _ = cat
        if any(c in low for c in cues):
            return cat
    return None


def latent_uses(resource: str) -> tuple[str, ...]:
    """Return the named latent uses for a resource.

    When the resource matches a known category, returns that
    category's named uses. When it doesn't, returns a small set of
    general convertible forms applicable across categories."""

    matched = _match_category(resource)
    if matched is not None:
        return matched[2]

    return (
        "evidence about the shape of the problem space",
        "calibration data for the next move",
        "raw material the user has already paid for",
    )


def evaluate(resource: Resource, goal: Goal) -> ConvertibilityScore:
    """Evaluate convertibility of a resource against the real goal.

    Score combines:
      - category match (matched category → 0.5 base; general → 0.25)
      - goal overlap (adds up to 0.3)
      - resource_relevance the caller supplied (adds up to 0.2)
    Capped at 1.0."""

    matched = _match_category(resource.text)
    if matched is None:
        base = 0.25
        category = "general"
        uses = latent_uses(resource.text)
        note = "general convertibility — no specific category matched"
    else:
        base = 0.50
        category = matched[0]
        uses = matched[2]
        note = f"category:{category}"

    overlap   = _overlap(resource.text, goal.real) * 0.30
    relevance = max(0.0, min(0.20, resource.goal_relevance * 0.20))
    score = min(1.0, base + overlap + relevance)

    return ConvertibilityScore(
        score=score,
        category=category,
        uses=uses,
        note=note,
    )
