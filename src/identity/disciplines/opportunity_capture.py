"""
Opportunity capture — the six-question test on a surfaced opening.

An opening is anything that might be capitalized on — a new piece of
evidence, a sudden constraint loosening, a market move, an offer, a
piece of feedback that contradicts the user's working assumption. The
discipline answers: capture, defer, or skip.

Six questions:

  1. Advances the goal — does this concretely move the user toward
     the real goal?
  2. Hidden cost named — is the cost (time, focus, reputation, sunk)
     visible, or is the opening pretending to be free?
  3. Fits current power — can the user actually act on this with the
     resources they have right now? Otherwise it's a future-self bet.
  4. Reversible — if it goes wrong, can the user back out? Irreversible
     openings need a higher bar.
  5. Doesn't dilute focus — does taking this fork shatter the singular
     path the user is on?
  6. Real opening — is this a genuine opening or a fashionable
     distraction wearing the costume of an opening?

Threshold: ≥4 / 6 to surface as a candidate; ≥6 / 6 to recommend
acting. The dossier builder uses the test to filter; the synthesizer
uses it to shape the response (surface vs recommend vs skip).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.identity.singular_path import Goal, Position


SIX_QUESTIONS: tuple[str, ...] = (
    "advances_goal",
    "hidden_cost_named",
    "fits_current_power",
    "reversible",
    "preserves_focus",
    "real_opening",
)


@dataclass(frozen=True)
class Opening:
    """A candidate opening to be tested.

    `description` is the short statement of what the opening is.
    `claimed_cost` is what the user (or the source) said the cost was;
    None means no cost was claimed, which is itself a signal.
    `requires` is a tuple of preconditions for acting on it — used to
    answer 'fits current power'."""

    description:  str
    claimed_cost: str | None         = None
    requires:     tuple[str, ...]    = ()


@dataclass(frozen=True)
class CaptureVerdict:
    """The verdict for a tested opening.

    `score` is in 0..6 — the number of questions answered 'yes'.
    `verdict` is one of "capture", "surface", "skip":
        capture  — score 6 / 6, recommend acting
        surface  — score 4 or 5, name it but do not recommend
        skip     — score 0..3, do not surface

    `answers` records the per-question answers (True / False) for
    auditability. `reasons` collects short clauses the synthesizer
    can fold into prompt context — never rendered verbatim, used to
    shape prose."""

    opening: Opening
    score:   int
    verdict: str
    answers: dict[str, bool]
    reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Per-question heuristics
# ---------------------------------------------------------------------------
#
# Each question is its own private predicate. Heuristics are
# deliberately legible — the score is auditable from the answers, and
# any individual predicate can be tuned without touching the others.

def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _overlap(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _advances_goal(opening: Opening, goal: Goal) -> bool:
    """Token overlap between the opening description and the real
    goal. Threshold deliberately low — we want to ask the question,
    not gate too aggressively. Real cases stand out clearly."""
    return _overlap(opening.description, goal.real) >= 0.10


def _hidden_cost_named(opening: Opening) -> bool:
    """Either the claimed_cost field is populated, OR the description
    itself names a cost (with words like 'costs', 'requires', 'risks',
    'sunk', 'tradeoff'). 'Free' or 'no-cost' framings fail this
    question — a free lunch is the most expensive meal in this
    discipline."""
    if opening.claimed_cost and opening.claimed_cost.strip():
        return True
    cost_cues = ("cost", "risk", "tradeoff", "sacrifice", "spend", "give up")
    free_cues = ("free", "no cost", "no risk", "easy win", "low-hanging")
    desc = opening.description.lower()
    if any(c in desc for c in free_cues):
        return False
    return any(c in desc for c in cost_cues)


def _fits_current_power(opening: Opening, position: Position | None) -> bool:
    """If the opening has explicit preconditions, the position must
    plausibly cover them. When position is None we conservatively
    return True only when the opening lists no preconditions; the
    user's resources are unknown and we don't want a False here to
    spuriously block surfacing."""
    if not opening.requires:
        return True
    if position is None:
        return False
    pos_low = position.text.lower()
    return all(req.lower() in pos_low for req in opening.requires)


def _reversible(opening: Opening) -> bool:
    """Irreversible cues failing the question; reversible cues
    passing. When neither appears the question passes by default —
    most decisions are reversible enough."""
    desc = opening.description.lower()
    irreversible_cues = (
        "irreversible",
        "permanent",
        "one-way door",
        "can't undo",
        "no going back",
        "burn the bridge",
        "burn bridges",
    )
    if any(c in desc for c in irreversible_cues):
        return False
    return True


def _preserves_focus(opening: Opening, goal: Goal) -> bool:
    """Focus is preserved when the opening operates inside the
    user's stated arena. If the opening's description has no overlap
    with the stated goal, it's a sidequest — fails the question."""
    return _overlap(opening.description, goal.stated) >= 0.05


def _real_opening(opening: Opening) -> bool:
    """Fashion / novelty filter. Cue words that, when present without
    cost-naming, suggest the opening is a trend wearing the costume
    of opportunity. Combined with hidden_cost_named — if the opening
    LOOKS fashionable AND fails to name a cost, this question fails."""
    fashion_cues = (
        "everyone is doing",
        "new wave",
        "latest",
        "trend",
        "hype",
        "viral",
        "breaking",
        "exclusive",
    )
    desc = opening.description.lower()
    if not any(c in desc for c in fashion_cues):
        return True
    return _hidden_cost_named(opening)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def test(opening: Opening, goal: Goal, position: Position | None = None) -> CaptureVerdict:
    """Run the six-question test on an opening.

    Returns a CaptureVerdict with structured answers, score, and a
    verdict bucket. The engine acts on the verdict (capture / surface
    / skip) and folds `reasons` into prompt context to shape prose."""

    answers: dict[str, bool] = {
        "advances_goal":      _advances_goal(opening, goal),
        "hidden_cost_named":  _hidden_cost_named(opening),
        "fits_current_power": _fits_current_power(opening, position),
        "reversible":         _reversible(opening),
        "preserves_focus":    _preserves_focus(opening, goal),
        "real_opening":       _real_opening(opening),
    }

    score = sum(1 for v in answers.values() if v)

    if score >= 6:
        verdict = "capture"
    elif score >= 4:
        verdict = "surface"
    else:
        verdict = "skip"

    reasons: list[str] = []
    for q, ans in answers.items():
        if not ans:
            reasons.append(f"fails:{q}")
        else:
            reasons.append(f"passes:{q}")

    return CaptureVerdict(
        opening=opening,
        score=score,
        verdict=verdict,
        answers=answers,
        reasons=tuple(reasons),
    )
