"""
Long-horizon vision — project a decision across timeframes.

Two operations.

project(decision, context)
    Produce a HorizonRead: a structured projection of the decision at
    6 months and 2 years (configurable via Context.horizon_months).
    The read is shaped by heuristics on the decision's text and the
    surrounding context; the synthesizer then folds the projection
    into the prompt as analytical scaffolding.

compounding_signal(action)
    Return +1 (compounds), 0 (neutral), -1 (decays). Cheap signal the
    dossier and synthesizer use to weight findings.

No LLM calls. The horizons here are heuristic — when the model
generates the response, it has both the user's brief and this
structured read in front of it, and produces the language. This
discipline produces the SKELETON, not the prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Verbs / phrases that historically compound: every increment buys
# more leverage on the next increment. Used for cheap signal scoring.
_COMPOUNDING_LEXICON = (
    "build",
    "ship",
    "publish",
    "compound",
    "invest",
    "automate",
    "document",
    "system",
    "process",
    "habit",
    "skill",
    "audience",
    "asset",
    "infrastructure",
    "leverage",
    "deepen",
    "specialize",
)

# Verbs / phrases that historically decay: the value drops if not
# refreshed, and the action itself doesn't accumulate.
_DECAYING_LEXICON = (
    "wait",
    "delay",
    "postpone",
    "polish",
    "rewrite",
    "redesign",
    "switch stack",
    "chase",
    "trend",
    "react",
    "respond to",
    "explain",
    "defend",
    "convince",
)


@dataclass(frozen=True)
class HorizonSignal:
    """A single projection point — value of the decision at a horizon.

    `direction` is one of: "compounds", "holds", "decays". The
    synthesizer maps these to language without rendering the labels
    verbatim. `note` is the heuristic clause that justified the
    direction — auditable, used to weight prose."""

    months_out: int
    direction:  str
    note:       str


@dataclass(frozen=True)
class HorizonRead:
    """A full projection across the horizons we care about.

    `signals` are ordered short-to-long horizon. `dominant` is the
    direction the discipline read at the longest horizon — the
    synthesizer treats this as the headline read."""

    decision: str
    signals:  tuple[HorizonSignal, ...]
    dominant: str  # one of "compounds", "holds", "decays"


def _hits(text: str, lexicon: tuple[str, ...]) -> tuple[str, ...]:
    low = text.lower()
    return tuple(w for w in lexicon if w in low)


def compounding_signal(action: str) -> int:
    """Return +1 (compounds), 0 (neutral), -1 (decays) for an action.

    Cheap signal — uses lexicon hits only. Net positive lexicon
    presence wins. Empty/neutral text returns 0."""
    if not action.strip():
        return 0
    comp_hits = _hits(action, _COMPOUNDING_LEXICON)
    dec_hits  = _hits(action, _DECAYING_LEXICON)
    if len(comp_hits) > len(dec_hits):
        return +1
    if len(dec_hits) > len(comp_hits):
        return -1
    return 0


def _direction_for(action: str, months_out: int) -> tuple[str, str]:
    """Heuristic direction at a horizon. Short horizons are softer —
    everything roughly 'holds' in the first quarter. Longer horizons
    let compounding and decay separate cleanly."""

    signal = compounding_signal(action)

    if months_out <= 3:
        # Near-term: most things look stable regardless of long-run
        # direction. Only surface a strong reading if signal is
        # already obvious.
        if signal > 0:
            return "compounds", "early leverage already visible"
        if signal < 0:
            return "decays", "near-term cost without accrual"
        return "holds", "no near-term swing"

    if signal > 0:
        comp_hits = _hits(action, _COMPOUNDING_LEXICON)
        note = f"compounds via: {','.join(comp_hits[:3])}"
        return "compounds", note

    if signal < 0:
        dec_hits = _hits(action, _DECAYING_LEXICON)
        note = f"decays without continued effort: {','.join(dec_hits[:3])}"
        return "decays", note

    return "holds", "neither clear accrual nor clear decay"


def project(decision: str, horizons: tuple[int, ...] = (3, 6, 24)) -> HorizonRead:
    """Project a decision across the supplied horizons (in months).

    Default horizons are 3 / 6 / 24 months. The synthesizer can pass
    custom horizons via Context — useful when the user's stated
    timeframe is shorter or longer than the default."""

    if not horizons:
        horizons = (3, 6, 24)

    signals = tuple(
        HorizonSignal(months_out=m, direction=d, note=n)
        for m, (d, n) in ((m, _direction_for(decision, m)) for m in horizons)
    )
    dominant = signals[-1].direction if signals else "holds"

    return HorizonRead(
        decision=decision,
        signals=signals,
        dominant=dominant,
    )
