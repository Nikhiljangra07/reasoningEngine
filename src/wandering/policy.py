"""
Wandering policy — chaos walk with anchor pull.

This module decides: given an agent's current position and accumulated
trace, where does it go NEXT?

The math is intentionally NOT optimization. Per Law 1: chaos IS the
feature. No learned priors, no Thompson sampling, no smart routing.

The policy has two impulses, balanced at every step:

  GRAVITY (toward anchor):   weight ~ 0.5
      Pull toward zones with structural overlap with the cushion.
      Prevents the agent from forgetting the problem.

  CHAOS (random jump):        weight ~ 0.5
      Pick a random direction, weighted by INVERSE visit count.
      Prevents the agent from getting stuck in adjacent fields.

The DOMAIN is unconstrained (Law 1 explicitly). Agents can go anywhere —
movies, jokes, ancient philosophy, sports. The chaos impulse uses a
broad seed domain list that spans human knowledge; per-step it picks
from there with inverse-frequency weighting (rarely-visited domains
preferred over recently-visited).

Drift detection: if the agent's last N positions have all been
semantically far from the cushion (no matches in many steps), the policy
forces a return-to-anchor on the next step. This is the structural
enforcement of "the anchor never moves" — it's not just policy, it's
math (Law 1 amended).

ISOLATION: imports cushion + trace types only. No LLM, no I/O. The
randomness is seedable for testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from src.wandering.cushion import CushionGraph
from src.wandering.trace import DecisionTrace, StepKind


log = logging.getLogger("constellax.wandering.policy")


# ---------------------------------------------------------------------------
# Seed domain list — the universe agents wander across
# ---------------------------------------------------------------------------

#: Broad, intentionally diverse seed list. Per Law 1, agents can be
#: ANYWHERE — this list isn't exhaustive, just a starting palette to
#: trigger cross-domain leaps. The policy picks from here weighted by
#: inverse visit-count.
#:
#: Editors: add domains liberally. Reducing diversity hurts the design.
SEED_DOMAINS: tuple[str, ...] = (
    # natural sciences
    "physics", "biology", "chemistry", "ecology", "neuroscience", "astronomy",
    # formal sciences
    "mathematics", "logic", "computer_science", "information_theory",
    # social sciences
    "psychology", "sociology", "anthropology", "economics", "linguistics",
    # humanities
    "history", "philosophy", "religion", "mythology", "literature",
    # arts
    "music", "film", "theater", "visual_arts", "poetry", "dance", "architecture",
    # craft & practice
    "cooking", "gardening", "carpentry", "textile_arts", "blacksmithing",
    # culture & lifeworld
    "sports", "games", "humor", "dark_humor", "memes", "advertising",
    # business & strategy
    "military_strategy", "diplomacy", "negotiation", "trade", "marketing",
    # spiritual & wisdom
    "taoism", "buddhism", "stoicism", "zen", "mysticism", "indigenous_traditions",
    # everyday & vernacular
    "parenting", "education", "relationships", "personal_finance", "habits",
    # technical & operational
    "engineering", "manufacturing", "logistics", "agriculture", "medicine",
    # boundary & edge
    "biographies", "investigative_journalism", "research_papers", "patents",
)


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

DRIFT_WINDOW = 4  # check the last N MATCHED steps
DRIFT_THRESHOLD = 0  # if total matched_count across window == 0 → drift


def detect_drift(trace: DecisionTrace) -> bool:
    """Return True if the agent has drifted too far from the anchor.

    Drift heuristic (intentionally simple for V0): scan the last
    DRIFT_WINDOW steps of kind MATCHED. If their summed matched_count is
    zero (no resonance found anywhere recently), the agent has drifted.

    Triggers a forced return-to-anchor on the next step.

    Note: the policy stays chaotic INSIDE the bounded radius. Drift
    detection is the outer boundary, not the inner search heuristic.
    """
    matched_steps = trace.steps_of(StepKind.MATCHED)
    if len(matched_steps) < DRIFT_WINDOW:
        return False
    recent = matched_steps[-DRIFT_WINDOW:]
    return sum(s.matched_count for s in recent) <= DRIFT_THRESHOLD


# ---------------------------------------------------------------------------
# Domain selection
# ---------------------------------------------------------------------------


def domain_visit_counts(trace: DecisionTrace) -> dict[str, int]:
    """Count how many MATCHED/DUG steps the agent has performed per domain.

    The trace's TraceStep.position field carries domain labels. We treat
    any non-empty position as one visit to that domain.
    """
    counts: dict[str, int] = {}
    for step in trace.steps:
        pos = step.position.strip()
        if pos and step.kind in (StepKind.MATCHED, StepKind.DUG, StepKind.FETCHED):
            counts[pos] = counts.get(pos, 0) + 1
    return counts


def pick_next_domain(
    trace: DecisionTrace,
    seed_domains: tuple[str, ...] = SEED_DOMAINS,
    random_fn: Callable[[float, float], float] | None = None,
    choice_fn: Callable[[list[str], list[float]], str] | None = None,
) -> str:
    """Pick the next domain to visit, weighted by inverse visit-count.

    Domains never visited get the highest weight. Recently-visited domains
    get less. This is the CHAOS impulse — it prefers genuine novelty in
    domain space without optimizing the walk inside any given domain.

    `random_fn` and `choice_fn` are injection points for tests. By default
    we use the stdlib `random` module. The policy is seedable for
    reproducibility, but in production we use real randomness — anything
    else would be "smart routing" and violate Law 1.
    """
    import random as _random

    if random_fn is None:
        random_fn = _random.uniform
    if choice_fn is None:
        def _default_choice(items: list[str], weights: list[float]) -> str:
            return _random.choices(items, weights=weights, k=1)[0]
        choice_fn = _default_choice

    visits = domain_visit_counts(trace)
    weights = []
    for domain in seed_domains:
        v = visits.get(domain, 0)
        # Inverse-frequency: never-visited = weight 2.0, each visit halves it.
        # Floor at 0.1 so heavily-visited domains aren't completely impossible
        # (chaos respects the long tail).
        w = max(0.1, 2.0 / (1.0 + v))
        weights.append(w)

    return choice_fn(list(seed_domains), weights)


# ---------------------------------------------------------------------------
# Top-level policy decision
# ---------------------------------------------------------------------------


@dataclass
class NextMove:
    """The policy's recommendation for the agent's next step.

    `kind` is the operation kind (typically FETCHED or RETURNED_TO_ANCHOR).
    `position` is the next domain / topic to investigate.
    `rationale` is a one-line human-readable reason (logged in trace).
    """

    kind: StepKind
    position: str
    rationale: str


def next_move(
    cushion: CushionGraph,
    trace: DecisionTrace,
    seed_domains: tuple[str, ...] = SEED_DOMAINS,
) -> NextMove:
    """Decide the agent's next move based on cushion + trace.

    Returns:
      NextMove(RETURNED_TO_ANCHOR, ...) if drift detected
      NextMove(FETCHED, <domain>, ...) for a fresh chaotic walk step

    The cushion parameter is currently used only for context (could later
    inform domain selection — but per Law 1, we don't do that). We accept
    it now so the API doesn't break when we add features that DO need it.
    """
    if detect_drift(trace):
        return NextMove(
            kind=StepKind.RETURNED_TO_ANCHOR,
            position=cushion.raw_input.problem.content[:80] or "(anchor)",
            rationale=(
                f"drift detected: last {DRIFT_WINDOW} match steps yielded "
                f"<= {DRIFT_THRESHOLD} hits"
            ),
        )

    domain = pick_next_domain(trace, seed_domains=seed_domains)
    return NextMove(
        kind=StepKind.FETCHED,
        position=domain,
        rationale=f"chaos pick: domain {domain!r} weighted by inverse visits",
    )


__all__ = [
    "SEED_DOMAINS",
    "DRIFT_WINDOW",
    "DRIFT_THRESHOLD",
    "detect_drift",
    "domain_visit_counts",
    "pick_next_domain",
    "NextMove",
    "next_move",
]
