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
import os
from dataclasses import dataclass, field
from typing import Callable

# Diversity steering strength (Phase 3, 2026-06-17). When a peer agent has
# already covered a domain, its pick-weight is multiplied by this factor —
# STEERING the swarm off already-covered ground to fight homogeneity collapse
# (the strongest-evidenced multi-agent failure: surface diversity collapses, so
# structural steering is the real lever). Stronger than the old 0.5. It is a
# WEIGHT, never an exclusion (floor below keeps it > 0), so it stays flow-not-
# judge (chaos law intact) and never STARVES a genuinely productive niche — which
# would fight the governor's REALLOCATE/exploit signal. Tune via env.
_NOTICEBOARD_DOWNWEIGHT = float(os.environ.get("WANDER_NOTICEBOARD_DOWNWEIGHT", "0.35"))
_NOTICEBOARD_FLOOR = float(os.environ.get("WANDER_NOTICEBOARD_FLOOR", "0.1"))

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
#:
#: RUNTIME NARROWING (control room): when the env var WANDER_SEED_DOMAINS
#: is set to a comma-separated list (e.g. "physics,mathematics"), the
#: policy restricts the wander to ONLY those domains. This is set by the
#: control room (scripts/control_room.py) via the runner. When the env
#: var is unset/empty, the full palette below is used (default behavior,
#: unchanged). The override only applies to callers using the default
#: seed_domains — an explicit seed_domains argument is always respected.
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
# Runtime domain narrowing (control room)
# ---------------------------------------------------------------------------

def _resolve_seed_domains(seed_domains: tuple[str, ...]) -> tuple[str, ...]:
    """Apply the WANDER_SEED_DOMAINS env override when present.

    Only narrows when the caller used the DEFAULT SEED_DOMAINS (identity
    check) — an explicit seed_domains argument is always respected, so
    tests and special callers are never silently overridden.

    Env format: comma-separated domain names, e.g. "physics,mathematics".
    Unknown names (not in SEED_DOMAINS) are dropped with a warning. If the
    override resolves to an empty set, the full palette is kept (a typo
    should not produce a zero-domain wander that finds nothing).
    """
    import os
    if seed_domains is not SEED_DOMAINS:
        return seed_domains  # caller narrowed explicitly; respect it
    raw = os.environ.get("WANDER_SEED_DOMAINS", "").strip()
    if not raw:
        return seed_domains  # no override → full palette
    requested = [d.strip() for d in raw.split(",") if d.strip()]
    valid = tuple(d for d in requested if d in SEED_DOMAINS)
    unknown = [d for d in requested if d not in SEED_DOMAINS]
    if unknown:
        log.warning(
            "WANDER_SEED_DOMAINS: ignoring unknown domain(s) %s "
            "(not in SEED_DOMAINS)", unknown,
        )
    if not valid:
        log.warning(
            "WANDER_SEED_DOMAINS resolved to zero valid domains; "
            "keeping full palette to avoid a no-op wander",
        )
        return seed_domains
    log.info("WANDER_SEED_DOMAINS narrowing wander to: %s", list(valid))
    return valid


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
    *,
    noticeboard_covered_domains: set[str] | None = None,
) -> str:
    """Pick the next domain to visit, weighted by inverse visit-count.

    Domains never visited get the highest weight. Recently-visited domains
    get less. This is the CHAOS impulse — it prefers genuine novelty in
    domain space without optimizing the walk inside any given domain.

    `random_fn` and `choice_fn` are injection points for tests. By default
    we use the stdlib `random` module. The policy is seedable for
    reproducibility, but in production we use real randomness — anything
    else would be "smart routing" and violate Law 1.

    WANDER_AGENT_NOTICEBOARD test scaffold (June 2026): when the optional
    `noticeboard_covered_domains` set is supplied (domains other agents
    have recently posted notices about), their weight is HALVED — a soft
    hint to prefer uncovered or complementary territory, NOT an exclusion.
    Heavily-noticed domains stay possible (the chaos floor still holds);
    they just become less likely. Each covered domain's halving is
    independent of its visit count, so a never-visited-by-me-but-covered-
    by-peers domain still gets a real-but-discounted chance.
    """
    import random as _random

    if random_fn is None:
        random_fn = _random.uniform
    if choice_fn is None:
        def _default_choice(items: list[str], weights: list[float]) -> str:
            return _random.choices(items, weights=weights, k=1)[0]
        choice_fn = _default_choice

    # Control-room narrowing: restrict to WANDER_SEED_DOMAINS when set.
    seed_domains = _resolve_seed_domains(seed_domains)

    covered: set[str] = noticeboard_covered_domains or set()
    visits = domain_visit_counts(trace)
    weights = []
    for domain in seed_domains:
        v = visits.get(domain, 0)
        # Inverse-frequency: never-visited = weight 2.0, each visit halves it.
        # Floor at 0.1 so heavily-visited domains aren't completely impossible
        # (chaos respects the long tail).
        w = max(0.1, 2.0 / (1.0 + v))
        # Noticeboard soft-downweight: scale weight down if a peer agent already
        # posted a notice about this domain — STEERING the swarm off covered
        # ground to fight homogeneity. Bounded by the chaos floor so it never
        # goes to zero (steering, not exclusion → chaos-safe, never starves a
        # productive niche the governor might want to REALLOCATE into).
        if domain in covered:
            w = max(_NOTICEBOARD_FLOOR, w * _NOTICEBOARD_DOWNWEIGHT)
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
    *,
    noticeboard_covered_domains: set[str] | None = None,
) -> NextMove:
    """Decide the agent's next move based on cushion + trace.

    Returns:
      NextMove(RETURNED_TO_ANCHOR, ...) if drift detected
      NextMove(FETCHED, <domain>, ...) for a fresh chaotic walk step

    The cushion parameter is currently used only for context (could later
    inform domain selection — but per Law 1, we don't do that). We accept
    it now so the API doesn't break when we add features that DO need it.

    WANDER_AGENT_NOTICEBOARD test scaffold (June 2026): `noticeboard_covered_
    domains` is the optional set of domains other agents have recently
    posted notices about. When supplied, pick_next_domain halves the
    weight of covered domains (soft hint, not exclusion) so the cohort
    spreads coverage. None = legacy behavior unchanged.
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

    domain = pick_next_domain(
        trace,
        seed_domains=seed_domains,
        noticeboard_covered_domains=noticeboard_covered_domains,
    )
    nb_note = (
        f" (noticeboard-aware; {len(noticeboard_covered_domains)} covered)"
        if noticeboard_covered_domains else ""
    )
    return NextMove(
        kind=StepKind.FETCHED,
        position=domain,
        rationale=f"chaos pick: domain {domain!r} weighted by inverse visits{nb_note}",
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
