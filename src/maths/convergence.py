"""
Maths Layer 5: Convergence & Stopping

The engine doesn't squeeze forever. It knows exactly when adding
more perspectives stops changing the answer.

4 Methods:
1. Elbow Method — diminishing returns detection
2. Occam's Razor — parsimony penalty (AIC/BIC)
3. Convergence Check — has the picture settled?
4. SVD Ranking — which perspectives carry the most weight?
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import FrameworkID, Perspective, Variable


@dataclass
class SVDRanking:
    """A perspective ranked by its singular value — how much it matters."""
    perspective: Perspective
    singular_value: float       # 0.0 to 1.0 — weight of this perspective
    rank: int                   # 1 = most important
    cumulative_info: float      # cumulative information explained up to this rank


@dataclass
class ConvergenceState:
    """The current state of the convergence check."""
    has_converged: bool
    iterations: int
    stability_score: float              # 0.0 to 1.0 — how stable is the picture?
    svd_rankings: list[SVDRanking]      # perspectives ranked by importance
    pillars: list[Perspective]          # the core perspectives (top SVD)
    decorations: list[Perspective]      # perspectives that add <5% information
    elbow_reached: bool                 # has diminishing returns kicked in?
    parsimony_score: float              # lower = simpler = better


def check_convergence(
    perspectives: list[Perspective],
    previous_state: ConvergenceState | None = None,
) -> ConvergenceState:
    """
    Check if the reasoning has converged.

    Convergence means: adding more perspectives doesn't change the picture.
    Uses all 4 methods to determine this.
    """
    iteration = (previous_state.iterations + 1) if previous_state else 1

    # Step 1: SVD Ranking — rank perspectives by weight
    rankings = _svd_rank(perspectives)

    # Step 2: Identify pillars (top contributors) and decorations (noise)
    pillars, decorations = _split_pillars_decorations(rankings)

    # Step 3: Elbow check — have we hit diminishing returns?
    elbow_reached = _elbow_check(rankings)

    # Step 4: Parsimony score — how complex is the current explanation?
    parsimony = _parsimony_score(perspectives)

    # Step 5: Stability check — has the picture changed since last iteration?
    stability = _stability_score(rankings, previous_state)

    # Convergence decision
    has_converged = (
        stability >= 0.85           # picture is 85% stable
        and elbow_reached           # diminishing returns reached
        and len(pillars) >= 2       # at least 2 core perspectives
    )

    return ConvergenceState(
        has_converged=has_converged,
        iterations=iteration,
        stability_score=stability,
        svd_rankings=rankings,
        pillars=pillars,
        decorations=decorations,
        elbow_reached=elbow_reached,
        parsimony_score=parsimony,
    )


# ---------------------------------------------------------------------------
# Method 1: SVD Ranking
# ---------------------------------------------------------------------------

def _svd_rank(perspectives: list[Perspective]) -> list[SVDRanking]:
    """
    Rank perspectives by their "singular value" — how much they
    contribute to the total picture.

    Weight factors:
    - perspective.weight (from the framework)
    - number of variables found
    - average confidence of variables
    - presence of hidden variables (bonus)
    """
    scored = []

    for p in perspectives:
        if not p.variables_found:
            sv = p.weight * 0.3  # framework ran but found nothing
        else:
            var_count_factor = min(len(p.variables_found) / 5.0, 1.0)
            avg_confidence = sum(v.confidence for v in p.variables_found) / len(p.variables_found)
            hidden_bonus = 0.2 if any(v.is_hidden for v in p.variables_found) else 0.0
            avg_magnitude = sum(v.magnitude for v in p.variables_found) / len(p.variables_found)

            sv = (
                p.weight * 0.3
                + avg_confidence * 0.25
                + avg_magnitude * 0.2
                + var_count_factor * 0.15
                + hidden_bonus * 0.1
            )

        scored.append((p, min(sv, 1.0)))

    # Sort by singular value descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build rankings with cumulative information
    rankings = []
    cumulative = 0.0
    total_sv = sum(sv for _, sv in scored) or 1.0

    for rank, (perspective, sv) in enumerate(scored, 1):
        proportion = sv / total_sv
        cumulative += proportion
        rankings.append(SVDRanking(
            perspective=perspective,
            singular_value=sv,
            rank=rank,
            cumulative_info=cumulative,
        ))

    return rankings


# ---------------------------------------------------------------------------
# Method 2: Pillar/Decoration split
# ---------------------------------------------------------------------------

def _split_pillars_decorations(
    rankings: list[SVDRanking],
) -> tuple[list[Perspective], list[Perspective]]:
    """
    Split perspectives into pillars (core) and decorations (noise).

    Pillars: perspectives that cumulatively explain 95% of the information.
    Decorations: the remaining 5%.
    """
    pillars = []
    decorations = []

    for r in rankings:
        if r.cumulative_info <= 0.95:
            pillars.append(r.perspective)
        else:
            decorations.append(r.perspective)

    # Ensure at least one pillar
    if not pillars and rankings:
        pillars.append(rankings[0].perspective)

    return pillars, decorations


# ---------------------------------------------------------------------------
# Method 3: Elbow check
# ---------------------------------------------------------------------------

def _elbow_check(rankings: list[SVDRanking]) -> bool:
    """
    Detect if the "elbow" has been reached — where adding more
    perspectives yields diminishing returns.

    The elbow is where the drop in singular value between consecutive
    ranks exceeds 50% of the previous drop.
    """
    if len(rankings) < 3:
        return True  # too few perspectives — elbow is trivially reached

    drops = []
    for i in range(1, len(rankings)):
        drop = rankings[i - 1].singular_value - rankings[i].singular_value
        drops.append(drop)

    # Find where the drop accelerates (the "knee")
    for i in range(1, len(drops)):
        if drops[i - 1] > 0 and drops[i] / drops[i - 1] > 1.5:
            return True  # sharp drop acceleration = elbow

    # Also check if the top 3 carry >80% of information
    if len(rankings) >= 3 and rankings[2].cumulative_info > 0.80:
        return True

    return False


# ---------------------------------------------------------------------------
# Method 4: Parsimony (Occam's Razor)
# ---------------------------------------------------------------------------

def _parsimony_score(perspectives: list[Perspective]) -> float:
    """
    Calculate parsimony score — lower is simpler is better.

    Penalizes:
    - Too many active perspectives
    - Variables with low confidence (uncertain = complex)
    - Redundant variables (same name across perspectives)
    """
    if not perspectives:
        return 0.0

    # Penalty for number of perspectives
    count_penalty = len(perspectives) / 20.0  # normalized: 20 perspectives = 1.0 penalty

    # Penalty for low-confidence variables
    all_vars = []
    for p in perspectives:
        all_vars.extend(p.variables_found)

    if all_vars:
        avg_confidence = sum(v.confidence for v in all_vars) / len(all_vars)
        confidence_penalty = 1.0 - avg_confidence
    else:
        confidence_penalty = 0.5

    # Penalty for redundant variables (same name appears multiple times)
    var_names = [v.name for v in all_vars]
    unique_names = set(var_names)
    redundancy_penalty = (
        (len(var_names) - len(unique_names)) / len(var_names)
        if var_names else 0.0
    )

    return (count_penalty * 0.4 + confidence_penalty * 0.3 + redundancy_penalty * 0.3)


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------

def _stability_score(
    current_rankings: list[SVDRanking],
    previous_state: ConvergenceState | None,
) -> float:
    """
    Check how much the picture has changed since the last iteration.

    If the top pillars haven't changed, the picture is stable.
    High stability (>0.85) + elbow reached = convergence.
    """
    if previous_state is None:
        return 0.5  # first iteration — stability is unknown

    if not previous_state.svd_rankings or not current_rankings:
        return 0.5

    # Compare top 3 frameworks between iterations
    prev_top = [r.perspective.framework for r in previous_state.svd_rankings[:3]]
    curr_top = [r.perspective.framework for r in current_rankings[:3]]

    matches = sum(1 for f in curr_top if f in prev_top)
    framework_stability = matches / 3.0

    # Compare top singular values
    prev_svs = [r.singular_value for r in previous_state.svd_rankings[:3]]
    curr_svs = [r.singular_value for r in current_rankings[:3]]

    sv_diffs = []
    for p, c in zip(prev_svs, curr_svs):
        if p > 0:
            sv_diffs.append(abs(p - c) / p)
    sv_stability = 1.0 - (sum(sv_diffs) / len(sv_diffs)) if sv_diffs else 0.5

    return framework_stability * 0.6 + sv_stability * 0.4
