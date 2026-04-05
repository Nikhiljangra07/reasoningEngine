"""
Convergence Protocol — Gibbs Free Energy Analog.

The system moves toward maximum stability and minimum conflict.
Convergence is checked after each complete cycle pass (both Sheng and Ke).

4 Criteria:
1. posterior_stability — Is the Bayesian posterior still shifting meaningfully?
2. dimensional_stability — Has dimensional reduction stabilized?
3. cycle_agreement — Does the Sheng output survive the Ke challenge?
4. energy_minimization — Has the system reached its lowest-energy state?

ISOLATION: Imports ONLY from src.core.types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    ChallengeOutput,
    Domain,
    DomainOutput,
    RootCause,
)


# Convergence thresholds — documented rationale for each:

# Posterior must shift less than 5% between iterations to be considered stable.
# 5% chosen because human problems typically converge within 0.03-0.07 delta
# per iteration after 3+ passes. Below 5% = diminishing returns.
POSTERIOR_STABILITY_THRESHOLD = 0.05

# Dimensional stability: number of new variables per iteration must drop below 2.
# After initial passes, domains recycle known variables. If <2 new variables
# emerge, the dimensional space has stabilized.
DIMENSIONAL_NEW_VARIABLE_THRESHOLD = 2

# Ke cycle scrutiny: average scrutiny score across all Ke pairs must be below 0.4.
# Scrutiny > 0.4 means the controlling cycle is still finding significant issues.
# Below 0.4 = the constructive output is surviving deconstruction.
KE_SCRUTINY_THRESHOLD = 0.4

# Gibbs energy: combined convergence score must exceed 0.75 for convergence.
# This is the master gate — all four criteria contribute to this score.
GIBBS_CONVERGENCE_THRESHOLD = 0.75

# Maximum iterations before forced stop.
# Empirically: 3-domain problems converge in 3-5 passes.
# 5-domain problems expected to converge in 5-8 passes.
# 12 gives headroom without infinite loops.
MAX_ITERATIONS = 12


@dataclass
class ConvergenceSnapshot:
    """State of convergence at a single iteration."""
    iteration: int
    posterior_delta: float              # how much Bayesian posterior shifted
    new_variables_count: int            # how many genuinely new variables emerged
    avg_ke_scrutiny: float              # average scrutiny from Ke cycle challenges
    gibbs_energy: float                 # combined convergence score (0=unstable, 1=converged)
    is_converged: bool
    contributing_scores: dict[str, float]   # breakdown of what contributed to gibbs


@dataclass
class ConvergenceHistory:
    """Full history of convergence across all iterations."""
    snapshots: list[ConvergenceSnapshot] = field(default_factory=list)
    final_converged: bool = False
    total_iterations: int = 0
    forced_stop: bool = False           # True if hit MAX_ITERATIONS without converging


def check_convergence(
    iteration: int,
    current_root_causes: list[RootCause],
    previous_root_causes: list[RootCause],
    current_outputs: dict[Domain, DomainOutput],
    previous_outputs: dict[Domain, DomainOutput],
    ke_results: list[ChallengeOutput],
) -> ConvergenceSnapshot:
    """
    Check convergence after a complete dual-cycle pass.

    Returns a snapshot of the convergence state.
    The orchestrator uses this to decide whether to loop or stop.
    """
    # Criterion 1: Posterior stability
    posterior_delta = _measure_posterior_shift(
        current_root_causes, previous_root_causes
    )
    posterior_stable = posterior_delta < POSTERIOR_STABILITY_THRESHOLD

    # Criterion 2: Dimensional stability
    new_vars = _count_new_variables(current_outputs, previous_outputs)
    dimensional_stable = new_vars < DIMENSIONAL_NEW_VARIABLE_THRESHOLD

    # Criterion 3: Cycle agreement (Ke scrutiny)
    avg_scrutiny = _average_ke_scrutiny(ke_results)
    cycle_agrees = avg_scrutiny < KE_SCRUTINY_THRESHOLD

    # Criterion 4: Energy minimization (Gibbs Free Energy analog)
    gibbs = _calculate_gibbs_energy(
        posterior_delta, new_vars, avg_scrutiny,
        posterior_stable, dimensional_stable, cycle_agrees,
    )

    is_converged = gibbs >= GIBBS_CONVERGENCE_THRESHOLD

    snapshot = ConvergenceSnapshot(
        iteration=iteration,
        posterior_delta=posterior_delta,
        new_variables_count=new_vars,
        avg_ke_scrutiny=avg_scrutiny,
        gibbs_energy=gibbs,
        is_converged=is_converged,
        contributing_scores={
            "posterior_stability": 1.0 - min(posterior_delta / POSTERIOR_STABILITY_THRESHOLD, 1.0),
            "dimensional_stability": 1.0 - min(new_vars / (DIMENSIONAL_NEW_VARIABLE_THRESHOLD * 3), 1.0),
            "cycle_agreement": 1.0 - min(avg_scrutiny / KE_SCRUTINY_THRESHOLD, 1.0) if KE_SCRUTINY_THRESHOLD > 0 else 1.0,
            "gibbs_energy": gibbs,
        },
    )

    return snapshot


# ---------------------------------------------------------------------------
# Criterion 1: Posterior Stability
# ---------------------------------------------------------------------------

def _measure_posterior_shift(
    current: list[RootCause],
    previous: list[RootCause],
) -> float:
    """
    Measure how much the root cause picture shifted between iterations.

    If the top root cause changed or its confidence shifted significantly,
    the posterior is still moving.
    """
    if not previous:
        return 1.0  # first iteration — maximum shift

    if not current:
        return 0.5  # lost all root causes — moderate shift

    # Compare top root causes by name
    current_top = sorted(current, key=lambda r: r.confidence, reverse=True)
    previous_top = sorted(previous, key=lambda r: r.confidence, reverse=True)

    # Did the #1 root cause change?
    if current_top[0].variable.name != previous_top[0].variable.name:
        return 0.8  # major shift — different root cause on top

    # Same root cause — measure confidence delta
    conf_delta = abs(current_top[0].confidence - previous_top[0].confidence)

    # Also check if the ranking order changed
    current_names = [r.variable.name for r in current_top[:5]]
    previous_names = [r.variable.name for r in previous_top[:5]]
    order_matches = sum(1 for c, p in zip(current_names, previous_names) if c == p)
    order_stability = order_matches / max(len(current_names), len(previous_names), 1)

    return conf_delta * 0.6 + (1.0 - order_stability) * 0.4


# ---------------------------------------------------------------------------
# Criterion 2: Dimensional Stability
# ---------------------------------------------------------------------------

def _count_new_variables(
    current: dict[Domain, DomainOutput],
    previous: dict[Domain, DomainOutput],
) -> int:
    """
    Count how many genuinely new variables emerged in this iteration.

    If domains are just restating known variables, the dimensional
    space has stabilized.
    """
    if not previous:
        # First iteration — everything is new
        total = 0
        for output in current.values():
            for p in output.perspectives:
                total += len(p.variables_found)
        return total

    # Collect all previous variable names
    prev_names = set()
    for output in previous.values():
        for p in output.perspectives:
            for v in p.variables_found:
                prev_names.add(v.name)

    # Count variables in current that weren't in previous
    new_count = 0
    for output in current.values():
        for p in output.perspectives:
            for v in p.variables_found:
                if v.name not in prev_names:
                    new_count += 1

    return new_count


# ---------------------------------------------------------------------------
# Criterion 3: Cycle Agreement
# ---------------------------------------------------------------------------

def _average_ke_scrutiny(ke_results: list[ChallengeOutput]) -> float:
    """
    Average scrutiny score across all Ke cycle challenges.

    Low scrutiny = the Sheng output is surviving Ke deconstruction.
    High scrutiny = significant issues remain.
    """
    if not ke_results:
        return 0.5  # no Ke results = moderate uncertainty

    return sum(r.scrutiny_score for r in ke_results) / len(ke_results)


# ---------------------------------------------------------------------------
# Criterion 4: Gibbs Free Energy
# ---------------------------------------------------------------------------

def _calculate_gibbs_energy(
    posterior_delta: float,
    new_vars: int,
    avg_scrutiny: float,
    posterior_stable: bool,
    dimensional_stable: bool,
    cycle_agrees: bool,
) -> float:
    """
    Calculate the Gibbs Free Energy analog.

    The system moves toward maximum stability (G = 1.0) and
    minimum conflict (G = 1.0). Convergence = G ≥ threshold.

    G = weighted combination of all four criteria.
    """
    # Posterior contribution (weight: 0.30)
    # Lower delta = more stable = higher G
    posterior_score = max(0.0, 1.0 - (posterior_delta / 0.5))

    # Dimensional contribution (weight: 0.20)
    # Fewer new variables = more stable
    dim_score = max(0.0, 1.0 - (new_vars / 10.0))

    # Cycle agreement contribution (weight: 0.30)
    # Lower scrutiny = Sheng survived Ke
    cycle_score = max(0.0, 1.0 - avg_scrutiny)

    # Boolean bonus (weight: 0.20)
    # All three boolean criteria met = bonus
    bool_count = sum([posterior_stable, dimensional_stable, cycle_agrees])
    bool_score = bool_count / 3.0

    gibbs = (
        posterior_score * 0.30
        + dim_score * 0.20
        + cycle_score * 0.30
        + bool_score * 0.20
    )

    return min(gibbs, 1.0)
