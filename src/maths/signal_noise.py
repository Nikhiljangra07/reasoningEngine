"""
Maths Layer 1: Signal vs. Noise (Contextual Relevance)

Every domain generates output. This layer decides what's signal for
THIS problem and stores the rest — because context changes, and
today's noise is tomorrow's key variable.

Nothing is useless; it's just waiting for the right equation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    Direction,
    DomainOutput,
    Perspective,
    Problem,
    SignalType,
    Variable,
)


@dataclass
class SignalNoiseResult:
    """Result of signal/noise filtering on a set of perspectives."""
    signal: list[Perspective]       # relevant to current problem
    noise: list[Perspective]        # not relevant now, stored for later
    latent: list[Perspective]       # looks like noise but may contain variable D
    orthogonal: list[Perspective]   # belongs to a different axis of the same reality


def filter_perspectives(
    problem: Problem,
    perspectives: list[Perspective],
) -> SignalNoiseResult:
    """
    Filter perspectives into signal, noise, latent, and orthogonal.

    Signal: Variables that directly relate to the problem's stated forces.
    Noise: Variables with low relevance and low magnitude.
    Latent: Variables that look irrelevant but have high confidence or
            contradict the current picture — potential hidden gold.
    Orthogonal: Variables on a different axis — not useless, just for
                a different equation.
    """
    signal = []
    noise = []
    latent = []
    orthogonal = []

    # Build the problem's "frequency" — what directions and magnitudes matter
    problem_directions = set()
    problem_magnitude_avg = 0.0
    if problem.variables:
        problem_directions = {v.direction for v in problem.variables}
        problem_magnitude_avg = (
            sum(v.magnitude for v in problem.variables) / len(problem.variables)
        )

    for perspective in perspectives:
        score = _relevance_score(perspective, problem, problem_directions, problem_magnitude_avg)

        if score >= 0.6:
            perspective.signal_type = SignalType.SIGNAL
            signal.append(perspective)
        elif score >= 0.3:
            # Medium relevance — check if it's latent gold
            if _is_latent_gold(perspective, problem):
                perspective.signal_type = SignalType.LATENT
                latent.append(perspective)
            elif _is_orthogonal(perspective, problem_directions):
                perspective.signal_type = SignalType.NOISE
                orthogonal.append(perspective)
            else:
                perspective.signal_type = SignalType.NOISE
                noise.append(perspective)
        else:
            # Low relevance — but check for latent indicators
            if _is_latent_gold(perspective, problem):
                perspective.signal_type = SignalType.LATENT
                latent.append(perspective)
            else:
                perspective.signal_type = SignalType.NOISE
                noise.append(perspective)

    return SignalNoiseResult(
        signal=signal,
        noise=noise,
        latent=latent,
        orthogonal=orthogonal,
    )


def _relevance_score(
    perspective: Perspective,
    problem: Problem,
    problem_directions: set[Direction],
    problem_magnitude_avg: float,
) -> float:
    """
    Score how relevant a perspective is to the current problem.

    Factors:
    - Do its variables share directions with the problem's forces?
    - Are its variables of similar magnitude (on the same scale)?
    - Does it contain hidden variables? (always relevant)
    - How many variables did it find? (more = more informative)
    """
    if not perspective.variables_found:
        return 0.3  # no variables = moderate baseline (framework ran but found nothing)

    score = 0.0
    weights_total = 0.0

    for var in perspective.variables_found:
        w = 1.0

        # Hidden variables are always high-signal
        if var.is_hidden:
            score += 0.9 * w
            weights_total += w
            continue

        # Direction alignment with problem
        direction_match = 0.5 if var.direction in problem_directions else 0.2
        if var.direction == Direction.CIRCULAR:
            direction_match = 0.7  # circular = feedback loop = always important

        # Magnitude relevance — variables near the problem's average are more relevant
        if problem_magnitude_avg > 0:
            magnitude_proximity = 1.0 - abs(var.magnitude - problem_magnitude_avg)
        else:
            magnitude_proximity = 0.5

        # Confidence — higher confidence = more reliable signal
        confidence_factor = var.confidence

        var_score = (
            direction_match * 0.4
            + magnitude_proximity * 0.3
            + confidence_factor * 0.3
        )
        score += var_score * w
        weights_total += w

    return score / weights_total if weights_total > 0 else 0.3


def _is_latent_gold(perspective: Perspective, problem: Problem) -> bool:
    """
    Check if a perspective looks like noise but is actually hidden gold.

    Latent indicators:
    - Contains a variable that contradicts the problem's stated picture
    - High confidence variable in an unexpected direction
    - Variable that would explain a gap in the current analysis
    """
    for var in perspective.variables_found:
        # High confidence but unexpected direction = potential hidden variable
        if var.confidence > 0.7 and var.direction not in {
            v.direction for v in problem.variables
        }:
            return True

        # Hidden variable with any confidence = always latent gold
        if var.is_hidden and var.confidence > 0.3:
            return True

    return False


def _is_orthogonal(
    perspective: Perspective, problem_directions: set[Direction]
) -> bool:
    """
    Check if a perspective is orthogonal — on a different axis entirely.

    Orthogonal perspectives aren't wrong, they're just tuned to a
    different frequency. They might become signal if the problem shifts.
    """
    if not perspective.variables_found:
        return False

    # If ALL variables are in directions not present in the problem, it's orthogonal
    var_directions = {v.direction for v in perspective.variables_found}
    overlap = var_directions & problem_directions

    return len(overlap) == 0 and len(var_directions) > 0
