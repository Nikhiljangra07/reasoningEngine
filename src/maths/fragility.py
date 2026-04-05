"""
Maths Layer 9: Ergodicity & Fragility (The Final Stress Test)

The quality control layer. After convergence captures variable y,
the system runs one final inspection: does the solution survive
the real world for THIS person in THIS moment?

This is Stage 5 of the Formation.

4 Operations:
1. Antifragility Assessment — breaks under stress, survives, or gets stronger?
2. Ergodicity Check — does the average outcome apply to THIS person?
3. Tail Risk Evaluation — what's the catastrophic downside?
4. Real-World Durability — can the user actually execute this?
"""

from __future__ import annotations

from src.core.types import (
    Consequence,
    FragilityRating,
    FragilityResult,
    Problem,
    RootCause,
    Severity,
    Variable,
)


def stress_test(
    problem: Problem,
    root_cause: RootCause,
    consequences: list[Consequence],
) -> FragilityResult:
    """
    Run the final stress test on the root cause and its consequences.

    This is the last gate before output. If the solution is fragile,
    it gets flagged. If the tail risk is catastrophic, it gets surfaced.
    If the user can't actually execute it, we say so.
    """
    # Step 1: Antifragility assessment
    rating, rating_reasoning = _assess_antifragility(root_cause, consequences)

    # Step 2: Ergodicity check
    is_ergodic, ergodic_reasoning = _check_ergodicity(root_cause, consequences)

    # Step 3: Tail risk evaluation
    tail_risk, tail_severity = _evaluate_tail_risk(consequences)

    # Step 4: Real-world durability
    executable, blockers = _check_executability(problem, root_cause)

    # Combine reasoning
    full_reasoning = (
        f"ANTIFRAGILITY: {rating_reasoning}\n"
        f"ERGODICITY: {ergodic_reasoning}\n"
        f"TAIL RISK: {tail_risk or 'None identified'}\n"
        f"EXECUTABLE: {'Yes' if executable else 'No — ' + '; '.join(blockers)}"
    )

    return FragilityResult(
        rating=rating,
        reasoning=full_reasoning,
        is_ergodic=is_ergodic,
        tail_risk=tail_risk,
        tail_risk_severity=tail_severity,
        real_world_executable=executable,
        execution_blockers=blockers,
    )


# ---------------------------------------------------------------------------
# Step 1: Antifragility
# ---------------------------------------------------------------------------

def _assess_antifragility(
    root_cause: RootCause,
    consequences: list[Consequence],
) -> tuple[FragilityRating, str]:
    """
    Assess whether the solution is fragile, robust, or antifragile.

    Fragile: high-confidence root cause but consequences are irreversible
             and depend on everything going right.
    Robust: root cause is clear and consequences are manageable.
    Antifragile: addressing the root cause opens new opportunities
                 and the solution gets stronger under pressure.
    """
    # Factors toward fragility
    irreversible_consequences = [c for c in consequences if not c.is_reversible]
    high_severity = [c for c in consequences if c.severity in (Severity.HIGH, Severity.CRITICAL)]
    low_confidence_root = root_cause.confidence < 0.5

    fragility_score = 0.0

    # Irreversible consequences = fragile
    if irreversible_consequences:
        fragility_score += 0.3 * len(irreversible_consequences) / max(len(consequences), 1)

    # High severity + many consequences = fragile
    if high_severity:
        fragility_score += 0.2 * len(high_severity) / max(len(consequences), 1)

    # Low confidence in root cause = fragile (we might be wrong)
    if low_confidence_root:
        fragility_score += 0.3

    # Factors toward antifragility
    reversible_consequences = [c for c in consequences if c.is_reversible]
    high_confidence_root = root_cause.confidence > 0.75

    antifragile_score = 0.0

    # High confidence root + reversible consequences = robust or antifragile
    if high_confidence_root:
        antifragile_score += 0.3

    if reversible_consequences:
        antifragile_score += 0.2 * len(reversible_consequences) / max(len(consequences), 1)

    # Root cause is hidden = addressing it opens new possibilities (antifragile)
    if root_cause.variable.is_hidden:
        antifragile_score += 0.2

    # Decision
    if fragility_score > 0.5:
        return (
            FragilityRating.FRAGILE,
            f"FRAGILE (score: {fragility_score:.2f}). "
            f"{len(irreversible_consequences)} irreversible consequences, "
            f"root confidence: {root_cause.confidence:.2f}. "
            "This solution breaks if conditions change."
        )
    elif antifragile_score > 0.5:
        return (
            FragilityRating.ANTIFRAGILE,
            f"ANTIFRAGILE (score: {antifragile_score:.2f}). "
            "Addressing this root cause opens new paths. "
            "The solution gets stronger under pressure."
        )
    else:
        return (
            FragilityRating.ROBUST,
            f"ROBUST (fragility: {fragility_score:.2f}, antifragility: {antifragile_score:.2f}). "
            "This solution survives stress but doesn't gain from it."
        )


# ---------------------------------------------------------------------------
# Step 2: Ergodicity
# ---------------------------------------------------------------------------

def _check_ergodicity(
    root_cause: RootCause,
    consequences: list[Consequence],
) -> tuple[bool, str]:
    """
    Check ergodicity — does the average outcome apply to THIS person?

    Non-ergodic situations: where the 10% failure means total ruin.
    A 90% success rate is meaningless if failure is catastrophic
    and non-recoverable.
    """
    # Check for non-ergodic indicators
    catastrophic_consequences = [
        c for c in consequences
        if c.severity == Severity.CRITICAL and not c.is_reversible
    ]

    high_probability_but_catastrophic = [
        c for c in consequences
        if c.probability > 0.1 and c.severity == Severity.CRITICAL
    ]

    if catastrophic_consequences:
        return (
            False,
            f"NON-ERGODIC. {len(catastrophic_consequences)} consequence(s) are both "
            "critical and irreversible. The average outcome does NOT apply — "
            "this person cannot recover from the downside. "
            "Expected value calculations are misleading here."
        )

    if high_probability_but_catastrophic:
        return (
            False,
            f"NON-ERGODIC. {len(high_probability_but_catastrophic)} critical consequence(s) "
            f"have >10% probability. Even a small chance of catastrophic, "
            "irreversible failure makes the average outcome meaningless "
            "for this specific person."
        )

    # Check root cause confidence — low confidence on a high-stakes problem is non-ergodic
    if root_cause.confidence < 0.5 and any(
        c.severity in (Severity.HIGH, Severity.CRITICAL) for c in consequences
    ):
        return (
            False,
            "BORDERLINE NON-ERGODIC. Root cause confidence is low "
            f"({root_cause.confidence:.2f}) but stakes are high. "
            "If we're wrong about the root cause, the consequences are severe."
        )

    return (
        True,
        "ERGODIC. The average outcome reasonably applies to this person. "
        "No catastrophic, irreversible consequences detected."
    )


# ---------------------------------------------------------------------------
# Step 3: Tail Risk
# ---------------------------------------------------------------------------

def _evaluate_tail_risk(
    consequences: list[Consequence],
) -> tuple[str | None, Severity]:
    """
    Evaluate tail risk — the worst-case scenario.

    What happens if everything goes wrong?
    """
    if not consequences:
        return None, Severity.LOW

    # Find the worst consequence
    severity_rank = {
        Severity.LOW: 0,
        Severity.MODERATE: 1,
        Severity.HIGH: 2,
        Severity.CRITICAL: 3,
    }

    worst = max(consequences, key=lambda c: severity_rank[c.severity])

    if worst.severity in (Severity.HIGH, Severity.CRITICAL):
        tail_desc = (
            f"TAIL RISK: {worst.description} "
            f"(severity: {worst.severity.value}, "
            f"probability: {worst.probability:.0%}, "
            f"timeframe: {worst.timeframe}, "
            f"reversible: {'yes' if worst.is_reversible else 'NO'}). "
        )
        if not worst.is_reversible:
            tail_desc += "This consequence is IRREVERSIBLE. Proceed with extreme caution."
        elif worst.reversal_window:
            tail_desc += f"Reversal window: {worst.reversal_window}."

        return tail_desc, worst.severity

    return None, Severity.LOW


# ---------------------------------------------------------------------------
# Step 4: Real-World Durability
# ---------------------------------------------------------------------------

def _check_executability(
    problem: Problem,
    root_cause: RootCause,
) -> tuple[bool, list[str]]:
    """
    Check if the solution is executable in the user's real context.

    A perfect solution the user can't execute is worthless.

    Checks:
    - Does the user have enough positive forces (resources/energy)?
    - Are there too many negative forces (blockers)?
    - Is the root cause within the user's control?
    """
    blockers = []

    # Check resource availability
    positive_forces = [
        v for v in problem.variables if v.direction.value == "positive"
    ]
    total_positive_energy = sum(v.magnitude for v in positive_forces) if positive_forces else 0.0

    if total_positive_energy < 0.3:
        blockers.append(
            "Low positive energy: the user may not have the resources, "
            "bandwidth, or support system to execute this."
        )

    # Check blocker count
    negative_forces = [
        v for v in problem.variables if v.direction.value == "negative"
    ]
    if len(negative_forces) > len(positive_forces) * 2:
        blockers.append(
            f"Blockers ({len(negative_forces)}) outnumber resources "
            f"({len(positive_forces)}) by >2:1. Execution will be uphill."
        )

    # Check if root cause is within user's control
    if root_cause.variable.is_hidden and root_cause.confidence < 0.6:
        blockers.append(
            "Root cause is hidden and uncertain — the user may not be able "
            "to act on it directly without more information."
        )

    # Check for external dependency
    if root_cause.bias_that_hid_it and "frame" in root_cause.bias_that_hid_it.lower():
        blockers.append(
            "Root cause involves the user's own perspective — "
            "execution requires self-awareness, which is the hardest change to make."
        )

    executable = len(blockers) == 0
    return executable, blockers
