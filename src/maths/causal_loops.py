"""
Maths Layer 8: Causal Loop Analysis (Circular Feedback Detection)

Physics trajectory projects a line. Most human problems are circular —
feedback loops that trap people.

Anxiety → avoidance → more anxiety.
Debt → stress → bad decisions → more debt.

Without this, the engine misses the spirals that keep people stuck.

Activates when trajectory detection finds circular rather than linear patterns.

4 Operations:
1. Reinforcing Loops — spirals (vicious or virtuous)
2. Balancing Loops — self-correcting equilibria
3. Loop Dominance — which cycle is winning?
4. Delay Effects — consequences that arrive late
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    CausalLoop,
    Direction,
    DomainOutput,
    FrameworkID,
    Problem,
    Variable,
)


@dataclass
class LoopAnalysis:
    """Complete causal loop analysis of the problem."""
    reinforcing_loops: list[CausalLoop]
    balancing_loops: list[CausalLoop]
    dominant_loop: CausalLoop | None
    delays: list[str]
    loop_variables: list[Variable]


def analyze_loops(
    problem: Problem,
    domain_outputs: list[DomainOutput],
) -> LoopAnalysis:
    """
    Run causal loop analysis across problem and domain outputs.

    Detects feedback cycles, identifies the dominant loop,
    and maps delay effects.
    """
    # Collect all variables from all sources
    all_variables = list(problem.variables)
    for output in domain_outputs:
        for p in output.perspectives:
            all_variables.extend(p.variables_found)

    # Detect reinforcing loops (spirals)
    reinforcing = _detect_reinforcing_loops(all_variables, problem)

    # Detect balancing loops (equilibria)
    balancing = _detect_balancing_loops(all_variables, problem)

    # Determine loop dominance
    all_loops = reinforcing + balancing
    dominant = _find_dominant_loop(all_loops)

    # Detect delay effects
    delays = _detect_delays(all_variables, all_loops)

    # Extract loop variables for the manifold
    loop_vars = _extract_loop_variables(all_loops, dominant, delays)

    return LoopAnalysis(
        reinforcing_loops=reinforcing,
        balancing_loops=balancing,
        dominant_loop=dominant,
        delays=delays,
        loop_variables=loop_vars,
    )


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def _detect_reinforcing_loops(
    variables: list[Variable], problem: Problem
) -> list[CausalLoop]:
    """
    Detect reinforcing loops — A causes more B, which causes more A.

    These loops amplify. They can spiral up (virtuous) or down (vicious).

    Detection: Find variable pairs where both push in the same direction
    and one's evidence references the other.
    """
    loops = []

    # Group variables by direction
    negative_vars = [v for v in variables if v.direction == Direction.NEGATIVE]
    positive_vars = [v for v in variables if v.direction == Direction.POSITIVE]

    # Look for vicious spirals: negative → more negative
    if len(negative_vars) >= 2:
        # Check for chain patterns: variables that reference each other
        for i, v1 in enumerate(negative_vars):
            for v2 in negative_vars[i + 1:]:
                connection = _find_connection(v1, v2)
                if connection:
                    loop = CausalLoop(
                        name=f"vicious_spiral_{v1.name}_{v2.name}",
                        description=(
                            f"Vicious reinforcing loop: '{v1.name}' and '{v2.name}' "
                            f"amplify each other. {connection}. "
                            "Each makes the other worse — the spiral accelerates."
                        ),
                        loop_type="reinforcing",
                        variables_in_loop=[v1.name, v2.name],
                        is_dominant=False,
                    )
                    loops.append(loop)

    # Look for virtuous spirals: positive → more positive
    if len(positive_vars) >= 2:
        for i, v1 in enumerate(positive_vars):
            for v2 in positive_vars[i + 1:]:
                connection = _find_connection(v1, v2)
                if connection:
                    loop = CausalLoop(
                        name=f"virtuous_spiral_{v1.name}_{v2.name}",
                        description=(
                            f"Virtuous reinforcing loop: '{v1.name}' and '{v2.name}' "
                            f"strengthen each other. {connection}. "
                            "Each makes the other better — the spiral accelerates upward."
                        ),
                        loop_type="reinforcing",
                        variables_in_loop=[v1.name, v2.name],
                        is_dominant=False,
                    )
                    loops.append(loop)

    # Look for mixed reinforcing: negative outcome triggers negative action triggers worse outcome
    for nv in negative_vars:
        for pv in positive_vars:
            # If a positive action is being cancelled by a negative outcome
            # and the negative outcome may be caused by the positive action failing
            if (nv.magnitude > 0.4 and pv.magnitude > 0.4
                    and abs(nv.magnitude - pv.magnitude) < 0.3):
                loop = CausalLoop(
                    name=f"effort_drain_loop_{pv.name}_{nv.name}",
                    description=(
                        f"Effort-drain loop: '{pv.name}' (positive, {pv.magnitude:.2f}) "
                        f"is being drained by '{nv.name}' (negative, {nv.magnitude:.2f}). "
                        "The effort creates the condition for the drain, "
                        "which demands more effort, which creates more drain."
                    ),
                    loop_type="reinforcing",
                    variables_in_loop=[pv.name, nv.name],
                    is_dominant=False,
                )
                loops.append(loop)

    return loops


def _detect_balancing_loops(
    variables: list[Variable], problem: Problem
) -> list[CausalLoop]:
    """
    Detect balancing loops — A causes B, which pushes back against A.

    These loops stabilize. They resist change in either direction.
    Different from Equilibrium (physics) — this is a self-correcting
    CYCLE, not just two opposing forces.
    """
    loops = []

    # Look for variables where pushing harder creates pushback
    for i, v1 in enumerate(variables):
        for v2 in variables[i + 1:]:
            if (v1.direction != v2.direction
                    and v1.direction != Direction.NEUTRAL
                    and v2.direction != Direction.NEUTRAL
                    and abs(v1.magnitude - v2.magnitude) < 0.2):
                loop = CausalLoop(
                    name=f"balancing_{v1.name}_{v2.name}",
                    description=(
                        f"Balancing loop: '{v1.name}' ({v1.direction.value}) "
                        f"triggers '{v2.name}' ({v2.direction.value}) which "
                        "pushes back to the original state. "
                        "The system resists change — any push is met with a counter-push."
                    ),
                    loop_type="balancing",
                    variables_in_loop=[v1.name, v2.name],
                    is_dominant=False,
                )
                loops.append(loop)

    return loops


def _find_dominant_loop(loops: list[CausalLoop]) -> CausalLoop | None:
    """
    Find the dominant loop — the one currently controlling the system.

    Heuristic: the loop with the most variables and strongest
    connections dominates.
    """
    if not loops:
        return None

    # Score loops by variable count and type (reinforcing > balancing for dominance)
    def loop_score(loop: CausalLoop) -> float:
        base = len(loop.variables_in_loop) * 0.5
        type_bonus = 0.3 if loop.loop_type == "reinforcing" else 0.1
        return base + type_bonus

    dominant = max(loops, key=loop_score)
    dominant.is_dominant = True
    return dominant


def _detect_delays(
    variables: list[Variable], loops: list[CausalLoop]
) -> list[str]:
    """
    Detect delay effects — consequences that arrive late.

    Delays are where humans lose the plot. They act, don't see results,
    change course — just as the original action was about to work.
    """
    delays = []

    # Look for high-magnitude variables with low confidence
    # (user is uncertain → likely hasn't seen the consequence yet)
    for var in variables:
        if var.magnitude > 0.5 and var.confidence < 0.5:
            delays.append(
                f"Potential delayed consequence in '{var.name}': "
                f"high magnitude ({var.magnitude:.2f}) but low confidence "
                f"({var.confidence:.2f}). The user may not have seen "
                "the full impact yet — it's still arriving."
            )

    # Look for reinforcing loops that haven't fully spiraled
    for loop in loops:
        if loop.loop_type == "reinforcing":
            delays.append(
                f"Reinforcing loop '{loop.name}' may have delayed amplification. "
                "The spiral's full force hasn't hit yet — each cycle "
                "increases the impact. The next turn will be bigger than the last."
            )

    return delays


# ---------------------------------------------------------------------------
# Variable extraction for manifold
# ---------------------------------------------------------------------------

def _extract_loop_variables(
    loops: list[CausalLoop],
    dominant: CausalLoop | None,
    delays: list[str],
) -> list[Variable]:
    """Extract variables from loop analysis for the manifold."""
    variables = []

    if dominant:
        variables.append(Variable(
            name=f"dominant_loop_{dominant.name}",
            description=(
                f"DOMINANT LOOP: {dominant.description} "
                "This loop currently controls the system's behavior. "
                "Intervening here has the highest leverage."
            ),
            magnitude=0.85,
            direction=Direction.CIRCULAR,
            confidence=0.7,
            source_framework=FrameworkID.CAUSAL_LOOPS,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Loop type: {dominant.loop_type}",
                f"Variables in loop: {', '.join(dominant.variables_in_loop)}",
                "This is the dominant feedback cycle.",
            ],
        ))

    for loop in loops:
        if loop != dominant:
            variables.append(Variable(
                name=f"loop_{loop.name}",
                description=loop.description,
                magnitude=0.5,
                direction=Direction.CIRCULAR,
                confidence=0.6,
                source_framework=FrameworkID.CAUSAL_LOOPS,
                is_hidden=False,
                is_user_stated=False,
                evidence=[
                    f"Loop type: {loop.loop_type}",
                    f"Variables: {', '.join(loop.variables_in_loop)}",
                ],
            ))

    if delays:
        variables.append(Variable(
            name="delay_effects",
            description=(
                f"{len(delays)} delay effects detected. "
                "Consequences are still arriving. "
                "The user may change course before seeing results."
            ),
            magnitude=0.6,
            direction=Direction.NEGATIVE,
            confidence=0.5,
            source_framework=FrameworkID.CAUSAL_LOOPS,
            is_hidden=True,
            is_user_stated=False,
            evidence=delays[:5],
        ))

    return variables


# ---------------------------------------------------------------------------
# Connection detection
# ---------------------------------------------------------------------------

def _find_connection(v1: Variable, v2: Variable) -> str | None:
    """
    Find if two variables are connected (one references the other).

    Checks names, descriptions, and evidence for cross-references.
    """
    # Check if v1 references v2
    v2_name_lower = v2.name.lower().replace("_", " ")
    v1_desc_lower = v1.description.lower()

    if v2_name_lower in v1_desc_lower or v2.name in v1.description:
        return f"'{v1.name}' references '{v2.name}' in its description"

    # Check if v2 references v1
    v1_name_lower = v1.name.lower().replace("_", " ")
    v2_desc_lower = v2.description.lower()

    if v1_name_lower in v2_desc_lower or v1.name in v2.description:
        return f"'{v2.name}' references '{v1.name}' in its description"

    # Check for shared evidence keywords
    v1_evidence = " ".join(v1.evidence).lower()
    v2_evidence = " ".join(v2.evidence).lower()

    if v2.name.lower() in v1_evidence or v1.name.lower() in v2_evidence:
        return f"Evidence chains reference each other"

    # Check if same source framework found both (suggesting causal connection)
    if v1.source_framework == v2.source_framework and v1.magnitude > 0.3 and v2.magnitude > 0.3:
        return (
            f"Same framework ({v1.source_framework.value}) identified both "
            f"at significant magnitude ({v1.magnitude:.2f}, {v2.magnitude:.2f})"
        )

    return None
