"""
Physics Phase 2: Bias Penetration

Operates on what the user DOESN'T tell you. Reads the gaps, contradictions,
and distortions in the user's story to surface what they can't or won't reveal.

This is the layer that no general-purpose chatbot performs — it challenges
the user's frame before reasoning about solutions.

5 Frameworks:
1. Anomalous Motion — detects wobble in the user's story (hidden variable pull)
2. Socratic Squeeze — strips assumptions to bedrock (first principles on beliefs)
3. Reference Frame Shift — rotates the user's perspective (removes blind spot)
4. Entropy Leak — finds what they're omitting (the mess points to truth)
5. Reductio ad Absurdum — breaks false claims to impossibility (forces hidden variable to surface)
"""

from __future__ import annotations

from src.core.types import (
    Consequence,
    Direction,
    Domain,
    DomainOutput,
    FrameworkID,
    Perspective,
    Problem,
    RootCause,
    Severity,
    SignalType,
    Variable,
)


def run_phase2(problem: Problem, phase1_output: DomainOutput) -> DomainOutput:
    """
    Execute all Phase 2 frameworks on the problem.

    Phase 2 uses both the original problem AND Phase 1's findings
    to detect where the user's story doesn't match the physics.
    The gap between the user's narrative and the system's reality IS variable y.
    """
    anomalous_result = _anomalous_motion(problem, phase1_output)
    socratic_result = _socratic_squeeze(problem, phase1_output)
    reference_result = _reference_frame_shift(problem, phase1_output)
    entropy_leak_result = _entropy_leak(problem, phase1_output)
    reductio_result = _reductio(problem, phase1_output)

    perspectives = [
        anomalous_result,
        socratic_result,
        reference_result,
        entropy_leak_result,
        reductio_result,
    ]

    # Extract all hidden variables found by bias penetration
    all_variables = []
    for p in perspectives:
        all_variables.extend(p.variables_found)

    root_causes = _extract_bias_root_causes(all_variables, perspectives)

    return DomainOutput(
        domain=Domain.PHYSICS,
        perspectives=perspectives,
        root_causes=root_causes,
        consequences=[],  # Phase 2 finds roots, Phase 1 projects consequences
        raw_analysis=_build_raw_analysis(perspectives),
    )


# ---------------------------------------------------------------------------
# Framework implementations
# ---------------------------------------------------------------------------

def _anomalous_motion(problem: Problem, phase1: DomainOutput) -> Perspective:
    """
    The "Anomalous Motion" Test — wobble detection.

    If a planet moves wrong, we don't assume the laws are wrong —
    we assume there's an unseen mass pulling on it.

    If the user's logic has a gap or contradiction, variable D
    is located exactly in that gap. Their bias is the "dark matter."
    """
    variables_found = []

    # Look for contradictions between user-stated variables and Phase 1 findings
    user_stated = [v for v in problem.variables if v.is_user_stated]
    phase1_hidden = []
    for p in phase1.perspectives:
        phase1_hidden.extend([v for v in p.variables_found if v.is_hidden])

    # Detect wobble: user says situation is X, but physics says it's Y
    for user_var in user_stated:
        for hidden_var in phase1_hidden:
            # If user claims positive but physics found a hidden negative drain
            if (user_var.direction == Direction.POSITIVE
                    and hidden_var.direction == Direction.NEGATIVE):
                wobble = Variable(
                    name=f"wobble_{user_var.name}",
                    description=(
                        f"Anomalous motion detected: user states '{user_var.name}' "
                        f"as positive (magnitude: {user_var.magnitude:.2f}), "
                        f"but physics found hidden negative force '{hidden_var.name}' "
                        f"(magnitude: {hidden_var.magnitude:.2f}). "
                        "The user's story wobbles — there's an unseen mass pulling on it."
                    ),
                    magnitude=hidden_var.magnitude,
                    direction=Direction.NEGATIVE,
                    confidence=min(hidden_var.confidence + 0.1, 1.0),
                    source_framework=FrameworkID.ANOMALOUS_MOTION,
                    is_hidden=True,
                    is_user_stated=False,
                    evidence=[
                        f"User claims: {user_var.name} is positive ({user_var.magnitude:.2f})",
                        f"Physics found: {hidden_var.name} is negative ({hidden_var.magnitude:.2f})",
                        "These contradict — the gap contains the hidden variable.",
                        "Bias type: the user's story doesn't match the system's physics.",
                    ],
                )
                variables_found.append(wobble)

    # Detect wobble: claimed actions should produce results but aren't
    for user_var in user_stated:
        if user_var.direction == Direction.POSITIVE and user_var.magnitude > 0.5:
            # User claims strong positive action — check if Phase 1 trajectory is negative
            trajectory_perspectives = [
                p for p in phase1.perspectives
                if p.framework == FrameworkID.TRAJECTORY_MOMENTUM
            ]
            for tp in trajectory_perspectives:
                for tv in tp.variables_found:
                    if tv.direction == Direction.NEGATIVE:
                        action_gap = Variable(
                            name=f"action_gap_{user_var.name}",
                            description=(
                                f"User claims strong positive action '{user_var.name}' "
                                f"(magnitude: {user_var.magnitude:.2f}), but trajectory "
                                f"is negative ({tv.description}). "
                                "The action should be producing results — it isn't. "
                                "Dark matter: something the user can't see is absorbing the effort."
                            ),
                            magnitude=user_var.magnitude * 0.8,
                            direction=Direction.NEGATIVE,
                            confidence=0.7,
                            source_framework=FrameworkID.ANOMALOUS_MOTION,
                            is_hidden=True,
                            is_user_stated=False,
                            evidence=[
                                f"Claimed action: {user_var.name} ({user_var.magnitude:.2f} positive)",
                                f"Actual trajectory: {tv.direction.value}",
                                "Gap between expected and actual results = hidden variable location.",
                            ],
                        )
                        variables_found.append(action_gap)

    return Perspective(
        framework=FrameworkID.ANOMALOUS_MOTION,
        domain=Domain.PHYSICS,
        content=_build_anomalous_analysis(variables_found),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )


def _socratic_squeeze(problem: Problem, phase1: DomainOutput) -> Perspective:
    """
    First Principles: The "Socratic Squeeze" — assumption stripping.

    Bias relies on analogy and assumptions.
    Strips away "it's just like last time" and "it's obviously because of X"
    to find what fundamental truth remains.
    """
    variables_found = []

    # Identify variables with high magnitude but low confidence — likely assumptions
    for var in problem.variables:
        if var.magnitude > 0.4 and var.confidence < 0.6:
            assumption = Variable(
                name=f"assumption_{var.name}",
                description=(
                    f"Potential assumption detected in '{var.name}': "
                    f"user treats this as significant (magnitude: {var.magnitude:.2f}) "
                    f"but confidence is low ({var.confidence:.2f}). "
                    "This may be an analogy or inherited belief rather than a verified fact. "
                    "Socratic question: What is the fundamental truth underneath this claim?"
                ),
                magnitude=var.magnitude,
                direction=var.direction,
                confidence=var.confidence * 0.5,  # halve confidence — it's an assumption
                source_framework=FrameworkID.SOCRATIC_SQUEEZE,
                is_hidden=True,
                is_user_stated=True,
                evidence=[
                    f"User stated: {var.name} = {var.description}",
                    f"Magnitude: {var.magnitude:.2f} (user treats as important)",
                    f"Confidence: {var.confidence:.2f} (but basis is weak)",
                    "Socratic test: Can this be broken down further? What raw facts support it?",
                ],
            )
            variables_found.append(assumption)

    # Check for variables that contradict each other — user holding two incompatible beliefs
    user_vars = problem.variables
    for i, v1 in enumerate(user_vars):
        for v2 in user_vars[i + 1:]:
            if (v1.direction == Direction.POSITIVE
                    and v2.direction == Direction.NEGATIVE
                    and abs(v1.magnitude - v2.magnitude) < 0.2):
                contradiction = Variable(
                    name=f"contradiction_{v1.name}_vs_{v2.name}",
                    description=(
                        f"Internal contradiction: user holds '{v1.name}' (positive) "
                        f"and '{v2.name}' (negative) at similar magnitudes. "
                        "These beliefs work against each other. "
                        "At least one is an assumption that hasn't been tested."
                    ),
                    magnitude=(v1.magnitude + v2.magnitude) / 2,
                    direction=Direction.NEUTRAL,
                    confidence=0.7,
                    source_framework=FrameworkID.SOCRATIC_SQUEEZE,
                    is_hidden=True,
                    is_user_stated=True,
                    evidence=[
                        f"Belief 1: {v1.name} ({v1.direction.value}, {v1.magnitude:.2f})",
                        f"Belief 2: {v2.name} ({v2.direction.value}, {v2.magnitude:.2f})",
                        "These are incompatible at their current magnitudes.",
                        "One must be an untested assumption.",
                    ],
                )
                variables_found.append(contradiction)

    return Perspective(
        framework=FrameworkID.SOCRATIC_SQUEEZE,
        domain=Domain.PHYSICS,
        content=_build_socratic_analysis(variables_found),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )


def _reference_frame_shift(problem: Problem, phase1: DomainOutput) -> Perspective:
    """
    Relativity: Changing the Reference Frame — perspective rotation.

    If the user is at the center of the problem, their bias is a
    "stationary" frame. Everything looks like it's moving at them.

    Force a shift: what do the forces look like from the other side?
    """
    variables_found = []

    # The user's frame: they see themselves as the subject being acted upon
    user_positive = [v for v in problem.variables if v.direction == Direction.POSITIVE]
    user_negative = [v for v in problem.variables if v.direction == Direction.NEGATIVE]

    # Frame shift: what if the user's positive forces are actually neutral or negative
    # from another reference frame?
    if len(user_negative) > len(user_positive) * 1.5:
        # User sees mostly negative — classic "everything is happening TO me" frame
        victim_frame = Variable(
            name="victim_reference_frame",
            description=(
                f"Reference frame bias detected: user identifies "
                f"{len(user_negative)} negative forces vs {len(user_positive)} positive. "
                "From this stationary frame, everything appears to be moving AT the user. "
                "Frame shift question: From the other party's perspective, "
                "which of these 'attacks' are actually responses to the user's own actions?"
            ),
            magnitude=0.7,
            direction=Direction.NEUTRAL,
            confidence=0.6,
            source_framework=FrameworkID.REFERENCE_FRAME_SHIFT,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"User-identified negative forces: {len(user_negative)}",
                f"User-identified positive forces: {len(user_positive)}",
                "Ratio suggests stationary-frame bias.",
                "The user may be the source of forces they perceive as external.",
            ],
        )
        variables_found.append(victim_frame)

    # Check for user-stated variables where the user is always the recipient, never the actor
    user_as_recipient = [
        v for v in problem.variables
        if v.is_user_stated and v.direction == Direction.NEGATIVE
    ]
    if len(user_as_recipient) >= 3:
        blind_spot = Variable(
            name="self_contribution_blind_spot",
            description=(
                f"The user identifies {len(user_as_recipient)} forces acting ON them "
                "but may not see their own contribution to the dynamic. "
                "Reference frame shift: the user's own behavior — visible from "
                "other frames — is likely a significant force they cannot see "
                "from their current position."
            ),
            magnitude=0.6,
            direction=Direction.NEGATIVE,
            confidence=0.65,
            source_framework=FrameworkID.REFERENCE_FRAME_SHIFT,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Forces user sees acting on them: {len(user_as_recipient)}",
                "User's own actions are invisible from their frame.",
                "Frame shift reveals: their behavior is part of the force field.",
            ],
        )
        variables_found.append(blind_spot)

    return Perspective(
        framework=FrameworkID.REFERENCE_FRAME_SHIFT,
        domain=Domain.PHYSICS,
        content=_build_reference_frame_analysis(
            user_positive, user_negative, variables_found
        ),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )


def _entropy_leak(problem: Problem, phase1: DomainOutput) -> Perspective:
    """
    Entropy and "The Leak" — omission detection.

    The user tells a story of order, but the system is falling apart.
    Chaos requires a cause. The mess points to what they're hiding.
    """
    variables_found = []

    # Compare user's stated picture with Phase 1's entropy finding
    entropy_perspectives = [
        p for p in phase1.perspectives
        if p.framework == FrameworkID.ENTROPY
    ]

    # Check if user paints a positive picture but entropy says decay
    user_positive_count = sum(
        1 for v in problem.variables if v.direction == Direction.POSITIVE
    )
    user_negative_count = sum(
        1 for v in problem.variables if v.direction == Direction.NEGATIVE
    )

    for ep in entropy_perspectives:
        decay_vars = [v for v in ep.variables_found if "decay" in v.name]
        for decay in decay_vars:
            if decay.magnitude > 0.3 and user_positive_count >= user_negative_count:
                # User says things are fine, but entropy says decay
                leak = Variable(
                    name="entropy_leak_omission",
                    description=(
                        f"The user presents a balanced or positive picture "
                        f"({user_positive_count} positive vs {user_negative_count} negative), "
                        f"but entropy analysis shows significant decay "
                        f"(rate: {decay.magnitude:.2f}). "
                        "The user is omitting the source of chaos. "
                        "Look for: conflict they're minimizing, effort they're not actually putting in, "
                        "or a deteriorating element they're embarrassed to admit."
                    ),
                    magnitude=decay.magnitude,
                    direction=Direction.NEGATIVE,
                    confidence=0.7,
                    source_framework=FrameworkID.ENTROPY_LEAK,
                    is_hidden=True,
                    is_user_stated=False,
                    evidence=[
                        f"User's positive variables: {user_positive_count}",
                        f"User's negative variables: {user_negative_count}",
                        f"Entropy decay rate: {decay.magnitude:.2f}",
                        "Mismatch: user says stable, physics says decaying.",
                        "The omission IS the leak.",
                    ],
                )
                variables_found.append(leak)

    return Perspective(
        framework=FrameworkID.ENTROPY_LEAK,
        domain=Domain.PHYSICS,
        content=_build_entropy_leak_analysis(variables_found),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.8,
    )


def _reductio(problem: Problem, phase1: DomainOutput) -> Perspective:
    """
    Reductio ad Absurdum — claim breaking.

    Take the user's claim to its logical extreme.
    If it leads to impossibility, their inputs are false.
    The correction IS the hidden variable.
    """
    variables_found = []

    # Find user claims with very high magnitude + high confidence
    # These are the "strong claims" worth stress-testing
    strong_claims = [
        v for v in problem.variables
        if v.is_user_stated and v.magnitude > 0.6 and v.confidence > 0.6
    ]

    for claim in strong_claims:
        # Take to extreme: if this claim is fully true (magnitude → 1.0),
        # does it conflict with other evidence?
        conflicting = []
        for p in phase1.perspectives:
            for v in p.variables_found:
                # If the claim is positive but physics found strong negative forces
                if (claim.direction == Direction.POSITIVE
                        and v.direction == Direction.NEGATIVE
                        and v.magnitude > 0.4):
                    conflicting.append(v)
                # If the claim is negative but physics found positive forces
                elif (claim.direction == Direction.NEGATIVE
                      and v.direction == Direction.POSITIVE
                      and v.magnitude > 0.4):
                    conflicting.append(v)

        if conflicting:
            absurdity = Variable(
                name=f"reductio_{claim.name}",
                description=(
                    f"Reductio test on '{claim.name}': if this claim is taken "
                    f"to its logical extreme (magnitude → 1.0, direction: {claim.direction.value}), "
                    f"it conflicts with {len(conflicting)} findings from physics analysis. "
                    f"Contradicting forces: {', '.join(v.name for v in conflicting[:3])}. "
                    "The claim cannot be fully true. The correction needed to "
                    "restore logical consistency IS variable D."
                ),
                magnitude=claim.magnitude * 0.7,
                direction=Direction.NEUTRAL,
                confidence=min(0.5 + len(conflicting) * 0.1, 0.9),
                source_framework=FrameworkID.REDUCTIO,
                is_hidden=True,
                is_user_stated=False,
                evidence=[
                    f"User claim: {claim.name} ({claim.direction.value}, {claim.magnitude:.2f})",
                    f"Taken to extreme: conflicts with {len(conflicting)} physics findings",
                    *[f"Conflict: {v.name} ({v.direction.value}, {v.magnitude:.2f})" for v in conflicting[:3]],
                    "The claim is partially or fully false.",
                    "The truth that replaces it is the hidden variable.",
                ],
            )
            variables_found.append(absurdity)

    return Perspective(
        framework=FrameworkID.REDUCTIO,
        domain=Domain.PHYSICS,
        content=_build_reductio_analysis(strong_claims, variables_found),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------

def _extract_bias_root_causes(
    all_variables: list[Variable], perspectives: list[Perspective]
) -> list[RootCause]:
    """Extract root causes from bias penetration — these include the bias that hid them."""
    candidates = []
    for var in all_variables:
        if var.is_hidden and var.confidence >= 0.5:
            agreeing = [
                p.framework for p in perspectives
                if any(v.name == var.name for v in p.variables_found)
            ]

            # Determine which bias hid this variable
            bias_map = {
                FrameworkID.ANOMALOUS_MOTION: "Dark matter bias — the user's story has a wobble they can't see",
                FrameworkID.SOCRATIC_SQUEEZE: "Assumption bias — treating an untested belief as fact",
                FrameworkID.REFERENCE_FRAME_SHIFT: "Stationary frame bias — can't see own contribution from own position",
                FrameworkID.ENTROPY_LEAK: "Omission bias — minimizing or hiding the source of chaos",
                FrameworkID.REDUCTIO: "Overclaim bias — holding a belief that breaks under logical pressure",
            }

            candidate = RootCause(
                variable=var,
                evidence_chain=var.evidence,
                bias_that_hid_it=bias_map.get(var.source_framework, "Unknown bias"),
                confidence=var.confidence,
                frameworks_that_agree=agreeing,
            )
            candidates.append(candidate)

    return candidates


# ---------------------------------------------------------------------------
# Analysis text builders
# ---------------------------------------------------------------------------

def _build_anomalous_analysis(variables_found: list[Variable]) -> str:
    lines = [
        "ANOMALOUS MOTION TEST (Wobble Detection)",
        "",
    ]
    if variables_found:
        lines.append("Wobbles detected in user's story:")
        for v in variables_found:
            lines.append(f"  - {v.name}: {v.description}")
    else:
        lines.append("No anomalous motion detected — user's story is consistent with physics.")
    return "\n".join(lines)


def _build_socratic_analysis(variables_found: list[Variable]) -> str:
    lines = [
        "SOCRATIC SQUEEZE (Assumption Stripping)",
        "",
    ]
    if variables_found:
        assumptions = [v for v in variables_found if "assumption" in v.name]
        contradictions = [v for v in variables_found if "contradiction" in v.name]
        if assumptions:
            lines.append("Assumptions detected (need first-principles verification):")
            for v in assumptions:
                lines.append(f"  - {v.name}: {v.description}")
        if contradictions:
            lines.append("Internal contradictions detected:")
            for v in contradictions:
                lines.append(f"  - {v.name}: {v.description}")
    else:
        lines.append("No unverified assumptions or contradictions detected.")
    return "\n".join(lines)


def _build_reference_frame_analysis(
    user_positive: list[Variable], user_negative: list[Variable],
    variables_found: list[Variable],
) -> str:
    lines = [
        "REFERENCE FRAME SHIFT (Perspective Rotation)",
        f"User's frame: {len(user_positive)} positive, {len(user_negative)} negative forces",
        "",
    ]
    if variables_found:
        lines.append("Frame shift findings:")
        for v in variables_found:
            lines.append(f"  - {v.name}: {v.description}")
    else:
        lines.append("User's reference frame appears balanced — no significant frame bias detected.")
    return "\n".join(lines)


def _build_entropy_leak_analysis(variables_found: list[Variable]) -> str:
    lines = [
        "ENTROPY LEAK (Omission Detection)",
        "",
    ]
    if variables_found:
        lines.append("Leaks detected — user is omitting sources of chaos:")
        for v in variables_found:
            lines.append(f"  - {v.name}: {v.description}")
    else:
        lines.append("No entropy leak detected — user's story matches entropy analysis.")
    return "\n".join(lines)


def _build_reductio_analysis(
    strong_claims: list[Variable], variables_found: list[Variable]
) -> str:
    lines = [
        "REDUCTIO AD ABSURDUM (Claim Breaking)",
        f"Strong claims tested: {len(strong_claims)}",
        "",
    ]
    if variables_found:
        lines.append("Claims that break under logical pressure:")
        for v in variables_found:
            lines.append(f"  - {v.name}: {v.description}")
    else:
        if strong_claims:
            lines.append("All strong claims survived the reductio test.")
        else:
            lines.append("No strong claims to test.")
    return "\n".join(lines)


def _build_raw_analysis(perspectives: list[Perspective]) -> str:
    """Combine all perspective analyses into one raw output."""
    sections = []
    for p in perspectives:
        sections.append(p.content)
    return "\n\n---\n\n".join(sections)
