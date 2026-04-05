"""
Physics Phase 1: Root Finding & Trajectory

Operates on what the user DOES tell you. Reads the system mechanics
to find where energy leaks, where pressure builds, where things are heading.

6 Frameworks:
1. First Principles — decomposes into irreducible forces (runs FIRST, upstream)
2. Conservation of Energy — audits input vs output, finds hidden drains/sources
3. Entropy — measures decay rate, timeline to breakdown
4. Trajectory & Momentum — plots where the problem lands and when
5. Potential → Kinetic — detects stored pressure about to release
6. Equilibrium & Net Force — finds hidden counter-forces causing stuckness
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


def run_phase1(problem: Problem) -> DomainOutput:
    """
    Execute all Phase 1 frameworks on the problem.

    First Principles runs first (decomposition step),
    then the remaining 5 operate on the decomposed forces.
    """
    # Step 1: First Principles decomposes the problem into irreducible forces
    first_principles_result = _first_principles(problem)

    # Step 2: Remaining 5 frameworks operate on the decomposed problem
    conservation_result = _conservation_of_energy(problem, first_principles_result)
    entropy_result = _entropy(problem, first_principles_result)
    trajectory_result = _trajectory_momentum(problem, first_principles_result)
    potential_result = _potential_kinetic(problem, first_principles_result)
    equilibrium_result = _equilibrium(problem, first_principles_result)

    # Collect all perspectives
    perspectives = [
        first_principles_result,
        conservation_result,
        entropy_result,
        trajectory_result,
        potential_result,
        equilibrium_result,
    ]

    # Extract all variables found across frameworks
    all_variables = []
    for p in perspectives:
        all_variables.extend(p.variables_found)

    # Build candidate root causes from high-confidence hidden variables
    root_causes = _extract_root_causes(all_variables, perspectives)

    # Build consequences from trajectory and entropy
    consequences = _extract_consequences(trajectory_result, entropy_result)

    return DomainOutput(
        domain=Domain.PHYSICS,
        perspectives=perspectives,
        root_causes=root_causes,
        consequences=consequences,
        raw_analysis=_build_raw_analysis(perspectives),
    )


# ---------------------------------------------------------------------------
# Framework implementations
# ---------------------------------------------------------------------------

def _first_principles(problem: Problem) -> Perspective:
    """
    First Principles (Classical Mechanics) — the decomposition step.

    Breaks the problem into its most basic, non-reducible parts.
    Doesn't solve the symptom — solves the force.
    Strips away everything that's noise to find the gravity of the situation.

    This runs FIRST. Its output feeds every other Phase 1 framework.
    """
    # Identify the irreducible forces from the user's stated variables
    forces = []
    for var in problem.variables:
        force = Variable(
            name=f"force_{var.name}",
            description=f"Irreducible force behind '{var.name}': {var.description}",
            magnitude=var.magnitude,
            direction=var.direction,
            confidence=var.confidence,
            source_framework=FrameworkID.FIRST_PRINCIPLES,
            is_hidden=False,
            is_user_stated=var.is_user_stated,
            evidence=[f"Decomposed from user variable: {var.name}"],
        )
        forces.append(force)

    return Perspective(
        framework=FrameworkID.FIRST_PRINCIPLES,
        domain=Domain.PHYSICS,
        content=_build_first_principles_analysis(problem, forces),
        variables_found=forces,
        signal_type=SignalType.SIGNAL,
        weight=1.0,  # upstream framework — always full weight
    )


def _conservation_of_energy(
    problem: Problem, decomposition: Perspective
) -> Perspective:
    """
    Conservation of Energy — the input/output audit.

    If output > input: there's a hidden energy source (variable D).
    If input >> output: there's a heat leak — energy is being wasted or stolen.

    Finds hidden drains and hidden fuel.
    """
    variables_found = []
    forces = decomposition.variables_found

    # Separate positive (input) forces from negative (output/drain) forces
    inputs = [f for f in forces if f.direction == Direction.POSITIVE]
    drains = [f for f in forces if f.direction == Direction.NEGATIVE]
    neutrals = [f for f in forces if f.direction == Direction.NEUTRAL]

    total_input = sum(f.magnitude for f in inputs) if inputs else 0.0
    total_drain = sum(f.magnitude for f in drains) if drains else 0.0

    # Conservation violation: the numbers don't add up
    imbalance = total_input - total_drain

    if abs(imbalance) > 0.2:  # threshold: significant imbalance
        if imbalance > 0:
            # Lots of input, little drain — where is the energy going?
            # There's a hidden drain the user hasn't mentioned
            hidden_drain = Variable(
                name="hidden_energy_drain",
                description=(
                    f"Conservation violation: input energy ({total_input:.2f}) "
                    f"significantly exceeds visible output ({total_drain:.2f}). "
                    f"There is a hidden drain absorbing {imbalance:.2f} units of effort."
                ),
                magnitude=min(abs(imbalance), 1.0),
                direction=Direction.NEGATIVE,
                confidence=min(abs(imbalance) / total_input, 1.0) if total_input > 0 else 0.5,
                source_framework=FrameworkID.CONSERVATION_OF_ENERGY,
                is_hidden=True,
                is_user_stated=False,
                evidence=[
                    f"Total input force: {total_input:.2f}",
                    f"Total visible drain: {total_drain:.2f}",
                    f"Unaccounted energy: {imbalance:.2f}",
                    "Conservation of Energy requires this gap to be filled by a hidden variable.",
                ],
            )
            variables_found.append(hidden_drain)
        else:
            # Little input, lots of output — where is the energy coming from?
            # There's a hidden source the user hasn't mentioned
            hidden_source = Variable(
                name="hidden_energy_source",
                description=(
                    f"Conservation violation: visible effort ({total_input:.2f}) "
                    f"is insufficient to explain the drain ({total_drain:.2f}). "
                    f"There is a hidden energy source powering {abs(imbalance):.2f} units."
                ),
                magnitude=min(abs(imbalance), 1.0),
                direction=Direction.POSITIVE,
                confidence=min(abs(imbalance) / total_drain, 1.0) if total_drain > 0 else 0.5,
                source_framework=FrameworkID.CONSERVATION_OF_ENERGY,
                is_hidden=True,
                is_user_stated=False,
                evidence=[
                    f"Total input force: {total_input:.2f}",
                    f"Total visible drain: {total_drain:.2f}",
                    f"Unexplained energy source: {abs(imbalance):.2f}",
                    "Conservation of Energy requires a hidden source to explain this output.",
                ],
            )
            variables_found.append(hidden_source)

    return Perspective(
        framework=FrameworkID.CONSERVATION_OF_ENERGY,
        domain=Domain.PHYSICS,
        content=_build_conservation_analysis(total_input, total_drain, imbalance, variables_found),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )


def _entropy(problem: Problem, decomposition: Perspective) -> Perspective:
    """
    Thermodynamics: Entropy — decay detection.

    Systems move toward chaos unless work is constantly added.
    Measures how fast the system is falling apart and the timeline to breakdown.
    """
    variables_found = []
    forces = decomposition.variables_found

    # Entropy indicators: negative forces without counterbalancing positive work
    negative_forces = [f for f in forces if f.direction == Direction.NEGATIVE]
    positive_work = [f for f in forces if f.direction == Direction.POSITIVE]

    total_decay = sum(f.magnitude for f in negative_forces) if negative_forces else 0.0
    total_maintenance = sum(f.magnitude for f in positive_work) if positive_work else 0.0

    # Net entropy: is the system decaying faster than it's being maintained?
    net_entropy = total_decay - total_maintenance

    if net_entropy > 0.1:  # system is decaying
        decay_rate = Variable(
            name="entropy_decay_rate",
            description=(
                f"System is decaying. Disorder ({total_decay:.2f}) exceeds "
                f"maintenance work ({total_maintenance:.2f}). "
                f"Net entropy increase: {net_entropy:.2f}. "
                "Without new energy input, this system trends toward breakdown."
            ),
            magnitude=min(net_entropy, 1.0),
            direction=Direction.NEGATIVE,
            confidence=0.8,
            source_framework=FrameworkID.ENTROPY,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Decay forces: {total_decay:.2f}",
                f"Maintenance forces: {total_maintenance:.2f}",
                f"Net entropy: +{net_entropy:.2f} (increasing disorder)",
            ],
        )
        variables_found.append(decay_rate)

        # Check for broken boundary — is there a specific leak point?
        if negative_forces:
            strongest_decay = max(negative_forces, key=lambda f: f.magnitude)
            if strongest_decay.magnitude > 0.5:
                boundary_breach = Variable(
                    name="entropy_boundary_breach",
                    description=(
                        f"Primary entropy source identified: '{strongest_decay.name}' "
                        f"(magnitude: {strongest_decay.magnitude:.2f}). "
                        "This is the broken boundary where energy escapes fastest."
                    ),
                    magnitude=strongest_decay.magnitude,
                    direction=Direction.NEGATIVE,
                    confidence=0.7,
                    source_framework=FrameworkID.ENTROPY,
                    is_hidden=False,
                    is_user_stated=False,
                    evidence=[
                        f"Strongest decay force: {strongest_decay.name}",
                        f"Magnitude: {strongest_decay.magnitude:.2f}",
                        "This is where the system is losing the most energy.",
                    ],
                )
                variables_found.append(boundary_breach)

    return Perspective(
        framework=FrameworkID.ENTROPY,
        domain=Domain.PHYSICS,
        content=_build_entropy_analysis(total_decay, total_maintenance, net_entropy, variables_found),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )


def _trajectory_momentum(
    problem: Problem, decomposition: Perspective
) -> Perspective:
    """
    Kinematics: Trajectory & Momentum — where this lands and when.

    Measures velocity (speed of change) and mass (importance/size).
    Plots the point of impact long before it happens.
    """
    variables_found = []
    forces = decomposition.variables_found

    # Calculate net direction and momentum
    positive_momentum = sum(
        f.magnitude for f in forces if f.direction == Direction.POSITIVE
    )
    negative_momentum = sum(
        f.magnitude for f in forces if f.direction == Direction.NEGATIVE
    )

    net_velocity = positive_momentum - negative_momentum  # positive = improving, negative = worsening
    total_mass = sum(f.magnitude for f in forces)  # total weight of the situation

    # Momentum = mass × velocity
    momentum = total_mass * abs(net_velocity) if total_mass > 0 else 0.0

    trajectory_direction = Direction.POSITIVE if net_velocity > 0 else Direction.NEGATIVE
    if abs(net_velocity) < 0.05:
        trajectory_direction = Direction.NEUTRAL

    trajectory = Variable(
        name="trajectory",
        description=(
            f"Net velocity: {net_velocity:+.2f} "
            f"({'improving' if net_velocity > 0 else 'worsening' if net_velocity < 0 else 'stagnant'}). "
            f"Total mass (situation weight): {total_mass:.2f}. "
            f"Momentum: {momentum:.2f}. "
            f"{'High momentum — hard to change course.' if momentum > 0.5 else 'Low momentum — course change still possible.'}"
        ),
        magnitude=min(momentum, 1.0),
        direction=trajectory_direction,
        confidence=0.75,
        source_framework=FrameworkID.TRAJECTORY_MOMENTUM,
        is_hidden=False,
        is_user_stated=False,
        evidence=[
            f"Positive forces: {positive_momentum:.2f}",
            f"Negative forces: {negative_momentum:.2f}",
            f"Net velocity: {net_velocity:+.2f}",
            f"Momentum: {momentum:.2f}",
        ],
    )
    variables_found.append(trajectory)

    return Perspective(
        framework=FrameworkID.TRAJECTORY_MOMENTUM,
        domain=Domain.PHYSICS,
        content=_build_trajectory_analysis(net_velocity, total_mass, momentum, trajectory_direction),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )


def _potential_kinetic(
    problem: Problem, decomposition: Perspective
) -> Perspective:
    """
    Potential vs. Kinetic Energy — pressure detection.

    Every action (kinetic) was once a tension (potential).
    Identifies where pressure is building that hasn't released yet.
    Predicts the next explosion before any kinetic movement occurs.
    """
    variables_found = []
    forces = decomposition.variables_found

    # High-magnitude neutral forces = stored potential energy
    # They haven't moved yet, but they're loaded
    potential_forces = [
        f for f in forces
        if f.direction == Direction.NEUTRAL and f.magnitude > 0.3
    ]

    # High-magnitude forces with low confidence = unstable energy
    # Something big is there but we're not sure which way it'll go
    unstable_forces = [
        f for f in forces
        if f.magnitude > 0.4 and f.confidence < 0.5
    ]

    for pf in potential_forces:
        stored_pressure = Variable(
            name=f"potential_energy_{pf.name}",
            description=(
                f"Stored potential energy in '{pf.name}' "
                f"(magnitude: {pf.magnitude:.2f}, direction: neutral/unreleased). "
                "This tension has not converted to action yet. "
                "When it releases, it will move fast and hard."
            ),
            magnitude=pf.magnitude,
            direction=Direction.NEUTRAL,
            confidence=0.7,
            source_framework=FrameworkID.POTENTIAL_KINETIC,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Source force: {pf.name}",
                f"Magnitude: {pf.magnitude:.2f} (significant stored energy)",
                "Direction: neutral — hasn't released yet",
                "Risk: sudden conversion to kinetic energy (crisis event)",
            ],
        )
        variables_found.append(stored_pressure)

    for uf in unstable_forces:
        if uf not in potential_forces:  # avoid double-counting
            instability = Variable(
                name=f"unstable_energy_{uf.name}",
                description=(
                    f"Unstable energy in '{uf.name}' "
                    f"(magnitude: {uf.magnitude:.2f}, confidence: {uf.confidence:.2f}). "
                    "High energy but uncertain direction. "
                    "This could tip either way — and when it does, it carries significant force."
                ),
                magnitude=uf.magnitude,
                direction=Direction.NEUTRAL,
                confidence=uf.confidence,
                source_framework=FrameworkID.POTENTIAL_KINETIC,
                is_hidden=False,
                is_user_stated=False,
                evidence=[
                    f"Source force: {uf.name}",
                    f"Magnitude: {uf.magnitude:.2f} (high energy)",
                    f"Confidence: {uf.confidence:.2f} (uncertain direction)",
                    "Risk: could release in either direction with significant impact",
                ],
            )
            variables_found.append(instability)

    return Perspective(
        framework=FrameworkID.POTENTIAL_KINETIC,
        domain=Domain.PHYSICS,
        content=_build_potential_analysis(potential_forces, unstable_forces, variables_found),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.8,
    )


def _equilibrium(
    problem: Problem, decomposition: Perspective
) -> Perspective:
    """
    Equilibrium & Net Force — stuckness diagnosis.

    If a situation is stuck, there are equal and opposite forces
    pushing against each other. Finds the hidden counter-force.
    """
    variables_found = []
    forces = decomposition.variables_found

    # Find opposing force pairs — positive and negative forces of similar magnitude
    positive_forces = [f for f in forces if f.direction == Direction.POSITIVE]
    negative_forces = [f for f in forces if f.direction == Direction.NEGATIVE]

    opposing_pairs = []
    for pf in positive_forces:
        for nf in negative_forces:
            # If magnitudes are within 0.15 of each other, they're in equilibrium
            if abs(pf.magnitude - nf.magnitude) < 0.15:
                opposing_pairs.append((pf, nf))

    for pf, nf in opposing_pairs:
        stuckness = Variable(
            name=f"equilibrium_{pf.name}_vs_{nf.name}",
            description=(
                f"Equilibrium detected: '{pf.name}' ({pf.magnitude:.2f} positive) "
                f"is being cancelled by '{nf.name}' ({nf.magnitude:.2f} negative). "
                "These forces are nearly equal — the system is stuck. "
                "Resolution requires removing the counter-force, not adding more power."
            ),
            magnitude=(pf.magnitude + nf.magnitude) / 2,
            direction=Direction.NEUTRAL,
            confidence=0.8,
            source_framework=FrameworkID.EQUILIBRIUM,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Positive force: {pf.name} ({pf.magnitude:.2f})",
                f"Negative force: {nf.name} ({nf.magnitude:.2f})",
                f"Imbalance: {abs(pf.magnitude - nf.magnitude):.2f} (near zero = stuck)",
                "These forces cancel each other out.",
            ],
        )
        variables_found.append(stuckness)

    # If no pairs but forces exist and net is near zero — hidden counter-force
    if not opposing_pairs and forces:
        net = sum(
            f.magnitude * (1 if f.direction == Direction.POSITIVE else -1)
            for f in forces
            if f.direction in (Direction.POSITIVE, Direction.NEGATIVE)
        )
        if abs(net) < 0.1 and len(forces) > 1:
            hidden_counter = Variable(
                name="hidden_counter_force",
                description=(
                    "Net force is near zero despite multiple active forces. "
                    "There is a hidden counter-force the user hasn't named. "
                    "This is likely a commitment, belief, or relationship "
                    "that perfectly cancels every move they make."
                ),
                magnitude=0.6,
                direction=Direction.NEGATIVE,
                confidence=0.6,
                source_framework=FrameworkID.EQUILIBRIUM,
                is_hidden=True,
                is_user_stated=False,
                evidence=[
                    f"Net force: {net:.2f} (near zero despite active forces)",
                    f"Active forces: {len(forces)}",
                    "Something unseen is holding the system in place.",
                ],
            )
            variables_found.append(hidden_counter)

    return Perspective(
        framework=FrameworkID.EQUILIBRIUM,
        domain=Domain.PHYSICS,
        content=_build_equilibrium_analysis(opposing_pairs, variables_found),
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )


# ---------------------------------------------------------------------------
# Output extraction helpers
# ---------------------------------------------------------------------------

def _extract_root_causes(
    all_variables: list[Variable], perspectives: list[Perspective]
) -> list[RootCause]:
    """Extract candidate root causes from high-confidence hidden variables."""
    candidates = []
    for var in all_variables:
        if var.is_hidden and var.confidence >= 0.5:
            # Find which frameworks agree this is significant
            agreeing = [
                p.framework for p in perspectives
                if any(v.name == var.name for v in p.variables_found)
            ]
            candidate = RootCause(
                variable=var,
                evidence_chain=var.evidence,
                confidence=var.confidence,
                frameworks_that_agree=agreeing,
            )
            candidates.append(candidate)
    return candidates


def _extract_consequences(
    trajectory: Perspective, entropy: Perspective
) -> list[Consequence]:
    """Build consequences from trajectory projection and entropy analysis."""
    consequences = []

    for var in trajectory.variables_found:
        if var.direction == Direction.NEGATIVE:
            severity = Severity.HIGH if var.magnitude > 0.6 else Severity.MODERATE
            consequences.append(Consequence(
                description=(
                    f"At current trajectory (velocity: {var.description}), "
                    "the situation is worsening."
                ),
                timeframe="3-6 months at current rate",
                severity=severity,
                probability=var.confidence,
                trajectory_framework=FrameworkID.TRAJECTORY_MOMENTUM,
                is_reversible=var.magnitude < 0.8,
                reversal_window="1-3 months" if var.magnitude < 0.8 else "closing",
            ))

    for var in entropy.variables_found:
        if "decay_rate" in var.name and var.magnitude > 0.3:
            consequences.append(Consequence(
                description=(
                    f"Entropy analysis: system decay rate is {var.magnitude:.2f}. "
                    "Without new energy input, breakdown is the trajectory."
                ),
                timeframe="6-12 months to significant deterioration",
                severity=Severity.HIGH if var.magnitude > 0.5 else Severity.MODERATE,
                probability=0.8,
                trajectory_framework=FrameworkID.ENTROPY,
                is_reversible=True,
                reversal_window="Requires immediate energy input to reverse",
            ))

    return consequences


# ---------------------------------------------------------------------------
# Analysis text builders
# ---------------------------------------------------------------------------

def _build_first_principles_analysis(
    problem: Problem, forces: list[Variable]
) -> str:
    lines = [
        "FIRST PRINCIPLES DECOMPOSITION",
        f"Problem: {problem.statement}",
        f"Irreducible forces found: {len(forces)}",
        "",
    ]
    for f in forces:
        lines.append(f"  - {f.name}: {f.description} [magnitude: {f.magnitude:.2f}, direction: {f.direction.value}]")
    return "\n".join(lines)


def _build_conservation_analysis(
    total_input: float, total_drain: float, imbalance: float,
    variables_found: list[Variable],
) -> str:
    lines = [
        "CONSERVATION OF ENERGY AUDIT",
        f"Total input energy: {total_input:.2f}",
        f"Total drain energy: {total_drain:.2f}",
        f"Imbalance: {imbalance:+.2f}",
        "",
    ]
    if variables_found:
        lines.append("Conservation violations found:")
        for v in variables_found:
            lines.append(f"  - {v.name}: {v.description}")
    else:
        lines.append("Energy is balanced — no hidden drains or sources detected.")
    return "\n".join(lines)


def _build_entropy_analysis(
    total_decay: float, total_maintenance: float, net_entropy: float,
    variables_found: list[Variable],
) -> str:
    lines = [
        "ENTROPY ANALYSIS",
        f"Decay forces: {total_decay:.2f}",
        f"Maintenance forces: {total_maintenance:.2f}",
        f"Net entropy: {net_entropy:+.2f}",
        "",
    ]
    if net_entropy > 0.1:
        lines.append("System is decaying faster than it's being maintained.")
    else:
        lines.append("System entropy is stable or being actively maintained.")
    for v in variables_found:
        lines.append(f"  - {v.name}: {v.description}")
    return "\n".join(lines)


def _build_trajectory_analysis(
    net_velocity: float, total_mass: float, momentum: float,
    direction: Direction,
) -> str:
    lines = [
        "TRAJECTORY & MOMENTUM ANALYSIS",
        f"Net velocity: {net_velocity:+.2f} ({direction.value})",
        f"Situation mass: {total_mass:.2f}",
        f"Momentum: {momentum:.2f}",
        "",
    ]
    if direction == Direction.NEGATIVE:
        lines.append("Trajectory is NEGATIVE — situation is worsening over time.")
        if momentum > 0.5:
            lines.append("HIGH MOMENTUM — changing course will require significant force.")
    elif direction == Direction.POSITIVE:
        lines.append("Trajectory is POSITIVE — situation is improving.")
    else:
        lines.append("Trajectory is STAGNANT — situation is not moving in either direction.")
    return "\n".join(lines)


def _build_potential_analysis(
    potential_forces: list[Variable], unstable_forces: list[Variable],
    variables_found: list[Variable],
) -> str:
    lines = [
        "POTENTIAL vs KINETIC ENERGY ANALYSIS",
        f"Stored potential forces detected: {len(potential_forces)}",
        f"Unstable energy sources detected: {len(unstable_forces)}",
        "",
    ]
    if variables_found:
        lines.append("Pressure points:")
        for v in variables_found:
            lines.append(f"  - {v.name}: {v.description}")
    else:
        lines.append("No significant stored pressure detected.")
    return "\n".join(lines)


def _build_equilibrium_analysis(
    opposing_pairs: list[tuple[Variable, Variable]],
    variables_found: list[Variable],
) -> str:
    lines = [
        "EQUILIBRIUM & NET FORCE ANALYSIS",
        f"Opposing force pairs found: {len(opposing_pairs)}",
        "",
    ]
    if opposing_pairs:
        lines.append("Stuckness points (equal and opposite forces):")
        for pf, nf in opposing_pairs:
            lines.append(f"  - {pf.name} ({pf.magnitude:.2f}) ←→ {nf.name} ({nf.magnitude:.2f})")
    for v in variables_found:
        if "hidden" in v.name:
            lines.append(f"  - {v.name}: {v.description}")
    if not opposing_pairs and not variables_found:
        lines.append("No equilibrium detected — forces are not balanced.")
    return "\n".join(lines)


def _build_raw_analysis(perspectives: list[Perspective]) -> str:
    """Combine all perspective analyses into one raw output."""
    sections = []
    for p in perspectives:
        sections.append(p.content)
    return "\n\n---\n\n".join(sections)
