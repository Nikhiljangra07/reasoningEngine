"""
Philosophy Epistemic Pipeline.

5 concepts forming a complete epistemic toolkit. The sequence is a logical
pipeline — each concept's output feeds the next:

1. Ontology — define reality (essential vs accidental)
2. Epistemology — audit knowledge (fact vs belief vs assumption)
3. Phenomenology — map perception (horizon of consciousness)
4. Dialectics — find conflict (thesis/antithesis/synthesis on the SITUATION)
5. Teleology — trace trajectory (hidden utility, why the problem persists)

NOTE: Philosophy's Dialectics operates on the SITUATION's structure.
      Psychology's Dialectical Thinking operates on the PERSON's experience.
      Same tool, different target.

ISOLATION: Imports ONLY from src.core.types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    Direction,
    Domain,
    DomainOutput,
    FrameworkID,
    Perspective,
    Problem,
    RootCause,
    SignalType,
    Variable,
)


# ===========================================================================
# Concept 1: Ontology (The Study of Being)
# ===========================================================================

@dataclass
class OntologyResult:
    """Result of ontological analysis — the irreducible essence."""
    essential_variables: list[Variable]
    accidental_variables: list[Variable]
    ontological_core: str
    essence_statement: str
    variables_reclassified: int


def run_ontology(
    problem: Problem,
    all_domain_outputs: list[DomainOutput],
) -> tuple[Perspective, OntologyResult]:
    """
    Ontology — strips the problem to its essence.

    Distinguishes between:
    - Essential properties: what the problem IS at its core.
      Remove this and the fundamental nature changes.
    - Accidental properties: surface noise, how the problem LOOKS.
      Remove this and the core problem remains unchanged.
    """
    # Collect all variables from problem + all upstream domain outputs
    all_vars = list(problem.variables)
    for output in all_domain_outputs:
        for p in output.perspectives:
            all_vars.extend(p.variables_found)

    essential = []
    accidental = []
    reclassified = 0

    for var in all_vars:
        is_essential = _essential_test(var, problem, all_vars)

        if is_essential:
            essential.append(var)
        else:
            accidental.append(var)
            if var.is_user_stated:
                reclassified += 1  # user thought this was important — ontology disagrees

    # Build the ontological core
    core_desc = _build_ontological_core(essential, problem)
    essence = _build_essence_statement(essential)

    result = OntologyResult(
        essential_variables=essential,
        accidental_variables=accidental,
        ontological_core=core_desc,
        essence_statement=essence,
        variables_reclassified=reclassified,
    )

    variables_found = []
    if reclassified > 0:
        variables_found.append(Variable(
            name="ontological_reclassification",
            description=(
                f"{reclassified} user-stated variable(s) reclassified as accidental. "
                "The user is treating surface features as core to the problem. "
                f"Essential core: {essence}"
            ),
            magnitude=min(reclassified / len(problem.variables), 1.0) if problem.variables else 0.5,
            direction=Direction.NEUTRAL,
            confidence=0.7,
            source_framework=FrameworkID.ONTOLOGY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Essential variables: {len(essential)}",
                f"Accidental variables: {len(accidental)}",
                f"User variables reclassified: {reclassified}",
                f"Ontological core: {core_desc[:100]}",
            ],
        ))

    content = (
        "ONTOLOGY ANALYSIS\n"
        f"Essential variables: {len(essential)}\n"
        f"Accidental variables: {len(accidental)}\n"
        f"User variables reclassified: {reclassified}\n\n"
        f"ONTOLOGICAL CORE: {core_desc}\n\n"
        f"ESSENCE: {essence}\n"
    )

    perspective = Perspective(
        framework=FrameworkID.ONTOLOGY,
        domain=Domain.PHILOSOPHY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )

    return perspective, result


def _essential_test(var: Variable, problem: Problem, all_vars: list[Variable]) -> bool:
    """
    Test if a variable is essential or accidental.

    Essential = remove it and the fundamental nature of the problem changes.
    Three tests:
    1. essential_test: would removing this change the problem fundamentally?
    2. substance_vs_attribute: does it describe WHAT or HOW?
    3. invariance_test: does it survive reframing?
    """
    # Test 1: High magnitude + high confidence + negative/circular = likely essential
    # (it's a strong force that the problem cannot exist without)
    fundamental_score = var.magnitude * var.confidence
    if var.direction in (Direction.NEGATIVE, Direction.CIRCULAR):
        fundamental_score *= 1.3  # problems are defined by their negative forces

    # Test 2: Substance vs attribute
    # Variables that are about the user's STATE are essential
    # Variables that are about CIRCUMSTANCES are more likely accidental
    is_state = var.is_hidden or var.direction == Direction.CIRCULAR

    # Test 3: Invariance — does this variable appear across multiple frameworks?
    # (if multiple domains found the same thing, it's invariant to reframing)
    framework_count = sum(
        1 for v in all_vars
        if v.name == var.name and v.source_framework != var.source_framework
    )
    is_invariant = framework_count >= 1

    # Essential if: high fundamental score, OR state-level, OR invariant across frameworks
    return fundamental_score > 0.5 or is_state or is_invariant


def _build_ontological_core(essential: list[Variable], problem: Problem) -> str:
    """Build a description of the problem's irreducible core."""
    if not essential:
        return "Unable to determine ontological core — insufficient essential variables."

    # The core is the intersection of essential forces
    directions = set(v.direction for v in essential)
    avg_magnitude = sum(v.magnitude for v in essential) / len(essential)

    core_names = [v.name for v in essential[:5]]

    return (
        f"The irreducible core of this problem consists of {len(essential)} essential forces: "
        f"{', '.join(core_names)}{'...' if len(essential) > 5 else ''}. "
        f"Average force magnitude: {avg_magnitude:.2f}. "
        f"Force directions: {', '.join(d.value for d in directions)}. "
        "Everything else is surface presentation."
    )


def _build_essence_statement(essential: list[Variable]) -> str:
    """Build a single-sentence essence statement."""
    if not essential:
        return "Essence undetermined."

    # Find the highest-magnitude essential variable — this is the problem's gravity
    core = max(essential, key=lambda v: v.magnitude * v.confidence)
    return (
        f"At its essence, this problem IS about '{core.name}' "
        f"({core.direction.value}, magnitude: {core.magnitude:.2f}). "
        "Strip away everything else, and this remains."
    )


# ===========================================================================
# Concept 2: Epistemology (The Study of Knowledge)
# ===========================================================================

@dataclass
class EpistemicEntry:
    """Classification of a single claim/variable."""
    variable_name: str
    classification: str             # "fact", "belief", "assumption", "opinion"
    evidence_basis: str             # "direct_experience", "secondhand", "inference", "none"
    justification_chain: str
    verification_status: str        # "verified", "unverified", "unfalsifiable"


@dataclass
class EpistemologyResult:
    """Full epistemic map of the problem."""
    epistemic_map: list[EpistemicEntry]
    false_prior_candidates: list[str]
    knowledge_gaps: list[str]
    blind_spots: list[str]


def run_epistemology(
    problem: Problem,
    ontology_result: OntologyResult,
) -> tuple[Perspective, EpistemologyResult]:
    """
    Epistemology — audits every piece of information.

    Classifies each claim as:
    - FACT: evidence + verification + justification
    - BELIEF: conviction without full evidence
    - ASSUMPTION: never examined, taken as given
    - OPINION: preference without evidence

    Flags all ASSUMPTIONS — these are where false priors live.
    """
    entries = []
    false_priors = []
    gaps = []
    blind_spots = []
    variables_found = []

    # Only analyze essential variables (ontology already filtered)
    for var in ontology_result.essential_variables:
        entry = _classify_epistemically(var)
        entries.append(entry)

        if entry.classification == "assumption":
            false_priors.append(
                f"'{var.name}' is classified as ASSUMPTION — taken as given, never examined. "
                f"Evidence basis: {entry.evidence_basis}. "
                "The system may be building on unstable ground here."
            )

    # Detect knowledge gaps — essential variables with low confidence
    for var in ontology_result.essential_variables:
        if var.confidence < 0.5:
            gaps.append(
                f"Knowledge gap on '{var.name}': essential to the problem "
                f"but confidence is only {var.confidence:.2f}. "
                "More information needed here."
            )

    # Detect blind spots — things the user should know but hasn't mentioned
    user_directions = {v.direction for v in problem.variables if v.is_user_stated}
    if Direction.NEGATIVE not in user_directions and any(
        v.direction == Direction.NEGATIVE for v in ontology_result.essential_variables
    ):
        blind_spots.append(
            "User has not acknowledged any negative forces, "
            "but ontology found essential negative variables. "
            "This is a structural blind spot."
        )
    if Direction.POSITIVE not in user_directions and any(
        v.direction == Direction.POSITIVE for v in ontology_result.essential_variables
    ):
        blind_spots.append(
            "User has not acknowledged any positive forces, "
            "but ontology found essential positive variables. "
            "Resources or strengths may be invisible to the user."
        )

    # Create variables for significant findings
    assumption_count = sum(1 for e in entries if e.classification == "assumption")
    if assumption_count > 0:
        variables_found.append(Variable(
            name="epistemic_assumptions",
            description=(
                f"{assumption_count} essential variable(s) classified as ASSUMPTION. "
                "These are unexamined beliefs the user treats as facts. "
                f"False prior candidates: {', '.join(f[:50] for f in false_priors[:3])}"
            ),
            magnitude=min(assumption_count / max(len(entries), 1), 1.0),
            direction=Direction.NEGATIVE,
            confidence=0.75,
            source_framework=FrameworkID.EPISTEMOLOGY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Assumptions found: {assumption_count}",
                *[f"False prior: {fp[:80]}" for fp in false_priors[:3]],
            ],
        ))

    if gaps:
        variables_found.append(Variable(
            name="knowledge_gaps",
            description=f"{len(gaps)} knowledge gap(s) in essential variables. {gaps[0][:100]}",
            magnitude=min(len(gaps) / 5.0, 1.0),
            direction=Direction.NEGATIVE,
            confidence=0.7,
            source_framework=FrameworkID.EPISTEMOLOGY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[g[:80] for g in gaps[:3]],
        ))

    result = EpistemologyResult(
        epistemic_map=entries,
        false_prior_candidates=false_priors,
        knowledge_gaps=gaps,
        blind_spots=blind_spots,
    )

    fact_count = sum(1 for e in entries if e.classification == "fact")
    belief_count = sum(1 for e in entries if e.classification == "belief")
    opinion_count = sum(1 for e in entries if e.classification == "opinion")

    content = (
        "EPISTEMOLOGY ANALYSIS\n"
        f"Facts: {fact_count} | Beliefs: {belief_count} | "
        f"Assumptions: {assumption_count} | Opinions: {opinion_count}\n"
        f"False prior candidates: {len(false_priors)}\n"
        f"Knowledge gaps: {len(gaps)}\n"
        f"Blind spots: {len(blind_spots)}\n\n"
    )
    for e in entries:
        content += f"  {e.variable_name}: {e.classification.upper()} (evidence: {e.evidence_basis})\n"

    perspective = Perspective(
        framework=FrameworkID.EPISTEMOLOGY,
        domain=Domain.PHILOSOPHY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )

    return perspective, result


def _classify_epistemically(var: Variable) -> EpistemicEntry:
    """Classify a single variable as fact, belief, assumption, or opinion."""
    evidence_count = len(var.evidence)

    # Determine evidence basis
    if evidence_count >= 3:
        evidence_basis = "direct_experience"
    elif evidence_count >= 1:
        evidence_basis = "inference"
    elif var.is_user_stated:
        evidence_basis = "secondhand"  # user says so but no supporting evidence
    else:
        evidence_basis = "none"

    # Determine verification status
    if evidence_count >= 2 and var.confidence > 0.7:
        verification = "verified"
    elif var.confidence > 0.5:
        verification = "unverified"
    else:
        verification = "unfalsifiable"

    # Build justification chain
    if var.evidence:
        justification = " → ".join(var.evidence[:3])
    elif var.is_user_stated:
        justification = "User stated without explicit justification"
    else:
        justification = "No justification chain available"

    # Classification decision
    if evidence_basis == "direct_experience" and verification == "verified":
        classification = "fact"
    elif var.confidence > 0.6 and evidence_count >= 1:
        classification = "belief"
    elif var.is_user_stated and evidence_count == 0:
        classification = "assumption"
    else:
        classification = "opinion"

    return EpistemicEntry(
        variable_name=var.name,
        classification=classification,
        evidence_basis=evidence_basis,
        justification_chain=justification,
        verification_status=verification,
    )


# ===========================================================================
# Concept 3: Phenomenology (The Study of Perspective)
# ===========================================================================

@dataclass
class PhenomenologyResult:
    """Mapping of the user's experiential horizon."""
    experiential_frame: str
    horizon_visible: list[str]
    horizon_invisible: list[str]
    frame_reality_gap: str
    perspective_limitations: list[str]
    bridge_recommendations: list[str]


def run_phenomenology(
    problem: Problem,
    ontology_result: OntologyResult,
    epistemology_result: EpistemologyResult,
    metacognition_score: float | None = None,
) -> tuple[Perspective, PhenomenologyResult]:
    """
    Phenomenology — maps how the problem is EXPERIENCED by the user.

    The user's bias is not a MISTAKE — it is a structural limit of where
    they're standing. Maps what they CAN see and what is structurally
    invisible from their current position.
    """
    variables_found = []

    # Determine experiential frame
    frame = _determine_experiential_frame(problem)

    # Map horizon — what's visible vs invisible
    visible = _map_visible_horizon(problem)
    invisible = _map_invisible_horizon(problem, ontology_result, epistemology_result)

    # Frame-reality gap
    gap = _calculate_frame_reality_gap(frame, ontology_result, epistemology_result)

    # Perspective limitations
    limitations = _identify_limitations(problem, invisible)

    # Bridge recommendations — what would shift the user's horizon
    bridges = _recommend_bridges(invisible, limitations, metacognition_score)

    result = PhenomenologyResult(
        experiential_frame=frame,
        horizon_visible=visible,
        horizon_invisible=invisible,
        frame_reality_gap=gap,
        perspective_limitations=limitations,
        bridge_recommendations=bridges,
    )

    # Create variable if significant gap detected
    if invisible:
        variables_found.append(Variable(
            name="phenomenological_horizon_limit",
            description=(
                f"User's experiential frame: {frame}. "
                f"{len(invisible)} element(s) are structurally invisible from their position. "
                f"Frame-reality gap: {gap[:100]}"
            ),
            magnitude=min(len(invisible) / 5.0, 1.0),
            direction=Direction.NEUTRAL,
            confidence=0.7,
            source_framework=FrameworkID.PHENOMENOLOGY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Frame: {frame}",
                f"Visible: {len(visible)} elements",
                f"Invisible: {len(invisible)} elements",
                *[f"Cannot see: {inv[:60]}" for inv in invisible[:3]],
            ],
        ))

    content = (
        "PHENOMENOLOGY ANALYSIS\n"
        f"Experiential frame: {frame}\n"
        f"Visible horizon: {len(visible)} elements\n"
        f"Invisible horizon: {len(invisible)} elements\n\n"
        f"FRAME-REALITY GAP: {gap}\n\n"
        "Perspective limitations:\n"
    )
    for lim in limitations:
        content += f"  - {lim}\n"
    content += "\nBridge recommendations:\n"
    for br in bridges:
        content += f"  → {br}\n"

    perspective = Perspective(
        framework=FrameworkID.PHENOMENOLOGY,
        domain=Domain.PHILOSOPHY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )

    return perspective, result


def _determine_experiential_frame(problem: Problem) -> str:
    """How is the user experiencing this problem?"""
    user_vars = [v for v in problem.variables if v.is_user_stated]
    neg_count = sum(1 for v in user_vars if v.direction == Direction.NEGATIVE)
    pos_count = sum(1 for v in user_vars if v.direction == Direction.POSITIVE)
    total = len(user_vars) or 1

    neg_ratio = neg_count / total

    if neg_ratio > 0.7:
        return "threat — user experiences this as something happening TO them"
    elif neg_ratio > 0.5:
        return "loss — user experiences this as something being taken from them"
    elif pos_count > neg_count:
        return "opportunity — user sees possibility despite challenges"
    else:
        return "test — user experiences this as a challenge to overcome"


def _map_visible_horizon(problem: Problem) -> list[str]:
    """What the user CAN see from their current position."""
    return [
        f"'{v.name}': {v.description[:60]}"
        for v in problem.variables if v.is_user_stated
    ]


def _map_invisible_horizon(
    problem: Problem,
    ontology: OntologyResult,
    epistemology: EpistemologyResult,
) -> list[str]:
    """What is structurally invisible to the user from their current position."""
    invisible = []

    # Essential variables the user didn't state
    user_names = {v.name for v in problem.variables if v.is_user_stated}
    for var in ontology.essential_variables:
        if var.name not in user_names and not var.is_user_stated:
            invisible.append(
                f"'{var.name}' — essential to the problem but not visible to the user "
                f"(magnitude: {var.magnitude:.2f}, direction: {var.direction.value})"
            )

    # Epistemic blind spots
    for bs in epistemology.blind_spots:
        invisible.append(f"Blind spot: {bs}")

    # Assumptions the user doesn't know are assumptions
    for fp in epistemology.false_prior_candidates:
        invisible.append(f"False prior (invisible to user): {fp[:80]}")

    return invisible


def _calculate_frame_reality_gap(
    frame: str,
    ontology: OntologyResult,
    epistemology: EpistemologyResult,
) -> str:
    """Calculate the gap between user's experience and assessed reality."""
    assumption_count = sum(
        1 for e in epistemology.epistemic_map if e.classification == "assumption"
    )
    accidental_count = len(ontology.accidental_variables)

    if assumption_count > 2 or accidental_count > 3:
        return (
            f"SIGNIFICANT GAP. User's frame ({frame}) is built on "
            f"{assumption_count} unexamined assumptions and focuses on "
            f"{accidental_count} accidental properties. The experienced reality "
            "diverges substantially from the ontological essence."
        )
    elif assumption_count > 0:
        return (
            f"MODERATE GAP. User's frame ({frame}) contains "
            f"{assumption_count} assumption(s) that may distort their view. "
            "Some accidental properties are being treated as essential."
        )
    return (
        f"MINIMAL GAP. User's frame ({frame}) is reasonably aligned "
        "with the ontological essence. Few untested assumptions detected."
    )


def _identify_limitations(problem: Problem, invisible: list[str]) -> list[str]:
    """Identify what structural limits prevent the user from seeing more."""
    limitations = []

    if invisible:
        limitations.append(
            f"Position limitation: {len(invisible)} element(s) cannot be seen from "
            "the user's current vantage point."
        )

    neg_vars = [v for v in problem.variables if v.direction == Direction.NEGATIVE and v.is_user_stated]
    if len(neg_vars) > len(problem.variables) * 0.6:
        limitations.append(
            "Emotional limitation: high negative focus narrows the visible horizon. "
            "Positive elements and opportunities become structurally harder to see."
        )

    high_conf_count = sum(1 for v in problem.variables if v.confidence > 0.8 and v.is_user_stated)
    if high_conf_count > len(problem.variables) * 0.5:
        limitations.append(
            "Certainty limitation: many variables held with high confidence, "
            "reducing openness to information that contradicts the current frame."
        )

    return limitations


def _recommend_bridges(
    invisible: list[str],
    limitations: list[str],
    metacognition_score: float | None,
) -> list[str]:
    """Recommend what would shift the user's horizon."""
    bridges = []

    if invisible:
        bridges.append(
            "Information bridge: present the invisible elements gradually, "
            "starting with the least threatening."
        )

    if any("emotional" in lim.lower() for lim in limitations):
        bridges.append(
            "Emotional processing bridge: the user needs to move through "
            "the emotional response before new information can land."
        )

    if any("certainty" in lim.lower() for lim in limitations):
        bridges.append(
            "Uncertainty bridge: introduce doubt gently — not by attacking beliefs "
            "but by presenting evidence that doesn't fit the current picture."
        )

    if metacognition_score is not None and metacognition_score < 0.4:
        bridges.append(
            "Metacognition bridge: the user has low self-awareness. "
            "Build reflective capacity before confronting findings directly."
        )

    return bridges if bridges else ["No specific bridge needed — user's frame is well-aligned."]


# ===========================================================================
# Concept 4: Dialectics (The Logic of Conflict)
# ===========================================================================

@dataclass
class PhilosophicalDialecticsResult:
    """Result of dialectical analysis on the SITUATION's structure."""
    thesis: str
    antithesis: str
    tension_point: str
    synthesis: str
    synthesis_is_variable_d: bool
    stability_score: float
    new_tensions: list[str]


def run_dialectics(
    problem: Problem,
    ontology_result: OntologyResult,
    epistemology_result: EpistemologyResult,
    phenomenology_result: PhenomenologyResult,
    physics_contradictions: list[Variable] | None = None,
) -> tuple[Perspective, PhilosophicalDialecticsResult]:
    """
    Dialectics — finds the internal tension driving the problem.

    Assumes every situation contains opposing forces fighting for control.
    Applies Thesis-Antithesis-Synthesis to find Variable D.

    NOTE: This operates on the SITUATION's structure, not the person's experience.
    """
    variables_found = []

    # Extract thesis: the dominant force/position in the problem
    thesis = _extract_situational_thesis(problem, ontology_result)

    # Extract antithesis: the opposing force the thesis suppresses
    antithesis = _extract_situational_antithesis(problem, ontology_result, physics_contradictions)

    # Find the tension point
    tension_point = _find_tension_point(thesis, antithesis, phenomenology_result)

    # Generate synthesis
    synthesis, is_var_d, stability = _generate_situational_synthesis(
        thesis, antithesis, tension_point, ontology_result
    )

    # Check for new tensions
    new_tensions = _check_new_tensions(synthesis, problem)

    result = PhilosophicalDialecticsResult(
        thesis=thesis,
        antithesis=antithesis,
        tension_point=tension_point,
        synthesis=synthesis,
        synthesis_is_variable_d=is_var_d,
        stability_score=stability,
        new_tensions=new_tensions,
    )

    # Create variable — the synthesis itself may be Variable D
    if is_var_d:
        variables_found.append(Variable(
            name="dialectical_variable_d",
            description=(
                f"Dialectical synthesis IS Variable D: {synthesis} "
                f"(stability: {stability:.2f})"
            ),
            magnitude=stability,
            direction=Direction.NEUTRAL,
            confidence=stability,
            source_framework=FrameworkID.DIALECTICS,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Thesis: {thesis[:80]}",
                f"Antithesis: {antithesis[:80]}",
                f"Tension point: {tension_point[:80]}",
                f"Synthesis: {synthesis[:80]}",
            ],
        ))

    variables_found.append(Variable(
        name="situational_tension",
        description=(
            f"Core tension: {tension_point} "
            f"(thesis: {thesis[:50]}... vs antithesis: {antithesis[:50]}...)"
        ),
        magnitude=0.8 if is_var_d else 0.6,
        direction=Direction.NEUTRAL,
        confidence=stability,
        source_framework=FrameworkID.DIALECTICS,
        is_hidden=False,
        is_user_stated=False,
        evidence=[
            f"Thesis: {thesis[:80]}",
            f"Antithesis: {antithesis[:80]}",
            f"Stability: {stability:.2f}",
        ],
    ))

    content = (
        "PHILOSOPHICAL DIALECTICS\n"
        f"THESIS: {thesis}\n"
        f"ANTITHESIS: {antithesis}\n"
        f"TENSION POINT: {tension_point}\n"
        f"SYNTHESIS: {synthesis}\n"
        f"Synthesis IS Variable D: {is_var_d}\n"
        f"Stability: {stability:.2f}\n"
    )
    if new_tensions:
        content += "\nNEW TENSIONS CREATED:\n"
        for nt in new_tensions:
            content += f"  - {nt}\n"

    perspective = Perspective(
        framework=FrameworkID.DIALECTICS,
        domain=Domain.PHILOSOPHY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )

    return perspective, result


def _extract_situational_thesis(problem: Problem, ontology: OntologyResult) -> str:
    """Extract the dominant force/position in the situation."""
    essential = ontology.essential_variables
    if not essential:
        return "No dominant thesis identified — essential variables insufficient."

    # The thesis is typically the user's primary stated position
    user_essential = [v for v in essential if v.is_user_stated]
    if user_essential:
        strongest = max(user_essential, key=lambda v: v.magnitude * v.confidence)
        return (
            f"The dominant force is '{strongest.name}' "
            f"({strongest.direction.value}, magnitude: {strongest.magnitude:.2f}): "
            f"{strongest.description}"
        )

    strongest = max(essential, key=lambda v: v.magnitude)
    return f"The dominant force is '{strongest.name}': {strongest.description}"


def _extract_situational_antithesis(
    problem: Problem,
    ontology: OntologyResult,
    physics_contradictions: list[Variable] | None,
) -> str:
    """Extract the opposing force that the thesis suppresses or ignores."""
    essential = ontology.essential_variables

    # Find the strongest essential force in the OPPOSITE direction of the thesis
    user_essential = [v for v in essential if v.is_user_stated]
    if user_essential:
        thesis_var = max(user_essential, key=lambda v: v.magnitude * v.confidence)
        # Find opposing
        opposing = [
            v for v in essential
            if v.direction != thesis_var.direction
            and v.direction != Direction.NEUTRAL
        ]
        if opposing:
            strongest_opp = max(opposing, key=lambda v: v.magnitude)
            return (
                f"The opposing force is '{strongest_opp.name}' "
                f"({strongest_opp.direction.value}, magnitude: {strongest_opp.magnitude:.2f}): "
                f"{strongest_opp.description}"
            )

    # Use physics contradictions if available
    if physics_contradictions:
        strongest = max(physics_contradictions, key=lambda v: v.magnitude)
        return (
            f"Physics contradiction: '{strongest.name}' — {strongest.description}"
        )

    return "No clear antithesis identified from available data."


def _find_tension_point(
    thesis: str, antithesis: str, phenomenology: PhenomenologyResult
) -> str:
    """Find the exact point where thesis and antithesis collide."""
    if phenomenology.frame_reality_gap and "SIGNIFICANT" in phenomenology.frame_reality_gap:
        return (
            f"The tension lives in the frame-reality gap: {phenomenology.frame_reality_gap[:100]}. "
            "The user's experience (thesis) and the structural reality (antithesis) "
            "collide at the point where the user's frame cannot accommodate the facts."
        )

    return (
        "The tension is between the dominant force and its suppressed opposite. "
        "Both exist simultaneously — the problem persists because neither can win."
    )


def _generate_situational_synthesis(
    thesis: str, antithesis: str, tension_point: str,
    ontology: OntologyResult,
) -> tuple[str, bool, float]:
    """Generate the dialectical synthesis — the truth from the contradiction."""
    essential = ontology.essential_variables
    hidden_essential = [v for v in essential if v.is_hidden]

    # If there are hidden essential variables, the synthesis reveals them
    if hidden_essential:
        strongest_hidden = max(hidden_essential, key=lambda v: v.magnitude * v.confidence)
        synthesis = (
            f"The thesis and antithesis are both incomplete views of the same reality. "
            f"The hidden force '{strongest_hidden.name}' ({strongest_hidden.description[:80]}) "
            f"is the Variable D that both thesis and antithesis orbit around. "
            "Neither can see it from their position alone."
        )
        stability = strongest_hidden.confidence
        return synthesis, True, stability

    # Otherwise, the synthesis is the integration of both forces
    synthesis = (
        "The thesis and antithesis represent two aspects of the same problem. "
        "Neither is wrong — they are partial views. The synthesis: both forces "
        "coexist and the problem will not resolve by choosing one over the other. "
        "Resolution requires addressing the STRUCTURE that produces both forces."
    )
    return synthesis, False, 0.6


def _check_new_tensions(synthesis: str, problem: Problem) -> list[str]:
    """Check if the synthesis creates new contradictions."""
    tensions = []

    # If any user variable is very high confidence and the synthesis challenges it
    for var in problem.variables:
        if var.is_user_stated and var.confidence > 0.85:
            tensions.append(
                f"The synthesis may conflict with user's strongly-held belief in "
                f"'{var.name}' (confidence: {var.confidence:.2f}). "
                "Delivery must account for this resistance."
            )

    return tensions


# ===========================================================================
# Concept 5: Teleology (The Study of Purpose/Ends)
# ===========================================================================

@dataclass
class TeleologyResult:
    """Analysis of the problem's hidden utility and purpose."""
    hidden_utility: str
    hidden_utility_confidence: float
    telos_trajectory: str
    function_as_solution: bool
    deeper_problem_hypothesis: str
    purpose_statement: str
    trajectory_divergence: str


def run_teleology(
    problem: Problem,
    ontology_result: OntologyResult,
    epistemology_result: EpistemologyResult,
    phenomenology_result: PhenomenologyResult,
    dialectics_result: PhilosophicalDialecticsResult,
    physics_trajectory: list[Variable] | None = None,
    motivated_reasoning: dict | None = None,
) -> tuple[Perspective, TeleologyResult]:
    """
    Teleology — reveals the hidden utility of the problem.

    Instead of looking at what CAUSED the problem (the past),
    looks at what PURPOSE the problem is serving (the future).

    Why does the problem persist despite the user wanting it solved?
    """
    variables_found = []

    # Search for hidden utility
    utility, utility_confidence = _search_hidden_utility(
        problem, ontology_result, epistemology_result, phenomenology_result
    )

    # Map the telos (trajectory of purpose)
    telos = _map_telos(problem, dialectics_result)

    # Check if the problem is functioning as a solution to a deeper problem
    is_function, deeper_problem = _check_function_as_solution(
        problem, ontology_result, utility
    )

    # Compare with physics trajectory
    divergence = _compare_trajectories(physics_trajectory, telos)

    # Purpose statement
    purpose = _build_purpose_statement(utility, is_function, deeper_problem, utility_confidence)

    result = TeleologyResult(
        hidden_utility=utility,
        hidden_utility_confidence=utility_confidence,
        telos_trajectory=telos,
        function_as_solution=is_function,
        deeper_problem_hypothesis=deeper_problem,
        purpose_statement=purpose,
        trajectory_divergence=divergence,
    )

    # Variable: the hidden purpose
    if utility_confidence > 0.4:
        variables_found.append(Variable(
            name="hidden_purpose",
            description=(
                f"Teleological finding: {purpose} "
                f"(confidence: {utility_confidence:.2f})"
            ),
            magnitude=utility_confidence,
            direction=Direction.NEGATIVE,  # hidden utility = resistance to change
            confidence=utility_confidence,
            source_framework=FrameworkID.TELEOLOGY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Hidden utility: {utility[:80]}",
                f"Function as solution: {is_function}",
                f"Deeper problem: {deeper_problem[:80]}",
                f"Trajectory divergence: {divergence[:80]}",
            ],
        ))

    if is_function:
        variables_found.append(Variable(
            name="problem_as_solution",
            description=(
                f"This problem is FUNCTIONING as a solution to: {deeper_problem}. "
                "Solving the surface problem would expose the deeper one."
            ),
            magnitude=0.8,
            direction=Direction.CIRCULAR,
            confidence=utility_confidence,
            source_framework=FrameworkID.TELEOLOGY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Surface problem: {problem.statement[:60]}",
                f"Deeper problem: {deeper_problem}",
                "The user may unconsciously resist solutions.",
            ],
        ))

    content = (
        "TELEOLOGY ANALYSIS\n"
        f"Hidden utility: {utility}\n"
        f"Utility confidence: {utility_confidence:.2f}\n"
        f"Telos trajectory: {telos}\n"
        f"Problem as solution: {is_function}\n"
        f"Deeper problem: {deeper_problem}\n"
        f"Trajectory divergence from physics: {divergence}\n\n"
        f"PURPOSE: {purpose}\n"
    )

    perspective = Perspective(
        framework=FrameworkID.TELEOLOGY,
        domain=Domain.PHILOSOPHY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )

    return perspective, result


def _search_hidden_utility(
    problem: Problem,
    ontology: OntologyResult,
    epistemology: EpistemologyResult,
    phenomenology: PhenomenologyResult,
) -> tuple[str, float]:
    """What does the user GAIN by having this problem remain unsolved?"""
    utilities = []

    # Check for identity preservation — the problem defines who they are
    high_magnitude_stated = [
        v for v in problem.variables
        if v.is_user_stated and v.magnitude > 0.7
    ]
    if len(high_magnitude_stated) >= 2:
        utilities.append(
            "identity preservation — the problem has become part of how the user "
            "defines themselves or their situation. Solving it means losing that identity."
        )

    # Check for avoidance — the problem prevents facing something worse
    if phenomenology.horizon_invisible:
        utilities.append(
            "avoidance — staying focused on this problem prevents the user "
            "from seeing the invisible elements in their horizon. "
            "The problem is a shield."
        )

    # Check for excuse — the problem justifies inaction on something else
    assumptions = [
        e for e in epistemology.epistemic_map
        if e.classification == "assumption"
    ]
    if assumptions:
        utilities.append(
            "excuse — unexamined assumptions allow the user to say "
            "'I can't because of X' without testing whether X is actually true."
        )

    if utilities:
        combined = "; ".join(utilities)
        confidence = min(0.4 + len(utilities) * 0.15, 0.85)
        return combined, confidence

    return "No clear hidden utility detected.", 0.2


def _map_telos(problem: Problem, dialectics: PhilosophicalDialecticsResult) -> str:
    """Where does the problem WANT to go?"""
    if dialectics.synthesis_is_variable_d:
        return (
            f"The problem's telos points toward the dialectical synthesis: "
            f"{dialectics.synthesis[:100]}. The situation is trying to resolve "
            "itself through the hidden variable."
        )

    return (
        "The problem's trajectory, if nothing changes, is toward continued tension. "
        "Without resolution of the core dialectical conflict, the situation "
        "will oscillate between thesis and antithesis indefinitely."
    )


def _check_function_as_solution(
    problem: Problem, ontology: OntologyResult, utility: str
) -> tuple[bool, str]:
    """Is this problem actually functioning as a solution to a deeper problem?"""
    # If hidden utility exists and involves avoidance, the problem IS a solution
    if "avoidance" in utility.lower():
        return True, (
            "The deeper problem is what the user would have to face "
            "if this surface problem were solved. The current problem "
            "serves as a protective barrier."
        )

    if "identity" in utility.lower():
        return True, (
            "The deeper problem is an identity crisis — who is the user "
            "without this problem? The surface problem provides structure "
            "and meaning the user doesn't know how to replace."
        )

    if "excuse" in utility.lower():
        return True, (
            "The deeper problem is the action the user is avoiding. "
            "The surface problem provides a socially acceptable reason "
            "for inaction."
        )

    return False, "No deeper problem detected — the surface problem appears to be the actual problem."


def _compare_trajectories(
    physics_trajectory: list[Variable] | None, telos: str
) -> str:
    """Compare physics trajectory with teleological trajectory."""
    if not physics_trajectory:
        return "No physics trajectory available for comparison."

    neg_trajectory = any(
        v.direction == Direction.NEGATIVE for v in physics_trajectory
    )

    if neg_trajectory and "resolve" in telos.lower():
        return (
            "DIVERGENCE: Physics says the situation is worsening, "
            "but teleology says the problem is trying to resolve itself. "
            "This divergence suggests a hidden force working toward resolution "
            "that physics hasn't measured — or that the 'worsening' IS the resolution "
            "path (things sometimes get worse before they get better)."
        )

    return "Physics trajectory and teleological trajectory are roughly aligned."


def _build_purpose_statement(
    utility: str, is_function: bool, deeper_problem: str,
    confidence: float,
) -> str:
    """Build the final purpose statement."""
    if is_function:
        return (
            f"This problem persists because it SERVES a purpose: {utility}. "
            f"It functions as a solution to a deeper problem: {deeper_problem}. "
            f"(confidence: {confidence:.0%})"
        )

    if confidence > 0.4:
        return (
            f"This problem persists partly because: {utility}. "
            f"(confidence: {confidence:.0%})"
        )

    return (
        "No clear hidden purpose detected. The problem may persist "
        "due to structural forces rather than hidden utility."
    )
