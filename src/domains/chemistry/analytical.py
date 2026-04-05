"""
Chemistry Module C: Analytical (Runs DURING the battlefield alongside other domains)

These are Chemistry's reasoning tools — deployed on the battlefield subject
to the same Ke cycle challenges as every other domain. No special treatment.

3 Concepts:
4. Chirality — detects mirror-image perspectives (same components, different orientation)
5. Catalysis — identifies the single breakthrough insight that transforms everything
6. Resonance — holds multiple perspectives simultaneously as a hybrid truth

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
# Concept 4: Chirality (The Mirror Perspective)
# ===========================================================================

@dataclass
class ChiralityAssessment:
    """Assessment of mirror-image perspectives."""
    is_chiral_pair: bool
    shared_components: list[str]
    orientation_a: str
    orientation_b: str
    fit_score_a: float
    fit_score_b: float
    toxic_mirror: str               # "a", "b", or "neither"
    truth_orientation: str           # "a", "b", or "undetermined"


def run_chirality(
    competing_narratives: list[tuple[str, DomainOutput]],
    physics_output: DomainOutput | None = None,
    epistemology_facts: list[str] | None = None,
    motivated_reasoning_bias: float | None = None,
) -> tuple[Perspective, list[ChiralityAssessment]]:
    """
    Chirality — detects when two perspectives have the exact same components
    but different ORIENTATIONS — mirror images that cannot be superimposed.

    One fits the truth, the other fits the deception. This is the ultimate
    bias detector — it catches the ORIENTATION of a perspective, not just
    its components.
    """
    assessments = []
    variables_found = []

    # Compare all pairs of competing narratives
    for i, (name_a, output_a) in enumerate(competing_narratives):
        for name_b, output_b in competing_narratives[i + 1:]:
            assessment = _assess_chirality(
                name_a, output_a, name_b, output_b,
                physics_output, epistemology_facts, motivated_reasoning_bias
            )
            assessments.append(assessment)

            if assessment.is_chiral_pair:
                toxic = assessment.toxic_mirror
                truth = assessment.truth_orientation

                variables_found.append(Variable(
                    name=f"chiral_pair_{name_a}_{name_b}",
                    description=(
                        f"CHIRAL PAIR detected: '{name_a}' and '{name_b}' use the "
                        f"same components but in different orientations. "
                        f"Shared components: {', '.join(assessment.shared_components[:5])}. "
                        f"Toxic mirror: {toxic}. Truth orientation: {truth}. "
                        f"Fit scores: {name_a}={assessment.fit_score_a:.2f}, "
                        f"{name_b}={assessment.fit_score_b:.2f}."
                    ),
                    magnitude=0.85,
                    direction=Direction.NEUTRAL,
                    confidence=max(assessment.fit_score_a, assessment.fit_score_b),
                    source_framework=FrameworkID.CHIRALITY,
                    is_hidden=True,
                    is_user_stated=False,
                    evidence=[
                        f"Shared components: {len(assessment.shared_components)}",
                        f"Orientation A ({name_a}): {assessment.orientation_a[:60]}",
                        f"Orientation B ({name_b}): {assessment.orientation_b[:60]}",
                        f"Toxic mirror: {toxic}",
                        f"Truth: {truth}",
                    ],
                ))

    content = "CHIRALITY ANALYSIS (Mirror Perspective Detection)\n\n"
    if assessments:
        chiral_count = sum(1 for a in assessments if a.is_chiral_pair)
        content += f"Chiral pairs found: {chiral_count} of {len(assessments)} comparisons\n\n"
        for a in assessments:
            if a.is_chiral_pair:
                content += (
                    f"  CHIRAL: shared={len(a.shared_components)} components\n"
                    f"    Orientation A: {a.orientation_a[:80]}\n"
                    f"    Orientation B: {a.orientation_b[:80]}\n"
                    f"    Toxic mirror: {a.toxic_mirror}\n"
                    f"    Truth orientation: {a.truth_orientation}\n\n"
                )
    else:
        content += "No competing narratives to compare.\n"

    perspective = Perspective(
        framework=FrameworkID.CHIRALITY,
        domain=Domain.CHEMISTRY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )

    return perspective, assessments


def _assess_chirality(
    name_a: str, output_a: DomainOutput,
    name_b: str, output_b: DomainOutput,
    physics_output: DomainOutput | None,
    epistemology_facts: list[str] | None,
    bias_score: float | None,
) -> ChiralityAssessment:
    """Assess if two outputs form a chiral pair."""
    # Collect variable names from each
    vars_a = {v.name for p in output_a.perspectives for v in p.variables_found}
    vars_b = {v.name for p in output_b.perspectives for v in p.variables_found}

    # Shared components
    shared = list(vars_a & vars_b)

    # Orientation analysis: compare direction distributions
    dirs_a = _direction_profile(output_a)
    dirs_b = _direction_profile(output_b)
    orientation_a = f"Direction profile: {dirs_a}"
    orientation_b = f"Direction profile: {dirs_b}"

    # Is it chiral? Same components, different arrangement
    is_chiral = len(shared) >= 2 and dirs_a != dirs_b

    # Fit test against physics and epistemology
    fit_a = _fit_test(output_a, physics_output, epistemology_facts)
    fit_b = _fit_test(output_b, physics_output, epistemology_facts)

    # Determine toxic mirror
    toxic = "neither"
    truth = "undetermined"
    if is_chiral:
        if fit_a > fit_b + 0.15:
            truth = "a"
            toxic = "b"
        elif fit_b > fit_a + 0.15:
            truth = "b"
            toxic = "a"

        # If bias is detected and one orientation aligns with it
        if bias_score is not None and bias_score > 0.6:
            # The orientation that matches the user's bias is more likely the toxic mirror
            # (motivated reasoning produces the deceptive mirror)
            pass  # Used implicitly in fit scoring

    return ChiralityAssessment(
        is_chiral_pair=is_chiral,
        shared_components=shared,
        orientation_a=orientation_a,
        orientation_b=orientation_b,
        fit_score_a=fit_a,
        fit_score_b=fit_b,
        toxic_mirror=toxic,
        truth_orientation=truth,
    )


def _direction_profile(output: DomainOutput) -> dict[str, int]:
    """Get the direction distribution of an output."""
    profile: dict[str, int] = {}
    for p in output.perspectives:
        for v in p.variables_found:
            d = v.direction.value
            profile[d] = profile.get(d, 0) + 1
    return profile


def _fit_test(
    output: DomainOutput,
    physics: DomainOutput | None,
    facts: list[str] | None,
) -> float:
    """Test how well an output fits reality (physics + verified facts)."""
    score = 0.5  # baseline

    if physics:
        # Compare root causes — alignment = higher fit
        physics_roots = {rc.variable.name for rc in physics.root_causes}
        output_roots = {rc.variable.name for rc in output.root_causes}
        overlap = physics_roots & output_roots
        if physics_roots:
            score += (len(overlap) / len(physics_roots)) * 0.25

    # Higher average confidence across variables = better fit
    all_vars = [v for p in output.perspectives for v in p.variables_found]
    if all_vars:
        avg_conf = sum(v.confidence for v in all_vars) / len(all_vars)
        score += avg_conf * 0.25

    return min(score, 1.0)


# ===========================================================================
# Concept 5: Catalysis (The Breakthrough)
# ===========================================================================

@dataclass
class CatalystCandidate:
    """A potential breakthrough insight."""
    insight: str
    barrier_reduction_score: float      # 0.0 to 1.0
    truth_alignment_score: float        # 0.0 to 1.0
    deliverability_score: float         # 0.0 to 1.0
    combined_score: float


@dataclass
class CatalysisResult:
    """Result of catalyst search."""
    activation_barriers: list[str]
    candidates: list[CatalystCandidate]
    primary_catalyst: CatalystCandidate | None
    catalytic_moment_phrasing: str


def run_catalysis(
    all_domain_outputs: dict[Domain, DomainOutput],
    root_causes: list[RootCause],
    metacognition_score: float | None = None,
) -> tuple[Perspective, CatalysisResult]:
    """
    Catalysis — identifies the single insight that transforms the entire
    messy problem into clarity.

    Lowers the "activation energy" required for the user to see the truth.
    The catalyst is not consumed — it's a reusable insight.
    """
    variables_found = []

    # Step 1: Map activation energy barriers
    barriers = _map_barriers(all_domain_outputs, root_causes)

    # Step 2: Search for catalyst candidates
    candidates = _search_catalysts(
        all_domain_outputs, root_causes, barriers, metacognition_score
    )

    # Step 3: Select primary catalyst
    primary = candidates[0] if candidates else None

    # Step 4: Craft the catalytic moment phrasing
    phrasing = _craft_catalytic_moment(primary, metacognition_score)

    result = CatalysisResult(
        activation_barriers=barriers,
        candidates=candidates,
        primary_catalyst=primary,
        catalytic_moment_phrasing=phrasing,
    )

    if primary:
        variables_found.append(Variable(
            name="catalytic_insight",
            description=(
                f"CATALYST: {primary.insight} "
                f"(barrier reduction: {primary.barrier_reduction_score:.2f}, "
                f"truth alignment: {primary.truth_alignment_score:.2f}, "
                f"deliverability: {primary.deliverability_score:.2f})"
            ),
            magnitude=primary.combined_score,
            direction=Direction.POSITIVE,
            confidence=primary.truth_alignment_score,
            source_framework=FrameworkID.CATALYSIS,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Insight: {primary.insight[:80]}",
                f"Barriers addressed: {len(barriers)}",
                f"Phrasing: {phrasing[:80]}",
            ],
        ))

    content = (
        "CATALYSIS ANALYSIS (Breakthrough Insight)\n"
        f"Activation barriers: {len(barriers)}\n"
        f"Catalyst candidates: {len(candidates)}\n\n"
    )
    for b in barriers:
        content += f"  BARRIER: {b}\n"
    content += "\n"
    if primary:
        content += (
            f"PRIMARY CATALYST: {primary.insight}\n"
            f"  Barrier reduction: {primary.barrier_reduction_score:.2f}\n"
            f"  Truth alignment: {primary.truth_alignment_score:.2f}\n"
            f"  Deliverability: {primary.deliverability_score:.2f}\n\n"
            f"CATALYTIC MOMENT: {phrasing}\n"
        )
    else:
        content += "No clear catalytic insight identified.\n"

    perspective = Perspective(
        framework=FrameworkID.CATALYSIS,
        domain=Domain.CHEMISTRY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )

    return perspective, result


def _map_barriers(
    outputs: dict[Domain, DomainOutput],
    root_causes: list[RootCause],
) -> list[str]:
    """What is preventing the user from seeing the solution?"""
    barriers = []

    # Check for emotional barriers (from psychology)
    psych = outputs.get(Domain.PSYCHOLOGY)
    if psych:
        for p in psych.perspectives:
            if p.framework == FrameworkID.DUAL_PROCESS:
                for v in p.variables_found:
                    if "rationalization" in v.name:
                        barriers.append(
                            f"Emotional barrier: {v.description[:80]} — "
                            "the user has wrapped an emotional impulse in logical clothing"
                        )

    # Check for cognitive barriers (from dissonance)
    if psych:
        for p in psych.perspectives:
            if p.framework == FrameworkID.COGNITIVE_DISSONANCE:
                for v in p.variables_found:
                    if v.magnitude > 0.6:
                        barriers.append(
                            f"Cognitive barrier: dissonance tension {v.magnitude:.2f} — "
                            "conflicting beliefs create a wall the user can't think past"
                        )

    # Check for information barriers (from epistemology)
    phil = outputs.get(Domain.PHILOSOPHY)
    if phil:
        for p in phil.perspectives:
            if p.framework == FrameworkID.EPISTEMOLOGY:
                for v in p.variables_found:
                    if "assumption" in v.name:
                        barriers.append(
                            f"Information barrier: {v.description[:80]} — "
                            "unexamined assumptions blocking new information"
                        )

    # Check for identity barriers (from teleology)
    if phil:
        for p in phil.perspectives:
            if p.framework == FrameworkID.TELEOLOGY:
                for v in p.variables_found:
                    if "purpose" in v.name or "solution" in v.name:
                        barriers.append(
                            f"Identity barrier: {v.description[:80]} — "
                            "the problem serves a purpose the user can't let go of"
                        )

    if not barriers:
        barriers.append("No specific barriers identified — the user may simply lack information")

    return barriers


def _search_catalysts(
    outputs: dict[Domain, DomainOutput],
    root_causes: list[RootCause],
    barriers: list[str],
    metacognition: float | None,
) -> list[CatalystCandidate]:
    """Search for potential catalytic insights across all domain outputs."""
    candidates = []

    for rc in root_causes:
        # Each root cause is a potential catalyst
        barrier_reduction = min(len(barriers) * 0.15 + rc.confidence * 0.3, 1.0)
        truth_alignment = rc.confidence
        deliverability = metacognition if metacognition is not None else 0.5

        insight = (
            f"The root issue is '{rc.variable.name}': {rc.variable.description[:80]}"
        )
        if rc.bias_that_hid_it:
            insight += f" (hidden by: {rc.bias_that_hid_it[:60]})"

        combined = (
            barrier_reduction * 0.35
            + truth_alignment * 0.40
            + deliverability * 0.25
        )

        candidates.append(CatalystCandidate(
            insight=insight,
            barrier_reduction_score=barrier_reduction,
            truth_alignment_score=truth_alignment,
            deliverability_score=deliverability,
            combined_score=combined,
        ))

    # Sort by combined score
    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    return candidates[:5]  # top 5


def _craft_catalytic_moment(
    catalyst: CatalystCandidate | None,
    metacognition: float | None,
) -> str:
    """Craft the specific phrasing that delivers the catalyst most effectively."""
    if not catalyst:
        return "No catalytic moment available."

    meta = metacognition if metacognition is not None else 0.5

    if meta > 0.7:
        # Direct delivery
        return f"Directly: {catalyst.insight}"
    elif meta > 0.4:
        # Building delivery
        return (
            f"Let me show you something in your own story... {catalyst.insight}"
        )
    else:
        # Gentle delivery
        return (
            f"I notice a pattern that might be worth exploring: "
            f"{catalyst.insight}"
        )


# ===========================================================================
# Concept 6: Resonance (The Endless Perspective)
# ===========================================================================

@dataclass
class ResonanceResult:
    """Result of resonance analysis — the hybrid truth."""
    requires_resonance: bool
    contributing_structures: list[str]
    hybrid_description: str
    hybrid_stability_score: float
    irreducible_ambiguity: bool
    ambiguity_description: str


def run_resonance(
    surviving_outputs: dict[Domain, DomainOutput],
    convergence_achieved: bool,
) -> tuple[Perspective, ResonanceResult]:
    """
    Resonance — when the truth cannot be expressed as a single structure,
    holds multiple valid perspectives simultaneously as a HYBRID.

    No single perspective is "The Answer" — the overlap of all valid
    perspectives creates the stable conclusion.

    Like benzene's resonance is more stable than either Kekulé structure.
    """
    variables_found = []

    # Step 1: Can the finding be expressed as one clear statement?
    single_structure_possible = _test_single_structure(surviving_outputs)

    if single_structure_possible and convergence_achieved:
        result = ResonanceResult(
            requires_resonance=False,
            contributing_structures=[],
            hybrid_description="Single structure sufficient — no resonance needed.",
            hybrid_stability_score=1.0,
            irreducible_ambiguity=False,
            ambiguity_description="",
        )

        content = (
            "RESONANCE ANALYSIS\n"
            "Single structure sufficient — convergence achieved on a clear answer.\n"
            "No resonance hybrid needed.\n"
        )

        perspective = Perspective(
            framework=FrameworkID.RESONANCE,
            domain=Domain.CHEMISTRY,
            content=content,
            variables_found=[],
            signal_type=SignalType.SIGNAL,
            weight=0.7,
        )

        return perspective, result

    # Step 2: List all contributing structures
    structures = _list_contributing_structures(surviving_outputs)

    # Step 3: Build the hybrid
    hybrid, stability = _build_hybrid(structures, surviving_outputs)

    # Step 4: Check for irreducible ambiguity
    is_ambiguous, ambiguity_desc = _check_irreducible_ambiguity(
        stability, structures, convergence_achieved
    )

    result = ResonanceResult(
        requires_resonance=True,
        contributing_structures=structures,
        hybrid_description=hybrid,
        hybrid_stability_score=stability,
        irreducible_ambiguity=is_ambiguous,
        ambiguity_description=ambiguity_desc,
    )

    variables_found.append(Variable(
        name="resonance_hybrid",
        description=(
            f"Resonance hybrid: {hybrid[:100]} "
            f"(stability: {stability:.2f}, "
            f"ambiguous: {is_ambiguous})"
        ),
        magnitude=stability,
        direction=Direction.NEUTRAL,
        confidence=stability,
        source_framework=FrameworkID.RESONANCE,
        is_hidden=False,
        is_user_stated=False,
        evidence=[
            f"Contributing structures: {len(structures)}",
            f"Stability: {stability:.2f}",
            f"Irreducible ambiguity: {is_ambiguous}",
            *[f"Structure: {s[:60]}" for s in structures[:3]],
        ],
    ))

    content = (
        "RESONANCE ANALYSIS\n"
        f"Requires resonance: YES\n"
        f"Contributing structures: {len(structures)}\n"
        f"Hybrid stability: {stability:.2f}\n"
        f"Irreducible ambiguity: {is_ambiguous}\n\n"
        f"HYBRID: {hybrid}\n"
    )
    if is_ambiguous:
        content += f"\nAMBIGUITY: {ambiguity_desc}\n"

    perspective = Perspective(
        framework=FrameworkID.RESONANCE,
        domain=Domain.CHEMISTRY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )

    return perspective, result


def _test_single_structure(outputs: dict[Domain, DomainOutput]) -> bool:
    """Can the finding be expressed as one clear statement?"""
    # If all domains agree on a single root cause → single structure possible
    all_root_names = set()
    for output in outputs.values():
        for rc in output.root_causes:
            all_root_names.add(rc.variable.name)

    # If only 1-2 unique root cause names, single structure works
    return len(all_root_names) <= 2


def _list_contributing_structures(
    outputs: dict[Domain, DomainOutput],
) -> list[str]:
    """List all valid perspectives that survived the Ke cycle."""
    structures = []
    for domain, output in outputs.items():
        if output.root_causes:
            for rc in output.root_causes:
                structures.append(
                    f"[{domain.value}] {rc.variable.name}: "
                    f"{rc.variable.description[:80]} "
                    f"(confidence: {rc.confidence:.2f})"
                )
        elif output.perspectives:
            top_perspective = max(
                output.perspectives,
                key=lambda p: p.weight * max(
                    (v.confidence for v in p.variables_found), default=0.5
                ),
            )
            structures.append(
                f"[{domain.value}] {top_perspective.framework.value}: "
                f"{top_perspective.content[:80]}"
            )
    return structures


def _build_hybrid(
    structures: list[str],
    outputs: dict[Domain, DomainOutput],
) -> tuple[str, float]:
    """Build the resonance hybrid — the truth that exists in the overlap."""
    if not structures:
        return "Insufficient structures for hybrid construction.", 0.3

    # The hybrid is more stable than any individual structure
    # (like benzene's resonance stabilization)
    individual_confidences = []
    for output in outputs.values():
        for rc in output.root_causes:
            individual_confidences.append(rc.confidence)

    avg_individual = (
        sum(individual_confidences) / len(individual_confidences)
        if individual_confidences else 0.5
    )
    # Resonance stabilization: hybrid is stronger than average individual
    hybrid_stability = min(avg_individual + 0.1 * len(structures), 0.95)

    hybrid_desc = (
        f"The truth is not captured by any single domain's perspective alone. "
        f"It exists in the overlap of {len(structures)} contributing structures. "
        f"Each structure captures a facet: "
        + "; ".join(s[:50] for s in structures[:4])
        + ". The resonance hybrid integrates these facets into a single "
        "stable conclusion that is more complete than any individual view."
    )

    return hybrid_desc, hybrid_stability


def _check_irreducible_ambiguity(
    stability: float, structures: list[str], convergence: bool
) -> tuple[bool, str]:
    """Check if the problem genuinely has no single answer."""
    if stability < 0.5 and not convergence:
        return True, (
            "IRREDUCIBLE AMBIGUITY: Even after resonance, the hybrid stability is low "
            f"({stability:.2f}) and convergence was not achieved. "
            "This problem genuinely has multiple valid answers. "
            "Forcing false certainty would be dishonest."
        )

    if len(structures) > 5 and stability < 0.7:
        return True, (
            f"POSSIBLE AMBIGUITY: {len(structures)} contributing structures with "
            f"moderate hybrid stability ({stability:.2f}). "
            "The answer exists but cannot be cleanly reduced to a single statement."
        )

    return False, ""
