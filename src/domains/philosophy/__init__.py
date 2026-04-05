"""
Philosophy Domain (Wood) — Isolated Island Module.

The energy of expansion, growth, and upward reach. A tree branches in
every direction, seeking light. Philosophy seeks to expand the boundaries
of understanding and asks the questions that other domains assume are settled.

5 concepts in logical pipeline sequence:
  Ontology → Epistemology → Phenomenology → Dialectics → Teleology
  (define reality → audit knowledge → map perception → find conflict → trace trajectory)

BRIDGE CONTRACT:
  input:  DomainInput (problem + all upstream domain outputs)
  output: DomainOutput (ontological core, epistemic map, horizon, dialectical synthesis, teleological purpose)
  challenge_input:  ChallengeInput (Physics output, per Ke cycle: Wood checks Earth)
  challenge_output: ChallengeOutput (assumptions in physics model, unexamined premises, alternative framings)

ISOLATION: This module imports ONLY from src.core.types and its own files.
           It does NOT import from any other domain.
"""

from __future__ import annotations

from src.core.types import (
    ChallengeInput,
    ChallengeOutput,
    Direction,
    Domain,
    DomainInput,
    DomainOutput,
    FrameworkID,
    Perspective,
    RootCause,
    Variable,
)
from src.domains.philosophy.epistemic_pipeline import (
    run_ontology,
    run_epistemology,
    run_phenomenology,
    run_dialectics,
    run_teleology,
)


# ---------------------------------------------------------------------------
# Sheng Cycle Entry — standard domain execution
# ---------------------------------------------------------------------------

def run_philosophy(domain_input: DomainInput) -> DomainOutput:
    """
    Execute the full Philosophy domain analysis.

    Accepts a DomainInput (bridge contract).
    Returns a DomainOutput (bridge contract).

    The 5 concepts run as a pipeline — each feeds the next:
    Ontology → Epistemology → Phenomenology → Dialectics → Teleology
    """
    problem = domain_input.problem
    upstream = domain_input.upstream_outputs

    # Collect all upstream domain outputs for Ontology (it needs everything)
    all_upstream_outputs = list(upstream.values())

    # Extract specific upstream findings via bridge (no direct imports)
    physics_contradictions = _extract_physics_contradictions(upstream)
    physics_trajectory = _extract_physics_trajectory(upstream)
    metacognition_score = _extract_metacognition_score(upstream)
    motivated_reasoning = _extract_motivated_reasoning(upstream)

    # ---------------------------------------------------------------
    # PIPELINE: each concept feeds the next
    # ---------------------------------------------------------------

    # Concept 1: Ontology ��� define reality
    ontology_perspective, ontology_result = run_ontology(
        problem, all_upstream_outputs
    )

    # Concept 2: Epistemology — audit knowledge (uses Ontology output)
    epistemology_perspective, epistemology_result = run_epistemology(
        problem, ontology_result
    )

    # Concept 3: Phenomenology — map perception (uses Ontology + Epistemology)
    phenomenology_perspective, phenomenology_result = run_phenomenology(
        problem, ontology_result, epistemology_result, metacognition_score
    )

    # Concept 4: Dialectics — find conflict (uses Ontology + Epistemology + Phenomenology)
    dialectics_perspective, dialectics_result = run_dialectics(
        problem, ontology_result, epistemology_result, phenomenology_result,
        physics_contradictions
    )

    # Concept 5: Teleology — trace trajectory (uses all previous)
    teleology_perspective, teleology_result = run_teleology(
        problem, ontology_result, epistemology_result,
        phenomenology_result, dialectics_result,
        physics_trajectory, motivated_reasoning
    )

    # ---------------------------------------------------------------
    # Assemble domain output
    # ---------------------------------------------------------------
    all_perspectives = [
        ontology_perspective,
        epistemology_perspective,
        phenomenology_perspective,
        dialectics_perspective,
        teleology_perspective,
    ]

    # Collect root causes
    root_causes = []

    # Dialectical Variable D
    if dialectics_result.synthesis_is_variable_d:
        root_causes.append(RootCause(
            variable=Variable(
                name="philosophical_variable_d",
                description=dialectics_result.synthesis,
                magnitude=dialectics_result.stability_score,
                direction=Direction.NEUTRAL,
                confidence=dialectics_result.stability_score,
                source_framework=FrameworkID.DIALECTICS,
                is_hidden=True,
                is_user_stated=False,
                evidence=[
                    f"Thesis: {dialectics_result.thesis[:80]}",
                    f"Antithesis: {dialectics_result.antithesis[:80]}",
                    f"Tension: {dialectics_result.tension_point[:80]}",
                ],
            ),
            evidence_chain=[
                dialectics_result.thesis,
                dialectics_result.antithesis,
                dialectics_result.tension_point,
                dialectics_result.synthesis,
            ],
            bias_that_hid_it="Dialectical blindness — thesis and antithesis each block view of the other",
            confidence=dialectics_result.stability_score,
            frameworks_that_agree=[FrameworkID.DIALECTICS],
        ))

    # Teleological hidden purpose
    if teleology_result.hidden_utility_confidence > 0.5:
        root_causes.append(RootCause(
            variable=Variable(
                name="teleological_purpose",
                description=teleology_result.purpose_statement,
                magnitude=teleology_result.hidden_utility_confidence,
                direction=Direction.CIRCULAR,
                confidence=teleology_result.hidden_utility_confidence,
                source_framework=FrameworkID.TELEOLOGY,
                is_hidden=True,
                is_user_stated=False,
                evidence=[
                    f"Hidden utility: {teleology_result.hidden_utility[:80]}",
                    f"Function as solution: {teleology_result.function_as_solution}",
                    f"Deeper problem: {teleology_result.deeper_problem_hypothesis[:80]}",
                ],
            ),
            evidence_chain=[
                teleology_result.hidden_utility,
                teleology_result.purpose_statement,
                teleology_result.deeper_problem_hypothesis,
            ],
            bias_that_hid_it="Teleological blindness — the problem's hidden utility prevents seeing why it persists",
            confidence=teleology_result.hidden_utility_confidence,
            frameworks_that_agree=[FrameworkID.TELEOLOGY],
        ))

    raw_parts = [
        f"Ontology: {len(ontology_result.essential_variables)} essential, {len(ontology_result.accidental_variables)} accidental, {ontology_result.variables_reclassified} reclassified",
        f"Epistemology: {sum(1 for e in epistemology_result.epistemic_map if e.classification == 'fact')} facts, {sum(1 for e in epistemology_result.epistemic_map if e.classification == 'assumption')} assumptions, {len(epistemology_result.false_prior_candidates)} false priors",
        f"Phenomenology: frame='{phenomenology_result.experiential_frame}', {len(phenomenology_result.horizon_invisible)} invisible elements",
        f"Dialectics: synthesis_is_variable_d={dialectics_result.synthesis_is_variable_d}, stability={dialectics_result.stability_score:.2f}",
        f"Teleology: hidden_utility_confidence={teleology_result.hidden_utility_confidence:.2f}, function_as_solution={teleology_result.function_as_solution}",
    ]

    return DomainOutput(
        domain=Domain.PHILOSOPHY,
        perspectives=all_perspectives,
        root_causes=root_causes,
        consequences=[],  # Philosophy finds meaning and purpose, not timeline consequences
        causal_loops=[],
        game_state=None,
        raw_analysis="\n".join(raw_parts),
    )


# ---------------------------------------------------------------------------
# Ke Cycle Entry — challenge another domain's output
# ---------------------------------------------------------------------------

def challenge(challenge_input: ChallengeInput) -> ChallengeOutput:
    """
    Philosophy challenges another domain's output (Ke cycle).

    Per Wu Xing: Wood penetrates Earth — Philosophy checks Physics.
    "Have you questioned the ground you're standing on?"

    Philosophy scrutinizes by checking:
    - Unexamined premises: what assumptions does the physics model make?
    - Alternative framings: could the same data support a different conclusion?
    - Settled facts questioned: which "facts" are actually beliefs or assumptions?
    - Ontological mismatch: is the analysis addressing WHAT the problem is, or just HOW it appears?
    """
    target = challenge_input.target_output
    contradictions = []
    unsupported = []
    confidence_adjustments: dict[str, float] = {}
    flags = []

    for perspective in target.perspectives:
        for var in perspective.variables_found:
            # Check 1: Unexamined premises — high confidence without questioning
            if var.confidence > 0.8 and not var.is_hidden:
                flags.append(
                    f"'{var.name}' is held at {var.confidence:.0%} confidence "
                    "without being questioned. Philosophy asks: what would have to "
                    "be true for this to be false? Has this premise been examined?"
                )

            # Check 2: Variables that might be accidental, not essential
            if var.is_user_stated and var.magnitude < 0.5 and var.confidence > 0.6:
                unsupported.append(
                    f"'{var.name}' may be an accidental property, not essential. "
                    "The analysis may be building on surface features rather than "
                    "the problem's actual nature."
                )

            # Check 3: Reframing potential — could the direction be wrong?
            if var.direction in (Direction.POSITIVE, Direction.NEGATIVE) and var.confidence < 0.7:
                flags.append(
                    f"'{var.name}' is classified as {var.direction.value} "
                    f"but with only {var.confidence:.0%} confidence. "
                    "An alternative framing might reverse this direction entirely."
                )

    # Check root causes for philosophical depth
    for rc in target.root_causes:
        if rc.confidence > 0.7 and not rc.bias_that_hid_it:
            unsupported.append(
                f"Root cause '{rc.variable.name}' has high confidence ({rc.confidence:.0%}) "
                "but no bias identified. Philosophy asks: if there's no bias hiding it, "
                "how was it hidden in the first place? Something is unexamined."
            )

        # Teleological question: why does this root cause persist?
        flags.append(
            f"Teleological question for '{rc.variable.name}': "
            "What PURPOSE does this root cause serve? Why hasn't the user already "
            "resolved it if it's this clear? What hidden utility does maintaining "
            "this problem provide?"
        )

    total_vars = sum(len(p.variables_found) for p in target.perspectives) or 1
    issue_count = len(contradictions) + len(unsupported) + len(flags)
    scrutiny_score = min(issue_count / total_vars, 1.0)

    return ChallengeOutput(
        challenger_domain=Domain.PHILOSOPHY,
        target_domain=challenge_input.target_domain,
        contradictions=contradictions,
        unsupported_claims=unsupported,
        confidence_adjustments=confidence_adjustments,
        scrutiny_score=scrutiny_score,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Bridge helpers — extract upstream findings WITHOUT importing domains
# ---------------------------------------------------------------------------

def _extract_physics_contradictions(
    upstream: dict[Domain, DomainOutput],
) -> list[Variable] | None:
    """Extract contradiction findings from Physics output via bridge."""
    physics = upstream.get(Domain.PHYSICS)
    if not physics:
        return None

    contradictions = []
    for p in physics.perspectives:
        if p.framework in (
            FrameworkID.ANOMALOUS_MOTION,
            FrameworkID.SOCRATIC_SQUEEZE,
            FrameworkID.REDUCTIO,
        ):
            contradictions.extend([v for v in p.variables_found if v.is_hidden])
    return contradictions if contradictions else None


def _extract_physics_trajectory(
    upstream: dict[Domain, DomainOutput],
) -> list[Variable] | None:
    """Extract trajectory variables from Physics output via bridge."""
    physics = upstream.get(Domain.PHYSICS)
    if not physics:
        return None

    trajectory = []
    for p in physics.perspectives:
        if p.framework == FrameworkID.TRAJECTORY_MOMENTUM:
            trajectory.extend(p.variables_found)
    return trajectory if trajectory else None


def _extract_metacognition_score(
    upstream: dict[Domain, DomainOutput],
) -> float | None:
    """Extract metacognition score from Psychology output via bridge."""
    psychology = upstream.get(Domain.PSYCHOLOGY)
    if not psychology:
        return None

    for p in psychology.perspectives:
        if p.framework == FrameworkID.METACOGNITION:
            for v in p.variables_found:
                if v.name == "metacognition_level":
                    return v.magnitude
    return None


def _extract_motivated_reasoning(
    upstream: dict[Domain, DomainOutput],
) -> dict | None:
    """Extract motivated reasoning assessment from Psychology via bridge."""
    psychology = upstream.get(Domain.PSYCHOLOGY)
    if not psychology:
        return None

    for p in psychology.perspectives:
        if p.framework == FrameworkID.MOTIVATED_REASONING:
            for v in p.variables_found:
                if "motivated_reasoning" in v.name:
                    return {"bias_score": v.magnitude, "description": v.description}
    return None
