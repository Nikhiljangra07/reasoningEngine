"""
Psychology Domain (Water) — Isolated Island Module.

The human layer. Flows into every crack. Finds the lowest point.
Operates in darkness, in the deep subconscious.

Explains WHY the physics readings are what they are. Treats the mind
as a physics system: bias is a drag coefficient, dissonance is
potential energy, motivated reasoning is anomalous motion.

Module 1 (Detection): Dual Process, Cognitive Dissonance, Motivated Reasoning
Module 2 (Integration): Dialectical Thinking, Metacognition

BRIDGE CONTRACT:
  input:  DomainInput (problem + upstream Physics/Math outputs via bridge)
  output: DomainOutput (system classifications, dissonance map, bias assessment, synthesis, delivery mode)
  challenge_input:  ChallengeInput (Chemistry output, per Ke cycle: Water checks Fire)
  challenge_output: ChallengeOutput (ethical concerns, bonding questioned, human impact flags)

ISOLATION: This module imports ONLY from src.core.types and its own module files.
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
from src.domains.psychology.mind_analysis import (
    run_dual_process,
    run_cognitive_dissonance,
    run_motivated_reasoning,
)
from src.domains.psychology.integration import (
    run_dialectical_thinking,
    run_metacognition,
)


# ---------------------------------------------------------------------------
# Sheng Cycle Entry — standard domain execution
# ---------------------------------------------------------------------------

def run_psychology(domain_input: DomainInput) -> DomainOutput:
    """
    Execute the full Psychology domain analysis.

    Accepts a DomainInput (bridge contract).
    Returns a DomainOutput (bridge contract).

    Receives Physics output (and optionally Math output) via the bridge.
    Uses Physics Phase 2 findings to cross-reference bias detection.
    """
    problem = domain_input.problem
    upstream = domain_input.upstream_outputs

    # Extract physics findings from bridge (if available)
    physics_contradictions = _extract_physics_contradictions(upstream)
    physics_anomalies = _extract_physics_anomalies(upstream)

    # Collect root causes from upstream for dialectical synthesis
    upstream_root_causes = _extract_upstream_root_causes(upstream)

    # ---------------------------------------------------------------
    # MODULE 1: Mind Analysis (Detection)
    # ---------------------------------------------------------------

    # Concept 1: Dual Process Theory
    dual_process_perspective, classifications = run_dual_process(problem)

    # Concept 2: Cognitive Dissonance
    dissonance_perspective, dissonance_pairs = run_cognitive_dissonance(
        problem, physics_contradictions
    )

    # Concept 3: Motivated Reasoning
    motivation_perspective, motivation_assessment = run_motivated_reasoning(
        problem, physics_anomalies
    )

    # ---------------------------------------------------------------
    # MODULE 2: Integration (Synthesis + Self-Awareness)
    # ---------------------------------------------------------------

    # Concept 4: Dialectical Thinking
    # Build upstream_outputs dict with string keys for the integration module
    upstream_dict = {d.value: o for d, o in upstream.items()} if upstream else None
    dialectical_perspective, syntheses = run_dialectical_thinking(
        problem, upstream_root_causes, upstream_dict
    )

    # Concept 5: Metacognition
    all_findings = []
    for p in [dual_process_perspective, dissonance_perspective, motivation_perspective]:
        all_findings.extend(p.variables_found)
    metacognition_perspective, meta_assessment = run_metacognition(
        problem, all_findings
    )

    # ---------------------------------------------------------------
    # Assemble domain output
    # ---------------------------------------------------------------
    all_perspectives = [
        dual_process_perspective,
        dissonance_perspective,
        motivation_perspective,
        dialectical_perspective,
        metacognition_perspective,
    ]

    # Collect all root causes from dissonance gap analysis
    root_causes = []
    for pair in dissonance_pairs:
        if pair.tension_score > 0.6:
            root_causes.append(RootCause(
                variable=Variable(
                    name=f"dissonance_root_{pair.var_a}_{pair.var_b}",
                    description=pair.variable_d_candidate,
                    magnitude=pair.tension_score,
                    direction=Direction.NEUTRAL,
                    confidence=pair.tension_score * 0.8,
                    source_framework=FrameworkID.COGNITIVE_DISSONANCE,
                    is_hidden=True,
                    is_user_stated=False,
                    evidence=[pair.gap_description],
                ),
                evidence_chain=[
                    pair.gap_description,
                    f"Resolution strategy: {pair.resolution_strategy}",
                    pair.variable_d_candidate,
                ],
                bias_that_hid_it=f"Dissonance resolution via {pair.resolution_strategy}",
                confidence=pair.tension_score * 0.8,
                frameworks_that_agree=[FrameworkID.COGNITIVE_DISSONANCE],
            ))

    raw_parts = [
        f"Dual Process: {sum(1 for c in classifications if c.system == 'S1')} S1, "
        f"{sum(1 for c in classifications if c.system == 'S2')} S2, "
        f"{sum(1 for c in classifications if c.flag)} rationalization flags",
        f"Cognitive Dissonance: {len(dissonance_pairs)} conflicting pairs",
        f"Motivated Reasoning: bias score {motivation_assessment.directional_bias_score:.2f}",
        f"Dialectical Synthesis: {len(syntheses)} syntheses generated",
        f"Metacognition: {meta_assessment.metacognition_score:.2f} → {meta_assessment.recommended_delivery_mode}",
    ]

    return DomainOutput(
        domain=Domain.PSYCHOLOGY,
        perspectives=all_perspectives,
        root_causes=root_causes,
        consequences=[],  # Psychology finds roots and biases, not consequences
        causal_loops=[],
        game_state=None,
        raw_analysis="\n".join(raw_parts),
    )


# ---------------------------------------------------------------------------
# Ke Cycle Entry — challenge another domain's output
# ---------------------------------------------------------------------------

def challenge(challenge_input: ChallengeInput) -> ChallengeOutput:
    """
    Psychology challenges another domain's output (Ke cycle).

    Per Wu Xing: Water extinguishes Fire — Psychology checks Chemistry.
    "You bonded these brilliantly but SHOULD you have?"

    Psychology scrutinizes by checking:
    - Ethical concerns: does the bonding serve the HUMAN or just elegance?
    - Bonding questioned: are the bonded elements compatible from a human perspective?
    - Should vs. Could: just because concepts CAN bond doesn't mean they SHOULD
    - Human impact: what does this bonding mean for the actual person?
    """
    target = challenge_input.target_output
    contradictions = []
    unsupported = []
    confidence_adjustments: dict[str, float] = {}
    flags = []

    # Check each perspective for human impact
    for perspective in target.perspectives:
        for var in perspective.variables_found:
            # Check 1: Hidden variables without human context
            if var.is_hidden and var.confidence > 0.6:
                flags.append(
                    f"'{var.name}' is hidden and high-confidence ({var.confidence:.2f}). "
                    "Psychology asks: what is the HUMAN cost of surfacing this? "
                    "Is the user ready to receive this finding?"
                )

            # Check 2: High magnitude claims about the user's mental state
            # without psychological grounding
            if (var.magnitude > 0.7
                    and var.source_framework not in (
                        FrameworkID.DUAL_PROCESS,
                        FrameworkID.COGNITIVE_DISSONANCE,
                        FrameworkID.MOTIVATED_REASONING,
                        FrameworkID.DIALECTICAL_THINKING,
                        FrameworkID.METACOGNITION,
                    )):
                # Non-psychology domain making strong psychological claims
                desc_lower = var.description.lower()
                psych_keywords = [
                    "bias", "emotion", "feel", "believe", "fear",
                    "deny", "avoid", "rationali", "dissonance",
                ]
                if any(kw in desc_lower for kw in psych_keywords):
                    unsupported.append(
                        f"'{var.name}' makes psychological claims "
                        f"('{var.description[:60]}...') from a non-psychology framework "
                        f"({var.source_framework.value}). "
                        "Psychology questions whether this assessment has "
                        "sufficient understanding of the human dynamics."
                    )
                    confidence_adjustments[var.name] = var.confidence * 0.8

    # Check root causes for human sensitivity
    for rc in target.root_causes:
        if rc.bias_that_hid_it and rc.confidence > 0.7:
            flags.append(
                f"Root cause '{rc.variable.name}' claims a specific bias "
                f"('{rc.bias_that_hid_it}'). Psychology asks: is this bias "
                "diagnosis accurate, or is it another domain projecting "
                "psychological language onto a structural finding?"
            )

    total_vars = sum(len(p.variables_found) for p in target.perspectives) or 1
    issue_count = len(contradictions) + len(unsupported) + len(flags)
    scrutiny_score = min(issue_count / total_vars, 1.0)

    return ChallengeOutput(
        challenger_domain=Domain.PSYCHOLOGY,
        target_domain=challenge_input.target_domain,
        contradictions=contradictions,
        unsupported_claims=unsupported,
        confidence_adjustments=confidence_adjustments,
        scrutiny_score=scrutiny_score,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Bridge helpers — extract findings from upstream WITHOUT importing domains
# ---------------------------------------------------------------------------

def _extract_physics_contradictions(
    upstream: dict[Domain, DomainOutput],
) -> list[Variable] | None:
    """Extract contradiction findings from Physics output (via bridge)."""
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
            contradictions.extend([
                v for v in p.variables_found if v.is_hidden
            ])
    return contradictions if contradictions else None


def _extract_physics_anomalies(
    upstream: dict[Domain, DomainOutput],
) -> list[Variable] | None:
    """Extract anomalous motion findings from Physics output (via bridge)."""
    physics = upstream.get(Domain.PHYSICS)
    if not physics:
        return None

    anomalies = []
    for p in physics.perspectives:
        if p.framework == FrameworkID.ANOMALOUS_MOTION:
            anomalies.extend(p.variables_found)
    return anomalies if anomalies else None


def _extract_upstream_root_causes(
    upstream: dict[Domain, DomainOutput],
) -> list[RootCause]:
    """Extract root causes from all upstream domain outputs (via bridge)."""
    root_causes = []
    for output in upstream.values():
        root_causes.extend(output.root_causes)
    return root_causes
