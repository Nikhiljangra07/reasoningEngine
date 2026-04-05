"""
Chemistry Domain (Fire) — Isolated Island Module.

The element of transformation. The ONLY domain that serves dual function:
governance AND analytical reasoning.

Module A (Governance): Runs BEFORE the battlefield opens.
  Self-Assembly, Valence, Chemical Equilibrium / Le Chatelier's.
  Sets the formation. Decides what activates, what bonds, what's inert.

Module C (Analytical): Runs DURING the battlefield alongside other domains.
  Chirality, Catalysis, Resonance.
  Fights as an equal participant. No special treatment. Subject to Ke cycle.

BRIDGE CONTRACT:
  input (Module A): Problem variables + problem type.
  output (Module A): Formation plan + activation map + signal integrity rules.
  input (Module C): All domain outputs + competing narratives + convergence state.
  output (Module C): { chirality assessment, catalyst identification, resonance hybrid }
  challenge_input:  ChallengeInput (Mathematics output, per Ke cycle: Fire checks Metal)
  challenge_output: ChallengeOutput (rigidity flags, real-world messiness, precision-without-accuracy warnings)

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
from src.domains.chemistry.governance import (
    run_self_assembly,
    run_valence,
    run_chemical_equilibrium,
    FormationPlan,
)
from src.domains.chemistry.analytical import (
    run_chirality,
    run_catalysis,
    run_resonance,
)


# ---------------------------------------------------------------------------
# Module A: Governance — runs BEFORE the battlefield
# ---------------------------------------------------------------------------

def run_governance(domain_input: DomainInput) -> tuple[DomainOutput, FormationPlan]:
    """
    Execute Chemistry's governance function (Module A).

    This runs FIRST — before any other domain activates.
    Sets the formation: what activates, what bonds, what's inert.

    Returns BOTH the DomainOutput (for the bridge) and the FormationPlan
    (for the orchestrator to use when deploying domains).
    """
    problem = domain_input.problem

    # Concept 1: Self-Assembly — determine natural structure and activation plan
    assembly_perspective, formation_plan = run_self_assembly(problem)

    perspectives = [assembly_perspective]

    raw_parts = [
        f"Self-Assembly: template={formation_plan.organizational_template}, "
        f"clusters={len(formation_plan.structural_affinity_clusters)}, "
        f"misfits={len(formation_plan.misfit_variables)}, "
        f"active_domains={[d.value for d in formation_plan.active_domains]}, "
        f"agents={formation_plan.estimated_agent_count}",
    ]

    output = DomainOutput(
        domain=Domain.CHEMISTRY,
        perspectives=perspectives,
        root_causes=[],
        consequences=[],
        causal_loops=[],
        game_state=None,
        raw_analysis="\n".join(raw_parts),
    )

    return output, formation_plan


# ---------------------------------------------------------------------------
# Module C: Analytical — runs DURING the battlefield
# ---------------------------------------------------------------------------

def run_chemistry(domain_input: DomainInput) -> DomainOutput:
    """
    Execute Chemistry's analytical function (Module C).

    This runs DURING the battlefield alongside other domains.
    Chemistry is an equal participant here — no special treatment.
    Subject to the same Ke cycle challenges as everyone else.

    Requires upstream outputs from other domains to analyze.
    """
    problem = domain_input.problem
    upstream = domain_input.upstream_outputs

    perspectives = []
    root_causes = []

    # Extract upstream data via bridge
    all_outputs = upstream
    physics_output = upstream.get(Domain.PHYSICS)
    metacognition_score = _extract_metacognition_score(upstream)
    upstream_root_causes = _extract_upstream_root_causes(upstream)

    # --- Concept 4: Chirality ---
    # Build competing narratives from domain outputs
    competing = []
    for domain, output in upstream.items():
        if output.root_causes:
            competing.append((domain.value, output))

    if len(competing) >= 2:
        chirality_perspective, chirality_assessments = run_chirality(
            competing, physics_output,
        )
        perspectives.append(chirality_perspective)

    # --- Concept 5: Catalysis ---
    if upstream_root_causes:
        catalysis_perspective, catalysis_result = run_catalysis(
            all_outputs, upstream_root_causes, metacognition_score
        )
        perspectives.append(catalysis_perspective)

        # The primary catalyst may be a root cause contribution
        if catalysis_result.primary_catalyst:
            root_causes.append(RootCause(
                variable=Variable(
                    name="catalytic_root",
                    description=catalysis_result.primary_catalyst.insight,
                    magnitude=catalysis_result.primary_catalyst.combined_score,
                    direction=Direction.NEUTRAL,
                    confidence=catalysis_result.primary_catalyst.truth_alignment_score,
                    source_framework=FrameworkID.CATALYSIS,
                    is_hidden=False,
                    is_user_stated=False,
                    evidence=[
                        catalysis_result.catalytic_moment_phrasing,
                    ],
                ),
                evidence_chain=[
                    f"Catalyst: {catalysis_result.primary_catalyst.insight[:80]}",
                    f"Barriers addressed: {len(catalysis_result.activation_barriers)}",
                    catalysis_result.catalytic_moment_phrasing,
                ],
                confidence=catalysis_result.primary_catalyst.truth_alignment_score,
                frameworks_that_agree=[FrameworkID.CATALYSIS],
            ))

    # --- Concept 6: Resonance ---
    # Check if resonance is needed (multiple valid answers?)
    convergence_achieved = len(upstream_root_causes) <= 2  # rough proxy
    resonance_perspective, resonance_result = run_resonance(
        all_outputs, convergence_achieved
    )
    perspectives.append(resonance_perspective)

    # Build raw analysis
    raw_parts = []
    if competing:
        raw_parts.append(f"Chirality: {len(competing)} narratives compared")
    if upstream_root_causes:
        raw_parts.append(
            f"Catalysis: {len(upstream_root_causes)} root causes analyzed, "
            f"primary catalyst {'found' if root_causes else 'not found'}"
        )
    raw_parts.append(
        f"Resonance: requires_resonance={resonance_result.requires_resonance}, "
        f"stability={resonance_result.hybrid_stability_score:.2f}, "
        f"ambiguity={resonance_result.irreducible_ambiguity}"
    )

    return DomainOutput(
        domain=Domain.CHEMISTRY,
        perspectives=perspectives,
        root_causes=root_causes,
        consequences=[],
        causal_loops=[],
        game_state=None,
        raw_analysis="\n".join(raw_parts),
    )


# ---------------------------------------------------------------------------
# Ke Cycle Entry — challenge another domain's output
# ---------------------------------------------------------------------------

def challenge(challenge_input: ChallengeInput) -> ChallengeOutput:
    """
    Chemistry challenges another domain's output (Ke cycle).

    Per Wu Xing: Fire melts Metal — Chemistry checks Mathematics.
    "Your precision is elegant but reality is messy."

    Chemistry scrutinizes by checking:
    - Rigidity: is the mathematical structure too rigid for real-world application?
    - Messiness not captured: has math oversimplified messy variables?
    - Precision without accuracy: precise numbers on the wrong question?
    - Oversimplification: variables that got flattened by dimensional reduction
    """
    target = challenge_input.target_output
    contradictions = []
    unsupported = []
    confidence_adjustments: dict[str, float] = {}
    flags = []

    all_vars = []
    for p in target.perspectives:
        all_vars.extend(p.variables_found)

    # Check 1: Rigidity — variables forced into clean categories that may not fit
    for var in all_vars:
        if var.confidence > 0.9:
            flags.append(
                f"'{var.name}' has {var.confidence:.0%} confidence. "
                "Chemistry asks: is this precision or false certainty? "
                "Real-world variables rarely reach 90%+ confidence. "
                "Has the math flattened uncertainty into artificial precision?"
            )
            confidence_adjustments[var.name] = var.confidence * 0.85

    # Check 2: Variables with magnitude exactly 0.0 or 1.0 — suspicious extremes
    for var in all_vars:
        if var.magnitude >= 0.99 or var.magnitude <= 0.01:
            contradictions.append(
                f"'{var.name}' has extreme magnitude ({var.magnitude:.2f}). "
                "Chemistry says: real-world variables are never this clean. "
                "Something has been oversimplified."
            )

    # Check 3: Root causes with single-framework agreement
    for rc in target.root_causes:
        if len(rc.frameworks_that_agree) <= 1:
            unsupported.append(
                f"Root cause '{rc.variable.name}' is supported by only "
                f"{len(rc.frameworks_that_agree)} framework(s). "
                "Chemistry requires cross-validation — a single framework "
                "can produce a precise answer to the wrong question."
            )

    # Check 4: Dimensional reduction may have lost important variables
    if target.perspectives:
        total_variables = len(all_vars)
        unique_names = len({v.name for v in all_vars})
        if total_variables > unique_names * 2.5:
            flags.append(
                f"High redundancy detected ({total_variables} variables, "
                f"{unique_names} unique). Dimensional reduction may have been "
                "too aggressive — important nuance could have been collapsed."
            )

    total_vars = len(all_vars) if all_vars else 1
    issue_count = len(contradictions) + len(unsupported) + len(flags)
    scrutiny_score = min(issue_count / total_vars, 1.0)

    return ChallengeOutput(
        challenger_domain=Domain.CHEMISTRY,
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


def _extract_upstream_root_causes(
    upstream: dict[Domain, DomainOutput],
) -> list[RootCause]:
    """Extract all root causes from upstream domain outputs."""
    root_causes = []
    for output in upstream.values():
        root_causes.extend(output.root_causes)
    return root_causes
