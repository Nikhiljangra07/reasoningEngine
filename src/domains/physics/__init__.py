"""
Physics Domain (Earth) — Isolated Island Module.

Root finder and path projector. Maps the terrain — finds the real problem,
projects where it leads, and penetrates user bias to surface what they
can't or won't tell you.

Phase 1: Squeezes the system from the outside (what the user tells you).
Phase 2: Squeezes the user's story from the inside (what they don't tell you).

Between the two, variable y has nowhere to hide.

BRIDGE CONTRACT:
  input:  DomainInput (problem + any upstream outputs)
  output: DomainOutput (root causes, force map, trajectory, bias flags)
  challenge_input:  ChallengeInput (another domain's output to scrutinize)
  challenge_output: ChallengeOutput (causal violations, energy imbalances, reality check)

ISOLATION: This module imports ONLY from src.core.types.
           It does NOT import from any other domain or from maths.
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
    Problem,
    Variable,
)
from src.domains.physics.phase1_root import run_phase1
from src.domains.physics.phase2_bias import run_phase2


# ---------------------------------------------------------------------------
# Sheng Cycle Entry — standard domain execution
# ---------------------------------------------------------------------------

def run_physics(domain_input: DomainInput) -> DomainOutput:
    """
    Execute the full Physics domain analysis.

    Accepts a DomainInput (bridge contract).
    Returns a DomainOutput (bridge contract).

    Phase 1 runs first (root finding + trajectory).
    Phase 2 runs second (bias penetration), using Phase 1's output
    to detect where the user's story doesn't match the physics.
    """
    problem = domain_input.problem

    # Phase 1: Root finding & trajectory
    phase1_output = run_phase1(problem)

    # Phase 2: Bias penetration (uses Phase 1 findings)
    phase2_output = run_phase2(problem, phase1_output)

    # Merge both phases into a single domain output
    return DomainOutput(
        domain=Domain.PHYSICS,
        perspectives=phase1_output.perspectives + phase2_output.perspectives,
        root_causes=phase1_output.root_causes + phase2_output.root_causes,
        consequences=phase1_output.consequences,
        causal_loops=[],
        game_state=None,
        raw_analysis=(
            "=== PHYSICS PHASE 1: ROOT FINDING & TRAJECTORY ===\n\n"
            + phase1_output.raw_analysis
            + "\n\n=== PHYSICS PHASE 2: BIAS PENETRATION ===\n\n"
            + phase2_output.raw_analysis
        ),
    )


# ---------------------------------------------------------------------------
# Ke Cycle Entry — challenge another domain's output
# ---------------------------------------------------------------------------

def challenge(challenge_input: ChallengeInput) -> ChallengeOutput:
    """
    Physics challenges another domain's output (Ke cycle).

    Per Wu Xing: Earth dams Water — Physics checks Psychology.
    "Your story is compelling but does it survive material reality?"

    Physics scrutinizes by checking:
    - Causal violations: do the claims follow cause-and-effect?
    - Energy imbalances: does the effort/output math add up?
    - Unsupported claims: are there assertions without physical evidence?
    - Reality check: does this survive contact with the laws of force/energy/trajectory?
    """
    target = challenge_input.target_output
    contradictions = []
    unsupported = []
    confidence_adjustments: dict[str, float] = {}
    flags = []

    for perspective in target.perspectives:
        for var in perspective.variables_found:
            # Check 1: Causal violation — high magnitude + low evidence = unsupported
            if var.magnitude > 0.6 and len(var.evidence) < 2:
                unsupported.append(
                    f"'{var.name}' claims magnitude {var.magnitude:.2f} "
                    f"but has only {len(var.evidence)} evidence point(s). "
                    "Physics requires causal evidence for high-magnitude claims."
                )
                confidence_adjustments[var.name] = var.confidence * 0.7

            # Check 2: Energy imbalance — positive claims without visible source
            if var.direction == Direction.POSITIVE and var.magnitude > 0.5 and var.is_hidden:
                flags.append(
                    f"'{var.name}' is a hidden positive force (magnitude: {var.magnitude:.2f}). "
                    "Physics asks: where is this energy coming from? "
                    "Hidden positive forces require a source."
                )

            # Check 3: Contradiction with physical constraints
            if var.direction == Direction.NEUTRAL and var.magnitude > 0.7:
                contradictions.append(
                    f"'{var.name}' is marked neutral but has high magnitude ({var.magnitude:.2f}). "
                    "In physics, high-magnitude forces HAVE a direction. "
                    "This neutrality may be masking the true direction."
                )

    # Check root causes for physical plausibility
    for rc in target.root_causes:
        if rc.confidence > 0.8 and len(rc.evidence_chain) < 3:
            unsupported.append(
                f"Root cause '{rc.variable.name}' claims {rc.confidence:.0%} confidence "
                f"but evidence chain has only {len(rc.evidence_chain)} step(s). "
                "Physics requires deeper causal chains for high-confidence root causes."
            )
            confidence_adjustments[rc.variable.name] = rc.confidence * 0.8

    # Calculate overall scrutiny score
    total_vars = sum(len(p.variables_found) for p in target.perspectives)
    issue_count = len(contradictions) + len(unsupported) + len(flags)
    scrutiny_score = min(issue_count / max(total_vars, 1), 1.0)

    return ChallengeOutput(
        challenger_domain=Domain.PHYSICS,
        target_domain=challenge_input.target_domain,
        contradictions=contradictions,
        unsupported_claims=unsupported,
        confidence_adjustments=confidence_adjustments,
        scrutiny_score=scrutiny_score,
        flags=flags,
    )
