"""
Mathematics Domain (Metal) — Isolated Island Module.

The rigid grid. Provides structure, inference, convergence detection,
and the Bayesian backbone that connects everything.

9 Layers:
1. Signal vs. Noise — contextual relevance filtering
2. Category Theory — universal translator between domains
3. Manifold Theory — multi-angle holder + dimensional reduction
4. N-Dimensional Capacity — integrated into Manifold
5. Convergence & Stopping — knows when the squeeze is done
6. Bayesian Inference — the living update engine
7. Game Theory — strategic multi-agent reasoning
8. Causal Loop Analysis — circular feedback detection
9. Ergodicity & Fragility — final stress test

BRIDGE CONTRACT:
  input:  DomainInput (problem + upstream outputs, especially Physics)
  output: DomainOutput (structured variables, convergence, bayesian, game theory, causal loops, ergodicity)
  challenge_input:  ChallengeInput (another domain's output to scrutinize)
  challenge_output: ChallengeOutput (logical gaps, contradictions, formal scrutiny, pruned claims)

ISOLATION: This module imports ONLY from src.core.types and its own layer files.
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
    Problem,
    RootCause,
    Variable,
)
from src.maths.signal_noise import filter_perspectives, SignalNoiseResult
from src.maths.category import analyze_categories, CategoryResult
from src.maths.manifold import build_manifold, find_least_action_path, ManifoldState
from src.maths.convergence import check_convergence, ConvergenceState
from src.maths.bayesian import (
    initialize_prior,
    update_with_evidence,
    get_root_candidates,
    BayesianState,
)
from src.maths.game_theory import analyze_game, extract_game_variables
from src.maths.causal_loops import analyze_loops, LoopAnalysis
from src.maths.fragility import stress_test


# ---------------------------------------------------------------------------
# Sheng Cycle Entry — standard domain execution
# ---------------------------------------------------------------------------

def run_mathematics(domain_input: DomainInput) -> DomainOutput:
    """
    Execute the full Mathematics domain analysis.

    Accepts a DomainInput (bridge contract).
    Returns a DomainOutput (bridge contract).

    Mathematics processes the outputs of other domains through
    its 9-layer infrastructure. It does NOT analyze the user's
    problem directly — it structures, filters, reduces, and infers.
    """
    problem = domain_input.problem
    upstream = domain_input.upstream_outputs

    # Collect all perspectives from upstream domain outputs
    all_perspectives: list[Perspective] = []
    all_domain_outputs: list[DomainOutput] = list(upstream.values())
    for output in all_domain_outputs:
        all_perspectives.extend(output.perspectives)

    # If no upstream data, work directly from problem variables
    if not all_perspectives:
        return _empty_output()

    # --- Layer 1: Signal vs. Noise ---
    signal_result = filter_perspectives(problem, all_perspectives)
    active_perspectives = signal_result.signal + signal_result.latent

    # --- Layer 2: Category Theory ---
    category_result = analyze_categories(all_domain_outputs)
    emergent_vars = category_result.emergent_variables
    if emergent_vars:
        active_perspectives.append(Perspective(
            framework=FrameworkID.CATEGORY_THEORY,
            domain=Domain.MATHEMATICS,
            content=(
                f"Category Theory: {len(emergent_vars)} emergent variables "
                f"from {len(category_result.morphisms)} morphisms."
            ),
            variables_found=emergent_vars,
            weight=0.8,
        ))

    # --- Layers 3+4: Manifold + N-Dimensional ---
    manifold = build_manifold(problem, active_perspectives)

    # --- Layer 5: Convergence ---
    convergence_state = check_convergence(active_perspectives)

    # --- Layer 6: Bayesian Inference ---
    bayesian_state = initialize_prior(problem)
    bayesian_state = update_with_evidence(bayesian_state, all_domain_outputs)
    root_candidates = get_root_candidates(bayesian_state)

    # --- Layer 7: Game Theory ---
    game_state = analyze_game(problem)
    game_variables: list[Variable] = []
    if game_state:
        game_variables = extract_game_variables(game_state)
        if game_variables:
            active_perspectives.append(Perspective(
                framework=FrameworkID.GAME_THEORY,
                domain=Domain.MATHEMATICS,
                content=f"Game Theory: {len(game_variables)} strategic variables.",
                variables_found=game_variables,
                weight=0.85,
            ))

    # --- Layer 8: Causal Loop Analysis ---
    loop_analysis = analyze_loops(problem, all_domain_outputs)
    all_loops = loop_analysis.reinforcing_loops + loop_analysis.balancing_loops
    if loop_analysis.loop_variables:
        active_perspectives.append(Perspective(
            framework=FrameworkID.CAUSAL_LOOPS,
            domain=Domain.MATHEMATICS,
            content=f"Causal Loops: {len(all_loops)} loops detected.",
            variables_found=loop_analysis.loop_variables,
            weight=0.85,
        ))

    # --- Layer 9: Ergodicity & Fragility ---
    # (Runs as post-convergence gate — included in output for orchestrator to use)
    fragility_result = None
    if root_candidates:
        # Collect all consequences from upstream
        all_consequences = []
        for output in all_domain_outputs:
            all_consequences.extend(output.consequences)
        fragility_result = stress_test(problem, root_candidates[0], all_consequences)

    # --- Least Action Path ---
    least_action = find_least_action_path(manifold)

    # Build raw analysis
    raw_parts = [
        f"Signal/Noise: {len(signal_result.signal)} signal, {len(signal_result.noise)} noise",
        f"Category Theory: {len(category_result.morphisms)} morphisms, {len(category_result.isomorphisms)} isomorphisms",
        f"Manifold: {manifold.total_dimensions} dims → {manifold.intrinsic_dimensions} intrinsic",
        f"Convergence: {'converged' if convergence_state.has_converged else 'not converged'} (stability: {convergence_state.stability_score:.2f})",
        f"Bayesian: {len(bayesian_state.beliefs)} beliefs, {len(root_candidates)} root candidates",
        f"Game Theory: {'active' if game_state else 'inactive'}",
        f"Causal Loops: {len(all_loops)} ({len(loop_analysis.reinforcing_loops)} reinforcing, {len(loop_analysis.balancing_loops)} balancing)",
        f"Fragility: {fragility_result.rating.value if fragility_result else 'not assessed'}",
    ]

    return DomainOutput(
        domain=Domain.MATHEMATICS,
        perspectives=active_perspectives,
        root_causes=root_candidates,
        consequences=[],  # Maths doesn't generate consequences, it processes them
        causal_loops=all_loops,
        game_state=game_state,
        raw_analysis="\n".join(raw_parts),
    )


# ---------------------------------------------------------------------------
# Ke Cycle Entry — challenge another domain's output
# ---------------------------------------------------------------------------

def challenge(challenge_input: ChallengeInput) -> ChallengeOutput:
    """
    Mathematics challenges another domain's output (Ke cycle).

    Per Wu Xing: Metal chops Wood — Mathematics checks Philosophy.
    "Your reframing is creative but does it survive formal scrutiny?"

    Mathematics scrutinizes by checking:
    - Logical gaps: does the reasoning chain hold?
    - Contradictions: do any claims conflict with each other?
    - Formal scrutiny: can the claims be expressed precisely?
    - Pruned claims: which claims don't survive Occam's Razor?
    """
    target = challenge_input.target_output
    contradictions = []
    unsupported = []
    confidence_adjustments: dict[str, float] = {}
    flags = []

    all_vars = []
    for p in target.perspectives:
        all_vars.extend(p.variables_found)

    # Check 1: Internal contradictions — variables that claim opposite things
    for i, v1 in enumerate(all_vars):
        for v2 in all_vars[i + 1:]:
            if (v1.name == v2.name
                    and v1.direction != v2.direction
                    and v1.direction != Direction.NEUTRAL
                    and v2.direction != Direction.NEUTRAL):
                contradictions.append(
                    f"Variable '{v1.name}' has contradictory directions: "
                    f"{v1.direction.value} (confidence: {v1.confidence:.2f}) vs "
                    f"{v2.direction.value} (confidence: {v2.confidence:.2f}). "
                    "Both cannot be true simultaneously."
                )

    # Check 2: Occam's Razor — are there too many variables for the explanation?
    unique_names = {v.name for v in all_vars}
    if len(all_vars) > len(unique_names) * 2:
        flags.append(
            f"Parsimony violation: {len(all_vars)} variables but only "
            f"{len(unique_names)} unique names. Redundancy detected — "
            "the explanation may be unnecessarily complex."
        )

    # Check 3: Confidence without evidence
    for var in all_vars:
        if var.confidence > 0.7 and len(var.evidence) < 2:
            unsupported.append(
                f"'{var.name}' claims high confidence ({var.confidence:.2f}) "
                f"with only {len(var.evidence)} evidence point(s). "
                "Mathematics requires sufficient evidence for high confidence."
            )
            confidence_adjustments[var.name] = var.confidence * 0.75

    # Check 4: Root cause plausibility
    for rc in target.root_causes:
        if rc.confidence > 0.9 and len(rc.frameworks_that_agree) < 2:
            flags.append(
                f"Root cause '{rc.variable.name}' has {rc.confidence:.0%} confidence "
                f"but only {len(rc.frameworks_that_agree)} framework(s) agree. "
                "High confidence requires multi-framework convergence."
            )

    total_vars = len(all_vars) if all_vars else 1
    issue_count = len(contradictions) + len(unsupported) + len(flags)
    scrutiny_score = min(issue_count / total_vars, 1.0)

    return ChallengeOutput(
        challenger_domain=Domain.MATHEMATICS,
        target_domain=challenge_input.target_domain,
        contradictions=contradictions,
        unsupported_claims=unsupported,
        confidence_adjustments=confidence_adjustments,
        scrutiny_score=scrutiny_score,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_output() -> DomainOutput:
    """Return empty output when no upstream data is available."""
    return DomainOutput(
        domain=Domain.MATHEMATICS,
        raw_analysis="No upstream data available for mathematical analysis.",
    )
