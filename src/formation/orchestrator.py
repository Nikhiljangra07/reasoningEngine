"""
Wu Xing Formation Orchestrator — The 7-Stage Dual-Cycle Engine.

Now with Funnel feedback loop between iterations and multi-answer output.

7 Stages:
  1. Chemistry Reads (Governance — sets formation, queries cache for priors)
  2. Manifold Opens (seed the space)
  3. Dual Cycles Deploy (Sheng + Ke simultaneously) + Funnel between iterations
  4. Bayesian Backbone (continuous update from all funneled outputs)
  5. Convergence Check (Gibbs Free Energy)
  6. Ergodicity & Fragility (final gate)
  7. Metacognitive Calibration (delivery tuning)

The Funnel runs AFTER each dual-cycle pass:
  - Filters by connection density (NOT by comfort or convergence direction)
  - Ke scores drive the funnel: high scrutiny = needs work, low = stable
  - Variable cap (30) bounds O(n²) without killing depth
  - Filtered variables go to cache for future problems

OUTPUT: NEVER a single answer. Always top 2-4 trajectories with confidence,
uncertainty, and a "more underneath" flag. The USER decides depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    ChallengeInput,
    ChallengeOutput,
    Consequence,
    Domain,
    DomainInput,
    DomainOutput,
    FragilityResult,
    Perspective,
    Problem,
    RootCause,
    Variable,
)
from src.formation.cycles import (
    get_sheng_order,
    get_active_ke_pairs,
)
from src.formation.convergence_protocol import (
    check_convergence,
    ConvergenceHistory,
    MAX_ITERATIONS,
)
from src.formation.funnel import (
    run_funnel,
    FunnelResult,
    ConnectionScore,
)
from src.formation.cache import (
    create_cache,
    cache_variable,
    query_cache,
    cache_to_priors,
    save_cache,
    CacheStore,
)

# Domain runner imports — bridge entry points only.
from src.domains.physics import (
    run_physics,
    challenge as physics_challenge,
)
from src.maths import (
    run_mathematics,
    challenge as maths_challenge,
)
from src.domains.psychology import (
    run_psychology,
    challenge as psychology_challenge,
)
from src.domains.philosophy import (
    run_philosophy,
    challenge as philosophy_challenge,
)
from src.domains.chemistry import (
    run_governance,
    run_chemistry,
    challenge as chemistry_challenge,
)
from src.domains.chemistry.governance import FormationPlan


# Domain registries
DOMAIN_RUNNERS = {
    Domain.PHYSICS: run_physics,
    Domain.MATHEMATICS: run_mathematics,
    Domain.PSYCHOLOGY: run_psychology,
    Domain.PHILOSOPHY: run_philosophy,
    Domain.CHEMISTRY: run_chemistry,
}

DOMAIN_CHALLENGERS = {
    Domain.PHYSICS: physics_challenge,
    Domain.MATHEMATICS: maths_challenge,
    Domain.PSYCHOLOGY: psychology_challenge,
    Domain.PHILOSOPHY: philosophy_challenge,
    Domain.CHEMISTRY: chemistry_challenge,
}


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A single probable answer/trajectory with confidence and consequences."""
    root_cause: RootCause
    confidence: float
    consequences: list[Consequence]
    cost_if_ignored: str
    source_domains: list[str]


@dataclass
class FormationResult:
    """
    The final output. NEVER a single answer.

    Always: top 2-4 trajectories, uncertainty, and a "more underneath" flag.
    The user decides whether to dig deeper. The engine does not decide
    when to stop exploring.
    """
    problem: Problem
    trajectories: list[Trajectory]              # top 2-4 probable answers
    uncertainty: str                             # what remains genuinely uncertain
    more_underneath: bool                        # always True — there's always more
    more_underneath_description: str
    bias_penetration: str
    hidden_purpose: str
    domain_outputs: dict[Domain, DomainOutput]
    formation_plan: FormationPlan
    convergence_history: ConvergenceHistory
    ke_results: list[ChallengeOutput]
    funnel_history: list[FunnelResult]
    fragility: FragilityResult | None
    delivery_mode: str
    catalytic_moment: str
    resonance_hybrid: str
    irreducible_ambiguity: bool


# ---------------------------------------------------------------------------
# The Engine
# ---------------------------------------------------------------------------

def run_formation(
    problem: Problem,
    cache_path: str = "",
) -> FormationResult:
    """
    Execute the full 7-stage Wu Xing Formation.

    Master entry point for the entire reasoning engine.
    """
    # =====================================================================
    # STAGE 1: CHEMISTRY READS (Governance)
    # =====================================================================
    # Query cache for priors from similar past problems
    cache = create_cache(cache_path)
    cache_hits = query_cache(cache, problem.statement)
    cached_priors = cache_to_priors(cache_hits)

    # If we have cached priors, inject them as additional problem variables
    if cached_priors:
        problem.variables.extend(cached_priors)

    gov_input = DomainInput(problem=problem)
    gov_output, formation_plan = run_governance(gov_input)

    active_domains = formation_plan.active_domains
    sheng_order = get_sheng_order(active_domains)
    ke_pairs = get_active_ke_pairs(active_domains)

    # =====================================================================
    # STAGE 2: MANIFOLD OPENS
    # =====================================================================
    domain_outputs: dict[Domain, DomainOutput] = {}
    domain_outputs[Domain.CHEMISTRY] = gov_output

    previous_outputs: dict[Domain, DomainOutput] = {}
    previous_root_causes: list[RootCause] = []
    convergence_history = ConvergenceHistory()
    all_ke_results: list[ChallengeOutput] = []
    funnel_history: list[FunnelResult] = []
    previous_connection_scores: dict[str, ConnectionScore] | None = None

    # =====================================================================
    # STAGES 3-5: DUAL CYCLES + FUNNEL + CONVERGENCE
    # =====================================================================
    for iteration in range(1, MAX_ITERATIONS + 1):

        # --- STAGE 3a: SHENG CYCLE ---
        for domain in sheng_order:
            if domain == Domain.CHEMISTRY and iteration == 1:
                continue  # governance already ran

            domain_input = DomainInput(
                problem=problem,
                upstream_outputs=dict(domain_outputs),
            )

            runner = DOMAIN_RUNNERS.get(domain)
            if runner:
                output = runner(domain_input)
                domain_outputs[domain] = output

        # --- STAGE 3b: KE CYCLE ---
        iteration_ke_results: list[ChallengeOutput] = []
        for challenger_domain, target_domain in ke_pairs:
            if target_domain not in domain_outputs:
                continue

            challenger_fn = DOMAIN_CHALLENGERS.get(challenger_domain)
            if challenger_fn:
                ke_input = ChallengeInput(
                    challenger_domain=challenger_domain,
                    target_domain=target_domain,
                    target_output=domain_outputs[target_domain],
                )
                ke_result = challenger_fn(ke_input)
                iteration_ke_results.append(ke_result)

        all_ke_results = iteration_ke_results

        # --- FUNNEL: Filter + Ke feedback ---
        funnel_result = run_funnel(
            domain_outputs=domain_outputs,
            ke_results=iteration_ke_results,
            previous_connection_scores=previous_connection_scores,
            iteration=iteration,
            problem_statement=problem.statement,
        )
        funnel_history.append(funnel_result)

        # Cache filtered-out variables
        for cached_var in funnel_result.cached_variables:
            cache_variable(
                cache, cached_var, problem.statement,
                f"Filtered at iteration {iteration}", iteration, 0
            )

        # The funnel's downstream becomes the effective domain output
        # for convergence checking. We don't replace domain_outputs
        # (domains re-run on full context), but we track what the funnel kept.
        previous_connection_scores = {
            v.name: ConnectionScore(
                variable_name=v.name, variable=v, connections=1,
                connected_domains=[], ke_status="unchallenged",
                ke_scrutiny=0.0, zero_connection_passes=0,
            )
            for v in funnel_result.downstream_variables
        }

        # --- STAGE 4: BAYESIAN BACKBONE ---
        current_root_causes = _collect_all_root_causes(domain_outputs)

        # --- STAGE 5: CONVERGENCE CHECK ---
        snapshot = check_convergence(
            iteration=iteration,
            current_root_causes=current_root_causes,
            previous_root_causes=previous_root_causes,
            current_outputs=domain_outputs,
            previous_outputs=previous_outputs,
            ke_results=iteration_ke_results,
        )
        convergence_history.snapshots.append(snapshot)

        if snapshot.is_converged:
            convergence_history.final_converged = True
            convergence_history.total_iterations = iteration
            break

        previous_outputs = dict(domain_outputs)
        previous_root_causes = list(current_root_causes)

    else:
        convergence_history.forced_stop = True
        convergence_history.total_iterations = MAX_ITERATIONS

    # =====================================================================
    # STAGE 6: ERGODICITY & FRAGILITY
    # =====================================================================
    all_root_causes = _collect_all_root_causes(domain_outputs)
    all_consequences = _collect_all_consequences(domain_outputs)

    # Build top 2-4 trajectories (NOT single answer)
    trajectories = _build_trajectories(all_root_causes, all_consequences, domain_outputs)

    fragility_result = None
    if trajectories:
        from src.maths.fragility import stress_test
        fragility_result = stress_test(
            problem, trajectories[0].root_cause, all_consequences
        )

    # =====================================================================
    # STAGE 7: METACOGNITIVE CALIBRATION
    # =====================================================================
    delivery_mode = _extract_delivery_mode(domain_outputs)
    catalytic_moment = _extract_catalytic_moment(domain_outputs)
    resonance_hybrid = _extract_resonance_hybrid(domain_outputs)
    irreducible_ambiguity = _extract_ambiguity(domain_outputs)
    bias_summary = _build_bias_summary(
        trajectories[0].root_cause if trajectories else None, domain_outputs
    )
    hidden_purpose = _extract_hidden_purpose(domain_outputs)

    # Build uncertainty description
    uncertainty = _build_uncertainty(convergence_history, funnel_history, all_ke_results)

    # Save cache for future problems
    if cache_path:
        save_cache(cache)

    # =====================================================================
    # ASSEMBLE
    # =====================================================================
    return FormationResult(
        problem=problem,
        trajectories=trajectories,
        uncertainty=uncertainty,
        more_underneath=True,  # always True — there's ALWAYS more underneath
        more_underneath_description=(
            "This analysis surfaced the top probable trajectories. "
            "Deeper analysis is available. The engine has cached "
            f"{sum(f.variables_cached for f in funnel_history)} variables "
            "that may contain additional insights. Continue if you want to dig deeper."
        ),
        bias_penetration=bias_summary,
        hidden_purpose=hidden_purpose,
        domain_outputs=domain_outputs,
        formation_plan=formation_plan,
        convergence_history=convergence_history,
        ke_results=all_ke_results,
        funnel_history=funnel_history,
        fragility=fragility_result,
        delivery_mode=delivery_mode,
        catalytic_moment=catalytic_moment,
        resonance_hybrid=resonance_hybrid,
        irreducible_ambiguity=irreducible_ambiguity,
    )


# ---------------------------------------------------------------------------
# Trajectory building (multi-answer, NEVER single root cause)
# ---------------------------------------------------------------------------

def _build_trajectories(
    root_causes: list[RootCause],
    consequences: list[Consequence],
    outputs: dict[Domain, DomainOutput],
) -> list[Trajectory]:
    """
    Build top 2-4 probable trajectories. NEVER a single answer.

    Each trajectory is a root cause + its confidence + consequences + cost.
    """
    if not root_causes:
        return []

    # Deduplicate and rank
    seen: set[str] = set()
    unique: list[RootCause] = []
    for rc in root_causes:
        if rc.variable.name not in seen:
            seen.add(rc.variable.name)
            unique.append(rc)

    # Score: confidence + hidden bonus + cross-domain agreement
    def score(rc: RootCause) -> float:
        conf = rc.confidence * 0.40
        hidden = 0.25 if rc.variable.is_hidden else 0.0
        agree = min(len(rc.frameworks_that_agree) / 5.0, 1.0) * 0.20
        # Cross-domain: which domains found this?
        domains = set()
        for fw in rc.frameworks_that_agree:
            domains.add(fw.value.split("_")[0])
        cross = min(len(domains) / 3.0, 1.0) * 0.15
        return conf + hidden + agree + cross

    unique.sort(key=score, reverse=True)

    # Take top 4
    top = unique[:4]

    trajectories = []
    for rc in top:
        # Find consequences related to this root cause
        related_consequences = [
            c for c in consequences
            if rc.variable.name in c.description.lower()
            or any(rc.variable.name in (e or "").lower() for e in rc.evidence_chain)
        ]
        if not related_consequences:
            related_consequences = consequences[:2]  # fallback: most severe

        # Determine which domains agree
        source_domains = list({
            fw.value for fw in rc.frameworks_that_agree
        })

        # Cost if ignored
        if rc.variable.direction.value == "negative":
            cost = (
                f"If '{rc.variable.name}' is left unaddressed, "
                f"consequences compound. Confidence: {rc.confidence:.0%}."
            )
        else:
            cost = f"Variable '{rc.variable.name}' continues unchecked."

        trajectories.append(Trajectory(
            root_cause=rc,
            confidence=rc.confidence,
            consequences=related_consequences,
            cost_if_ignored=cost,
            source_domains=source_domains,
        ))

    return trajectories


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def _collect_all_root_causes(outputs: dict[Domain, DomainOutput]) -> list[RootCause]:
    all_rc = []
    for output in outputs.values():
        all_rc.extend(output.root_causes)
    return all_rc


def _collect_all_consequences(outputs: dict[Domain, DomainOutput]) -> list[Consequence]:
    consequences = []
    for output in outputs.values():
        consequences.extend(output.consequences)
    return consequences


# ---------------------------------------------------------------------------
# Stage 7 extractors
# ---------------------------------------------------------------------------

def _extract_delivery_mode(outputs: dict[Domain, DomainOutput]) -> str:
    psych = outputs.get(Domain.PSYCHOLOGY)
    if psych:
        for p in psych.perspectives:
            if p.framework.value == "metacognition":
                for v in p.variables_found:
                    if "metacognition_level" in v.name:
                        score = v.magnitude
                        if score > 0.7:
                            return "direct"
                        elif score > 0.4:
                            return "building"
                        return "gentle"
    return "building"


def _extract_catalytic_moment(outputs: dict[Domain, DomainOutput]) -> str:
    chem = outputs.get(Domain.CHEMISTRY)
    if chem:
        for p in chem.perspectives:
            if p.framework.value == "catalysis":
                for v in p.variables_found:
                    if "catalytic_insight" in v.name:
                        return v.description
    return "No catalytic moment identified."


def _extract_resonance_hybrid(outputs: dict[Domain, DomainOutput]) -> str:
    chem = outputs.get(Domain.CHEMISTRY)
    if chem:
        for p in chem.perspectives:
            if p.framework.value == "resonance":
                for v in p.variables_found:
                    if "resonance_hybrid" in v.name:
                        return v.description
    return "Single structure sufficient."


def _extract_ambiguity(outputs: dict[Domain, DomainOutput]) -> bool:
    chem = outputs.get(Domain.CHEMISTRY)
    if chem:
        for p in chem.perspectives:
            if p.framework.value == "resonance":
                if "IRREDUCIBLE AMBIGUITY" in p.content:
                    return True
    return False


def _extract_hidden_purpose(outputs: dict[Domain, DomainOutput]) -> str:
    phil = outputs.get(Domain.PHILOSOPHY)
    if phil:
        for p in phil.perspectives:
            if p.framework.value == "teleology":
                for v in p.variables_found:
                    if "hidden_purpose" in v.name:
                        return v.description
    return "No hidden purpose identified."


def _build_bias_summary(
    root: RootCause | None, outputs: dict[Domain, DomainOutput]
) -> str:
    parts = []
    if root and root.bias_that_hid_it:
        parts.append(f"Primary bias: {root.bias_that_hid_it}")

    psych = outputs.get(Domain.PSYCHOLOGY)
    if psych:
        for p in psych.perspectives:
            if p.framework.value == "motivated_reasoning":
                for v in p.variables_found:
                    if "motivated_reasoning" in v.name:
                        parts.append(f"Motivated reasoning: {v.description[:100]}")

    phil = outputs.get(Domain.PHILOSOPHY)
    if phil:
        for p in phil.perspectives:
            if p.framework.value == "phenomenology":
                for v in p.variables_found:
                    if "horizon" in v.name:
                        parts.append(f"Horizon limit: {v.description[:100]}")

    return " | ".join(parts) if parts else "No specific bias identified."


def _build_uncertainty(
    convergence: ConvergenceHistory,
    funnel_history: list[FunnelResult],
    ke_results: list[ChallengeOutput],
) -> str:
    """Build honest uncertainty description."""
    parts = []

    if convergence.forced_stop:
        parts.append(
            f"Engine did not fully converge in {convergence.total_iterations} iterations. "
            "Results represent the best current state, not a final answer."
        )

    # High Ke scrutiny pairs = unresolved challenges
    high_ke = [ke for ke in ke_results if ke.scrutiny_score > 0.5]
    if high_ke:
        pairs = [f"{ke.challenger_domain.value}→{ke.target_domain.value}" for ke in high_ke]
        parts.append(
            f"{len(high_ke)} domain pair(s) still under high scrutiny: {', '.join(pairs)}. "
            "These findings have not fully survived challenge."
        )

    # Funnel cached variables = potential hidden insights
    total_cached = sum(f.variables_cached for f in funnel_history)
    if total_cached > 0:
        parts.append(
            f"{total_cached} variable(s) were filtered to cache. "
            "They may contain additional insights on deeper analysis."
        )

    if parts:
        return " ".join(parts)
    return "Analysis converged with no significant remaining uncertainty."
