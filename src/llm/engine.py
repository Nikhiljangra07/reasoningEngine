"""
Async Formation Engine — Steps 1.4 through 1.7.

This is the async runtime that:
1.4 — Spawns tributaries (dynamic fan-out based on formation plan)
1.5 — Merges results (fan-in reducer, preserves contradictions)
1.6 — Runs Ke cycle as second fan-out (5 critic calls in parallel)
1.7 — Loops until convergence (Gibbs check + Le Chatelier re-run logic)

The engine uses the LLMClient for all calls and the funnel between
iterations to keep variables bounded and Ke-informed.

ISOLATION: Imports from src.core.types, src.llm.client, src.llm.router,
           src.llm.validator, and src.formation modules.
           Domain logic is accessed via bridge contracts only.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from src.core.types import (
    ChallengeOutput,
    Consequence,
    Direction,
    Domain,
    DomainInput,
    DomainOutput,
    FrameworkID,
    Perspective,
    Problem,
    RootCause,
    SignalType,
    Variable,
)
from src.llm.client import LLMClient, LLMResponse
from src.llm.router import LLMFormationPlan, route_problem, ALL_CONCEPTS
from src.llm.validator import validate_formation, ValidationResult
from src.llm.prompts import get_domain_prompt, get_ke_critic_prompt
from src.formation.cycles import get_sheng_order, get_active_ke_pairs, KE_PAIRS
from src.formation.funnel import run_funnel, FunnelResult, ConnectionScore
from src.formation.convergence_protocol import (
    check_convergence,
    ConvergenceHistory,
    MAX_ITERATIONS,
)
from src.formation.cache import (
    create_cache,
    cache_variable,
    query_cache,
    cache_to_priors,
    save_cache,
    CacheStore,
)


# Domain law prompts are in src/llm/prompts.py
# Accessed via get_domain_prompt() and get_ke_critic_prompt()


# ---------------------------------------------------------------------------
# Engine Output
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A single probable answer/trajectory."""
    root_cause: RootCause
    confidence: float
    consequences: list[Consequence]
    cost_if_ignored: str
    source_domains: list[str]


@dataclass
class EngineResult:
    """Full result of the async formation engine."""
    problem: Problem
    trajectories: list[Trajectory]
    uncertainty: str
    more_underneath: bool
    more_underneath_description: str
    bias_penetration: str
    hidden_purpose: str
    domain_outputs: dict[Domain, DomainOutput]
    formation_plan: LLMFormationPlan
    validation_result: ValidationResult
    convergence_history: ConvergenceHistory
    ke_results: list[ChallengeOutput]
    funnel_history: list[FunnelResult]
    delivery_mode: str
    catalytic_moment: str
    irreducible_ambiguity: bool
    call_summary: dict


# ---------------------------------------------------------------------------
# The Async Engine
# ---------------------------------------------------------------------------

async def run_async_formation(
    problem: Problem,
    client: LLMClient,
    cache_path: str = "",
    max_iterations: int | None = None,
) -> EngineResult:
    """
    Run the full async Wu Xing Formation Engine.

    This is the async version of the orchestrator that uses real (or mock)
    LLM calls via the client, with dynamic fan-out and fan-in.
    """
    max_iter = max_iterations or MAX_ITERATIONS

    # =====================================================================
    # STAGE 1: CHEMISTRY READS (Router + Validation)
    # =====================================================================

    # Query cache for priors
    cache = create_cache(cache_path)
    cache_hits = query_cache(cache, problem.statement)
    if cache_hits:
        cached_priors = cache_to_priors(cache_hits)
        problem.variables.extend(cached_priors)

    # Chemistry Self-Assembly decides formation (Step 1.2)
    formation_plan = await route_problem(client, problem.statement, problem.context)

    # Math validates formation (Step 1.3)
    validation = validate_formation(formation_plan, problem)
    plan = validation.adjusted_plan

    active_domains = plan.active_domains
    sheng_order = get_sheng_order(active_domains)
    ke_pairs = get_active_ke_pairs(active_domains)

    # =====================================================================
    # STAGE 2: MANIFOLD OPENS
    # =====================================================================
    domain_outputs: dict[Domain, DomainOutput] = {}
    previous_outputs: dict[Domain, DomainOutput] = {}
    previous_root_causes: list[RootCause] = []
    convergence_history = ConvergenceHistory()
    all_ke_results: list[ChallengeOutput] = []
    funnel_history: list[FunnelResult] = []
    prev_connection_scores: dict[str, ConnectionScore] | None = None

    # Track which domains need re-run (Le Chatelier's)
    domains_to_skip: set[Domain] = set()

    # =====================================================================
    # STAGES 3-5: DUAL CYCLES + FUNNEL + CONVERGENCE
    # =====================================================================
    for iteration in range(1, max_iter + 1):

        # --- STEP 1.4: TRIBUTARY SPAWNING (Sheng Fan-Out) ---
        sheng_calls = []
        sheng_domains = []

        for domain in sheng_order:
            if domain in domains_to_skip:
                continue  # Le Chatelier's: low-scrutiny domain skips

            domain_str = domain.value
            concepts = plan.concepts_per_domain.get(domain_str, [])
            if not concepts:
                continue

            # Build the user message with problem + upstream context
            user_msg = _build_domain_user_message(problem, domain_outputs, domain)

            # Get the full domain law prompt (90% laws, 10% guidance)
            system_prompt = get_domain_prompt(domain_str, concepts)

            sheng_calls.append({
                "system_prompt": system_prompt,
                "user_message": user_msg,
                "domain": domain_str,
                "concept": "full_domain",
            })
            sheng_domains.append(domain)

        # Fan-out: all domain calls in parallel
        if sheng_calls:
            sheng_responses = await client.call_batch(sheng_calls)

            # --- STEP 1.5: FAN-IN REDUCER ---
            for domain, response in zip(sheng_domains, sheng_responses):
                if response.success:
                    output = _parse_domain_response(domain, response)
                    domain_outputs[domain] = output

        # --- STEP 1.6: KE CYCLE FAN-OUT ---
        ke_calls = []
        ke_pair_list = []

        for challenger, target in ke_pairs:
            if target not in domain_outputs:
                continue

            target_summary = _summarize_domain_output(domain_outputs[target])
            system_prompt = get_ke_critic_prompt(challenger.value, target.value)
            user_msg = (
                f"Challenge {target.value}'s output:\n\n{target_summary}\n\n"
                "Find weaknesses. Be specific."
            )

            ke_calls.append({
                "system_prompt": system_prompt,
                "user_message": user_msg,
                "domain": "critic",
                "concept": f"{challenger.value}_checks_{target.value}",
            })
            ke_pair_list.append((challenger, target))

        iteration_ke_results: list[ChallengeOutput] = []
        if ke_calls:
            ke_responses = await client.call_batch(ke_calls)

            for (challenger, target), response in zip(ke_pair_list, ke_responses):
                ke_output = _parse_ke_response(challenger, target, response)
                iteration_ke_results.append(ke_output)

        all_ke_results = iteration_ke_results

        # --- FUNNEL ---
        funnel_result = run_funnel(
            domain_outputs=domain_outputs,
            ke_results=iteration_ke_results,
            previous_connection_scores=prev_connection_scores,
            iteration=iteration,
            problem_statement=problem.statement,
        )
        funnel_history.append(funnel_result)

        # Cache filtered variables
        for cached_var in funnel_result.cached_variables:
            cache_variable(
                cache, cached_var, problem.statement,
                f"Filtered at iteration {iteration}", iteration, 0
            )

        # Update connection scores for next iteration
        prev_connection_scores = {
            v.name: ConnectionScore(
                variable_name=v.name, variable=v, connections=1,
                connected_domains=[], ke_status="unchallenged",
                ke_scrutiny=0.0, zero_connection_passes=0,
            )
            for v in funnel_result.downstream_variables
        }

        # --- LE CHATELIER'S RE-RUN LOGIC ---
        # High Ke scrutiny (>0.5) → domain MUST re-run next iteration
        # Low Ke scrutiny (<0.2) → domain CAN skip next iteration
        domains_to_skip = set()
        for ke in iteration_ke_results:
            if ke.scrutiny_score < 0.2 and ke.target_domain in active_domains:
                domains_to_skip.add(ke.target_domain)
            # High scrutiny domains are NOT added to skip → they re-run

        # --- STAGE 4: COLLECT ROOT CAUSES ---
        current_root_causes = []
        for output in domain_outputs.values():
            current_root_causes.extend(output.root_causes)

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
        convergence_history.total_iterations = max_iter

    # =====================================================================
    # STAGE 6 + 7: POST-CONVERGENCE
    # =====================================================================
    all_root_causes = []
    for output in domain_outputs.values():
        all_root_causes.extend(output.root_causes)

    trajectories = _build_trajectories(all_root_causes, domain_outputs)
    delivery_mode = _extract_field(domain_outputs, Domain.PSYCHOLOGY, "delivery_mode", "building")
    catalytic_moment = _extract_field(domain_outputs, Domain.CHEMISTRY, "catalyst", "No catalyst identified.")
    bias_summary = _build_bias_summary(trajectories, domain_outputs)
    hidden_purpose = _extract_field(domain_outputs, Domain.PHILOSOPHY, "hidden_utility", "No hidden purpose identified.")
    uncertainty = _build_uncertainty(convergence_history, funnel_history, all_ke_results)

    # Save cache
    if cache_path:
        save_cache(cache)

    return EngineResult(
        problem=problem,
        trajectories=trajectories,
        uncertainty=uncertainty,
        more_underneath=True,
        more_underneath_description=(
            f"Deeper analysis available. {sum(f.variables_cached for f in funnel_history)} "
            "variables cached. Continue to dig deeper."
        ),
        bias_penetration=bias_summary,
        hidden_purpose=hidden_purpose,
        domain_outputs=domain_outputs,
        formation_plan=plan,
        validation_result=validation,
        convergence_history=convergence_history,
        ke_results=all_ke_results,
        funnel_history=funnel_history,
        delivery_mode=delivery_mode,
        catalytic_moment=catalytic_moment,
        irreducible_ambiguity=False,
        call_summary=client.get_call_summary(),
    )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _build_domain_user_message(
    problem: Problem,
    current_outputs: dict[Domain, DomainOutput],
    target_domain: Domain,
) -> str:
    """Build the user message for a domain agent, including upstream context."""
    parts = [f"PROBLEM: {problem.statement}"]

    if problem.context:
        parts.append(f"CONTEXT: {problem.context}")

    # Include problem variables
    if problem.variables:
        var_lines = []
        for v in problem.variables[:10]:  # cap to prevent token explosion
            var_lines.append(
                f"  - {v.name}: {v.description} "
                f"(magnitude: {v.magnitude:.2f}, direction: {v.direction.value}, "
                f"confidence: {v.confidence:.2f})"
            )
        parts.append("VARIABLES:\n" + "\n".join(var_lines))

    # Include upstream findings (summarized)
    if current_outputs:
        upstream_parts = []
        for domain, output in current_outputs.items():
            if domain != target_domain and output.raw_analysis:
                summary = output.raw_analysis[:300]
                upstream_parts.append(f"[{domain.value}] {summary}")
        if upstream_parts:
            parts.append("UPSTREAM FINDINGS:\n" + "\n".join(upstream_parts))

    return "\n\n".join(parts)


def _summarize_domain_output(output: DomainOutput) -> str:
    """Summarize a domain output for Ke challenge input."""
    parts = [output.raw_analysis[:500]]

    if output.root_causes:
        rc_lines = []
        for rc in output.root_causes[:5]:
            rc_lines.append(
                f"  ROOT CAUSE: {rc.variable.name} (confidence: {rc.confidence:.2f}) "
                f"— {rc.variable.description[:100]}"
            )
        parts.append("ROOT CAUSES:\n" + "\n".join(rc_lines))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses. Sonnet often wraps JSON in ```json ... ```."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        start = 1
        end = len(lines)
        if lines[-1].strip() == "```":
            end = -1
        cleaned = "\n".join(lines[start:end]).strip()
    return cleaned


def _parse_domain_response(domain: Domain, response: LLMResponse) -> DomainOutput:
    """Parse an LLM response into a DomainOutput via bridge contract."""
    perspectives = []
    root_causes = []

    try:
        data = json.loads(_strip_code_fences(response.content))
        findings = data.get("findings", [])

        for finding in findings:
            # The LLM is required to specify which concept produced this finding.
            # Map it to the actual FrameworkID — fall back to domain primary only if missing.
            concept_str = finding.get("concept", "") or finding.get("type", "")
            framework = _concept_to_framework(concept_str, domain)

            var = Variable(
                name=finding.get("name", f"llm_{domain.value}_finding"),
                description=finding.get("description", ""),
                magnitude=float(finding.get("magnitude", 0.5)),
                direction=Direction(finding.get("direction", "neutral")),
                confidence=float(finding.get("confidence", 0.5)),
                source_framework=framework,
                is_hidden=bool(finding.get("is_hidden", False)) or
                          finding.get("label", "") in ("ROOT_CAUSE", "HIDDEN") or
                          finding.get("type", "") in ("ROOT_CAUSE", "BIAS_DETECTION"),
                is_user_stated=False,
                evidence=finding.get("evidence", []),
            )

            perspectives.append(Perspective(
                framework=framework,
                domain=domain,
                content=finding.get("description", ""),
                variables_found=[var],
                signal_type=SignalType.SIGNAL,
                weight=var.confidence,
            ))

            # High-confidence findings become root cause candidates
            label = finding.get("label", "")
            if label in ("ROOT_CAUSE", "VERIFIED") or var.confidence > 0.7:
                root_causes.append(RootCause(
                    variable=var,
                    evidence_chain=var.evidence,
                    confidence=var.confidence,
                    frameworks_that_agree=[framework],
                ))

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Unparseable response — create a minimal output
        perspectives.append(Perspective(
            framework=_domain_to_framework(domain),
            domain=domain,
            content=response.content[:500],
            variables_found=[],
            signal_type=SignalType.SIGNAL,
            weight=0.3,
        ))

    return DomainOutput(
        domain=domain,
        perspectives=perspectives,
        root_causes=root_causes,
        consequences=[],
        causal_loops=[],
        game_state=None,
        raw_analysis=response.content[:1000],
    )


def _parse_ke_response(
    challenger: Domain, target: Domain, response: LLMResponse
) -> ChallengeOutput:
    """Parse a Ke critic LLM response into a ChallengeOutput.

    Computes scrutiny_score from the 5-dimension evaluation if individual
    dimensions are present (more reliable than the LLM's stated score).
    """
    try:
        data = json.loads(_strip_code_fences(response.content))

        # Compute scrutiny from the 5 dimensions if available — more reliable
        # than trusting the LLM's stated overall score.
        dim_keys = [
            "evidence_gaps", "unexamined_assumptions", "missing_perspectives",
            "logical_coherence", "overconfidence",
        ]
        dim_scores: list[float] = []
        for k in dim_keys:
            v = data.get(k)
            if isinstance(v, dict) and "score" in v:
                try:
                    dim_scores.append(float(v["score"]))
                except (TypeError, ValueError):
                    pass
            elif isinstance(v, (int, float)):
                dim_scores.append(float(v))

        if len(dim_scores) >= 3:
            # Use computed average — bypasses LLM defaulting to round numbers
            scrutiny = round(sum(dim_scores) / len(dim_scores), 3)
        else:
            scrutiny = float(data.get("scrutiny_score", 0.3))

        # Build flags from dimension justifications when present
        flags = list(data.get("flags", []))
        for k in dim_keys:
            v = data.get(k)
            if isinstance(v, dict):
                score = v.get("score", 0)
                just = v.get("justification", "")
                if isinstance(score, (int, float)) and float(score) >= 0.5 and just:
                    flags.append(f"{k}: {just}")

        return ChallengeOutput(
            challenger_domain=challenger,
            target_domain=target,
            contradictions=data.get("contradictions", []),
            unsupported_claims=data.get("unsupported_claims", []),
            confidence_adjustments=data.get("confidence_adjustments", {}),
            scrutiny_score=max(0.0, min(1.0, scrutiny)),
            flags=flags[:8],
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ChallengeOutput(
            challenger_domain=challenger,
            target_domain=target,
            scrutiny_score=0.3,
            flags=["Ke response could not be parsed"],
        )


def _domain_to_framework(domain: Domain) -> FrameworkID:
    """Map a domain to its primary framework ID."""
    mapping = {
        Domain.PHYSICS: FrameworkID.FIRST_PRINCIPLES,
        Domain.MATHEMATICS: FrameworkID.BAYESIAN,
        Domain.PSYCHOLOGY: FrameworkID.DUAL_PROCESS,
        Domain.PHILOSOPHY: FrameworkID.ONTOLOGY,
        Domain.CHEMISTRY: FrameworkID.CATALYSIS,
    }
    return mapping.get(domain, FrameworkID.BAYESIAN)


# Concept name → FrameworkID lookup. Accepts the LLM's concept/type output
# in any reasonable form (case-insensitive, with or without underscores).
_CONCEPT_ALIASES: dict[str, FrameworkID] = {
    # Physics
    "first_principles": FrameworkID.FIRST_PRINCIPLES,
    "conservation": FrameworkID.CONSERVATION_OF_ENERGY,
    "conservation_of_energy": FrameworkID.CONSERVATION_OF_ENERGY,
    "entropy": FrameworkID.ENTROPY,
    "trajectory": FrameworkID.TRAJECTORY_MOMENTUM,
    "trajectory_momentum": FrameworkID.TRAJECTORY_MOMENTUM,
    "momentum": FrameworkID.TRAJECTORY_MOMENTUM,
    "potential_kinetic": FrameworkID.POTENTIAL_KINETIC,
    "potential_energy": FrameworkID.POTENTIAL_KINETIC,
    "equilibrium": FrameworkID.EQUILIBRIUM,
    "anomalous_motion": FrameworkID.ANOMALOUS_MOTION,
    "anomaly": FrameworkID.ANOMALOUS_MOTION,
    "socratic_squeeze": FrameworkID.SOCRATIC_SQUEEZE,
    "socratic": FrameworkID.SOCRATIC_SQUEEZE,
    "reference_frame": FrameworkID.REFERENCE_FRAME_SHIFT,
    "reference_frame_shift": FrameworkID.REFERENCE_FRAME_SHIFT,
    "entropy_leak": FrameworkID.ENTROPY_LEAK,
    "reductio": FrameworkID.REDUCTIO,
    "reductio_ad_absurdum": FrameworkID.REDUCTIO,
    # Mathematics
    "signal_noise": FrameworkID.SIGNAL_NOISE,
    "signal_vs_noise": FrameworkID.SIGNAL_NOISE,
    "category_theory": FrameworkID.CATEGORY_THEORY,
    "category": FrameworkID.CATEGORY_THEORY,
    "manifold": FrameworkID.MANIFOLD,
    "dimensional_reduction": FrameworkID.DIMENSIONAL_REDUCTION,
    "convergence": FrameworkID.CONVERGENCE,
    "bayesian": FrameworkID.BAYESIAN,
    "bayesian_inference": FrameworkID.BAYESIAN,
    "game_theory": FrameworkID.GAME_THEORY,
    "causal_loops": FrameworkID.CAUSAL_LOOPS,
    "causal_loop": FrameworkID.CAUSAL_LOOPS,
    "fragility": FrameworkID.FRAGILITY,
    "ergodicity": FrameworkID.FRAGILITY,
    "ergodicity_fragility": FrameworkID.FRAGILITY,
    # Psychology
    "dual_process": FrameworkID.DUAL_PROCESS,
    "system_1_2": FrameworkID.DUAL_PROCESS,
    "cognitive_dissonance": FrameworkID.COGNITIVE_DISSONANCE,
    "dissonance": FrameworkID.COGNITIVE_DISSONANCE,
    "motivated_reasoning": FrameworkID.MOTIVATED_REASONING,
    "dialectical_thinking": FrameworkID.DIALECTICAL_THINKING,
    "dialectical": FrameworkID.DIALECTICAL_THINKING,
    "metacognition": FrameworkID.METACOGNITION,
    # Philosophy
    "ontology": FrameworkID.ONTOLOGY,
    "ontological": FrameworkID.ONTOLOGY,
    "epistemology": FrameworkID.EPISTEMOLOGY,
    "epistemic": FrameworkID.EPISTEMOLOGY,
    "phenomenology": FrameworkID.PHENOMENOLOGY,
    "phenomenological": FrameworkID.PHENOMENOLOGY,
    "dialectics": FrameworkID.DIALECTICS,
    "teleology": FrameworkID.TELEOLOGY,
    "teleological": FrameworkID.TELEOLOGY,
    # Chemistry
    "self_assembly": FrameworkID.SELF_ASSEMBLY,
    "valence": FrameworkID.VALENCE,
    "chemical_equilibrium": FrameworkID.CHEMICAL_EQUILIBRIUM,
    "le_chateliers": FrameworkID.CHEMICAL_EQUILIBRIUM,
    "chirality": FrameworkID.CHIRALITY,
    "catalysis": FrameworkID.CATALYSIS,
    "catalyst": FrameworkID.CATALYSIS,
    "resonance": FrameworkID.RESONANCE,
}


def _concept_to_framework(concept_str: str, domain: Domain) -> FrameworkID:
    """
    Map an LLM-output concept name to a FrameworkID.
    Accepts case variations, spaces, and common synonyms.
    Falls back to the domain's primary framework only if unrecognized.
    """
    if not concept_str:
        return _domain_to_framework(domain)

    # Normalize: lowercase, strip, replace spaces/dashes with underscores
    key = concept_str.strip().lower().replace(" ", "_").replace("-", "_")

    if key in _CONCEPT_ALIASES:
        return _CONCEPT_ALIASES[key]

    # Try matching as a substring (e.g. "ontological_core" → "ontology")
    for alias, fw in _CONCEPT_ALIASES.items():
        if alias in key or key in alias:
            return fw

    return _domain_to_framework(domain)


# ---------------------------------------------------------------------------
# Trajectory building
# ---------------------------------------------------------------------------

def _build_trajectories(
    root_causes: list[RootCause],
    outputs: dict[Domain, DomainOutput],
) -> list[Trajectory]:
    """Build top 2-4 trajectories. NEVER a single answer."""
    if not root_causes:
        return []

    # Step 1: Exact-name deduplication (cheap)
    seen: set[str] = set()
    unique: list[RootCause] = []
    for rc in root_causes:
        if rc.variable.name not in seen:
            seen.add(rc.variable.name)
            unique.append(rc)

    # Step 2: Semantic deduplication across domains (Problem 7 fix)
    # Two root causes from different domains describing the same underlying
    # insight are merged into one with combined confidence and source domains.
    unique = _semantic_dedupe_root_causes(unique)

    unique.sort(key=lambda rc: rc.confidence, reverse=True)

    trajectories = []
    for rc in unique[:4]:
        cost = (
            f"If '{rc.variable.name}' is left unaddressed, consequences compound."
            if rc.variable.direction == Direction.NEGATIVE
            else f"Variable '{rc.variable.name}' continues unchecked."
        )
        trajectories.append(Trajectory(
            root_cause=rc,
            confidence=rc.confidence,
            consequences=[],
            cost_if_ignored=cost,
            source_domains=[fw.value for fw in rc.frameworks_that_agree],
        ))

    return trajectories


def _semantic_dedupe_root_causes(root_causes: list[RootCause]) -> list[RootCause]:
    """
    Cluster root causes by semantic similarity (TF-IDF cosine on description+evidence).
    Two findings from different domains describing the same underlying insight get
    merged: highest-confidence wins, frameworks combine, name describes the cluster.

    Threshold: 0.6 (looser than the 0.7 bond threshold — we want to dedupe more
    aggressively here since these are top-level findings going to the user).
    """
    if len(root_causes) <= 1:
        return root_causes

    try:
        from src.llm.semantic import _tokenize, _compute_idf, _tfidf_vector, _cosine_similarity
    except ImportError:
        return root_causes

    DEDUPE_THRESHOLD = 0.6

    # Build TF-IDF over the variables
    variables = [rc.variable for rc in root_causes]
    docs = [_tokenize(v) for v in variables]
    idf = _compute_idf(docs)
    vectors = [_tfidf_vector(v, idf) for v in variables]

    # Cluster: assign each rc to a cluster id
    cluster_id = list(range(len(root_causes)))  # everyone starts in own cluster
    for i in range(len(root_causes)):
        for j in range(i + 1, len(root_causes)):
            if cluster_id[i] == cluster_id[j]:
                continue
            sim = _cosine_similarity(vectors[i], vectors[j])
            if sim >= DEDUPE_THRESHOLD:
                # Merge cluster of j into cluster of i
                old_cluster = cluster_id[j]
                new_cluster = cluster_id[i]
                for k in range(len(cluster_id)):
                    if cluster_id[k] == old_cluster:
                        cluster_id[k] = new_cluster

    # Build clusters
    clusters: dict[int, list[RootCause]] = {}
    for idx, cid in enumerate(cluster_id):
        clusters.setdefault(cid, []).append(root_causes[idx])

    # Merge each cluster into a single root cause
    merged: list[RootCause] = []
    for cluster in clusters.values():
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue

        # Pick the highest-confidence rc as the anchor
        anchor = max(cluster, key=lambda rc: rc.confidence)

        # Combine all unique frameworks
        all_frameworks: list[FrameworkID] = []
        seen_fw: set[str] = set()
        for rc in cluster:
            for fw in rc.frameworks_that_agree:
                if fw.value not in seen_fw:
                    all_frameworks.append(fw)
                    seen_fw.add(fw.value)

        # Combine evidence
        all_evidence: list[str] = []
        for rc in cluster:
            all_evidence.extend(rc.evidence_chain)

        # Boost confidence slightly for cross-domain agreement (capped)
        boost = min(0.1 * (len(cluster) - 1), 0.15)
        merged_confidence = min(anchor.confidence + boost, 0.99)

        merged_var = Variable(
            name=anchor.variable.name,
            description=anchor.variable.description,
            magnitude=anchor.variable.magnitude,
            direction=anchor.variable.direction,
            confidence=merged_confidence,
            source_framework=anchor.variable.source_framework,
            is_hidden=any(rc.variable.is_hidden for rc in cluster),
            is_user_stated=False,
            evidence=all_evidence[:8],
        )

        merged.append(RootCause(
            variable=merged_var,
            evidence_chain=all_evidence[:8],
            confidence=merged_confidence,
            frameworks_that_agree=all_frameworks,
            bias_that_hid_it=anchor.bias_that_hid_it,
        ))

    return merged


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_field(
    outputs: dict[Domain, DomainOutput],
    domain: Domain,
    field_name: str,
    default: str,
) -> str:
    """Extract a field from a domain's raw analysis."""
    output = outputs.get(domain)
    if output and output.raw_analysis:
        try:
            data = json.loads(_strip_code_fences(output.raw_analysis))
            val = data.get(field_name)
            if val:
                return str(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return default


def _build_bias_summary(
    trajectories: list[Trajectory],
    outputs: dict[Domain, DomainOutput],
) -> str:
    """Build bias summary from root causes and domain outputs."""
    parts = []
    for t in trajectories[:2]:
        if t.root_cause.bias_that_hid_it:
            parts.append(t.root_cause.bias_that_hid_it)
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
            f"Did not fully converge in {convergence.total_iterations} iterations."
        )
    high_ke = [ke for ke in ke_results if ke.scrutiny_score > 0.5]
    if high_ke:
        pairs = [f"{ke.challenger_domain.value}→{ke.target_domain.value}" for ke in high_ke]
        parts.append(f"{len(high_ke)} pair(s) under high scrutiny: {', '.join(pairs)}.")
    total_cached = sum(f.variables_cached for f in funnel_history)
    if total_cached > 0:
        parts.append(f"{total_cached} variables cached for deeper analysis.")
    return " ".join(parts) if parts else "Converged with no significant uncertainty."
