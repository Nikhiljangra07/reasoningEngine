"""
Funnel Mechanism — The Feedback Loop Between Iterations.

After each dual-cycle pass (Sheng + Ke), the funnel runs.
It does two things:
1. Filters what flows downstream to the next iteration
2. Caches what gets filtered out for future use

CRITICAL RULES:
- Filter by CONNECTION DENSITY, never by comfort or convergence direction.
- A variable gets filtered out ONLY if it has ZERO connections to ANY
  domain's findings after two full passes.
- If a variable connects to even ONE domain — it stays downstream.
- Contradictions are SIGNAL, not noise. They may be where Variable D hides.
- High Ke scrutiny (>0.5) means NEEDS MORE WORK — keep it, force another pass.
- Low Ke scrutiny (<0.2) means survived challenge — stable, higher confidence.
- Zero connections after 2 passes = genuine noise → cache.
- Variable cap per iteration (default 30) bounds O(n²) without killing depth.

ISOLATION: Imports ONLY from src.core.types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    ChallengeOutput,
    Domain,
    DomainOutput,
    Perspective,
    SignalType,
    Variable,
)


# Maximum variables flowing downstream per iteration.
# Bounds O(n²) in Category Theory, Causal Loops, Cognitive Dissonance.
# Connection density determines which 30 survive — not convergence direction.
VARIABLE_CAP = 30

# Ke scrutiny thresholds
KE_HIGH_SCRUTINY = 0.5      # needs more work — keep downstream, force pass
KE_LOW_SCRUTINY = 0.2       # survived challenge — stable, boost confidence
KE_ZERO_CONNECTION_PASSES = 2  # must survive 2 full passes with zero connections before filtering


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConnectionScore:
    """Connection density for a single variable."""
    variable_name: str
    variable: Variable
    connections: int                 # how many domain outputs reference this variable
    connected_domains: list[str]     # which domains connect to it
    ke_status: str                   # "needs_work", "stable", "unchallenged"
    ke_scrutiny: float               # average Ke scrutiny on domains that found this variable
    zero_connection_passes: int      # how many consecutive passes with zero connections


@dataclass
class FunnelResult:
    """Result of a funnel pass."""
    downstream_variables: list[Variable]        # what flows to next iteration (capped)
    downstream_perspectives: list[Perspective]   # perspectives that survived
    cached_variables: list[Variable]             # filtered out — stored for future
    variables_kept: int
    variables_cached: int
    variables_needing_work: int                 # high Ke scrutiny — forced another pass
    variables_stable: int                       # low Ke scrutiny — high confidence


@dataclass
class CachedCombination:
    """A filtered-out variable stored for future use."""
    variable: Variable
    problem_context: str            # what problem produced this
    result_produced: str            # what this variable contributed
    iteration_cached: int           # when it was filtered out
    connection_score_at_cache: int  # its connection density when cached


# ---------------------------------------------------------------------------
# The Funnel
# ---------------------------------------------------------------------------

def run_funnel(
    domain_outputs: dict[Domain, DomainOutput],
    ke_results: list[ChallengeOutput],
    previous_connection_scores: dict[str, ConnectionScore] | None,
    iteration: int,
    problem_statement: str,
) -> FunnelResult:
    """
    Run the funnel after a dual-cycle pass.

    Filters variables by connection density + Ke scrutiny.
    Caps at VARIABLE_CAP. Caches everything else.
    """
    # Step 1: Collect ALL variables from all domain outputs
    all_variables = _collect_all_variables(domain_outputs)

    # Step 2: Score connection density for each variable
    connection_scores = _score_connections(
        all_variables, domain_outputs, ke_results, previous_connection_scores
    )

    # Step 3: Apply Ke-driven classification
    _apply_ke_classification(connection_scores, ke_results, domain_outputs)

    # Step 4: Filter by connection density
    # Keep: anything with connections >= 1 OR ke_status == "needs_work"
    # Cache: zero connections for KE_ZERO_CONNECTION_PASSES consecutive passes
    downstream = []
    to_cache = []

    for name, cs in connection_scores.items():
        if cs.connections == 0 and cs.zero_connection_passes >= KE_ZERO_CONNECTION_PASSES:
            # Genuine noise — cache it
            to_cache.append(cs)
        else:
            # Has connections OR hasn't had enough zero-connection passes — keep
            downstream.append(cs)

    # Step 5: Cap at VARIABLE_CAP
    # Sort by connection density (highest first). Keep top N.
    # Variables with ke_status == "needs_work" get priority (sorted first).
    downstream.sort(
        key=lambda cs: (
            cs.ke_status == "needs_work",  # True sorts after False... invert
            cs.connections,
            cs.ke_scrutiny,
        ),
        reverse=True,
    )

    # Separate needs_work (always keep) from the rest
    needs_work = [cs for cs in downstream if cs.ke_status == "needs_work"]
    rest = [cs for cs in downstream if cs.ke_status != "needs_work"]

    # needs_work always stays. Fill remaining cap with highest-connection rest.
    remaining_cap = max(VARIABLE_CAP - len(needs_work), 0)
    rest_kept = rest[:remaining_cap]
    rest_cached = rest[remaining_cap:]

    final_downstream = needs_work + rest_kept
    to_cache.extend(rest_cached)

    # Step 6: Build output
    downstream_vars = [cs.variable for cs in final_downstream]
    cached_vars = [cs.variable for cs in to_cache]

    # Adjust confidence based on Ke status
    for cs in final_downstream:
        if cs.ke_status == "stable":
            # Survived Ke challenge — boost confidence (cap at 0.95)
            cs.variable.confidence = min(cs.variable.confidence * 1.1, 0.95)
        elif cs.ke_status == "needs_work":
            # High Ke scrutiny — don't lower confidence, but flag for re-examination
            pass  # confidence stays — the variable needs more analysis, not less trust

    # Build downstream perspectives — keep perspectives that contain surviving variables
    surviving_names = {cs.variable_name for cs in final_downstream}
    downstream_perspectives = _filter_perspectives(domain_outputs, surviving_names)

    return FunnelResult(
        downstream_variables=downstream_vars,
        downstream_perspectives=downstream_perspectives,
        cached_variables=cached_vars,
        variables_kept=len(final_downstream),
        variables_cached=len(to_cache),
        variables_needing_work=len(needs_work),
        variables_stable=sum(1 for cs in final_downstream if cs.ke_status == "stable"),
    )


# ---------------------------------------------------------------------------
# Connection density scoring
# ---------------------------------------------------------------------------

def _collect_all_variables(
    domain_outputs: dict[Domain, DomainOutput],
) -> dict[str, Variable]:
    """Collect all unique variables across all domain outputs."""
    variables: dict[str, Variable] = {}
    for output in domain_outputs.values():
        for perspective in output.perspectives:
            for var in perspective.variables_found:
                # Keep the highest-confidence version if duplicate names
                if var.name not in variables or var.confidence > variables[var.name].confidence:
                    variables[var.name] = var
    return variables


def _score_connections(
    all_variables: dict[str, Variable],
    domain_outputs: dict[Domain, DomainOutput],
    ke_results: list[ChallengeOutput],
    previous_scores: dict[str, ConnectionScore] | None,
) -> dict[str, ConnectionScore]:
    """
    Score connection density for each variable.

    A variable's connection count = how many DIFFERENT domains reference it
    (by name in their variables, evidence chains, or descriptions).
    """
    scores: dict[str, ConnectionScore] = {}

    for var_name, var in all_variables.items():
        connected_domains = []

        for domain, output in domain_outputs.items():
            # Check if this domain references this variable
            if _domain_references_variable(output, var_name):
                connected_domains.append(domain.value)

        # Track zero-connection passes from previous iteration
        prev_zero = 0
        if previous_scores and var_name in previous_scores:
            if len(connected_domains) == 0:
                prev_zero = previous_scores[var_name].zero_connection_passes + 1
            # else: reset to 0 (it got connected again)

        scores[var_name] = ConnectionScore(
            variable_name=var_name,
            variable=var,
            connections=len(connected_domains),
            connected_domains=connected_domains,
            ke_status="unchallenged",
            ke_scrutiny=0.0,
            zero_connection_passes=prev_zero if len(connected_domains) == 0 else 0,
        )

    return scores


def _domain_references_variable(output: DomainOutput, var_name: str) -> bool:
    """Check if a domain output references a variable (by name or semantically)."""
    var_name_lower = var_name.lower().replace("_", " ")

    for perspective in output.perspectives:
        # Direct: variable with this name exists
        for v in perspective.variables_found:
            if v.name == var_name:
                return True

            # Semantic: variable description or evidence references this name
            combined_text = v.description.lower() + " ".join(v.evidence).lower()
            if var_name_lower in combined_text:
                return True

    # Check root causes
    for rc in output.root_causes:
        if rc.variable.name == var_name:
            return True
        combined = rc.variable.description.lower() + " ".join(rc.evidence_chain).lower()
        if var_name_lower in combined:
            return True

    # Check raw analysis
    if var_name_lower in output.raw_analysis.lower():
        return True

    return False


# ---------------------------------------------------------------------------
# Ke classification
# ---------------------------------------------------------------------------

def _apply_ke_classification(
    connection_scores: dict[str, ConnectionScore],
    ke_results: list[ChallengeOutput],
    domain_outputs: dict[Domain, DomainOutput],
) -> None:
    """
    Apply Ke scrutiny scores to classify variables.

    - High scrutiny (>0.5) on the domain that found this variable = needs_work
    - Low scrutiny (<0.2) on the domain that found it = stable
    """
    # Build a map: domain → its Ke scrutiny score (when it was the TARGET)
    domain_scrutiny: dict[str, float] = {}
    for ke in ke_results:
        domain_scrutiny[ke.target_domain.value] = ke.scrutiny_score

        # Also check if specific variables were called out
        for var_name, adjusted_conf in ke.confidence_adjustments.items():
            if var_name in connection_scores:
                cs = connection_scores[var_name]
                cs.ke_scrutiny = max(cs.ke_scrutiny, ke.scrutiny_score)
                if ke.scrutiny_score > KE_HIGH_SCRUTINY:
                    cs.ke_status = "needs_work"

    # For variables not directly challenged, inherit their source domain's scrutiny
    for name, cs in connection_scores.items():
        if cs.ke_status == "unchallenged":
            source_domain = cs.variable.source_framework.value.split("_")[0]
            # Try to find the domain this framework belongs to
            for domain_name, scrutiny in domain_scrutiny.items():
                if _framework_belongs_to_domain(cs.variable.source_framework.value, domain_name):
                    cs.ke_scrutiny = scrutiny
                    if scrutiny > KE_HIGH_SCRUTINY:
                        cs.ke_status = "needs_work"
                    elif scrutiny < KE_LOW_SCRUTINY:
                        cs.ke_status = "stable"
                    else:
                        cs.ke_status = "unchallenged"
                    break


def _framework_belongs_to_domain(framework_value: str, domain_value: str) -> bool:
    """Check if a framework ID belongs to a domain (rough matching)."""
    # Physics frameworks
    physics_frameworks = {
        "first_principles", "conservation_of_energy", "entropy",
        "trajectory_momentum", "potential_kinetic", "equilibrium",
        "anomalous_motion", "socratic_squeeze", "reference_frame_shift",
        "entropy_leak", "reductio_ad_absurdum",
    }
    # Maths frameworks
    maths_frameworks = {
        "signal_noise", "category_theory", "manifold", "dimensional_reduction",
        "convergence", "bayesian_inference", "game_theory", "causal_loops",
        "ergodicity_fragility",
    }
    # Psychology frameworks
    psych_frameworks = {
        "dual_process", "cognitive_dissonance", "motivated_reasoning",
        "dialectical_thinking", "metacognition",
    }
    # Philosophy frameworks
    phil_frameworks = {
        "ontology", "epistemology", "phenomenology", "dialectics", "teleology",
    }
    # Chemistry frameworks
    chem_frameworks = {
        "self_assembly", "valence", "chemical_equilibrium",
        "chirality", "catalysis", "resonance",
    }

    domain_map = {
        "physics": physics_frameworks,
        "mathematics": maths_frameworks,
        "psychology": psych_frameworks,
        "philosophy": phil_frameworks,
        "chemistry": chem_frameworks,
    }

    return framework_value in domain_map.get(domain_value, set())


# ---------------------------------------------------------------------------
# Perspective filtering
# ---------------------------------------------------------------------------

def _filter_perspectives(
    domain_outputs: dict[Domain, DomainOutput],
    surviving_names: set[str],
) -> list[Perspective]:
    """Keep perspectives that contain at least one surviving variable."""
    kept = []
    for output in domain_outputs.values():
        for perspective in output.perspectives:
            has_survivor = any(
                v.name in surviving_names for v in perspective.variables_found
            )
            if has_survivor or not perspective.variables_found:
                kept.append(perspective)
    return kept
