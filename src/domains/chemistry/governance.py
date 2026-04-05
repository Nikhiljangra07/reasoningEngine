"""
Chemistry Module A: Governance (Runs BEFORE the battlefield opens)

Sets the formation. Decides which domains activate, which concepts bond,
and how the system absorbs stress. This is the inner-layer governance
that manages signal integrity between domain bridges.

3 Concepts:
1. Self-Assembly — creates conditions for data to naturally organize into structure
2. Valence — determines bonding compatibility between concepts/outputs
3. Chemical Equilibrium (Le Chatelier's) — absorbs stress when new heavy variables enter

ISOLATION: Imports ONLY from src.core.types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    BondType,
    Direction,
    Domain,
    DomainOutput,
    FrameworkID,
    Perspective,
    Problem,
    SignalType,
    Variable,
)


# ===========================================================================
# Concept 1: Self-Assembly (The Formation)
# ===========================================================================

@dataclass
class AffinityCluster:
    """A natural grouping of variables that belong together."""
    name: str
    variables: list[str]                # variable names in this cluster
    affinity_type: str                  # "semantic", "causal", "temporal", "directional"
    strength: float                     # 0.0 to 1.0


@dataclass
class FormationPlan:
    """The governance decision: what activates, what's inert."""
    structural_affinity_clusters: list[AffinityCluster]
    organizational_template: str        # "linear", "web", "tree", "cycle", "hub_and_spoke"
    misfit_variables: list[str]         # variables that don't fit any cluster
    active_domains: list[Domain]
    active_concepts_per_domain: dict[str, list[str]]
    estimated_agent_count: int


def run_self_assembly(problem: Problem) -> tuple[Perspective, FormationPlan]:
    """
    Self-Assembly — creates the environment where data points naturally
    organize into structured, functional units.

    Rather than forcing a solution, it sets up conditions where the
    "atoms" (perspectives, variables, findings) click into place
    because of their logical shapes.
    """
    variables_found = []
    user_vars = problem.variables

    # Step 1: Find natural clusters based on structural affinity
    clusters = _find_affinity_clusters(user_vars)

    # Step 2: Determine organizational template
    template = _determine_template(clusters, user_vars)

    # Step 3: Detect misfits — variables that don't fit any cluster
    clustered_names = set()
    for c in clusters:
        clustered_names.update(c.variables)
    misfits = [v.name for v in user_vars if v.name not in clustered_names]

    # Step 4: Determine which domains and concepts should activate
    active_domains = _determine_active_domains(user_vars, clusters)
    active_concepts = _determine_active_concepts(user_vars, clusters, active_domains)
    agent_count = _estimate_agent_count(active_domains, active_concepts)

    plan = FormationPlan(
        structural_affinity_clusters=clusters,
        organizational_template=template,
        misfit_variables=misfits,
        active_domains=active_domains,
        active_concepts_per_domain=active_concepts,
        estimated_agent_count=agent_count,
    )

    # Misfits are significant — they might be the most important hidden variables
    if misfits:
        variables_found.append(Variable(
            name="self_assembly_misfits",
            description=(
                f"{len(misfits)} variable(s) don't fit any natural cluster: "
                f"{', '.join(misfits)}. "
                "Misfits are either noise OR the most important hidden variables — "
                "they don't fit because the current structure can't accommodate them."
            ),
            magnitude=min(len(misfits) / max(len(user_vars), 1), 1.0),
            direction=Direction.NEUTRAL,
            confidence=0.6,
            source_framework=FrameworkID.SELF_ASSEMBLY,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Misfits: {', '.join(misfits)}",
                f"Clusters formed: {len(clusters)}",
                f"Template: {template}",
                "Misfits don't fit the natural structure of the problem.",
            ],
        ))

    content = (
        "SELF-ASSEMBLY (FORMATION PLAN)\n"
        f"Organizational template: {template}\n"
        f"Clusters found: {len(clusters)}\n"
        f"Misfit variables: {len(misfits)}\n"
        f"Active domains: {[d.value for d in active_domains]}\n"
        f"Estimated agents: {agent_count}\n\n"
    )
    for c in clusters:
        content += f"  Cluster '{c.name}': {', '.join(c.variables)} ({c.affinity_type}, strength: {c.strength:.2f})\n"
    if misfits:
        content += f"\n  MISFITS (potential hidden variables): {', '.join(misfits)}\n"

    perspective = Perspective(
        framework=FrameworkID.SELF_ASSEMBLY,
        domain=Domain.CHEMISTRY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )

    return perspective, plan


def _find_affinity_clusters(variables: list[Variable]) -> list[AffinityCluster]:
    """Find natural groupings based on structural affinity."""
    clusters = []

    # Cluster by direction (same direction = directional affinity)
    direction_groups: dict[Direction, list[str]] = {}
    for v in variables:
        if v.direction not in direction_groups:
            direction_groups[v.direction] = []
        direction_groups[v.direction].append(v.name)

    for direction, names in direction_groups.items():
        if len(names) >= 2:
            # Calculate average magnitude for strength
            mags = [v.magnitude for v in variables if v.name in names]
            strength = sum(mags) / len(mags) if mags else 0.5

            clusters.append(AffinityCluster(
                name=f"{direction.value}_forces",
                variables=names,
                affinity_type="directional",
                strength=strength,
            ))

    # Cluster by magnitude proximity (similar weight = causal affinity)
    sorted_vars = sorted(variables, key=lambda v: v.magnitude, reverse=True)
    high_mag = [v.name for v in sorted_vars if v.magnitude > 0.7]
    low_mag = [v.name for v in sorted_vars if v.magnitude < 0.4]

    if len(high_mag) >= 2:
        clusters.append(AffinityCluster(
            name="high_impact_forces",
            variables=high_mag,
            affinity_type="causal",
            strength=0.8,
        ))

    if len(low_mag) >= 2:
        clusters.append(AffinityCluster(
            name="background_forces",
            variables=low_mag,
            affinity_type="causal",
            strength=0.4,
        ))

    return clusters


def _determine_template(
    clusters: list[AffinityCluster], variables: list[Variable]
) -> str:
    """Determine the natural organizational structure of the problem."""
    direction_set = {v.direction for v in variables}

    # If circular forces present → cycle
    if Direction.CIRCULAR in direction_set:
        return "cycle"

    # If one dominant cluster → hub_and_spoke
    if clusters:
        max_cluster = max(clusters, key=lambda c: len(c.variables))
        if len(max_cluster.variables) > len(variables) * 0.6:
            return "hub_and_spoke"

    # If clear positive vs negative split → linear (two poles)
    pos = sum(1 for v in variables if v.direction == Direction.POSITIVE)
    neg = sum(1 for v in variables if v.direction == Direction.NEGATIVE)
    if pos > 0 and neg > 0 and abs(pos - neg) < 2:
        return "linear"

    # If many variables with complex relationships → web
    if len(variables) > 5:
        return "web"

    # Default → tree
    return "tree"


def _determine_active_domains(
    variables: list[Variable], clusters: list[AffinityCluster]
) -> list[Domain]:
    """Determine which domains should activate for this problem."""
    # Physics and Maths always activate (they're the foundation)
    active = [Domain.PHYSICS, Domain.MATHEMATICS]

    # Psychology activates if there are user-stated variables (always for human problems)
    if any(v.is_user_stated for v in variables):
        active.append(Domain.PSYCHOLOGY)

    # Philosophy activates if there are assumptions or conflicting directions
    directions = {v.direction for v in variables}
    has_conflict = Direction.POSITIVE in directions and Direction.NEGATIVE in directions
    has_low_confidence = any(v.confidence < 0.5 for v in variables)
    if has_conflict or has_low_confidence:
        active.append(Domain.PHILOSOPHY)

    # Chemistry always activates (it's governance + analytical)
    active.append(Domain.CHEMISTRY)

    return active


def _determine_active_concepts(
    variables: list[Variable],
    clusters: list[AffinityCluster],
    active_domains: list[Domain],
) -> dict[str, list[str]]:
    """Determine which specific concepts should activate per domain."""
    concepts: dict[str, list[str]] = {}

    # Physics — always full activation (both phases needed for any human problem)
    if Domain.PHYSICS in active_domains:
        concepts["physics"] = [
            "first_principles", "conservation_of_energy", "entropy",
            "trajectory_momentum", "potential_kinetic", "equilibrium",
            "anomalous_motion", "socratic_squeeze", "reference_frame_shift",
            "entropy_leak", "reductio_ad_absurdum",
        ]

    # Mathematics — core layers always, conditional layers based on problem
    if Domain.MATHEMATICS in active_domains:
        core = [
            "signal_noise", "category_theory", "manifold",
            "convergence", "bayesian_inference",
        ]
        # Game theory only if multiple agents detected
        agent_keywords = [
            "partner", "cofounder", "boss", "team", "company",
            "competitor", "spouse", "parent", "friend",
        ]
        has_agents = any(
            any(kw in v.description.lower() or kw in v.name.lower() for kw in agent_keywords)
            for v in variables
        )
        if has_agents:
            core.append("game_theory")

        # Causal loops if circular direction detected
        if Direction.CIRCULAR in {v.direction for v in variables}:
            core.append("causal_loops")
        else:
            core.append("causal_loops")  # always useful for human problems

        core.append("ergodicity_fragility")  # always needed as final gate
        concepts["mathematics"] = core

    # Psychology — always full activation for human problems
    if Domain.PSYCHOLOGY in active_domains:
        concepts["psychology"] = [
            "dual_process", "cognitive_dissonance", "motivated_reasoning",
            "dialectical_thinking", "metacognition",
        ]

    # Philosophy — full pipeline when activated
    if Domain.PHILOSOPHY in active_domains:
        concepts["philosophy"] = [
            "ontology", "epistemology", "phenomenology",
            "dialectics", "teleology",
        ]

    # Chemistry — governance always, analytical always when chemistry is active
    if Domain.CHEMISTRY in active_domains:
        concepts["chemistry"] = [
            "self_assembly", "valence", "chemical_equilibrium",
            "chirality", "catalysis", "resonance",
        ]

    return concepts


def _estimate_agent_count(
    active_domains: list[Domain],
    active_concepts: dict[str, list[str]],
) -> int:
    """Estimate how many parallel agents will run."""
    # Each domain is at least 1 agent; complex domains may spawn sub-agents
    total = 0
    for domain_name, concepts in active_concepts.items():
        # Base: 1 agent per domain
        total += 1
        # If more than 5 concepts, add sub-agents
        if len(concepts) > 5:
            total += 1
    return total


# ===========================================================================
# Concept 2: Valence (The Compatibility)
# ===========================================================================

@dataclass
class BondAssessment:
    """Assessment of bonding compatibility between two outputs."""
    output_a_name: str
    output_b_name: str
    polarity_a: str                     # "positive" or "negative"
    polarity_b: str
    shared_electrons: list[str]         # shared variables/facts
    bond_type: BondType
    bond_strength: float                # 0.0 to 1.0


def run_valence(
    output_a: DomainOutput,
    output_b: DomainOutput,
) -> BondAssessment:
    """
    Valence — determines bonding compatibility between two domain outputs.

    Every output has a "valence" — its capacity to share or exchange
    logical connections with other outputs.

    Bond types:
    - IONIC: opposites held by attraction (e.g., Physics force + Psychology emotion)
    - COVALENT: similar outputs sharing a common variable equally
    - NONE: genuinely unrelated for this problem
    """
    # Collect variables from both outputs
    vars_a = []
    for p in output_a.perspectives:
        vars_a.extend(p.variables_found)
    vars_b = []
    for p in output_b.perspectives:
        vars_b.extend(p.variables_found)

    # Determine polarity
    polarity_a = _determine_polarity(vars_a)
    polarity_b = _determine_polarity(vars_b)

    # Find shared electrons using semantic matching (TF-IDF cosine similarity)
    # Falls back to exact name matching + description word overlap
    try:
        from src.llm.semantic import find_semantic_matches, matches_to_shared_electrons
        matches = find_semantic_matches(vars_a, vars_b)
        shared = matches_to_shared_electrons(matches)
    except ImportError:
        # Fallback: use the built-in matching
        shared = _find_shared_electrons(vars_a, vars_b)

    # Determine bond type
    bond_type = _determine_bond_type(polarity_a, polarity_b, shared, vars_a, vars_b)

    # Calculate bond strength
    strength = _calculate_bond_strength(shared, vars_a, vars_b, bond_type)

    return BondAssessment(
        output_a_name=output_a.domain.value,
        output_b_name=output_b.domain.value,
        polarity_a=polarity_a,
        polarity_b=polarity_b,
        shared_electrons=shared,
        bond_type=bond_type,
        bond_strength=strength,
    )


def _determine_polarity(variables: list[Variable]) -> str:
    """Is this output electropositive (providing support) or electronegative (pulling)?"""
    if not variables:
        return "neutral"

    positive_weight = sum(v.magnitude for v in variables if v.direction == Direction.POSITIVE)
    negative_weight = sum(v.magnitude for v in variables if v.direction == Direction.NEGATIVE)

    if positive_weight > negative_weight * 1.3:
        return "positive"
    elif negative_weight > positive_weight * 1.3:
        return "negative"
    return "neutral"


def _find_shared_electrons(
    vars_a: list[Variable], vars_b: list[Variable]
) -> list[str]:
    """
    Find variables/facts shared between two outputs — the 'shared electrons'.

    Uses SEMANTIC SIMILARITY on descriptions, not just name matching.
    Two domains can describe the same underlying variable with completely
    different names. If two variables describe the same underlying reality
    with different terminology, they share an electron.

    Primary: cosine similarity on description word sets.
    Fallback: exact name matching.
    """
    shared: list[str] = []
    already_matched: set[str] = set()

    # Layer 1: Exact name matches (fast fallback)
    names_a = {v.name for v in vars_a}
    names_b = {v.name for v in vars_b}
    for name in names_a & names_b:
        shared.append(name)
        already_matched.add(name)

    # Layer 2: Semantic similarity on descriptions
    # For each pair across domains, compute word-overlap cosine similarity.
    # If similarity > threshold, they describe the same underlying reality.
    SEMANTIC_THRESHOLD = 0.25  # 25% word overlap = same underlying variable

    for va in vars_a:
        if va.name in already_matched:
            continue
        va_words = _description_words(va)
        if not va_words:
            continue

        for vb in vars_b:
            if vb.name in already_matched:
                continue
            vb_words = _description_words(vb)
            if not vb_words:
                continue

            # Cosine similarity via word overlap (Jaccard-like)
            intersection = len(va_words & vb_words)
            union = len(va_words | vb_words)
            similarity = intersection / union if union > 0 else 0.0

            if similarity > SEMANTIC_THRESHOLD:
                bond_name = f"{va.name}≈{vb.name}"
                if bond_name not in shared:
                    shared.append(bond_name)
                    already_matched.add(va.name)
                    already_matched.add(vb.name)
                break  # one match per va variable

    # Layer 3: Cross-reference in evidence chains
    for va in vars_a:
        if va.name in already_matched:
            continue
        for vb in vars_b:
            if vb.name in already_matched:
                continue
            va_text = va.description.lower() + " ".join(va.evidence).lower()
            vb_text = vb.description.lower() + " ".join(vb.evidence).lower()

            vb_name_readable = vb.name.lower().replace("_", " ")
            va_name_readable = va.name.lower().replace("_", " ")

            if vb_name_readable in va_text:
                ref_name = f"{va.name}→{vb.name}"
                if ref_name not in shared:
                    shared.append(ref_name)
                    already_matched.add(va.name)
                break
            elif va_name_readable in vb_text:
                ref_name = f"{vb.name}→{va.name}"
                if ref_name not in shared:
                    shared.append(ref_name)
                    already_matched.add(vb.name)
                break

    return shared


def _description_words(var: Variable) -> set[str]:
    """Extract meaningful words from a variable's description + evidence for semantic matching."""
    # Combine description and evidence into one text
    text = var.description.lower()
    if var.evidence:
        text += " " + " ".join(var.evidence).lower()

    # Split and filter stopwords (lightweight — no NLP dependency)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "to", "for",
        "with", "on", "at", "from", "by", "about", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "and", "but",
        "or", "not", "no", "so", "if", "then", "than", "that", "this", "it",
        "its", "they", "their", "them", "we", "our", "you", "your", "he",
        "she", "his", "her", "which", "what", "where", "when", "how", "all",
        "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "only", "very", "just", "also", "any",
    }

    words = set()
    for word in text.split():
        # Strip punctuation
        cleaned = word.strip(".,;:!?()[]{}\"'`-—")
        if len(cleaned) > 2 and cleaned not in stopwords:
            words.add(cleaned)

    return words


def _determine_bond_type(
    polarity_a: str, polarity_b: str,
    shared: list[str],
    vars_a: list[Variable], vars_b: list[Variable],
) -> BondType:
    """Determine the type of bond between two outputs."""
    if not shared:
        return BondType.NONE

    # Ionic: opposite polarities with shared electrons
    if polarity_a != polarity_b and polarity_a != "neutral" and polarity_b != "neutral":
        return BondType.IONIC

    # Covalent: same polarity sharing variables equally
    if shared:
        return BondType.COVALENT

    return BondType.NONE


def _calculate_bond_strength(
    shared: list[str],
    vars_a: list[Variable], vars_b: list[Variable],
    bond_type: BondType,
) -> float:
    """Calculate the strength of the bond."""
    if bond_type == BondType.NONE:
        return 0.0

    # Base: shared electron count relative to total
    total_unique = len({v.name for v in vars_a} | {v.name for v in vars_b})
    shared_ratio = len(shared) / max(total_unique, 1)

    # Ionic bonds get a bonus (opposites attracting = stronger connection)
    type_bonus = 0.15 if bond_type == BondType.IONIC else 0.0

    # Confidence alignment
    avg_conf_a = sum(v.confidence for v in vars_a) / max(len(vars_a), 1)
    avg_conf_b = sum(v.confidence for v in vars_b) / max(len(vars_b), 1)
    confidence_factor = (avg_conf_a + avg_conf_b) / 2

    return min(shared_ratio * 0.5 + confidence_factor * 0.35 + type_bonus, 1.0)


# ===========================================================================
# Concept 3: Chemical Equilibrium / Le Chatelier's Principle
# ===========================================================================

@dataclass
class EquilibriumShift:
    """Result of absorbing a new variable into the system."""
    stress_magnitude: str               # "low", "medium", "high"
    domains_to_rerun: list[Domain]
    bonds_to_reevaluate: list[str]      # pairs of domain names
    new_equilibrium_reached: bool
    cascade_risk_score: float           # 0.0 to 1.0


def run_chemical_equilibrium(
    new_variable: Variable,
    current_outputs: dict[Domain, DomainOutput],
) -> tuple[Perspective, EquilibriumShift]:
    """
    Le Chatelier's Principle — absorbs stress when a new heavy variable enters.

    When Variable D is discovered or a domain produces a finding that
    changes the picture, the entire system shifts. This ensures the
    system finds a new stable state rather than breaking.
    """
    variables_found = []

    # Assess stress magnitude
    stress = _assess_stress(new_variable, current_outputs)

    # Determine shift direction
    domains_to_rerun, bonds_to_reeval = _determine_shift(
        new_variable, stress, current_outputs
    )

    # Check cascade risk
    cascade_risk = _assess_cascade_risk(stress, domains_to_rerun, current_outputs)

    # Check if new equilibrium is achievable
    equilibrium_reached = stress != "high" or cascade_risk < 0.7

    shift = EquilibriumShift(
        stress_magnitude=stress,
        domains_to_rerun=domains_to_rerun,
        bonds_to_reevaluate=bonds_to_reeval,
        new_equilibrium_reached=equilibrium_reached,
        cascade_risk_score=cascade_risk,
    )

    if stress in ("medium", "high"):
        variables_found.append(Variable(
            name=f"equilibrium_shift_{new_variable.name}",
            description=(
                f"Le Chatelier's: new variable '{new_variable.name}' "
                f"(magnitude: {new_variable.magnitude:.2f}) caused {stress} stress. "
                f"{len(domains_to_rerun)} domain(s) need to re-run. "
                f"Cascade risk: {cascade_risk:.2f}. "
                f"New equilibrium: {'reached' if equilibrium_reached else 'NOT reached — system unstable'}."
            ),
            magnitude=new_variable.magnitude,
            direction=Direction.NEUTRAL,
            confidence=0.7,
            source_framework=FrameworkID.CHEMICAL_EQUILIBRIUM,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Stress: {stress}",
                f"Domains to rerun: {[d.value for d in domains_to_rerun]}",
                f"Cascade risk: {cascade_risk:.2f}",
            ],
        ))

    content = (
        "CHEMICAL EQUILIBRIUM (LE CHATELIER'S)\n"
        f"New variable: {new_variable.name} (magnitude: {new_variable.magnitude:.2f})\n"
        f"Stress magnitude: {stress}\n"
        f"Domains to rerun: {[d.value for d in domains_to_rerun]}\n"
        f"Bonds to reevaluate: {bonds_to_reeval}\n"
        f"Cascade risk: {cascade_risk:.2f}\n"
        f"New equilibrium: {'reached' if equilibrium_reached else 'NOT reached'}\n"
    )

    perspective = Perspective(
        framework=FrameworkID.CHEMICAL_EQUILIBRIUM,
        domain=Domain.CHEMISTRY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.8,
    )

    return perspective, shift


def _assess_stress(
    new_variable: Variable,
    current_outputs: dict[Domain, DomainOutput],
) -> str:
    """How much does this new variable change the existing analysis?"""
    # Count how many existing variables it contradicts
    contradiction_count = 0
    for output in current_outputs.values():
        for p in output.perspectives:
            for v in p.variables_found:
                # Same name but different direction = contradiction
                if v.name == new_variable.name and v.direction != new_variable.direction:
                    contradiction_count += 1
                # Opposing direction and similar magnitude = stress
                if (v.direction != new_variable.direction
                        and v.direction != Direction.NEUTRAL
                        and new_variable.direction != Direction.NEUTRAL
                        and abs(v.magnitude - new_variable.magnitude) < 0.2):
                    contradiction_count += 1

    if new_variable.magnitude > 0.7 and contradiction_count >= 2:
        return "high"
    elif new_variable.magnitude > 0.5 or contradiction_count >= 1:
        return "medium"
    return "low"


def _determine_shift(
    new_variable: Variable,
    stress: str,
    current_outputs: dict[Domain, DomainOutput],
) -> tuple[list[Domain], list[str]]:
    """Determine which domains and bonds need recalculation."""
    domains_to_rerun = []
    bonds_to_reeval = []

    if stress == "high":
        # All domains need to re-run
        domains_to_rerun = list(current_outputs.keys())
        # All bonds need reevaluation
        domain_names = [d.value for d in current_outputs.keys()]
        for i, a in enumerate(domain_names):
            for b in domain_names[i + 1:]:
                bonds_to_reeval.append(f"{a}↔{b}")

    elif stress == "medium":
        # Only domains whose variables are affected
        for domain, output in current_outputs.items():
            for p in output.perspectives:
                for v in p.variables_found:
                    if (v.name == new_variable.name
                            or v.direction != new_variable.direction):
                        if domain not in domains_to_rerun:
                            domains_to_rerun.append(domain)
                        break

    return domains_to_rerun, bonds_to_reeval


def _assess_cascade_risk(
    stress: str,
    domains_to_rerun: list[Domain],
    current_outputs: dict[Domain, DomainOutput],
) -> float:
    """Does the shift in one area cause a chain reaction?"""
    if stress == "low":
        return 0.1

    # Risk increases with number of domains affected
    domain_ratio = len(domains_to_rerun) / max(len(current_outputs), 1)

    # Risk increases with number of root causes already found
    total_roots = sum(len(o.root_causes) for o in current_outputs.values())
    root_factor = min(total_roots / 10.0, 0.5)

    stress_factor = {"low": 0.1, "medium": 0.3, "high": 0.6}.get(stress, 0.3)

    return min(domain_ratio * 0.4 + root_factor * 0.3 + stress_factor * 0.3, 1.0)
