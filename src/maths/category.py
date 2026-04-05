"""
Maths Layer 2: Category Theory (The Universal Translator)

Doesn't care what the objects are (numbers, atoms, or emotions) —
only cares about the relationships between them.

This is what lets Physics talk to Psychology and realize they're
solving the same equation.

4 Operations:
1. Morphisms — how does this transform the situation? (the flow)
2. Isomorphism — are two different-looking things structurally identical? (the pattern)
3. Topos — can different internal logics coexist? (the room)
4. Compositionality — how do small things combine into emergent properties? (the glue)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    DomainOutput,
    FrameworkID,
    Perspective,
    Variable,
)


@dataclass
class Morphism:
    """
    An arrow: how one variable transforms into another.

    In physics this is a force. In psychology this is an influence.
    Category Theory treats them identically — both are arrows
    from point A to point B.
    """
    source: Variable
    target: Variable
    transformation: str     # description of how source becomes/affects target
    strength: float         # 0.0 to 1.0 — how strong is this transformation?
    domain_source: str      # which domain found the source
    domain_target: str      # which domain found the target


@dataclass
class Isomorphism:
    """
    Two things that are different in appearance but identical in structure.

    The Golden Key — if you solve tension in a bridge (Physics),
    you can apply the same logic to tension in a social group (Psychology).
    """
    perspective_a: Perspective
    perspective_b: Perspective
    structural_match: str       # description of what structure they share
    confidence: float           # 0.0 to 1.0


@dataclass
class CategoryResult:
    """Result of category theory analysis across domain outputs."""
    morphisms: list[Morphism] = field(default_factory=list)
    isomorphisms: list[Isomorphism] = field(default_factory=list)
    cross_domain_translations: list[str] = field(default_factory=list)
    emergent_variables: list[Variable] = field(default_factory=list)


def analyze_categories(domain_outputs: list[DomainOutput]) -> CategoryResult:
    """
    Run category theory analysis across all domain outputs.

    Finds morphisms (transformations), isomorphisms (structural matches),
    and emergent variables from compositionality.
    """
    all_perspectives = []
    all_variables = []

    for output in domain_outputs:
        all_perspectives.extend(output.perspectives)
        for p in output.perspectives:
            all_variables.extend(p.variables_found)

    morphisms = _find_morphisms(all_variables)
    isomorphisms = _find_isomorphisms(all_perspectives)
    emergent = _find_emergent_variables(morphisms, all_variables)

    translations = []
    for iso in isomorphisms:
        translations.append(
            f"'{iso.perspective_a.framework.value}' and '{iso.perspective_b.framework.value}' "
            f"are structurally identical: {iso.structural_match}"
        )

    return CategoryResult(
        morphisms=morphisms,
        isomorphisms=isomorphisms,
        cross_domain_translations=translations,
        emergent_variables=emergent,
    )


def _find_morphisms(variables: list[Variable]) -> list[Morphism]:
    """
    Find morphisms — transformation arrows between variables.

    If variable A's output feeds variable B's input, that's a morphism.
    Detected by: matching directions, causal language in descriptions,
    and magnitude relationships.
    """
    morphisms = []

    for i, source in enumerate(variables):
        for target in variables[i + 1:]:
            # Skip same-framework connections (those are internal, not cross-domain)
            if source.source_framework == target.source_framework:
                continue

            strength = _morphism_strength(source, target)
            if strength > 0.3:
                morphisms.append(Morphism(
                    source=source,
                    target=target,
                    transformation=(
                        f"'{source.name}' ({source.direction.value}, {source.magnitude:.2f}) "
                        f"→ '{target.name}' ({target.direction.value}, {target.magnitude:.2f})"
                    ),
                    strength=strength,
                    domain_source=source.source_framework.value,
                    domain_target=target.source_framework.value,
                ))

    return morphisms


def _morphism_strength(source: Variable, target: Variable) -> float:
    """
    Calculate the strength of a morphism between two variables.

    Strong morphisms:
    - Source is positive, target is the consequence (negative drain from that positive)
    - Source and target have similar magnitudes (energy conservation in the arrow)
    - Both have high confidence
    """
    score = 0.0

    # Magnitude similarity — energy is conserved in the transformation
    magnitude_diff = abs(source.magnitude - target.magnitude)
    score += (1.0 - magnitude_diff) * 0.3

    # Causal direction — positive source feeding negative target is a strong arrow
    if source.direction != target.direction:
        score += 0.3  # opposing directions = transformation is happening

    # Confidence — both must be reasonably confident
    avg_confidence = (source.confidence + target.confidence) / 2
    score += avg_confidence * 0.2

    # Hidden variables connecting to visible ones = strong morphisms
    if source.is_hidden != target.is_hidden:
        score += 0.2

    return min(score, 1.0)


def _find_isomorphisms(perspectives: list[Perspective]) -> list[Isomorphism]:
    """
    Find isomorphisms — perspectives from different frameworks that
    are structurally identical (different appearance, same pattern).

    Two perspectives are isomorphic if:
    - They found similar variable patterns (same directions, similar magnitudes)
    - They came from different frameworks
    - Their conclusions point to the same underlying structure
    """
    isomorphisms = []

    for i, pa in enumerate(perspectives):
        for pb in perspectives[i + 1:]:
            # Must be different frameworks
            if pa.framework == pb.framework:
                continue

            confidence = _structural_similarity(pa, pb)
            if confidence > 0.5:
                match_desc = _describe_structural_match(pa, pb)
                isomorphisms.append(Isomorphism(
                    perspective_a=pa,
                    perspective_b=pb,
                    structural_match=match_desc,
                    confidence=confidence,
                ))

    return isomorphisms


def _structural_similarity(pa: Perspective, pb: Perspective) -> float:
    """
    Calculate structural similarity between two perspectives.

    Ignores the domain-specific language. Looks at:
    - Variable count similarity
    - Direction distribution similarity
    - Magnitude distribution similarity
    - Hidden variable pattern similarity
    """
    if not pa.variables_found or not pb.variables_found:
        return 0.0

    score = 0.0

    # Variable count similarity
    count_diff = abs(len(pa.variables_found) - len(pb.variables_found))
    max_count = max(len(pa.variables_found), len(pb.variables_found))
    score += (1.0 - count_diff / max_count) * 0.25 if max_count > 0 else 0.0

    # Direction distribution similarity
    dirs_a = [v.direction for v in pa.variables_found]
    dirs_b = [v.direction for v in pb.variables_found]
    common_dirs = set(dirs_a) & set(dirs_b)
    all_dirs = set(dirs_a) | set(dirs_b)
    score += (len(common_dirs) / len(all_dirs)) * 0.25 if all_dirs else 0.0

    # Magnitude distribution similarity
    avg_mag_a = sum(v.magnitude for v in pa.variables_found) / len(pa.variables_found)
    avg_mag_b = sum(v.magnitude for v in pb.variables_found) / len(pb.variables_found)
    score += (1.0 - abs(avg_mag_a - avg_mag_b)) * 0.25

    # Hidden variable pattern
    hidden_a = any(v.is_hidden for v in pa.variables_found)
    hidden_b = any(v.is_hidden for v in pb.variables_found)
    score += 0.25 if hidden_a == hidden_b else 0.0

    return score


def _describe_structural_match(pa: Perspective, pb: Perspective) -> str:
    """Describe what structural pattern two perspectives share."""
    parts = []

    dirs_a = {v.direction for v in pa.variables_found}
    dirs_b = {v.direction for v in pb.variables_found}
    common = dirs_a & dirs_b
    if common:
        parts.append(f"shared force directions: {', '.join(d.value for d in common)}")

    avg_a = sum(v.magnitude for v in pa.variables_found) / len(pa.variables_found)
    avg_b = sum(v.magnitude for v in pb.variables_found) / len(pb.variables_found)
    if abs(avg_a - avg_b) < 0.2:
        parts.append(f"similar magnitude scale ({avg_a:.2f} ≈ {avg_b:.2f})")

    hidden_a = [v for v in pa.variables_found if v.is_hidden]
    hidden_b = [v for v in pb.variables_found if v.is_hidden]
    if hidden_a and hidden_b:
        parts.append("both surfaced hidden variables")

    return "; ".join(parts) if parts else "structural pattern match"


def _find_emergent_variables(
    morphisms: list[Morphism], all_variables: list[Variable]
) -> list[Variable]:
    """
    Compositionality — find emergent variables from morphism chains.

    When A → B → C forms a chain, the combined effect (A → C)
    may reveal a variable D that neither A, B, nor C show individually.
    """
    emergent = []

    # Find variables that appear as targets of multiple morphisms
    # (convergence point = emergent property)
    target_counts: dict[str, int] = {}
    target_strengths: dict[str, float] = {}
    target_vars: dict[str, Variable] = {}

    for m in morphisms:
        name = m.target.name
        target_counts[name] = target_counts.get(name, 0) + 1
        target_strengths[name] = target_strengths.get(name, 0.0) + m.strength
        target_vars[name] = m.target

    for name, count in target_counts.items():
        if count >= 2:
            # Multiple morphisms converge on this variable — it's emergent
            var = target_vars[name]
            emergent_var = Variable(
                name=f"emergent_{name}",
                description=(
                    f"Emergent variable: '{name}' is the convergence point "
                    f"of {count} morphisms from different frameworks. "
                    f"Combined transformation strength: {target_strengths[name]:.2f}. "
                    "This variable gains new properties from the combination "
                    "that none of the individual sources show alone."
                ),
                magnitude=min(target_strengths[name] / count, 1.0),
                direction=var.direction,
                confidence=min(0.5 + count * 0.1, 0.95),
                source_framework=FrameworkID.CATEGORY_THEORY,
                is_hidden=var.is_hidden,
                is_user_stated=False,
                evidence=[
                    f"Convergence point of {count} morphisms",
                    f"Total transformation strength: {target_strengths[name]:.2f}",
                    "Emergent from compositionality — the whole exceeds the parts.",
                ],
            )
            emergent.append(emergent_var)

    return emergent
