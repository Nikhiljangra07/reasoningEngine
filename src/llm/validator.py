"""
Math Formation Validation — Deterministic Safety Net.

After Chemistry Self-Assembly decides the formation, Math validates it
using rule-based checks. This is NOT an LLM call — it's fast, reliable,
and unhackable.

If Math flags adjustments, Chemistry's formation is modified before
agents spawn. This catches mistriaging.

ISOLATION: Imports from src.core.types and src.llm.router only.
           Does NOT import from any domain module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import Domain, Problem
from src.llm.router import LLMFormationPlan, ALL_CONCEPTS


@dataclass
class ValidationResult:
    """Result of Math's validation of Chemistry's formation plan."""
    approved: bool
    adjustments: list[str]              # human-readable adjustment descriptions
    adjusted_plan: LLMFormationPlan     # the plan after adjustments (may be same as input)


def validate_formation(
    plan: LLMFormationPlan,
    problem: Problem,
) -> ValidationResult:
    """
    Validate Chemistry's formation plan using deterministic rules.

    Rules check for:
    - Missing critical domains/concepts based on problem signals
    - Over-activation (too many agents)
    - Under-activation (too few agents)
    - Mandatory concept dependencies

    Returns the plan with adjustments applied.
    """
    adjustments: list[str] = []
    adjusted_domains = list(plan.active_domains)
    adjusted_concepts = {k: list(v) for k, v in plan.concepts_per_domain.items()}

    # Combine problem statement + variable descriptions for signal detection
    problem_text = problem.statement.lower()
    for var in problem.variables:
        problem_text += " " + var.description.lower()
        problem_text += " " + var.name.lower().replace("_", " ")

    # -----------------------------------------------------------------------
    # Rule 1: Problem mentions other people/actors → Game Theory MUST be active
    # -----------------------------------------------------------------------
    agent_keywords = [
        "partner", "cofounder", "boss", "manager", "team", "company",
        "competitor", "spouse", "parent", "friend", "colleague", "client",
        "investor", "employee", "landlord", "bank", "they ", "them ",
        "he ", "she ", "his ", "her ",
    ]
    has_agents = any(kw in problem_text for kw in agent_keywords)

    if has_agents:
        if "game_theory" not in adjusted_concepts.get("mathematics", []):
            adjusted_concepts.setdefault("mathematics", []).append("game_theory")
            adjustments.append(
                "ADDED game_theory to mathematics: problem involves other actors/agents."
            )

    # -----------------------------------------------------------------------
    # Rule 2: Internal conflict or emotional distress → Psychology MUST include
    #         Cognitive Dissonance + Motivated Reasoning
    # -----------------------------------------------------------------------
    conflict_keywords = [
        "conflict", "torn", "doubt", "confused", "unsure", "struggle",
        "fight", "argue", "disagree", "stress", "anxiety", "fear",
        "angry", "frustrated", "overwhelm", "guilt", "shame",
    ]
    has_conflict = any(kw in problem_text for kw in conflict_keywords)

    if has_conflict:
        if Domain.PSYCHOLOGY not in adjusted_domains:
            adjusted_domains.append(Domain.PSYCHOLOGY)
            adjustments.append(
                "ADDED psychology domain: problem involves emotional conflict."
            )
        psych_concepts = adjusted_concepts.setdefault("psychology", [])
        for required in ["cognitive_dissonance", "motivated_reasoning"]:
            if required not in psych_concepts:
                psych_concepts.append(required)
                adjustments.append(
                    f"ADDED {required} to psychology: problem involves internal conflict."
                )

    # -----------------------------------------------------------------------
    # Rule 3: Decision between options → Philosophy Dialectics MUST be active
    # -----------------------------------------------------------------------
    decision_keywords = [
        "should i", "decide", "choice", "option", "alternative",
        "choose", "either", "or ", "vs", "versus", "trade-off",
        "dilemma", "crossroads",
    ]
    has_decision = any(kw in problem_text for kw in decision_keywords)

    if has_decision:
        if Domain.PHILOSOPHY not in adjusted_domains:
            adjusted_domains.append(Domain.PHILOSOPHY)
            adjustments.append(
                "ADDED philosophy domain: problem involves a decision."
            )
        phil_concepts = adjusted_concepts.setdefault("philosophy", [])
        for required in ["dialectics", "teleology"]:
            if required not in phil_concepts:
                phil_concepts.append(required)
                adjustments.append(
                    f"ADDED {required} to philosophy: problem involves decision-making."
                )

    # -----------------------------------------------------------------------
    # Rule 4: Time pressure or trajectory → Physics Trajectory MUST be active
    # -----------------------------------------------------------------------
    time_keywords = [
        "deadline", "time", "month", "year", "week", "soon",
        "running out", "urgent", "before", "after", "how long",
        "trajectory", "heading", "direction", "momentum",
    ]
    has_time = any(kw in problem_text for kw in time_keywords)

    if has_time:
        phys_concepts = adjusted_concepts.setdefault("physics", [])
        if "trajectory_momentum" not in phys_concepts:
            phys_concepts.append("trajectory_momentum")
            adjustments.append(
                "ADDED trajectory_momentum to physics: problem involves time/trajectory."
            )
        if "entropy" not in phys_concepts:
            phys_concepts.append("entropy")
            adjustments.append(
                "ADDED entropy to physics: problem involves time-bound decay."
            )

    # -----------------------------------------------------------------------
    # Rule 5: Physics and Mathematics are ALWAYS required
    # -----------------------------------------------------------------------
    if Domain.PHYSICS not in adjusted_domains:
        adjusted_domains.append(Domain.PHYSICS)
        adjustments.append("ADDED physics domain: always required as foundation.")
    if Domain.MATHEMATICS not in adjusted_domains:
        adjusted_domains.append(Domain.MATHEMATICS)
        adjustments.append("ADDED mathematics domain: always required as foundation.")

    # Ensure minimum physics concepts
    phys_concepts = adjusted_concepts.setdefault("physics", [])
    for required in ["first_principles", "conservation_of_energy", "equilibrium"]:
        if required not in phys_concepts:
            phys_concepts.append(required)

    # Ensure minimum math concepts
    math_concepts = adjusted_concepts.setdefault("mathematics", [])
    for required in ["signal_noise", "bayesian_inference", "convergence"]:
        if required not in math_concepts:
            math_concepts.append(required)

    # -----------------------------------------------------------------------
    # Rule 6: Chemistry governance is ALWAYS required
    # -----------------------------------------------------------------------
    if Domain.CHEMISTRY not in adjusted_domains:
        adjusted_domains.append(Domain.CHEMISTRY)
    chem_concepts = adjusted_concepts.setdefault("chemistry", [])
    for required in ["self_assembly", "valence", "chemical_equilibrium"]:
        if required not in chem_concepts:
            chem_concepts.append(required)

    # -----------------------------------------------------------------------
    # Rule 7: Metacognition is ALWAYS required (delivery calibration)
    # -----------------------------------------------------------------------
    if Domain.PSYCHOLOGY not in adjusted_domains:
        adjusted_domains.append(Domain.PSYCHOLOGY)
    psych_concepts = adjusted_concepts.setdefault("psychology", [])
    if "metacognition" not in psych_concepts:
        psych_concepts.append("metacognition")

    # -----------------------------------------------------------------------
    # Rule 8: Agent count bounds
    # -----------------------------------------------------------------------
    total_agents = sum(len(v) for v in adjusted_concepts.values())

    if total_agents > 20:
        adjustments.append(
            f"WARNING: {total_agents} agents estimated — high activation. "
            "Review if all concepts are truly needed."
        )
    elif total_agents < 5:
        adjustments.append(
            f"WARNING: only {total_agents} agents — likely under-activation. "
            "Consider adding more concepts."
        )

    # -----------------------------------------------------------------------
    # Build adjusted plan
    # -----------------------------------------------------------------------
    adjusted_plan = LLMFormationPlan(
        active_domains=adjusted_domains,
        concepts_per_domain=adjusted_concepts,
        estimated_agent_count=total_agents,
        estimated_iterations=plan.estimated_iterations,
        estimated_credit_cost=total_agents * 1.5,  # rough: 1.5 credits per agent
        problem_complexity=plan.problem_complexity,
        reasoning=plan.reasoning + (
            f" | Math validation applied {len(adjustments)} adjustment(s)."
            if adjustments else " | Math validation: approved as-is."
        ),
    )

    return ValidationResult(
        approved=len(adjustments) == 0,
        adjustments=adjustments,
        adjusted_plan=adjusted_plan,
    )
