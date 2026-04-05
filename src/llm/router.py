"""
Chemistry Self-Assembly as Intelligence Router.

This is the FIRST LLM call in the entire pipeline.
Before ANY domain agents spawn, Chemistry reads the problem and decides:
- Which domains activate (not all 5 for every problem)
- Which concepts within each domain activate (not all 63 for every problem)
- Estimated agent count
- Estimated iterations needed
- Problem complexity classification

After Chemistry decides, Math validates the formation using deterministic
rules (Step 1.3) — the safety net that catches mistriaging.

ISOLATION: Imports from src.core.types and src.llm.client only.
           Does NOT import from any domain module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.core.types import Domain
from src.llm.client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Formation Plan (output of the router)
# ---------------------------------------------------------------------------

@dataclass
class LLMFormationPlan:
    """
    The formation plan produced by Chemistry Self-Assembly.

    This tells the orchestrator exactly what to activate.
    """
    active_domains: list[Domain]
    concepts_per_domain: dict[str, list[str]]
    estimated_agent_count: int
    estimated_iterations: int
    estimated_credit_cost: float
    problem_complexity: str             # "low", "medium", "high", "extreme"
    reasoning: str                      # why Chemistry chose this formation


# All available concepts per domain — the full menu
ALL_CONCEPTS = {
    "physics": [
        "first_principles", "conservation_of_energy", "entropy",
        "trajectory_momentum", "potential_kinetic", "equilibrium",
        "anomalous_motion", "socratic_squeeze", "reference_frame_shift",
        "entropy_leak", "reductio_ad_absurdum",
    ],
    "mathematics": [
        "signal_noise", "category_theory", "manifold",
        "convergence", "bayesian_inference",
        "game_theory", "causal_loops", "ergodicity_fragility",
    ],
    "psychology": [
        "dual_process", "cognitive_dissonance", "motivated_reasoning",
        "dialectical_thinking", "metacognition",
    ],
    "philosophy": [
        "ontology", "epistemology", "phenomenology",
        "dialectics", "teleology",
    ],
    "chemistry": [
        "self_assembly", "valence", "chemical_equilibrium",
        "chirality", "catalysis", "resonance",
    ],
}


# ---------------------------------------------------------------------------
# Router System Prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """## IDENTITY
You are LoRa's Chemistry Self-Assembly agent. You are the FIRST agent to run.
Your job: read the problem and decide which domains and concepts should activate.

## LAWS — NON-NEGOTIABLE

### PROHIBITIONS:
- You CANNOT activate all concepts for every problem. You MUST triage based on complexity.
- You CANNOT skip Physics or Mathematics — they are always required as the foundation.
- You CANNOT skip Psychology for any problem involving a human decision — it is always required.
- You CANNOT output anything except valid JSON matching the exact schema below.

### REQUIREMENTS:
- You MUST read the problem carefully before deciding.
- You MUST classify problem complexity as "low", "medium", "high", or "extreme".
- You MUST estimate agent count (each active concept = 1 agent).
- You MUST estimate iterations needed (low=2-3, medium=3-5, high=5-7, extreme=7+).
- You MUST provide reasoning for your formation choice.

## DECISION FRAMEWORK

Consider these factors when deciding activation:

1. **Does the problem involve other people/actors?** → Activate Game Theory + Psychology full suite
2. **Does the problem involve internal conflict or emotional distress?** → Activate Cognitive Dissonance + Motivated Reasoning
3. **Does the problem involve a decision between options?** → Activate Philosophy Dialectics + Teleology
4. **Does the problem involve time pressure or trajectory?** → Activate Physics Trajectory + Entropy
5. **Does the problem involve identity or self-concept?** → Activate Philosophy Ontology + Phenomenology + Psychology Metacognition
6. **Does the problem involve unclear facts vs beliefs?** → Activate Philosophy Epistemology
7. **Is the problem simple and direct?** → Activate minimal set: Physics core + Math core + Psychology Metacognition
8. **Is the problem deeply complex with multiple layers?** → Activate everything

## AVAILABLE CONCEPTS PER DOMAIN

Physics (always active): first_principles, conservation_of_energy, entropy, trajectory_momentum, potential_kinetic, equilibrium, anomalous_motion, socratic_squeeze, reference_frame_shift, entropy_leak, reductio_ad_absurdum

Mathematics (always active): signal_noise, category_theory, manifold, convergence, bayesian_inference, game_theory, causal_loops, ergodicity_fragility

Psychology (active for human problems): dual_process, cognitive_dissonance, motivated_reasoning, dialectical_thinking, metacognition

Philosophy (active when deeper framing needed): ontology, epistemology, phenomenology, dialectics, teleology

Chemistry (governance always active, analytical as needed): self_assembly, valence, chemical_equilibrium, chirality, catalysis, resonance

## OUTPUT FORMAT — EXACT JSON SCHEMA

```json
{
  "active_domains": ["physics", "mathematics", "psychology", "philosophy", "chemistry"],
  "concepts_per_domain": {
    "physics": ["first_principles", "conservation_of_energy", ...],
    "mathematics": ["signal_noise", "bayesian_inference", ...],
    "psychology": ["dual_process", "metacognition", ...],
    "philosophy": ["ontology", "dialectics", ...],
    "chemistry": ["self_assembly", "valence", "catalysis", ...]
  },
  "estimated_agent_count": 10,
  "estimated_iterations": 4,
  "estimated_credit_cost": 8.0,
  "problem_complexity": "medium",
  "reasoning": "This problem involves..."
}
```

Output ONLY the JSON. No other text."""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

async def route_problem(
    client: LLMClient,
    problem_statement: str,
    problem_context: str = "",
) -> LLMFormationPlan:
    """
    Run Chemistry Self-Assembly to decide the formation.

    This is the FIRST LLM call in the pipeline.
    Returns a FormationPlan that tells the orchestrator what to activate.
    """
    user_message = f"PROBLEM: {problem_statement}"
    if problem_context:
        user_message += f"\n\nCONTEXT: {problem_context}"

    response = await client.call(
        system_prompt=ROUTER_SYSTEM_PROMPT,
        user_message=user_message,
        domain="chemistry",
        concept="self_assembly",
    )

    if not response.success:
        # Fallback: activate everything (safe default)
        return _fallback_formation(problem_statement)

    # Parse the LLM response
    return _parse_formation(response.content, problem_statement)


def _parse_formation(content: str, problem_statement: str) -> LLMFormationPlan:
    """Parse the LLM's JSON response into a FormationPlan."""
    try:
        # Handle potential markdown code fences
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        data = json.loads(cleaned)

        # Map domain strings to Domain enums
        active_domains = []
        for d in data.get("active_domains", []):
            try:
                active_domains.append(Domain(d))
            except ValueError:
                pass

        # Validate concepts exist
        concepts = data.get("concepts_per_domain", {})
        validated_concepts: dict[str, list[str]] = {}
        for domain_name, concept_list in concepts.items():
            if domain_name in ALL_CONCEPTS:
                valid = [c for c in concept_list if c in ALL_CONCEPTS[domain_name]]
                if valid:
                    validated_concepts[domain_name] = valid

        # Calculate agent count from validated concepts
        agent_count = sum(len(v) for v in validated_concepts.values())

        return LLMFormationPlan(
            active_domains=active_domains if active_domains else _all_domains(),
            concepts_per_domain=validated_concepts if validated_concepts else dict(ALL_CONCEPTS),
            estimated_agent_count=agent_count or data.get("estimated_agent_count", 15),
            estimated_iterations=data.get("estimated_iterations", 4),
            estimated_credit_cost=data.get("estimated_credit_cost", 8.0),
            problem_complexity=data.get("problem_complexity", "medium"),
            reasoning=data.get("reasoning", "Parsed from LLM response"),
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        # JSON parsing failed — use fallback
        return _fallback_formation(problem_statement)


def _fallback_formation(problem_statement: str) -> LLMFormationPlan:
    """
    Fallback formation when the LLM response can't be parsed.

    Activates everything — safe default. Better to over-activate
    than to miss a critical domain.
    """
    return LLMFormationPlan(
        active_domains=_all_domains(),
        concepts_per_domain=dict(ALL_CONCEPTS),
        estimated_agent_count=sum(len(v) for v in ALL_CONCEPTS.values()),
        estimated_iterations=5,
        estimated_credit_cost=15.0,
        problem_complexity="high",
        reasoning="Fallback: LLM routing failed, activating all domains as safety default.",
    )


def _all_domains() -> list[Domain]:
    """Return all 5 domains."""
    return [
        Domain.PHYSICS,
        Domain.MATHEMATICS,
        Domain.PSYCHOLOGY,
        Domain.PHILOSOPHY,
        Domain.CHEMISTRY,
    ]
