"""
Core types for the Reasoning Engine.

These are the shared data structures that every domain module operates on.
No domain logic lives here — just the language the engine speaks.

Every domain is an isolated island. These types are the ONLY shared contract.
Bridges between islands carry these types and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Domain(Enum):
    """The five elements of Wu Xing."""
    PHYSICS = "physics"             # Earth — ground of reality
    MATHEMATICS = "mathematics"     # Metal — precision grid
    PSYCHOLOGY = "psychology"       # Water — hidden depths
    PHILOSOPHY = "philosophy"       # Wood — expansion, questions the question
    CHEMISTRY = "chemistry"         # Fire — transformation, governance + analytical


class FrameworkID(Enum):
    """Every concept in the engine has a unique ID."""

    # --- Physics (Earth) — 11 concepts ---
    # Phase 1: Root Finding & Trajectory
    FIRST_PRINCIPLES = "first_principles"
    CONSERVATION_OF_ENERGY = "conservation_of_energy"
    ENTROPY = "entropy"
    TRAJECTORY_MOMENTUM = "trajectory_momentum"
    POTENTIAL_KINETIC = "potential_kinetic"
    EQUILIBRIUM = "equilibrium"
    # Phase 2: Bias Penetration
    ANOMALOUS_MOTION = "anomalous_motion"
    SOCRATIC_SQUEEZE = "socratic_squeeze"
    REFERENCE_FRAME_SHIFT = "reference_frame_shift"
    ENTROPY_LEAK = "entropy_leak"
    REDUCTIO = "reductio_ad_absurdum"

    # --- Mathematics (Metal) — 9 layers ---
    SIGNAL_NOISE = "signal_noise"
    CATEGORY_THEORY = "category_theory"
    MANIFOLD = "manifold"
    DIMENSIONAL_REDUCTION = "dimensional_reduction"
    CONVERGENCE = "convergence"
    BAYESIAN = "bayesian_inference"
    GAME_THEORY = "game_theory"
    CAUSAL_LOOPS = "causal_loops"
    FRAGILITY = "ergodicity_fragility"

    # --- Psychology (Water) — 5 concepts ---
    DUAL_PROCESS = "dual_process"
    COGNITIVE_DISSONANCE = "cognitive_dissonance"
    MOTIVATED_REASONING = "motivated_reasoning"
    DIALECTICAL_THINKING = "dialectical_thinking"
    METACOGNITION = "metacognition"

    # --- Philosophy (Wood) — 5 concepts ---
    ONTOLOGY = "ontology"
    EPISTEMOLOGY = "epistemology"
    PHENOMENOLOGY = "phenomenology"
    DIALECTICS = "dialectics"
    TELEOLOGY = "teleology"

    # --- Chemistry (Fire) — 6 concepts ---
    # Module A: Governance
    SELF_ASSEMBLY = "self_assembly"
    VALENCE = "valence"
    CHEMICAL_EQUILIBRIUM = "chemical_equilibrium"
    # Module C: Analytical
    CHIRALITY = "chirality"
    CATALYSIS = "catalysis"
    RESONANCE = "resonance"


class Direction(Enum):
    """Direction of a force or variable — is it helping or hurting?"""
    POSITIVE = "positive"       # moving toward resolution
    NEGATIVE = "negative"       # moving toward breakdown
    NEUTRAL = "neutral"         # no clear direction
    CIRCULAR = "circular"       # feedback loop


class Severity(Enum):
    """How critical is this finding?"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class FragilityRating(Enum):
    """Final stress test result."""
    FRAGILE = "fragile"
    ROBUST = "robust"
    ANTIFRAGILE = "antifragile"


class SignalType(Enum):
    """Signal vs. Noise classification."""
    SIGNAL = "signal"
    NOISE = "noise"
    LATENT = "latent"


class BondType(Enum):
    """Chemistry Valence — how two outputs bond."""
    IONIC = "ionic"             # opposites held by attraction
    COVALENT = "covalent"       # similar outputs sharing a common variable
    NONE = "none"               # genuinely unrelated


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Variable:
    """
    A named force or factor in the problem space.

    Variables are what every domain finds, transforms, and reduces.
    """
    name: str
    description: str
    magnitude: float                    # 0.0 to 1.0
    direction: Direction
    confidence: float                   # 0.0 to 1.0
    source_framework: FrameworkID
    is_hidden: bool = False
    is_user_stated: bool = True
    evidence: list[str] = field(default_factory=list)


@dataclass
class Perspective:
    """
    A single "chart" in the manifold — one angle on the problem.
    Each framework produces perspectives.
    """
    framework: FrameworkID
    domain: Domain
    content: str
    variables_found: list[Variable] = field(default_factory=list)
    signal_type: SignalType = SignalType.SIGNAL
    weight: float = 1.0


@dataclass
class RootCause:
    """Variable y — the underlying issue, often hidden by user bias."""
    variable: Variable
    evidence_chain: list[str]
    bias_that_hid_it: Optional[str] = None
    confidence: float = 0.0
    frameworks_that_agree: list[FrameworkID] = field(default_factory=list)


@dataclass
class Consequence:
    """
    A concrete, time-bound projection.
    "Things might get harder" is UNACCEPTABLE.
    "In 6 months at this trajectory, you'll be 30% below market rate" is the standard.
    """
    description: str
    timeframe: str
    severity: Severity
    probability: float
    trajectory_framework: FrameworkID = FrameworkID.TRAJECTORY_MOMENTUM
    is_reversible: bool = True
    reversal_window: Optional[str] = None


@dataclass
class CausalLoop:
    """A detected feedback loop in the problem."""
    name: str
    description: str
    loop_type: str                      # "reinforcing" or "balancing"
    variables_in_loop: list[str]
    is_dominant: bool = False
    delay: Optional[str] = None


@dataclass
class GameState:
    """Strategic multi-agent analysis."""
    agents: list[str]
    is_zero_sum: bool
    nash_equilibrium: Optional[str] = None
    dominant_strategies: dict[str, str] = field(default_factory=dict)
    prisoners_dilemma: bool = False


@dataclass
class FragilityResult:
    """Ergodicity & Fragility stress test."""
    rating: FragilityRating
    reasoning: str
    is_ergodic: bool
    tail_risk: Optional[str] = None
    tail_risk_severity: Severity = Severity.LOW
    real_world_executable: bool = True
    execution_blockers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Problem — the user's input
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """
    The user's input. What they tell us about their situation.
    x1, x2, ... are the variables they provide.
    Variable y is what we're looking for.
    """
    statement: str
    variables: list[Variable] = field(default_factory=list)
    context: str = ""
    domain_hints: list[Domain] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bridge Contracts — the ONLY way domains talk to each other
# ---------------------------------------------------------------------------

@dataclass
class DomainOutput:
    """
    Standardized output from any domain island.

    This is what crosses the bridge FROM a domain.
    Every domain produces exactly one DomainOutput.
    """
    domain: Domain
    perspectives: list[Perspective] = field(default_factory=list)
    root_causes: list[RootCause] = field(default_factory=list)
    consequences: list[Consequence] = field(default_factory=list)
    causal_loops: list[CausalLoop] = field(default_factory=list)
    game_state: Optional[GameState] = None
    raw_analysis: str = ""


@dataclass
class DomainInput:
    """
    Standardized input TO a domain island.

    This is what crosses the bridge INTO a domain.
    Contains the problem + any upstream domain outputs the orchestrator
    decides to pass (based on Wu Xing Sheng cycle).
    """
    problem: Problem
    upstream_outputs: dict[Domain, DomainOutput] = field(default_factory=dict)


@dataclass
class ChallengeInput:
    """
    Input for the Ke cycle (Controlling / Deconstruction).

    A domain receives another domain's output and challenges it.
    Physics checks Psychology. Psychology checks Chemistry. etc.
    """
    challenger_domain: Domain
    target_domain: Domain
    target_output: DomainOutput


@dataclass
class ChallengeOutput:
    """
    Result of a Ke cycle challenge.

    What the challenging domain found wrong, questionable,
    or unsupported in the target domain's output.
    """
    challenger_domain: Domain
    target_domain: Domain
    contradictions: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    confidence_adjustments: dict[str, float] = field(default_factory=dict)
    scrutiny_score: float = 0.0         # 0.0 = nothing wrong, 1.0 = total rejection
    flags: list[str] = field(default_factory=list)
