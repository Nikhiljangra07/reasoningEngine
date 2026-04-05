"""
Maths Layer 6: Bayesian Inference (The Living Engine)

The living mathematical engine that collects every bit of information
on its way to finding a solution. Treats every data point — even biased
ones — as a gold update to the truth.

4 Operations:
1. Prior — initial belief based on what we know
2. Likelihood — how probable is this evidence given our current model?
3. Posterior — updated truth that accounts for all evidence
4. Latent Variable Inference — surface variable y from gaps in visible data
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    Direction,
    DomainOutput,
    FrameworkID,
    Perspective,
    Problem,
    RootCause,
    SignalType,
    Variable,
)


@dataclass
class BayesianBelief:
    """
    A belief about a variable — updated with each new evidence.

    Prior → Evidence → Posterior → (new evidence) → Updated Posterior → ...
    """
    variable_name: str
    prior: float                # initial probability (0.0 to 1.0)
    posterior: float             # current updated probability
    evidence_count: int         # how many pieces of evidence have updated this
    evidence_log: list[str] = field(default_factory=list)
    direction: Direction = Direction.NEUTRAL
    is_root_candidate: bool = False


@dataclass
class BayesianState:
    """The current state of all Bayesian beliefs."""
    beliefs: dict[str, BayesianBelief]          # name → belief
    latent_variables: list[Variable]            # inferred hidden variables
    global_truth: list[BayesianBelief]          # beliefs sorted by posterior (highest first)
    iteration: int


def initialize_prior(problem: Problem) -> BayesianState:
    """
    Step 1: Build the Prior from the problem's stated variables.

    The prior is not zero — the user's input is the foundation.
    Each stated variable becomes an initial belief.
    """
    beliefs = {}

    for var in problem.variables:
        belief = BayesianBelief(
            variable_name=var.name,
            prior=var.confidence,
            posterior=var.confidence,  # starts equal to prior
            evidence_count=0,
            evidence_log=[f"Prior from user input: {var.description}"],
            direction=var.direction,
        )
        beliefs[var.name] = belief

    return BayesianState(
        beliefs=beliefs,
        latent_variables=[],
        global_truth=sorted(beliefs.values(), key=lambda b: b.posterior, reverse=True),
        iteration=0,
    )


def update_with_evidence(
    state: BayesianState,
    domain_outputs: list[DomainOutput],
) -> BayesianState:
    """
    Step 2 + 3: Update beliefs with new evidence from domain outputs.

    Every variable found by any framework is treated as evidence.
    The posterior updates. The global truth reshapes.
    """
    for output in domain_outputs:
        for perspective in output.perspectives:
            for var in perspective.variables_found:
                _update_single_belief(state, var, perspective)

    # After processing all evidence, infer latent variables
    latent = _infer_latent_variables(state)
    state.latent_variables = latent

    # Add latent variables as new beliefs
    for lv in latent:
        if lv.name not in state.beliefs:
            state.beliefs[lv.name] = BayesianBelief(
                variable_name=lv.name,
                prior=0.1,  # low prior — we inferred this, didn't observe it
                posterior=lv.confidence,
                evidence_count=1,
                evidence_log=[f"Inferred latent variable: {lv.description}"],
                direction=lv.direction,
                is_root_candidate=True,
            )

    # Update global truth ranking
    state.global_truth = sorted(
        state.beliefs.values(),
        key=lambda b: b.posterior,
        reverse=True,
    )
    state.iteration += 1

    return state


def get_root_candidates(state: BayesianState) -> list[RootCause]:
    """
    Extract the most likely root causes from Bayesian beliefs.

    Root candidates are beliefs that:
    - Have high posterior probability
    - Were either inferred (latent) or significantly updated from prior
    - Have negative or circular direction (problems, not solutions)
    """
    candidates = []

    for belief in state.global_truth:
        is_candidate = (
            belief.is_root_candidate
            or (belief.posterior > belief.prior * 1.5 and belief.posterior > 0.6)
            or (belief.direction in (Direction.NEGATIVE, Direction.CIRCULAR)
                and belief.posterior > 0.65)
        )

        if is_candidate:
            variable = Variable(
                name=belief.variable_name,
                description=(
                    f"Bayesian root candidate: posterior={belief.posterior:.2f} "
                    f"(prior was {belief.prior:.2f}), "
                    f"updated by {belief.evidence_count} evidence points."
                ),
                magnitude=belief.posterior,
                direction=belief.direction,
                confidence=belief.posterior,
                source_framework=FrameworkID.BAYESIAN,
                is_hidden=belief.is_root_candidate,
                is_user_stated=False,
                evidence=belief.evidence_log[-5:],  # last 5 evidence entries
            )

            root = RootCause(
                variable=variable,
                evidence_chain=belief.evidence_log,
                confidence=belief.posterior,
                frameworks_that_agree=[FrameworkID.BAYESIAN],
            )
            candidates.append(root)

    return candidates


# ---------------------------------------------------------------------------
# Internal: Bayesian update logic
# ---------------------------------------------------------------------------

def _update_single_belief(
    state: BayesianState,
    evidence: Variable,
    source: Perspective,
) -> None:
    """
    Update a single belief with new evidence.

    Uses simplified Bayesian update:
    posterior = prior × likelihood / normalizer

    The likelihood is derived from the evidence's confidence and
    how well it aligns with the existing belief.
    """
    name = evidence.name

    if name in state.beliefs:
        belief = state.beliefs[name]

        # Calculate likelihood: how probable is this evidence given our current belief?
        likelihood = _calculate_likelihood(belief, evidence)

        # Bayesian update: posterior ∝ prior × likelihood
        unnormalized = belief.posterior * likelihood
        # Simple normalization: scale to [0, 1]
        belief.posterior = min(unnormalized / (unnormalized + (1 - belief.posterior) * (1 - likelihood) + 1e-10), 0.99)

        belief.evidence_count += 1
        belief.evidence_log.append(
            f"[{source.framework.value}] {evidence.description[:80]} "
            f"(likelihood: {likelihood:.2f})"
        )

        # Update direction if evidence is strongly directional
        if evidence.confidence > belief.posterior * 0.8:
            belief.direction = evidence.direction

        # Mark as root candidate if it's hidden or significantly negative
        if evidence.is_hidden or (evidence.direction == Direction.NEGATIVE and evidence.magnitude > 0.5):
            belief.is_root_candidate = True

    else:
        # New variable — create belief from evidence
        state.beliefs[name] = BayesianBelief(
            variable_name=name,
            prior=evidence.confidence * 0.5,  # discount: not user-stated
            posterior=evidence.confidence,
            evidence_count=1,
            evidence_log=[
                f"[{source.framework.value}] New variable discovered: {evidence.description[:80]}"
            ],
            direction=evidence.direction,
            is_root_candidate=evidence.is_hidden,
        )


def _calculate_likelihood(belief: BayesianBelief, evidence: Variable) -> float:
    """
    Calculate how likely this evidence is given the current belief.

    High likelihood if:
    - Evidence direction matches belief direction (confirming)
    - Evidence magnitude is consistent with belief posterior
    - Evidence confidence is high

    Low likelihood if:
    - Evidence contradicts the belief (but this is still informative!)
    """
    # Direction alignment
    if evidence.direction == belief.direction:
        direction_factor = 0.8
    elif evidence.direction == Direction.NEUTRAL:
        direction_factor = 0.5
    else:
        direction_factor = 0.3  # contradiction — low likelihood but informative

    # Magnitude consistency
    magnitude_factor = 1.0 - abs(evidence.magnitude - belief.posterior) * 0.5

    # Evidence quality
    quality_factor = evidence.confidence

    return (direction_factor * 0.4 + magnitude_factor * 0.3 + quality_factor * 0.3)


# ---------------------------------------------------------------------------
# Step 4: Latent Variable Inference
# ---------------------------------------------------------------------------

def _infer_latent_variables(state: BayesianState) -> list[Variable]:
    """
    Infer latent variables — hidden forces that must exist because
    the visible variables don't make sense without them.

    Detection methods:
    1. Beliefs with high evidence count but unstable posterior (something is pushing it around)
    2. Gaps in the direction profile (all positive but posterior is negative — missing negative)
    3. Multiple beliefs pointing to the same hidden cause
    """
    latent = []

    # Method 1: Unstable beliefs — something hidden is affecting them
    for name, belief in state.beliefs.items():
        if belief.evidence_count >= 3 and abs(belief.posterior - belief.prior) > 0.3:
            latent_var = Variable(
                name=f"latent_driver_{name}",
                description=(
                    f"Latent variable inferred: '{name}' has been updated "
                    f"{belief.evidence_count} times and shifted from "
                    f"prior={belief.prior:.2f} to posterior={belief.posterior:.2f}. "
                    "Something unseen is driving this variable — the visible "
                    "evidence alone doesn't explain the magnitude of change."
                ),
                magnitude=abs(belief.posterior - belief.prior),
                direction=belief.direction,
                confidence=min(0.5 + belief.evidence_count * 0.05, 0.85),
                source_framework=FrameworkID.BAYESIAN,
                is_hidden=True,
                is_user_stated=False,
                evidence=[
                    f"Belief: {name}",
                    f"Prior: {belief.prior:.2f}",
                    f"Posterior: {belief.posterior:.2f}",
                    f"Evidence points: {belief.evidence_count}",
                    f"Shift magnitude: {abs(belief.posterior - belief.prior):.2f}",
                ],
            )
            latent.append(latent_var)

    # Method 2: Direction profile gap
    directions = [b.direction for b in state.beliefs.values()]
    has_positive = Direction.POSITIVE in directions
    has_negative = Direction.NEGATIVE in directions

    if has_positive and not has_negative:
        # All positive beliefs but no negatives — where's the resistance?
        latent.append(Variable(
            name="latent_hidden_resistance",
            description=(
                "Direction profile gap: all beliefs are positive but "
                "the system isn't resolving. There must be a hidden "
                "negative force that no framework has surfaced yet."
            ),
            magnitude=0.5,
            direction=Direction.NEGATIVE,
            confidence=0.5,
            source_framework=FrameworkID.BAYESIAN,
            is_hidden=True,
            is_user_stated=False,
            evidence=["All beliefs positive but system unresolved."],
        ))

    return latent
