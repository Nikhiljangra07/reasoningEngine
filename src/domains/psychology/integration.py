"""
Psychology Module 2: Integration (Synthesis + Self-Awareness)

Takes the detection findings from Module 1 and the analytical findings from
Physics/Math (received via bridge) and produces:
- Dialectical synthesis (thesis + antithesis → integrated truth)
- Metacognition assessment (how much self-awareness the user has → delivery calibration)

2 Concepts:
4. Dialectical Thinking (The Integration) — synthesizes opposing views on the PERSON's experience
5. Metacognition (The Observer) — assesses self-awareness, calibrates delivery

NOTE: Psychology's Dialectical Thinking operates on the PERSON's internal experience.
      Philosophy's Dialectics operates on the SITUATION's structure.
      Same tool, different target.

ISOLATION: Imports ONLY from src.core.types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.types import (
    Direction,
    Domain,
    DomainOutput,
    FrameworkID,
    Perspective,
    Problem,
    RootCause,
    SignalType,
    Variable,
)


# ---------------------------------------------------------------------------
# Concept 4: Dialectical Thinking
# ---------------------------------------------------------------------------

@dataclass
class DialecticalSynthesis:
    """Result of thesis/antithesis/synthesis on a root cause."""
    root_cause_name: str
    thesis: str                         # user's view
    antithesis: str                     # physics/math view
    shared_ground: str                  # what both agree on
    synthesis: str                      # the integrated truth
    synthesis_confidence: float         # 0.0 to 1.0
    new_contradictions: list[str]       # contradictions the synthesis introduces


def run_dialectical_thinking(
    problem: Problem,
    root_causes: list[RootCause],
    upstream_outputs: dict[str, DomainOutput] | None = None,
) -> tuple[Perspective, list[DialecticalSynthesis]]:
    """
    Dialectical Thinking — for each root cause, generate thesis (user's view)
    and antithesis (physics/math view), then find the synthesis.

    The synthesis is the integrated truth that neither perspective alone
    could produce. The counter-argument STRENGTHENS the final solution.
    """
    syntheses = []
    variables_found = []

    for rc in root_causes:
        # Extract thesis: what does the USER believe about this root cause?
        thesis = _extract_thesis(problem, rc)

        # Extract antithesis: what do the DOMAINS say that contradicts the user?
        antithesis = _extract_antithesis(rc, upstream_outputs)

        # Find shared ground
        shared = _find_shared_ground(thesis, antithesis, rc)

        # Generate synthesis
        synthesis_text, confidence = _generate_synthesis(thesis, antithesis, shared, rc)

        # Check for new contradictions the synthesis creates
        new_contradictions = _check_synthesis_stability(synthesis_text, problem, rc)

        ds = DialecticalSynthesis(
            root_cause_name=rc.variable.name,
            thesis=thesis,
            antithesis=antithesis,
            shared_ground=shared,
            synthesis=synthesis_text,
            synthesis_confidence=confidence,
            new_contradictions=new_contradictions,
        )
        syntheses.append(ds)

        # Create variable for manifold
        variables_found.append(Variable(
            name=f"synthesis_{rc.variable.name}",
            description=(
                f"Dialectical synthesis on '{rc.variable.name}': "
                f"{synthesis_text} "
                f"(confidence: {confidence:.2f})"
            ),
            magnitude=confidence,
            direction=Direction.NEUTRAL,  # synthesis transcends direction
            confidence=confidence,
            source_framework=FrameworkID.DIALECTICAL_THINKING,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Thesis (user): {thesis[:80]}",
                f"Antithesis (domains): {antithesis[:80]}",
                f"Shared ground: {shared[:80]}",
                f"Synthesis: {synthesis_text[:80]}",
                f"New contradictions: {len(new_contradictions)}",
            ],
        ))

    content = "DIALECTICAL THINKING ANALYSIS\n\n"
    for ds in syntheses:
        content += (
            f"--- {ds.root_cause_name} ---\n"
            f"  Thesis (user): {ds.thesis}\n"
            f"  Antithesis (domains): {ds.antithesis}\n"
            f"  Shared ground: {ds.shared_ground}\n"
            f"  SYNTHESIS: {ds.synthesis}\n"
            f"  Confidence: {ds.synthesis_confidence:.2f}\n"
        )
        if ds.new_contradictions:
            content += f"  NEW CONTRADICTIONS: {len(ds.new_contradictions)}\n"
            for nc in ds.new_contradictions:
                content += f"    - {nc}\n"
        content += "\n"

    perspective = Perspective(
        framework=FrameworkID.DIALECTICAL_THINKING,
        domain=Domain.PSYCHOLOGY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )

    return perspective, syntheses


def _extract_thesis(problem: Problem, rc: RootCause) -> str:
    """Extract the user's view on this root cause."""
    # Find user variables that relate to this root cause
    related = [
        v for v in problem.variables
        if v.is_user_stated and (
            v.name in rc.variable.name
            or rc.variable.name in v.name
            or any(v.name in e for e in rc.evidence_chain)
        )
    ]

    if related:
        user_view = related[0]
        return (
            f"The user sees '{user_view.name}' as {user_view.direction.value} "
            f"(magnitude: {user_view.magnitude:.2f}, confidence: {user_view.confidence:.2f}). "
            f"Their framing: {user_view.description}"
        )

    return (
        f"The user has not directly addressed '{rc.variable.name}'. "
        "Their silence on this topic may itself be significant."
    )


def _extract_antithesis(
    rc: RootCause,
    upstream_outputs: dict[str, DomainOutput] | None,
) -> str:
    """Extract the domain findings that contradict the user's view."""
    parts = []

    # Root cause's own evidence chain contains the domain's view
    if rc.evidence_chain:
        parts.append(f"Domain analysis: {rc.evidence_chain[-1]}")

    if rc.bias_that_hid_it:
        parts.append(f"Bias identified: {rc.bias_that_hid_it}")

    if rc.variable.is_hidden:
        parts.append(
            f"This root cause was HIDDEN — the user did not and could not "
            f"see '{rc.variable.name}' from their current frame."
        )

    if parts:
        return " | ".join(parts)

    return (
        f"Domain analysis identified '{rc.variable.name}' as a root cause "
        f"with {rc.confidence:.0%} confidence, which the user did not identify."
    )


def _find_shared_ground(thesis: str, antithesis: str, rc: RootCause) -> str:
    """Find what both the user's view and domain view agree on."""
    # The shared ground is often the EXISTENCE of the problem —
    # both agree something is wrong, they disagree on WHAT or WHY
    return (
        f"Both perspectives agree that '{rc.variable.name}' is significant "
        f"(magnitude: {rc.variable.magnitude:.2f}). "
        "The disagreement is about its nature, cause, or direction — not its importance."
    )


def _generate_synthesis(
    thesis: str, antithesis: str, shared: str, rc: RootCause
) -> tuple[str, float]:
    """
    Generate the dialectical synthesis.

    The synthesis integrates valid elements of both thesis and antithesis
    into a truth that neither alone could produce.
    """
    # Confidence is based on root cause confidence + whether bias was identified
    base_confidence = rc.confidence * 0.7
    bias_bonus = 0.15 if rc.bias_that_hid_it else 0.0
    hidden_bonus = 0.1 if rc.variable.is_hidden else 0.0
    confidence = min(base_confidence + bias_bonus + hidden_bonus, 0.95)

    synthesis = (
        f"The user's experience of '{rc.variable.name}' is valid as EXPERIENCE — "
        f"but the domain analysis reveals a deeper structure. "
    )

    if rc.bias_that_hid_it:
        synthesis += (
            f"The bias ({rc.bias_that_hid_it}) explains WHY the user sees it "
            f"the way they do. The integrated truth: the user's feeling is real, "
            f"but its source is not what they think. The actual driver is "
            f"'{rc.variable.description[:80]}' operating beneath awareness."
        )
    else:
        synthesis += (
            f"The user's partial view and the domain's analytical view converge on "
            f"'{rc.variable.name}' as a central factor. The integrated truth: "
            f"the user is right that this matters, but may be wrong about "
            f"why it matters or where it leads."
        )

    return synthesis, confidence


def _check_synthesis_stability(
    synthesis: str, problem: Problem, rc: RootCause
) -> list[str]:
    """
    Check if the synthesis creates new contradictions.

    If it does, flag for another dialectical pass.
    """
    contradictions = []

    # Check: does the synthesis contradict any user variable?
    for var in problem.variables:
        if var.is_user_stated and var.direction == Direction.POSITIVE and rc.variable.direction == Direction.NEGATIVE:
            if var.confidence > 0.8:
                contradictions.append(
                    f"Synthesis may conflict with user's strongly-held positive belief "
                    f"in '{var.name}' (confidence: {var.confidence:.2f}). "
                    "Delivering this synthesis may face resistance."
                )

    return contradictions


# ---------------------------------------------------------------------------
# Concept 5: Metacognition
# ---------------------------------------------------------------------------

@dataclass
class MetacognitionAssessment:
    """Assessment of user's self-awareness capacity."""
    metacognition_score: float          # 0.0 to 1.0
    acknowledges_uncertainty: float     # 0.0 to 1.0
    presents_both_sides: float          # 0.0 to 1.0
    references_own_role: float          # 0.0 to 1.0
    receptivity_to_challenge: float     # 0.0 to 1.0
    recommended_delivery_mode: str      # "direct", "building", or "gentle"


def run_metacognition(
    problem: Problem,
    all_findings: list[Variable] | None = None,
) -> tuple[Perspective, MetacognitionAssessment]:
    """
    Metacognition — assess the user's capacity for self-awareness.

    Determines HOW to deliver findings:
    - Score > 0.7: HIGH → "Here's what you're not seeing."
    - Score 0.4-0.7: MEDIUM → "Let's look at this from another angle."
    - Score < 0.4: LOW → "I notice something interesting in what you've described."

    The answer is the same. The delivery is calibrated to the human.
    """
    user_vars = [v for v in problem.variables if v.is_user_stated]

    # Factor 1: Acknowledges uncertainty
    # Does the user have any variables with low confidence? (they admit doubt)
    uncertain_vars = [v for v in user_vars if v.confidence < 0.5]
    acknowledges_uncertainty = min(len(uncertain_vars) / max(len(user_vars), 1), 1.0)

    # Factor 2: Presents both sides
    # Does the user voluntarily offer variables in BOTH directions?
    has_positive = any(v.direction == Direction.POSITIVE for v in user_vars)
    has_negative = any(v.direction == Direction.NEGATIVE for v in user_vars)
    if has_positive and has_negative:
        pos_count = sum(1 for v in user_vars if v.direction == Direction.POSITIVE)
        neg_count = sum(1 for v in user_vars if v.direction == Direction.NEGATIVE)
        total = pos_count + neg_count
        # Balance: closer to 50/50 = higher score
        balance = 1.0 - abs(pos_count - neg_count) / total
        presents_both_sides = balance
    elif has_positive or has_negative:
        presents_both_sides = 0.2  # only one side
    else:
        presents_both_sides = 0.5  # all neutral

    # Factor 3: References own role
    # Does the user include themselves as a causal factor?
    self_reference_keywords = [
        "i ", "my ", "me ", "myself", "i'm", "i've", "my fault",
        "i should", "i could", "i did", "i didn't",
    ]
    statement_lower = problem.statement.lower()
    self_refs = sum(1 for kw in self_reference_keywords if kw in statement_lower)
    # Check if any negative variable references the user as cause
    user_as_cause = any(
        v.direction == Direction.NEGATIVE and
        any(kw in v.description.lower() for kw in ["i ", "my ", "myself"])
        for v in user_vars
    )
    references_own_role = min(self_refs / 5.0 + (0.3 if user_as_cause else 0.0), 1.0)

    # Factor 4: Receptivity to challenge
    # Proxy: does the user present neutral/uncertain variables alongside confident ones?
    # High receptivity = comfortable with ambiguity
    neutral_count = sum(1 for v in user_vars if v.direction == Direction.NEUTRAL)
    mixed_confidence = any(v.confidence < 0.5 for v in user_vars) and any(v.confidence > 0.7 for v in user_vars)
    receptivity = (
        min(neutral_count / max(len(user_vars), 1), 0.5)
        + (0.3 if mixed_confidence else 0.0)
        + (0.2 if acknowledges_uncertainty > 0.3 else 0.0)
    )
    receptivity = min(receptivity, 1.0)

    # Overall metacognition score
    meta_score = (
        acknowledges_uncertainty * 0.25
        + presents_both_sides * 0.25
        + references_own_role * 0.25
        + receptivity * 0.25
    )

    # Delivery mode
    if meta_score > 0.7:
        delivery = "direct"
    elif meta_score > 0.4:
        delivery = "building"
    else:
        delivery = "gentle"

    assessment = MetacognitionAssessment(
        metacognition_score=meta_score,
        acknowledges_uncertainty=acknowledges_uncertainty,
        presents_both_sides=presents_both_sides,
        references_own_role=references_own_role,
        receptivity_to_challenge=receptivity,
        recommended_delivery_mode=delivery,
    )

    # Variable for manifold
    variables_found = [Variable(
        name="metacognition_level",
        description=(
            f"User metacognition: {meta_score:.2f} ({delivery} delivery recommended). "
            f"Uncertainty acknowledgment: {acknowledges_uncertainty:.2f}, "
            f"Both sides: {presents_both_sides:.2f}, "
            f"Self-reference: {references_own_role:.2f}, "
            f"Receptivity: {receptivity:.2f}."
        ),
        magnitude=meta_score,
        direction=Direction.NEUTRAL,
        confidence=0.75,
        source_framework=FrameworkID.METACOGNITION,
        is_hidden=False,
        is_user_stated=False,
        evidence=[
            f"Metacognition score: {meta_score:.2f}",
            f"Delivery mode: {delivery}",
            f"Acknowledges uncertainty: {acknowledges_uncertainty:.2f}",
            f"Presents both sides: {presents_both_sides:.2f}",
            f"References own role: {references_own_role:.2f}",
            f"Receptivity to challenge: {receptivity:.2f}",
        ],
    )]

    content = (
        "METACOGNITION ASSESSMENT\n"
        f"Overall score: {meta_score:.2f}\n"
        f"Recommended delivery: {delivery.upper()}\n\n"
        f"  Acknowledges uncertainty: {acknowledges_uncertainty:.2f}\n"
        f"  Presents both sides: {presents_both_sides:.2f}\n"
        f"  References own role: {references_own_role:.2f}\n"
        f"  Receptivity to challenge: {receptivity:.2f}\n\n"
        f"Delivery calibration:\n"
    )
    if delivery == "direct":
        content += "  → HIGH self-awareness. Deliver directly: 'Here's what you're not seeing.'\n"
    elif delivery == "building":
        content += "  → MEDIUM self-awareness. Build toward: 'Let's look at this from another angle.'\n"
    else:
        content += "  → LOW self-awareness. Gentle reveal: 'I notice something interesting...'\n"

    perspective = Perspective(
        framework=FrameworkID.METACOGNITION,
        domain=Domain.PSYCHOLOGY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.8,
    )

    return perspective, assessment
