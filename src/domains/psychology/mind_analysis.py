"""
Psychology Module 1: Mind Analysis (Detection)

Detects the psychological forces operating beneath the user's story.
Treats the mind as a physics system: bias is a drag coefficient,
dissonance is potential energy, motivated reasoning is anomalous motion.

3 Concepts:
1. Dual Process Theory (System 1 vs. System 2) — classifies thought speed and flags rationalization
2. Cognitive Dissonance (The Tension) — finds conflicting beliefs where Variable D hides
3. Motivated Reasoning (Goal-Oriented Bias) — detects one-sided data filtering

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
# Concept 1: Dual Process Theory
# ---------------------------------------------------------------------------

@dataclass
class SystemClassification:
    """Classification of a single variable as System 1 or System 2 origin."""
    variable_name: str
    system: str                     # "S1", "S2", or "S2_justifying_S1"
    confidence_score: float         # 0.0 to 1.0
    emotional_charge_score: float   # 0.0 to 1.0
    justification_depth: float     # 0.0 to 1.0
    flag: bool                      # True if S2_justifying_S1 detected


def run_dual_process(problem: Problem) -> tuple[Perspective, list[SystemClassification]]:
    """
    Dual Process Theory — classify each user variable as System 1 or System 2.

    System 1 (Fast/Intuitive): Where bias lives. Survival mechanism.
      Indicators: high emotional charge, high confidence without evidence, snap judgments.

    System 2 (Slow/Analytical): The reasoning center.
      Indicators: low emotional charge, evidence cited, measured confidence.

    FLAG: S2 justifying S1 — elaborate justification of an emotional position.
      This is the most dangerous pattern: the user THINKS they're being logical.
    """
    classifications = []
    variables_found = []

    for var in problem.variables:
        if not var.is_user_stated:
            continue

        # Score emotional charge from variable properties
        # High magnitude + user stated + high confidence with thin evidence = emotional
        emotional_charge = _score_emotional_charge(var)

        # Score justification depth
        # Lots of evidence + elaborate description = deep justification
        justification_depth = _score_justification_depth(var)

        # Classify
        system, flag = _classify_system(
            var.confidence, emotional_charge, justification_depth
        )

        classification = SystemClassification(
            variable_name=var.name,
            system=system,
            confidence_score=var.confidence,
            emotional_charge_score=emotional_charge,
            justification_depth=justification_depth,
            flag=flag,
        )
        classifications.append(classification)

        # Create a variable for the manifold
        if flag:
            # S2 justifying S1 — this is a significant finding
            variables_found.append(Variable(
                name=f"rationalization_{var.name}",
                description=(
                    f"Post-hoc rationalization detected on '{var.name}': "
                    f"emotional charge is high ({emotional_charge:.2f}) but user "
                    f"has constructed elaborate justification (depth: {justification_depth:.2f}). "
                    "System 2 is being used to defend a System 1 impulse. "
                    "The user THINKS this is a calculated decision — it isn't."
                ),
                magnitude=emotional_charge,
                direction=var.direction,
                confidence=0.7,
                source_framework=FrameworkID.DUAL_PROCESS,
                is_hidden=True,
                is_user_stated=False,
                evidence=[
                    f"Variable: {var.name}",
                    f"Emotional charge: {emotional_charge:.2f}",
                    f"Justification depth: {justification_depth:.2f}",
                    f"User confidence: {var.confidence:.2f}",
                    "Pattern: high emotion + elaborate justification = rationalization",
                ],
            ))
        elif system == "S1":
            variables_found.append(Variable(
                name=f"impulse_{var.name}",
                description=(
                    f"System 1 impulse detected on '{var.name}': "
                    f"emotional charge {emotional_charge:.2f}, "
                    f"minimal justification ({justification_depth:.2f}). "
                    "This is a fast, intuitive reaction — not a calculated position."
                ),
                magnitude=emotional_charge * 0.7,
                direction=var.direction,
                confidence=0.6,
                source_framework=FrameworkID.DUAL_PROCESS,
                is_hidden=False,
                is_user_stated=False,
                evidence=[
                    f"Variable: {var.name}",
                    f"Emotional charge: {emotional_charge:.2f}",
                    f"Justification depth: {justification_depth:.2f}",
                    "Pattern: high emotion + low justification = impulse",
                ],
            ))

    # Build analysis text
    s1_count = sum(1 for c in classifications if c.system == "S1")
    s2_count = sum(1 for c in classifications if c.system == "S2")
    flag_count = sum(1 for c in classifications if c.flag)

    content = (
        "DUAL PROCESS ANALYSIS\n"
        f"System 1 (impulse): {s1_count} variables\n"
        f"System 2 (calculated): {s2_count} variables\n"
        f"S2 justifying S1 (rationalization): {flag_count} variables\n\n"
    )
    for c in classifications:
        marker = " *** FLAG ***" if c.flag else ""
        content += (
            f"  {c.variable_name}: {c.system} "
            f"(emotion: {c.emotional_charge_score:.2f}, "
            f"justification: {c.justification_depth:.2f}){marker}\n"
        )

    perspective = Perspective(
        framework=FrameworkID.DUAL_PROCESS,
        domain=Domain.PSYCHOLOGY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )

    return perspective, classifications


def _score_emotional_charge(var: Variable) -> float:
    """
    Score how emotionally charged a variable is.

    Indicators:
    - High magnitude (user cares a lot about this)
    - High confidence without proportional evidence (gut feeling)
    - Extreme direction (strong positive or negative = emotional investment)
    """
    magnitude_factor = var.magnitude  # 0-1

    # Confidence without evidence = gut feeling = emotional
    evidence_ratio = len(var.evidence) / max(var.confidence * 5, 1)
    gut_feeling_factor = max(0, 1.0 - evidence_ratio)

    # Extreme direction = emotional investment
    direction_factor = 0.7 if var.direction in (Direction.POSITIVE, Direction.NEGATIVE) else 0.3

    return (magnitude_factor * 0.4 + gut_feeling_factor * 0.35 + direction_factor * 0.25)


def _score_justification_depth(var: Variable) -> float:
    """
    Score how deeply the user has justified this variable.

    Deep justification = lots of evidence, detailed description.
    """
    evidence_score = min(len(var.evidence) / 4.0, 1.0)
    description_length = min(len(var.description) / 100.0, 1.0)

    return evidence_score * 0.6 + description_length * 0.4


def _classify_system(
    confidence: float,
    emotional_charge: float,
    justification_depth: float,
) -> tuple[str, bool]:
    """
    Classify as S1, S2, or S2_justifying_S1.

    Rules (from spec):
    - High confidence + high emotion + user_stated = S1
    - High confidence + low emotion + evidence = S2
    - High confidence + high emotion + elaborate justification = S2 justifying S1 (FLAG)
    """
    is_emotional = emotional_charge > 0.5
    is_justified = justification_depth > 0.4
    is_confident = confidence > 0.6

    # The dangerous pattern: emotional AND heavily justified
    if is_emotional and is_justified and is_confident:
        return "S2_justifying_S1", True

    # Pure System 1: emotional without justification
    if is_emotional and not is_justified:
        return "S1", False

    # System 2: low emotion, justified
    if not is_emotional and is_justified:
        return "S2", False

    # Default: moderate — classify by dominant signal
    if emotional_charge > justification_depth:
        return "S1", False
    return "S2", False


# ---------------------------------------------------------------------------
# Concept 2: Cognitive Dissonance
# ---------------------------------------------------------------------------

@dataclass
class DissonancePair:
    """A pair of conflicting beliefs with tension score."""
    var_a: str
    var_b: str
    tension_score: float                # 0.0 to 1.0
    gap_description: str
    resolution_strategy: str            # "denial", "minimization", "compartmentalization", "none"
    variable_d_candidate: str           # description of what might be hiding in the gap


def run_cognitive_dissonance(
    problem: Problem,
    physics_contradictions: list[Variable] | None = None,
) -> tuple[Perspective, list[DissonancePair]]:
    """
    Cognitive Dissonance — find conflicting belief pairs.

    The gap between conflicting beliefs is where Variable D hides.
    This is the Potential Energy of psychology — high dissonance = stored
    pressure about to release.

    Uses Physics Phase 2 contradiction findings as additional input
    (received via bridge, not direct import).
    """
    pairs = []
    variables_found = []

    user_vars = [v for v in problem.variables if v.is_user_stated]

    # Find semantically opposing variable pairs
    for i, va in enumerate(user_vars):
        for vb in user_vars[i + 1:]:
            tension = _calculate_tension(va, vb)

            if tension > 0.3:
                gap = _describe_gap(va, vb, tension)
                resolution = _detect_resolution_strategy(va, vb, problem)
                d_candidate = _hypothesize_variable_d(va, vb, tension)

                pair = DissonancePair(
                    var_a=va.name,
                    var_b=vb.name,
                    tension_score=tension,
                    gap_description=gap,
                    resolution_strategy=resolution,
                    variable_d_candidate=d_candidate,
                )
                pairs.append(pair)

                # High tension pairs produce hidden variables
                if tension > 0.5:
                    variables_found.append(Variable(
                        name=f"dissonance_{va.name}_vs_{vb.name}",
                        description=(
                            f"Cognitive dissonance detected: '{va.name}' and '{vb.name}' "
                            f"conflict with tension score {tension:.2f}. "
                            f"Gap: {gap}. "
                            f"Resolution strategy: {resolution}. "
                            f"Variable D candidate: {d_candidate}"
                        ),
                        magnitude=tension,
                        direction=Direction.NEUTRAL,
                        confidence=0.7,
                        source_framework=FrameworkID.COGNITIVE_DISSONANCE,
                        is_hidden=True,
                        is_user_stated=False,
                        evidence=[
                            f"Belief A: {va.name} ({va.direction.value}, {va.magnitude:.2f})",
                            f"Belief B: {vb.name} ({vb.direction.value}, {vb.magnitude:.2f})",
                            f"Tension: {tension:.2f}",
                            f"Resolution strategy: {resolution}",
                            f"Variable D candidate: {d_candidate}",
                        ],
                    ))

    # Cross-reference with physics contradiction findings if available
    if physics_contradictions:
        for pc in physics_contradictions:
            # Physics found something that contradicts user's picture
            # This amplifies any existing dissonance on the same variable
            for pair in pairs:
                if pc.name in pair.var_a or pc.name in pair.var_b:
                    # Physics confirms this tension is real
                    pair.tension_score = min(pair.tension_score + 0.15, 1.0)

    # Sort by tension (highest first)
    pairs.sort(key=lambda p: p.tension_score, reverse=True)

    # Overall dissonance energy
    total_energy = sum(p.tension_score for p in pairs)

    content = (
        "COGNITIVE DISSONANCE ANALYSIS\n"
        f"Conflicting pairs found: {len(pairs)}\n"
        f"Total dissonance energy: {total_energy:.2f}\n"
        f"Highest tension: {pairs[0].tension_score:.2f} ({pairs[0].var_a} vs {pairs[0].var_b})\n\n"
        if pairs else
        "COGNITIVE DISSONANCE ANALYSIS\nNo significant conflicting beliefs detected.\n"
    )
    for p in pairs:
        content += (
            f"  {p.var_a} vs {p.var_b}: tension={p.tension_score:.2f}\n"
            f"    Gap: {p.gap_description}\n"
            f"    Resolution: {p.resolution_strategy}\n"
            f"    Variable D: {p.variable_d_candidate}\n\n"
        )

    perspective = Perspective(
        framework=FrameworkID.COGNITIVE_DISSONANCE,
        domain=Domain.PSYCHOLOGY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.9,
    )

    return perspective, pairs


def _calculate_tension(va: Variable, vb: Variable) -> float:
    """
    Calculate tension between two variables.

    High tension when:
    - Opposite directions at similar magnitudes (direct conflict)
    - Both have high confidence (user holds both beliefs firmly)
    - Both are user-stated (internal conflict, not inferred)
    """
    # Direction opposition
    if va.direction == vb.direction:
        direction_tension = 0.1  # same direction = low tension
    elif (va.direction == Direction.POSITIVE and vb.direction == Direction.NEGATIVE) or \
         (va.direction == Direction.NEGATIVE and vb.direction == Direction.POSITIVE):
        direction_tension = 0.8  # direct opposition
    else:
        direction_tension = 0.3  # one neutral

    # Magnitude similarity (similar magnitudes = stronger conflict)
    magnitude_tension = 1.0 - abs(va.magnitude - vb.magnitude)

    # Confidence of both (high confidence in conflicting beliefs = high dissonance)
    confidence_tension = (va.confidence + vb.confidence) / 2

    return (direction_tension * 0.45 + magnitude_tension * 0.25 + confidence_tension * 0.3)


def _describe_gap(va: Variable, vb: Variable, tension: float) -> str:
    """Describe the gap between two conflicting variables."""
    if va.direction == Direction.POSITIVE and vb.direction == Direction.NEGATIVE:
        return (
            f"User holds positive belief about '{va.name}' "
            f"and negative belief about '{vb.name}' simultaneously. "
            f"These pull in opposite directions with tension {tension:.2f}."
        )
    elif va.direction == vb.direction:
        return (
            f"Both '{va.name}' and '{vb.name}' point {va.direction.value} "
            f"but at different magnitudes ({va.magnitude:.2f} vs {vb.magnitude:.2f}). "
            "The inconsistency suggests incomplete understanding."
        )
    return (
        f"'{va.name}' ({va.direction.value}) and '{vb.name}' ({vb.direction.value}) "
        f"create cognitive tension at score {tension:.2f}."
    )


def _detect_resolution_strategy(
    va: Variable, vb: Variable, problem: Problem
) -> str:
    """
    Detect how the user is resolving the dissonance.

    - Denial: one variable has very low confidence (user is dismissing it)
    - Minimization: one has artificially low magnitude (downplaying)
    - Compartmentalization: both high confidence, no acknowledgment of conflict
    - None: user acknowledges the tension
    """
    # Denial: dramatically different confidence levels
    if abs(va.confidence - vb.confidence) > 0.4:
        return "denial"

    # Minimization: one magnitude is suspiciously low relative to the other
    if va.magnitude > 0.6 and vb.magnitude < 0.3:
        return "minimization"
    if vb.magnitude > 0.6 and va.magnitude < 0.3:
        return "minimization"

    # Compartmentalization: both high confidence, high magnitude, no resolution
    if va.confidence > 0.6 and vb.confidence > 0.6 and va.magnitude > 0.5 and vb.magnitude > 0.5:
        return "compartmentalization"

    return "none"


def _hypothesize_variable_d(va: Variable, vb: Variable, tension: float) -> str:
    """Hypothesize what Variable D might be hiding in the dissonance gap."""
    if tension > 0.7:
        return (
            f"The high tension between '{va.name}' and '{vb.name}' suggests "
            "a fundamental unresolved truth the user cannot face. "
            "Variable D is likely the resolution they're avoiding — "
            "the choice or admission that would dissolve both beliefs."
        )
    elif tension > 0.5:
        return (
            f"Moderate tension between '{va.name}' and '{vb.name}'. "
            "Variable D may be a missing piece of information that "
            "would reconcile these beliefs — or prove one of them wrong."
        )
    return (
        f"Low tension between '{va.name}' and '{vb.name}'. "
        "The conflict is present but not severe. "
        "Variable D may emerge if the situation intensifies."
    )


# ---------------------------------------------------------------------------
# Concept 3: Motivated Reasoning
# ---------------------------------------------------------------------------

@dataclass
class MotivatedReasoningAssessment:
    """Assessment of goal-oriented bias in user's data presentation."""
    directional_bias_score: float       # 0.0 to 1.0 (>0.8 = one-sided)
    likely_pre_set_conclusion: str | None
    missing_counter_evidence: list[str]
    filter_patterns: list[str]
    motivation_source_hypothesis: str | None


def run_motivated_reasoning(
    problem: Problem,
    physics_anomalies: list[Variable] | None = None,
) -> tuple[Perspective, MotivatedReasoningAssessment]:
    """
    Motivated Reasoning — detect goal-oriented bias.

    The user acts as Lawyer (defending position) not Scientist (searching truth).
    They subconsciously filter data to show only what supports their position.

    Cross-references with Physics Anomalous Motion — if physics detected wobble,
    motivated reasoning may explain WHY.
    """
    variables_found = []
    user_vars = [v for v in problem.variables if v.is_user_stated]

    # Calculate directional bias
    positive_count = sum(1 for v in user_vars if v.direction == Direction.POSITIVE)
    negative_count = sum(1 for v in user_vars if v.direction == Direction.NEGATIVE)
    total = len(user_vars) or 1

    # How one-sided is the presentation?
    if positive_count > negative_count:
        dominant_direction = Direction.POSITIVE
        bias_score = positive_count / total
    elif negative_count > positive_count:
        dominant_direction = Direction.NEGATIVE
        bias_score = negative_count / total
    else:
        dominant_direction = Direction.NEUTRAL
        bias_score = 0.5

    # Detect pre-set conclusion
    pre_set = None
    if bias_score > 0.7:
        if dominant_direction == Direction.POSITIVE:
            pre_set = (
                "User appears to have already decided this situation is GOOD/RIGHT. "
                "Evidence is curated to support this conclusion."
            )
        elif dominant_direction == Direction.NEGATIVE:
            pre_set = (
                "User appears to have already decided this situation is BAD/WRONG. "
                "Evidence is curated to support this conclusion."
            )

    # Detect missing counter-evidence
    missing = []
    if dominant_direction == Direction.POSITIVE and negative_count < 2:
        missing.append(
            "Very few negative variables stated despite presenting a 'problem'. "
            "If everything is positive, why is there a problem at all? "
            "Missing: the downsides the user isn't mentioning."
        )
    if dominant_direction == Direction.NEGATIVE and positive_count < 2:
        missing.append(
            "Very few positive variables stated. The user sees no bright spots. "
            "Missing: the resources, strengths, or options the user is ignoring."
        )

    # Detect high-magnitude variables with suspiciously high confidence
    for var in user_vars:
        if var.magnitude > 0.7 and var.confidence > 0.8 and len(var.evidence) < 2:
            missing.append(
                f"'{var.name}' has high magnitude ({var.magnitude:.2f}) and high confidence "
                f"({var.confidence:.2f}) but minimal evidence ({len(var.evidence)} points). "
                "This looks like conviction without basis — a key ingredient of motivated reasoning."
            )

    # Detect filter patterns
    filters = []
    # Pattern: acknowledge-then-dismiss
    for var in user_vars:
        if var.direction != dominant_direction and var.magnitude < 0.3 and var.confidence > 0.5:
            filters.append(
                f"'{var.name}' is acknowledged but minimized (magnitude: {var.magnitude:.2f}). "
                "User knows this factor exists but has artificially reduced its weight."
            )

    # Motivation source hypothesis
    motivation = None
    if bias_score > 0.7:
        motivation = (
            f"The user's story is {bias_score:.0%} skewed toward {dominant_direction.value}. "
            "Hypothesis: the user is protecting a decision they've already made, "
            "or avoiding a truth that contradicts their preferred narrative."
        )

    # Cross-reference with physics anomalies
    if physics_anomalies and bias_score > 0.6:
        for anomaly in physics_anomalies:
            filters.append(
                f"Physics detected anomalous motion in '{anomaly.name}': "
                f"{anomaly.description[:100]}. "
                "Motivated reasoning may explain why the user's story doesn't "
                "match the system's physics."
            )

    # Create variable if significant bias detected
    if bias_score > 0.65:
        variables_found.append(Variable(
            name="motivated_reasoning_bias",
            description=(
                f"Motivated reasoning detected: {bias_score:.0%} directional bias "
                f"toward {dominant_direction.value}. "
                f"{'Pre-set conclusion: ' + pre_set if pre_set else 'No clear pre-set conclusion.'} "
                f"Missing counter-evidence: {len(missing)} items."
            ),
            magnitude=bias_score,
            direction=Direction.NEGATIVE,
            confidence=min(bias_score, 0.85),
            source_framework=FrameworkID.MOTIVATED_REASONING,
            is_hidden=True,
            is_user_stated=False,
            evidence=[
                f"Directional bias: {bias_score:.2f}",
                f"Dominant direction: {dominant_direction.value}",
                f"Positive variables: {positive_count}",
                f"Negative variables: {negative_count}",
                *[f"Missing: {m[:80]}" for m in missing[:3]],
            ],
        ))

    assessment = MotivatedReasoningAssessment(
        directional_bias_score=bias_score,
        likely_pre_set_conclusion=pre_set,
        missing_counter_evidence=missing,
        filter_patterns=filters,
        motivation_source_hypothesis=motivation,
    )

    content = (
        "MOTIVATED REASONING ANALYSIS\n"
        f"Directional bias score: {bias_score:.2f}\n"
        f"Dominant direction: {dominant_direction.value}\n"
        f"Pre-set conclusion: {'Yes' if pre_set else 'No'}\n"
        f"Missing counter-evidence: {len(missing)} items\n"
        f"Filter patterns detected: {len(filters)}\n\n"
    )
    if pre_set:
        content += f"CONCLUSION DETECTED: {pre_set}\n\n"
    for m in missing:
        content += f"  MISSING: {m}\n"
    for f in filters:
        content += f"  FILTER: {f}\n"

    perspective = Perspective(
        framework=FrameworkID.MOTIVATED_REASONING,
        domain=Domain.PSYCHOLOGY,
        content=content,
        variables_found=variables_found,
        signal_type=SignalType.SIGNAL,
        weight=0.85,
    )

    return perspective, assessment
