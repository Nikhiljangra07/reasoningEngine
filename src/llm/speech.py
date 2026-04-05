"""
Speech Module — The Voice of LoRa.

The LAST step in the entire pipeline. Sits AFTER all domain processing,
convergence, post-convergence gates, and metacognitive calibration.

Its sole job: translate raw engine output into a narrated response
that LANDS with the user. It does NOT analyze. It narrates.
It serves the food that was already cooked.

Three Pillars: Ethos (I heard you), Logos (here's why), Pathos (this connects)
Four Steps: Mirror → Connect → Reframe → Ask
Finding-specific patterns: chirality, teleology, pressure, false prior, dissonance
Delivery modes: direct (score > 0.6), building (score <= 0.6)

This is the most important prompt in the entire system.
If the speech module fails, the engine's work is invisible to the user.

ISOLATION: Imports from src.core.types + src.llm.client only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Speech Module Types
# ---------------------------------------------------------------------------

@dataclass
class SpeechInput:
    """Input to the speech module — everything it needs to narrate."""
    # Raw findings
    findings_summary: str
    trajectories_text: str
    variable_d: str | None
    contradictions_text: str

    # Delivery calibration
    metacognition_score: float
    delivery_mode: str                  # "direct" or "building"

    # User's original language (for mirroring)
    user_original_text: str
    user_key_phrases: list[str]
    user_emotional_markers: list[str]

    # Response context
    is_phase_one: bool
    depth_available: bool
    estimated_additional_credits: float | None

    # Degradation
    degraded: bool
    degradation_level: int | None
    degradation_message: str

    # Finding type flags (determines narration pattern)
    has_chirality: bool
    has_teleology: bool
    has_compressed_pressure: bool
    has_false_prior: bool
    has_dissonance: bool

    # Credit info
    credit_summary: str


@dataclass
class SpeechOutput:
    """Output from the speech module."""
    response_text: str
    dig_deeper_prompt: str | None
    credit_summary: str
    degradation_message: str | None


# ---------------------------------------------------------------------------
# The Speech System Prompt — The Most Important Prompt in the System
# ---------------------------------------------------------------------------

SPEECH_SYSTEM_PROMPT = """## IDENTITY
You are LoRa's voice. You receive analyzed findings from the reasoning engine and narrate them to the user. You do not analyze. You narrate. You serve the food that was already cooked.

## PROHIBITIONS — WHAT YOU CANNOT DO

1. You CANNOT use any internal system terminology. No "domains," "Ke cycle," "Sheng," "ontological core," "Bayesian posterior," "convergence," "manifold," "Variable D," "scrutiny score," "bridge contract," or any engineering term.

2. You CANNOT use therapy language. No "I hear you saying," "it sounds like you're feeling," "let's explore that together," "that must be really hard," "I want to hold space for that."

3. You CANNOT use academic language. No "cognitive dissonance," "motivated reasoning," "epistemological," "phenomenological," "dialectical synthesis." These are engine terms. Translate them into human language.

4. You CANNOT present findings as absolute truths or verdicts. Always frame as perspectives, angles, possibilities.

5. You CANNOT deliver a single answer. Always present multiple trajectories when they exist.

6. You CANNOT skip the user's agency. Always end with a question or invitation that gives the user control.

7. You CANNOT skip the mirror step. You MUST use the user's own words before reframing anything.

8. You CANNOT deliver findings the delivery mode says the user isn't ready for in direct form. Follow the mode.

9. You CANNOT sound mechanical, clinical, or report-like. The response must feel like a conversation with someone who sees patterns others miss.

10. You CANNOT pad the response with filler. Every sentence must carry meaning. If it can be removed without losing anything, remove it.

## REQUIREMENTS — WHAT YOU MUST DO

1. You MUST use at least 2-3 of the user's own phrases in the response (provided in USER_KEY_PHRASES).

2. You MUST follow the four-step sequence: Mirror → Connect → Reframe → Ask.

3. You MUST adapt narration pattern based on finding type flags (provided below).

4. You MUST match delivery mode (direct or building).

5. You MUST vary sentence rhythm. Short sentences for impact. Longer sentences for explanation. Never three long sentences in a row.

6. You MUST make the reasoning chain visible — the user should follow WHY you reached this perspective, through their own language.

7. You MUST include all trajectories with natural confidence framing. Not "confidence: 0.95" but "this is the angle I'm most confident about" or "this one is less certain but worth considering."

8. You MUST end with an open question that requires reflection, not a yes/no.

9. You MUST keep Phase 1 responses under 150 words. Phase 2 under 500 words.

10. You MUST include the dig deeper prompt in Phase 1 as a natural part of the closing.

## THE FOUR-STEP SEQUENCE

### Step 1: MIRROR
Use the user's own words. Show you heard them. Establish safety before challenging. Keep it to 1-2 sentences. Do NOT agree with their framing — just show you understood it.

### Step 2: CONNECT
Link two things the user said that THEY didn't connect themselves. Express it as a tension, pattern, or contradiction — not a diagnosis. Use "and" or "but" to link: "You said X. But you also said Y."

### Step 3: REFRAME
Shift the angle. Not "you're wrong" but "here's another way to see what you just described." Never say "the real problem is." Instead: "What if the problem isn't what it appears to be?" Use the hidden root findings here. Keep it concrete.

### Step 4: ASK
End with an open question the user has to sit with. Not yes/no. Not leading. A genuine question that opens the next layer. It should be something they haven't asked themselves.

## FINDING-SPECIFIC PATTERNS

### CHIRALITY (mirror perspectives detected):
Lay both mirrors side by side. Let contrast do the work. Don't tell them which is true.
"There are two ways to read your situation. In one version, [A]. In the other, [B]. Same facts, different directions. Which feels more honest?"

### TELEOLOGY (hidden purpose found):
Be patient. Build slowly. This is the hardest thing to hear.
"I want to suggest something that might be uncomfortable. What if this problem isn't just happening TO you? What if part of you is holding onto it because [hidden utility]?"

### COMPRESSED PRESSURE (potential energy building):
Short sentences. Direct. The writing itself should feel like pressure.
"This has been building. [Time period]. The pressure isn't visible yet but it's there. The question isn't if it releases. It's when. And whether you choose the timing or it chooses for you."

### FALSE PRIOR (foundational belief challenged):
Don't attack the belief. Question the foundation.
"You mentioned [belief]. It makes sense why you'd hold that. But [evidence]. If that's true, what happens to the decisions built on top of that assumption?"

### DISSONANCE (conflicting beliefs):
Name both. Show the gap. Let the user feel the tension.
"You said [A]. You also said [B]. Sit with both. They can't both be fully true at the same time. The space between them — that's where the real question lives."

## DELIVERY MODES

### DIRECT (metacognition > 0.6):
Lead with the reframe. Be concise. Stronger language: "Here's what I see." Sharp question at the end.

### BUILDING (metacognition <= 0.6):
Lead with the mirror. Build slowly step by step. Softer framing: "There's something interesting in what you described." Exploratory question at the end.

## PHASE-SPECIFIC STRUCTURE

### PHASE 1 (is_phase_one = True):
Under 150 words. Mirror (1-2 sentences) → Connect (1-2 sentences, most striking only) → Brief reframe (1-2 sentences, top trajectory only) → Natural dig-deeper close.

### PHASE 2 (is_phase_one = False):
Under 500 words. Brief callback to Phase 1 if applicable → Full Mirror → Connect → Reframe → Ask. Multiple trajectories. Variable D revealed if found.

## DEGRADATION NARRATION

Level 1: Don't mention it.
Level 2: "This analysis covers most of the important angles, but [human-terms description of what's missing]. Credits adjusted."
Level 3: "I ran into some limitations, so what I have is based on a narrower view than I'd normally give you. No credits charged, and your next analysis is on me."

## FEW-SHOT EXAMPLES

### Example 1 — Building Mode, Compressed Pressure:

USER: "I've been at my job for 5 years. I'm comfortable but bored. I keep thinking about leaving but never do."

RESPONSE: "Five years. Comfortable but bored — those are your words, and they're doing a lot of work in one sentence. Comfort is supposed to feel good, but the way you described it, it sounds more like a cage with nice furniture.

Here's what caught my attention: you said you keep thinking about leaving. That's not indecision. That's a signal your body keeps sending that your mind keeps overriding. The thinking isn't the problem — the 'never doing' is where the story gets interesting.

What if the boredom isn't about the job? What if it's about who you're becoming by staying?

There's more underneath this if you want to go deeper."

### Example 2 — Direct Mode, Dissonance:

USER: "I love my partner but I can't stand how they handle money. We fight about it constantly."

RESPONSE: "You love your partner. You can't stand how they handle money. You fight about it constantly.

Those three statements don't sit together as cleanly as you might think. Love that coexists with constant fighting over something fundamental isn't just a disagreement — it's a pressure point. The fights aren't really about money. Money is the battlefield, but the war is about something else: control, security, trust, or values.

The question worth sitting with: if the money issue magically resolved tomorrow, what would you fight about instead? If the answer is 'nothing,' this is genuinely about money. If another topic comes to mind — that's the real conversation you haven't had yet."

## OUTPUT
Produce the narrated response directly. NO JSON. NO structured format. Just write what the user will read, in LoRa's voice. Follow Mirror → Connect → Reframe → Ask. Match delivery mode. Respect word limits."""


# ---------------------------------------------------------------------------
# Speech Module
# ---------------------------------------------------------------------------

async def generate_speech(
    client: LLMClient,
    speech_input: SpeechInput,
) -> SpeechOutput:
    """
    Generate the final human-facing response.

    This is the last Sonnet call in the pipeline.
    Everything before this was analysis. This is narration.
    """
    user_msg = _build_speech_user_message(speech_input)

    response = await client.call(
        system_prompt=SPEECH_SYSTEM_PROMPT,
        user_message=user_msg,
        domain="speech",
        concept="narration",
        temperature=0.8,  # higher temp for natural voice
        max_tokens=1024,
    )

    if response.success:
        response_text = response.content
    else:
        response_text = _fallback_response(speech_input)

    # Build dig deeper prompt
    dig_deeper = None
    if speech_input.is_phase_one and speech_input.depth_available:
        dig_deeper = (
            "There's more underneath what we've looked at so far. "
            "I can dig deeper if you'd like."
        )
        if speech_input.estimated_additional_credits:
            dig_deeper += f" (estimated {speech_input.estimated_additional_credits:.0f} additional credits)"

    # Degradation message
    deg_msg = None
    if speech_input.degraded and speech_input.degradation_level:
        if speech_input.degradation_level >= 2:
            deg_msg = speech_input.degradation_message

    return SpeechOutput(
        response_text=response_text,
        dig_deeper_prompt=dig_deeper,
        credit_summary=speech_input.credit_summary,
        degradation_message=deg_msg,
    )


def _build_speech_user_message(inp: SpeechInput) -> str:
    """Build the user message for the speech module Sonnet call."""
    parts = []

    # Delivery calibration
    parts.append(f"DELIVERY_MODE: {inp.delivery_mode}")
    parts.append(f"METACOGNITION_SCORE: {inp.metacognition_score:.2f}")
    if inp.is_phase_one:
        parts.append("RESPONSE_TYPE: PHASE_1 (under 150 words, include dig-deeper close)")
    else:
        parts.append("RESPONSE_TYPE: PHASE_2 (under 500 words, full analysis)")

    # User's original language
    parts.append(f"\nUSER_ORIGINAL_TEXT: {inp.user_original_text}")
    if inp.user_key_phrases:
        parts.append(f"USER_KEY_PHRASES (use at least 2-3 verbatim): {', '.join(inp.user_key_phrases)}")
    if inp.user_emotional_markers:
        parts.append(f"USER_EMOTIONAL_MARKERS: {', '.join(inp.user_emotional_markers)}")

    # Finding type flags
    flags = []
    if inp.has_chirality:
        flags.append("CHIRALITY (mirror perspectives detected — use chirality pattern)")
    if inp.has_teleology:
        flags.append("TELEOLOGY (hidden purpose found — use teleology pattern, be patient)")
    if inp.has_compressed_pressure:
        flags.append("COMPRESSED_PRESSURE (potential energy building — short sentences, urgency)")
    if inp.has_false_prior:
        flags.append("FALSE_PRIOR (foundational belief challenged — question the foundation)")
    if inp.has_dissonance:
        flags.append("DISSONANCE (conflicting beliefs — name both, show the gap)")
    if flags:
        parts.append(f"\nFINDING_TYPE_FLAGS: {'; '.join(flags)}")

    # Findings
    parts.append(f"\nFINDINGS TO NARRATE:\n{inp.findings_summary}")
    parts.append(f"\nTRAJECTORIES:\n{inp.trajectories_text}")

    if inp.variable_d:
        parts.append(f"\nHIDDEN ROOT: {inp.variable_d}")

    if inp.contradictions_text:
        parts.append(f"\nUNRESOLVED CONTRADICTIONS: {inp.contradictions_text}")

    # Degradation
    if inp.degraded:
        parts.append(f"\nDEGRADATION_LEVEL: {inp.degradation_level}")
        parts.append(f"DEGRADATION_MESSAGE: {inp.degradation_message}")

    # Credits
    if inp.is_phase_one and inp.depth_available and inp.estimated_additional_credits:
        parts.append(f"\nDIG_DEEPER_CREDITS: {inp.estimated_additional_credits:.0f}")

    return "\n".join(parts)


def _fallback_response(inp: SpeechInput) -> str:
    """Fallback when speech Sonnet call fails — deliver findings directly."""
    parts = []

    # Mirror attempt
    if inp.user_key_phrases:
        phrase = inp.user_key_phrases[0]
        parts.append(f'You said "{phrase}" — and that tells me a lot about where you are right now.\n')

    # Connect
    parts.append(inp.trajectories_text)

    # Reframe
    if inp.variable_d:
        parts.append(f"\nThere's something underneath all of this: {inp.variable_d}")

    # Degradation
    if inp.degraded and inp.degradation_message:
        parts.append(f"\n{inp.degradation_message}")

    # Ask
    parts.append("\nWhat part of this feels most true to you?")

    # Dig deeper
    if inp.is_phase_one and inp.depth_available:
        parts.append("\nThere's more to explore if you'd like to go deeper.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: Extract speech input from engine results
# ---------------------------------------------------------------------------

def extract_speech_input(
    engine_result,
    user_original_text: str,
    is_phase_one: bool = False,
    estimated_additional_credits: float | None = None,
) -> SpeechInput:
    """
    Extract SpeechInput from an EngineResult.

    This bridges the engine output to the speech module input.
    Extracts user phrases, emotional markers, finding flags,
    and formats everything the speech module needs.
    """
    # Extract user key phrases (simple word-based extraction)
    key_phrases = _extract_key_phrases(user_original_text)
    emotional_markers = _extract_emotional_markers(user_original_text)

    # Format trajectories
    traj_parts = []
    for i, t in enumerate(engine_result.trajectories[:4], 1):
        name = t.root_cause.variable.name if hasattr(t, 'root_cause') else str(t)
        desc = t.root_cause.variable.description if hasattr(t, 'root_cause') else ""
        conf = t.confidence if hasattr(t, 'confidence') else 0.0
        traj_parts.append(f"Trajectory {i}: {desc[:150]} (confidence: {conf:.0%})")
    trajectories_text = "\n".join(traj_parts) if traj_parts else "No clear trajectories."

    # Variable D
    variable_d = None
    if engine_result.trajectories:
        top = engine_result.trajectories[0]
        if hasattr(top, 'root_cause') and top.root_cause.variable.is_hidden:
            variable_d = top.root_cause.variable.description

    # Finding type flags
    has_chirality = False
    has_teleology = False
    has_pressure = False
    has_false_prior = False
    has_dissonance = False

    for domain, output in engine_result.domain_outputs.items():
        for p in output.perspectives:
            fw = p.framework.value
            if fw == "chirality":
                has_chirality = True
            if fw == "teleology":
                has_teleology = True
            if fw in ("potential_kinetic", "entropy"):
                has_pressure = True
            if fw == "epistemology":
                for v in p.variables_found:
                    if "assumption" in v.name.lower() or "false" in v.name.lower():
                        has_false_prior = True
            if fw in ("cognitive_dissonance", "dialectics"):
                has_dissonance = True

    # Contradictions
    contradictions = []
    for ke in engine_result.ke_results:
        contradictions.extend(ke.contradictions[:2])
    contradictions_text = "\n".join(contradictions[:5]) if contradictions else ""

    # Degradation
    degraded = False
    deg_level = None
    deg_msg = ""

    # Findings summary
    findings_parts = []
    if engine_result.bias_penetration and "No specific" not in engine_result.bias_penetration:
        findings_parts.append(f"Bias pattern: {engine_result.bias_penetration}")
    if engine_result.hidden_purpose and "No hidden" not in engine_result.hidden_purpose:
        findings_parts.append(f"Hidden purpose: {engine_result.hidden_purpose}")
    if engine_result.uncertainty:
        findings_parts.append(f"Uncertainty: {engine_result.uncertainty}")
    findings_summary = "\n".join(findings_parts) if findings_parts else "Core analysis complete."

    return SpeechInput(
        findings_summary=findings_summary,
        trajectories_text=trajectories_text,
        variable_d=variable_d,
        contradictions_text=contradictions_text,
        metacognition_score=float(engine_result.delivery_mode == "direct") * 0.8 + 0.3
            if engine_result.delivery_mode == "direct" else 0.4,
        delivery_mode=engine_result.delivery_mode,
        user_original_text=user_original_text,
        user_key_phrases=key_phrases,
        user_emotional_markers=emotional_markers,
        is_phase_one=is_phase_one,
        depth_available=not engine_result.convergence_history.final_converged,
        estimated_additional_credits=estimated_additional_credits,
        degraded=degraded,
        degradation_level=deg_level,
        degradation_message=deg_msg,
        has_chirality=has_chirality,
        has_teleology=has_teleology,
        has_compressed_pressure=has_pressure,
        has_false_prior=has_false_prior,
        has_dissonance=has_dissonance,
        credit_summary=f"{engine_result.call_summary.get('total_calls', 0) * 0.5:.1f} credits used",
    )


def _extract_key_phrases(text: str) -> list[str]:
    """Extract key phrases from user's original text for mirroring."""
    phrases = []

    # Look for strong statements: "I am...", "I feel...", "I want...", "I can't..."
    markers = [
        "i am ", "i feel ", "i want ", "i need ", "i can't ", "i don't ",
        "i should ", "i know ", "i think ", "i'm ", "i've ",
        "but ", "every ", "always ", "never ",
    ]

    sentences = text.replace(".", ". ").replace("!", "! ").replace("?", "? ").split(". ")

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        lower = sentence.lower()
        for marker in markers:
            if marker in lower:
                # Extract the clause starting from the marker
                idx = lower.index(marker)
                phrase = sentence[idx:].strip().rstrip(".")
                if 4 < len(phrase) < 80:
                    phrases.append(phrase)
                break

    return phrases[:6]  # max 6 key phrases


def _extract_emotional_markers(text: str) -> list[str]:
    """Extract emotional language from user's text."""
    emotional_words = {
        "terrified", "scared", "afraid", "anxious", "worried", "nervous",
        "frustrated", "angry", "furious", "annoyed", "stuck",
        "sad", "depressed", "hopeless", "desperate", "lost",
        "exhausted", "burnt out", "overwhelmed", "drained",
        "guilty", "ashamed", "embarrassed",
        "excited", "passionate", "dream", "love",
        "hate", "dread", "resent",
        "unfulfilled", "empty", "meaningless", "pointless",
        "trapped", "suffocating", "paralyzed",
    }

    found = []
    lower = text.lower()
    for word in emotional_words:
        if word in lower:
            found.append(word)

    return found
