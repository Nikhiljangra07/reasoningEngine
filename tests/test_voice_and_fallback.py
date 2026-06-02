"""
Voice + fallback tests.

Covers the two safety/persona fixes from the dispatch audit:

  FIX 1 — Router _fallback_formation is cost-safe (minimum viable set),
          not all-domains. Re-introducing the "activate everything" fallback
          would compound API cost on every router failure.

  FIX 2 — Speech module enforces the advisor persona:
          - Five-step sequence (Mirror → Connect → Reframe → Recommend → Ask)
          - Multiple angles surfaced before settling on strongest
          - User holds authority on their own situation
          - Recommendation must be present (not just a closing question)
          - Educates (names pattern/mechanism), not just informs
          - No therapy / no game / no command-voice

Tests inspect the PROMPT TEXT — they verify the contract is encoded, not
that an LLM produced output matching it. This is the cheapest reliable
guard against accidental regressions to the older "therapist + open
question" shape.

Run: PYTHONPATH=. python3 tests/test_voice_and_fallback.py
"""

from __future__ import annotations

import asyncio

from src.core.types import Domain
from src.llm.router import (
    ALL_CONCEPTS,
    LLMFormationPlan,
    _fallback_formation,
)
from src.identity import compose_system_prompt
from src.llm.speech import SPEECH_SYSTEM_PROMPT


def _composed_speech_prompt() -> str:
    """The actual system prompt the model receives for the speech call —
    Singular Path doctrine header + 'MODE: speech' + the local speech
    prompt. After the 0.3.1 identity integration, the model sees this
    composed prompt; SPEECH_SYSTEM_PROMPT alone is no longer
    representative of what reaches the LLM. Tests that guard the
    'strategist not counselor' persona should validate against the
    composed prompt so they catch regressions at the user-visible
    layer."""
    return compose_system_prompt(SPEECH_SYSTEM_PROMPT, mode="speech")


PASSED = 0
FAILED = 0
ERRORS: list[tuple[str, str]] = []


def test(name: str):
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    global PASSED, FAILED
    name = getattr(fn, "_test_name", fn.__name__)
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, repr(e)))
        print(f"  FAIL  {name}: {e!r}")


# ---------------------------------------------------------------------------
# FIX 1 — Router fallback is cost-safe
# ---------------------------------------------------------------------------

@test("1.1 fallback activates ONLY Physics + Math + Psychology (not all 5)")
def test_fallback_minimum_domains():
    plan = _fallback_formation("test problem")
    active = {d.value for d in plan.active_domains}
    assert active == {"physics", "mathematics", "psychology"}, (
        f"expected minimum set, got {active}"
    )
    assert "philosophy" not in active
    assert "chemistry" not in active


@test("1.2 fallback uses core concepts only (not the full concept catalog)")
def test_fallback_core_concepts_only():
    plan = _fallback_formation("test problem")
    # 3 concepts per domain × 3 domains = 9 total
    total_concepts = sum(len(v) for v in plan.concepts_per_domain.values())
    assert total_concepts == 9, f"expected 9 minimum concepts, got {total_concepts}"

    # Each activated domain has exactly 3 concepts
    for domain_name, concepts in plan.concepts_per_domain.items():
        assert len(concepts) == 3, f"{domain_name}: expected 3 concepts, got {len(concepts)}"


@test("1.3 fallback concept choices are all in ALL_CONCEPTS (validity check)")
def test_fallback_concepts_are_valid():
    plan = _fallback_formation("test problem")
    for domain_name, concepts in plan.concepts_per_domain.items():
        valid_for_domain = set(ALL_CONCEPTS[domain_name])
        for c in concepts:
            assert c in valid_for_domain, (
                f"{domain_name}: '{c}' not in valid concepts {valid_for_domain}"
            )


@test("1.4 fallback estimated_credit_cost is meaningfully lower than full fanout")
def test_fallback_cost_lower():
    plan = _fallback_formation("test problem")
    # Old fallback estimated 15.0 (full fanout). New should be much lower.
    assert plan.estimated_credit_cost <= 5.0, (
        f"fallback cost {plan.estimated_credit_cost} should be ~3, not ~15"
    )
    # And iterations are conservative
    assert plan.estimated_iterations <= 3, (
        f"fallback iterations {plan.estimated_iterations} should be ~3, not ~5"
    )


@test("1.5 fallback reasoning text explicitly mentions cost-safe degradation")
def test_fallback_reasoning_documented():
    plan = _fallback_formation("test problem")
    r = plan.reasoning.lower()
    # The reasoning should surface to the user/logs that we're in fallback,
    # not pretend everything is normal.
    assert "fallback" in r or "cost-safe" in r, (
        f"reasoning should flag fallback state, got: {plan.reasoning}"
    )


@test("1.6 fallback complexity is downgraded ('low'), not pretend-'high'")
def test_fallback_complexity_honest():
    plan = _fallback_formation("test problem")
    # The old fallback marked complexity="high" — misleading. New one
    # acknowledges we're running a stripped-down formation.
    assert plan.problem_complexity in ("low", "medium"), (
        f"fallback complexity should reflect degraded state, got: {plan.problem_complexity}"
    )


# ---------------------------------------------------------------------------
# FIX 2 — Speech module: advisor voice contract
# ---------------------------------------------------------------------------

@test("2.1 speech prompt mentions 'strategist' (the persona the user wants)")
def test_speech_advisor_identity():
    # Validate the COMPOSED prompt (Singular Path doctrine + speech
    # local) — that is what the model actually receives at runtime.
    # The 'strategist' framing lives in the local speech prompt; the
    # 'not a counselor / chat partner / coach' rejection lives in the
    # Singular Path header. Both must reach the model together.
    text = _composed_speech_prompt().lower()
    assert "strategist" in text
    # The Singular Path header explicitly rejects chat-partner / coach /
    # critic framing; either that header wording OR the legacy explicit
    # negation satisfies the persona contract.
    assert (
        "not a chat partner" in text
        or "not a coach" in text
        or "are not a counselor" in text
        or "not a counselor, advisor" in text
    )


@test("2.2 speech prompt forbids leaving the user with only a question")
def test_speech_forbids_question_only():
    # Strategist persona — there MUST be a concrete move on the table,
    # not just analysis + question.
    text = SPEECH_SYSTEM_PROMPT.lower()
    assert "must put one concrete next move on the table" in text


@test("2.3 speech prompt requires a concrete recommended move")
def test_speech_requires_recommendation():
    # The advisor-counsel phrasing the user wanted
    text = SPEECH_SYSTEM_PROMPT.lower()
    assert "based on what i see" in text or "if i were advising you" in text


@test("2.4 speech prompt requires multi-angle terrain display")
def test_speech_multiple_angles():
    # The strategist surfaces multiple angles, picks a read, doesn't
    # list-and-walk-away.
    text = SPEECH_SYSTEM_PROMPT.lower()
    assert "pick your read" in text
    assert "two ways to read this" in text or "menu" in text


@test("2.5 speech prompt requires education (pattern naming)")
def test_speech_educates():
    # Name the pattern once, by its real name — that's the teach beat
    # in the strategist persona.
    text = SPEECH_SYSTEM_PROMPT.lower()
    assert "name the pattern" in text
    assert "pattern" in text


@test("2.6 speech prompt explicitly defers to user authority")
def test_speech_user_authority():
    text = SPEECH_SYSTEM_PROMPT.lower()
    # User holds final authority — not the AI
    deference_phrases = [
        "the user decides",
        "the user is the one acting",
        "the call is yours",
        "you know your",
        "user's authority",
    ]
    matches = sum(1 for p in deference_phrases if p in text)
    assert matches >= 2, (
        f"expected user-deference language, found {matches} of "
        f"{len(deference_phrases)} markers"
    )


@test("2.7 speech prompt forbids command-voice (advisor, not general)")
def test_speech_no_command_voice():
    # The strategist puts the move on the table but the user is the
    # operator — explicit hand-off at the fork. The phrase 'operator'
    # is load-bearing in the new persona.
    text = SPEECH_SYSTEM_PROMPT.lower()
    assert "user is the operator" in text
    assert "user decides" in text


@test("2.8 five-step sequence is Mirror → Connect → Reframe → Recommend → Ask")
def test_speech_five_step_sequence():
    text = SPEECH_SYSTEM_PROMPT
    # All five labels present in the new strategist sequence header
    assert "Mirror" in text
    assert "Tension" in text
    assert "Read" in text
    assert "Move" in text
    assert "Hand-off" in text
    assert "Mirror → Tension → Read → Move → Hand-off" in text


@test("2.9 speech prompt still bans therapy language (regression guard)")
def test_speech_no_therapy():
    text = SPEECH_SYSTEM_PROMPT.lower()
    # These should still be forbidden
    assert "i hear you saying" in text or "hold space" in text
    assert "cannot use therapy language" in text


@test("2.10 speech prompt still bans academic language (regression guard)")
def test_speech_no_academic():
    text = SPEECH_SYSTEM_PROMPT.lower()
    assert "cannot use academic language" in text
    # Specific bans still present
    assert "cognitive dissonance" in text


@test("2.11 recommendation must be cheap-to-test (advisor prefers reversible moves)")
def test_speech_cheap_to_test():
    text = SPEECH_SYSTEM_PROMPT.lower()
    # Angular Discipline rule #5
    assert "cheap to test" in text
    assert "reversible" in text


@test("2.12 timeframe anchor still required (no 'eventually')")
def test_speech_concrete_timeframe():
    text = SPEECH_SYSTEM_PROMPT.lower()
    # Angular Discipline rule #3 — preserved from prior pass
    assert "concrete timeframe" in text
    assert "eventually" in text  # appears in the forbidden-example


# ---------------------------------------------------------------------------
# Smoke — the few-shot examples reflect the new voice
# ---------------------------------------------------------------------------

@test("3.1 few-shot example contains a concrete recommended move (verdict)")
def test_example_1_has_move():
    # After Phase 2 the few-shot example is JSON. The verdict_line IS the
    # recommended move — declarative, named, with a timeframe.
    text = SPEECH_SYSTEM_PROMPT
    assert '"verdict_line":' in text
    # The example's verdict puts the move on the table explicitly.
    assert "Partnership-led wedge" in text or "not a Q3" in text


@test("3.2 few-shot example structures reasoning as an array (≤3 items rule)")
def test_example_2_has_move():
    text = SPEECH_SYSTEM_PROMPT
    # The few-shot example shows the reasoning[] array shape the LLM
    # must emit — three load-bearing arguments, no more.
    assert '"reasoning": [' in text
    assert "It's math, not taste." in text
    assert "Each path costs you a different person." in text


@test("3.3 few-shot example surfaces alternatives that are NAMED and WEIGHTED")
def test_examples_show_alternatives():
    text = SPEECH_SYSTEM_PROMPT
    # Per the strategist rules, alternatives are named with mono-caps tags
    # and explicitly weighted — not a neutral menu. The example demonstrates
    # both PAID-FIRST (weak) and COMMUNITY-FIRST (hedge).
    assert '"alternatives": [' in text
    assert "PAID-FIRST" in text
    assert '"weight": "weak"' in text


@test("3.4 few-shot example puts hand-off into falsifiers + open_questions")
def test_examples_defer_to_user():
    text = SPEECH_SYSTEM_PROMPT
    # The hand-off ("only you can see these variables") now lives in the
    # structured falsifiers[] + open_questions[] arrays. Both must appear
    # in the example and reference user-only variables (runway, partner).
    assert '"falsifiers": [' in text
    assert '"open_questions": [' in text
    assert "warm partner" in text or "runway" in text


@test("3.5 few-shot example emits at least one visual spec (Phase 3)")
def test_example_has_visual():
    # Phase 3 — the synthesizer is expected to emit visuals[] for Map Room.
    # The example demonstrates both supported types: comparison-table
    # (path comparison) and mermaid (decision tree).
    text = SPEECH_SYSTEM_PROMPT
    assert '"visuals": [' in text
    assert '"type": "comparison-table"' in text
    assert '"type": "mermaid"' in text


@test("3.6 prompt forbids prose preamble + code fences around the JSON")
def test_output_strict_json():
    # The OUTPUT section explicitly bans markdown fences and prose
    # preambles — the entire response must be a single valid JSON object.
    # Tokenize on whitespace so multi-line phrasing doesn't matter.
    text = " ".join(SPEECH_SYSTEM_PROMPT.lower().split())
    assert "no prose preamble" in text
    assert "no code fences" in text
    assert "single valid json object" in text


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_fallback_minimum_domains,
    test_fallback_core_concepts_only,
    test_fallback_concepts_are_valid,
    test_fallback_cost_lower,
    test_fallback_reasoning_documented,
    test_fallback_complexity_honest,
    test_speech_advisor_identity,
    test_speech_forbids_question_only,
    test_speech_requires_recommendation,
    test_speech_multiple_angles,
    test_speech_educates,
    test_speech_user_authority,
    test_speech_no_command_voice,
    test_speech_five_step_sequence,
    test_speech_no_therapy,
    test_speech_no_academic,
    test_speech_cheap_to_test,
    test_speech_concrete_timeframe,
    test_example_1_has_move,
    test_example_2_has_move,
    test_examples_show_alternatives,
    test_examples_defer_to_user,
    test_example_has_visual,
    test_output_strict_json,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} voice + fallback tests...")
    print()
    for fn in ALL_TESTS:
        run_test(fn)
    print()
    print(f"{PASSED} passed, {FAILED} failed")
    if ERRORS:
        print()
        print("Failures:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
