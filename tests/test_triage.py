"""
Triage gate tests.

No LLM calls, no API hits. All tests run against the deterministic mock
classifier so they're fast (~ms) and stable. The live-mode path is exercised
indirectly by _parse_triage_json fallbacks.

Run with: python -m pytest tests/test_triage.py -v
Or directly: PYTHONPATH=. python3 tests/test_triage.py
"""

from __future__ import annotations

import asyncio

from src.llm.effort import Effort
from src.llm.triage import (
    MCPNeed,
    Route,
    TriageResult,
    _parse_triage_json,
    triage,
)


# ---------------------------------------------------------------------------
# Test infrastructure (matches the style of test_integration.py + test_bridge.py)
# ---------------------------------------------------------------------------

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
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ---------------------------------------------------------------------------
# 1. TRIVIAL route — greetings, acks, single-word replies
# ---------------------------------------------------------------------------

@test("1.1 empty string → TRIVIAL")
async def test_empty():
    r = await triage("")
    assert r.route == Route.TRIVIAL
    assert r.recommended_effort == Effort.LOW
    assert r.classifier_mode == "mock"


@test("1.2 'hi' → TRIVIAL")
async def test_hi():
    r = await triage("hi")
    assert r.route == Route.TRIVIAL


@test("1.3 'thanks!' → TRIVIAL")
async def test_thanks():
    r = await triage("thanks!")
    assert r.route == Route.TRIVIAL


@test("1.4 'ok' → TRIVIAL")
async def test_ok():
    r = await triage("ok")
    assert r.route == Route.TRIVIAL


@test("1.5 'yes' → TRIVIAL")
async def test_yes():
    r = await triage("yes")
    assert r.route == Route.TRIVIAL


@test("1.6 'are you there?' → TRIVIAL")
async def test_are_you_there():
    r = await triage("are you there?")
    assert r.route == Route.TRIVIAL


# ---------------------------------------------------------------------------
# 2. DIRECT_PLUS route — memory / project-context queries
# ---------------------------------------------------------------------------

@test("2.1 'what did we decide about auth' → DIRECT_PLUS + memory_v2")
async def test_what_did_we_decide():
    r = await triage("what did we decide about auth last week?")
    assert r.route == Route.DIRECT_PLUS
    assert any(m.name == "memory_v2" for m in r.mcps_needed)


@test("2.2 'where is the routing logic' → DIRECT_PLUS")
async def test_where_is():
    r = await triage("where is the routing logic in this codebase?")
    assert r.route == Route.DIRECT_PLUS


@test("2.3 'summarize this repo area' → DIRECT_PLUS")
async def test_summarize_repo():
    r = await triage("summarize this repo area for me")
    assert r.route == Route.DIRECT_PLUS


# ---------------------------------------------------------------------------
# 3. DEEP route — explicit user-forced depth triggers
# ---------------------------------------------------------------------------

@test("3.1 'go deeper' → DEEP")
async def test_go_deeper():
    r = await triage("go deeper on this problem")
    assert r.route == Route.DEEP
    assert r.recommended_effort in (Effort.MEDIUM, Effort.HIGH)


@test("3.2 'pressure-test this' → DEEP")
async def test_pressure_test():
    r = await triage("pressure-test this approach before I commit")
    assert r.route == Route.DEEP


@test("3.3 'open this up' → DEEP")
async def test_open_this_up():
    r = await triage("open this up for me — I want the bigger picture")
    assert r.route == Route.DEEP


# ---------------------------------------------------------------------------
# 4. DEEP route — semantic depth triggers
# ---------------------------------------------------------------------------

@test("4.1 'should I X or Y' → DEEP")
async def test_should_i():
    r = await triage("should I take the founder offer or stay employed?")
    assert r.route == Route.DEEP


@test("4.2 'I'm torn between' → DEEP")
async def test_torn_between():
    r = await triage("I'm torn between rewriting this module or extending it")
    assert r.route == Route.DEEP


@test("4.3 'what would you do here' → DEEP")
async def test_what_would_you_do():
    r = await triage("what would you do here?")
    assert r.route == Route.DEEP


@test("4.4 'whether to' framing → DEEP")
async def test_whether_to():
    r = await triage(
        "I'm thinking about whether to migrate off Heroku — interested in your take"
    )
    assert r.route == Route.DEEP


# ---------------------------------------------------------------------------
# 5. DEEP route — effort calibration
# ---------------------------------------------------------------------------

@test("5.1 architecture word → DEEP / MEDIUM + flag")
async def test_architecture_medium():
    r = await triage("we need to refactor the auth layer next sprint")
    assert r.route == Route.DEEP
    assert r.recommended_effort == Effort.MEDIUM
    assert "architecture_decision" in r.risk_flags


@test("5.2 long-term decision → DEEP / HIGH + flag")
async def test_long_term_high():
    r = await triage("should I quit my job to start a company?")
    assert r.route == Route.DEEP
    assert r.recommended_effort == Effort.HIGH
    assert "long_term_consequence" in r.risk_flags


@test("5.3 founder question → DEEP / HIGH")
async def test_founder_high():
    r = await triage(
        "I'm thinking about whether to raise money from this investor or bootstrap"
    )
    assert r.route == Route.DEEP
    assert r.recommended_effort == Effort.HIGH


# ---------------------------------------------------------------------------
# 6. Interrupt flag — short statement of intent to do something irreversible
# ---------------------------------------------------------------------------

@test("6.1 short delete statement → interrupt=True + irreversible flag")
async def test_interrupt_short_delete():
    r = await triage("delete the old branch")
    assert r.interrupt is True
    assert "irreversible_action" in r.risk_flags


@test("6.2 long deliberation about deletion → interrupt=False")
async def test_interrupt_long_deliberation():
    r = await triage(
        "I've been thinking about whether to delete the old feature flag system. "
        "We had a long debate on the team last week and I'm leaning toward removing it "
        "but I want to think through what could go wrong before I commit."
    )
    # Deliberation, not action statement.
    assert r.interrupt is False


@test("6.3 'deploy to prod' short statement → interrupt=True")
async def test_interrupt_deploy():
    r = await triage("about to push to prod")
    assert r.interrupt is True
    assert "irreversible_action" in r.risk_flags


# ---------------------------------------------------------------------------
# 7. DIRECT route — factual / generic fallback
# ---------------------------------------------------------------------------

@test("7.1 generic factual question → DIRECT")
async def test_direct_factual():
    r = await triage("what is the difference between async and threading in Python?")
    assert r.route == Route.DIRECT


# ---------------------------------------------------------------------------
# 8. Classifier mode + raw response handling
# ---------------------------------------------------------------------------

@test("8.1 mock mode tags classifier_mode='mock'")
async def test_classifier_mode_mock():
    r = await triage("hi")
    assert r.classifier_mode == "mock"


@test("8.2 _parse_triage_json strips code fences")
def test_parse_strips_fences():
    raw = '```json\n{"route": "deep", "recommended_effort": "high", "why": "x"}\n```'
    r = _parse_triage_json(raw, "test input")
    assert r.route == Route.DEEP
    assert r.recommended_effort == Effort.HIGH
    assert r.classifier_mode == "live"


@test("8.3 _parse_triage_json falls back on garbage")
def test_parse_garbage_fallback():
    r = _parse_triage_json("this is not json at all", "hi")
    assert r.classifier_mode == "live_unparseable_fallback_mock"
    # The fallback runs the mock classifier on the original text.
    assert r.route == Route.TRIVIAL  # "hi" → TRIVIAL


@test("8.4 _parse_triage_json bias-toward-depth on unknown route")
def test_parse_unknown_route():
    raw = '{"route": "bogus_route", "recommended_effort": "medium", "why": "x"}'
    r = _parse_triage_json(raw, "test")
    # Unknown route → DEEP (safest bias).
    assert r.route == Route.DEEP


@test("8.5 _parse_triage_json captures mcps_needed with all fields")
def test_parse_mcps():
    raw = (
        '{"route": "direct_plus", "recommended_effort": "low", '
        '"mcps_needed": [{"name": "github", "why": "needed for PR context", "required": true}], '
        '"why": "x"}'
    )
    r = _parse_triage_json(raw, "test")
    assert len(r.mcps_needed) == 1
    assert r.mcps_needed[0].name == "github"
    assert r.mcps_needed[0].required is True


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

@test("9.1 whitespace-only input → TRIVIAL")
async def test_whitespace():
    r = await triage("   \n\n   ")
    assert r.route == Route.TRIVIAL


@test("9.2 None client falls back to mock cleanly")
async def test_none_client():
    r = await triage("should I refactor this?", client=None)
    assert r.route == Route.DEEP
    assert r.classifier_mode == "mock"


@test("9.3 every TriageResult has populated 'why' field")
async def test_why_populated():
    for msg in ["hi", "what's async", "should I leave my job?", "what did we decide?"]:
        r = await triage(msg)
        assert r.why and len(r.why) > 5, f"empty why for: {msg}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_empty,
    test_hi,
    test_thanks,
    test_ok,
    test_yes,
    test_are_you_there,
    test_what_did_we_decide,
    test_where_is,
    test_summarize_repo,
    test_go_deeper,
    test_pressure_test,
    test_open_this_up,
    test_should_i,
    test_torn_between,
    test_what_would_you_do,
    test_whether_to,
    test_architecture_medium,
    test_long_term_high,
    test_founder_high,
    test_interrupt_short_delete,
    test_interrupt_long_deliberation,
    test_interrupt_deploy,
    test_direct_factual,
    test_classifier_mode_mock,
    test_parse_strips_fences,
    test_parse_garbage_fallback,
    test_parse_unknown_route,
    test_parse_mcps,
    test_whitespace,
    test_none_client,
    test_why_populated,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} triage gate tests...")
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
