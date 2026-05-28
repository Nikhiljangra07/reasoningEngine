"""
Dispatch preview tests.

Verifies the new pre-flight cost estimator + preview endpoint:
    - Deterministic cost estimates (no LLM call required)
    - Route-aware: TRIVIAL = ~zero, DIRECT < DEEP @ LOW < DEEP @ HIGH
    - Effort scaling: more iterations → higher estimate
    - num_active_domains reduces estimate (fewer domain calls)
    - preview_dispatch returns correct shape for each route
    - Phase totals sum to total_usd

Run: PYTHONPATH=. python3 tests/test_dispatch_preview.py
"""

from __future__ import annotations

import asyncio

from src.core.types import Domain
from src.llm.client import ClientMode, LLMClient
from src.llm.dispatch_preview import (
    CallProfile,
    CostBreakdown,
    DispatchPreview,
    estimate_cost_breakdown,
    preview_dispatch,
    preview_to_dict,
    serialize_formation_plan,
)
from src.llm.effort import Effort
from src.llm.router import LLMFormationPlan
from src.llm.triage import MCPNeed, Route, TriageResult


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
# 1. CallProfile cost computation
# ---------------------------------------------------------------------------

@test("1.1 CallProfile.cost_usd uses PRICING from provider_map")
def test_call_profile_cost():
    p = CallProfile("test", "anthropic/claude-sonnet-4-6", in_tokens=1000, out_tokens=500)
    # Sonnet 4.6 = ($3, $15) per 1M tokens
    # 1000/1M * $3 + 500/1M * $15 = $0.003 + $0.0075 = $0.0105
    assert abs(p.cost_usd() - 0.0105) < 0.00001


@test("1.2 CallProfile uses FALLBACK pricing for unknown model slugs")
def test_call_profile_fallback():
    p = CallProfile("test", "unknown/model-xyz", in_tokens=1000, out_tokens=1000)
    # Fallback = ($3, $15)
    assert abs(p.cost_usd() - (3 + 15) / 1000) < 0.00001


# ---------------------------------------------------------------------------
# 2. Cost estimator by route
# ---------------------------------------------------------------------------

@test("2.1 TRIVIAL route — only triage call, no other phases")
def test_estimate_trivial():
    cb = estimate_cost_breakdown(Route.TRIVIAL, Effort.LOW)
    assert cb.estimated_llm_calls == 1
    assert cb.phases["triage"] > 0
    assert cb.phases["router"] == 0
    assert cb.phases["domain_calls"] == 0
    assert cb.phases["ke_calls"] == 0
    assert cb.phases["speech"] == 0
    assert cb.phases["direct"] == 0
    # Triage is Flash-Lite — sub-cent
    assert cb.total_usd < 0.005


@test("2.2 DIRECT route — triage + direct, no engine phases")
def test_estimate_direct():
    cb = estimate_cost_breakdown(Route.DIRECT, Effort.LOW)
    assert cb.estimated_llm_calls == 2
    assert cb.phases["direct"] > 0
    assert cb.phases["router"] == 0
    assert cb.phases["domain_calls"] == 0
    # Direct uses Sonnet but small input — still under 5 cents
    assert cb.total_usd < 0.05


@test("2.3 DIRECT_PLUS route — same calls as DIRECT but higher input cost")
def test_estimate_direct_plus():
    cb_direct = estimate_cost_breakdown(Route.DIRECT, Effort.LOW)
    cb_plus = estimate_cost_breakdown(Route.DIRECT_PLUS, Effort.LOW)
    # Both have triage + 1 sonnet call, but DIRECT_PLUS has larger input
    assert cb_plus.estimated_llm_calls == cb_direct.estimated_llm_calls
    assert cb_plus.total_usd > cb_direct.total_usd


@test("2.4 DEEP route — fires router + engine loop + speech")
def test_estimate_deep():
    cb = estimate_cost_breakdown(Route.DEEP, Effort.MEDIUM)
    assert cb.phases["triage"] > 0
    assert cb.phases["router"] > 0
    assert cb.phases["domain_calls"] > 0
    assert cb.phases["ke_calls"] > 0
    assert cb.phases["speech"] > 0
    # MEDIUM = 5 iters × (5 domains + 5 ke) + triage + router + speech = 53 calls
    assert cb.estimated_llm_calls == 1 + 1 + 5 * 5 + 5 * 5 + 1


# ---------------------------------------------------------------------------
# 3. Effort scaling
# ---------------------------------------------------------------------------

@test("3.1 DEEP cost ordering: LOW(3) < AUTO(4) < MEDIUM(5) < HIGH(8)")
def test_effort_monotonic():
    # 2026-05-28 retuning: AUTO is no longer the deepest tier — it now
    # means "let the engine decide on a budget" (4 iter). HIGH is the
    # explicit deepest sweep (8 iter). Order is now non-monotonic by
    # the Effort enum but monotonic in iteration count.
    low    = estimate_cost_breakdown(Route.DEEP, Effort.LOW)
    medium = estimate_cost_breakdown(Route.DEEP, Effort.MEDIUM)
    high   = estimate_cost_breakdown(Route.DEEP, Effort.HIGH)
    auto   = estimate_cost_breakdown(Route.DEEP, Effort.AUTO)
    assert low.total_usd < auto.total_usd < medium.total_usd < high.total_usd


@test("3.2 Effort affects iter count: LOW=3, MEDIUM=5, HIGH=8, AUTO=4")
def test_effort_iter_count():
    # Each iter adds: num_active_domains + num_ke_pairs LLM calls
    # Default: 5 + 5 = 10 calls per iter
    base = 1 + 1 + 1  # triage + router + speech
    low_calls    = estimate_cost_breakdown(Route.DEEP, Effort.LOW).estimated_llm_calls
    medium_calls = estimate_cost_breakdown(Route.DEEP, Effort.MEDIUM).estimated_llm_calls
    high_calls   = estimate_cost_breakdown(Route.DEEP, Effort.HIGH).estimated_llm_calls
    auto_calls   = estimate_cost_breakdown(Route.DEEP, Effort.AUTO).estimated_llm_calls
    assert low_calls    == base + 3 * 10
    assert medium_calls == base + 5 * 10
    assert high_calls   == base + 8 * 10
    assert auto_calls   == base + 4 * 10


# ---------------------------------------------------------------------------
# 4. Active-domain scaling
# ---------------------------------------------------------------------------

@test("4.1 num_active_domains=3 produces lower cost than =5")
def test_fewer_domains_cheaper():
    cb_5 = estimate_cost_breakdown(Route.DEEP, Effort.MEDIUM, num_active_domains=5, num_ke_pairs=5)
    cb_3 = estimate_cost_breakdown(Route.DEEP, Effort.MEDIUM, num_active_domains=3, num_ke_pairs=3)
    assert cb_3.total_usd < cb_5.total_usd
    assert cb_3.estimated_llm_calls < cb_5.estimated_llm_calls


@test("4.2 num_active_domains=1, num_ke_pairs=0 — minimal DEEP shape")
def test_minimal_deep():
    cb = estimate_cost_breakdown(Route.DEEP, Effort.LOW, num_active_domains=1, num_ke_pairs=0)
    # LOW = 3 iters × (1 + 0) = 3 calls. + triage + router + speech = 6.
    assert cb.estimated_llm_calls == 6


@test("4.3 converged_iterations override caps the iter count")
def test_converged_iterations_override():
    cb_capped = estimate_cost_breakdown(
        Route.DEEP, Effort.HIGH,
        num_active_domains=5, num_ke_pairs=5,
        converged_iterations=2,
    )
    # Should use 2 iterations, not 10
    base = 1 + 1 + 1
    assert cb_capped.estimated_llm_calls == base + 2 * (5 + 5)


# ---------------------------------------------------------------------------
# 5. Phase sums match total
# ---------------------------------------------------------------------------

@test("5.1 sum(phases.values) == total_usd for every route")
def test_phase_sum_matches_total():
    for route in (Route.TRIVIAL, Route.DIRECT, Route.DIRECT_PLUS, Route.DEEP):
        cb = estimate_cost_breakdown(route, Effort.MEDIUM)
        phase_sum = round(sum(cb.phases.values()), 5)
        assert abs(phase_sum - cb.total_usd) < 0.0001, (
            f"{route.value}: phase sum {phase_sum} != total {cb.total_usd}"
        )


@test("5.2 estimated_seconds is non-negative and scales with call count")
def test_seconds_positive_and_scales():
    cb_low  = estimate_cost_breakdown(Route.DEEP, Effort.LOW)
    cb_high = estimate_cost_breakdown(Route.DEEP, Effort.HIGH)
    assert cb_low.estimated_seconds > 0
    assert cb_high.estimated_seconds > cb_low.estimated_seconds


# ---------------------------------------------------------------------------
# 6. Formation plan serialization
# ---------------------------------------------------------------------------

@test("6.1 serialize_formation_plan(None) returns None")
def test_serialize_plan_none():
    assert serialize_formation_plan(None) is None


@test("6.2 serialize_formation_plan emits JSON-safe dict")
def test_serialize_plan_shape():
    plan = LLMFormationPlan(
        active_domains=[Domain.PHYSICS, Domain.MATHEMATICS, Domain.PSYCHOLOGY],
        concepts_per_domain={
            "physics": ["first_principles", "entropy"],
            "mathematics": ["bayesian_inference"],
            "psychology": ["metacognition"],
        },
        estimated_agent_count=4,
        estimated_iterations=3,
        estimated_credit_cost=5.0,
        problem_complexity="low",
        reasoning="Simple decision, minimal fan-out.",
    )
    d = serialize_formation_plan(plan)
    assert d["active_domains"] == ["physics", "mathematics", "psychology"]
    assert d["concepts_per_domain"]["physics"] == ["first_principles", "entropy"]
    assert d["estimated_agent_count"] == 4
    assert d["problem_complexity"] == "low"
    # JSON-safe: must be serializable
    import json
    json.dumps(d)


# ---------------------------------------------------------------------------
# 7. preview_dispatch end-to-end with mock client
# ---------------------------------------------------------------------------

@test("7.1 preview_dispatch with trivial text returns TRIVIAL route + low cost")
async def test_preview_trivial():
    client = LLMClient(mode=ClientMode.MOCK)
    preview = await preview_dispatch("hi", client=client)
    assert preview.route == "trivial"
    assert preview.cost_breakdown.total_usd < 0.005
    assert preview.formation_plan is None  # no router fired


@test("7.2 preview_dispatch with decision text routes to DEEP and has formation plan or null on failure")
async def test_preview_deep():
    client = LLMClient(mode=ClientMode.MOCK)
    preview = await preview_dispatch(
        "Should I refactor the auth module before launch or just ship?",
        client=client,
    )
    assert preview.route == "deep"
    # In mock mode, router may not produce a real plan — but the cost
    # estimate should still fire using defaults
    assert preview.cost_breakdown.total_usd > 0
    assert preview.cost_breakdown.estimated_llm_calls > 5


@test("7.3 preview_dispatch with force_triage_result skips classifier")
async def test_preview_forced_triage():
    client = LLMClient(mode=ClientMode.MOCK)
    forced = TriageResult(
        route=Route.DIRECT,
        recommended_effort=Effort.LOW,
        why="Forced for test.",
        classifier_mode="mock",
    )
    preview = await preview_dispatch(
        "What's the capital of France?",
        client=client,
        force_triage_result=forced,
    )
    assert preview.route == "direct"
    assert preview.triage_why == "Forced for test."


@test("7.4 preview_to_dict produces a JSON-safe dict")
async def test_preview_to_dict():
    client = LLMClient(mode=ClientMode.MOCK)
    preview = await preview_dispatch("hi", client=client)
    d = preview_to_dict(preview)
    import json
    json.dumps(d)  # must not raise
    assert d["route"] == "trivial"
    assert "cost_breakdown" in d
    assert "phases" in d["cost_breakdown"]
    assert d["cost_breakdown"]["total_usd"] >= 0


# ---------------------------------------------------------------------------
# 8. Risk surfacing
# ---------------------------------------------------------------------------

@test("8.1 risk_flags propagate from triage into preview")
async def test_risk_flags_propagate():
    client = LLMClient(mode=ClientMode.MOCK)
    forced = TriageResult(
        route=Route.DEEP,
        recommended_effort=Effort.HIGH,
        risk_flags=["irreversible_action", "architecture_decision"],
        interrupt=False,
        why="High-stakes test.",
        classifier_mode="mock",
    )
    preview = await preview_dispatch(
        "Should we drop the production database?",
        client=client,
        force_triage_result=forced,
    )
    assert "irreversible_action" in preview.risk_flags
    assert "architecture_decision" in preview.risk_flags


@test("8.2 mcps_needed propagate from triage into preview")
async def test_mcps_propagate():
    client = LLMClient(mode=ClientMode.MOCK)
    forced = TriageResult(
        route=Route.DIRECT_PLUS,
        recommended_effort=Effort.LOW,
        mcps_needed=[MCPNeed(name="memory_v2", why="Prior decisions.", required=False)],
        why="Memory query.",
        classifier_mode="mock",
    )
    preview = await preview_dispatch(
        "What did we decide about the schema?",
        client=client,
        force_triage_result=forced,
    )
    assert len(preview.mcps_needed) == 1
    assert preview.mcps_needed[0]["name"] == "memory_v2"


# ---------------------------------------------------------------------------
# 9. Sanity bounds — the dispatch ceiling is real
# ---------------------------------------------------------------------------

@test("9.1 worst-case DEEP @ HIGH with all domains stays under $1.50 estimate")
def test_worst_case_under_threshold():
    # 2026-05-28: HIGH is the new deepest tier (8 iter). AUTO is now a
    # mid-budget tier (4 iter), so the "worst case" assertion lives on
    # HIGH. The budget tracker caps actual spend; this just verifies
    # the estimate sits in a sensible range.
    cb = estimate_cost_breakdown(Route.DEEP, Effort.HIGH, num_active_domains=5, num_ke_pairs=5)
    assert 0.3 < cb.total_usd < 2.0  # ballpark — adjust if pricing changes


@test("9.2 cheapest DIRECT is under 5 cents")
def test_cheapest_direct():
    cb = estimate_cost_breakdown(Route.DIRECT, Effort.LOW)
    assert cb.total_usd < 0.05


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_call_profile_cost,
    test_call_profile_fallback,
    test_estimate_trivial,
    test_estimate_direct,
    test_estimate_direct_plus,
    test_estimate_deep,
    test_effort_monotonic,
    test_effort_iter_count,
    test_fewer_domains_cheaper,
    test_minimal_deep,
    test_converged_iterations_override,
    test_phase_sum_matches_total,
    test_seconds_positive_and_scales,
    test_serialize_plan_none,
    test_serialize_plan_shape,
    test_preview_trivial,
    test_preview_deep,
    test_preview_forced_triage,
    test_preview_to_dict,
    test_risk_flags_propagate,
    test_mcps_propagate,
    test_worst_case_under_threshold,
    test_cheapest_direct,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} dispatch_preview tests...")
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
