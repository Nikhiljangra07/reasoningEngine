"""
Budget enforcer tests.

No LLM calls. Verifies:
    1. Fresh tracker: check() returns allowed=True with remaining = caps
    2. Each cap breach is detected individually
    3. Cost accumulation from LLM responses (success + failure paths)
    4. Custom caps override defaults
    5. summary() shape is stable and includes caps + state
    6. Wall-time breach simulated via state.started_at manipulation
    7. Tracker stays breached once breached (idempotence)
    8. Unknown model uses fallback pricing without raising

Run: PYTHONPATH=. python3 tests/test_budget.py
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.budget import BudgetCaps, BudgetCheck, BudgetState, BudgetTracker


# ---------------------------------------------------------------------------
# Test infrastructure
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
        fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ---------------------------------------------------------------------------
# Fixtures — a lightweight LLMResponse-shaped object for testing record_llm_response
# ---------------------------------------------------------------------------

@dataclass
class FakeLLMResponse:
    model: str
    input_tokens: int
    output_tokens: int
    success: bool = True


# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------

@test("1.1 default caps match locked design (12 iter / 720s / $1.00 / 20 mcps)")
def test_default_caps():
    t = BudgetTracker()
    assert t.caps.max_iterations == 12
    assert t.caps.max_wall_time_sec == 720.0
    assert t.caps.max_cost_usd == 1.00
    assert t.caps.max_mcp_calls == 20


@test("1.2 fresh tracker: check.allowed=True, remaining = caps")
def test_fresh_check():
    t = BudgetTracker()
    c = t.check()
    assert c.allowed is True
    assert c.remaining_iterations == 12
    assert c.remaining_cost_usd == 1.00
    assert c.remaining_mcp_calls == 20
    # remaining_wall_time should be ~720 (allow tiny epsilon for elapsed)
    assert 719.0 <= c.remaining_wall_time_sec <= 720.0


# ---------------------------------------------------------------------------
# 2. Each cap breach
# ---------------------------------------------------------------------------

@test("2.1 iteration cap breach")
def test_iteration_breach():
    t = BudgetTracker(caps=BudgetCaps(max_iterations=3))
    for _ in range(3):
        t.increment_iteration()
    c = t.check()
    assert c.allowed is False
    assert "iteration cap" in c.reason


@test("2.2 cost cap breach")
def test_cost_breach():
    t = BudgetTracker(caps=BudgetCaps(max_cost_usd=0.01))
    # anthropic/claude-sonnet-4-6 costs $3/M input, $15/M output → 10k+10k = $0.18
    t.record_llm_call("anthropic/claude-sonnet-4-6", 10_000, 10_000)
    c = t.check()
    assert c.allowed is False
    assert "cost exceeded" in c.reason


@test("2.3 wall-time cap breach (simulated via started_at)")
def test_wall_time_breach():
    t = BudgetTracker(caps=BudgetCaps(max_wall_time_sec=10.0))
    # Travel 15s into the past — simulates 15s elapsed
    t.state.started_at -= 15.0
    c = t.check()
    assert c.allowed is False
    assert "wall time exceeded" in c.reason


@test("2.4 MCP call cap breach")
def test_mcp_breach():
    t = BudgetTracker(caps=BudgetCaps(max_mcp_calls=2))
    t.record_mcp_call("github")
    t.record_mcp_call("web_search")
    c = t.check()
    assert c.allowed is False
    assert "MCP call cap" in c.reason


# ---------------------------------------------------------------------------
# 3. Cost accumulation from LLM responses
# ---------------------------------------------------------------------------

@test("3.1 record_llm_call with Sonnet 4.6 computes correct cost")
def test_record_sonnet():
    t = BudgetTracker()
    # Sonnet 4.6: $3/M input, $15/M output
    cost = t.record_llm_call("anthropic/claude-sonnet-4-6", 1_000, 500)
    # Expected: 1000 * 3 / 1M + 500 * 15 / 1M = 0.003 + 0.0075 = 0.0105
    assert abs(cost - 0.0105) < 1e-6
    assert abs(t.state.cost_usd - 0.0105) < 1e-6


@test("3.2 record_llm_call with unknown model uses fallback pricing (no raise)")
def test_record_unknown_model():
    t = BudgetTracker()
    cost = t.record_llm_call("nobody/never-heard-of-it", 1_000, 1_000)
    # Fallback is Sonnet-tier ($3, $15), so 0.003 + 0.015 = 0.018
    assert cost > 0
    assert abs(cost - 0.018) < 1e-6


@test("3.3 record_llm_response from successful response adds cost")
def test_record_response_success():
    t = BudgetTracker()
    resp = FakeLLMResponse(model="anthropic/claude-haiku-4-5",
                           input_tokens=100, output_tokens=100, success=True)
    cost = t.record_llm_response(resp)
    # Haiku 4.5: $1/M input, $5/M output → 0.0001 + 0.0005 = 0.0006
    assert abs(cost - 0.0006) < 1e-6


@test("3.4 record_llm_response from failed response adds zero cost")
def test_record_response_failure():
    t = BudgetTracker()
    resp = FakeLLMResponse(model="anthropic/claude-sonnet-4-6",
                           input_tokens=1000, output_tokens=1000, success=False)
    cost = t.record_llm_response(resp)
    assert cost == 0.0
    assert t.state.cost_usd == 0.0


@test("3.5 record_llm_response with missing fields handled safely")
def test_record_response_missing_fields():
    # Object without `model`, `input_tokens`, etc.
    class Stub:
        success = True
    resp = Stub()
    t = BudgetTracker()
    cost = t.record_llm_response(resp)
    # No tokens → no cost, no crash
    assert cost == 0.0


# ---------------------------------------------------------------------------
# 4. Custom caps + counter increments
# ---------------------------------------------------------------------------

@test("4.1 custom caps fully override defaults")
def test_custom_caps():
    t = BudgetTracker(caps=BudgetCaps(
        max_iterations=5, max_wall_time_sec=60.0,
        max_cost_usd=0.50, max_mcp_calls=10,
    ))
    assert t.caps.max_iterations == 5
    assert t.caps.max_wall_time_sec == 60.0
    assert t.caps.max_cost_usd == 0.50
    assert t.caps.max_mcp_calls == 10


@test("4.2 increment_iteration counts up")
def test_increment_iter():
    t = BudgetTracker()
    for _ in range(5):
        t.increment_iteration()
    assert t.state.iterations == 5


@test("4.3 record_mcp_call counts up")
def test_increment_mcp():
    t = BudgetTracker()
    for name in ("github", "web_search", "docs"):
        t.record_mcp_call(name)
    assert t.state.mcp_calls == 3


# ---------------------------------------------------------------------------
# 5. summary() shape
# ---------------------------------------------------------------------------

@test("5.1 summary() includes all live counters + caps")
def test_summary_shape():
    t = BudgetTracker(caps=BudgetCaps(max_iterations=8))
    t.increment_iteration()
    t.record_llm_call("anthropic/claude-haiku-4-5", 500, 500)
    t.record_mcp_call("github")
    s = t.summary()
    assert s["iterations"] == 1
    assert s["mcp_calls"] == 1
    assert s["cost_usd"] > 0
    assert s["breached"] is False
    assert s["caps"]["max_iterations"] == 8


@test("5.2 summary().breach_reason populated after breach")
def test_summary_breach_reason():
    t = BudgetTracker(caps=BudgetCaps(max_iterations=1))
    t.increment_iteration()
    t.check()  # triggers breach
    s = t.summary()
    assert s["breached"] is True
    assert "iteration cap" in s["breach_reason"]


# ---------------------------------------------------------------------------
# 6. Idempotence + state guarantees
# ---------------------------------------------------------------------------

@test("6.1 once breached, stays breached across multiple checks")
def test_breach_idempotent():
    t = BudgetTracker(caps=BudgetCaps(max_iterations=1))
    t.increment_iteration()
    c1 = t.check()
    c2 = t.check()
    c3 = t.check()
    assert c1.allowed is False
    assert c2.allowed is False
    assert c3.allowed is False
    assert t.breached is True


@test("6.2 .breached property reflects state.breached")
def test_breached_property():
    t = BudgetTracker()
    assert t.breached is False
    t.state.breached = True
    assert t.breached is True


@test("6.3 multiple check() calls don't accidentally consume budget")
def test_check_no_side_effects():
    t = BudgetTracker()
    for _ in range(10):
        t.check()
    # Counters untouched by check() alone
    assert t.state.iterations == 0
    assert t.state.cost_usd == 0.0
    assert t.state.mcp_calls == 0


# ---------------------------------------------------------------------------
# 7. Breach precedence
# ---------------------------------------------------------------------------

@test("7.1 wall-time breach takes precedence over cost breach when both apply")
def test_precedence_walltime():
    t = BudgetTracker(caps=BudgetCaps(max_wall_time_sec=10.0, max_cost_usd=0.01))
    t.state.started_at -= 20.0  # 20s elapsed
    t.state.cost_usd = 100.0     # cost also breached
    c = t.check()
    assert c.allowed is False
    assert "wall time" in c.reason  # wall-time wins (checked first)


@test("7.2 remaining values clamp to zero, never negative")
def test_remaining_nonneg():
    t = BudgetTracker(caps=BudgetCaps(max_iterations=1, max_cost_usd=0.01))
    t.increment_iteration()
    t.increment_iteration()
    t.state.cost_usd = 5.00  # massively over
    c = t.check()
    assert c.remaining_iterations >= 0
    assert c.remaining_cost_usd >= 0
    assert c.remaining_wall_time_sec >= 0
    assert c.remaining_mcp_calls >= 0


# ---------------------------------------------------------------------------
# 8. Engine integration — budget passed through run_async_formation
#    stops the loop early on cap breach with forced_stop=True.
# ---------------------------------------------------------------------------

@test("8.1 engine stops on first iteration when budget is pre-breached")
def test_engine_budget_pre_breached():
    import asyncio
    from src.core.types import (
        Direction, FrameworkID, Problem, Variable,
    )
    from src.llm.client import ClientMode, LLMClient
    from src.llm.engine import run_async_formation

    # Pre-breach the budget by pretending wall-time already elapsed.
    # The engine's first pre-iteration check will see the breach and stop
    # without running any iteration body. Deterministic regardless of mock
    # convergence behavior.
    client = LLMClient(mode=ClientMode.MOCK)
    budget = BudgetTracker(caps=BudgetCaps(max_wall_time_sec=1.0))
    budget.state.started_at -= 100.0  # 100s already elapsed → breached

    problem = Problem(
        statement="should I refactor this or extend it?",
        variables=[Variable(
            name="v", description="x", magnitude=0.5,
            direction=Direction.NEUTRAL, confidence=0.7,
            source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True,
        )],
    )

    result = asyncio.run(run_async_formation(
        problem=problem, client=client, max_iterations=10, budget=budget,
    ))
    # Pre-breached → engine stops at iteration 1 pre-check → total_iterations=0
    assert result.convergence_history.total_iterations == 0
    assert result.convergence_history.forced_stop is True
    assert budget.breached is True


@test("8.2 engine WITHOUT budget runs to its own max_iterations (back-compat)")
def test_engine_no_budget():
    import asyncio
    from src.core.types import (
        Direction, FrameworkID, Problem, Variable,
    )
    from src.llm.client import ClientMode, LLMClient
    from src.llm.engine import run_async_formation

    client = LLMClient(mode=ClientMode.MOCK)
    problem = Problem(
        statement="hi",
        variables=[Variable(
            name="v", description="x", magnitude=0.5,
            direction=Direction.NEUTRAL, confidence=0.7,
            source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True,
        )],
    )

    # No budget passed — engine runs normally.
    result = asyncio.run(run_async_formation(
        problem=problem, client=client, max_iterations=2,
    ))
    # Should have completed without raising.
    assert result is not None


@test("8.3 engine increments budget.iterations as it runs")
def test_engine_increments_iterations():
    import asyncio
    from src.core.types import (
        Direction, FrameworkID, Problem, Variable,
    )
    from src.llm.client import ClientMode, LLMClient
    from src.llm.engine import run_async_formation

    client = LLMClient(mode=ClientMode.MOCK)
    budget = BudgetTracker()  # default caps — won't constrain
    problem = Problem(
        statement="should I X or Y?",
        variables=[Variable(
            name="v", description="x", magnitude=0.5,
            direction=Direction.NEUTRAL, confidence=0.7,
            source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True,
        )],
    )

    asyncio.run(run_async_formation(
        problem=problem, client=client, max_iterations=2, budget=budget,
    ))
    # Engine ran ≥1 iteration → budget should know
    assert budget.state.iterations >= 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_default_caps,
    test_fresh_check,
    test_iteration_breach,
    test_cost_breach,
    test_wall_time_breach,
    test_mcp_breach,
    test_record_sonnet,
    test_record_unknown_model,
    test_record_response_success,
    test_record_response_failure,
    test_record_response_missing_fields,
    test_custom_caps,
    test_increment_iter,
    test_increment_mcp,
    test_summary_shape,
    test_summary_breach_reason,
    test_breach_idempotent,
    test_breached_property,
    test_check_no_side_effects,
    test_precedence_walltime,
    test_remaining_nonneg,
    test_engine_budget_pre_breached,
    test_engine_no_budget,
    test_engine_increments_iterations,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} budget enforcer tests...")
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
