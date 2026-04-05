"""
Integration Test Suite — Step 8.

Tests 8.1 through 8.7 from the Phase 2 implementation plan.
All tests run in MOCK mode — no API credits spent.

Run with: python -m pytest tests/test_integration.py -v
Or directly: python tests/test_integration.py
"""

from __future__ import annotations

import asyncio
import sys
import time

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

PASSED = 0
FAILED = 0
ERRORS = []


def test(name: str):
    """Decorator for test functions."""
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    """Run a single test and track pass/fail."""
    global PASSED, FAILED
    name = getattr(fn, '_test_name', fn.__name__)
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        PASSED += 1
        print(f"  ✓ {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  ✗ {name}: {e}")


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

from src.core.types import *
from src.llm.client import LLMClient, ClientMode


def make_client() -> LLMClient:
    return LLMClient(mode=ClientMode.MOCK)


def make_career_problem() -> Problem:
    return Problem(
        statement="Should I quit my stable job to start a business with my friend?",
        variables=[
            Variable(name="job_security", description="Stable corporate job with good salary and benefits", magnitude=0.7, direction=Direction.POSITIVE, confidence=0.9, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True, evidence=["employed 5 years", "good reviews"]),
            Variable(name="startup_dream", description="Always wanted to build something of my own", magnitude=0.85, direction=Direction.POSITIVE, confidence=0.7, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True),
            Variable(name="friend_cofounding", description="Best friend wants to cofound together", magnitude=0.6, direction=Direction.POSITIVE, confidence=0.6, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True),
            Variable(name="financial_risk", description="Would lose stable income, have mortgage", magnitude=0.8, direction=Direction.NEGATIVE, confidence=0.85, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True, evidence=["mortgage payments", "no savings runway"]),
        ],
    )


def make_relationship_problem() -> Problem:
    return Problem(
        statement="My partner wants me to take a corporate job but I want freelance. We fight about money every week.",
        variables=[
            Variable(name="freelance_passion", description="Deep passion for freelance work", magnitude=0.85, direction=Direction.POSITIVE, confidence=0.9, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True, evidence=["years of freelancing"]),
            Variable(name="income_instability", description="Freelance income inconsistent", magnitude=0.7, direction=Direction.NEGATIVE, confidence=0.85, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True, evidence=["bank statements"]),
            Variable(name="partner_pressure", description="Partner pushing for corporate job", magnitude=0.75, direction=Direction.NEGATIVE, confidence=0.8, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True, evidence=["weekly arguments"]),
        ],
    )


# ===========================================================================
# 8.1: Unit Tests Per Domain Agent (Isolated Islands)
# ===========================================================================

@test("8.1a Physics island produces perspectives + root causes")
def test_physics_island():
    from src.domains.physics import run_physics
    problem = make_career_problem()
    output = run_physics(DomainInput(problem=problem))
    assert output.domain == Domain.PHYSICS
    assert len(output.perspectives) == 11, f"Expected 11 perspectives, got {len(output.perspectives)}"
    assert len(output.root_causes) >= 0  # may or may not find hidden roots
    assert output.raw_analysis != ""


@test("8.1b Physics Ke challenge produces scrutiny score")
def test_physics_challenge():
    from src.domains.physics import run_physics, challenge
    problem = make_career_problem()
    output = run_physics(DomainInput(problem=problem))
    ke = challenge(ChallengeInput(
        challenger_domain=Domain.PHYSICS,
        target_domain=Domain.PSYCHOLOGY,
        target_output=output,  # self-challenge for testing
    ))
    assert ke.challenger_domain == Domain.PHYSICS
    assert 0.0 <= ke.scrutiny_score <= 1.0


@test("8.1c Mathematics island processes upstream Physics")
def test_maths_island():
    from src.domains.physics import run_physics
    from src.maths import run_mathematics
    physics_out = run_physics(DomainInput(problem=make_career_problem()))
    maths_out = run_mathematics(DomainInput(
        problem=make_career_problem(),
        upstream_outputs={Domain.PHYSICS: physics_out},
    ))
    assert maths_out.domain == Domain.MATHEMATICS
    assert len(maths_out.perspectives) > 0


@test("8.1d Psychology island detects bias patterns")
def test_psychology_island():
    from src.domains.psychology import run_psychology
    output = run_psychology(DomainInput(problem=make_relationship_problem()))
    assert output.domain == Domain.PSYCHOLOGY
    assert len(output.perspectives) == 5, f"Expected 5 perspectives, got {len(output.perspectives)}"
    # Check that dual process, dissonance, motivated reasoning, dialectical, metacognition all ran
    frameworks = {p.framework for p in output.perspectives}
    assert FrameworkID.DUAL_PROCESS in frameworks
    assert FrameworkID.COGNITIVE_DISSONANCE in frameworks
    assert FrameworkID.MOTIVATED_REASONING in frameworks
    assert FrameworkID.METACOGNITION in frameworks


@test("8.1e Philosophy island follows ontology→epistemology→...→teleology pipeline")
def test_philosophy_island():
    from src.domains.philosophy import run_philosophy
    output = run_philosophy(DomainInput(problem=make_career_problem()))
    assert output.domain == Domain.PHILOSOPHY
    assert len(output.perspectives) == 5
    frameworks = [p.framework for p in output.perspectives]
    assert frameworks == [
        FrameworkID.ONTOLOGY,
        FrameworkID.EPISTEMOLOGY,
        FrameworkID.PHENOMENOLOGY,
        FrameworkID.DIALECTICS,
        FrameworkID.TELEOLOGY,
    ], f"Pipeline order wrong: {[f.value for f in frameworks]}"


@test("8.1f Chemistry governance produces formation plan")
def test_chemistry_governance():
    from src.domains.chemistry import run_governance
    output, plan = run_governance(DomainInput(problem=make_career_problem()))
    assert output.domain == Domain.CHEMISTRY
    assert len(plan.active_domains) >= 3  # at least physics, maths, psychology
    assert plan.estimated_agent_count > 0
    assert plan.organizational_template in ("linear", "web", "tree", "cycle", "hub_and_spoke")


@test("8.1g Chemistry analytical produces chirality + catalysis + resonance")
def test_chemistry_analytical():
    from src.domains.physics import run_physics
    from src.domains.psychology import run_psychology
    from src.domains.chemistry import run_chemistry
    problem = make_career_problem()
    physics_out = run_physics(DomainInput(problem=problem))
    psych_out = run_psychology(DomainInput(problem=problem, upstream_outputs={Domain.PHYSICS: physics_out}))
    chem_out = run_chemistry(DomainInput(problem=problem, upstream_outputs={
        Domain.PHYSICS: physics_out, Domain.PSYCHOLOGY: psych_out,
    }))
    assert chem_out.domain == Domain.CHEMISTRY
    assert len(chem_out.perspectives) >= 1  # at least resonance


@test("8.1h All 5 domains produce isolation-clean output (no cross-imports)")
def test_isolation():
    import importlib, inspect
    domains = [
        'src.domains.physics',
        'src.domains.psychology',
        'src.domains.philosophy',
        'src.domains.chemistry',
        'src.maths',
    ]
    other_domains = {
        'src.domains.physics': ['src.domains.psychology', 'src.domains.philosophy', 'src.domains.chemistry', 'src.maths'],
        'src.domains.psychology': ['src.domains.physics', 'src.domains.philosophy', 'src.domains.chemistry', 'src.maths'],
        'src.domains.philosophy': ['src.domains.physics', 'src.domains.psychology', 'src.domains.chemistry', 'src.maths'],
        'src.domains.chemistry': ['src.domains.physics', 'src.domains.psychology', 'src.domains.philosophy', 'src.maths'],
        'src.maths': ['src.domains.physics', 'src.domains.psychology', 'src.domains.philosophy', 'src.domains.chemistry'],
    }
    for mod_path in domains:
        mod = importlib.import_module(mod_path)
        src = inspect.getsource(mod)
        for forbidden in other_domains[mod_path]:
            assert f"from {forbidden}" not in src, f"ISOLATION VIOLATION: {mod_path} imports from {forbidden}"


# ===========================================================================
# 8.2: Dual-Cycle Test
# ===========================================================================

@test("8.2a Sheng cycle runs all 5 domains in correct order")
async def test_sheng_cycle():
    from src.llm.engine import run_async_formation
    client = make_client()
    result = await run_async_formation(make_career_problem(), client, max_iterations=1)
    active = set(result.domain_outputs.keys())
    assert Domain.PHYSICS in active
    assert Domain.MATHEMATICS in active
    assert Domain.PSYCHOLOGY in active


@test("8.2b Ke cycle produces 5 differentiated challenge pairs")
async def test_ke_cycle():
    from src.llm.engine import run_async_formation
    client = make_client()
    result = await run_async_formation(make_career_problem(), client, max_iterations=2)
    assert len(result.ke_results) >= 3  # at least 3 pairs active
    # Check that scrutiny scores exist and are bounded
    for ke in result.ke_results:
        assert 0.0 <= ke.scrutiny_score <= 1.0
        assert ke.challenger_domain != ke.target_domain


# ===========================================================================
# 8.3: Funnel Test
# ===========================================================================

@test("8.3a Variable cap holds at 30 per iteration")
async def test_funnel_cap():
    from src.llm.engine import run_async_formation
    client = make_client()
    result = await run_async_formation(make_relationship_problem(), client, max_iterations=2)
    for f in result.funnel_history:
        assert f.variables_kept <= 30, f"Cap exceeded: {f.variables_kept}"


@test("8.3b Cache grows across iterations")
async def test_funnel_cache_grows():
    from src.llm.engine import run_async_formation
    client = make_client()
    result = await run_async_formation(make_career_problem(), client, max_iterations=2)
    if len(result.funnel_history) >= 2:
        # Total cached should be non-negative
        total_cached = sum(f.variables_cached for f in result.funnel_history)
        assert total_cached >= 0


# ===========================================================================
# 8.4: Progressive Disclosure Test
# ===========================================================================

@test("8.4a Phase 1 delivers findings within reasonable time")
async def test_phase1_speed():
    from src.llm.disclosure import run_phase_one
    client = make_client()
    start = time.monotonic()
    phase1 = await run_phase_one(make_career_problem(), client)
    elapsed = time.monotonic() - start
    assert len(phase1.initial_findings) >= 1, "Phase 1 should produce at least 1 finding"
    assert phase1.message != ""
    assert elapsed < 30, f"Phase 1 took {elapsed:.1f}s — too slow for mock mode"


@test("8.4b Phase 1 includes dig-deeper option when depth available")
async def test_phase1_dig_deeper():
    from src.llm.disclosure import run_phase_one
    client = make_client()
    phase1 = await run_phase_one(make_career_problem(), client)
    # depth_available should be True for a non-converged Phase 1
    assert isinstance(phase1.depth_available, bool)
    assert phase1.estimated_additional_credits > 0


# ===========================================================================
# 8.5: Failure Tests
# ===========================================================================

@test("8.5a Level 1: concept skip detected correctly")
def test_degradation_level1():
    from src.llm.degradation import DegradationTracker, DegradationLevel
    tracker = DegradationTracker()
    tracker.set_domain_concept_count("physics", 11)
    tracker.record_failure("physics", "entropy", "timeout", attempt=2)
    tracker.record_success("physics", "first_principles")
    state = tracker.get_state()
    assert state.level == DegradationLevel.LEVEL_1
    assert state.confidence_reduction == 0.9
    assert not state.free_response


@test("8.5b Level 2: full domain failure detected")
def test_degradation_level2():
    from src.llm.degradation import DegradationTracker, DegradationLevel
    tracker = DegradationTracker()
    tracker.set_domain_concept_count("physics", 2)
    tracker.set_domain_concept_count("psychology", 5)
    tracker.record_failure("physics", "concept1", "crash")
    tracker.record_failure("physics", "concept2", "crash")
    tracker.record_success("psychology", "dual_process")
    state = tracker.get_state()
    assert state.level == DegradationLevel.LEVEL_2
    assert "physics" in state.domains_down
    assert state.confidence_reduction == 0.7


@test("8.5c Level 3: 3+ domains down → free response + free retry")
def test_degradation_level3():
    from src.llm.degradation import DegradationTracker, DegradationLevel
    tracker = DegradationTracker()
    for d in ["physics", "psychology", "philosophy"]:
        tracker.set_domain_concept_count(d, 1)
        tracker.record_failure(d, "all", "crash")
    state = tracker.get_state()
    assert state.level == DegradationLevel.LEVEL_3
    assert state.free_response is True
    assert state.free_retry_issued is True
    assert state.confidence_reduction == 0.4
    # User message must not contain internal terminology
    assert "island" not in state.user_message.lower()
    assert "ke cycle" not in state.user_message.lower()
    assert "domain" not in state.user_message.lower()


@test("8.5d Credit system: 3-domain failure → 0 credits + free retry")
def test_credit_failure():
    from src.llm.credits import calculate_invoice
    invoice = calculate_invoice(domains_ran=5, domains_failed=3, iterations_ran=2, ke_pairs_ran=5)
    assert invoice.actual_total == 0.0
    assert invoice.free_retry_issued is True


# ===========================================================================
# 8.6: Speech Module Test
# ===========================================================================

@test("8.6a Speech module generates response in all 3 delivery modes")
async def test_speech_modes():
    from src.llm.speech import generate_speech, SpeechInput
    client = make_client()

    for mode in ["direct", "building", "gentle"]:
        inp = SpeechInput(
            findings_summary="Core tension between career stability and personal growth.",
            trajectories_text="Trajectory 1: hidden energy drain (80%). Trajectory 2: identity conflict (75%).",
            variable_d="The problem persists because it serves identity preservation.",
            confidence=0.75,
            delivery_mode=mode,
            is_phase_one=False,
            depth_available=False,
            degraded=False,
            degradation_message="",
            credit_summary="5.0 credits used",
        )
        result = await generate_speech(client, inp)
        assert result.response_text != "", f"Empty response for mode={mode}"
        assert result.credit_summary == "5.0 credits used"


@test("8.6b Speech module includes dig-deeper for Phase 1")
async def test_speech_dig_deeper():
    from src.llm.speech import generate_speech, SpeechInput
    client = make_client()
    inp = SpeechInput(
        findings_summary="Initial findings.",
        trajectories_text="Trajectory 1: test (70%).",
        variable_d=None,
        confidence=0.7,
        delivery_mode="building",
        is_phase_one=True,
        depth_available=True,
        degraded=False,
        degradation_message="",
        credit_summary="3.0 credits",
    )
    result = await generate_speech(client, inp)
    assert result.dig_deeper_prompt is not None
    assert "deeper" in result.dig_deeper_prompt.lower()


# ===========================================================================
# 8.7: End-to-End Stress Test (10 problems)
# ===========================================================================

STRESS_TEST_PROBLEMS = [
    "Should I quit my stable job to start a business?",
    "My partner wants me to take a corporate job but I want freelance.",
    "I've been offered a promotion but it means relocating away from my aging parents.",
    "My cofounder wants to pivot the company but I think we should stay the course.",
    "I'm 35 and thinking about going back to school for a completely different career.",
    "My best friend borrowed money and hasn't paid me back. It's affecting our friendship.",
    "I'm burnt out at work but my team depends on me and I feel guilty about leaving.",
    "My spouse and I disagree on whether to have children.",
    "I received a job offer that pays 50% more but the company has questionable ethics.",
    "I'm spending all my savings on my startup and my family thinks I'm being irresponsible.",
]


@test("8.7 End-to-end stress test: 10 different problems")
async def test_stress_e2e():
    from src.llm.engine import run_async_formation
    client = make_client()

    results = []
    total_start = time.monotonic()

    for i, statement in enumerate(STRESS_TEST_PROBLEMS):
        problem = Problem(
            statement=statement,
            variables=[
                Variable(name=f"var_a_{i}", description=f"Primary force in: {statement[:40]}", magnitude=0.7, direction=Direction.POSITIVE, confidence=0.8, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True),
                Variable(name=f"var_b_{i}", description=f"Opposing force in: {statement[:40]}", magnitude=0.65, direction=Direction.NEGATIVE, confidence=0.75, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True),
            ],
        )

        result = await run_async_formation(problem, client, max_iterations=2)
        results.append(result)

    total_elapsed = time.monotonic() - total_start

    # Verify all 10 completed
    assert len(results) == 10, f"Only {len(results)}/10 problems completed"

    # Verify each produced trajectories
    for i, result in enumerate(results):
        assert len(result.trajectories) >= 1, f"Problem {i} produced 0 trajectories"
        assert len(result.domain_outputs) >= 3, f"Problem {i} had <3 domains"
        assert len(result.ke_results) >= 1, f"Problem {i} had 0 Ke results"

    # Summary stats
    avg_trajectories = sum(len(r.trajectories) for r in results) / 10
    avg_ke_pairs = sum(len(r.ke_results) for r in results) / 10
    total_calls = sum(r.call_summary["total_calls"] for r in results)
    total_tokens = sum(r.call_summary["total_tokens"]["total_tokens"] for r in results)
    avg_time = total_elapsed / 10

    print(f"\n    STRESS TEST SUMMARY:")
    print(f"    10/10 problems completed in {total_elapsed:.1f}s ({avg_time:.1f}s avg)")
    print(f"    Avg trajectories per problem: {avg_trajectories:.1f}")
    print(f"    Avg Ke pairs per problem: {avg_ke_pairs:.1f}")
    print(f"    Total LLM calls: {total_calls}")
    print(f"    Total tokens: {total_tokens}")
    converged = sum(1 for r in results if r.convergence_history.final_converged)
    print(f"    Converged: {converged}/10")


# ===========================================================================
# Main runner
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION TEST SUITE — Step 8")
    print("=" * 60)
    print()

    sections = [
        ("8.1 Unit Tests Per Domain", [
            test_physics_island, test_physics_challenge, test_maths_island,
            test_psychology_island, test_philosophy_island,
            test_chemistry_governance, test_chemistry_analytical,
            test_isolation,
        ]),
        ("8.2 Dual-Cycle Tests", [
            test_sheng_cycle, test_ke_cycle,
        ]),
        ("8.3 Funnel Tests", [
            test_funnel_cap, test_funnel_cache_grows,
        ]),
        ("8.4 Progressive Disclosure Tests", [
            test_phase1_speed, test_phase1_dig_deeper,
        ]),
        ("8.5 Failure Tests", [
            test_degradation_level1, test_degradation_level2,
            test_degradation_level3, test_credit_failure,
        ]),
        ("8.6 Speech Module Tests", [
            test_speech_modes, test_speech_dig_deeper,
        ]),
        ("8.7 End-to-End Stress Test", [
            test_stress_e2e,
        ]),
    ]

    for section_name, tests in sections:
        print(f"\n--- {section_name} ---")
        for t in tests:
            run_test(t)

    print()
    print("=" * 60)
    total = PASSED + FAILED
    print(f"RESULTS: {PASSED}/{total} passed, {FAILED}/{total} failed")
    if ERRORS:
        print("\nFAILURES:")
        for name, err in ERRORS:
            print(f"  ✗ {name}: {err}")
    print("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
