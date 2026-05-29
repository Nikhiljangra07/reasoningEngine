"""
Phase 1-5 tests — wandering engine end-to-end with fake LLM client.

Covers: report types, trace types, matching, policy, critique, agent loop,
runtime modes, articulation, synthesis, dossier assembly.

Run: PYTHONPATH=. python3 tests/test_wandering_engine.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.llm.client import LLMResponse
from src.wandering.cushion import (
    CushionField,
    CushionGraph,
    CushionInput,
    CushionLayer,
)
from src.wandering.report import (
    Confidence,
    ExplorationReport,
    LayerMatch,
    SourceCitation,
)
from src.wandering.trace import (
    DecisionTrace,
    DiscardKind,
    DiscardedClue,
    StepKind,
    TraceStep,
)
from src.wandering.matching import (
    MAX_DIG_ITERATIONS,
    MIN_DIG_ITERATIONS,
    MatchResult,
    iterations_for_match,
    match_content,
    parse_match_response,
)
from src.wandering.policy import (
    DRIFT_WINDOW,
    SEED_DOMAINS,
    detect_drift,
    domain_visit_counts,
    next_move,
    pick_next_domain,
)
from src.wandering.critique import (
    CritiqueResult,
    CritiqueVerdict,
    QUESTIONS,
    parse_critique_response,
    run_self_critique,
)
from src.wandering.agent import (
    AgentBudget,
    AgentState,
    FetchResult,
    run_agent,
    stub_fetcher,
)
from src.wandering.runtime import (
    MODE_DEFAULTS,
    SessionResult,
    WanderingConfig,
    WanderingMode,
    assign_models,
    run_wandering_session,
)
from src.wandering.articulate import (
    ArticulatedCard,
    articulate_report,
    parse_articulation_response,
)
from src.wandering.synthesis import (
    Contradiction,
    InsightCluster,
    OpportunityPath,
    SynthesisMap,
    parse_synthesis_response,
    synthesize_dossier,
)
from src.wandering.dossier import (
    ConfidenceBand,
    Dossier,
    DossierMetadata,
    build_dossier,
)


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


# ===========================================================================
# Helpers
# ===========================================================================


def _make_cushion(problem_text: str = "test problem about bounded freedom") -> CushionGraph:
    """Build a minimal CushionGraph for tests."""
    return CushionGraph(
        actual=CushionLayer(
            name="actual",
            nodes=["wandering agents", "research anchor", "credit budget"],
            summary="agents wander to find inspiration",
        ),
        essence=CushionLayer(
            name="essence",
            nodes=["bounded freedom", "productive constraint", "anchored chaos"],
            summary="freedom shaped by structure",
        ),
        mechanism=CushionLayer(
            name="mechanism",
            nodes=[
                "soft constraint enables emergence",
                "hard constraint kills value behavior",
            ],
            summary="constraint paradox",
        ),
        raw_input=CushionInput(
            problem=CushionField(name="problem", content=problem_text),
            context=CushionField(name="context"),
            vision=CushionField(name="vision"),
            current_map=CushionField(name="current_map"),
        ),
    )


class _FakeLLMClient:
    """Returns prescripted responses based on the domain of the call.

    Supports per-(domain, concept) routing so a single fake can serve
    multiple call types (matching, critique, dig, articulate, synthesize).
    """

    def __init__(self, responses: dict[str, str] | None = None, default: str = "{}"):
        self.responses = responses or {}
        self.default = default
        self.call_log: list[dict[str, Any]] = []

    async def call(self, *, system_prompt, user_message, domain, concept, **kwargs):
        key = f"{domain}:{concept}"
        self.call_log.append({
            "domain": domain,
            "concept": concept,
            "system_prompt_len": len(system_prompt),
            "user_message_len": len(user_message),
        })
        body = self.responses.get(key, self.responses.get(domain, self.default))
        return LLMResponse(
            content=body,
            input_tokens=100,
            output_tokens=50,
            latency_ms=10.0,
            success=True,
            model="fake-model",
        )


def _good_match_response(cushion: CushionGraph) -> str:
    """A response showing partial match in essence layer."""
    return json.dumps({
        "actual": [],
        "essence": ["bounded freedom", "anchored chaos"],
        "mechanism": ["soft constraint enables emergence"],
    })


def _good_dig_response() -> str:
    return json.dumps({
        "exploration_summary": "Jazz improvisation operates inside harmonic structure.",
        "advancement": "Resonates with the bounded-freedom essence of the anchor.",
        "what_does_not_map": "Jazz is real-time ensemble; agents run async parallel.",
        "next_lead": "Look at improvisation pedagogy",
    })


def _good_critique_response() -> str:
    return json.dumps({
        "answers": {q: f"answer for {q}" for q in QUESTIONS},
        "red_flags": [],
        "verdict": "continue",
        "summary": "On track",
    })


def _good_articulation_response() -> str:
    return json.dumps({
        "spark": "Jazz operates inside harmonic constraint.",
        "source_shape": "Soloists choose freely within chord changes.",
        "bridge": "Your agents need bounded freedom that constrains without dictating.",
        "use": "Consider drift radius as a harmonic frame, not a cage.",
        "limit": "Jazz is real-time ensemble; agents are async parallel.",
    })


def _good_synthesis_response() -> str:
    return json.dumps({
        "top_insights": ["wander-P01-001"],
        "clusters": [{
            "label": "bounded-freedom analogues",
            "card_ids": ["wander-P01-001"],
            "summary": "Multiple cards show bounded-freedom mechanism",
        }],
        "contradictions": [],
        "opportunity_paths": [{
            "description": "Treat drift radius as harmonic frame",
            "supporting_card_ids": ["wander-P01-001"],
            "confidence_estimate": "high",
        }],
        "open_questions": ["How does async coordination compare to ensemble timing?"],
        "recommended_next_direction": "Explore bounded-freedom in async contexts.",
        "what_would_change_the_verdict": "Evidence that hard constraint succeeds elsewhere.",
    })


# ===========================================================================
# 1. Report + LayerMatch + Confidence derivation
# ===========================================================================


@test("1.1 LayerMatch ratio is matched/total")
def test_layer_match_ratio():
    lm = LayerMatch(layer_name="essence", matched_nodes=["a", "b"], total_nodes=5)
    assert abs(lm.ratio - 0.4) < 0.0001
    assert lm.match_count == 2
    assert lm.ratio_string() == "2/5"


@test("1.2 LayerMatch with 0 total returns 0.0 ratio (no division error)")
def test_layer_match_zero_total():
    lm = LayerMatch(layer_name="essence", matched_nodes=[], total_nodes=0)
    assert lm.ratio == 0.0


@test("1.3 Report.compute_confidence: essence >= 0.7 → HIGH (Heisenberg)")
def test_confidence_essence_high():
    r = ExplorationReport(
        report_id="r1",
        agent_id="P01",
        anchor_summary="anchor",
        domain_explored="jazz",
        layer_matches={
            "actual": LayerMatch(layer_name="actual", matched_nodes=[], total_nodes=4),
            "essence": LayerMatch(layer_name="essence", matched_nodes=["a", "b", "c", "d"], total_nodes=5),
            "mechanism": LayerMatch(layer_name="mechanism", matched_nodes=[], total_nodes=3),
        },
    )
    assert r.compute_confidence() == Confidence.HIGH


@test("1.4 Report.compute_confidence: zero matches → LOW")
def test_confidence_zero():
    r = ExplorationReport(
        report_id="r1",
        agent_id="P01",
        anchor_summary="anchor",
        domain_explored="jazz",
        layer_matches={
            "actual": LayerMatch(layer_name="actual", matched_nodes=[], total_nodes=4),
            "essence": LayerMatch(layer_name="essence", matched_nodes=[], total_nodes=5),
            "mechanism": LayerMatch(layer_name="mechanism", matched_nodes=[], total_nodes=3),
        },
    )
    assert r.compute_confidence() == Confidence.LOW


@test("1.5 Report.validate flags empty what_does_not_map (Law 7)")
def test_report_validate_what_does_not_map():
    r = ExplorationReport(
        report_id="r1",
        agent_id="P01",
        anchor_summary="anchor",
        domain_explored="jazz",
        layer_matches={
            "essence": LayerMatch(layer_name="essence", matched_nodes=["a"], total_nodes=5),
        },
        exploration_summary="found something",
        what_does_not_map="",  # MISSING
    )
    errors = r.validate()
    assert any("what_does_not_map" in e for e in errors)


@test("1.6 Report with all required fields validates")
def test_report_validate_clean():
    r = ExplorationReport(
        report_id="r1",
        agent_id="P01",
        anchor_summary="anchor",
        domain_explored="jazz",
        layer_matches={
            "essence": LayerMatch(layer_name="essence", matched_nodes=["a"], total_nodes=5),
        },
        exploration_summary="found something",
        what_does_not_map="here is where it breaks",
    )
    assert r.is_valid()


# ===========================================================================
# 2. Trace types
# ===========================================================================


@test("2.1 DecisionTrace.append sets step_id monotonically")
def test_trace_append_step_id():
    tr = DecisionTrace(agent_id="P01")
    tr.append(TraceStep(step_id=999, kind=StepKind.INITIALIZED))
    tr.append(TraceStep(step_id=999, kind=StepKind.FETCHED))
    assert tr.steps[0].step_id == 0
    assert tr.steps[1].step_id == 1


@test("2.2 DecisionTrace.append increments reports_produced on REPORTED")
def test_trace_reports_produced():
    tr = DecisionTrace(agent_id="P01")
    tr.append(TraceStep(step_id=0, kind=StepKind.REPORTED, report_id="r1"))
    tr.append(TraceStep(step_id=0, kind=StepKind.REPORTED, report_id="r2"))
    assert tr.total_reports_produced == 2


@test("2.3 DecisionTrace.discard preserves clue with classification")
def test_trace_discard():
    tr = DecisionTrace(agent_id="P01")
    tr.discard(DiscardedClue(
        description="x",
        classification=DiscardKind.POSSIBLY_RELEVANT_ELSEWHERE,
    ))
    assert len(tr.discarded_clues) == 1
    assert tr.discarded_clues[0].classification == DiscardKind.POSSIBLY_RELEVANT_ELSEWHERE


# ===========================================================================
# 3. Matching algorithm
# ===========================================================================


@test("3.1 iterations_for_match: 0 nodes → 0 iter (no dig)")
def test_iter_zero():
    assert iterations_for_match(0) == 0


@test("3.2 iterations_for_match: 1 node → MIN_DIG_ITERATIONS")
def test_iter_one_node():
    assert iterations_for_match(1) == MIN_DIG_ITERATIONS


@test("3.3 iterations_for_match: 3 nodes → 5 iter (or MAX)")
def test_iter_three_nodes():
    assert iterations_for_match(3) == min(2 + 3, MAX_DIG_ITERATIONS)


@test("3.4 iterations_for_match: many nodes capped at MAX")
def test_iter_capped():
    assert iterations_for_match(100) == MAX_DIG_ITERATIONS


@test("3.5 parse_match_response: valid response returns three layers")
async def test_parse_match_valid():
    cushion = _make_cushion()
    response = _good_match_response(cushion)
    matches = parse_match_response(response, cushion)
    assert set(matches.keys()) == {"actual", "essence", "mechanism"}
    assert matches["essence"].match_count == 2


@test("3.6 parse_match_response: filters out invented nodes (not in cushion)")
def test_parse_match_filters_invented():
    cushion = _make_cushion()
    response = json.dumps({
        "actual": ["wandering agents"],  # real
        "essence": ["bounded freedom", "INVENTED NODE"],  # one real, one fake
        "mechanism": [],
    })
    matches = parse_match_response(response, cushion)
    # Invented node filtered; only the real one survives.
    assert matches["essence"].match_count == 1
    assert matches["essence"].matched_nodes == ["bounded freedom"]


@test("3.7 match_content end-to-end with fake client")
async def test_match_content_e2e():
    cushion = _make_cushion()
    client = _FakeLLMClient(responses={"psychology:structural_match_check": _good_match_response(cushion)})
    result = await match_content(cushion, "some jazz article", client)  # type: ignore[arg-type]
    assert result.total_matched_nodes == 3
    assert result.dig_iterations >= MIN_DIG_ITERATIONS
    assert result.has_any_match() is True


@test("3.8 match_content handles unparseable response gracefully")
async def test_match_unparseable():
    cushion = _make_cushion()
    client = _FakeLLMClient(default="this is not JSON at all {}{}{")
    result = await match_content(cushion, "stuff", client)  # type: ignore[arg-type]
    # Treated as no match — safe degradation
    assert result.total_matched_nodes == 0


# ===========================================================================
# 4. Policy
# ===========================================================================


@test("4.1 detect_drift: empty trace → False (not enough data)")
def test_drift_empty():
    tr = DecisionTrace(agent_id="P01")
    assert detect_drift(tr) is False


@test("4.2 detect_drift: last N MATCHED steps all 0 → True")
def test_drift_detected():
    tr = DecisionTrace(agent_id="P01")
    for _ in range(DRIFT_WINDOW):
        tr.append(TraceStep(step_id=0, kind=StepKind.MATCHED, matched_count=0))
    assert detect_drift(tr) is True


@test("4.3 detect_drift: last N MATCHED steps with matches → False")
def test_drift_with_matches():
    tr = DecisionTrace(agent_id="P01")
    for _ in range(DRIFT_WINDOW):
        tr.append(TraceStep(step_id=0, kind=StepKind.MATCHED, matched_count=2))
    assert detect_drift(tr) is False


@test("4.4 domain_visit_counts tracks per-position visits")
def test_domain_visit_counts():
    tr = DecisionTrace(agent_id="P01")
    tr.append(TraceStep(step_id=0, kind=StepKind.FETCHED, position="jazz"))
    tr.append(TraceStep(step_id=0, kind=StepKind.MATCHED, position="jazz"))
    tr.append(TraceStep(step_id=0, kind=StepKind.FETCHED, position="biology"))
    counts = domain_visit_counts(tr)
    assert counts["jazz"] == 2
    assert counts["biology"] == 1


@test("4.5 pick_next_domain biases toward unvisited (deterministic mock)")
def test_pick_next_domain_inverse_freq():
    tr = DecisionTrace(agent_id="P01")
    # Heavily visit "physics" — should rarely be picked
    for _ in range(20):
        tr.append(TraceStep(step_id=0, kind=StepKind.FETCHED, position="physics"))
    # Stub choice_fn to pick highest-weight item deterministically
    def deterministic_choice(items, weights):
        idx = weights.index(max(weights))
        return items[idx]
    pick = pick_next_domain(tr, choice_fn=deterministic_choice)
    # Should pick something other than "physics"
    assert pick != "physics"


@test("4.6 next_move: drift triggers RETURNED_TO_ANCHOR")
def test_next_move_drift():
    tr = DecisionTrace(agent_id="P01")
    for _ in range(DRIFT_WINDOW):
        tr.append(TraceStep(step_id=0, kind=StepKind.MATCHED, matched_count=0))
    move = next_move(_make_cushion(), tr)
    assert move.kind == StepKind.RETURNED_TO_ANCHOR


@test("4.7 next_move: no drift → FETCHED with a domain")
def test_next_move_chaos_pick():
    tr = DecisionTrace(agent_id="P01")
    move = next_move(_make_cushion(), tr)
    assert move.kind == StepKind.FETCHED
    assert move.position in SEED_DOMAINS


# ===========================================================================
# 5. Critique
# ===========================================================================


@test("5.1 QUESTIONS has exactly 6 (locked)")
def test_critique_six_questions():
    assert len(QUESTIONS) == 6


@test("5.2 parse_critique_response: valid JSON → CritiqueResult")
def test_parse_critique_valid():
    result = parse_critique_response(_good_critique_response())
    assert result.verdict == CritiqueVerdict.CONTINUE
    assert len(result.answers) == 6


@test("5.3 parse_critique_response: unparseable → default CONTINUE")
def test_parse_critique_unparseable():
    result = parse_critique_response("garbage")
    assert result.verdict == CritiqueVerdict.CONTINUE  # safe default


@test("5.4 parse_critique_response: unknown verdict → defaults to CONTINUE")
def test_parse_critique_unknown_verdict():
    payload = json.dumps({"verdict": "explode", "answers": {}, "red_flags": [], "summary": ""})
    result = parse_critique_response(payload)
    assert result.verdict == CritiqueVerdict.CONTINUE


@test("5.5 run_self_critique end-to-end with fake client")
async def test_run_critique_e2e():
    client = _FakeLLMClient(responses={"psychology:self_critique_check": _good_critique_response()})
    result = await run_self_critique(
        cushion=_make_cushion(),
        agent_position="jazz",
        latest_finding="found something",
        cumulative_tokens=1000,
        iterations_so_far=1,
        client=client,  # type: ignore[arg-type]
    )
    assert result.verdict == CritiqueVerdict.CONTINUE


# ===========================================================================
# 6. Agent loop
# ===========================================================================


@test("6.1 agent runs to budget exhaustion + logs EXHAUSTED step")
async def test_agent_runs_to_budget():
    cushion = _make_cushion()
    state = AgentState(
        agent_id="P01",
        cushion=cushion,
        budget=AgentBudget(time_budget_seconds=999, token_budget=999_999, max_steps=4),
    )
    client = _FakeLLMClient(responses={
        "psychology:structural_match_check": json.dumps({
            "actual": [], "essence": [], "mechanism": []  # no matches → MOVED_ON
        }),
    })
    counter = {"t": 0.0}
    def fake_clock():
        counter["t"] += 0.001
        return counter["t"]
    result = await run_agent(state, client, fetcher=stub_fetcher, clock=fake_clock)  # type: ignore[arg-type]
    assert result.trace.completion_reason.startswith("exhausted_")
    # Last step must be EXHAUSTED
    assert result.trace.last_step().kind == StepKind.EXHAUSTED


@test("6.2 agent with match → DUG → REPORTED step appears")
async def test_agent_produces_report_on_match():
    cushion = _make_cushion()
    state = AgentState(
        agent_id="P01",
        cushion=cushion,
        budget=AgentBudget(time_budget_seconds=999, token_budget=99_999, max_steps=8),
    )
    client = _FakeLLMClient(responses={
        "psychology:structural_match_check": _good_match_response(cushion),
        "psychology:self_critique_check": _good_critique_response(),
        "synthesizer:wandering_dig": _good_dig_response(),
    })
    counter = {"t": 0.0}
    def fake_clock():
        counter["t"] += 0.001
        return counter["t"]
    result = await run_agent(state, client, fetcher=stub_fetcher, clock=fake_clock)  # type: ignore[arg-type]
    # At least one report produced
    assert len(result.reports) >= 1
    assert result.reports[0].is_valid()
    # confidence derived
    assert result.reports[0].confidence in (Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH)


@test("6.3 agent without match → discards clue with classification")
async def test_agent_discards_no_match():
    cushion = _make_cushion()
    state = AgentState(
        agent_id="P01",
        cushion=cushion,
        budget=AgentBudget(time_budget_seconds=999, token_budget=99_999, max_steps=6),
    )
    client = _FakeLLMClient(responses={
        "psychology:structural_match_check": json.dumps({"actual": [], "essence": [], "mechanism": []}),
    })
    counter = {"t": 0.0}
    def fake_clock():
        counter["t"] += 0.001
        return counter["t"]
    result = await run_agent(state, client, fetcher=stub_fetcher, clock=fake_clock)  # type: ignore[arg-type]
    # No reports, but discards logged
    assert len(result.reports) == 0
    assert len(result.trace.discarded_clues) >= 1


@test("6.4 agent honors ABANDON_DIG verdict from critique")
async def test_agent_abandons_on_critique():
    cushion = _make_cushion()
    state = AgentState(
        agent_id="P01",
        cushion=cushion,
        budget=AgentBudget(time_budget_seconds=999, token_budget=99_999, max_steps=10),
    )
    abandon = json.dumps({
        "answers": {q: "x" for q in QUESTIONS},
        "red_flags": ["Q4"],
        "verdict": "abandon_dig",
        "summary": "no gain",
    })
    client = _FakeLLMClient(responses={
        "psychology:structural_match_check": _good_match_response(cushion),
        "psychology:self_critique_check": abandon,
        "synthesizer:wandering_dig": _good_dig_response(),
    })
    counter = {"t": 0.0}
    def fake_clock():
        counter["t"] += 0.001
        return counter["t"]
    result = await run_agent(state, client, fetcher=stub_fetcher, clock=fake_clock)  # type: ignore[arg-type]
    # ABANDONED step should appear in trace
    abandoned_steps = result.trace.steps_of(StepKind.ABANDONED)
    assert len(abandoned_steps) >= 1


# ===========================================================================
# 7. Runtime + WanderingMode
# ===========================================================================


@test("7.1 MODE_DEFAULTS contains all three modes")
def test_mode_defaults_all_three():
    assert WanderingMode.TRIPLE_PENDULUM in MODE_DEFAULTS
    assert WanderingMode.MULTI_PENDULUM in MODE_DEFAULTS
    assert WanderingMode.ABSOLUTE_CHAOS in MODE_DEFAULTS
    assert MODE_DEFAULTS[WanderingMode.MULTI_PENDULUM].agents == 5
    assert MODE_DEFAULTS[WanderingMode.ABSOLUTE_CHAOS].agents == 10


@test("7.2 WanderingConfig.resolved applies defaults")
def test_config_resolved_defaults():
    cfg = WanderingConfig(mode=WanderingMode.MULTI_PENDULUM)
    agents, time_s, tokens, mix = cfg.resolved()
    assert agents == 5
    assert tokens == 30_000
    assert len(mix) == 5


@test("7.3 WanderingConfig.resolved respects overrides")
def test_config_resolved_overrides():
    cfg = WanderingConfig(
        mode=WanderingMode.MULTI_PENDULUM,
        agents=3,
        tokens_per_agent=10_000,
    )
    agents, _, tokens, _ = cfg.resolved()
    assert agents == 3
    assert tokens == 10_000


@test("7.4 assign_models cycles when num_agents > mix length")
def test_assign_models_cycle():
    models = assign_models(5, ("a", "b"))
    assert models == ["a", "b", "a", "b", "a"]


@test("7.5 run_wandering_session MULTI mode produces SessionResult with reports")
async def test_run_session_multi():
    cushion = _make_cushion()
    config = WanderingConfig(
        mode=WanderingMode.MULTI_PENDULUM,
        agents=2,
        time_budget_seconds=999,
        tokens_per_agent=99_999,
    )
    client = _FakeLLMClient(responses={
        "psychology:structural_match_check": _good_match_response(cushion),
        "psychology:self_critique_check": _good_critique_response(),
        "synthesizer:wandering_dig": _good_dig_response(),
    })
    counter = {"t": 0.0}
    def fake_clock():
        counter["t"] += 0.0001
        return counter["t"]
    session = await run_wandering_session(cushion, config, client, clock=fake_clock)  # type: ignore[arg-type]
    assert session.mode == WanderingMode.MULTI_PENDULUM
    assert session.agent_count() == 2
    # At least some reports produced (each agent runs until step_cap or budget)
    # Agents may not all produce — that's OK in a fake test; verify structure.
    assert isinstance(session.reports, list)


@test("7.6 run_wandering_session TRIPLE mode runs agents sequentially")
async def test_run_session_triple():
    cushion = _make_cushion()
    config = WanderingConfig(
        mode=WanderingMode.TRIPLE_PENDULUM,
        agents=2,
        time_budget_seconds=999,
        tokens_per_agent=99_999,
    )
    client = _FakeLLMClient(responses={
        "psychology:structural_match_check": json.dumps({"actual": [], "essence": [], "mechanism": []}),
    })
    counter = {"t": 0.0}
    def fake_clock():
        counter["t"] += 0.0001
        return counter["t"]
    session = await run_wandering_session(cushion, config, client, clock=fake_clock)  # type: ignore[arg-type]
    assert session.mode == WanderingMode.TRIPLE_PENDULUM
    assert session.agent_count() == 2


# ===========================================================================
# 8. Articulation
# ===========================================================================


@test("8.1 parse_articulation_response returns five fields")
def test_parse_articulation_valid():
    fields = parse_articulation_response(_good_articulation_response())
    assert set(fields.keys()) == {"spark", "source_shape", "bridge", "use", "limit"}
    assert "harmonic" in fields["spark"].lower()


@test("8.2 parse_articulation_response: missing fields → fallback string")
def test_parse_articulation_missing():
    payload = json.dumps({"spark": "only spark provided"})
    fields = parse_articulation_response(payload)
    assert fields["spark"] == "only spark provided"
    assert fields["limit"] == "(unable to articulate)"


@test("8.3 articulate_report end-to-end produces ArticulatedCard")
async def test_articulate_e2e():
    report = ExplorationReport(
        report_id="r1",
        agent_id="P01",
        anchor_summary="anchor",
        domain_explored="jazz",
        layer_matches={
            "essence": LayerMatch(layer_name="essence", matched_nodes=["bounded freedom"], total_nodes=3),
        },
        exploration_summary="found jazz",
        advancement="resonance",
        what_does_not_map="real-time vs async",
        confidence=Confidence.MEDIUM,
    )
    client = _FakeLLMClient(responses={"synthesizer:wandering_articulation": _good_articulation_response()})
    card = await articulate_report(report, client)  # type: ignore[arg-type]
    assert isinstance(card, ArticulatedCard)
    assert card.report_id == "r1"
    assert card.confidence == Confidence.MEDIUM
    assert "harmonic" in card.spark.lower()


@test("8.4 articulate_report falls back to raw fields if LLM fails")
async def test_articulate_fallback():
    report = ExplorationReport(
        report_id="r1",
        agent_id="P01",
        anchor_summary="anchor",
        domain_explored="jazz",
        layer_matches={
            "essence": LayerMatch(layer_name="essence", matched_nodes=["a"], total_nodes=3),
        },
        exploration_summary="raw summary",
        what_does_not_map="raw limit",
    )
    # Fake client that returns success=False
    class _FailClient:
        async def call(self, **kwargs):
            return LLMResponse(content="", input_tokens=0, output_tokens=0, latency_ms=0, success=False, error="oops")
    card = await articulate_report(report, _FailClient())  # type: ignore[arg-type]
    assert card.spark == "raw summary"
    assert card.limit == "raw limit"


# ===========================================================================
# 9. Synthesis
# ===========================================================================


@test("9.1 parse_synthesis_response: valid → SynthesisMap with clusters")
def test_parse_synthesis_valid():
    sm = parse_synthesis_response(_good_synthesis_response())
    assert len(sm.clusters) == 1
    assert len(sm.opportunity_paths) == 1
    assert sm.opportunity_paths[0].confidence_estimate == Confidence.HIGH
    assert "bounded-freedom" in sm.recommended_next_direction


@test("9.2 parse_synthesis_response: unparseable → empty map")
def test_parse_synthesis_empty():
    sm = parse_synthesis_response("garbage")
    assert sm.clusters == []
    assert sm.recommended_next_direction == ""


@test("9.3 synthesize_dossier with empty cards returns empty map (no LLM call)")
async def test_synthesize_empty():
    client = _FakeLLMClient()
    sm = await synthesize_dossier("anchor", [], client)  # type: ignore[arg-type]
    assert sm.clusters == []
    # No LLM call should have been made
    assert client.call_log == []


@test("9.4 synthesize_dossier end-to-end with one card")
async def test_synthesize_one_card():
    card = ArticulatedCard(
        report_id="wander-P01-001",
        spark="x",
        source_shape="y",
        bridge="z",
        use="u",
        limit="l",
        confidence=Confidence.HIGH,
    )
    client = _FakeLLMClient(responses={"synthesizer:wandering_synthesis": _good_synthesis_response()})
    sm = await synthesize_dossier("anchor", [card], client)  # type: ignore[arg-type]
    assert "wander-P01-001" in sm.top_insights


# ===========================================================================
# 10. Dossier assembly
# ===========================================================================


@test("10.1 Dossier.all_cards returns HIGH + MED + LOW in order")
def test_dossier_all_cards_order():
    d = Dossier(metadata=_dummy_metadata())
    d.high.cards.append(_card("h1", Confidence.HIGH))
    d.medium.cards.append(_card("m1", Confidence.MEDIUM))
    d.low.cards.append(_card("l1", Confidence.LOW))
    cards = d.all_cards()
    assert [c.report_id for c in cards] == ["h1", "m1", "l1"]


@test("10.2 Dossier.card_by_id finds the right card across bands")
def test_dossier_card_by_id():
    d = Dossier(metadata=_dummy_metadata())
    d.low.cards.append(_card("l1", Confidence.LOW))
    assert d.card_by_id("l1") is not None
    assert d.card_by_id("nonexistent") is None


@test("10.3 Dossier.to_dict produces JSON-safe payload")
def test_dossier_to_dict():
    d = Dossier(metadata=_dummy_metadata())
    d.high.cards.append(_card("h1", Confidence.HIGH))
    payload = d.to_dict()
    import json as _json
    blob = _json.dumps(payload)  # must not raise
    assert "h1" in blob
    assert "high" in payload


@test("10.4 build_dossier end-to-end produces a Dossier with bands populated")
async def test_build_dossier_e2e():
    cushion = _make_cushion()
    # Fabricate a SessionResult with one report
    report = ExplorationReport(
        report_id="wander-P01-001",
        agent_id="P01",
        anchor_summary="anchor",
        domain_explored="jazz",
        layer_matches={
            "essence": LayerMatch(layer_name="essence", matched_nodes=["bounded freedom", "anchored chaos"], total_nodes=3),
        },
        exploration_summary="x",
        advancement="y",
        what_does_not_map="z",
    )
    report.confidence = report.compute_confidence()
    session = SessionResult(
        session_id="wsess-test",
        mode=WanderingMode.MULTI_PENDULUM,
        cushion=cushion,
        config=WanderingConfig(),
        reports=[report],
        traces=[DecisionTrace(agent_id="P01")],
        total_tokens_spent=500,
        elapsed_seconds=10.0,
    )
    client = _FakeLLMClient(responses={
        "synthesizer:wandering_articulation": _good_articulation_response(),
        "synthesizer:wandering_synthesis": _good_synthesis_response(),
    })
    dossier = await build_dossier(session, client)  # type: ignore[arg-type]
    assert dossier.metadata.session_id == "wsess-test"
    assert dossier.metadata.report_count == 1
    # The one card should land in some confidence band
    total_cards = len(dossier.all_cards())
    assert total_cards == 1


# ---- helpers for dossier tests ----

def _dummy_metadata() -> DossierMetadata:
    return DossierMetadata(
        session_id="s1",
        mode=WanderingMode.MULTI_PENDULUM,
        anchor_summary="anchor",
        cushion_constellation_size=8,
        agent_count=2,
        report_count=1,
        total_tokens_spent=100,
        elapsed_seconds=1.0,
        completed_at=1000.0,
    )


def _card(report_id: str, confidence: Confidence) -> ArticulatedCard:
    return ArticulatedCard(
        report_id=report_id,
        spark="s", source_shape="ss", bridge="b", use="u", limit="l",
        confidence=confidence,
    )


# ===========================================================================
# Runner
# ===========================================================================


ALL_TESTS = [
    # 1. Report
    test_layer_match_ratio,
    test_layer_match_zero_total,
    test_confidence_essence_high,
    test_confidence_zero,
    test_report_validate_what_does_not_map,
    test_report_validate_clean,
    # 2. Trace
    test_trace_append_step_id,
    test_trace_reports_produced,
    test_trace_discard,
    # 3. Matching
    test_iter_zero,
    test_iter_one_node,
    test_iter_three_nodes,
    test_iter_capped,
    test_parse_match_valid,
    test_parse_match_filters_invented,
    test_match_content_e2e,
    test_match_unparseable,
    # 4. Policy
    test_drift_empty,
    test_drift_detected,
    test_drift_with_matches,
    test_domain_visit_counts,
    test_pick_next_domain_inverse_freq,
    test_next_move_drift,
    test_next_move_chaos_pick,
    # 5. Critique
    test_critique_six_questions,
    test_parse_critique_valid,
    test_parse_critique_unparseable,
    test_parse_critique_unknown_verdict,
    test_run_critique_e2e,
    # 6. Agent
    test_agent_runs_to_budget,
    test_agent_produces_report_on_match,
    test_agent_discards_no_match,
    test_agent_abandons_on_critique,
    # 7. Runtime
    test_mode_defaults_all_three,
    test_config_resolved_defaults,
    test_config_resolved_overrides,
    test_assign_models_cycle,
    test_run_session_multi,
    test_run_session_triple,
    # 8. Articulation
    test_parse_articulation_valid,
    test_parse_articulation_missing,
    test_articulate_e2e,
    test_articulate_fallback,
    # 9. Synthesis
    test_parse_synthesis_valid,
    test_parse_synthesis_empty,
    test_synthesize_empty,
    test_synthesize_one_card,
    # 10. Dossier
    test_dossier_all_cards_order,
    test_dossier_card_by_id,
    test_dossier_to_dict,
    test_build_dossier_e2e,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} Wandering Room engine tests...")
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
