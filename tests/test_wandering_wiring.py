"""
Wiring tests — sub-agent spawning, persistence, fetcher adapter, API routes.

Covers everything added beyond the core engine: subagent.py, fetcher.py,
memory_enrichment.py, persistence.py, routes.py.

Run: PYTHONPATH=. python3 tests/test_wandering_wiring.py
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
from src.wandering.runtime import (
    SessionResult,
    WanderingConfig,
    WanderingMode,
)
from src.wandering.subagent import (
    MAX_CHAIN_DEPTH,
    SpawnRequest,
    should_spawn,
    spawn_request_from_high_match_report,
    spawn_request_from_user_dig_deeper,
)
from src.wandering.persistence import (
    InMemoryWanderingStore,
    build_wandering_store_from_env,
    session_to_json,
)
from src.wandering.fetcher import (
    _build_query_for_domain,
    _stitch_from_hit,
)
from src.wandering.agent import AgentState, AgentBudget
from src.bridge.web_search import SearchHit, SearchResult


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


def _make_cushion() -> CushionGraph:
    return CushionGraph(
        actual=CushionLayer(name="actual", nodes=["wandering agents", "research anchor", "credit budget"]),
        essence=CushionLayer(name="essence", nodes=["bounded freedom", "productive constraint", "anchored chaos"]),
        mechanism=CushionLayer(name="mechanism", nodes=["soft constraint enables emergence", "hard constraint kills value"]),
        raw_input=CushionInput(
            problem=CushionField(name="problem", content="how to control wandering agents"),
            context=CushionField(name="context"),
            vision=CushionField(name="vision"),
            current_map=CushionField(name="current_map"),
        ),
    )


def _make_report(report_id: str, confidence: Confidence, essence_match: int = 2) -> ExplorationReport:
    r = ExplorationReport(
        report_id=report_id,
        agent_id="P01",
        anchor_summary="anchor",
        domain_explored="jazz",
        layer_matches={
            "actual": LayerMatch(layer_name="actual", matched_nodes=[], total_nodes=3),
            "essence": LayerMatch(
                layer_name="essence",
                matched_nodes=["bounded freedom"][:essence_match] if essence_match else [],
                total_nodes=3,
            ),
            "mechanism": LayerMatch(
                layer_name="mechanism",
                matched_nodes=["soft constraint enables emergence"][:1] if essence_match else [],
                total_nodes=2,
            ),
        },
        confidence=confidence,
        exploration_summary="summary",
        what_does_not_map="limit",
    )
    return r


# ===========================================================================
# 1. SpawnRequest validation (should_spawn)
# ===========================================================================


@test("1.1 should_spawn rejects exceeded chain depth for triple_pendulum")
def test_should_spawn_depth_triple():
    req = SpawnRequest(
        parent_agent_id="P01",
        cushion=_make_cushion(),
        focus_area="something",
        distance_budget_tokens=5000,
        chain_depth=5,  # > MAX_CHAIN_DEPTH["triple_pendulum"] (=3)
        mode_key="triple_pendulum",
    )
    allowed, reason = should_spawn(req, session_tokens_spent=0, session_token_cap=100_000)
    assert allowed is False
    assert "chain_depth" in reason


@test("1.2 should_spawn rejects budget overflow")
def test_should_spawn_budget_overflow():
    req = SpawnRequest(
        parent_agent_id="P01",
        cushion=_make_cushion(),
        focus_area="something",
        distance_budget_tokens=50_000,
        chain_depth=2,
        mode_key="absolute_chaos",
    )
    allowed, reason = should_spawn(req, session_tokens_spent=90_000, session_token_cap=100_000)
    assert allowed is False
    assert "token_cap" in reason


@test("1.3 should_spawn rejects zero budget")
def test_should_spawn_zero_budget():
    req = SpawnRequest(
        parent_agent_id="P01",
        cushion=_make_cushion(),
        focus_area="something",
        distance_budget_tokens=0,
        chain_depth=2,
        mode_key="absolute_chaos",
    )
    allowed, reason = should_spawn(req, session_tokens_spent=0, session_token_cap=100_000)
    assert allowed is False
    assert "zero" in reason.lower()


@test("1.4 should_spawn rejects empty focus_area")
def test_should_spawn_empty_focus():
    req = SpawnRequest(
        parent_agent_id="P01",
        cushion=_make_cushion(),
        focus_area="   ",
        distance_budget_tokens=5000,
        chain_depth=2,
        mode_key="absolute_chaos",
    )
    allowed, reason = should_spawn(req, session_tokens_spent=0, session_token_cap=100_000)
    assert allowed is False
    assert "focus" in reason.lower()


@test("1.5 should_spawn allows valid request")
def test_should_spawn_allowed():
    req = SpawnRequest(
        parent_agent_id="P01",
        cushion=_make_cushion(),
        focus_area="explore further",
        distance_budget_tokens=5000,
        chain_depth=2,
        mode_key="absolute_chaos",
    )
    allowed, _ = should_spawn(req, session_tokens_spent=0, session_token_cap=100_000)
    assert allowed is True


# ===========================================================================
# 2. SpawnRequest builders
# ===========================================================================


@test("2.1 spawn_request_from_high_match_report returns None for non-HIGH")
def test_spawn_builder_skips_non_high():
    state = AgentState(agent_id="P01", cushion=_make_cushion(), budget=AgentBudget())
    medium_report = _make_report("r1", Confidence.MEDIUM)
    req = spawn_request_from_high_match_report(
        parent_state=state, report=medium_report,
        mode_key="absolute_chaos", chain_depth=2,
    )
    assert req is None


@test("2.2 spawn_request_from_high_match_report builds focus from matched essence + mechanism")
def test_spawn_builder_high():
    state = AgentState(agent_id="P01", cushion=_make_cushion(), budget=AgentBudget())
    high_report = _make_report("r1", Confidence.HIGH)
    req = spawn_request_from_high_match_report(
        parent_state=state, report=high_report,
        mode_key="absolute_chaos", chain_depth=2,
    )
    assert req is not None
    assert req.chain_depth == 2
    assert "bounded freedom" in req.focus_area
    assert req.parent_agent_id == "P01"


@test("2.3 spawn_request_from_high_match_report returns None if no essence/mech match")
def test_spawn_builder_no_structural_match():
    state = AgentState(agent_id="P01", cushion=_make_cushion(), budget=AgentBudget())
    surface_only = _make_report("r1", Confidence.HIGH, essence_match=0)
    # Override essence/mech to have NO matched nodes
    surface_only.layer_matches["essence"].matched_nodes = []
    surface_only.layer_matches["mechanism"].matched_nodes = []
    req = spawn_request_from_high_match_report(
        parent_state=state, report=surface_only,
        mode_key="absolute_chaos", chain_depth=2,
    )
    assert req is None  # HIGH on actual alone doesn't earn auto-spawn


@test("2.4 spawn_request_from_user_dig_deeper uses user focus if provided")
def test_spawn_user_focus():
    cushion = _make_cushion()
    report = _make_report("r1", Confidence.LOW)
    req = spawn_request_from_user_dig_deeper(
        cushion=cushion,
        report=report,
        user_request_text="look at music theory examples",
    )
    assert req.focus_area == "look at music theory examples"
    assert req.chain_depth == 2


@test("2.5 spawn_request_from_user_dig_deeper derives focus if user text empty")
def test_spawn_user_derived_focus():
    cushion = _make_cushion()
    report = _make_report("r1", Confidence.MEDIUM, essence_match=1)
    req = spawn_request_from_user_dig_deeper(
        cushion=cushion,
        report=report,
        user_request_text="",
    )
    assert "r1" in req.focus_area
    assert "essence" in req.focus_area.lower()


# ===========================================================================
# 3. Fetcher adapter
# ===========================================================================


@test("3.1 _build_query_for_domain combines anchor + domain hint")
def test_query_build():
    q = _build_query_for_domain("jazz", "controlling wandering agents")
    assert "controlling wandering agents" in q
    assert "jazz" in q


@test("3.2 _build_query_for_domain handles empty domain")
def test_query_no_domain():
    q = _build_query_for_domain("", "anchor")
    assert q == "anchor"


@test("3.3 _stitch_from_hit returns explanatory body when no hits")
def test_stitch_no_hits():
    result = SearchResult(query="test", provider="test")
    title, url, body = _stitch_from_hit(result, 0)
    assert "no results" in body.lower()
    assert url == ""


@test("3.4 _stitch_from_hit combines multiple hits with separator")
def test_stitch_multiple_hits():
    result = SearchResult(
        query="jazz",
        provider="tavily",
        hits=[
            SearchHit(title="A", url="http://a.com", snippet="snippet A"),
            SearchHit(title="B", url="http://b.com", snippet="snippet B"),
        ],
    )
    title, url, body = _stitch_from_hit(result, 0)
    assert title == "A"
    assert url == "http://a.com"
    assert "snippet A" in body
    assert "snippet B" in body
    assert "---" in body  # separator


# ===========================================================================
# 4. In-memory persistence
# ===========================================================================


@test("4.1 InMemoryWanderingStore.save_session + list_sessions round-trips")
async def test_inmemory_save_list():
    store = InMemoryWanderingStore()
    cushion = _make_cushion()
    session = SessionResult(
        session_id="wsess-1",
        mode=WanderingMode.MULTI_PENDULUM,
        cushion=cushion,
        config=WanderingConfig(),
    )
    assert await store.save_session("user-A", session) is True
    sessions = await store.list_sessions("user-A")
    assert "wsess-1" in sessions


@test("4.2 InMemoryWanderingStore.get_session returns saved session")
async def test_inmemory_get_session():
    store = InMemoryWanderingStore()
    session = SessionResult(
        session_id="wsess-2",
        mode=WanderingMode.TRIPLE_PENDULUM,
        cushion=_make_cushion(),
        config=WanderingConfig(),
    )
    await store.save_session("user-A", session)
    fetched = await store.get_session("wsess-2")
    assert fetched is not None
    assert fetched.session_id == "wsess-2"


@test("4.3 save_session harvests discarded clues into user's bucket")
async def test_inmemory_discards_harvested():
    store = InMemoryWanderingStore()
    cushion = _make_cushion()
    trace = DecisionTrace(agent_id="P01")
    trace.discard(DiscardedClue(
        description="ancient pottery analogy",
        classification=DiscardKind.POSSIBLY_RELEVANT_ELSEWHERE,
    ))
    session = SessionResult(
        session_id="wsess-3",
        mode=WanderingMode.MULTI_PENDULUM,
        cushion=cushion,
        config=WanderingConfig(),
        traces=[trace],
    )
    await store.save_session("user-A", session)
    clues = await store.list_discarded_clues_for_user("user-A")
    assert len(clues) == 1
    assert clues[0].description == "ancient pottery analogy"


@test("4.4 list_discarded_clues filters by classification")
async def test_inmemory_filter_classification():
    store = InMemoryWanderingStore()
    trace = DecisionTrace(agent_id="P01")
    trace.discard(DiscardedClue(
        description="x",
        classification=DiscardKind.DISCARDED_FOR_CURRENT_ANCHOR,
    ))
    trace.discard(DiscardedClue(
        description="y",
        classification=DiscardKind.REVISIT_LATER,
    ))
    session = SessionResult(
        session_id="wsess-4",
        mode=WanderingMode.MULTI_PENDULUM,
        cushion=_make_cushion(),
        config=WanderingConfig(),
        traces=[trace],
    )
    await store.save_session("user-A", session)
    # Default kinds filter excludes DISCARDED_FOR_CURRENT_ANCHOR
    clues = await store.list_discarded_clues_for_user("user-A")
    assert len(clues) == 1
    assert clues[0].description == "y"


@test("4.5 list_discarded_clues respects limit")
async def test_inmemory_discards_limit():
    store = InMemoryWanderingStore()
    trace = DecisionTrace(agent_id="P01")
    for i in range(10):
        trace.discard(DiscardedClue(
            description=f"clue {i}",
            classification=DiscardKind.REVISIT_LATER,
        ))
    session = SessionResult(
        session_id="wsess-5",
        mode=WanderingMode.MULTI_PENDULUM,
        cushion=_make_cushion(),
        config=WanderingConfig(),
        traces=[trace],
    )
    await store.save_session("user-A", session)
    clues = await store.list_discarded_clues_for_user("user-A", limit=3)
    assert len(clues) == 3


@test("4.6 save_session unknown user (None) doesn't crash")
async def test_inmemory_no_user():
    store = InMemoryWanderingStore()
    session = SessionResult(
        session_id="wsess-anon",
        mode=WanderingMode.MULTI_PENDULUM,
        cushion=_make_cushion(),
        config=WanderingConfig(),
    )
    assert await store.save_session(None, session) is True


# ===========================================================================
# 5. session_to_json round-trip
# ===========================================================================


@test("5.1 session_to_json produces parseable JSON with all keys")
def test_session_to_json():
    session = SessionResult(
        session_id="wsess-x",
        mode=WanderingMode.ABSOLUTE_CHAOS,
        cushion=_make_cushion(),
        config=WanderingConfig(),
        reports=[_make_report("r1", Confidence.HIGH)],
        traces=[DecisionTrace(agent_id="P01")],
        total_tokens_spent=1234,
        elapsed_seconds=15.5,
    )
    raw = session_to_json(session)
    parsed = json.loads(raw)
    assert parsed["session_id"] == "wsess-x"
    assert parsed["mode"] == "absolute_chaos"
    assert parsed["total_tokens_spent"] == 1234
    assert len(parsed["reports"]) == 1
    assert parsed["reports"][0]["report_id"] == "r1"


# ===========================================================================
# 6. build_wandering_store_from_env fallback
# ===========================================================================


@test("6.1 build_wandering_store_from_env without CONSTELLAX_DB_BACKEND → InMemory")
def test_build_store_fallback():
    import os
    old = os.environ.pop("CONSTELLAX_DB_BACKEND", None)
    try:
        store = build_wandering_store_from_env()
        assert isinstance(store, InMemoryWanderingStore)
    finally:
        if old is not None:
            os.environ["CONSTELLAX_DB_BACKEND"] = old


@test("6.2 build_wandering_store_from_env never raises on missing config")
def test_build_store_safe():
    # Even with backend=neo4j but no URI, must fall back to in-memory, not raise.
    import os
    old_be = os.environ.get("CONSTELLAX_DB_BACKEND")
    old_uri = os.environ.pop("NEO4J_URI", None)
    os.environ["CONSTELLAX_DB_BACKEND"] = "neo4j"
    try:
        store = build_wandering_store_from_env()
        # In most CI/dev envs this should fall back gracefully.
        # It should NEVER raise.
        assert store is not None
    finally:
        if old_be is not None:
            os.environ["CONSTELLAX_DB_BACKEND"] = old_be
        else:
            os.environ.pop("CONSTELLAX_DB_BACKEND", None)
        if old_uri is not None:
            os.environ["NEO4J_URI"] = old_uri


# ===========================================================================
# 7. Memory enrichment (defensive paths)
# ===========================================================================


@test("7.1 fetch_memory_enrichment_real returns empty when no user_id")
async def test_memory_enrichment_no_user():
    from src.wandering.memory_enrichment import fetch_memory_enrichment_real
    out = await fetch_memory_enrichment_real(None)
    assert out == ""


@test("7.2 fetch_memory_enrichment_real returns empty when store unavailable")
async def test_memory_enrichment_no_store():
    from src.wandering.memory_enrichment import fetch_memory_enrichment_real
    # No Neo4j env vars set in test env → store builds as InMemory which
    # is empty for a fresh user. Result: empty enrichment.
    out = await fetch_memory_enrichment_real("usr-test-no-data")
    assert out == ""


@test("7.3 enrich_cushion_input does not overwrite existing enrichment")
async def test_enrich_respects_existing():
    from src.wandering.memory_enrichment import enrich_cushion_input
    inp = CushionInput(
        problem=CushionField(name="problem", content="x"),
        context=CushionField(name="context"),
        vision=CushionField(name="vision"),
        current_map=CushionField(name="current_map"),
        memory_enrichment="pre-existing context",
    )
    await enrich_cushion_input(inp, "user-A")
    assert inp.memory_enrichment == "pre-existing context"


# ===========================================================================
# 8. Routes — import + register
# ===========================================================================


@test("8.1 routes.get_router returns a FastAPI APIRouter with 9 routes")
def test_router_shape():
    from src.wandering.routes import get_router
    router = get_router()
    # APIRouter exposes routes via .routes
    paths = [r.path for r in router.routes if hasattr(r, "path")]
    assert "/brief" in paths
    assert "/session" in paths
    assert "/session/{session_id}" in paths
    assert "/session/{session_id}/dig-deeper" in paths
    assert "/session/{session_id}/memo" in paths
    assert "/session/{session_id}/status" in paths
    assert "/session/{session_id}/abort" in paths
    assert "/session/{session_id}/continue" in paths
    assert "/sessions" in paths


@test("8.2 router mounted on server.py exposes all /api/v2/wandering paths")
def test_server_mount():
    # Lazy import to avoid module init in other tests
    import server
    server_paths = [r.path for r in server.app.routes if hasattr(r, "path")]
    assert "/api/v2/wandering/brief" in server_paths
    assert "/api/v2/wandering/session" in server_paths
    assert "/api/v2/wandering/session/{session_id}" in server_paths
    assert "/api/v2/wandering/session/{session_id}/dig-deeper" in server_paths
    assert "/api/v2/wandering/session/{session_id}/memo" in server_paths
    assert "/api/v2/wandering/session/{session_id}/status" in server_paths
    assert "/api/v2/wandering/session/{session_id}/abort" in server_paths
    assert "/api/v2/wandering/session/{session_id}/continue" in server_paths
    assert "/api/v2/wandering/sessions" in server_paths


# ===========================================================================
# Runner
# ===========================================================================


ALL_TESTS = [
    # 1. should_spawn
    test_should_spawn_depth_triple,
    test_should_spawn_budget_overflow,
    test_should_spawn_zero_budget,
    test_should_spawn_empty_focus,
    test_should_spawn_allowed,
    # 2. Spawn builders
    test_spawn_builder_skips_non_high,
    test_spawn_builder_high,
    test_spawn_builder_no_structural_match,
    test_spawn_user_focus,
    test_spawn_user_derived_focus,
    # 3. Fetcher adapter
    test_query_build,
    test_query_no_domain,
    test_stitch_no_hits,
    test_stitch_multiple_hits,
    # 4. In-memory persistence
    test_inmemory_save_list,
    test_inmemory_get_session,
    test_inmemory_discards_harvested,
    test_inmemory_filter_classification,
    test_inmemory_discards_limit,
    test_inmemory_no_user,
    # 5. session_to_json
    test_session_to_json,
    # 6. build_wandering_store_from_env
    test_build_store_fallback,
    test_build_store_safe,
    # 7. Memory enrichment
    test_memory_enrichment_no_user,
    test_memory_enrichment_no_store,
    test_enrich_respects_existing,
    # 8. Routes
    test_router_shape,
    test_server_mount,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} Wandering Room wiring tests...")
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
