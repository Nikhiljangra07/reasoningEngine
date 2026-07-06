"""
Tests for src/wandering/map_adapter.py — Dossier → Map Room Memo.

Pure-data adapter; no LLM calls, no network. Tests build a fixture
Dossier directly (skipping build_dossier's LLM path) and assert the
resulting wire-shape dict has the expected structure.

Run: PYTHONPATH=. python3 tests/test_wandering_map_adapter.py
"""

from __future__ import annotations

from src.wandering.articulate import ArticulatedCard
from src.wandering.cushion import (
    CushionField,
    CushionGraph,
    CushionInput,
    CushionLayer,
)
from src.wandering.dossier import ConfidenceBand, Dossier, DossierMetadata
from src.wandering.map_adapter import session_to_memo
from src.wandering.report import Confidence
from src.wandering.runtime import SessionResult, WanderingConfig, WanderingMode
from src.wandering.synthesis import (
    Contradiction,
    InsightCluster,
    OpportunityPath,
    SynthesisMap,
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
        fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ===========================================================================
# Fixture builders
# ===========================================================================


def _make_cushion(problem: str = "build a sketchbook tool for solo designers") -> CushionGraph:
    return CushionGraph(
        actual=CushionLayer(
            name="actual",
            nodes=["sketchbook", "solo designers", "affordance theory"],
            summary="solo designers extending Norman's affordance theory",
        ),
        essence=CushionLayer(
            name="essence",
            nodes=["affordance", "perceived action", "embodied tool"],
            summary="how tools invite specific kinds of use",
        ),
        mechanism=CushionLayer(
            name="mechanism",
            nodes=["physical affordance shapes mental affordance"],
            summary="materiality drives perception",
        ),
        raw_input=CushionInput(
            problem=CushionField(name="problem", content=problem),
            context=CushionField(name="context"),
            vision=CushionField(name="vision"),
            hunches=CushionField(name="hunches"),
        ),
    )


def _make_card(
    report_id: str,
    spark: str,
    confidence: Confidence = Confidence.MEDIUM,
    domain: str = "design",
    match_strength: float = 0.55,
) -> ArticulatedCard:
    return ArticulatedCard(
        report_id=report_id,
        spark=spark,
        source_shape="some source shape text",
        bridge="some bridge text",
        use="some use text",
        limit="some limit text",
        confidence=confidence,
        confidence_detail="act:1/3 ess:2/3 mec:0/1",
        agent_id=f"agent-{report_id}",
        domain=domain,
        citations=[],
        match_strength=match_strength,
    )


def _make_session(cushion: CushionGraph) -> SessionResult:
    config = WanderingConfig(mode=WanderingMode.MULTI_PENDULUM, session_id="wsess-test")
    return SessionResult(
        session_id="wsess-test",
        mode=WanderingMode.MULTI_PENDULUM,
        cushion=cushion,
        config=config,
        reports=[],
        traces=[],
        total_tokens_spent=12_345,
        elapsed_seconds=240.0,
        ended_at=0.0,
    )


def _make_dossier(
    cards_high: list[ArticulatedCard],
    cards_medium: list[ArticulatedCard],
    cards_low: list[ArticulatedCard],
    synthesis: SynthesisMap,
) -> Dossier:
    metadata = DossierMetadata(
        session_id="wsess-test",
        mode=WanderingMode.MULTI_PENDULUM,
        anchor_summary="anchor",
        cushion_constellation_size=10,
        agent_count=3,
        report_count=len(cards_high) + len(cards_medium) + len(cards_low),
        total_tokens_spent=12_345,
        elapsed_seconds=240.0,
        completed_at=1.0,
    )
    high = ConfidenceBand(Confidence.HIGH)
    high.cards = cards_high
    medium = ConfidenceBand(Confidence.MEDIUM)
    medium.cards = cards_medium
    low = ConfidenceBand(Confidence.LOW)
    low.cards = cards_low
    return Dossier(
        metadata=metadata,
        high=high,
        medium=medium,
        low=low,
        synthesis=synthesis,
    )


# ===========================================================================
# Tests
# ===========================================================================


@test("memo: top-level fields all present")
def t_top_level_fields():
    cushion = _make_cushion()
    session = _make_session(cushion)
    card = _make_card("r1", "spark A")
    dossier = _make_dossier([], [card], [], SynthesisMap())
    memo = session_to_memo(session, dossier)
    for key in (
        "verdict_line", "verdict_body", "confidence", "reasoning",
        "alternatives", "falsifiers", "open_questions", "visuals",
    ):
        assert key in memo, f"missing top-level field: {key}"
    assert memo["alternatives"] == []
    assert memo["falsifiers"] == []
    assert isinstance(memo["reasoning"], list)
    assert isinstance(memo["visuals"], list)


@test("memo: visual is knowledge-graph with pursuit node first")
def t_visual_shape():
    cushion = _make_cushion("the pursuit text")
    session = _make_session(cushion)
    card = _make_card("r1", "spark A", domain="biology")
    dossier = _make_dossier([], [card], [], SynthesisMap())
    memo = session_to_memo(session, dossier)
    assert len(memo["visuals"]) == 1
    vis = memo["visuals"][0]
    assert vis["type"] == "knowledge-graph"
    assert vis["layout"] == "cose"
    assert vis["title"] == "Dossier map"
    nodes = vis["nodes"]
    assert nodes[0]["id"] == "pursuit"
    assert nodes[0]["kind"] == "decision"
    assert "pursuit text" in nodes[0]["label"]


@test("memo: every card becomes a node + edge from pursuit")
def t_cards_to_nodes():
    cushion = _make_cushion()
    session = _make_session(cushion)
    cards = [
        _make_card("r-a", "spark A", domain="biology", match_strength=0.81),
        _make_card("r-b", "spark B", domain="music",   match_strength=0.42),
        _make_card("r-c", "spark C", domain="finance", match_strength=0.18, confidence=Confidence.LOW),
    ]
    dossier = _make_dossier([cards[0]], [cards[1]], [cards[2]], SynthesisMap())
    memo = session_to_memo(session, dossier)
    vis = memo["visuals"][0]
    node_ids = {n["id"] for n in vis["nodes"]}
    assert {"pursuit", "r-a", "r-b", "r-c"} <= node_ids
    # Every card has an edge FROM pursuit.
    pursuit_targets = {e["target"] for e in vis["edges"] if e["source"] == "pursuit"}
    assert {"r-a", "r-b", "r-c"} <= pursuit_targets
    # Match strength surfaces as a percentage on the edge label.
    edges_by_target = {e["target"]: e for e in vis["edges"] if e["source"] == "pursuit"}
    assert edges_by_target["r-a"]["label"] == "81%"
    assert edges_by_target["r-b"]["label"] == "42%"
    assert edges_by_target["r-c"]["label"] == "18%"
    # Domain rides the edge relation.
    assert edges_by_target["r-a"]["relation"] == "biology"
    # LOW confidence cards render as 'claim' kind; non-LOW as 'concept'.
    cards_by_id = {n["id"]: n for n in vis["nodes"]}
    assert cards_by_id["r-a"]["kind"] == "concept"  # HIGH
    assert cards_by_id["r-b"]["kind"] == "concept"  # MEDIUM
    assert cards_by_id["r-c"]["kind"] == "claim"    # LOW


@test("memo: clusters become concept nodes with contains edges")
def t_clusters():
    cushion = _make_cushion()
    session = _make_session(cushion)
    cards = [
        _make_card("r-a", "spark A"),
        _make_card("r-b", "spark B"),
    ]
    cluster = InsightCluster(
        label="Shared affordance pattern",
        card_ids=["r-a", "r-b", "r-missing"],
        summary="they all share X",
    )
    dossier = _make_dossier([], cards, [], SynthesisMap(clusters=[cluster]))
    memo = session_to_memo(session, dossier)
    vis = memo["visuals"][0]
    node_ids = {n["id"] for n in vis["nodes"]}
    assert "cluster-1" in node_ids
    # contains edges only for cards that EXIST (dangling card_ids are skipped).
    contains_edges = [e for e in vis["edges"] if e["relation"] == "contains" and e["source"] == "cluster-1"]
    contains_targets = {e["target"] for e in contains_edges}
    assert contains_targets == {"r-a", "r-b"}
    # cluster node carries concept kind.
    cluster_node = next(n for n in vis["nodes"] if n["id"] == "cluster-1")
    assert cluster_node["kind"] == "concept"


@test("memo: contradictions become tension edges between TWO existing cards")
def t_contradictions():
    cushion = _make_cushion()
    session = _make_session(cushion)
    cards = [
        _make_card("r-a", "spark A"),
        _make_card("r-b", "spark B"),
        _make_card("r-c", "spark C"),
    ]
    contradictions = [
        # Valid: both cards exist.
        Contradiction(description="A vs B disagree on X", card_ids=["r-a", "r-b"]),
        # Invalid: only one card exists — must NOT emit a dangling edge.
        Contradiction(description="A vs ghost", card_ids=["r-a", "r-ghost"]),
        # Invalid: only one card_id total.
        Contradiction(description="solo", card_ids=["r-c"]),
    ]
    dossier = _make_dossier([], cards, [], SynthesisMap(contradictions=contradictions))
    memo = session_to_memo(session, dossier)
    vis = memo["visuals"][0]
    tension_edges = [e for e in vis["edges"] if e["relation"] == "contradicts"]
    assert len(tension_edges) == 1
    edge = tension_edges[0]
    assert {edge["source"], edge["target"]} == {"r-a", "r-b"}
    assert edge["label"] == "tension"
    # Contradictions do NOT create new nodes.
    node_ids = {n["id"] for n in vis["nodes"]}
    assert "r-ghost" not in node_ids


@test("memo: opportunity_paths become outcome nodes with supports edges")
def t_opportunity_paths():
    cushion = _make_cushion()
    session = _make_session(cushion)
    cards = [_make_card("r-a", "spark A"), _make_card("r-b", "spark B")]
    paths = [
        OpportunityPath(
            description="Path 1: build minimum viable prototype",
            supporting_card_ids=["r-a", "r-b"],
        ),
        OpportunityPath(
            description="Path 2: dangling reference",
            supporting_card_ids=["r-missing"],
        ),
    ]
    dossier = _make_dossier([], cards, [], SynthesisMap(opportunity_paths=paths))
    memo = session_to_memo(session, dossier)
    vis = memo["visuals"][0]
    node_ids = {n["id"]: n for n in vis["nodes"]}
    assert "path-1" in node_ids
    assert "path-2" in node_ids
    assert node_ids["path-1"]["kind"] == "outcome"
    # Path 1: both cards exist → two supports edges.
    p1_supports = [e for e in vis["edges"] if e["target"] == "path-1" and e["relation"] == "supports"]
    assert {e["source"] for e in p1_supports} == {"r-a", "r-b"}
    # Path 2: dangling card_id → zero edges.
    p2_supports = [e for e in vis["edges"] if e["target"] == "path-2"]
    assert p2_supports == []


@test("memo: top_insights with valid card_ids resolve to spark + domain")
def t_top_insights_resolve():
    cushion = _make_cushion()
    session = _make_session(cushion)
    cards = [
        _make_card("r-a", "spark for A", domain="biology"),
        _make_card("r-b", "spark for B", domain="music"),
    ]
    synthesis = SynthesisMap(top_insights=["r-a", "r-b", "freeform insight text", "r-missing"])
    dossier = _make_dossier([], cards, [], synthesis)
    memo = session_to_memo(session, dossier)
    reasoning = memo["reasoning"]
    # Order preserved; card_ids resolved; freeform passed through as body.
    assert reasoning[0]["title"] == "biology"
    assert reasoning[0]["body"]  == "spark for A"
    assert reasoning[1]["title"] == "music"
    assert reasoning[1]["body"]  == "spark for B"
    assert reasoning[2]["title"] == ""
    assert reasoning[2]["body"]  == "freeform insight text"
    # Missing card_id passes through verbatim — caller doesn't lose info.
    assert reasoning[3]["title"] == ""
    assert reasoning[3]["body"]  == "r-missing"


@test("memo: confidence aggregates from card distribution")
def t_confidence_aggregation():
    cushion = _make_cushion()
    session = _make_session(cushion)

    # All-HIGH → high.
    h = _make_dossier([_make_card("a", "")], [], [], SynthesisMap())
    assert session_to_memo(session, h)["confidence"] == "high"

    # Mostly MEDIUM → moderate.
    m = _make_dossier(
        [],
        [_make_card("a", ""), _make_card("b", "")],
        [_make_card("c", "", confidence=Confidence.LOW)],
        SynthesisMap(),
    )
    assert session_to_memo(session, m)["confidence"] == "moderate"

    # Mostly LOW → low.
    low = _make_dossier(
        [],
        [],
        [_make_card("a", "", confidence=Confidence.LOW),
         _make_card("b", "", confidence=Confidence.LOW)],
        SynthesisMap(),
    )
    assert session_to_memo(session, low)["confidence"] == "low"

    # Empty → low (honest).
    e = _make_dossier([], [], [], SynthesisMap())
    assert session_to_memo(session, e)["confidence"] == "low"


@test("memo: verdict_line falls back when synthesis has no recommendation")
def t_verdict_line_fallback():
    cushion = _make_cushion()
    session = _make_session(cushion)
    dossier = _make_dossier([], [_make_card("a", "")], [], SynthesisMap())
    memo = session_to_memo(session, dossier)
    assert "no single verdict" in memo["verdict_line"].lower()


@test("memo: verdict_line uses recommended_next_direction when set")
def t_verdict_line_from_synthesis():
    cushion = _make_cushion()
    session = _make_session(cushion)
    syn = SynthesisMap(recommended_next_direction="Prototype the sketchbook tool.")
    dossier = _make_dossier([], [_make_card("a", "")], [], syn)
    memo = session_to_memo(session, dossier)
    assert memo["verdict_line"] == "Prototype the sketchbook tool."


@test("memo: verdict_body summarises counts")
def t_verdict_body_counts():
    cushion = _make_cushion()
    session = _make_session(cushion)
    dossier = _make_dossier(
        [_make_card("a", "")],
        [_make_card("b", ""), _make_card("c", "")],
        [_make_card("d", "", confidence=Confidence.LOW)],
        SynthesisMap(
            clusters=[InsightCluster(label="C1", card_ids=["a", "b"])],
            contradictions=[Contradiction(description="x", card_ids=["a", "b"])],
        ),
    )
    body = session_to_memo(session, dossier)["verdict_body"]
    assert "4 partial matches" in body
    assert "1 low" in body and "2 medium" in body and "1 high" in body
    assert "1 cluster" in body
    assert "1 contradiction" in body


@test("memo: open_questions are passed through as Q&A items")
def t_open_questions():
    cushion = _make_cushion()
    session = _make_session(cushion)
    syn = SynthesisMap(
        open_questions=["What scale do you want?", "Who is the user?", ""],
    )
    dossier = _make_dossier([], [_make_card("a", "")], [], syn)
    qs = session_to_memo(session, dossier)["open_questions"]
    # Empty strings dropped; valid ones kept.
    assert len(qs) == 2
    assert qs[0] == {"question": "What scale do you want?", "answer": ""}


@test("memo: degenerate dossier (zero cards) emits pursuit-only graph")
def t_zero_cards():
    cushion = _make_cushion()
    session = _make_session(cushion)
    dossier = _make_dossier([], [], [], SynthesisMap())
    memo = session_to_memo(session, dossier)
    vis = memo["visuals"][0]
    # Just the pursuit node, no edges. Frontend's <2-node gate will skip
    # rendering the graph; the memo body still surfaces in the right column.
    assert len(vis["nodes"]) == 1
    assert vis["nodes"][0]["id"] == "pursuit"
    assert vis["edges"] == []


@test("memo: card labels are truncated to keep the graph readable")
def t_label_truncation():
    cushion = _make_cushion()
    session = _make_session(cushion)
    long_spark = "x" * 200
    card = _make_card("r-a", long_spark)
    dossier = _make_dossier([], [card], [], SynthesisMap())
    memo = session_to_memo(session, dossier)
    nodes = memo["visuals"][0]["nodes"]
    card_node = next(n for n in nodes if n["id"] == "r-a")
    # Truncated to <= 80 chars (with ellipsis when cut).
    assert len(card_node["label"]) <= 80
    assert card_node["label"].endswith("…")


@test("memo: match_strength edge label clamps to [0, 100]%")
def t_match_strength_clamp():
    cushion = _make_cushion()
    session = _make_session(cushion)
    cards = [
        _make_card("over", "spark", match_strength=1.5),
        _make_card("under", "spark", match_strength=-0.3),
        _make_card("nan",   "spark", match_strength=float("nan")),
    ]
    dossier = _make_dossier([], cards, [], SynthesisMap())
    memo = session_to_memo(session, dossier)
    edges_by_target = {e["target"]: e for e in memo["visuals"][0]["edges"]}
    assert edges_by_target["over"]["label"]  == "100%"
    assert edges_by_target["under"]["label"] == "0%"
    assert edges_by_target["nan"]["label"]   == "0%"


# ===========================================================================
# Runner
# ===========================================================================


if __name__ == "__main__":
    tests = [
        t_top_level_fields,
        t_visual_shape,
        t_cards_to_nodes,
        t_clusters,
        t_contradictions,
        t_opportunity_paths,
        t_top_insights_resolve,
        t_confidence_aggregation,
        t_verdict_line_fallback,
        t_verdict_line_from_synthesis,
        t_verdict_body_counts,
        t_open_questions,
        t_zero_cards,
        t_label_truncation,
        t_match_strength_clamp,
    ]
    print(f"\nRunning {len(tests)} map_adapter tests...\n")
    for fn in tests:
        run_test(fn)
    print(f"\n{'=' * 60}")
    print(f"  {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 60}")
    if FAILED:
        for name, msg in ERRORS:
            print(f"  ✗ {name}: {msg}")
        raise SystemExit(1)
