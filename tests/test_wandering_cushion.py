"""
Phase 0 tests — cushion data structures + brief composer.

Run: PYTHONPATH=. python3 tests/test_wandering_cushion.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from src.llm.client import LLMResponse
from src.wandering.cushion import (
    CushionField,
    CushionGraph,
    CushionInput,
    CushionLayer,
    SkipReason,
)
from src.wandering.composer import (
    MAX_NODES_PER_LAYER,
    MIN_NODES_PER_LAYER,
    build_extraction_user_message,
    compose_cushion,
    fetch_memory_enrichment,
    parse_extraction_response,
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
# Fake LLM client — for testing compose_cushion without live API
# ===========================================================================


class _FakeLLMClient:
    """Minimal async stub that returns a prescripted LLMResponse.

    Tests inject whatever response text they want. Real LLMClient interface
    is wider; we only need .call() for the composer.
    """

    def __init__(self, response_text: str, success: bool = True, error: str | None = None):
        self.response_text = response_text
        self.success = success
        self.error = error
        self.last_system_prompt: str = ""
        self.last_user_message: str = ""
        self.last_domain: str = ""
        self.last_concept: str = ""

    async def call(self, *, system_prompt, user_message, domain, concept, **kwargs):
        self.last_system_prompt = system_prompt
        self.last_user_message = user_message
        self.last_domain = domain
        self.last_concept = concept
        return LLMResponse(
            content=self.response_text,
            input_tokens=100,
            output_tokens=50,
            latency_ms=42.0,
            success=self.success,
            model="claude-sonnet-4-6",
            error=self.error,
        )


def _good_extraction_json() -> str:
    """A valid 3-layer extraction response Sonnet might return."""
    payload = {
        "actual": {
            "summary": "AI agents that wander the internet for inspirations.",
            "nodes": [
                "wandering AI agents",
                "internet exploration",
                "research anchor",
                "credit budget",
            ],
        },
        "essence": {
            "summary": "Bounded freedom: chaos within structural anchor.",
            "nodes": [
                "bounded freedom",
                "productive constraint",
                "anchored chaos",
                "soft-vs-hard constraint paradox",
                "trust through observation",
            ],
        },
        "mechanism": {
            "summary": "Systems wanting unpredictable output need soft constraint.",
            "nodes": [
                "unpredictable output under resource limits",
                "soft constraint enables emergence",
                "hard constraint kills the value behavior",
                "observation as low-touch control",
            ],
        },
    }
    return json.dumps(payload)


# ===========================================================================
# 1. CushionField behavior
# ===========================================================================


@test("1.1 CushionField filled with content is is_filled()")
def test_field_filled():
    f = CushionField(name="problem", content="figuring out how to control agents")
    assert f.is_filled() is True
    assert f.is_skipped() is False


@test("1.2 CushionField with empty content is is_skipped()")
def test_field_empty():
    f = CushionField(name="problem", content="")
    assert f.is_filled() is False
    assert f.is_skipped() is True


@test("1.3 CushionField with whitespace-only content is skipped")
def test_field_whitespace():
    f = CushionField(name="problem", content="   \n  ")
    assert f.is_filled() is False


@test("1.4 SkipReason.SKIPPED_AFTER_PROMPT field is not filled even with content")
def test_field_skipped_after_prompt():
    # User typed something but then explicitly skipped — treat as skipped.
    f = CushionField(
        name="problem",
        content="text that was abandoned",
        skip_reason=SkipReason.SKIPPED_AFTER_PROMPT,
    )
    assert f.is_filled() is False
    assert f.is_skipped() is True


# ===========================================================================
# 2. CushionInput viability
# ===========================================================================


@test("2.1 CushionInput with all four fields filled is minimally viable")
def test_input_full():
    inp = CushionInput(
        problem=CushionField(name="problem", content="the problem"),
        context=CushionField(name="context", content="the context"),
        vision=CushionField(name="vision", content="the vision"),
        hunches=CushionField(name="hunches", content="threads"),
    )
    assert inp.is_minimally_viable() is True
    assert inp.filled_field_count() == 4


@test("2.2 CushionInput with only problem filled is minimally viable")
def test_input_problem_only():
    # Per design — problem is the floor; user can skip everything else.
    inp = CushionInput(
        problem=CushionField(name="problem", content="the problem"),
        context=CushionField(name="context"),
        vision=CushionField(name="vision"),
        hunches=CushionField(name="hunches"),
    )
    assert inp.is_minimally_viable() is True
    assert inp.filled_field_count() == 1


@test("2.3 CushionInput with problem skipped is NOT minimally viable")
def test_input_no_problem():
    inp = CushionInput(
        problem=CushionField(name="problem"),
        context=CushionField(name="context", content="x"),
        vision=CushionField(name="vision", content="y"),
        hunches=CushionField(name="hunches", content="z"),
    )
    assert inp.is_minimally_viable() is False


@test("2.4 CushionInput.fields() returns four in canonical order; question excluded")
def test_input_fields_order():
    inp = CushionInput(
        problem=CushionField(name="problem", content="p"),
        context=CushionField(name="context", content="c"),
        vision=CushionField(name="vision", content="v"),
        hunches=CushionField(name="hunches", content="m"),
        question=CushionField(name="question", content="the checkpoint"),
    )
    names = [f.name for f in inp.fields()]
    # `question` is intentionally EXCLUDED from fields() — it is a judge-facing
    # checkpoint, never extracted into the wander anchor — yet still stored.
    assert names == ["problem", "context", "vision", "hunches"]
    assert "question" not in names
    assert inp.question.content == "the checkpoint"


# ===========================================================================
# 3. CushionLayer + CushionGraph behavior
# ===========================================================================


@test("3.1 CushionLayer node_count reflects actual list size")
def test_layer_count():
    layer = CushionLayer(name="essence", nodes=["a", "b", "c"])
    assert layer.node_count() == 3


@test("3.2 CushionGraph auto-computes constellation_size on init")
def test_graph_constellation_size():
    g = _make_minimal_graph()
    # 3 + 4 + 5 = 12 nodes
    assert g.constellation_size == 12


@test("3.3 CushionGraph layers() returns three in canonical order")
def test_graph_layers_order():
    g = _make_minimal_graph()
    names = [layer.name for layer in g.layers()]
    assert names == ["actual", "essence", "mechanism"]


@test("3.4 CushionGraph is_well_formed when each layer has >=1 node + input viable")
def test_graph_well_formed():
    g = _make_minimal_graph()
    assert g.is_well_formed() is True


@test("3.5 CushionGraph is NOT well-formed if any layer is empty")
def test_graph_not_well_formed_empty_layer():
    g = _make_minimal_graph()
    g.essence = CushionLayer(name="essence", nodes=[])
    assert g.is_well_formed() is False


@test("3.6 CushionGraph.to_anchor_prompt contains all three layer names")
def test_anchor_prompt_shape():
    g = _make_minimal_graph()
    prompt = g.to_anchor_prompt()
    assert "ACTUAL" in prompt
    assert "ESSENCE" in prompt
    assert "MECHANISM" in prompt
    assert "do not detach" in prompt.lower()


def _make_minimal_graph() -> CushionGraph:
    """Helper: a small but valid CushionGraph for tests."""
    return CushionGraph(
        actual=CushionLayer(
            name="actual",
            nodes=["a1", "a2", "a3"],
            summary="actual summary",
        ),
        essence=CushionLayer(
            name="essence",
            nodes=["e1", "e2", "e3", "e4"],
            summary="essence summary",
        ),
        mechanism=CushionLayer(
            name="mechanism",
            nodes=["m1", "m2", "m3", "m4", "m5"],
            summary="mechanism summary",
        ),
        raw_input=CushionInput(
            problem=CushionField(name="problem", content="x"),
            context=CushionField(name="context"),
            vision=CushionField(name="vision"),
            hunches=CushionField(name="hunches"),
        ),
    )


# ===========================================================================
# 4. parse_extraction_response — happy path + degradations
# ===========================================================================


@test("4.1 parse happy path returns three layers with correct names")
def test_parse_happy():
    layers = parse_extraction_response(_good_extraction_json())
    assert set(layers.keys()) == {"actual", "essence", "mechanism"}
    assert layers["actual"].node_count() == 4
    assert layers["essence"].node_count() == 5


@test("4.2 parse strips ```json code fences if model added them")
def test_parse_code_fences():
    wrapped = "```json\n" + _good_extraction_json() + "\n```"
    layers = parse_extraction_response(wrapped)
    assert len(layers) == 3


@test("4.3 parse extracts JSON even when wrapped in prose")
def test_parse_prose_wrapped():
    wrapped = "Sure, here is the JSON:\n\n" + _good_extraction_json() + "\n\nLet me know!"
    layers = parse_extraction_response(wrapped)
    assert len(layers) == 3


@test("4.4 parse rejects malformed JSON")
def test_parse_malformed():
    try:
        parse_extraction_response("{not valid json at all")
    except ValueError:
        return
    raise AssertionError("expected ValueError on malformed JSON")


@test("4.5 parse rejects missing layer")
def test_parse_missing_layer():
    payload = json.loads(_good_extraction_json())
    del payload["mechanism"]
    try:
        parse_extraction_response(json.dumps(payload))
    except ValueError as e:
        assert "mechanism" in str(e)
        return
    raise AssertionError("expected ValueError on missing layer")


@test("4.6 parse rejects layer with fewer than MIN_NODES_PER_LAYER")
def test_parse_too_few_nodes():
    payload = json.loads(_good_extraction_json())
    payload["actual"]["nodes"] = ["only", "two"]  # < 3
    try:
        parse_extraction_response(json.dumps(payload))
    except ValueError as e:
        assert "minimum" in str(e).lower()
        return
    raise AssertionError("expected ValueError on too few nodes")


@test("4.7 parse truncates layer with more than MAX_NODES_PER_LAYER (no error)")
def test_parse_too_many_nodes_truncates():
    payload = json.loads(_good_extraction_json())
    payload["essence"]["nodes"] = [f"node{i}" for i in range(15)]  # > 8
    layers = parse_extraction_response(json.dumps(payload))
    assert layers["essence"].node_count() == MAX_NODES_PER_LAYER


@test("4.8 parse drops empty/whitespace nodes in a layer")
def test_parse_drops_whitespace_nodes():
    payload = json.loads(_good_extraction_json())
    payload["actual"]["nodes"] = ["good", "", "  ", "also good", "third"]
    layers = parse_extraction_response(json.dumps(payload))
    assert layers["actual"].node_count() == 3


# ===========================================================================
# 5. build_extraction_user_message
# ===========================================================================


@test("5.1 build_user_message includes all filled fields by label")
def test_user_message_filled():
    inp = CushionInput(
        problem=CushionField(name="problem", content="control wandering agents"),
        context=CushionField(name="context", content="part of Constellax"),
        vision=CushionField(name="vision", content="cognitive augmentation"),
        hunches=CushionField(name="hunches", content="Heisenberg thread"),
    )
    msg = build_extraction_user_message(inp)
    assert "PROBLEM" in msg
    assert "CONTEXT" in msg
    assert "VISION" in msg
    assert "HUNCHES" in msg
    assert "control wandering agents" in msg
    assert "cognitive augmentation" in msg


@test("5.2 build_user_message marks skipped fields explicitly")
def test_user_message_skipped():
    inp = CushionInput(
        problem=CushionField(name="problem", content="x"),
        context=CushionField(name="context"),  # skipped
        vision=CushionField(name="vision"),
        hunches=CushionField(name="hunches"),
    )
    msg = build_extraction_user_message(inp)
    assert "skipped by user" in msg.lower()


@test("5.3 build_user_message includes memory enrichment block when present")
def test_user_message_memory():
    inp = CushionInput(
        problem=CushionField(name="problem", content="x"),
        context=CushionField(name="context"),
        vision=CushionField(name="vision"),
        hunches=CushionField(name="hunches"),
        memory_enrichment="user has been working on Constellax memory pipeline",
    )
    msg = build_extraction_user_message(inp)
    assert "AUTO-ENRICHED" in msg
    assert "memory pipeline" in msg


# ===========================================================================
# 6. fetch_memory_enrichment (Phase 0 stub)
# ===========================================================================


@test("6.1 fetch_memory_enrichment returns empty string in Phase 0")
async def test_memory_phase0_stub():
    out = await fetch_memory_enrichment(user_id="usr-test-123")
    assert out == ""


@test("6.2 fetch_memory_enrichment with no user_id returns empty")
async def test_memory_no_user():
    out = await fetch_memory_enrichment(user_id=None)
    assert out == ""


# ===========================================================================
# 7. compose_cushion — end-to-end with fake client
# ===========================================================================


@test("7.1 compose_cushion happy path produces well-formed graph")
async def test_compose_happy():
    inp = _good_input()
    client = _FakeLLMClient(response_text=_good_extraction_json())
    graph = await compose_cushion(inp, client)  # type: ignore[arg-type]

    assert graph.is_well_formed() is True
    assert graph.actual.node_count() >= 3
    assert graph.essence.node_count() >= 3
    assert graph.mechanism.node_count() >= 3
    assert graph.extraction_model == "claude-sonnet-4-6"
    assert graph.extracted_at > 0
    # Routing
    assert client.last_domain == "synthesizer"
    assert client.last_concept == "cushion_extraction"


@test("7.2 compose_cushion rejects input with skipped problem field")
async def test_compose_no_problem():
    inp = CushionInput(
        problem=CushionField(name="problem"),  # SKIPPED
        context=CushionField(name="context", content="x"),
        vision=CushionField(name="vision", content="y"),
        hunches=CushionField(name="hunches", content="z"),
    )
    client = _FakeLLMClient(response_text=_good_extraction_json())
    try:
        await compose_cushion(inp, client)  # type: ignore[arg-type]
    except ValueError as e:
        assert "minimally viable" in str(e)
        return
    raise AssertionError("expected ValueError on skipped problem field")


@test("7.3 compose_cushion raises RuntimeError on LLM failure")
async def test_compose_llm_failure():
    inp = _good_input()
    client = _FakeLLMClient(
        response_text="",
        success=False,
        error="connection timeout",
    )
    try:
        await compose_cushion(inp, client)  # type: ignore[arg-type]
    except RuntimeError as e:
        assert "timeout" in str(e)
        return
    raise AssertionError("expected RuntimeError on LLM failure")


@test("7.4 compose_cushion injects memory enrichment when auto_enrich=True")
async def test_compose_auto_enrich():
    inp = _good_input()
    client = _FakeLLMClient(response_text=_good_extraction_json())
    # Phase 0 stub returns "" so we just verify the call path doesn't error.
    graph = await compose_cushion(  # type: ignore[arg-type]
        inp, client, user_id="usr-test-123", auto_enrich=True
    )
    assert graph.is_well_formed() is True


@test("7.5 compose_cushion preserves raw_input for audit")
async def test_compose_preserves_raw():
    inp = _good_input()
    client = _FakeLLMClient(response_text=_good_extraction_json())
    graph = await compose_cushion(inp, client)  # type: ignore[arg-type]
    assert graph.raw_input is inp
    assert graph.raw_input.problem.content == inp.problem.content


def _good_input() -> CushionInput:
    return CushionInput(
        problem=CushionField(
            name="problem",
            content="How do I control 10 wandering agents without killing their freedom?",
        ),
        context=CushionField(
            name="context",
            content="Part of Constellax research-mode feature.",
        ),
        vision=CushionField(
            name="vision",
            content="A cognitive augmentation tool that simulates partial-match inspiration.",
        ),
        hunches=CushionField(
            name="hunches",
            content="Looking at Heisenberg uncertainty, Taoist Wuxing, pendulum chaos.",
        ),
    )


# ===========================================================================
# Runner
# ===========================================================================


ALL_TESTS = [
    # 1. Field
    test_field_filled,
    test_field_empty,
    test_field_whitespace,
    test_field_skipped_after_prompt,
    # 2. Input
    test_input_full,
    test_input_problem_only,
    test_input_no_problem,
    test_input_fields_order,
    # 3. Layer + Graph
    test_layer_count,
    test_graph_constellation_size,
    test_graph_layers_order,
    test_graph_well_formed,
    test_graph_not_well_formed_empty_layer,
    test_anchor_prompt_shape,
    # 4. parse_extraction_response
    test_parse_happy,
    test_parse_code_fences,
    test_parse_prose_wrapped,
    test_parse_malformed,
    test_parse_missing_layer,
    test_parse_too_few_nodes,
    test_parse_too_many_nodes_truncates,
    test_parse_drops_whitespace_nodes,
    # 5. build_extraction_user_message
    test_user_message_filled,
    test_user_message_skipped,
    test_user_message_memory,
    # 6. fetch_memory_enrichment
    test_memory_phase0_stub,
    test_memory_no_user,
    # 7. compose_cushion
    test_compose_happy,
    test_compose_no_problem,
    test_compose_llm_failure,
    test_compose_auto_enrich,
    test_compose_preserves_raw,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} Wandering Room cushion tests...")
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
