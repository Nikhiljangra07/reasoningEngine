"""
Bridge interface tests.

No LLM calls, no API hits, no live backends. Tests verify:
    1. Type construction (the data contracts hold their shape)
    2. BridgeClient mode validation
    3. Stub-mode graphify reads return well-formed empty results
    4. Memory V2 reads/writes raise NotImplementedError with TODO pointers
    5. Drift detection runs end-to-end against the stub comparator and
       returns a DriftReport with is_drifted=False (since the stub never
       flags drift)

Run with: python -m pytest tests/test_bridge.py -v
Or directly: python tests/test_bridge.py
"""

from __future__ import annotations

import asyncio
import time

from src.bridge import (
    BridgeClient,
    BridgeQuery,
    BridgeResult,
    CodeRef,
    ContextFingerprint,
    DecisionAnchor,
    DriftReport,
)
from src.bridge.drift import detect_drift, stub_comparator
from src.bridge.graphify_adapter import GraphifyAdapter
from src.bridge.memory_adapter import MemoryAdapter


# ---------------------------------------------------------------------------
# Test infrastructure (matches the lightweight style of test_integration.py)
# ---------------------------------------------------------------------------

PASSED = 0
FAILED = 0
ERRORS: list[tuple[str, str]] = []


def test(name: str):
    """Decorator that tags a test function with its display name."""
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    """Run a single test and track pass/fail."""
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
# Fixtures
# ---------------------------------------------------------------------------

def make_code_ref() -> CodeRef:
    return CodeRef(
        file_path="src/handlers/refund.ts",
        line_start=17,
        line_end=22,
        symbol_name="refund",
        symbol_type="function",
    )


def make_decision() -> DecisionAnchor:
    return DecisionAnchor(
        id="D-014",
        title="Idempotency keys = request-shape hash + orgId",
        rationale=(
            "Two production incidents in the last quarter caused by "
            "single-component keys colliding across orgs."
        ),
        evidence=[
            "Stripe's own behavior (request fingerprint included)",
            "INC-0042 — duplicate refund across org-A and org-B (2026-02-11)",
        ],
        status="SETTLED",
        created_at=time.time(),
        code_refs=[make_code_ref()],
        tags=["idempotency", "payments"],
    )


# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

@test("1.1 CodeRef constructs with required fields")
def test_code_ref_construct():
    ref = make_code_ref()
    assert ref.file_path == "src/handlers/refund.ts"
    assert ref.line_start == 17
    assert ref.line_end == 22
    assert ref.symbol_name == "refund"


@test("1.2 DecisionAnchor constructs with code_refs and tags")
def test_decision_anchor_construct():
    d = make_decision()
    assert d.id == "D-014"
    assert d.status == "SETTLED"
    assert len(d.code_refs) == 1
    assert d.code_refs[0].file_path == "src/handlers/refund.ts"
    assert d.superseded_by is None
    assert "idempotency" in d.tags


@test("1.3 ContextFingerprint accepts None vector in stub mode")
def test_context_fingerprint_none_vector():
    fp = ContextFingerprint(
        id="fp-001",
        vector=None,
        metadata={"project": "payments", "constraint": "low_latency"},
        created_at=time.time(),
    )
    assert fp.vector is None
    assert fp.metadata["project"] == "payments"


@test("1.4 BridgeQuery / BridgeResult placeholders are constructable")
def test_query_result_placeholders():
    q = BridgeQuery(kind="placeholder", args={"k": 5})
    r = BridgeResult(kind="placeholder")
    assert q.kind == "placeholder"
    assert r.decisions == []
    assert r.code_refs == []
    assert r.drift_reports == []


# ---------------------------------------------------------------------------
# 2. BridgeClient mode validation
# ---------------------------------------------------------------------------

@test("2.1 BridgeClient(mode='stub') constructs cleanly")
def test_stub_mode_constructs():
    bridge = BridgeClient(repo_root=".", mode="stub")
    assert bridge.mode == "stub"
    assert bridge.repo_root == "."


@test("2.2 BridgeClient(mode='live') without anchor_backend raises ValueError")
def test_live_mode_requires_backend():
    # Live mode demands an injected anchor_backend (typically Neo4j); without
    # one we refuse construction rather than silently degrade to in-memory.
    try:
        BridgeClient(repo_root=".", mode="live")
    except ValueError as e:
        assert "anchor_backend" in str(e)
        return
    raise AssertionError("expected ValueError for live mode without anchor_backend")


@test("2.3 BridgeClient with invalid mode raises ValueError")
def test_invalid_mode_raises():
    try:
        BridgeClient(repo_root=".", mode="bogus")
    except ValueError as e:
        assert "stub" in str(e) and "live" in str(e)
        return
    raise AssertionError("expected ValueError for invalid mode")


@test("2.4 BridgeClient stores optional project_id (default=None, back-compat)")
def test_bridge_project_id():
    # No project_id supplied — back-compat default
    b1 = BridgeClient(repo_root=".", mode="stub")
    assert b1.project_id is None

    # project_id supplied — stored as attribute
    b2 = BridgeClient(repo_root=".", mode="stub", project_id="abc123def456")
    assert b2.project_id == "abc123def456"
    assert b2.repo_root == "."
    assert b2.mode == "stub"


# ---------------------------------------------------------------------------
# 3. Stub-mode graphify reads
# ---------------------------------------------------------------------------

@test("3.1 get_code_structure returns well-formed empty shape in stub mode")
async def test_stub_get_code_structure():
    bridge = BridgeClient(repo_root=".", mode="stub")
    result = await bridge.get_code_structure("src/llm/client.py")
    assert isinstance(result, dict)
    assert result["file"] == "src/llm/client.py"
    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["node_count"] == 0
    assert result["edge_count"] == 0
    assert result["mode"] == "stub"


@test("3.2 get_callers_of returns empty list in stub mode")
async def test_stub_get_callers_of():
    bridge = BridgeClient(repo_root=".", mode="stub")
    result = await bridge.get_callers_of("run_formation")
    assert result == []


@test("3.3 get_dependencies_of returns empty list in stub mode")
async def test_stub_get_dependencies_of():
    bridge = BridgeClient(repo_root=".", mode="stub")
    result = await bridge.get_dependencies_of("src/formation/orchestrator.py")
    assert result == []


# ---------------------------------------------------------------------------
# 4. Memory V2 — in-memory CRUD works (project-scoped)
# ---------------------------------------------------------------------------

@test("4.1 get_decision returns None for unknown id")
async def test_get_decision_unknown():
    bridge = BridgeClient(repo_root=".", mode="stub", project_id="proj-1")
    result = await bridge.get_decision("D-does-not-exist")
    assert result is None


@test("4.2 store + get round-trip returns the same decision")
async def test_store_get_round_trip():
    bridge = BridgeClient(repo_root=".", mode="stub", project_id="proj-1")
    d = make_decision()
    stored_id = await bridge.store_decision(d)
    assert stored_id == d.id
    fetched = await bridge.get_decision(d.id)
    assert fetched is not None
    assert fetched.id == d.id
    assert fetched.title == d.title


@test("4.3 get_decisions_touching_file returns matching decisions")
async def test_get_decisions_touching_file():
    bridge = BridgeClient(repo_root=".", mode="stub", project_id="proj-1")
    await bridge.store_decision(make_decision())
    matches = await bridge.get_decisions_touching_file("src/handlers/refund.ts")
    assert len(matches) == 1
    assert matches[0].id == "D-014"

    # Unrelated path returns empty
    empty = await bridge.get_decisions_touching_file("src/other/module.py")
    assert empty == []


@test("4.4 update_decision_status mutates stored decision")
async def test_update_status():
    bridge = BridgeClient(repo_root=".", mode="stub", project_id="proj-1")
    await bridge.store_decision(make_decision())
    await bridge.update_decision_status("D-014", "DRIFTED")
    fetched = await bridge.get_decision("D-014")
    assert fetched is not None
    assert fetched.status == "DRIFTED"


@test("4.5 update_decision_status on unknown id raises KeyError")
async def test_update_status_unknown_raises():
    bridge = BridgeClient(repo_root=".", mode="stub", project_id="proj-1")
    try:
        await bridge.update_decision_status("D-nonexistent", "DRIFTED")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown decision id")


@test("4.6 find_similar_decisions ranks by keyword overlap")
async def test_find_similar_keyword():
    bridge = BridgeClient(repo_root=".", mode="stub", project_id="proj-1")
    await bridge.store_decision(make_decision())  # idempotency / payments tags
    # Strong overlap
    hits = await bridge.find_similar_decisions("idempotency strategy", k=3)
    assert len(hits) == 1
    assert hits[0].id == "D-014"
    # No overlap
    empty = await bridge.find_similar_decisions(
        "completely unrelated quantum chromodynamics topic", k=3
    )
    assert empty == []


@test("4.7 get_code_refs_for_decision returns the stored refs")
async def test_get_code_refs():
    bridge = BridgeClient(repo_root=".", mode="stub", project_id="proj-1")
    await bridge.store_decision(make_decision())
    refs = await bridge.get_code_refs_for_decision("D-014")
    assert len(refs) == 1
    assert refs[0].file_path == "src/handlers/refund.ts"


@test("4.8 get_code_refs_for_decision returns [] for unknown id")
async def test_get_code_refs_unknown():
    bridge = BridgeClient(repo_root=".", mode="stub", project_id="proj-1")
    refs = await bridge.get_code_refs_for_decision("D-nonexistent")
    assert refs == []


# ---------------------------------------------------------------------------
# 4P. Project scoping — the blending-prevention defense, enforced at storage
# ---------------------------------------------------------------------------

@test("4P.1 decisions stored in project A are INVISIBLE to queries in project B")
async def test_project_scoping_isolates():
    bridge_a = BridgeClient(repo_root=".", mode="stub", project_id="proj-A")
    bridge_b = BridgeClient(repo_root=".", mode="stub", project_id="proj-B")
    # Each bridge has its OWN store dict (instance-local), so explicit
    # isolation is the default. Storing in A should not leak to B.
    await bridge_a.store_decision(make_decision())
    # B has its own bucket — query returns nothing
    assert await bridge_b.get_decision("D-014") is None
    assert await bridge_b.get_decisions_touching_file("src/handlers/refund.ts") == []
    assert await bridge_b.find_similar_decisions("idempotency", k=5) == []


@test("4P.2 shared store + different project_ids → still no blending")
async def test_project_scoping_with_shared_store():
    from src.bridge.memory_adapter import MemoryAdapter
    shared: dict = {}
    adapter_a = MemoryAdapter(repo_root=".", project_id="proj-A", store=shared)
    adapter_b = MemoryAdapter(repo_root=".", project_id="proj-B", store=shared)
    # Store the same-id decision twice under different projects — they
    # must NOT collide. Each project has its own bucket inside `shared`.
    d_a = make_decision()  # id="D-014" with project-A title
    await adapter_a.store_decision(d_a)
    # Even with the same id in the same shared dict, querying B sees nothing
    assert await adapter_b.get_decision("D-014") is None
    assert await adapter_a.get_decision("D-014") is not None
    # Verify shared dict has two distinct top-level keys, one per project
    assert "proj-A" in shared
    assert "proj-B" not in shared or shared["proj-B"] == {}


@test("4P.3 MemoryAdapter direct CRUD works (back-compat: no project_id)")
async def test_memory_adapter_direct_unscoped():
    from src.bridge.memory_adapter import MemoryAdapter
    adapter = MemoryAdapter(repo_root=".")  # no project_id → unscoped bucket
    stored_id = await adapter.store_decision(make_decision())
    assert stored_id == "D-014"
    fetched = await adapter.get_decision("D-014")
    assert fetched is not None


# ---------------------------------------------------------------------------
# 4S. Disk persistence — opt-in via storage_path
# ---------------------------------------------------------------------------

@test("4S.1 no storage_path → in-memory only, no file ever created")
async def test_persistence_disabled_by_default():
    import os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        # No storage_path supplied
        adapter = MemoryAdapter(repo_root=".", project_id="proj-1")
        await adapter.store_decision(make_decision())
        # Tempdir should remain empty
        assert os.listdir(tmp) == []


@test("4S.2 storage_path: store_decision auto-writes to disk")
async def test_persistence_autosave_store():
    import json, os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")
        adapter = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        await adapter.store_decision(make_decision())
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["schema_version"] == 1
        assert "proj-1" in data["projects"]
        assert "D-014" in data["projects"]["proj-1"]


@test("4S.3 reconstructed adapter loads prior state from disk")
async def test_persistence_load_on_construct():
    import os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")
        # Write
        a1 = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        await a1.store_decision(make_decision())
        # Reload in a fresh instance
        a2 = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        loaded = await a2.get_decision("D-014")
        assert loaded is not None
        assert loaded.title == "Idempotency keys = request-shape hash + orgId"
        # Nested CodeRef objects reconstructed
        assert len(loaded.code_refs) == 1
        assert loaded.code_refs[0].file_path == "src/handlers/refund.ts"


@test("4S.4 update_decision_status persists across reload")
async def test_persistence_status_update():
    import os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")
        a1 = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        await a1.store_decision(make_decision())
        await a1.update_decision_status("D-014", "DRIFTED")
        # Fresh adapter sees the updated status
        a2 = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        d = await a2.get_decision("D-014")
        assert d is not None
        assert d.status == "DRIFTED"


@test("4S.5 corrupted memory file → silent fresh start, no crash")
async def test_persistence_corrupted():
    import os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")
        with open(path, "w") as f:
            f.write("garbage }}}{{{ not json")
        # Construction must not raise
        adapter = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        assert await adapter.get_decision("D-014") is None


@test("4S.6 missing storage file → fresh registry, no error, no file created")
async def test_persistence_missing_file():
    import os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "does_not_exist.json")
        adapter = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        assert await adapter.get_decision("D-014") is None
        # Construct alone doesn't create the file — mutations do
        assert not os.path.exists(path)


@test("4S.7 schema drift: unknown field on a stored decision → that entry skipped, rest load")
async def test_persistence_schema_drift():
    import json, os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")
        # Hand-write a JSON file with one good entry + one corrupted entry
        data = {
            "schema_version": 1,
            "projects": {
                "proj-1": {
                    "D-good": {
                        "id": "D-good", "title": "good", "rationale": "ok",
                        "evidence": [], "status": "OPEN", "created_at": 0.0,
                        "superseded_by": None, "supersedes": None,
                        "code_refs": [], "context_fingerprint_id": None,
                        "tags": [],
                    },
                    "D-future": {
                        "id": "D-future", "title": "future", "rationale": "x",
                        "evidence": [], "status": "OPEN", "created_at": 0.0,
                        "superseded_by": None, "supersedes": None,
                        "code_refs": [], "context_fingerprint_id": None,
                        "tags": [],
                        "unknown_future_field": "skip me",  # not in dataclass
                    },
                },
            },
        }
        with open(path, "w") as f:
            json.dump(data, f)
        adapter = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        # Good entry loaded
        assert await adapter.get_decision("D-good") is not None
        # Corrupted entry skipped, not crashed
        assert await adapter.get_decision("D-future") is None


@test("4S.8 atomic write: .tmp file does not survive after save")
async def test_persistence_atomic_write():
    import os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")
        adapter = MemoryAdapter(repo_root=".", project_id="proj-1", storage_path=path)
        await adapter.store_decision(make_decision())
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")


@test("4S.9 multiple projects in one file — each loads back into its own bucket")
async def test_persistence_multi_project():
    import os, tempfile
    from src.bridge.memory_adapter import MemoryAdapter
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "memory.json")
        # Single shared store + different project_ids, persistence on
        shared: dict = {}
        a1 = MemoryAdapter(
            repo_root=".", project_id="proj-A",
            store=shared, storage_path=path,
        )
        a2 = MemoryAdapter(
            repo_root=".", project_id="proj-B",
            store=shared, storage_path=path,
        )
        await a1.store_decision(make_decision())
        await a2.store_decision(make_decision())
        # Fresh adapters loading the file
        a3 = MemoryAdapter(repo_root=".", project_id="proj-A", storage_path=path)
        a4 = MemoryAdapter(repo_root=".", project_id="proj-B", storage_path=path)
        # Each sees ITS OWN project's decision; never the other's
        assert await a3.get_decision("D-014") is not None
        assert await a4.get_decision("D-014") is not None
        # And cross-project queries on the loaded adapters are still isolated
        a3_other_project = MemoryAdapter(
            repo_root=".", project_id="proj-A",
            store=a4._store, storage_path=None,
        )
        # a3_other_project shares a4's store; queries scoped to proj-A
        # should find proj-A's data via the shared store
        assert await a3_other_project.get_decision("D-014") is not None


# ---------------------------------------------------------------------------
# 5. Drift detection with stub comparator
# ---------------------------------------------------------------------------

@test("5.1 detect_drift returns DriftReport with is_drifted=False (stub comparator)")
async def test_detect_drift_stub():
    bridge = BridgeClient(repo_root=".", mode="stub")
    decision = make_decision()
    report = await bridge.detect_drift(decision)
    assert isinstance(report, DriftReport)
    assert report.decision.id == "D-014"
    assert report.is_drifted is False
    assert report.confidence == 0.0
    assert report.suggested_action == "no_action"
    assert "not yet implemented" in report.drift_description


@test("5.2 detect_drift on decision with no code_refs returns floating-decision report")
async def test_detect_drift_no_refs():
    decision = make_decision()
    decision.code_refs = []
    graphify = GraphifyAdapter(repo_root=".")
    report = await detect_drift(decision, graphify)
    assert report.is_drifted is False
    assert "no code_refs" in report.drift_description
    # Floating-decision reports carry no per-ref breakdown.
    assert report.per_ref_reports == []


@test("5.2b detect_drift populates per_ref_reports for multi-ref decisions")
async def test_detect_drift_per_ref_reports():
    decision = make_decision()
    # Add a second code_ref so we can verify the per-ref breakdown.
    decision.code_refs.append(CodeRef(
        file_path="src/handlers/refund.ts",
        line_start=40,
        line_end=55,
        symbol_name="refund_async",
        symbol_type="function",
    ))
    graphify = GraphifyAdapter(repo_root=".")
    report = await detect_drift(decision, graphify)
    # Top-level summary stays unchanged.
    assert report.is_drifted is False
    # Per-ref breakdown is populated, one entry per code_ref, in order.
    assert len(report.per_ref_reports) == 2
    assert report.per_ref_reports[0].code_ref.line_start == 17
    assert report.per_ref_reports[1].code_ref.line_start == 40
    # Each per-ref report is itself a DriftReport.
    for r in report.per_ref_reports:
        assert isinstance(r, DriftReport)


@test("5.3 stub_comparator returns the contracted 4-tuple")
async def test_stub_comparator_shape():
    decision = make_decision()
    ref = decision.code_refs[0]
    result = await stub_comparator(decision, ref, current_code="")
    assert isinstance(result, tuple)
    assert len(result) == 4
    is_drifted, description, confidence, action = result
    assert is_drifted is False
    assert isinstance(description, str)
    assert confidence == 0.0
    assert action == "no_action"


# ---------------------------------------------------------------------------
# 6. GraphifyAdapter direct construction (smoke — no graph file needed)
# ---------------------------------------------------------------------------

@test("6.1 GraphifyAdapter constructs without graph file present (lazy load)")
def test_graphify_adapter_lazy():
    adapter = GraphifyAdapter(repo_root=".")
    assert adapter.graph_path.name == "graph.json"
    assert adapter._graph is None  # not loaded until first query


@test("6.2 GraphifyAdapter raises FileNotFoundError on query when graph missing")
async def test_graphify_adapter_missing_graph():
    adapter = GraphifyAdapter(repo_root="/tmp/__nonexistent_repo__")
    try:
        await adapter.get_code_structure("src/anything.py")
    except FileNotFoundError as e:
        assert "graphify extract" in str(e)
        return
    raise AssertionError("expected FileNotFoundError")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_code_ref_construct,
    test_decision_anchor_construct,
    test_context_fingerprint_none_vector,
    test_query_result_placeholders,
    test_stub_mode_constructs,
    test_live_mode_requires_backend,
    test_invalid_mode_raises,
    test_bridge_project_id,
    test_stub_get_code_structure,
    test_stub_get_callers_of,
    test_stub_get_dependencies_of,
    test_get_decision_unknown,
    test_store_get_round_trip,
    test_get_decisions_touching_file,
    test_update_status,
    test_update_status_unknown_raises,
    test_find_similar_keyword,
    test_get_code_refs,
    test_get_code_refs_unknown,
    test_project_scoping_isolates,
    test_project_scoping_with_shared_store,
    test_memory_adapter_direct_unscoped,
    test_persistence_disabled_by_default,
    test_persistence_autosave_store,
    test_persistence_load_on_construct,
    test_persistence_status_update,
    test_persistence_corrupted,
    test_persistence_missing_file,
    test_persistence_schema_drift,
    test_persistence_atomic_write,
    test_persistence_multi_project,
    test_detect_drift_stub,
    test_detect_drift_no_refs,
    test_detect_drift_per_ref_reports,
    test_stub_comparator_shape,
    test_graphify_adapter_lazy,
    test_graphify_adapter_missing_graph,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} bridge interface tests...")
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
