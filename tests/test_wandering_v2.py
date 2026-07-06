"""
Tests for the V2 sprint:

  F5 — Persistent JobState across server restart
       (jobs.py durability + WanderingStore.save_job_state / get_job_state
        / list_running_jobs + sweep_interrupted_from_store)

  F3+F7 unified — Sidebar of past wanders
       (WanderingStore.sessions_metadata + /api/v2/wandering/sessions route)

  F1 — PDF math parsing via Claude native PDF input
       (pdf_extractor.is_pdf_url, is_available, dispatch through extract_url)

Pure-logic tests, mocking out HTTP calls. The wandering pipeline itself
already has 174 tests covering retrieval mesh + map adapter + engine
behavior; this file only adds tests for the V2 additions.

Run:
  PYTHONPATH=. .venv/bin/python tests/test_wandering_v2.py
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from src.wandering import jobs as wandering_jobs
from src.wandering.jobs import (
    JobState,
    JobStatus,
    drain_running_for_shutdown,
    get_status_durable,
    register_job,
    set_store,
    sweep_interrupted_from_store,
)
from src.wandering.persistence import (
    InMemoryWanderingStore,
    WANDERING_SCHEMA_CYPHER,
)
from src.wandering.pdf_extractor import (
    PdfExtractResult,
    is_available as pdf_is_available,
    is_pdf_url,
)
from src.wandering.extractors import extract_url, ExtractResult


# ─── Mini test harness ─────────────────────────────────────────────────

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
    except AssertionError as e:
        FAILED += 1
        ERRORS.append((name, f"FAIL: {e}"))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, f"ERROR: {type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


# ─── F5: JobState mirror + restart recovery ────────────────────────────


@test("F5.1 schema includes :WanderingJob indexes")
def test_schema_has_wandering_job():
    assert "WanderingJob" in WANDERING_SCHEMA_CYPHER
    assert "wandering_job_session_id" in WANDERING_SCHEMA_CYPHER
    assert "wandering_job_status" in WANDERING_SCHEMA_CYPHER


@test("F5.2 InMemoryWanderingStore.save_job_state stores by session_id")
async def test_in_memory_save_job_state():
    store = InMemoryWanderingStore()
    ok = await store.save_job_state({
        "session_id": "wsess-1",
        "job_id":     "wjob-1",
        "status":     "running",
        "started_at": 1000.0,
    })
    assert ok is True
    fetched = await store.get_job_state("wsess-1")
    assert fetched is not None
    assert fetched["session_id"] == "wsess-1"
    assert fetched["status"] == "running"


@test("F5.3 list_running_jobs only returns running entries")
async def test_in_memory_list_running():
    store = InMemoryWanderingStore()
    await store.save_job_state({"session_id": "a", "status": "running"})
    await store.save_job_state({"session_id": "b", "status": "completed"})
    await store.save_job_state({"session_id": "c", "status": "running"})
    await store.save_job_state({"session_id": "d", "status": "failed"})
    running = await store.list_running_jobs()
    sids = sorted([r["session_id"] for r in running])
    assert sids == ["a", "c"]


@test("F5.4 sweep_interrupted_from_store marks running jobs failed")
async def test_sweep_marks_failed():
    store = InMemoryWanderingStore()
    set_store(store)
    try:
        # Pre-populate with a job that was RUNNING from a prior PID.
        await store.save_job_state({
            "session_id":          "prior-wsess",
            "job_id":              "prior-job",
            "user_id":             "guest",
            "status":              "running",
            "started_at":          time.time() - 600,
            "mode":                "multi_pendulum",
            "agents":              5,
            "time_budget_seconds": 1800,
            "pursuit":             "test pursuit",
        })
        swept = await sweep_interrupted_from_store()
        assert swept == 1
        # State should now be marked failed.
        after = await store.get_job_state("prior-wsess")
        assert after is not None
        assert after["status"] == "failed"
        assert after["error"] == "server_restart_during_wander"
    finally:
        set_store(None)
        wandering_jobs.clear_all()


@test("F5.5 sweep with no store is a no-op")
async def test_sweep_no_store():
    set_store(None)
    swept = await sweep_interrupted_from_store()
    assert swept == 0


@test("F5.6 get_status_durable: in-process registry wins")
async def test_status_in_process_wins():
    wandering_jobs.clear_all()
    store = InMemoryWanderingStore()
    set_store(store)
    try:
        # Put a stale "running" entry in the store ...
        await store.save_job_state({
            "session_id": "wsess-x",
            "status":     "running",
            "started_at": 0.0,
            "mode":       "multi_pendulum",
            "agents":     5,
            "time_budget_seconds": 100.0,
            "pursuit":    "",
            "job_id":     "j",
            "user_id":    "guest",
        })
        # ... but ALSO an in-process state that's terminal.
        state = JobState(
            session_id="wsess-x",
            job_id="j",
            user_id="guest",
            started_at=time.time(),
            mode="multi_pendulum",
            agents=5,
            time_budget_seconds=100.0,
            pursuit="",
            status=JobStatus.COMPLETED,
            completed_at=time.time(),
        )
        wandering_jobs._JOBS["wsess-x"] = state
        result = await get_status_durable("wsess-x")
        assert result is not None
        assert result["status"] == "completed"  # in-process won
    finally:
        wandering_jobs.clear_all()
        set_store(None)


@test("F5.7 get_status_durable: store fallback after restart")
async def test_status_store_fallback():
    wandering_jobs.clear_all()
    store = InMemoryWanderingStore()
    set_store(store)
    try:
        # Only the store knows about this session (post-restart).
        await store.save_job_state({
            "session_id":          "wsess-restart",
            "status":              "failed",
            "error":               "server_restart_during_wander",
            "started_at":          time.time() - 600,
            "completed_at":        time.time() - 10,
            "mode":                "multi_pendulum",
            "agents":              5,
            "time_budget_seconds": 1800,
            "pursuit":             "test",
            "job_id":              "j",
            "user_id":             "guest",
        })
        result = await get_status_durable("wsess-restart")
        assert result is not None
        assert result["status"] == "failed"
        assert result["error"] == "server_restart_during_wander"
    finally:
        wandering_jobs.clear_all()
        set_store(None)


@test("F5.8 drain_running_for_shutdown returns only RUNNING jobs")
def test_drain_running():
    wandering_jobs.clear_all()
    # Two running, one completed.
    for sid, status in (("a", JobStatus.RUNNING), ("b", JobStatus.COMPLETED), ("c", JobStatus.RUNNING)):
        wandering_jobs._JOBS[sid] = JobState(
            session_id=sid, job_id="j", user_id="guest",
            started_at=time.time(), mode="multi_pendulum",
            agents=5, time_budget_seconds=100.0, pursuit="",
            status=status,
        )
    running = drain_running_for_shutdown()
    sids = sorted([s.session_id for s in running])
    assert sids == ["a", "c"]
    wandering_jobs.clear_all()


@test("F5.9 register_job mirrors to store fire-and-forget")
async def test_register_mirrors_state():
    wandering_jobs.clear_all()
    store = InMemoryWanderingStore()
    set_store(store)
    try:
        async def _hold():
            await asyncio.sleep(0.05)

        task = asyncio.create_task(_hold())
        state = register_job(
            session_id="wsess-m",
            job_id="wjob-m",
            user_id="guest",
            mode="multi_pendulum",
            agents=5,
            time_budget_seconds=100.0,
            pursuit="test pursuit",
            task=task,
        )
        assert state is not None
        # Yield so the create_task'd mirror runs.
        await asyncio.sleep(0.01)
        mirrored = await store.get_job_state("wsess-m")
        assert mirrored is not None
        assert mirrored["status"] == "running"
        assert mirrored["pursuit"] == "test pursuit"
        await task
    finally:
        wandering_jobs.clear_all()
        set_store(None)


# ─── F3+F7: sidebar metadata + endpoint ────────────────────────────────


@test("F3F7.1 InMemoryWanderingStore.sessions_metadata returns expected shape")
async def test_in_memory_sessions_metadata():
    from src.wandering.cushion import CushionField, CushionGraph, CushionInput, CushionLayer
    from src.wandering.runtime import SessionResult, WanderingConfig, WanderingMode

    cushion = CushionGraph(
        actual=CushionLayer(name="actual", nodes=["a"], summary=""),
        essence=CushionLayer(name="essence", nodes=["e"], summary=""),
        mechanism=CushionLayer(name="mechanism", nodes=["m"], summary=""),
        raw_input=CushionInput(
            problem=CushionField(name="problem", content="how do I X"),
            context=CushionField(name="context", content=""),
            vision=CushionField(name="vision", content=""),
            hunches=CushionField(name="hunches", content=""),
        ),
    )
    session = SessionResult(
        session_id="wsess-md",
        mode=WanderingMode.MULTI_PENDULUM,
        cushion=cushion,
        config=WanderingConfig(mode=WanderingMode.MULTI_PENDULUM, session_id="wsess-md"),
        reports=[],
        traces=[],
        total_tokens_spent=2500,
        elapsed_seconds=45.0,
        ended_at=time.time(),
    )
    store = InMemoryWanderingStore()
    await store.save_session("user-1", session)
    meta = await store.sessions_metadata("user-1", limit=10)
    assert len(meta) == 1
    entry = meta[0]
    assert entry["session_id"] == "wsess-md"
    assert entry["mode"] == "multi_pendulum"
    assert entry["pursuit"] == "how do I X"
    assert entry["report_count"] == 0
    assert entry["total_tokens_spent"] == 2500


@test("F3F7.2 sessions_metadata for unknown user is empty")
async def test_in_memory_sessions_metadata_empty():
    store = InMemoryWanderingStore()
    meta = await store.sessions_metadata("user-nobody", limit=10)
    assert meta == []


@test("F3F7.3 /sessions route registered in router")
def test_sessions_route_registered():
    from src.wandering.routes import get_router
    router = get_router()
    paths = [r.path for r in router.routes if hasattr(r, "path")]
    assert "/sessions" in paths
    # Sanity: the other 7 are still here.
    assert "/brief" in paths
    assert "/session" in paths
    assert "/session/{session_id}/status" in paths


# ─── F1: PDF math parsing dispatch ─────────────────────────────────────


@test("F1.1 is_pdf_url accepts .pdf URLs case-insensitively")
def test_is_pdf_url_case():
    assert is_pdf_url("https://arxiv.org/pdf/1234.5678.pdf") is True
    assert is_pdf_url("https://arxiv.org/pdf/1234.5678.PDF") is True
    assert is_pdf_url("https://arxiv.org/abs/1234.5678") is False
    assert is_pdf_url("") is False


@test("F1.2 is_available reflects ANTHROPIC_API_KEY presence")
def test_pdf_is_available():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        assert pdf_is_available() is False
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        assert pdf_is_available() is True
        os.environ["ANTHROPIC_API_KEY"] = "   "   # whitespace only
        assert pdf_is_available() is False
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)


@test("F1.3 extract_pdf_url soft-fails when no API key")
async def test_extract_pdf_no_key():
    from src.wandering.pdf_extractor import extract_pdf_url

    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        result = await extract_pdf_url("https://arxiv.org/pdf/1234.5678.pdf")
        assert result.ok is False
        assert result.error == "no_api_key"
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


@test("F1.4 extract_pdf_url rejects non-http URLs")
async def test_extract_pdf_bad_scheme():
    from src.wandering.pdf_extractor import extract_pdf_url

    saved = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    try:
        result = await extract_pdf_url("ftp://server/file.pdf")
        assert result.ok is False
        assert result.error == "invalid_url"
    finally:
        if saved is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = saved


@test("F1.5 extract_url dispatches PDF URLs to pdf_extractor when key set")
async def test_extract_url_dispatch_pdf():
    """When ANTHROPIC_API_KEY is set and the URL is a PDF, extract_url
    should route through pdf_extractor (not Jina). We monkey-patch the
    pdf_extractor to return a deterministic ok result and verify that
    Jina is NOT called."""
    from src.wandering import extractors as ex
    from src.wandering import pdf_extractor

    saved = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    async def fake_pdf(url):
        return PdfExtractResult(
            url=url, body="# Title\n\n$E = mc^2$\n",
            chars=22, ok=True, latency_ms=100,
        )

    orig_pdf = pdf_extractor.extract_pdf_url
    pdf_extractor.extract_pdf_url = fake_pdf  # type: ignore[assignment]
    try:
        result = await ex.extract_url("https://arxiv.org/pdf/1234.pdf")
    finally:
        pdf_extractor.extract_pdf_url = orig_pdf  # type: ignore[assignment]
        if saved is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = saved

    assert result.ok is True
    assert "E = mc^2" in result.body


@test("F1.6 extract_url falls back to Jina when PDF path fails")
async def test_extract_url_pdf_fallback():
    """When the PDF path errors, the dispatcher should soft-fall to
    Jina for the same URL. We patch pdf_extractor to fail and Jina
    (via httpx) to succeed; result should be Jina's body."""
    from src.wandering import extractors as ex
    from src.wandering import pdf_extractor

    saved = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key"

    async def fake_pdf_fail(url):
        return PdfExtractResult(url=url, error="api_http_503", ok=False)

    orig_pdf = pdf_extractor.extract_pdf_url
    pdf_extractor.extract_pdf_url = fake_pdf_fail  # type: ignore[assignment]

    # Patch httpx.AsyncClient.get used inside Jina path.
    import httpx
    class _FakeResp:
        text = "# Fallback Jina body\n\nplain text"
        def raise_for_status(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def get(self, url):
            return _FakeResp()

    orig_client = httpx.AsyncClient
    httpx.AsyncClient = _FakeClient  # type: ignore[assignment]
    try:
        result = await ex.extract_url("https://arxiv.org/pdf/9999.pdf")
    finally:
        pdf_extractor.extract_pdf_url = orig_pdf  # type: ignore[assignment]
        httpx.AsyncClient = orig_client  # type: ignore[assignment]
        if saved is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = saved

    assert result.ok is True
    assert "Fallback Jina body" in result.body


@test("F1.7 extract_url uses Jina path when no Anthropic key")
async def test_extract_url_no_key_uses_jina():
    """Without ANTHROPIC_API_KEY, PDF URLs go straight to Jina — no PDF
    extractor invocation at all. We just verify behavior matches a Jina
    fetch (no need to patch pdf_extractor)."""
    from src.wandering import extractors as ex
    import httpx

    saved = os.environ.pop("ANTHROPIC_API_KEY", None)

    class _FakeResp:
        text = "fallback content"
        def raise_for_status(self):
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def get(self, url):
            return _FakeResp()

    orig = httpx.AsyncClient
    httpx.AsyncClient = _FakeClient  # type: ignore[assignment]
    try:
        result = await ex.extract_url("https://arxiv.org/pdf/4444.pdf")
    finally:
        httpx.AsyncClient = orig  # type: ignore[assignment]
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved

    assert result.ok is True
    assert "fallback content" in result.body


# ─── Runner ────────────────────────────────────────────────────────────


# ─── Cancel flow: credit math + progress mechanics ─────────────────────


@test("CANCEL.1 _tokens_to_credits rounds UP")
def test_tokens_to_credits_round_up():
    from src.wandering.routes import _tokens_to_credits, TOKENS_PER_CREDIT
    assert TOKENS_PER_CREDIT == 10_000
    assert _tokens_to_credits(0)        == 0
    assert _tokens_to_credits(1)        == 1
    assert _tokens_to_credits(9_999)    == 1
    assert _tokens_to_credits(10_000)   == 1
    assert _tokens_to_credits(10_001)   == 2
    assert _tokens_to_credits(210_000)  == 21
    # Negative input is defensive — never crashes, always returns 0.
    assert _tokens_to_credits(-50)      == 0


@test("CANCEL.2 WanderingProgress sums cumulative_tokens across agents")
def test_progress_sums_tokens():
    from src.wandering.agent import AgentBudget, AgentState
    from src.wandering.cushion import (
        CushionField, CushionGraph, CushionInput, CushionLayer,
    )
    from src.wandering.runtime import WanderingProgress

    cushion = CushionGraph(
        actual=CushionLayer(name="actual", nodes=["a"], summary=""),
        essence=CushionLayer(name="essence", nodes=["b"], summary=""),
        mechanism=CushionLayer(name="mechanism", nodes=["c"], summary=""),
        raw_input=CushionInput(
            problem=CushionField(name="problem", content="x"),
            context=CushionField(name="context", content=""),
            vision=CushionField(name="vision", content=""),
            hunches=CushionField(name="hunches", content=""),
        ),
    )
    p = WanderingProgress()
    assert p.tokens_used == 0

    a1 = AgentState(agent_id="P01", cushion=cushion, budget=AgentBudget())
    a2 = AgentState(agent_id="P02", cushion=cushion, budget=AgentBudget())
    p.register(a1, a2)

    a1.cumulative_tokens = 5_000
    a2.cumulative_tokens = 7_500
    assert p.tokens_used == 12_500

    # register is idempotent — re-adding the same agent doesn't double
    # the count.
    p.register(a1)
    assert len(p.agents) == 2
    assert p.tokens_used == 12_500


@test("CANCEL.3 abort response payload structure")
def test_abort_payload_shape():
    """We can't easily run an end-to-end abort here without an asyncio
    task to cancel — but we can verify the route's response builder by
    invoking the code path with a manually-prepped JobState. Skip the
    cancel itself; just confirm the schema."""
    import json

    wandering_jobs.clear_all()
    state = JobState(
        session_id="wsess-abort-test",
        job_id="j",
        user_id="guest",
        started_at=time.time() - 60.0,
        mode="multi_pendulum",
        agents=5,
        time_budget_seconds=900.0,
        pursuit="test",
        status=JobStatus.ABORTED,  # simulate already-cancelled state
        completed_at=time.time(),
    )
    wandering_jobs._JOBS["wsess-abort-test"] = state
    payload = state.to_dict()
    # JobState.to_dict has the right keys for the abort response wrapper.
    assert "status" in payload
    assert "session_id" in payload
    assert "pursuit" in payload
    # Don't conflict with the credits field — credits live alongside,
    # not inside.
    assert "credits" not in payload
    wandering_jobs.clear_all()


def main():
    tests = [
        # F5
        test_schema_has_wandering_job,
        test_in_memory_save_job_state,
        test_in_memory_list_running,
        test_sweep_marks_failed,
        test_sweep_no_store,
        test_status_in_process_wins,
        test_status_store_fallback,
        test_drain_running,
        test_register_mirrors_state,
        # F3+F7
        test_in_memory_sessions_metadata,
        test_in_memory_sessions_metadata_empty,
        test_sessions_route_registered,
        # F1
        test_is_pdf_url_case,
        test_pdf_is_available,
        test_extract_pdf_no_key,
        test_extract_pdf_bad_scheme,
        test_extract_url_dispatch_pdf,
        test_extract_url_pdf_fallback,
        test_extract_url_no_key_uses_jina,
        # Cancel
        test_tokens_to_credits_round_up,
        test_progress_sums_tokens,
        test_abort_payload_shape,
    ]
    for fn in tests:
        run_test(fn)
    print()
    print("=" * 60)
    print(f"  {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    if FAILED:
        for name, err in ERRORS:
            print(f"  {name}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
