"""
ThreadPersistence — the wire from /api/v2/trace into the memory pipeline.

OWNS THREE COLLABORATORS AS PROCESS SINGLETONS
==============================================
    1. ThreadStore           — KV/index storage (FalkorThreadStore in prod)
    2. GeminiEmbeddingService — produces iteration embeddings
    3. IterationMetadataExtractor — extracts entities/tags/user_mode/etc

These are built lazily on first use. Init is idempotent. If the Redis
client isn't available, falls back to InMemoryThreadStore (so dev without
a running container still works).

PUBLIC ENTRY POINTS
===================
    record_iteration(...)      — fire-and-forget persistence after a trace
    get_router()               — FastAPI APIRouter with 4 endpoints
    get_thread_store()         — direct access for tests / advanced callers

EVERY ENTRY POINT IS DEFENSIVE
==============================
Persistence failures NEVER raise into the caller. The dispatcher hook
schedules persistence as a background task; if it crashes, the user's
trace response is unaffected. Read endpoints return 404 / empty list
rather than 500 on any error.

WORKSPACE TAGS
==============
workspace_id is a free-form string today: typical values are
"claude" | "cursor" | "codex" | "antigravity" | "aider" | "web" | None.
Free-form keeps us open to new IDEs without schema changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from src.auth.supabase_auth import get_effective_user_id

from src.bridge.thread_store import (
    FalkorThreadStore, InMemoryThreadStore, ThreadStore,
    build_thread_store_from_env, init_store_schema,
)
from src.bridge.embedding_service import GeminiEmbeddingService
from src.bridge.iteration_metadata import IterationMetadataExtractor
from src.core.thread_types import (
    ThreadRecord, IterationRecord, SegmentedResponse, Segment, OpinionSegment,
    ProspectsSegment, ProspectBranch, MapRoomArtifacts, VisualBlock,
    TriageSnapshot, BudgetSnapshot, ProvenanceRecord, ModelCall,
    OutcomeRecord, MemoryContext, Entity,
    DEFAULT_UNCERTAINTY_DISCLAIMER, SCHEMA_VERSION,
)

log = logging.getLogger("constellax.thread_persistence")


# ─── Singletons (lazy-initialized) ────────────────────────────────────

_store: ThreadStore | None = None
_embedder: GeminiEmbeddingService | None = None
_extractor: IterationMetadataExtractor | None = None
_init_lock = asyncio.Lock()


async def _ensure_initialized() -> tuple[ThreadStore, GeminiEmbeddingService, IterationMetadataExtractor]:
    global _store, _embedder, _extractor
    if _store is not None:
        return _store, _embedder, _extractor    # type: ignore[return-value]
    async with _init_lock:
        if _store is not None:
            return _store, _embedder, _extractor    # type: ignore[return-value]
        backed = build_thread_store_from_env()
        if backed is not None:
            _store = backed
            log.info("ThreadPersistence: %s active", type(backed).__name__)
            # No-op for Falkor; runs Cypher DDL for Neo4j (constraints + vector index).
            await init_store_schema(_store)
        else:
            _store = InMemoryThreadStore()
            log.warning(
                "ThreadPersistence: in-memory fallback — neither CONSTELLAX_DB_BACKEND=neo4j "
                "(NEO4J_URI/NEO4J_PASSWORD) nor CONSTELLAX_REDIS_URL is configured"
            )
        _embedder = GeminiEmbeddingService()
        _extractor = IterationMetadataExtractor()
    return _store, _embedder, _extractor


def get_thread_store() -> ThreadStore | None:
    """Synchronous accessor for tests / inspection. May return None if
    nothing has been recorded yet (singletons not initialized)."""
    return _store


# ─── Public: record_iteration (fire-and-forget) ───────────────────────

def schedule_record_iteration(
    *,
    request_id: str,
    question: str,
    dispatch_result: Any,
    triage: dict | None = None,
    cached_memo: dict | None = None,
    user_id: str | None = None,
    project_id: str | None = None,
    workspace_id: str | None = None,
    surface_id: str | None = None,
    parent_thread_id: str | None = None,
    effort_picked: str = "medium",
    started_at: float | None = None,
    web_search_meta: dict | None = None,
    title_question: str | None = None,
    explicit_thread_id: str | None = None,
) -> None:
    """Schedule iteration persistence as a background task.

    Never blocks the caller. Never raises. If anything inside the persistence
    task fails (embedding service down, network error, malformed memo,
    Redis hiccup) it's logged and swallowed.

    `explicit_thread_id` is used by the segmented streaming flow: the
    synth phase commits to a thread_id (memo_id) that the frontend
    stores, but the engine runs in a LATER phase with a different
    request_id. Without an explicit pin, `_resolve_thread_id` would
    derive `thr-{opinion_request_id}` which doesn't match the
    frontend's stored ID — Reasoning Trace 404s. Pass the synth's
    memo_id here and the iteration is stored under the right key.
    """
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_record_iteration_async(
            request_id=request_id, question=question, dispatch_result=dispatch_result,
            triage=triage, cached_memo=cached_memo,
            user_id=user_id, project_id=project_id,
            workspace_id=workspace_id, surface_id=surface_id,
            parent_thread_id=parent_thread_id,
            effort_picked=effort_picked, started_at=started_at,
            web_search_meta=web_search_meta,
            title_question=title_question,
            explicit_thread_id=explicit_thread_id,
        ))
    except Exception as e:
        log.warning("schedule_record_iteration failed to enqueue: %s", e)


async def _record_iteration_async(
    *,
    request_id: str,
    question: str,
    dispatch_result: Any,
    triage: dict | None,
    cached_memo: dict | None,
    user_id: str | None,
    project_id: str | None,
    workspace_id: str | None,
    surface_id: str | None,
    parent_thread_id: str | None,
    effort_picked: str,
    started_at: float | None,
    web_search_meta: dict | None = None,
    title_question: str | None = None,
    explicit_thread_id: str | None = None,
) -> None:
    """Build an IterationRecord from the dispatch result and persist it,
    plus update the parent thread. Best-effort end to end."""
    try:
        store, embedder, extractor = await _ensure_initialized()
    except Exception as e:
        log.warning("persistence init failed: %s", e)
        return

    try:
        completed_at = time.time()
        if explicit_thread_id:
            # Caller has already committed to a thread_id (the synth
            # phase's memo_id). Pin to it. If a thread already exists
            # under this id, append; otherwise the upsert below creates
            # it. This is the pin the segmented flow relies on.
            try:
                existing = await store.get_thread(explicit_thread_id)
                thread_id, is_new_thread = explicit_thread_id, existing is None
            except Exception as e:
                log.warning(
                    "explicit_thread_id lookup failed for %s: %s — treating as new",
                    explicit_thread_id, e,
                )
                thread_id, is_new_thread = explicit_thread_id, True
        else:
            thread_id, is_new_thread = await _resolve_thread_id(
                store, request_id, parent_thread_id,
            )

        # Sequence number: 1 if new thread, otherwise (existing iteration_count + 1).
        # Read straight from the authoritative list to avoid drift if the
        # ThreadRecord was updated concurrently.
        seq_num = 1
        if not is_new_thread:
            try:
                existing_iters = await store.list_iterations_for_thread(thread_id)
                seq_num = len(existing_iters) + 1
            except Exception:
                seq_num = 1

        # Build the SegmentedResponse from the existing memo shape.
        # During the transitional phase, all three segments are derived from
        # the single memo emitted by speech.py — they're effectively staged
        # views of the same content. Once segment streaming ships, this
        # function fans out into three separate writes per segment.
        memo = _safe_attr(dispatch_result, "memo") or cached_memo or {}
        response = _build_segmented_response(memo, dispatch_result)

        iter_id = f"itr-{request_id}"
        # API-boundary defaults for the new Phase 1 provenance fields. When
        # the caller (browser / IDE / mobile) doesn't supply them, default
        # workspace_id="web" + surface_id="chat" — the safe assumption for
        # any traffic that reached the trace endpoint without an explicit
        # workspace tag. The fields end up on the Iteration node as
        # indexed properties, so the sweeper and retrievers find them
        # directly without parsing payload_json.
        effective_workspace = workspace_id or "web"
        effective_surface = surface_id or "chat"
        iteration = IterationRecord(
            id=iter_id, thread_id=thread_id, sequence_num=seq_num,
            workspace_id=effective_workspace,
            surface_id=effective_surface,
            question=question,
            effort_picked=effort_picked,
            triage=_build_triage_snapshot(triage),
            engine=_extract_engine_artifacts(dispatch_result),
            response=response,
            budget=_build_budget_snapshot(dispatch_result),
            provenance=_build_provenance(dispatch_result),
            status="done",
            created_at=started_at or completed_at,
            completed_at=completed_at,
            memory_context=MemoryContext(),
        )

        # Stash the search footprint on the iteration's meta escape
        # hatch. Lives at iteration.meta["web_search"] and round-trips
        # through to_payload/from_payload without a schema change.
        # Reasoning Trace pulls it from /api/v2/thread/<id>/full.
        if web_search_meta:
            iteration.meta["web_search"] = web_search_meta

        # Memory-layer enrichment: embedding + metadata extraction.
        # Both calls run in parallel, both are best-effort, both can fail
        # without breaking the iteration save.
        synth_text = response.synthesizer.text if response.synthesizer else ""
        emb_task = asyncio.create_task(embedder.embed(f"{question}\n\n{synth_text}"))
        meta_task = asyncio.create_task(extractor.extract(question, synth_text))
        emb_res, meta_res = await asyncio.gather(emb_task, meta_task, return_exceptions=True)

        if isinstance(emb_res, Exception):
            log.warning("embedding raised: %s", emb_res)
        elif emb_res is not None and emb_res.success:
            iteration.embedding = emb_res.vector
            iteration.embedding_model = emb_res.model
            iteration.provenance.model_calls.append(ModelCall(
                purpose="embedding", model=emb_res.model, backend="gemini",
                tokens_in=emb_res.tokens, tokens_out=0,
                latency_ms=emb_res.latency_ms,
            ))

        if isinstance(meta_res, Exception):
            log.warning("metadata extraction raised: %s", meta_res)
        elif meta_res is not None and meta_res.success:
            iteration.entities = meta_res.entities
            iteration.tags = meta_res.tags
            iteration.domains = meta_res.domains
            iteration.user_mode = meta_res.user_mode
            iteration.time_horizon = meta_res.time_horizon
            # Promote load_bearing_assumption to the response if not already set
            if response.load_bearing_assumption is None and meta_res.load_bearing_assumption:
                response.load_bearing_assumption = meta_res.load_bearing_assumption
            iteration.provenance.model_calls.append(ModelCall(
                purpose="metadata_extraction", model=meta_res.model, backend="gemini",
                tokens_in=meta_res.tokens_in, tokens_out=meta_res.tokens_out,
                cost_usd=meta_res.cost_usd, latency_ms=meta_res.latency_ms,
            ))

        # Persist
        try:
            await store.save_iteration(iteration)
        except Exception as e:
            log.warning("save_iteration failed for %s: %s", iter_id, e)
            return

        # Build / update the parent thread
        try:
            existing_thread = await store.get_thread(thread_id) if not is_new_thread else None
            # Prefer the user's literal question for the title — falls
            # back to the augmented `question` only when title_question
            # wasn't passed (legacy callers).
            title_source = title_question if title_question else question
            thread = existing_thread or ThreadRecord(
                id=thread_id, user_id=user_id, project_id=project_id, workspace_id=workspace_id,
                title=_derive_title(title_source), created_at=started_at or completed_at,
                updated_at=completed_at, status="active",
            )
            thread.iteration_ids = list(set(thread.iteration_ids + [iter_id]))
            thread.iteration_count = len(thread.iteration_ids)
            thread.last_route = _safe_attr(dispatch_result, "route")
            thread.last_confidence = response.overall_confidence
            # Union new memory signals into the thread's aggregates
            thread.all_entities = sorted(set(thread.all_entities + [e.name for e in iteration.entities]))
            thread.all_tags = sorted(set(thread.all_tags + iteration.tags))
            thread.all_domains = sorted(set(thread.all_domains + iteration.domains))
            thread.updated_at = completed_at

            # Cost / time / perspective rollups — accumulate this iteration
            # into the thread's totals. Every figure defensively coerces to
            # 0 so a partial dispatch result never corrupts the running sum.
            iter_time_ms = 0
            if iteration.created_at and iteration.completed_at:
                iter_time_ms = max(0, int((iteration.completed_at - iteration.created_at) * 1000))
            iter_cost = sum(
                float(getattr(mc, "cost_usd", 0.0) or 0.0)
                for mc in iteration.provenance.model_calls
            )
            iter_perspectives = len(_safe_attr(dispatch_result, "perspectives") or [])
            thread.aggregate_time_ms  = (thread.aggregate_time_ms or 0) + iter_time_ms
            thread.aggregate_cost_usd = float(thread.aggregate_cost_usd or 0.0) + iter_cost
            thread.perspectives_run   = (thread.perspectives_run or 0) + iter_perspectives

            await store.save_thread(thread)
        except Exception as e:
            log.warning("thread upsert failed for %s: %s", thread_id, e)
    except Exception as e:
        # Outer guard — must never escape this function
        log.exception("record_iteration crashed unexpectedly: %s", e)


# ─── FastAPI router (4 read endpoints + 1 outcome write) ──────────────

def get_router() -> APIRouter:
    """Returns a fresh APIRouter with the thread/iteration endpoints.
    server.py calls app.include_router(get_router()) at startup."""
    router = APIRouter()

    @router.get("/api/v2/threads")
    async def list_threads_endpoint(
        request: Request,
        user_id: str | None = Query(None, alias="user"),
        project_id: str | None = Query(None, alias="project"),
        workspace_id: str | None = Query(None, alias="workspace"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        # Phase 3C: when the request carries a verified Supabase JWT, the
        # verified `sub` claim trumps the `?user=` query param. This closes
        # the race where the frontend caches the legacy localStorage UUID
        # and sends it as ?user= even after sign-in — the backend now
        # always scopes the query to the verified identity instead.
        # Anonymous requests still use the query param value, exactly as
        # before (Phase 3A non-breaking contract).
        effective_user_id = get_effective_user_id(request, user_id)

        store = await _maybe_store()
        if store is None:
            return JSONResponse({"threads": [], "warning": "store not initialized"}, status_code=200)
        try:
            threads = await store.list_threads(
                user_id=effective_user_id,
                project_id=project_id, workspace_id=workspace_id,
                limit=limit, offset=offset,
            )
            return {"threads": [t.to_payload() for t in threads], "count": len(threads)}
        except Exception as e:
            log.warning("list_threads failed: %s", e)
            return JSONResponse({"threads": [], "error": str(e)}, status_code=200)

    @router.get("/api/v2/thread/{thread_id}/full")
    async def get_thread_full_endpoint(thread_id: str):
        """Full thread + all its iterations. Used by the thread page."""
        store = await _maybe_store()
        if store is None:
            raise HTTPException(404, "store not initialized")
        thread = await store.get_thread(thread_id)
        if not thread:
            raise HTTPException(404, f"thread {thread_id} not found")
        iters = await store.list_iterations_for_thread(thread_id)
        return {
            "thread": thread.to_payload(),
            "iterations": [it.to_payload() for it in iters],
        }

    @router.delete("/api/v2/thread/{thread_id}")
    async def delete_thread_endpoint(thread_id: str):
        """Delete a thread and every iteration it owns. Cascades through
        the store's authoritative iteration list so leftover index
        entries get cleaned up too."""
        store = await _maybe_store()
        if store is None:
            raise HTTPException(503, "store not initialized")
        try:
            ok = await store.delete_thread(thread_id)
        except Exception as e:
            log.warning("delete_thread failed for %s: %s", thread_id, e)
            raise HTTPException(500, f"delete failed: {e}")
        if not ok:
            raise HTTPException(404, f"thread {thread_id} not found")
        return {"ok": True, "thread_id": thread_id}

    @router.get("/api/v2/iteration/{iter_id}")
    async def get_iteration_endpoint(iter_id: str):
        """One iteration — used by the Map Room when opened in a fresh tab."""
        store = await _maybe_store()
        if store is None:
            raise HTTPException(404, "store not initialized")
        it = await store.get_iteration(iter_id)
        if not it:
            raise HTTPException(404, f"iteration {iter_id} not found")
        return it.to_payload()

    @router.post("/api/v2/iteration/{iter_id}/outcome")
    async def post_iteration_outcome(iter_id: str, payload: dict):
        """Record user's report of how the recommendation actually played out."""
        store = await _maybe_store()
        if store is None:
            raise HTTPException(503, "store not initialized")
        outcome = OutcomeRecord(
            reported_at=time.time(),
            outcome_text=str(payload.get("outcome_text", "")).strip(),
            followed_advice=payload.get("followed_advice"),
            accuracy=payload.get("accuracy"),
            surprise_factor=payload.get("surprise_factor"),
            meta=payload.get("meta") or {},
        )
        if not outcome.outcome_text:
            raise HTTPException(400, "outcome_text is required")
        ok = await store.attach_outcome(iter_id, outcome)
        if not ok:
            raise HTTPException(404, f"iteration {iter_id} not found")
        return {"ok": True, "iter_id": iter_id}

    @router.get("/api/v2/threads/similar")
    async def find_similar_threads(
        iter_id: str = Query(...),
        k: int = Query(5, ge=1, le=20),
    ):
        """Find iterations similar to the given one via embedding cosine
        similarity. The pattern-matching primitive — used by both the
        reasoning engine (for memory recall during a trace) and the UI
        (for 'related threads' panel)."""
        store = await _maybe_store()
        if store is None:
            return JSONResponse({"matches": []}, status_code=200)
        base = await store.get_iteration(iter_id)
        if not base or base.embedding is None:
            return {"matches": [], "warning": "iteration missing embedding"}
        try:
            results = await store.find_similar_iterations(base.embedding, k=k, exclude_iter_id=iter_id)
            return {"matches": [
                {"score": s, "iteration": it.to_payload()} for s, it in results
            ]}
        except Exception as e:
            log.warning("find_similar failed: %s", e)
            return JSONResponse({"matches": [], "error": str(e)}, status_code=200)

    return router


async def _maybe_store() -> ThreadStore | None:
    """Endpoint-safe accessor. Returns None on init failure rather than raising."""
    try:
        store, _, _ = await _ensure_initialized()
        return store
    except Exception as e:
        log.warning("store unavailable: %s", e)
        return None


# ─── Helpers (kept private — used only inside this module) ────────────

def _safe_attr(obj: Any, name: str, default=None):
    """Defensive getattr that works on dataclasses, dicts, and None."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _derive_title(question: str) -> str:
    """First 60 chars or first sentence, whichever is shorter."""
    q = (question or "").strip()
    if not q:
        return "Untitled thread"
    cutoff = q.find(".")
    if 0 < cutoff < 60:
        return q[:cutoff].strip()
    return q[:60].strip() + ("…" if len(q) > 60 else "")


async def _resolve_thread_id(
    store: ThreadStore,
    request_id: str,
    parent_thread_id: str | None,
) -> tuple[str, bool]:
    """Decide whether this trace continues an existing thread or starts a new one.

    Frontend sends `parent_thread_id` on followups. If it resolves to a known
    thread, append to it (is_new=False). Otherwise derive a fresh thread_id
    from the request_id — first turn in a new conversation."""
    if parent_thread_id:
        try:
            existing = await store.get_thread(parent_thread_id)
            if existing is not None:
                return parent_thread_id, False
        except Exception as e:
            log.warning("parent_thread_id lookup failed for %s: %s", parent_thread_id, e)
    return f"thr-{request_id}", True


def _build_triage_snapshot(triage: dict | None) -> TriageSnapshot | None:
    if not triage:
        return None
    return TriageSnapshot(
        route=triage.get("route", "deep"),
        recommended_effort=triage.get("recommended_effort", "medium"),
        risk_flags=list(triage.get("risk_flags") or []),
        why=triage.get("why", ""),
        classifier_mode=triage.get("classifier_mode", ""),
    )


def _build_budget_snapshot(dispatch_result: Any) -> BudgetSnapshot:
    b = _safe_attr(dispatch_result, "budget_summary") or {}
    return BudgetSnapshot(
        iterations=int(b.get("iterations", 0) or 0),
        wall_time_sec=float(b.get("wall_time_sec", 0.0) or 0.0),
        cost_usd=float(b.get("cost_usd", 0.0) or 0.0),
        mcp_calls=int(b.get("mcp_calls", 0) or 0),
        breached=bool(b.get("breached", False)),
        breach_reason=str(b.get("breach_reason", "") or ""),
    )


def _build_provenance(dispatch_result: Any) -> ProvenanceRecord:
    capability_state = _safe_attr(dispatch_result, "capability_state") or {}
    return ProvenanceRecord(
        model_calls=[],
        graphify_queries=[],
        mcps_invoked=[],
        capability_state=dict(capability_state) if isinstance(capability_state, dict) else {},
        backend_versions={"schema": str(SCHEMA_VERSION)},
    )


def _extract_engine_artifacts(dispatch_result: Any) -> dict:
    """Pull the engine's analytical findings into a plain dict for storage.
    We don't try to deserialize Variable/Perspective trees here — the dict
    form preserves everything for replay without coupling the store to
    engine internals."""
    er = _safe_attr(dispatch_result, "engine_result")
    if er is None:
        return {}
    return {
        "actual_iterations": _safe_attr(er, "actual_iterations"),
        "actual_active_domains": list(_safe_attr(er, "actual_active_domains") or []),
        # Variables/Perspectives are large — store summaries only at this layer
        "had_perspectives": bool(_safe_attr(er, "perspectives")),
    }


def _build_segmented_response(memo: dict, dispatch_result: Any) -> SegmentedResponse:
    """Map the existing flat Memo shape into the new 3-segment shape.

    PHASE 1 — staged from single memo:
      synthesizer = verdict_line + verdict_body
      opinion     = reasoning items joined into peer prose
      prospects   = falsifiers + open_questions converted into conditional branches

    PHASE 2 (segment streaming) — three separate LLM calls fill these
    fields independently as they complete. The shape is already correct
    for that flow; only the producer changes."""
    response_text = _safe_attr(dispatch_result, "response_text") or ""
    confidence = memo.get("confidence", "moderate") if isinstance(memo, dict) else "moderate"

    if not memo:
        # Non-deep route (trivial/direct) — synthesize a one-segment response.
        return SegmentedResponse(
            overall_confidence=confidence,
            synthesizer=Segment(text=response_text, confidence=confidence, delivered_at=time.time()),
            opinion=None,
            prospects=None,
            map_room=MapRoomArtifacts(),
            user_interjections=[],
            memo=None,
        )

    verdict_line = memo.get("verdict_line", "")
    verdict_body = memo.get("verdict_body", "")
    synth_text = "\n\n".join(p for p in [verdict_line, verdict_body] if p) or response_text

    reasoning = memo.get("reasoning") or []
    alternatives = memo.get("alternatives") or []
    opinion_paragraphs = []
    for r in reasoning:
        if isinstance(r, dict):
            opinion_paragraphs.append(f"**{r.get('title','')}** — {r.get('body','')}")
    if alternatives:
        alts = "; ".join(f"{a.get('tag','')}: {a.get('body','')}" for a in alternatives if isinstance(a, dict))
        opinion_paragraphs.append(f"\nAlternatives considered: {alts}")
    opinion_text = "\n\n".join(opinion_paragraphs)

    # Prospects from falsifiers + open_questions, in branch shape
    branches: list[ProspectBranch] = []
    for f in (memo.get("falsifiers") or [])[:5]:
        if not isinstance(f, dict): continue
        q = f.get("question", ""); a = f.get("answer", "")
        if q and a:
            branches.append(ProspectBranch(
                condition=q, outcome=a, confidence=confidence,
            ))
    prospects_text = (
        "Likely conditional paths — each marked with confidence. "
        "These are insights, not predictions."
    ) if branches else ""

    return SegmentedResponse(
        overall_confidence=confidence,
        synthesizer=Segment(text=synth_text, confidence=confidence, delivered_at=time.time()),
        opinion=OpinionSegment(
            text=opinion_text,
            peer_commentary="",  # filled when segment streaming lands
            confidence=confidence,
            delivered_at=time.time(),
        ) if opinion_text else None,
        prospects=ProspectsSegment(
            text=prospects_text,
            branches=branches,
            uncertainty_disclaimer=DEFAULT_UNCERTAINTY_DISCLAIMER,
            confidence=confidence,
            delivered_at=time.time(),
        ) if branches else None,
        map_room=MapRoomArtifacts(
            visuals=[
                VisualBlock(kind=v.get("type"), title=v.get("title"), spec=v.get("spec"), meta={})
                for v in (memo.get("visuals") or []) if isinstance(v, dict)
            ],
        ),
        user_interjections=[],
        # Carry the raw memo dict (including visuals) verbatim so the
        # Map Room can rehydrate structured fields on a fresh-tab fetch
        # via /api/v2/thread/{id}/full. Without this, the persistence
        # boundary strips structure and the Map Room falls back to
        # parsing synthesizer prose — losing visuals entirely.
        memo=memo if isinstance(memo, dict) else None,
    )
