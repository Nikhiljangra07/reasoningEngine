"""
API endpoints for Wandering Room.

FOUR endpoints under /api/v2/wandering/:

  POST /brief
    body: { problem, context, vision, current_map, user_id?, auto_enrich? }
    response: { cushion: <CushionGraph dict>, brief_ok: bool, warnings: [...] }
    purpose: turn user's four-field intake into the three-layer cushion.
             Used at session start; returns the cushion the user reviews
             before launching wandering.

  POST /session
    body: { cushion, mode, agents?, time_seconds?, tokens_per_agent?,
            model_mix?, user_id? }
    response: { session_id, dossier: <Dossier dict>, summary: {...} }
    purpose: run a Wandering Room session end-to-end. Returns the final
             dossier ready for rendering. Synchronous in V1 (frontend
             shows progress via streaming polish later).

  GET /session/{session_id}
    response: { session_id, exists: bool, dossier: <Dossier dict> | null }
    purpose: re-fetch a stored session's dossier (post-session view).

  POST /session/{session_id}/dig-deeper
    body: { report_id, focus?, tokens? }
    response: { subagent_id, reports: [...], dossier_delta: {...} }
    purpose: user clicks "dig deeper" on one report card. Spawns a
             sub-agent that inherits the cushion + focuses on that report's
             matched layers. Returns the new reports for the dossier.

DEFENSIVE: all endpoints handle the "no auth, no Neo4j, no user_id" path
gracefully. Guest users get in-memory storage (lost on restart);
authenticated users get Neo4j persistence (if configured).

Per Law 4: read-only on user state. We write ONLY to the wandering
namespace. The user's project memory and IDE state are never touched.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.auth.supabase_auth import get_effective_user_id
from src.llm.client import LLMClient, ClientMode
from src.wandering import jobs
from src.wandering.composer import compose_cushion
from src.wandering.cushion import (
    CushionField,
    CushionInput,
    CushionGraph,
    CushionLayer,
    SkipReason,
)
from src.wandering.dossier import Dossier, build_dossier
from src.wandering.fetcher import web_search_fetcher
from src.wandering.map_adapter import session_to_memo
from src.wandering.persistence import (
    WanderingStore,
    build_wandering_store_from_env,
)
from src.wandering.report import Confidence
from src.wandering.runtime import (
    SessionResult,
    WanderingConfig,
    WanderingMode,
    WanderingProgress,
    run_wandering_session,
)
from src.wandering.credits import (
    CreditService,
    CreditTxKind,
    InsufficientCredits,
    Reservation,
    get_credit_service,
    tokens_to_credits,
)
from src.wandering.subagent import (
    run_subagent,
    should_spawn,
    spawn_request_from_user_dig_deeper,
)


log = logging.getLogger("constellax.wandering.routes")


# ---------------------------------------------------------------------------
# Process-singleton store + client
# ---------------------------------------------------------------------------

_STORE: WanderingStore | None = None
_LLM_CLIENT: LLMClient | None = None


def get_store() -> WanderingStore:
    """Lazy build of the WanderingStore. Idempotent."""
    global _STORE
    if _STORE is None:
        _STORE = build_wandering_store_from_env()
    return _STORE


def get_llm_client() -> LLMClient:
    """Lazy build of LLMClient. Uses LIVE mode when OPENROUTER_API_KEY is set;
    falls back to MOCK in tests / when keys are missing."""
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        import os
        mode = ClientMode.LIVE if os.environ.get("OPENROUTER_API_KEY", "").strip() else ClientMode.MOCK
        _LLM_CLIENT = LLMClient(mode=mode)
    return _LLM_CLIENT


# Override hooks for tests (set these before calling routes in test code).
def _set_store(store: WanderingStore) -> None:
    """Test helper: inject a specific store."""
    global _STORE
    _STORE = store


def _set_llm_client(client: LLMClient) -> None:
    """Test helper: inject a specific LLM client."""
    global _LLM_CLIENT
    _LLM_CLIENT = client


# In-memory cache of recent cushions for the dig-deeper path. Wandering
# sessions store the cushion in the SessionResult; for dig-deeper we need
# to find the cushion quickly given a session_id. The store's
# get_session() is V1-incomplete (Neo4j read-back is lossy), so we keep
# a session_id → cushion mapping in memory. Lost on restart; users who
# want to dig deeper after restart can re-run their original session.
_CUSHION_CACHE: dict[str, CushionGraph] = {}
_SESSION_CACHE: dict[str, SessionResult] = {}
# Dossier is built once per job at completion time and cached here so
# repeat polls of GET /session/{id} (and GET /session/{id}/memo) don't
# re-run build_dossier — which articulates every report via LLM and is
# the most expensive read on the wandering path.
_DOSSIER_CACHE: dict[str, Dossier] = {}
# Live progress handles for in-flight wanders. Populated by the worker
# right before run_wandering_session() spawns agents; agents register
# themselves into the handle as they're created. The abort route reads
# from here to compute an HONEST credit-refund based on real tokens
# spent at the moment of cancel — not a time-ratio estimate.
# Entries are cleaned up in the worker's `finally`.
_LIVE_PROGRESS: dict[str, WanderingProgress] = {}

# Credit display unit. The canonical TOKENS_PER_CREDIT and conversion
# function live in src/wandering/credits.py — they are imported here so
# the wander lifecycle and the credit ledger never disagree on what a
# credit is worth in tokens. The local names are kept as aliases for
# backwards compatibility with anything that already imports them from
# this module.
from src.wandering.credits import TOKENS_PER_CREDIT  # noqa: F401


def _tokens_to_credits(tokens: int) -> int:
    """Alias for credits.tokens_to_credits. Keeps the old call sites
    inside routes.py working without churn."""
    return tokens_to_credits(tokens)


def _derived_credit_payload(
    state: Any, live_tokens: int,
) -> dict[str, Any]:
    """Best-effort credit breakdown used when the live reservation isn't
    available — guest wanders (no user_id), zero-budget configs, or
    the rare race where the reservation closed before /abort could
    commit it. Computes budget from MODE_DEFAULTS, used from live
    tokens, refunded as the clamped difference.

    This path does NOT touch the credit ledger — it's display-only.
    Real settlement happens via CreditService.commit/release on the
    primary path."""
    budgeted_credits = 0
    used_credits     = tokens_to_credits(live_tokens)
    try:
        mode = WanderingMode(state.mode)
        from src.wandering.runtime import MODE_DEFAULTS
        defaults = MODE_DEFAULTS[mode]
        budgeted_tokens  = defaults.tokens_per_agent * state.agents
        budgeted_credits = tokens_to_credits(budgeted_tokens)
    except (ValueError, KeyError, AttributeError):
        pass
    refunded_credits = max(0, budgeted_credits - used_credits)
    return {
        "budgeted":          budgeted_credits,
        "used":              used_credits,
        "refunded":          refunded_credits,
        "unit":              "credit",
        "tokens_per_credit": TOKENS_PER_CREDIT,
    }


# Map a wander's session_id to its open Reservation so the /abort and
# worker-completion paths can find the right hold without threading
# the reservation_id through JobState. Same lifecycle as _LIVE_PROGRESS:
# populated when the wander starts, cleaned up in the worker's `finally`
# regardless of how the worker exited.
_RESERVATIONS: dict[str, Reservation] = {}


# ---------------------------------------------------------------------------
# Helpers — parse request bodies into typed objects
# ---------------------------------------------------------------------------


def _build_cushion_input_from_body(body: dict[str, Any]) -> CushionInput:
    """Turn the JSON body of /brief into a CushionInput.

    Required fields are pulled from `problem`, `context`, `vision`,
    `current_map` keys (matching the four-field intake form). A missing
    or whitespace-only value is treated as a skip (SKIPPED_AFTER_PROMPT
    if `skipped_after_prompt` key set on the body's skip_reasons map).
    """
    skip_reasons_raw = body.get("skip_reasons") or {}

    def _field(name: str) -> CushionField:
        content = str(body.get(name, "") or "").strip()
        reason_raw = str(skip_reasons_raw.get(name, "not_skipped") or "not_skipped").strip()
        try:
            reason = SkipReason(reason_raw)
        except ValueError:
            reason = SkipReason.NOT_SKIPPED
        return CushionField(name=name, content=content, skip_reason=reason)

    return CushionInput(
        problem=_field("problem"),
        context=_field("context"),
        vision=_field("vision"),
        current_map=_field("current_map"),
    )


def _cushion_to_response_dict(cushion: CushionGraph) -> dict[str, Any]:
    """Render a cushion for the response payload — user-facing shape."""
    return {
        "actual": {
            "name": cushion.actual.name,
            "nodes": list(cushion.actual.nodes),
            "summary": cushion.actual.summary,
        },
        "essence": {
            "name": cushion.essence.name,
            "nodes": list(cushion.essence.nodes),
            "summary": cushion.essence.summary,
        },
        "mechanism": {
            "name": cushion.mechanism.name,
            "nodes": list(cushion.mechanism.nodes),
            "summary": cushion.mechanism.summary,
        },
        "constellation_size": cushion.constellation_size,
        "extraction_model": cushion.extraction_model,
        "extracted_at": cushion.extracted_at,
        "raw_input": {
            "problem": cushion.raw_input.problem.content,
            "context": cushion.raw_input.context.content,
            "vision": cushion.raw_input.vision.content,
            "current_map": cushion.raw_input.current_map.content,
            "memory_enrichment": cushion.raw_input.memory_enrichment,
        },
    }


def _cushion_from_request_dict(d: dict[str, Any]) -> CushionGraph | None:
    """Hydrate a CushionGraph from a request payload (the inverse of
    _cushion_to_response_dict). Used by /session when the client sends
    back the cushion from a prior /brief call."""
    try:
        raw_input_raw = d.get("raw_input") or {}
        raw_input = CushionInput(
            problem=CushionField(name="problem", content=str(raw_input_raw.get("problem", "")).strip()),
            context=CushionField(name="context", content=str(raw_input_raw.get("context", "")).strip()),
            vision=CushionField(name="vision", content=str(raw_input_raw.get("vision", "")).strip()),
            current_map=CushionField(name="current_map", content=str(raw_input_raw.get("current_map", "")).strip()),
            memory_enrichment=str(raw_input_raw.get("memory_enrichment", "")),
        )
        actual_raw = d.get("actual") or {}
        essence_raw = d.get("essence") or {}
        mech_raw = d.get("mechanism") or {}
        return CushionGraph(
            actual=CushionLayer(
                name="actual",
                nodes=list(actual_raw.get("nodes", [])),
                summary=str(actual_raw.get("summary", "")),
            ),
            essence=CushionLayer(
                name="essence",
                nodes=list(essence_raw.get("nodes", [])),
                summary=str(essence_raw.get("summary", "")),
            ),
            mechanism=CushionLayer(
                name="mechanism",
                nodes=list(mech_raw.get("nodes", [])),
                summary=str(mech_raw.get("summary", "")),
            ),
            raw_input=raw_input,
            constellation_size=int(d.get("constellation_size", 0) or 0),
            extraction_model=str(d.get("extraction_model", "")),
            extracted_at=float(d.get("extracted_at", 0.0) or 0.0),
        )
    except Exception as e:
        log.warning("could not hydrate cushion from request: %s", e)
        return None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Build the Wandering Room APIRouter.

    Mounted into the main FastAPI app by server.py via
    app.include_router(get_router(), prefix='/api/v2/wandering').
    """
    router = APIRouter()

    @router.post("/brief")
    async def post_brief(request: Request) -> JSONResponse:
        """Build the three-layer cushion from a four-field intake."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail={"error": "invalid_json"})

        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail={"error": "body_must_be_object"})

        user_id = get_effective_user_id(request, body.get("user_id"))

        input_data = _build_cushion_input_from_body(body)

        if not input_data.is_minimally_viable():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "brief_too_thin",
                    "detail": "the 'problem' field is required; the cushion needs an anchor",
                },
            )

        try:
            cushion = await compose_cushion(
                input_data=input_data,
                client=get_llm_client(),
                user_id=user_id,
                auto_enrich=bool(body.get("auto_enrich", True)),
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": "brief_invalid", "detail": str(e)})
        except RuntimeError as e:
            log.warning("compose_cushion runtime error: %s", e)
            return JSONResponse(
                status_code=502,
                content={"error": "extraction_failed", "detail": str(e)},
            )

        # Surface warnings for skipped fields (informational).
        warnings = []
        if input_data.context.is_skipped():
            warnings.append("context skipped — cushion will be less dimensionally rich")
        if input_data.vision.is_skipped():
            warnings.append("vision skipped — fewer cross-domain Heisenberg-zone hits expected")
        if input_data.current_map.is_skipped():
            warnings.append("current_map skipped — agents start cold instead of from your partial map")

        return JSONResponse(content={
            "brief_ok": cushion.is_well_formed(),
            "cushion": _cushion_to_response_dict(cushion),
            "warnings": warnings,
            "user_id": user_id,
        })

    @router.post("/session")
    async def post_session(request: Request) -> JSONResponse:
        """Accept a wander job — spawns it in the background and returns
        202 IMMEDIATELY with the session_id + job_id. The browser tab can
        close at any time; the wander keeps running inside this process.

        Lifecycle:
          1. Caller POSTs /session with the cushion + mode (and an optional
             session_id they want to claim).
          2. We validate, build the WanderingConfig, register a JobState,
             and spawn the wander as asyncio.create_task.
          3. We return 202 with {session_id, job_id, status: "running",
             started_at, mode, agents, time_budget_seconds, pursuit}.
          4. The client polls GET /session/<id>/status until status flips
             to "completed" (or "failed" / "aborted"), then GETs
             /session/<id> for the full dossier.
          5. POST /session/<id>/abort cancels the running task; the
             worker catches CancelledError and marks the job aborted.

        We deliberately DON'T await run_wandering_session here — that was
        the bug. A 60-minute synchronous request meant the client never
        learned the session_id when it disconnected, so the "you can
        close the tab and come back" UI promise was a lie.
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail={"error": "invalid_json"})

        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail={"error": "body_must_be_object"})

        user_id = get_effective_user_id(request, body.get("user_id"))

        cushion_raw = body.get("cushion")
        if not isinstance(cushion_raw, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing_cushion",
                    "detail": "send the cushion object from /brief in the 'cushion' field",
                },
            )

        cushion = _cushion_from_request_dict(cushion_raw)
        if cushion is None or not cushion.is_well_formed():
            return JSONResponse(
                status_code=400,
                content={"error": "cushion_malformed"},
            )

        mode_raw = str(body.get("mode", "multi_pendulum")).strip().lower()
        try:
            mode = WanderingMode(mode_raw)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_mode",
                    "valid": [m.value for m in WanderingMode],
                },
            )

        # Build config from optional overrides
        agents_raw = body.get("agents")
        time_seconds_raw = body.get("time_seconds")
        tokens_per_agent_raw = body.get("tokens_per_agent")
        model_mix_raw = body.get("model_mix")
        session_id = body.get("session_id") or f"wsess-{uuid.uuid4().hex[:8]}"
        job_id = f"wjob-{uuid.uuid4().hex[:8]}"

        config = WanderingConfig(
            mode=mode,
            agents=int(agents_raw) if isinstance(agents_raw, int) else None,
            time_budget_seconds=float(time_seconds_raw) if isinstance(time_seconds_raw, (int, float)) else None,
            tokens_per_agent=int(tokens_per_agent_raw) if isinstance(tokens_per_agent_raw, int) else None,
            model_mix=tuple(model_mix_raw) if isinstance(model_mix_raw, list) else None,
            session_id=session_id,
        )

        # Reject a duplicate submission for a session_id that's already
        # running. The caller's correct move is to resume via
        # GET /session/<id>/status against the existing job.
        existing = jobs.get_job(session_id)
        if existing is not None and existing.status == jobs.JobStatus.RUNNING:
            return JSONResponse(
                status_code=409,
                content={
                    "error":       "session_already_running",
                    "detail":      "a wander is already running for this session_id; poll /status to resume",
                    "session_id":  session_id,
                    "job_id":      existing.job_id,
                },
            )

        client = get_llm_client()

        # Snapshot resolved config values for the JobState — the
        # frontend uses these to drive the LiveWandering wait screen
        # even when the user reopened the tab and lost their local draft.
        resolved_agents, resolved_time, resolved_tokens_per_agent, _ = config.resolved()
        pursuit_text = cushion.raw_input.problem.content

        # ── Credit reservation ─────────────────────────────────────────
        # Reserve the budget BEFORE spawning the worker. If the user
        # doesn't have enough credits, we return 402 here and never burn
        # an agent. The reservation handle is stored against the
        # session_id so /abort and worker completion can find it later.
        # Guests (empty user_id) skip the gate — they're transient and
        # don't have credit accounts.
        budgeted_credits = tokens_to_credits(resolved_tokens_per_agent * resolved_agents)
        reservation: Reservation | None = None
        if user_id and budgeted_credits > 0:
            credits = get_credit_service()
            try:
                reservation = await credits.reserve(
                    user_id=user_id,
                    amount=budgeted_credits,
                    ref_id=session_id,
                    note=f"{mode.value} wander budget ({resolved_agents} agents × "
                         f"{resolved_tokens_per_agent:,} tokens)",
                )
            except InsufficientCredits as ic:
                return JSONResponse(
                    status_code=402,
                    content={
                        "error":             "insufficient_credits",
                        "balance":           ic.balance,
                        "needed":            ic.needed,
                        "gap":               ic.needed - ic.balance,
                        "mode":              mode.value,
                        "agents":            resolved_agents,
                        "tokens_per_credit": TOKENS_PER_CREDIT,
                        "detail":            (
                            f"This {mode.value} wander needs {ic.needed} credits; "
                            f"you have {ic.balance}. Top up to continue."
                        ),
                    },
                )

        async def _run_wander() -> None:
            """Background worker — runs the wander, builds the dossier,
            caches both. Exceptions are surfaced via JobState.error; the
            task itself never propagates an error (it has nowhere to
            surface to — the HTTP response already returned).

            Credit lifecycle:
              - On COMPLETED: commit the reservation with actual tokens.
              - On CANCELLED: leave the reservation open — the /abort
                handler runs commit() with the snapshot tokens to debit
                the user fairly for what was actually consumed.
              - On FAILED (exception): release the full reservation —
                the wander didn't deliver, the user shouldn't pay.
            """
            # Live progress handle — registered in _LIVE_PROGRESS so the
            # abort route can read real cumulative_tokens at cancel time.
            # Cleaned up in `finally` regardless of how the worker exits.
            progress = WanderingProgress()
            _LIVE_PROGRESS[session_id] = progress
            if reservation is not None:
                _RESERVATIONS[session_id] = reservation
            try:
                # Use the real fetcher in LIVE mode; stub in MOCK so tests
                # don't hit the network.
                fetcher = (
                    web_search_fetcher
                    if client.mode == ClientMode.LIVE  # type: ignore[attr-defined]
                    else None
                )
                if fetcher is None:
                    session = await run_wandering_session(
                        cushion, config, client, progress=progress,
                    )
                else:
                    session = await run_wandering_session(
                        cushion, config, client,
                        fetcher=fetcher, progress=progress,
                    )

                dossier = await build_dossier(session, client)

                # Best-effort persistence to the durable store. Failure
                # here doesn't fail the job — the in-memory cache is
                # still good for the immediate read.
                store = get_store()
                try:
                    await store.save_session(user_id, session)
                except Exception as e:
                    log.warning("save_session failed for %s: %s", session_id, e)

                # Cache results in three places: cushion for dig-deeper,
                # session for dig-deeper + memo adapter, dossier for the
                # GET /session/<id> fast path.
                _CUSHION_CACHE[session_id] = cushion
                _SESSION_CACHE[session_id] = session
                _DOSSIER_CACHE[session_id] = dossier

                # Settle credits: commit the reservation against the
                # actual tokens spent. Unused budget returns to the
                # user's spendable balance automatically.
                if reservation is not None:
                    try:
                        result = await get_credit_service().commit(
                            reservation_id=reservation.reservation_id,
                            actual_tokens=session.total_tokens_spent,
                        )
                        log.info(
                            "wander %s charged: budgeted=%d used=%d refunded=%d balance=%d",
                            session_id,
                            result.budgeted, result.used,
                            result.refunded, result.balance_after,
                        )
                    except Exception as e:
                        log.warning(
                            "credit commit failed for %s: %s (reservation may leak)",
                            session_id, e,
                        )

                jobs.mark_completed(session_id)
                log.info(
                    "wander %s completed (%d reports, %d tokens, %.1fs)",
                    session_id,
                    session.report_count(),
                    session.total_tokens_spent,
                    session.elapsed_seconds,
                )
            except asyncio.CancelledError:
                # User clicked Abort; the task was cancelled via
                # asyncio.Task.cancel(). We mark the job and re-raise so
                # the runtime knows the task was cancelled cleanly.
                # The /abort handler is responsible for committing the
                # reservation against the snapshot tokens — we leave it
                # alone here so abort's pre-cancel snapshot is authoritative.
                jobs.mark_aborted(session_id)
                log.info("wander %s aborted by user", session_id)
                raise
            except Exception as e:
                # Wander failed mid-flight (LLM error, network drop, etc.).
                # Release the full reservation — the user shouldn't pay
                # for a wander that didn't deliver.
                if reservation is not None and get_credit_service().is_open(
                    reservation.reservation_id,
                ):
                    try:
                        await get_credit_service().release(reservation.reservation_id)
                        log.info(
                            "wander %s failed; released %d credits",
                            session_id, reservation.held_credits,
                        )
                    except Exception as rel_e:
                        log.warning(
                            "credit release failed for %s: %s", session_id, rel_e,
                        )
                jobs.mark_failed(session_id, str(e))
                log.exception("wander %s failed: %s", session_id, e)
            finally:
                # Tear down the live progress handle regardless of how
                # the worker exited. Cancel reads happened pre-cancel
                # in the abort route, so by this point the credit math
                # is already done.
                _LIVE_PROGRESS.pop(session_id, None)
                _RESERVATIONS.pop(session_id, None)

        # Spawn the worker on the running event loop. We do NOT await it
        # — that's the whole point of this refactor.
        task = asyncio.create_task(_run_wander(), name=f"wander-{session_id}")

        try:
            jobs.register_job(
                session_id=session_id,
                job_id=job_id,
                user_id=user_id,
                mode=mode.value,
                agents=resolved_agents,
                time_budget_seconds=resolved_time,
                pursuit=pursuit_text,
                task=task,
            )
        except RuntimeError as e:
            # Race condition: another request registered between our
            # earlier check and here. Cancel our task and surface 409.
            task.cancel()
            return JSONResponse(
                status_code=409,
                content={"error": "session_already_running", "detail": str(e)},
            )

        return JSONResponse(
            status_code=202,
            content={
                "session_id":          session_id,
                "job_id":              job_id,
                "user_id":             user_id,
                "mode":                mode.value,
                "agents":              resolved_agents,
                "time_budget_seconds": resolved_time,
                "pursuit":             pursuit_text,
                "status":              jobs.JobStatus.RUNNING.value,
                "started_at":          time.time(),
            },
        )

    @router.get("/session/{session_id}/status")
    async def get_session_status(session_id: str) -> JSONResponse:
        """Cheap poll endpoint. Returns the current JobState as JSON.

        Frontend polls this every few seconds during the wander. Body
        carries enough to drive the LiveWandering wait screen (mode,
        agents, time_budget_seconds, pursuit) so a user who reopened
        the tab and lost their local draft still sees their session in
        full fidelity.

        Consults the in-process registry first; on a miss, falls back to
        the durable store so post-restart polls see the FAILED/aborted
        state from the prior PID instead of a 404 that confuses the
        frontend's resume path.

        404 only when neither layer knows the session — at that point
        it never existed.

        LIVE TELEMETRY (running jobs only): when the in-process registry
        has the job AND there's a registered WanderingProgress handle,
        we merge real per-agent state into the response under `live`:
          - `tokens_used`       total tokens spent so far across agents
          - `reports_count`     reports finalized so far
          - `urls_visited`      distinct URLs touched
          - `followon_queue`    depth of the shared follow-on queue
          - `agents[]`          per-agent snapshot (id, model, tokens,
                                phase, position, ...)
        These are REAL counters from in-memory state — not estimates.
        They're absent on completed/failed/aborted/restored-from-store
        responses because the worker's `finally` block tears down the
        progress handle when the wander ends.
        """
        payload = await jobs.get_status_durable(session_id)
        if payload is None:
            return JSONResponse(
                status_code=404,
                content={"error": "session_not_found"},
            )

        # Attach live per-agent state when the wander is in-flight.
        progress = _LIVE_PROGRESS.get(session_id)
        if progress is not None and payload.get("status") == "running":
            payload["live"] = {
                "tokens_used":        progress.tokens_used,
                "reports_count":      progress.reports_count,
                "urls_visited":       progress.urls_visited,
                "followon_queue":     progress.followon_queue_size,
                "agents":             progress.live_state(),
            }

        return JSONResponse(content=payload)

    @router.post("/session/{session_id}/abort")
    async def post_session_abort(session_id: str) -> JSONResponse:
        """Cancel a running wander. Returns the (possibly-updated) job
        state plus an HONEST credit breakdown — what the user spent vs
        what we refunded.

        ORDER OF OPERATIONS (important):
          1. Read the live WanderingProgress handle BEFORE firing
             task.cancel(). This snapshots the agents' cumulative
             tokens at the moment of user intent. Reads are pure
             memory access; no awaits.
          2. Fire task.cancel(). The cooperative cancellation surfaces
             at the worker's next await point — the worker catches
             CancelledError, marks the JobState aborted, and the
             `finally` block tears down the progress handle.
          3. Build the response from the snapshot + the JobState the
             abort_job mutation already updated.

        We snapshot pre-cancel because the worker MIGHT race to clean
        up _LIVE_PROGRESS before we read it (rare; only happens if
        cancel propagates synchronously somehow). Reading first is
        safe — at worst we report tokens that include 1-2 microseconds
        of post-snapshot work, which rounds down to zero credits.

        Idempotent: aborting a job that's already terminal is a no-op;
        we still return the JobState with whatever progress info we
        have (which may be empty if the worker already cleaned up)."""
        # Snapshot live token usage BEFORE firing cancel.
        progress = _LIVE_PROGRESS.get(session_id)
        live_tokens = progress.tokens_used if progress is not None else 0

        # Look up the open reservation BEFORE firing cancel — the
        # worker's `finally` could prune _RESERVATIONS the moment the
        # cancellation propagates.
        reservation = _RESERVATIONS.get(session_id)

        # Fire the cancel. Returns True only when there was a RUNNING
        # job to actually cancel; False on already-terminal sessions.
        cancelled = jobs.abort_job(session_id)
        state = jobs.get_job(session_id)
        if state is None:
            return JSONResponse(
                status_code=404,
                content={"error": "session_not_found"},
            )

        # Settle the reservation. There are three paths to think about:
        #   (a) Reservation exists AND is still open: commit against
        #       the snapshot tokens. This is the normal cancel path.
        #       commit() writes the CHARGE entry for what was used and
        #       releases the unused portion back to spendable balance.
        #   (b) Reservation exists but already closed: the worker beat
        #       us to it (already committed or released). Read the
        #       reservation handle for the breakdown — used/refunded
        #       are best-effort derived from the snapshot.
        #   (c) No reservation (guest user or budget was zero): fall
        #       back to mode-default math so the UI still shows numbers.
        credits_payload: dict[str, Any]
        if reservation is not None and get_credit_service().is_open(
            reservation.reservation_id,
        ):
            try:
                result = await get_credit_service().commit(
                    reservation_id=reservation.reservation_id,
                    actual_tokens=live_tokens,
                    note=f"Wander {session_id[:8]} cancelled at "
                         f"{live_tokens:,} tokens",
                )
                credits_payload = {
                    "budgeted":          result.budgeted,
                    "used":              result.used,
                    "refunded":          result.refunded,
                    "balance_after":     result.balance_after,
                    "unit":              "credit",
                    "tokens_per_credit": TOKENS_PER_CREDIT,
                }
            except Exception as e:
                log.warning(
                    "credit commit on abort failed for %s: %s — "
                    "falling back to derived breakdown",
                    session_id, e,
                )
                credits_payload = _derived_credit_payload(state, live_tokens)
        else:
            credits_payload = _derived_credit_payload(state, live_tokens)

        return JSONResponse(content={
            "session_id":       session_id,
            "aborted":          cancelled,
            **state.to_dict(),
            # Credit breakdown — the trust-building bit. Shape stays
            # consistent regardless of which settlement path fired.
            "credits": credits_payload,
        })

    @router.get("/session/{session_id}")
    async def get_session_dossier(session_id: str) -> JSONResponse:
        """Fetch a session's dossier. Job-aware:

          - status=running  → 202 with current job state (no dossier yet)
          - status=completed → 200 with the cached dossier
          - status=failed    → 500 with error
          - status=aborted   → 410 (Gone)
          - no job state     → fall back to Neo4j read (returns 404 if
                                the session never existed)
        """
        state = jobs.get_job(session_id)

        if state is not None:
            if state.status == jobs.JobStatus.RUNNING:
                # No dossier yet; signal to the client to keep polling.
                return JSONResponse(
                    status_code=202,
                    content={
                        "session_id": session_id,
                        "exists":     True,
                        "dossier":    None,
                        "status":     state.status.value,
                        "elapsed_seconds": state.to_dict()["elapsed_seconds"],
                    },
                )
            if state.status == jobs.JobStatus.FAILED:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error":      "wander_failed",
                        "session_id": session_id,
                        "detail":     state.error or "unknown error",
                    },
                )
            if state.status == jobs.JobStatus.ABORTED:
                return JSONResponse(
                    status_code=410,
                    content={
                        "error":      "wander_aborted",
                        "session_id": session_id,
                        "detail":     "this wander was cancelled before completion",
                    },
                )
            # status == COMPLETED — fall through to the cached dossier.

        # Cache-first read. _DOSSIER_CACHE is populated by the worker on
        # successful completion; if the job is COMPLETED, this hits.
        cached_dossier = _DOSSIER_CACHE.get(session_id)
        if cached_dossier is not None:
            return JSONResponse(content={
                "session_id": session_id,
                "exists":     True,
                "dossier":    cached_dossier.to_dict(),
                "status":     jobs.JobStatus.COMPLETED.value,
            })

        # Legacy / fallback path: SessionResult cached but dossier not
        # (can happen if the session was started before _DOSSIER_CACHE
        # existed, or if the in-process cache is hot but the dossier
        # cache was cleared). Re-run build_dossier.
        cached_session = _SESSION_CACHE.get(session_id)
        if cached_session is not None:
            client = get_llm_client()
            dossier = await build_dossier(cached_session, client)
            _DOSSIER_CACHE[session_id] = dossier
            return JSONResponse(content={
                "session_id": session_id,
                "exists":     True,
                "dossier":    dossier.to_dict(),
                "status":     jobs.JobStatus.COMPLETED.value,
            })

        # Neo4j fallback — fetch reports only and build a minimal dossier
        store = get_store()
        try:
            reports = await store.get_reports(session_id)
        except Exception as e:
            log.warning("get_reports failed: %s", e)
            reports = []

        if not reports:
            return JSONResponse(content={"session_id": session_id, "exists": False, "dossier": None})

        # Build a minimal dossier wrapper around just the reports.
        # The synthesis layer needs a fake session; we synthesize what we have.
        from src.wandering.articulate import articulate_report
        from src.wandering.dossier import (
            ConfidenceBand, Dossier, DossierMetadata,
        )

        client = get_llm_client()
        cards = []
        for r in reports:
            try:
                card = await articulate_report(r, client)
                cards.append(card)
            except Exception as e:
                log.debug("articulate_report failed for %s: %s", r.report_id, e)

        high_band = ConfidenceBand(Confidence.HIGH)
        medium_band = ConfidenceBand(Confidence.MEDIUM)
        low_band = ConfidenceBand(Confidence.LOW)
        for c in cards:
            if c.confidence == Confidence.HIGH:
                high_band.cards.append(c)
            elif c.confidence == Confidence.MEDIUM:
                medium_band.cards.append(c)
            else:
                low_band.cards.append(c)

        metadata = DossierMetadata(
            session_id=session_id,
            mode=WanderingMode.MULTI_PENDULUM,  # unknown; default
            anchor_summary="(reconstructed from persisted reports)",
            cushion_constellation_size=0,
            agent_count=0,
            report_count=len(reports),
            total_tokens_spent=0,
            elapsed_seconds=0.0,
            completed_at=time.time(),
        )
        dossier = Dossier(metadata=metadata, high=high_band, medium=medium_band, low=low_band)
        return JSONResponse(content={
            "session_id": session_id,
            "exists": True,
            "dossier": dossier.to_dict(),
        })

    @router.post("/session/{session_id}/dig-deeper")
    async def post_dig_deeper(session_id: str, request: Request) -> JSONResponse:
        """Spawn a sub-agent against one report from a prior session."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail={"error": "invalid_json"})

        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail={"error": "body_must_be_object"})

        report_id = str(body.get("report_id", "")).strip()
        if not report_id:
            return JSONResponse(status_code=400, content={"error": "missing_report_id"})

        cushion = _CUSHION_CACHE.get(session_id)
        cached_session = _SESSION_CACHE.get(session_id)
        if cushion is None or cached_session is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "session_not_in_cache",
                    "detail": (
                        "dig-deeper requires the original session to still be in memory; "
                        "if the server was restarted, re-run the original brief."
                    ),
                },
            )

        # Find the original report
        target = next(
            (r for r in cached_session.reports if r.report_id == report_id),
            None,
        )
        if target is None:
            return JSONResponse(status_code=404, content={"error": "report_not_found"})

        focus = str(body.get("focus", "")).strip()
        tokens_raw = body.get("tokens", 20_000)
        try:
            tokens = int(tokens_raw)
        except (TypeError, ValueError):
            tokens = 20_000

        req = spawn_request_from_user_dig_deeper(
            cushion=cushion,
            report=target,
            user_request_text=focus,
            distance_budget_tokens=tokens,
        )

        allowed, reason = should_spawn(
            req,
            session_tokens_spent=cached_session.total_tokens_spent,
            session_token_cap=cached_session.config.session_token_cap,
        )
        if not allowed:
            return JSONResponse(
                status_code=400,
                content={"error": "spawn_blocked", "reason": reason},
            )

        client = get_llm_client()
        fetcher = web_search_fetcher if client.mode == ClientMode.LIVE else None  # type: ignore[attr-defined]
        if fetcher is None:
            outcome = await run_subagent(req, client=client)
        else:
            outcome = await run_subagent(req, client=client, fetcher=fetcher)

        # Fold reports into the cached session for future dig-deeper calls
        cached_session.reports.extend(outcome.reports)
        cached_session.total_tokens_spent += outcome.tokens_spent

        # Render the new reports as cards
        from src.wandering.articulate import articulate_report

        new_cards = []
        for r in outcome.reports:
            try:
                card = await articulate_report(r, client)
                new_cards.append(card.to_dict())
            except Exception as e:
                log.debug("articulate_report failed for %s: %s", r.report_id, e)

        return JSONResponse(content={
            "session_id": session_id,
            "subagent_id": outcome.subagent_id,
            "aborted": outcome.aborted,
            "tokens_spent": outcome.tokens_spent,
            "new_reports_count": len(outcome.reports),
            "new_cards": new_cards,
        })

    @router.post("/session/{session_id}/continue")
    async def post_session_continue(
        session_id: str, request: Request,
    ) -> JSONResponse:
        """Continue an existing wander with new pursuit + new hunches.

        The original session's vision is the stable anchor — the user
        doesn't re-state it. They supply the next pursuit (derived from
        what the dossier surfaced) and any new analogies that came up
        while reading. The backend rebuilds the cushion from
        (new pursuit) + (new hunches) + (original vision) and spawns a
        fresh wander burst against the SAME session_id. New reports are
        APPENDED to the cached session — the dossier grows over time.

        Behaviour matches POST /session: returns 202 immediately with
        the JobInfo for the new burst; the wander runs in the background
        on the server.

        Failure modes:
          400 — invalid input (missing pursuit, body not an object)
          404 — session not in cache (server restart between runs;
                user can start a new wander, or we can rehydrate from
                Neo4j once the lossy SessionResult reconstruction is
                fully wired — F-followup)
          409 — a previous job for this session_id is still RUNNING
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail={"error": "invalid_json"})

        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400, detail={"error": "body_must_be_object"},
            )

        new_pursuit = str(body.get("pursuit", "") or "").strip()
        if len(new_pursuit) < 10:
            return JSONResponse(
                status_code=400,
                content={
                    "error":  "pursuit_too_thin",
                    "detail": "the next pursuit field needs at least one full sentence",
                },
            )

        cached_session = _SESSION_CACHE.get(session_id)
        if cached_session is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error":  "session_not_in_cache",
                    "detail": (
                        "continue requires the original session to still be in memory; "
                        "if the server was restarted, start a new wander instead"
                    ),
                },
            )

        # Block re-entry while a previous burst is still running.
        existing_job = jobs.get_job(session_id)
        if existing_job is not None and existing_job.status == jobs.JobStatus.RUNNING:
            return JSONResponse(
                status_code=409,
                content={
                    "error":      "session_already_running",
                    "detail":     "a wander is already running for this session_id; poll /status",
                    "session_id": session_id,
                    "job_id":     existing_job.job_id,
                },
            )

        user_id = get_effective_user_id(request, body.get("user_id"))

        # Build the continuation's CushionInput. Vision and current_map
        # are CARRIED from the original session (the stable anchor).
        # Pursuit (problem) and hunches (context) are fresh from the user.
        new_hunches = str(body.get("hunches", "") or "").strip()
        original_input = cached_session.cushion.raw_input
        continuation_input = CushionInput(
            problem=CushionField(
                name="problem",
                content=new_pursuit,
            ),
            context=CushionField(
                name="context",
                content=new_hunches,
            ),
            vision=CushionField(
                name="vision",
                content=original_input.vision.content,
                skip_reason=original_input.vision.skip_reason,
            ),
            current_map=CushionField(
                name="current_map",
                content=original_input.current_map.content,
                skip_reason=original_input.current_map.skip_reason,
            ),
        )

        # Resolve mode + budgets. Defaults match the brief composer's.
        mode_raw = str(body.get("mode", "multi_pendulum")).strip().lower()
        try:
            mode = WanderingMode(mode_raw)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_mode",
                    "valid": [m.value for m in WanderingMode],
                },
            )

        agents_raw       = body.get("agents")
        time_seconds_raw = body.get("time_seconds")
        job_id = f"wjob-{uuid.uuid4().hex[:8]}"

        config = WanderingConfig(
            mode=mode,
            agents=int(agents_raw) if isinstance(agents_raw, int) else None,
            time_budget_seconds=(
                float(time_seconds_raw) if isinstance(time_seconds_raw, (int, float)) else None
            ),
            session_id=session_id,
        )

        resolved_agents, resolved_time, resolved_tokens_per_agent, _ = config.resolved()

        client = get_llm_client()

        # Continuations are billed identically to new wanders — same
        # mode-defaults math, same reserve/commit/release lifecycle.
        # The cushion-rebuild and dossier-merge work doesn't change the
        # credit story.
        budgeted_credits = tokens_to_credits(resolved_tokens_per_agent * resolved_agents)
        reservation: Reservation | None = None
        if user_id and budgeted_credits > 0:
            credits = get_credit_service()
            try:
                reservation = await credits.reserve(
                    user_id=user_id,
                    amount=budgeted_credits,
                    ref_id=session_id,
                    note=f"{mode.value} continuation ({resolved_agents} agents)",
                )
            except InsufficientCredits as ic:
                return JSONResponse(
                    status_code=402,
                    content={
                        "error":             "insufficient_credits",
                        "balance":           ic.balance,
                        "needed":            ic.needed,
                        "gap":               ic.needed - ic.balance,
                        "mode":              mode.value,
                        "agents":            resolved_agents,
                        "tokens_per_credit": TOKENS_PER_CREDIT,
                        "detail":            (
                            f"This {mode.value} continuation needs {ic.needed} credits; "
                            f"you have {ic.balance}. Top up to continue."
                        ),
                    },
                )

        async def _run_continuation() -> None:
            """Background worker for the continuation burst.

            Re-extracts the cushion from the updated inputs (so the
            agents anchor against the EVOLVED problem statement),
            spawns the wander, then APPENDS new reports to the cached
            session. Final dossier is rebuilt across the merged report
            set so HIGH/MEDIUM/LOW bands include both the original and
            the continuation findings."""
            progress = WanderingProgress()
            _LIVE_PROGRESS[session_id] = progress
            if reservation is not None:
                _RESERVATIONS[session_id] = reservation
            try:
                # Step 1 — refreshed cushion (problem updated, vision held).
                continuation_cushion = await compose_cushion(
                    input_data=continuation_input,
                    client=client,
                    user_id=user_id,
                    auto_enrich=False,  # avoid double-enriching across continuations
                )

                # Step 2 — run the new wander burst.
                fetcher = web_search_fetcher if client.mode == ClientMode.LIVE else None  # type: ignore[attr-defined]
                if fetcher is None:
                    burst = await run_wandering_session(
                        continuation_cushion, config, client, progress=progress,
                    )
                else:
                    burst = await run_wandering_session(
                        continuation_cushion, config, client,
                        fetcher=fetcher, progress=progress,
                    )

                # Step 3 — merge: append new reports + traces onto the
                # cached session, sum tokens. Keep the original cushion
                # cached separately so dig-deeper still works against it,
                # AND store the continuation cushion as the latest.
                cached_session.reports.extend(burst.reports)
                cached_session.traces.extend(burst.traces)
                cached_session.total_tokens_spent += burst.total_tokens_spent
                cached_session.elapsed_seconds   += burst.elapsed_seconds
                cached_session.ended_at           = burst.ended_at
                cached_session.cushion            = continuation_cushion
                # config update kept narrow: only the mode (LOW/MED/HIGH)
                # might have changed for this burst.
                cached_session.config.mode = config.mode

                # Step 4 — rebuild the dossier across the merged set.
                merged_dossier = await build_dossier(cached_session, client)

                # Step 5 — persist + cache the merged state.
                store = get_store()
                try:
                    await store.save_session(user_id, cached_session)
                except Exception as e:
                    log.warning("save_session (continuation) failed for %s: %s",
                                session_id, e)

                _CUSHION_CACHE[session_id] = continuation_cushion
                _SESSION_CACHE[session_id] = cached_session
                _DOSSIER_CACHE[session_id] = merged_dossier

                # Settle credits for THIS burst only — burst.total_tokens_spent
                # is the delta added by the continuation, not the merged
                # session total. Charging on the merged total would
                # double-charge tokens from the original wander.
                if reservation is not None:
                    try:
                        result = await get_credit_service().commit(
                            reservation_id=reservation.reservation_id,
                            actual_tokens=burst.total_tokens_spent,
                        )
                        log.info(
                            "continuation %s charged: budgeted=%d used=%d refunded=%d balance=%d",
                            session_id,
                            result.budgeted, result.used,
                            result.refunded, result.balance_after,
                        )
                    except Exception as e:
                        log.warning(
                            "credit commit failed for continuation %s: %s",
                            session_id, e,
                        )

                jobs.mark_completed(session_id)
                log.info(
                    "continuation %s completed (+%d reports, +%d tokens; total: %d, %d)",
                    session_id,
                    len(burst.reports),
                    burst.total_tokens_spent,
                    cached_session.report_count(),
                    cached_session.total_tokens_spent,
                )
            except asyncio.CancelledError:
                # /abort handler will commit the reservation against the
                # snapshot tokens. Same contract as new-wander aborts.
                jobs.mark_aborted(session_id)
                log.info("continuation %s aborted by user", session_id)
                raise
            except Exception as e:
                if reservation is not None and get_credit_service().is_open(
                    reservation.reservation_id,
                ):
                    try:
                        await get_credit_service().release(reservation.reservation_id)
                        log.info(
                            "continuation %s failed; released %d credits",
                            session_id, reservation.held_credits,
                        )
                    except Exception as rel_e:
                        log.warning(
                            "credit release failed for continuation %s: %s",
                            session_id, rel_e,
                        )
                jobs.mark_failed(session_id, str(e))
                log.exception("continuation %s failed: %s", session_id, e)
            finally:
                _LIVE_PROGRESS.pop(session_id, None)
                _RESERVATIONS.pop(session_id, None)

        # Spawn worker, register job. Same lifecycle pattern as
        # post_session — async-accept, in-process JobState mirrors to
        # the durable store (F5).
        task = asyncio.create_task(
            _run_continuation(), name=f"wander-continue-{session_id}",
        )
        try:
            jobs.register_job(
                session_id=session_id,
                job_id=job_id,
                user_id=user_id,
                mode=mode.value,
                agents=resolved_agents,
                time_budget_seconds=resolved_time,
                pursuit=new_pursuit,
                task=task,
            )
        except RuntimeError as e:
            task.cancel()
            return JSONResponse(
                status_code=409,
                content={"error": "session_already_running", "detail": str(e)},
            )

        return JSONResponse(
            status_code=202,
            content={
                "session_id":          session_id,
                "job_id":              job_id,
                "user_id":             user_id,
                "mode":                mode.value,
                "agents":              resolved_agents,
                "time_budget_seconds": resolved_time,
                "pursuit":             new_pursuit,
                "status":              jobs.JobStatus.RUNNING.value,
                "started_at":          time.time(),
                "continuation":        True,
            },
        )

    @router.get("/sessions")
    async def get_sessions(request: Request) -> JSONResponse:
        """List the requesting user's past wandering sessions.

        Powers the frontend sidebar — each session is its own isolated
        wander graph and can be revisited any time. Each session's
        graph memory is hermetic — no contamination across sessions.

        Returns reverse-chronological sessions, each with:
          { session_id, mode, pursuit, completed_at, report_count,
            total_tokens_spent }

        Empty list when the user has no past wanders. Never 500s — store
        errors degrade to []. Guests get their guest-scoped sessions.
        """
        user_id = get_effective_user_id(request, request.query_params.get("user_id"))
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            limit = 50
        limit = max(1, min(limit, 100))

        store = get_store()
        try:
            sessions = await store.sessions_metadata(user_id, limit=limit)
        except Exception as e:
            log.warning("sessions_metadata failed: %s", e)
            sessions = []

        return JSONResponse(content={
            "user_id":  user_id,
            "sessions": sessions,
            "count":    len(sessions),
        })

    @router.get("/session/{session_id}/memo")
    async def get_session_memo(session_id: str) -> JSONResponse:
        """Adapt a wandering session's dossier into the Map Room's Memo
        shape (with a knowledge-graph visual). The frontend's
        /map/wander/<session_id> route hits this endpoint, then renders
        the result with the existing MapRoomPage primitives.

        Cache-only in V1 — same constraint as dig-deeper. If the original
        session isn't in memory (server restart, or a stale bookmark),
        return 404 with a clear remediation. The list endpoint that will
        let users walk back to old sessions is a separate concern.
        """
        cached_session = _SESSION_CACHE.get(session_id)
        if cached_session is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "session_not_in_cache",
                    "detail": (
                        "this wandering session is not in memory; "
                        "the server may have been restarted. "
                        "re-run the wander to view its map."
                    ),
                },
            )

        # Build the dossier the same way the GET /session/{id} route does.
        # The adapter is pure + sync once the dossier is in hand.
        client = get_llm_client()
        dossier = await build_dossier(cached_session, client)

        try:
            memo = session_to_memo(cached_session, dossier)
        except Exception as e:
            log.warning("session_to_memo failed for %s: %s", session_id, e)
            return JSONResponse(
                status_code=500,
                content={"error": "memo_adapter_failed", "detail": str(e)},
            )

        # Map Room's frontend loader expects the SAME shape that
        # /api/v2/thread/<id>/full returns — but ours is simpler because
        # wandering has no multi-iteration concept. We return one
        # synthesized iteration in the array, plus a synthetic thread
        # summary so the existing fetchThreadMemo fallback path is happy.
        # Frontend's fetchWanderingMemo helper consumes this directly;
        # the shape parallels (not reuses) the Thinking Room shape so
        # the frontend adapter is local + explicit.
        return JSONResponse(content={
            "session_id": session_id,
            "question":   cached_session.cushion.raw_input.problem.content,
            "memo":       memo,
            "mode":       cached_session.config.mode.value,
            "report_count": cached_session.report_count(),
        })

    return router


__all__ = [
    "get_router",
    "get_store",
    "get_llm_client",
]
