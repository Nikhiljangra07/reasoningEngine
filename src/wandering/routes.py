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

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.auth.supabase_auth import get_effective_user_id
from src.llm.client import LLMClient, ClientMode
from src.wandering.composer import compose_cushion
from src.wandering.cushion import (
    CushionField,
    CushionInput,
    CushionGraph,
    CushionLayer,
    SkipReason,
)
from src.wandering.dossier import build_dossier
from src.wandering.fetcher import web_search_fetcher
from src.wandering.persistence import (
    WanderingStore,
    build_wandering_store_from_env,
)
from src.wandering.report import Confidence
from src.wandering.runtime import (
    SessionResult,
    WanderingConfig,
    WanderingMode,
    run_wandering_session,
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
        """Run a Wandering Room session end-to-end. Returns the dossier."""
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
        agents = body.get("agents")
        time_seconds = body.get("time_seconds")
        tokens_per_agent = body.get("tokens_per_agent")
        model_mix = body.get("model_mix")
        session_id = body.get("session_id") or f"wsess-{uuid.uuid4().hex[:8]}"

        config = WanderingConfig(
            mode=mode,
            agents=int(agents) if isinstance(agents, int) else None,
            time_budget_seconds=float(time_seconds) if isinstance(time_seconds, (int, float)) else None,
            tokens_per_agent=int(tokens_per_agent) if isinstance(tokens_per_agent, int) else None,
            model_mix=tuple(model_mix) if isinstance(model_mix, list) else None,
            session_id=session_id,
        )

        client = get_llm_client()
        # Use the real fetcher in LIVE mode; stub in MOCK so tests don't hit the network.
        fetcher = web_search_fetcher if client.mode == ClientMode.LIVE else None  # type: ignore[attr-defined]
        # Fallback: if fetcher is None, runtime uses stub_fetcher default
        if fetcher is None:
            session = await run_wandering_session(cushion, config, client)
        else:
            session = await run_wandering_session(cushion, config, client, fetcher=fetcher)

        # Build the dossier
        dossier = await build_dossier(session, client)

        # Persist (best-effort)
        store = get_store()
        try:
            await store.save_session(user_id, session)
        except Exception as e:
            log.warning("save_session failed: %s", e)

        # Cache cushion + session for dig-deeper
        _CUSHION_CACHE[session.session_id] = cushion
        _SESSION_CACHE[session.session_id] = session

        return JSONResponse(content={
            "session_id": session.session_id,
            "mode": mode.value,
            "user_id": user_id,
            "summary": {
                "agent_count": session.agent_count(),
                "report_count": session.report_count(),
                "total_tokens_spent": session.total_tokens_spent,
                "elapsed_seconds": session.elapsed_seconds,
                "high_count": len(dossier.high.cards),
                "medium_count": len(dossier.medium.cards),
                "low_count": len(dossier.low.cards),
            },
            "dossier": dossier.to_dict(),
        })

    @router.get("/session/{session_id}")
    async def get_session_dossier(session_id: str) -> JSONResponse:
        """Re-fetch a session's dossier. V1 reads from in-memory cache;
        Neo4j path returns reports-only because get_session() doesn't
        reconstruct full SessionResult yet."""
        # In-memory cache hit?
        cached_session = _SESSION_CACHE.get(session_id)
        if cached_session is not None:
            client = get_llm_client()
            dossier = await build_dossier(cached_session, client)
            return JSONResponse(content={
                "session_id": session_id,
                "exists": True,
                "dossier": dossier.to_dict(),
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

    return router


__all__ = [
    "get_router",
    "get_store",
    "get_llm_client",
]
