"""
Constellax Reasoning Engine — FastAPI server.

Serves the chat UI from /web and exposes /api/trace which runs the full
async Wu Xing engine on a user's problem.

Production-ready features:
- Configurable port/host/CORS via environment variables
- Per-request trace state (no race conditions on concurrent requests)
- Structured logging
- Health check endpoint
- Input validation with size caps
- Graceful error handling at the endpoint level
- Loads .env via python-dotenv

Run:
    python server.py
Then open: http://localhost:$PORT (default 8100)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Load .env from repo root before importing anything that reads env vars.
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from src.bridge.conversation_store import ConversationStore
from src.bridge.thread_persistence import (
    get_router as _get_thread_persistence_router,
    schedule_record_iteration as _persist_iteration,
)
from src.bridge.web_search import (
    format_web_context_block as _format_web_context_block,
    web_search as _web_search,
)
from src.bridge.search_router import route as _route_search
from src.bridge.neo4j_backend import (
    Neo4jAnchorBackend,
    Neo4jConversationBackend,
    build_neo4j_driver_from_env,
)
from src.bridge.memory_retriever import MemoryRetriever
from src.bridge.embedding_service import GeminiEmbeddingService
from src.bridge.github_client import (
    build_github_client_from_env,
    make_github_handler,
)
from src.llm.memory_injection import build_memory_directive
from src.mcp_router import McpHandlerRegistry
from src.core.types import Direction, FrameworkID, Problem, Variable
from src.dispatcher import DispatchResult, dispatch, resume_with_choice
from src.llm.budget import BudgetCaps
from src.llm.client import ClientMode, LLMClient
from src.llm.dispatch_preview import (
    estimate_cost_breakdown,
    preview_dispatch,
    preview_to_dict,
    serialize_formation_plan,
    serialize_perspectives,
)
from src.llm.effort import Effort, iterations_for, normalize_effort
from src.llm.engine import run_async_formation
from src.llm.speech import (
    SegmentMemo,
    extract_speech_input,
    generate_clarification,
    generate_opinion_segment,
    generate_prospects_segment,
    generate_speech,
    generate_synthesizer_only,
    generate_synthesizer_segment,
)
from src.llm.triage import Route, triage as run_triage
from src.llm.visualizer import build_visuals


# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("constellax.server")

# Observability log — per-LLM-call records + per-request summary.
# Writes to stdout (with the rest) AND to logs/llm-calls.log so the
# data persists across server restarts and is easy to grep/parse.
# Disable with OBS_LOG=0; change the file path with OBS_LOG_FILE.
_obs = logging.getLogger("constellax.obs")
_obs.propagate = True  # also reach root → stdout via basicConfig handler
if os.environ.get("OBS_LOG", "1") not in ("0", "false", "no"):
    _obs_path = os.environ.get(
        "OBS_LOG_FILE",
        os.path.join(os.path.dirname(__file__), "logs", "llm-calls.log"),
    )
    try:
        os.makedirs(os.path.dirname(_obs_path), exist_ok=True)
        _h = logging.FileHandler(_obs_path, mode="a", encoding="utf-8")
        _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _h.setLevel(logging.INFO)
        _obs.addHandler(_h)
        log.info("Observability log: writing per-call records to %s", _obs_path)
    except OSError as _e:
        log.warning("Could not open observability log file: %s", _e)


# ── Config from environment ────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", "8100"))
HOST = os.environ.get("HOST", "0.0.0.0")
CORS_ORIGINS_ENV = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS_ENV.split(",") if o.strip()]

DEFAULT_MAX_ITERATIONS = int(os.environ.get("DEFAULT_MAX_ITERATIONS", "2"))
MAX_PHASE2_ITERATIONS = int(os.environ.get("MAX_PHASE2_ITERATIONS", "6"))
MAX_QUESTION_CHARS = int(os.environ.get("MAX_QUESTION_CHARS", "8000"))
MAX_PHASE1_SUMMARY_CHARS = int(os.environ.get("MAX_PHASE1_SUMMARY_CHARS", "20000"))

# Default effort tier when a request does not specify one. Mapped to an
# iteration budget via src/llm/effort.py (low=3, medium=6, high=10).
DEFAULT_EFFORT = normalize_effort(os.environ.get("LORA_EFFORT"))

# Conversation store backend selection.
#
# Backend selection (in priority order):
#   CONSTELLAX_DB_BACKEND=neo4j → Neo4jConversationBackend + Neo4jAnchorBackend
#                                  sharing one driver (requires NEO4J_URI +
#                                  NEO4J_PASSWORD; a missing creds with the
#                                  flag set is a LOUD WARNING — we fall back
#                                  to in-memory only as a dev-only escape
#                                  hatch).
#   (no flag / missing creds)   → in-memory dict (single-process,
#                                  optionally JSON-file backed via
#                                  LORA_CONVERSATION_STORE_PATH). Dev only.
#
# Redis support was removed in the Phase 6 cleanup of the Neo4j migration —
# Neo4j is the sole persistent backend now.
CONVERSATION_STORE_PATH = os.environ.get("LORA_CONVERSATION_STORE_PATH", "")
CONSTELLAX_DB_BACKEND = os.environ.get("CONSTELLAX_DB_BACKEND", "neo4j").strip().lower()

_CONV_BACKEND: dict = {}
# Active conversation backend — Neo4jConversationBackend when Neo4j is
# configured, otherwise None (falls through to in-memory).
_CONV_BACKEND_ACTIVE: "object | None" = None
# Active anchor backend — Neo4jAnchorBackend when Neo4j is configured,
# otherwise None (MemoryAdapter falls through to in-memory in stub mode).
# BridgeClient(mode="live", anchor_backend=_ANCHOR_BACKEND_ACTIVE, ...)
# wires up the persistent path.
_ANCHOR_BACKEND_ACTIVE: "object | None" = None
# Shared Neo4j driver (module-scope so both conv + anchor backends use one
# connection pool against the Aura instance, and the Decision Trace writer
# / sweeper / memory retriever can reuse it without spinning up duplicates).
_NEO4J_DRIVER: "object | None" = None
_NEO4J_DATABASE: str | None = None

if CONSTELLAX_DB_BACKEND == "neo4j":
    _built = build_neo4j_driver_from_env()
    if _built is not None:
        _NEO4J_DRIVER, _NEO4J_DATABASE = _built
        # Conversation backend owns the shared driver — its .close() cleans up
        # the pool on shutdown. Anchor backend uses the same driver with
        # owns_driver=False so a stray close() can't pull the rug out from
        # under the other backend.
        _CONV_BACKEND_ACTIVE = Neo4jConversationBackend(
            _NEO4J_DRIVER, database=_NEO4J_DATABASE, owns_driver=True,
        )
        _ANCHOR_BACKEND_ACTIVE = Neo4jAnchorBackend(
            _NEO4J_DRIVER, database=_NEO4J_DATABASE, owns_driver=False,
        )
        log.info(
            "Neo4j: ConversationStore + AnchorBackend bound to shared driver "
            "(database=%s)",
            _NEO4J_DATABASE,
        )
    else:
        # Explicit-intent signal — log loudly so this is discoverable in
        # production logs instead of silently degrading to in-memory.
        log.warning(
            "CONSTELLAX_DB_BACKEND=neo4j but NEO4J_URI/NEO4J_PASSWORD are not "
            "set — falling back to in-memory backends (dev-only). Set the "
            "Neo4j env vars for persistent storage."
        )

# JSON-file persistence is only meaningful for the in-memory backend.
# When Neo4j is active, that backend IS the persistence.
if CONVERSATION_STORE_PATH and _CONV_BACKEND_ACTIVE is None:
    _bootstrap = ConversationStore(storage_path=CONVERSATION_STORE_PATH)
    _CONV_BACKEND = _bootstrap._backend.raw


def _make_conv_store(project_id: str | None) -> ConversationStore:
    """Build a per-request ConversationStore sharing the process backend."""
    if _CONV_BACKEND_ACTIVE is not None:
        return ConversationStore(
            project_id=project_id,
            backend=_CONV_BACKEND_ACTIVE,
        )
    return ConversationStore(
        project_id=project_id,
        store=_CONV_BACKEND,
        storage_path=CONVERSATION_STORE_PATH or None,
    )


def _make_bridge_client(
    repo_root: str,
    project_id: str | None,
):
    """Build a per-request BridgeClient sharing the process anchor backend.

    Returns a live-mode BridgeClient when Neo4j is active (decisions persist
    across restarts), and a stub-mode one otherwise (in-memory, dev-only).
    Import is lazy so the bridge module doesn't pull GraphifyAdapter +
    similarity stack at startup when no caller asks for it.
    """
    from src.bridge.client import BridgeClient

    if _ANCHOR_BACKEND_ACTIVE is not None:
        return BridgeClient(
            repo_root=repo_root,
            mode="live",
            project_id=project_id,
            anchor_backend=_ANCHOR_BACKEND_ACTIVE,
        )
    return BridgeClient(
        repo_root=repo_root,
        mode="stub",
        project_id=project_id,
    )


# ---------------------------------------------------------------------------
# MCP handler registry — Phase B startup wiring
# ---------------------------------------------------------------------------
# Constructed once at module load. Handlers are registered for each
# capability whose backing service is configured (env-detected). The
# matching capability gets flipped MISSING → AVAILABLE on the per-request
# CapabilityRegistry via _MCP_AVAILABLE_AT_STARTUP so the registry stays
# honest with the actual wiring.

_MCP_HANDLERS = McpHandlerRegistry()
_MCP_AVAILABLE_AT_STARTUP: list[str] = []

# GitHub — read-only PRs / issues / repo / commits / file contents.
_github_client = build_github_client_from_env()
if _github_client is not None:
    _MCP_HANDLERS.register("github", make_github_handler(_github_client))
    _MCP_AVAILABLE_AT_STARTUP.append("github")
    log.info("MCP: github handler registered (GITHUB_TOKEN detected)")
else:
    log.info("MCP: github handler NOT registered (GITHUB_TOKEN unset)")


def _make_capability_registry():
    """Build a per-request CapabilityRegistry with startup-detected MCPs
    flipped to AVAILABLE.

    Constructed fresh per request so usage_log stays request-scoped.
    The set of "actually available" external MCPs is determined ONCE
    at startup by env detection (above); this helper just applies that
    state to each new registry instance.
    """
    from src.capabilities import CapabilityRegistry
    reg = CapabilityRegistry()
    for name in _MCP_AVAILABLE_AT_STARTUP:
        reg.mark_available(name)
    return reg


# ---------------------------------------------------------------------------
# Memo cache — Phase 4 (browser-tab Map Room)
#
# Stores the structured memo + question per trace request_id so the new
# browser-tab Map Room (/map/<id> on the web app) can fetch it via
# GET /api/v2/thread/<id>. In-process dict with a soft TTL; for multi-
# worker production we'll move this to Redis alongside ConversationStore.
# Single-worker dev is fine as-is.
# ---------------------------------------------------------------------------

from threading import Lock as _Lock

_MEMO_CACHE: dict[str, dict] = {}
_MEMO_CACHE_LOCK = _Lock()
_MEMO_CACHE_TTL_S = 60 * 60 * 24      # 24h — Map Room links are session-scoped
_MEMO_CACHE_MAX_ENTRIES = 1000        # hard cap on memory


def _cache_memo(thread_id: str, *, question: str, payload: dict) -> None:
    """Store a memo + question under a thread_id for the Map Room to fetch."""
    if not thread_id or not payload:
        return
    now = time.time()
    with _MEMO_CACHE_LOCK:
        # Sweep stale entries first (cheap, opportunistic GC).
        if len(_MEMO_CACHE) > _MEMO_CACHE_MAX_ENTRIES // 2:
            stale = [
                k for k, v in _MEMO_CACHE.items()
                if now - v.get("_cached_at", now) > _MEMO_CACHE_TTL_S
            ]
            for k in stale:
                _MEMO_CACHE.pop(k, None)
            # Hard cap: drop oldest if still over limit (LRU-ish).
            if len(_MEMO_CACHE) >= _MEMO_CACHE_MAX_ENTRIES:
                oldest = sorted(
                    _MEMO_CACHE.items(),
                    key=lambda kv: kv[1].get("_cached_at", 0),
                )[: len(_MEMO_CACHE) - _MEMO_CACHE_MAX_ENTRIES + 1]
                for k, _ in oldest:
                    _MEMO_CACHE.pop(k, None)
        _MEMO_CACHE[thread_id] = {
            "question":   question,
            "payload":    payload,
            "_cached_at": now,
        }


def _get_cached_memo(thread_id: str) -> dict | None:
    """Fetch a cached memo; None if not found or expired."""
    with _MEMO_CACHE_LOCK:
        entry = _MEMO_CACHE.get(thread_id)
        if entry is None:
            return None
        if time.time() - entry.get("_cached_at", 0) > _MEMO_CACHE_TTL_S:
            _MEMO_CACHE.pop(thread_id, None)
            return None
        return entry


# ---------------------------------------------------------------------------
# Segment cache — backs POST /api/v2/trace/segment
#
# When phase=synthesizer fires, the dispatcher produces a full memo (the
# expensive engine + speech step). We cache (speech_input, memo, etc.)
# under `memo_id` so the follow-on phases can either return a cached
# slice instantly OR re-shape a specific slice when the user splices in.
#
# In-process dict, soft TTL — same constraints as _MEMO_CACHE. For a
# multi-worker deployment this moves to Redis alongside the conversation
# store; single-worker dev is fine.
# ---------------------------------------------------------------------------

_SEGMENT_CACHE: dict[str, dict] = {}
_SEGMENT_CACHE_LOCK = _Lock()
_SEGMENT_CACHE_TTL_S = 60 * 30            # 30 min — segments are session-scoped
_SEGMENT_CACHE_MAX_ENTRIES = 500


def _cache_segments(
    memo_id: str,
    *,
    speech_input,
    full_memo: dict,
    question: str,
    thread_id: str,
    extra: dict | None = None,
) -> None:
    """Store the segment state for a memo_id.

    We cache even when `full_memo` is empty (non-DEEP routes) so any
    accidental follow-on opinion/prospects call returns an empty slice
    instead of a confusing 404. `next_phase=null` from the synthesizer
    response already tells well-behaved clients not to call.
    """
    if not memo_id:
        return
    now = time.time()
    with _SEGMENT_CACHE_LOCK:
        # Opportunistic GC
        if len(_SEGMENT_CACHE) > _SEGMENT_CACHE_MAX_ENTRIES // 2:
            stale = [
                k for k, v in _SEGMENT_CACHE.items()
                if now - v.get("_cached_at", now) > _SEGMENT_CACHE_TTL_S
            ]
            for k in stale:
                _SEGMENT_CACHE.pop(k, None)
            if len(_SEGMENT_CACHE) >= _SEGMENT_CACHE_MAX_ENTRIES:
                oldest = sorted(
                    _SEGMENT_CACHE.items(),
                    key=lambda kv: kv[1].get("_cached_at", 0),
                )[: len(_SEGMENT_CACHE) - _SEGMENT_CACHE_MAX_ENTRIES + 1]
                for k, _ in oldest:
                    _SEGMENT_CACHE.pop(k, None)
        _SEGMENT_CACHE[memo_id] = {
            "speech_input": speech_input,
            "full_memo":    dict(full_memo),   # copy — we mutate slices in-place
            "question":     question,
            "thread_id":    thread_id,
            "extra":        extra or {},
            "_cached_at":   now,
        }


def _get_cached_segments(memo_id: str) -> dict | None:
    """Fetch a cached segment state; None if not found or expired."""
    with _SEGMENT_CACHE_LOCK:
        entry = _SEGMENT_CACHE.get(memo_id)
        if entry is None:
            return None
        if time.time() - entry.get("_cached_at", 0) > _SEGMENT_CACHE_TTL_S:
            _SEGMENT_CACHE.pop(memo_id, None)
            return None
        return entry


def _update_cached_segment(memo_id: str, slice_fields: dict) -> None:
    """Merge a regenerated phase slice into the cached full memo."""
    if not memo_id or not slice_fields:
        return
    with _SEGMENT_CACHE_LOCK:
        entry = _SEGMENT_CACHE.get(memo_id)
        if entry is None:
            return
        entry["full_memo"].update(slice_fields)
        entry["_cached_at"] = time.time()       # bump TTL


# ---------------------------------------------------------------------------
# Segment slice extractors — pull the phase slice out of a full memo
# ---------------------------------------------------------------------------

def _synthesizer_slice(memo: dict | None) -> dict:
    if not isinstance(memo, dict):
        return {"verdict_line": "", "verdict_body": "", "confidence": "moderate"}
    return {
        "verdict_line": memo.get("verdict_line", ""),
        "verdict_body": memo.get("verdict_body", ""),
        "confidence":   memo.get("confidence", "moderate"),
    }


def _opinion_slice(memo: dict | None) -> dict:
    if not isinstance(memo, dict):
        return {"reasoning": [], "alternatives": [], "falsifiers": []}
    return {
        "reasoning":    memo.get("reasoning", []) or [],
        "alternatives": memo.get("alternatives", []) or [],
        "falsifiers":   memo.get("falsifiers", []) or [],
    }


def _prospects_slice(memo: dict | None) -> dict:
    if not isinstance(memo, dict):
        return {"open_questions": [], "visuals": []}
    return {
        "open_questions": memo.get("open_questions", []) or [],
        "visuals":        memo.get("visuals", []) or [],
    }


def _persist_after_engine_done(*, request_id: str, memo_id: str) -> None:
    """Persistence for the DEEP segment flow.

    Called once per memo_id when the opinion phase has finished
    running the engine. Reads the cached engine/dispatch state, fires
    `_persist_iteration` (fire-and-forget), and marks the cache entry
    as persisted so subsequent splice-driven calls don't double-persist.

    Persistence fires HERE (at end of opinion, after the engine has
    produced the canonical memo) rather than at end of prospects, so
    the async Falkor write has 5-15 seconds to land while the user
    advances through the prospects segment + dwell. Fixes the
    "thread isn't indexed yet" race in the Reasoning Trace.

    Non-DEEP routes call `_persist_iteration` directly from the
    synthesizer phase (their full memo is complete there); this helper
    is the DEEP-only path.
    """
    with _SEGMENT_CACHE_LOCK:
        entry = _SEGMENT_CACHE.get(memo_id)
        if entry is None:
            return
        extra = entry.get("extra") or {}
        if extra.get("persisted"):
            return
        dispatch_result = extra.get("dispatch_result")
        if dispatch_result is None:
            log.warning(
                "[%s] _persist_after_engine_done: no dispatch_result for memo_id=%s — skipping",
                request_id, memo_id,
            )
            return
        # Mark first so we don't race on concurrent prospects calls.
        extra["persisted"] = True
        entry["extra"] = extra
        question_augmented = extra.get("question_augmented") or ""
        original_question = extra.get("original_question") or question_augmented
        triage_payload = extra.get("triage_payload") or {}
        user_id = extra.get("user_id")
        project_id = extra.get("project_id")
        workspace_id = extra.get("workspace_id")
        surface_id = extra.get("surface_id")
        parent_thread_id = extra.get("parent_thread_id")
        user_effort_value = extra.get("effort") or "medium"
        request_start = entry.get("request_start") or time.time()
        web_search_meta = extra.get("web_search_meta")

    try:
        _persist_iteration(
            request_id=request_id,
            question=question_augmented,
            dispatch_result=dispatch_result,
            triage=triage_payload,
            cached_memo=dispatch_result.memo,
            user_id=user_id,
            project_id=project_id,
            workspace_id=workspace_id,
            surface_id=surface_id,
            parent_thread_id=parent_thread_id,
            effort_picked=user_effort_value,
            started_at=request_start,
            web_search_meta=web_search_meta,
            title_question=original_question,
            # Pin persistence to the synth-phase memo_id so the
            # Reasoning Trace finds the thread by the same id the
            # frontend already has. Without this, _resolve_thread_id
            # derives `thr-{opinion_request_id}` — a fresh id that
            # the frontend never saw — and getThreadFull 404s.
            explicit_thread_id=memo_id,
        )
        log.info(
            "[%s] _persist_after_engine_done scheduled for memo_id=%s (pinned)",
            request_id, memo_id,
        )
    except Exception as e:
        log.warning(
            "[%s] _persist_after_engine_done enqueue failed (non-fatal): %s",
            request_id, e,
        )


# Background sweep — periodic deletion of expired conversation entries.
# Filter-on-read already hides them from queries; this is what actually
# frees the heap (and disk, when persistence is enabled).
SWEEP_ENABLED = os.environ.get("LORA_SWEEP_ENABLED", "1") not in ("0", "false", "False", "")
SWEEP_INTERVAL_SECONDS = int(os.environ.get("LORA_SWEEP_INTERVAL_SECONDS", "3600"))
_sweep_task: "asyncio.Task | None" = None  # populated on startup, cancelled on shutdown


# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="Constellax Reasoning Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    # DELETE is required by the sidebar's per-row delete (the browser
    # does a CORS preflight on DELETE — without it listed here, fetch()
    # throws "Failed to fetch" before any HTTP exchange happens, which
    # looks like "backend is down" but isn't. PUT/PATCH added for the
    # same future-proofing reason. allow_headers=["*"] already covers
    # Content-Type and any custom headers we add later.
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Thread/iteration memory pipeline — adds:
#   GET  /api/v2/threads
#   GET  /api/v2/thread/{id}/full
#   GET  /api/v2/iteration/{id}
#   POST /api/v2/iteration/{id}/outcome
#   GET  /api/v2/threads/similar?iter_id=...
app.include_router(_get_thread_persistence_router())

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ── Helpers ────────────────────────────────────────────────────────────────

NEGATIVE_SIGNALS = {
    "but", "struggle", "fight", "doubt", "fear", "worried", "stuck", "hate",
    "dread", "can't", "don't", "terrified", "frustrated", "unfulfilled",
    "overwhelm", "anxious", "lost", "trapped",
}
POSITIVE_SIGNALS = {
    "love", "passionate", "dream", "want", "excited", "enjoy", "growing",
    "opportunity", "happy", "grateful",
}
EMPHASIS_HIGH = {"every", "always", "never", "completely", "totally"}


def parse_problem(text: str) -> Problem:
    """Auto-extract variables from raw user text."""
    variables: list[Variable] = []
    sentences = text.replace(".", ". ").split(". ")
    for i, raw in enumerate(sentences):
        sentence = raw.strip()
        if not sentence or len(sentence) < 10:
            continue
        lower = sentence.lower()
        nc = sum(1 for w in NEGATIVE_SIGNALS if w in lower)
        pc = sum(1 for w in POSITIVE_SIGNALS if w in lower)
        if nc > pc:
            direction = Direction.NEGATIVE
        elif pc > nc:
            direction = Direction.POSITIVE
        else:
            direction = Direction.NEUTRAL
        magnitude = 0.85 if any(w in lower for w in EMPHASIS_HIGH) else 0.6
        variables.append(Variable(
            name=f"v{i}",
            description=sentence[:200],
            magnitude=magnitude,
            direction=direction,
            confidence=0.8,
            source_framework=FrameworkID.FIRST_PRINCIPLES,
            is_user_stated=True,
        ))
    return Problem(statement=text, variables=variables[:8])


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines[1:]).strip()
    return cleaned


def _parse_findings_from_response(content: str) -> list[dict]:
    """Best-effort parsing of findings from an LLM response. Returns [] on any error."""
    try:
        data = json.loads(_strip_code_fences(content))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    findings_raw = data.get("findings", [])
    if not isinstance(findings_raw, list):
        return []
    findings: list[dict] = []
    for f in findings_raw:
        if not isinstance(f, dict):
            continue
        findings.append({
            "name": str(f.get("name", ""))[:80],
            "type": str(f.get("type", f.get("concept", "")))[:40],
            "description": str(f.get("description", ""))[:120],
            "magnitude": float(f.get("magnitude", 0.5) or 0.5),
            "direction": str(f.get("direction", "neutral"))[:16],
            "confidence": float(f.get("confidence", 0.5) or 0.5),
        })
    return findings


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", response_model=None)
async def root():
    """Serve the chat UI."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "status": "ok",
        "service": "Constellax Reasoning Engine",
        "version": "2.0.0",
        "domains": 5,
        "concepts": 63,
    })


def _has_live_key() -> bool:
    """LIVE mode runs if any supported provider key is set. OpenRouter is primary."""
    return bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


@app.get("/health")
async def health() -> dict:
    """Liveness probe for load balancers and orchestrators."""
    return {
        "status": "ok",
        "mode": "live" if _has_live_key() else "mock",
        "default_effort": DEFAULT_EFFORT.value,
    }


@app.post("/api/trace")
async def trace(request: Request) -> JSONResponse:
    """Run the engine on a user problem and return the full trace."""
    request_id = f"req-{int(time.time() * 1000)}"
    request_start = time.time()

    # ── Input parsing & validation ──
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "Field 'question' is required"}, status_code=400)
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse(
            {"error": f"Question too long. Max {MAX_QUESTION_CHARS} characters."},
            status_code=400,
        )

    # Effort tier (preferred) → iteration budget. Raw max_iterations still
    # wins if explicitly passed, so existing clients keep working.
    effort = normalize_effort(body.get("effort", DEFAULT_EFFORT))
    effort_iters = iterations_for(effort)

    raw_max_iters = body.get("max_iterations")
    if raw_max_iters is None:
        max_iterations = effort_iters
    else:
        try:
            max_iterations = int(raw_max_iters)
        except (TypeError, ValueError):
            return JSONResponse({"error": "max_iterations must be an integer"}, status_code=400)

    # Hard ceiling: never exceed 2× MAX_PHASE2_ITERATIONS (legacy cap) or 12 (engine cap).
    max_iterations = max(1, min(max_iterations, MAX_PHASE2_ITERATIONS * 2, 12))
    is_phase_one = max_iterations <= DEFAULT_MAX_ITERATIONS

    phase1_summary = str(body.get("phase1_summary", ""))
    if len(phase1_summary) > MAX_PHASE1_SUMMARY_CHARS:
        phase1_summary = phase1_summary[:MAX_PHASE1_SUMMARY_CHARS]

    log.info(
        "[%s] /api/trace start | qlen=%d | effort=%s | iters=%d | phase1=%s",
        request_id, len(question), effort.value, max_iterations, bool(phase1_summary),
    )

    # ── Per-request state (no globals → no race conditions) ──
    trace_events: list[dict] = []

    def emit(event_type: str, data: dict) -> None:
        trace_events.append({
            "type": event_type,
            "data": data,
            "t": (time.time() - request_start) * 1000,
        })

    # ── Engine execution ──
    try:
        mode = ClientMode.LIVE if _has_live_key() else ClientMode.MOCK
        client = LLMClient(mode=mode)

        # Instrument client to capture every call into this request's trace
        original_call = client.call

        async def instrumented_call(system_prompt, user_message, domain, concept, **kwargs):
            t0 = time.time()
            emit("call_start", {"domain": domain, "concept": concept})
            result = await original_call(system_prompt, user_message, domain, concept, **kwargs)
            elapsed = (time.time() - t0) * 1000
            findings = _parse_findings_from_response(result.content) if result.success else []
            emit("call_done", {
                "domain": domain,
                "concept": concept,
                "success": result.success,
                "elapsed_ms": round(elapsed),
                "tokens": result.input_tokens + result.output_tokens,
                "findings": findings,
                "preview": result.content[:300] if result.success else (result.error or ""),
            })
            return result

        client.call = instrumented_call

        problem = parse_problem(question)
        if phase1_summary:
            problem.context = (
                "PHASE 2 — DEEPER ANALYSIS. The user has already seen Phase 1 findings below. "
                "Do NOT repeat the same analysis. Go deeper. Challenge Phase 1's conclusions. "
                "Find what Phase 1 missed. Surface second-order effects and hidden variables "
                "that only emerge with more iterations.\n\n"
                f"PHASE 1 FINDINGS (already delivered to user):\n{phase1_summary}"
            )

        emit("stage", {"stage": 1, "name": "Chemistry Reads"})

        engine_result = await run_async_formation(
            problem, client, max_iterations=max_iterations,
        )

        speech_input = extract_speech_input(
            engine_result, question,
            is_phase_one=is_phase_one,
            estimated_additional_credits=15.0 if is_phase_one else 0,
        )
        emit("stage", {"stage": 7, "name": "Speech Module"})
        speech_result = await generate_speech(client, speech_input)

        total_ms = (time.time() - request_start) * 1000

        # ── Build response ──
        domain_data: dict[str, dict] = {}
        for dom, out in engine_result.domain_outputs.items():
            perspectives = []
            for p in out.perspectives:
                pvars = []
                for v in p.variables_found:
                    pvars.append({
                        "name": v.name,
                        "desc": v.description[:80],
                        "mag": v.magnitude,
                        "dir": v.direction.value,
                        "conf": v.confidence,
                        "hidden": v.is_hidden,
                    })
                perspectives.append({
                    "framework": p.framework.value,
                    "weight": p.weight,
                    "variables": pvars,
                })
            roots = []
            for r in out.root_causes:
                roots.append({
                    "name": r.variable.name,
                    "desc": r.variable.description[:120],
                    "confidence": r.confidence,
                    "bias": r.bias_that_hid_it or "",
                    "hidden": r.variable.is_hidden,
                })
            domain_data[dom.value] = {
                "perspectives": perspectives,
                "root_causes": roots,
                "raw_preview": out.raw_analysis[:400],
            }

        ke_data = [
            {
                "challenger": ke.challenger_domain.value,
                "target": ke.target_domain.value,
                "scrutiny": ke.scrutiny_score,
                "contradictions": ke.contradictions[:3],
                "flags": ke.flags[:3],
            }
            for ke in engine_result.ke_results
        ]

        funnel_data = [
            {
                "kept": f.variables_kept,
                "cached": f.variables_cached,
                "needs_work": f.variables_needing_work,
                "stable": f.variables_stable,
            }
            for f in engine_result.funnel_history
        ]

        conv_data = [
            {
                "iter": s.iteration,
                "gibbs": s.gibbs_energy,
                "converged": s.is_converged,
                "posterior_delta": s.posterior_delta,
                "new_vars": s.new_variables_count,
                "ke_scrutiny": s.avg_ke_scrutiny,
            }
            for s in engine_result.convergence_history.snapshots
        ]

        traj_data = [
            {
                "name": t.root_cause.variable.name,
                "desc": t.root_cause.variable.description[:200],
                "confidence": t.confidence,
                "hidden": t.root_cause.variable.is_hidden,
                "bias": t.root_cause.bias_that_hid_it or "",
            }
            for t in engine_result.trajectories[:4]
        ]

        summary = client.get_call_summary()

        log.info(
            "[%s] /api/trace done | %dms | calls=%d | tokens=%d | cost=$%.2f | converged=%s",
            request_id, round(total_ms), summary["total_calls"],
            summary["total_tokens"]["total_tokens"], summary["estimated_cost_usd"],
            engine_result.convergence_history.final_converged,
        )

        return JSONResponse({
            "request_id": request_id,
            "question": question,
            "total_time_ms": round(total_ms),
            "mode": "live" if has_key else "mock",
            "events": trace_events,
            "domains": domain_data,
            "ke": ke_data,
            "funnel": funnel_data,
            "convergence": conv_data,
            "trajectories": traj_data,
            "formation": {
                "complexity": engine_result.formation_plan.problem_complexity,
                "active_domains": [d.value for d in engine_result.formation_plan.active_domains],
                "agent_count": engine_result.formation_plan.estimated_agent_count,
            },
            "speech": speech_result.response_text,
            "dig_deeper": speech_result.dig_deeper_prompt,
            "delivery_mode": engine_result.delivery_mode,
            "stats": {
                "calls": summary["total_calls"],
                "tokens": summary["total_tokens"]["total_tokens"],
                "cost": summary["estimated_cost_usd"],
                "iterations": engine_result.convergence_history.total_iterations,
                "converged": engine_result.convergence_history.final_converged,
            },
        })

    except HTTPException:
        raise
    except Exception as exc:
        log.error(
            "[%s] /api/trace failed: %s\n%s",
            request_id, exc, traceback.format_exc(),
        )
        return JSONResponse(
            {
                "error": "Internal engine error. The reasoning engine failed mid-execution.",
                "detail": str(exc),
                "request_id": request_id,
            },
            status_code=500,
        )


@app.post("/api/v2/trace")
async def trace_v2(request: Request) -> JSONResponse:
    """
    Brain-extension routing endpoint — the new path.

    Wires triage gate + capability registry + budget + engine into one flow.
    Request body:
      {
        "question":     "...",                  # required
        "effort":       "low|medium|high",      # optional; default = LORA_EFFORT
        "policy":       "strict|ask|auto",      # optional; default = "ask"
        "caps": {                               # optional; per-request budget overrides
          "max_iterations": 12,
          "max_wall_time_sec": 720,
          "max_cost_usd": 1.00,
          "max_mcp_calls": 20
        }
      }

    Response body:
      {
        "route":         "trivial|direct|direct_plus|deep",
        "response_text": "...",                 # the synthesized advisor voice
        "triage":        { ... },               # what the gate decided + why
        "formation_plan": null | { ... },       # what the engine activated (DEEP only)
        "actual_iterations":     null | int,    # convergence iterations actually run
        "actual_active_domains": [ ... ],       # domain names that participated
        "perspectives":  [                      # per-trajectory cards for the UI
          { "angle", "title", "body", "reasoning" }, ...
        ],                                      # empty list when no engine ran
        "budget":        { ... },               # cost + time + iterations + caps
        "capabilities":  { available, missing, absent_by_design },
        "escalation_offer": null | { ... },     # set when policy=ask + gate>user
        "missing_capability_offers": [ ... ],   # MCPs the gate wanted but couldn't fire
        "debug":         { ... }
      }

    The existing /api/trace endpoint is preserved and unchanged.
    """
    request_id = f"req-{int(time.time() * 1000)}"
    request_start = time.time()

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "Field 'question' is required"}, status_code=400)
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse(
            {"error": f"Question too long. Max {MAX_QUESTION_CHARS} characters."},
            status_code=400,
        )

    # Preserve the user's literal question so the sidebar title and
    # iteration record reflect what THEY typed — not the augmented
    # prompt we send to the engine (CSV blocks, web context, Map Room
    # framing). Sidebars showing "[WEB CONTEXT — live search r…]" or
    # "[Map Room follow-up — the use…]" come from the augmented field
    # leaking into _derive_title; this `original_question` is the
    # value we hand to persistence for title derivation.
    original_question = question

    # Phase 5 — CSV attachments for data-driven charts.
    # Frontend sends `attachments: [{ name, content, mime }]` for small
    # CSV (< 50KB raw text) uploads from the Map Room. We inline them
    # into the question prefix so the synthesizer can read them and
    # emit a Vega-Lite spec. Large files / multi-CSV / Stripe / PostHog
    # are deferred to a v2 data-connector platform.
    attachments_in = body.get("attachments") or []
    if not isinstance(attachments_in, list):
        return JSONResponse(
            {"error": "attachments must be a list of {name, content} objects"},
            status_code=400,
        )
    attachment_blocks: list[str] = []
    MAX_ATTACHMENT_BYTES = 50_000   # ~12K tokens — fits comfortably in any budget
    total_attachment_size = 0
    for att in attachments_in[:3]:  # hard cap 3 files per request
        if not isinstance(att, dict):
            continue
        att_name    = str(att.get("name", "attachment.csv"))[:128]
        att_content = att.get("content")
        if not isinstance(att_content, str):
            continue
        if len(att_content) > MAX_ATTACHMENT_BYTES:
            return JSONResponse(
                {
                    "error":
                        f"Attachment '{att_name}' is {len(att_content)} bytes; "
                        f"max {MAX_ATTACHMENT_BYTES} bytes per file. Aggregate "
                        f"before uploading.",
                },
                status_code=400,
            )
        total_attachment_size += len(att_content)
        if total_attachment_size > MAX_ATTACHMENT_BYTES * 2:
            return JSONResponse(
                {"error": "Combined attachment size exceeds budget."},
                status_code=400,
            )
        attachment_blocks.append(
            f"[ATTACHED_CSV name=\"{att_name}\"]\n{att_content.rstrip()}\n[/ATTACHED_CSV]"
        )

    if attachment_blocks:
        # Prepend attachments to the question so the synthesizer sees the
        # data before the prose. Triage/router see the same prefix —
        # they're cheap enough that the extra tokens don't hurt.
        question = "\n\n".join(attachment_blocks) + "\n\n" + question

    # ─── Adaptive web search ────────────────────────────────────────────
    # The router (Gemini Flash) decides whether the question needs live
    # web data AND rewrites the user's phrasing into a search-engine
    # query. When it says yes, we hit Tavily (or DDG fallback) and
    # prepend the results as a "WEB CONTEXT" block to the prompt.
    #
    # Every step is recorded into `web_search_meta` so the Reasoning
    # Trace surfaces the full footprint: decision + reason + refined
    # query + provider + URLs visited + latency.
    web_context = ""
    web_search_meta: dict | None = None
    try:
        decision = await _route_search(question)
        web_search_meta = {
            "decision_via":   decision.via,         # "llm" | "heuristic" | "fallback"
            "needs_search":   decision.needs_search,
            "router_reason":  decision.reason,
            "router_model":   decision.model,
            "router_ms":      decision.latency_ms,
            "refined_query":  decision.refined_query if decision.needs_search else "",
            "results":        [],
            "provider":       None,
            "search_ms":      0,
            "cached":         False,
            "answer":         None,
        }
        log.info(
            "[%s] search.route via=%s needs=%s ms=%d reason=%r",
            request_id, decision.via, decision.needs_search,
            decision.latency_ms, decision.reason,
        )

        if decision.needs_search and decision.refined_query:
            try:
                ws_result = await _web_search(decision.refined_query, max_results=5)
                web_search_meta["provider"]   = ws_result.provider
                web_search_meta["search_ms"]  = ws_result.latency_ms
                web_search_meta["cached"]     = ws_result.cached
                web_search_meta["answer"]     = ws_result.answer
                web_search_meta["results"]    = [
                    {"title": h.title, "url": h.url, "snippet": h.snippet}
                    for h in ws_result.hits
                ]
                if ws_result.ok:
                    web_context = _format_web_context_block(ws_result)
                    log.info(
                        "[%s] search.run provider=%s hits=%d cached=%s ms=%d",
                        request_id, ws_result.provider, len(ws_result.hits),
                        ws_result.cached, ws_result.latency_ms,
                    )
                else:
                    web_search_meta["error"] = ws_result.error
                    log.info(
                        "[%s] search.run produced no results (error=%s)",
                        request_id, ws_result.error,
                    )
            except Exception as e:
                log.warning("[%s] search.run raised (non-fatal): %s", request_id, e)
                web_search_meta["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        # Outer guard — search must never break a trace.
        log.warning("[%s] search routing raised (non-fatal): %s", request_id, e)
        web_search_meta = {"error": f"{type(e).__name__}: {e}", "decision_via": "error"}

    if web_context:
        # Prepend web context — comes before the user question itself so
        # the model treats it as preceding evidence, not an inline note.
        question = web_context + "\n" + question

    user_effort = normalize_effort(body.get("effort", DEFAULT_EFFORT))
    policy = str(body.get("policy", "ask")).strip().lower()
    if policy not in ("strict", "ask", "auto"):
        return JSONResponse(
            {"error": "policy must be one of: strict | ask | auto"},
            status_code=400,
        )

    # Optional per-request cap overrides.
    caps_in = body.get("caps") or {}
    try:
        caps = BudgetCaps(
            max_iterations=int(caps_in.get("max_iterations", 12)),
            max_wall_time_sec=float(caps_in.get("max_wall_time_sec", 720.0)),
            max_cost_usd=float(caps_in.get("max_cost_usd", 1.00)),
            max_mcp_calls=int(caps_in.get("max_mcp_calls", 20)),
        )
    except (TypeError, ValueError):
        return JSONResponse({"error": "caps fields must be numeric"}, status_code=400)

    # Optional conversation context — if both fields present, the dispatch
    # records the iteration into the conversation store.
    session_id = body.get("session_id")
    project_id = body.get("project_id")
    if session_id is not None:
        session_id = str(session_id).strip() or None
    if project_id is not None:
        project_id = str(project_id).strip() or None

    log.info(
        "[%s] /api/v2/trace start | qlen=%d | effort=%s | policy=%s | session=%s",
        request_id, len(question), user_effort.value, policy,
        session_id or "(none)",
    )

    try:
        client = LLMClient(mode=(ClientMode.LIVE if _has_live_key() else ClientMode.MOCK))
        conv_store = _make_conv_store(project_id) if session_id else None
        # Phase 4: build Decision Trace prior-memory block. Empty string when
        # disabled / no user_id / Neo4j unavailable. Adds ~0.5-1s when active.
        memory_directive = await _build_memory_for_request(
            question=question, body=body,
            parent_thread_id=body.get("parent_thread_id"),
        )
        result: DispatchResult = await dispatch(
            text=question,
            client=client,
            user_effort=user_effort,
            policy=policy,
            caps=caps,
            registry=_make_capability_registry(),
            conversation_store=conv_store,
            session_id=session_id,
            memory_directive=memory_directive,
            mcp_handlers=_MCP_HANDLERS,
        )
        total_ms = (time.time() - request_start) * 1000

        log.info(
            "[%s] /api/v2/trace done | route=%s | total_ms=%d | cost=$%.4f",
            request_id, result.route.value, int(total_ms),
            result.budget_summary.get("cost_usd", 0.0),
        )

        # Per-request LLM-call breakdown — model × count × latency × cost.
        # Emits a multi-line summary block to the constellax.obs logger.
        try:
            client.summarize_calls(request_id=request_id)
        except Exception:
            log.exception("[%s] summarize_calls failed (non-fatal)", request_id)

        # Pull the formation plan (and engine convergence stats) out of
        # engine_result so the UI can render which agents actually ran.
        # Only DEEP routes have these — TRIVIAL/DIRECT/DIRECT_PLUS skip
        # the engine entirely.
        formation_plan_dict = None
        actual_iterations = None
        actual_active_domains = []
        if result.engine_result is not None:
            formation_plan_dict = serialize_formation_plan(
                result.engine_result.formation_plan
            )
            ch = result.engine_result.convergence_history
            actual_iterations = ch.total_iterations or len(ch.snapshots)
            actual_active_domains = [
                d.value for d in result.engine_result.formation_plan.active_domains
            ]

        # Per-trajectory cards for the UI. Empty list when the engine
        # didn't run; the UI then renders only the advisor voice.
        perspectives = serialize_perspectives(result.engine_result)

        payload = {
            "route": result.route.value,
            "response_text": result.response_text,
            "triage": {
                "route": result.triage_result.route.value,
                "recommended_effort": result.triage_result.recommended_effort.value,
                "interrupt": result.triage_result.interrupt,
                "risk_flags": result.triage_result.risk_flags,
                "mcps_needed": [
                    {"name": m.name, "why": m.why, "required": m.required}
                    for m in result.triage_result.mcps_needed
                ],
                "why": result.triage_result.why,
                "classifier_mode": result.triage_result.classifier_mode,
            },
            # Formation plan — the engine's per-request dispatch decision.
            # null for non-DEEP routes (no engine ran). When set, the UI
            # can render only the agents that actually fired.
            "formation_plan": formation_plan_dict,
            "actual_iterations": actual_iterations,
            "actual_active_domains": actual_active_domains,
            # Trajectory cards — one entry per finding the engine surfaced.
            # Empty when the engine didn't run (TRIVIAL/DIRECT/DIRECT_PLUS).
            "perspectives": perspectives,
            # Structured peer memo (verdict / reasoning / alternatives /
            # falsifiers / open_questions / confidence / visuals). Populated
            # only for DEEP route; the frontend's Thinking Room + Map Room
            # render from this directly. None for trivial/direct routes —
            # the frontend then renders response_text as legacy prose.
            "memo": result.memo,
            # Thread ID — the key under which the browser-tab Map Room
            # (/map/<thread_id>) fetches this memo, AND the conversation
            # identifier the frontend echoes back as parent_thread_id on
            # followups so multi-turn chats accumulate iterations on the
            # same thread. If the request carried a parent_thread_id we
            # mirror it back; otherwise derive a fresh thr-<request_id>
            # — same logic as _resolve_thread_id in thread_persistence.py.
            "thread_id": (body.get("parent_thread_id") or f"thr-{request_id}"),
            "budget": result.budget_summary,
            "capabilities": result.capability_state,
            "escalation_offer": result.escalation_offer,
            "missing_capability_offers": result.missing_capability_offers,
            "debug": {
                **result.debug,
                "request_id": request_id,
                "total_ms": round(total_ms),
                "web_search": web_search_meta,
            },
        }

        # Cache the memo for the browser-tab Map Room. Only DEEP route
        # produces a memo worth caching; non-deep routes are chat-style
        # replies that don't have a Map Room view.
        if result.memo:
            _cache_memo(request_id, question=question, payload=payload)

        # Memory pipeline — persist this iteration to FalkorDB as a
        # ThreadRecord + IterationRecord with embedding + extracted metadata
        # (entities, tags, user_mode, time_horizon, load_bearing_assumption).
        # Fire-and-forget; failures inside never affect this response.
        try:
            _persist_iteration(
                request_id=request_id,
                question=question,
                dispatch_result=result,
                triage=payload["triage"],
                cached_memo=result.memo,
                user_id=body.get("user_id"),
                project_id=project_id,
                workspace_id=body.get("workspace_id"),
                surface_id=body.get("surface_id"),
                parent_thread_id=body.get("parent_thread_id"),
                effort_picked=user_effort.value,
                started_at=request_start,
                web_search_meta=web_search_meta,
                title_question=original_question,
            )
        except Exception as e:
            log.warning("[%s] thread persistence enqueue failed (non-fatal): %s", request_id, e)

        return JSONResponse(payload)

    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] /api/v2/trace failed", request_id)
        return JSONResponse(
            {
                "error": "Internal server error",
                "message": str(exc),
                "request_id": request_id,
            },
            status_code=500,
        )


# ---------------------------------------------------------------------------
# POST /api/v2/trace/segment  —  per-phase streaming endpoint
# ---------------------------------------------------------------------------
#
# Replaces the timer-faked "3-segment streaming" the frontend used to
# do client-side with real, server-driven segment generation. Three
# phases:
#
#   1. synthesizer  — first call. Runs the same dispatch flow as
#                     /api/v2/trace (engine + speech), caches the
#                     resulting full memo + speech_input under a
#                     memo_id (= thread_id), and returns ONLY the
#                     verdict slice. Same shape as the synthesizer
#                     fields on the full memo.
#
#   2. opinion      — second call. Looks up the cache by memo_id.
#                     With no splice: returns the cached reasoning/
#                     alternatives/falsifiers slice instantly (no
#                     LLM call). With a splice: calls
#                     `generate_opinion_segment` to re-shape just
#                     this phase using the user's mid-stream input,
#                     merges back into the cache, returns the new
#                     slice.
#
#   3. prospects    — third call. Same shape as opinion but for the
#                     open_questions + visuals slice.
#
# /api/v2/trace remains unchanged — the Map Room rehydration path,
# the escalation/resume flow, and any non-streaming consumer keep
# working exactly as before.
# ---------------------------------------------------------------------------

@app.post("/api/v2/trace/segment")
async def trace_segment(request: Request) -> JSONResponse:
    request_id = f"req-{int(time.time() * 1000)}"
    request_start = time.time()

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    phase = str(body.get("phase", "")).strip().lower()
    if phase not in ("synthesizer", "opinion", "prospects", "clarify"):
        return JSONResponse(
            {"error": "phase must be one of: synthesizer | opinion | prospects | clarify"},
            status_code=400,
        )

    # ─── Phase: clarify ────────────────────────────────────────────────
    # Multi-turn Q&A inside the breathing room between synthesizer and
    # opinion. Each clarify call answers ONE follow-up question and
    # stores the (Q, A) pair on the cache. When the opinion phase
    # finally fires the engine, the accumulated clarifications get
    # threaded into the dispatch question so the engine sees them.
    if phase == "clarify":
        memo_id = str(body.get("memo_id", "")).strip()
        if not memo_id:
            return JSONResponse(
                {"error": "memo_id is required for clarify phase"},
                status_code=400,
            )
        clarify_question = str(body.get("question", "")).strip()
        if not clarify_question:
            return JSONResponse(
                {"error": "Field 'question' is required for clarify phase"},
                status_code=400,
            )
        if len(clarify_question) > MAX_QUESTION_CHARS:
            return JSONResponse(
                {"error": f"Clarification question too long. Max {MAX_QUESTION_CHARS} characters."},
                status_code=400,
            )

        entry = _get_cached_segments(memo_id)
        if entry is None:
            return JSONResponse(
                {
                    "error":  "memo_id not found or expired",
                    "memo_id": memo_id,
                    "hint":   "Re-run phase=synthesizer to start a fresh stream.",
                },
                status_code=404,
            )

        extra = entry.get("extra") or {}
        synth_fields = _synthesizer_slice(entry.get("full_memo") or {})
        # If full_memo is empty (DEEP route synth-only), fall back to the
        # synth slice we cached separately. We stash that under
        # extra["synth_fields_only"] when synth-only completes.
        if not synth_fields.get("verdict_line") and not synth_fields.get("verdict_body"):
            synth_fields = extra.get("synth_fields_only") or synth_fields
        parent_question = extra.get("original_question") or ""
        prior_qas: list[dict] = extra.get("clarifications") or []

        # Cap to keep prompt size bounded — 5 Q&As is plenty.
        if len(prior_qas) >= 5:
            return JSONResponse({
                "phase":   "clarify",
                "memo_id": memo_id,
                "answer":  "We've covered five clarifications — let's move into the deep analysis where the bigger picture lands.",
                "qa_count": len(prior_qas),
                "capped":   True,
            })

        try:
            client = LLMClient(mode=(ClientMode.LIVE if _has_live_key() else ClientMode.MOCK))
            answer = await generate_clarification(
                client=client,
                parent_question=parent_question,
                synth_fields=synth_fields,
                prior_qas=prior_qas,
                new_question=clarify_question,
            )
            try:
                client.summarize_calls(request_id=request_id)
            except Exception:
                log.exception("[%s] summarize_calls failed (non-fatal)", request_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] clarify generation failed", request_id)
            return JSONResponse({
                "phase":   "clarify",
                "memo_id": memo_id,
                "answer":  f"I couldn't reach the model for that one ({type(exc).__name__}). Continuing with the deep analysis when you're ready.",
                "qa_count": len(prior_qas),
                "error":    True,
            })

        # Persist (Q, A) into the cache so opinion phase sees it.
        with _SEGMENT_CACHE_LOCK:
            ce = _SEGMENT_CACHE.get(memo_id)
            if ce is not None:
                ce_extra = ce.get("extra") or {}
                qas = list(ce_extra.get("clarifications") or [])
                qas.append({"q": clarify_question, "a": answer})
                ce_extra["clarifications"] = qas
                ce["extra"] = ce_extra
                ce["_cached_at"] = time.time()
                qa_count_after = len(qas)
            else:
                qa_count_after = len(prior_qas) + 1

        log.info(
            "[%s] clarify memo_id=%s qa_count=%d qlen=%d alen=%d",
            request_id, memo_id, qa_count_after,
            len(clarify_question), len(answer),
        )
        return JSONResponse({
            "phase":    "clarify",
            "memo_id":  memo_id,
            "answer":   answer,
            "qa_count": qa_count_after,
        })

    splice = body.get("splice")
    if splice is not None and not isinstance(splice, str):
        return JSONResponse({"error": "splice must be a string"}, status_code=400)
    if isinstance(splice, str):
        splice = splice.strip()
        if len(splice) > MAX_QUESTION_CHARS:
            return JSONResponse(
                {"error": f"splice too long. Max {MAX_QUESTION_CHARS} characters."},
                status_code=400,
            )

    # ─── Phase: opinion / prospects ────────────────────────────────────
    # opinion = where the Wu Xing engine runs (if it hasn't yet) +
    # opinion narration. This is the long-running phase for DEEP
    # routes — the user has just advanced past the synthesizer
    # breathing room and is now committed to the deep analysis.
    # prospects = engine is cached, narrate the prospects slice +
    # fire persistence at the end-of-stream.
    if phase in ("opinion", "prospects"):
        memo_id = str(body.get("memo_id", "")).strip()
        if not memo_id:
            return JSONResponse(
                {"error": "memo_id is required for opinion/prospects phases"},
                status_code=400,
            )

        entry = _get_cached_segments(memo_id)
        if entry is None:
            return JSONResponse(
                {
                    "error":  "memo_id not found or expired",
                    "memo_id": memo_id,
                    "hint":   "Re-run phase=synthesizer to start a fresh stream.",
                },
                status_code=404,
            )

        extra = entry.get("extra") or {}
        cached_memo: dict = entry["full_memo"]
        speech_input = entry["speech_input"]
        deep_route = extra.get("route") == Route.DEEP.value

        # ─── Engine-not-yet-run path ────────────────────────────────
        # On a DEEP route, the synth phase produced only the verdict
        # slice; the engine is still pending. The first opinion call
        # is where it actually fires.
        if phase == "opinion" and deep_route and not extra.get("engine_already_ran"):
            try:
                client = LLMClient(mode=(ClientMode.LIVE if _has_live_key() else ClientMode.MOCK))
                conv_store = _make_conv_store(extra.get("project_id")) if extra.get("session_id") else None
                caps_dict = extra.get("caps") or {}
                caps = BudgetCaps(
                    max_iterations=int(caps_dict.get("max_iterations", 12)),
                    max_wall_time_sec=float(caps_dict.get("max_wall_time_sec", 720.0)),
                    max_cost_usd=float(caps_dict.get("max_cost_usd", 1.00)),
                    max_mcp_calls=int(caps_dict.get("max_mcp_calls", 20)),
                )
                user_effort = normalize_effort(extra.get("effort"))
                policy = extra.get("policy") or "ask"
                question_augmented = extra.get("question_augmented") or ""
                original_question = extra.get("original_question") or question_augmented

                # Thread the breathing-room clarifications into the
                # question the engine sees. The Q&As from phase=clarify
                # are conversational context — the engine should reason
                # about the user's REFINED understanding, not just the
                # original prompt. Capped at 5 per phase=clarify guard.
                clarifications = extra.get("clarifications") or []
                if clarifications:
                    clarify_block_lines = [
                        "",
                        "[CLARIFICATIONS — the user asked the following follow-up "
                        "questions during the breathing room AFTER reading the "
                        "synthesizer verdict, and you answered each one. Treat "
                        "these as part of the question you are reasoning about. "
                        "Do NOT reanswer them; let them shape your read.]",
                    ]
                    for i, qa in enumerate(clarifications, 1):
                        qq = (qa.get("q") or "").strip()
                        aa = (qa.get("a") or "").strip()
                        if qq:
                            clarify_block_lines.append(f"\nFOLLOWUP {i}: {qq}")
                        if aa:
                            clarify_block_lines.append(f"YOUR ANSWER: {aa}")
                    question_augmented = (
                        question_augmented + "\n" + "\n".join(clarify_block_lines)
                    )
                    log.info(
                        "[%s] opinion: threaded %d clarification(s) into dispatch question",
                        request_id, len(clarifications),
                    )

                # Pass synth-phase triage result via force_triage_result.
                # This bypasses dispatch's internal triage so the
                # augmented question (with clarifications appended)
                # cannot be reclassified to DIRECT/DIRECT_PLUS — which
                # was producing an empty memo in user-visible runs.
                forced_triage = extra.get("triage_result_obj")
                log.info(
                    "[%s] /api/v2/trace/segment phase=opinion engine starting | memo_id=%s | effort=%s | clarifications=%d | force_triage=%s",
                    request_id, memo_id, user_effort.value, len(clarifications),
                    "yes" if forced_triage is not None else "no",
                )
                result: DispatchResult = await dispatch(
                    text=question_augmented,
                    client=client,
                    user_effort=user_effort,
                    policy=policy,
                    caps=caps,
                    registry=_make_capability_registry(),
                    conversation_store=conv_store,
                    session_id=extra.get("session_id"),
                    force_triage_result=forced_triage,
                    mcp_handlers=_MCP_HANDLERS,
                )
                engine_ms = (time.time() - request_start) * 1000
                log.info(
                    "[%s] /api/v2/trace/segment opinion engine done | route=%s | ms=%d | cost=$%.4f",
                    request_id, result.route.value, int(engine_ms),
                    result.budget_summary.get("cost_usd", 0.0),
                )
                try:
                    client.summarize_calls(request_id=request_id)
                except Exception:
                    log.exception("[%s] summarize_calls failed (non-fatal)", request_id)

                # Merge engine result into cache.
                new_full_memo = result.memo or {}
                speech_input = None
                if result.engine_result is not None:
                    try:
                        speech_input = extract_speech_input(
                            result.engine_result,
                            user_original_text=original_question,
                            is_phase_one=False,
                        )
                    except Exception:
                        log.exception("[%s] extract_speech_input failed (non-fatal)", request_id)

                # Update cache in-place — same memo_id, now with engine
                # state and the full memo from generate_speech.
                with _SEGMENT_CACHE_LOCK:
                    cache_entry = _SEGMENT_CACHE.get(memo_id)
                    if cache_entry is not None:
                        cache_entry["full_memo"] = dict(new_full_memo)
                        cache_entry["speech_input"] = speech_input
                        cache_entry["extra"]["engine_already_ran"] = True
                        cache_entry["extra"]["dispatch_route"] = result.route.value
                        cache_entry["extra"]["dispatch_result"] = result
                        cache_entry["_cached_at"] = time.time()
                cached_memo = new_full_memo

                # ─── Visual pipeline (Codex architecture) ─────────────
                # Classifier → spec generator → validator runs here, on
                # the canonical engine-built memo. Speech.py's free-form
                # visuals[] emission is intentionally quiet now — the
                # procedural pipeline owns Map Room visuals end-to-end.
                # Failures degrade silently to visuals=[] (Map Room
                # renders that case gracefully).
                if new_full_memo:
                    try:
                        visuals = await build_visuals(
                            client=client,
                            memo=new_full_memo,
                            question=question_augmented,
                        )
                    except Exception:
                        log.exception("[%s] opinion: visualizer pipeline failed (non-fatal)", request_id)
                        visuals = []

                    if visuals:
                        new_full_memo["visuals"] = visuals
                        with _SEGMENT_CACHE_LOCK:
                            ce = _SEGMENT_CACHE.get(memo_id)
                            if ce is not None and isinstance(ce.get("full_memo"), dict):
                                ce["full_memo"]["visuals"] = visuals
                        cached_memo = new_full_memo
                    log.info(
                        "[%s] opinion: visuals built (%d emitted)",
                        request_id, len(visuals),
                    )

                # Diagnostic: when synth said DEEP but dispatch's
                # internal triage on the augmented question reclassified
                # to non-DEEP, the engine doesn't run inside dispatch
                # and `memo` comes back None/empty. Surface this loudly
                # so we can see it in logs + flag it to the frontend.
                synth_route = extra.get("route")
                if synth_route == Route.DEEP.value and result.route != Route.DEEP:
                    log.warning(
                        "[%s] ROUTE DRIFT — synth said DEEP, dispatch with "
                        "augmented question routed to %s. Full memo will be "
                        "empty. Augmented question length: %d",
                        request_id, result.route.value, len(question_augmented),
                    )
                if not new_full_memo:
                    log.warning(
                        "[%s] opinion: full_memo is empty after dispatch "
                        "(route=%s). Frontend will render the placeholder.",
                        request_id, result.route.value,
                    )

                # ─── Persist NOW (at end of opinion, not prospects) ────
                # Engine work is done. Persistence is fire-and-forget; by
                # the time the user reaches prospects + clicks Reasoning
                # Trace, the Falkor write has had ~5-15 seconds to land.
                # Fixes the "thread isn't indexed yet" race the user hit.
                if synth_route == Route.DEEP.value and not extra.get("persisted"):
                    _persist_after_engine_done(
                        request_id=request_id, memo_id=memo_id,
                    )
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s] /api/v2/trace/segment opinion engine failed", request_id)
                return JSONResponse({
                    "phase":     "opinion",
                    "memo_id":   memo_id,
                    "thread_id": entry.get("thread_id"),
                    "segment":   _opinion_slice(cached_memo),
                    "regenerated": False,
                    "error":     f"engine failed: {type(exc).__name__}: {exc}",
                    "next_phase": "prospects",
                    "done":      False,
                }, status_code=500)

        # ─── Splice-free path: return cached slice instantly ─────────
        if not splice:
            if phase == "opinion":
                slice_fields = _opinion_slice(cached_memo)
            else:
                slice_fields = _prospects_slice(cached_memo)
                # End-of-stream persistence for DEEP routes — engine
                # ran in opinion phase, prospects landing is the
                # canonical end of stream.
                if deep_route and not extra.get("persisted"):
                    _persist_after_engine_done(
                        request_id=request_id, memo_id=memo_id,
                    )
            log.info(
                "[%s] /api/v2/trace/segment phase=%s memo_id=%s splice=no cached_slice ms=%d",
                request_id, phase, memo_id,
                int((time.time() - request_start) * 1000),
            )
            return JSONResponse({
                "phase":     phase,
                "memo_id":   memo_id,
                "thread_id": entry.get("thread_id"),
                "segment":   slice_fields,
                "regenerated": False,
                "next_phase": "prospects" if phase == "opinion" else None,
                "done":      phase == "prospects",
            })

        # ─── Splice path: regenerate this phase with the user's input ─
        if speech_input is None:
            log.info(
                "[%s] /api/v2/trace/segment phase=%s memo_id=%s splice=yes (no speech_input, returning cached)",
                request_id, phase, memo_id,
            )
            slice_fields = (
                _opinion_slice(cached_memo) if phase == "opinion"
                else _prospects_slice(cached_memo)
            )
            if phase == "prospects" and deep_route and not extra.get("persisted"):
                _persist_after_engine_done(request_id=request_id, memo_id=memo_id)
            return JSONResponse({
                "phase":       phase,
                "memo_id":     memo_id,
                "thread_id":   entry.get("thread_id"),
                "segment":     slice_fields,
                "regenerated": False,
                "next_phase":  "prospects" if phase == "opinion" else None,
                "done":        phase == "prospects",
            })

        try:
            client = LLMClient(mode=(ClientMode.LIVE if _has_live_key() else ClientMode.MOCK))
            if phase == "opinion":
                seg: SegmentMemo = await generate_opinion_segment(
                    client=client,
                    speech_input=speech_input,
                    prior_segments=cached_memo,
                    splice=splice,
                )
            else:
                seg = await generate_prospects_segment(
                    client=client,
                    speech_input=speech_input,
                    prior_segments=cached_memo,
                    splice=splice,
                )

            try:
                client.summarize_calls(request_id=request_id)
            except Exception:
                log.exception("[%s] summarize_calls failed (non-fatal)", request_id)

            if seg.success and seg.fields:
                _update_cached_segment(memo_id, seg.fields)
                slice_fields = seg.fields
                regenerated = True
            else:
                log.warning(
                    "[%s] /api/v2/trace/segment phase=%s splice regen failed, using cached slice",
                    request_id, phase,
                )
                slice_fields = (
                    _opinion_slice(cached_memo) if phase == "opinion"
                    else _prospects_slice(cached_memo)
                )
                regenerated = False

            if phase == "prospects" and deep_route and not extra.get("persisted"):
                _persist_after_engine_done(request_id=request_id, memo_id=memo_id)

            log.info(
                "[%s] /api/v2/trace/segment phase=%s memo_id=%s splice=yes regen=%s ms=%d",
                request_id, phase, memo_id, regenerated,
                int((time.time() - request_start) * 1000),
            )
            return JSONResponse({
                "phase":       phase,
                "memo_id":     memo_id,
                "thread_id":   entry.get("thread_id"),
                "segment":     slice_fields,
                "regenerated": regenerated,
                "next_phase":  "prospects" if phase == "opinion" else None,
                "done":        phase == "prospects",
            })
        except Exception as exc:  # noqa: BLE001
            log.exception("[%s] /api/v2/trace/segment (%s splice) failed", request_id, phase)
            slice_fields = (
                _opinion_slice(cached_memo) if phase == "opinion"
                else _prospects_slice(cached_memo)
            )
            return JSONResponse({
                "phase":       phase,
                "memo_id":     memo_id,
                "thread_id":   entry.get("thread_id"),
                "segment":     slice_fields,
                "regenerated": False,
                "error":       f"{type(exc).__name__}: {exc}",
                "next_phase":  "prospects" if phase == "opinion" else None,
                "done":        phase == "prospects",
            })

    # ─── Phase: synthesizer ────────────────────────────────────────────
    # Same input contract as /api/v2/trace. Runs dispatch normally,
    # caches the full memo, returns ONLY the synthesizer slice.

    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "Field 'question' is required"}, status_code=400)
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse(
            {"error": f"Question too long. Max {MAX_QUESTION_CHARS} characters."},
            status_code=400,
        )
    original_question = question

    # Attachments — mirror /api/v2/trace.
    attachments_in = body.get("attachments") or []
    if not isinstance(attachments_in, list):
        return JSONResponse(
            {"error": "attachments must be a list of {name, content} objects"},
            status_code=400,
        )
    attachment_blocks: list[str] = []
    MAX_ATTACHMENT_BYTES = 50_000
    total_attachment_size = 0
    for att in attachments_in[:3]:
        if not isinstance(att, dict):
            continue
        att_name = str(att.get("name", "attachment.csv"))[:128]
        att_content = att.get("content")
        if not isinstance(att_content, str):
            continue
        if len(att_content) > MAX_ATTACHMENT_BYTES:
            return JSONResponse(
                {"error":
                    f"Attachment '{att_name}' is {len(att_content)} bytes; "
                    f"max {MAX_ATTACHMENT_BYTES} bytes per file."},
                status_code=400,
            )
        total_attachment_size += len(att_content)
        if total_attachment_size > MAX_ATTACHMENT_BYTES * 2:
            return JSONResponse(
                {"error": "Combined attachment size exceeds budget."},
                status_code=400,
            )
        attachment_blocks.append(
            f"[ATTACHED_CSV name=\"{att_name}\"]\n{att_content.rstrip()}\n[/ATTACHED_CSV]"
        )
    if attachment_blocks:
        question = "\n\n".join(attachment_blocks) + "\n\n" + question

    # Web search — mirror /api/v2/trace.
    web_context = ""
    web_search_meta: dict | None = None
    try:
        decision = await _route_search(question)
        web_search_meta = {
            "decision_via":  decision.via,
            "needs_search":  decision.needs_search,
            "router_reason": decision.reason,
            "router_model":  decision.model,
            "router_ms":     decision.latency_ms,
            "refined_query": decision.refined_query if decision.needs_search else "",
            "results":       [],
            "provider":      None,
            "search_ms":     0,
            "cached":        False,
            "answer":        None,
        }
        if decision.needs_search and decision.refined_query:
            try:
                ws_result = await _web_search(decision.refined_query, max_results=5)
                web_search_meta["provider"]  = ws_result.provider
                web_search_meta["search_ms"] = ws_result.latency_ms
                web_search_meta["cached"]    = ws_result.cached
                web_search_meta["answer"]    = ws_result.answer
                web_search_meta["results"]   = [
                    {"title": h.title, "url": h.url, "snippet": h.snippet}
                    for h in ws_result.hits
                ]
                if ws_result.ok:
                    web_context = _format_web_context_block(ws_result)
                else:
                    web_search_meta["error"] = ws_result.error
            except Exception as e:
                log.warning("[%s] search.run raised (non-fatal): %s", request_id, e)
                web_search_meta["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        log.warning("[%s] search routing raised (non-fatal): %s", request_id, e)
        web_search_meta = {"error": f"{type(e).__name__}: {e}", "decision_via": "error"}
    if web_context:
        question = web_context + "\n" + question

    user_effort = normalize_effort(body.get("effort", DEFAULT_EFFORT))
    policy = str(body.get("policy", "ask")).strip().lower()
    if policy not in ("strict", "ask", "auto"):
        return JSONResponse(
            {"error": "policy must be one of: strict | ask | auto"},
            status_code=400,
        )
    caps_in = body.get("caps") or {}
    try:
        caps = BudgetCaps(
            max_iterations=int(caps_in.get("max_iterations", 12)),
            max_wall_time_sec=float(caps_in.get("max_wall_time_sec", 720.0)),
            max_cost_usd=float(caps_in.get("max_cost_usd", 1.00)),
            max_mcp_calls=int(caps_in.get("max_mcp_calls", 20)),
        )
    except (TypeError, ValueError):
        return JSONResponse({"error": "caps fields must be numeric"}, status_code=400)

    session_id = body.get("session_id")
    project_id = body.get("project_id")
    if session_id is not None:
        session_id = str(session_id).strip() or None
    if project_id is not None:
        project_id = str(project_id).strip() or None

    log.info(
        "[%s] /api/v2/trace/segment phase=synthesizer | qlen=%d | effort=%s | policy=%s | session=%s",
        request_id, len(question), user_effort.value, policy,
        session_id or "(none)",
    )

    try:
        client = LLMClient(mode=(ClientMode.LIVE if _has_live_key() else ClientMode.MOCK))
        conv_store = _make_conv_store(project_id) if session_id else None

        # ─── Triage first — decides whether to skip the engine ────────
        # Cheap classifier (~500ms live, instant in mock). Determines
        # whether this question needs the full Wu Xing engine. If yes,
        # we DEFER the engine to the opinion phase and produce only a
        # synth-only first-read here. If no, we run dispatch normally
        # and return the full memo in one shot (same as legacy
        # /api/v2/trace behavior).
        triage_result = await run_triage(text=question, client=client)
        log.info(
            "[%s] triage route=%s effort=%s mode=%s",
            request_id, triage_result.route.value,
            triage_result.recommended_effort.value,
            triage_result.classifier_mode,
        )
        triage_payload = {
            "route": triage_result.route.value,
            "recommended_effort": triage_result.recommended_effort.value,
            "interrupt": triage_result.interrupt,
            "risk_flags": triage_result.risk_flags,
            "mcps_needed": [
                {"name": m.name, "why": m.why, "required": m.required}
                for m in triage_result.mcps_needed
            ],
            "why": triage_result.why,
            "classifier_mode": triage_result.classifier_mode,
        }

        # DEEP route → skip dispatch in synthesizer phase. The engine
        # fires in opinion phase. User gets a fast first read here.
        if triage_result.route == Route.DEEP:
            try:
                # Quick key-phrase extraction (reuses the helper that
                # the full speech module uses) — gives the synth-only
                # prompt something to mirror without requiring engine
                # output.
                from src.llm.speech import _extract_key_phrases
                key_phrases = _extract_key_phrases(original_question)
            except Exception:
                key_phrases = None

            synth_seg: SegmentMemo = await generate_synthesizer_only(
                client=client,
                question=original_question,
                user_key_phrases=key_phrases,
            )
            synth_ms = (time.time() - request_start) * 1000
            log.info(
                "[%s] /api/v2/trace/segment synth-only done (DEEP, no engine yet) | ms=%d",
                request_id, int(synth_ms),
            )
            try:
                client.summarize_calls(request_id=request_id)
            except Exception:
                log.exception("[%s] summarize_calls failed (non-fatal)", request_id)

            synth_fields = synth_seg.fields if synth_seg.success else {
                "verdict_line": "",
                "verdict_body": "Working on your question. The deep analysis is coming next.",
                "confidence":   "moderate",
            }

            # Honesty cap: the synth-only read is engine-free. The Wu
            # Xing pipeline has not run yet, no domains have stress-
            # tested anything, no Ke cycle has challenged the verdict.
            # Claiming "high confidence" at this point is a lie — the
            # opinion segment is where the real read gets earned. Cap
            # at moderate. Engine-confidence (which may be high) will
            # override this when the opinion phase merges the full
            # memo back into the cache.
            if synth_fields.get("confidence") == "high":
                synth_fields["confidence"] = "moderate"

            memo_id = body.get("parent_thread_id") or f"thr-{request_id}"

            # Seed the cache with synth-only state. opinion phase reads
            # `extra` to run dispatch with the right context.
            _cache_segments(
                memo_id,
                speech_input=None,        # filled in when opinion runs the engine
                full_memo={},              # filled in when opinion runs the engine
                question=question,
                thread_id=memo_id,
                extra={
                    "route":             triage_result.route.value,
                    "question_augmented": question,
                    "original_question": original_question,
                    "effort":            user_effort.value,
                    "policy":            policy,
                    "caps": {
                        "max_iterations":    caps.max_iterations,
                        "max_wall_time_sec": caps.max_wall_time_sec,
                        "max_cost_usd":      caps.max_cost_usd,
                        "max_mcp_calls":     caps.max_mcp_calls,
                    },
                    "session_id":        session_id,
                    "project_id":        project_id,
                    "user_id":           body.get("user_id"),
                    "workspace_id":      body.get("workspace_id"),
                    "surface_id":        body.get("surface_id"),
                    "parent_thread_id":  body.get("parent_thread_id"),
                    "web_search_meta":   web_search_meta,
                    "triage_payload":    triage_payload,
                    # The actual TriageResult object — opinion phase
                    # passes this to dispatch() via force_triage_result
                    # so internal triage doesn't reclassify the
                    # augmented question (with clarifications appended)
                    # to a non-DEEP route. Without this we got an
                    # empty memo because dispatch fell to DIRECT.
                    "triage_result_obj": triage_result,
                    "engine_already_ran": False,
                    "persisted":         False,
                    # opinion phase needs these for the dispatch call:
                    "request_start":     request_start,
                    # Synth-only fields stashed so clarify can read them
                    # before the engine has populated full_memo.
                    "synth_fields_only": synth_fields,
                    # Clarification Q&A list — populated by phase=clarify.
                    "clarifications":    [],
                },
            )
            # Also store request_start at top level for persistence helper
            with _SEGMENT_CACHE_LOCK:
                ce = _SEGMENT_CACHE.get(memo_id)
                if ce is not None:
                    ce["request_start"] = request_start

            total_ms = (time.time() - request_start) * 1000
            return JSONResponse({
                "phase":           "synthesizer",
                "memo_id":         memo_id,
                "thread_id":       memo_id,
                "route":           triage_result.route.value,
                # No engine ran yet — the heavy fields are empty/null.
                "response_text":   "",
                "perspectives":    [],
                "formation_plan":  None,
                "actual_iterations": None,
                "actual_active_domains": [],
                "escalation_offer":  None,
                "missing_capability_offers": [],
                "capabilities":      {},
                "triage":            triage_payload,
                "budget":            {},
                "segment":           synth_fields,
                "memo":              None,        # full memo arrives at end of prospects
                "regenerated":       False,
                "next_phase":        "opinion",
                "done":              False,
                "debug": {
                    "request_id": request_id,
                    "total_ms":   round(total_ms),
                    "engine_pending": True,
                    "web_search": web_search_meta,
                    "triage_ms":  int(synth_ms),
                },
            })

        # Non-DEEP route → keep the original one-shot dispatch behavior.
        # Triage already ran; dispatch will re-run it internally (cheap)
        # and route appropriately. Persistence fires here since the
        # full memo IS complete in one round-trip for these routes.
        result: DispatchResult = await dispatch(
            text=question,
            client=client,
            user_effort=user_effort,
            policy=policy,
            caps=caps,
            registry=_make_capability_registry(),
            conversation_store=conv_store,
            session_id=session_id,
            mcp_handlers=_MCP_HANDLERS,
        )
        total_ms = (time.time() - request_start) * 1000
        log.info(
            "[%s] /api/v2/trace/segment(synthesizer) done | route=%s | ms=%d | cost=$%.4f",
            request_id, result.route.value, int(total_ms),
            result.budget_summary.get("cost_usd", 0.0),
        )
        try:
            client.summarize_calls(request_id=request_id)
        except Exception:
            log.exception("[%s] summarize_calls failed (non-fatal)", request_id)

        # Memo id = thread id (mirrors /api/v2/trace).
        memo_id = body.get("parent_thread_id") or f"thr-{request_id}"

        # Pull engine context for segment regen on splice. None for non-DEEP routes.
        speech_input = None
        if result.engine_result is not None:
            try:
                speech_input = extract_speech_input(
                    result.engine_result,
                    user_original_text=original_question,
                    is_phase_one=False,   # segment endpoint always runs the full pipe
                )
            except Exception:
                log.exception("[%s] extract_speech_input failed (non-fatal)", request_id)

        formation_plan_dict = None
        actual_iterations = None
        actual_active_domains = []
        if result.engine_result is not None:
            formation_plan_dict = serialize_formation_plan(
                result.engine_result.formation_plan
            )
            ch = result.engine_result.convergence_history
            actual_iterations = ch.total_iterations or len(ch.snapshots)
            actual_active_domains = [
                d.value for d in result.engine_result.formation_plan.active_domains
            ]
        perspectives = serialize_perspectives(result.engine_result)

        full_memo = result.memo or {}
        syn_slice = _synthesizer_slice(full_memo) if full_memo else {
            # Non-DEEP routes don't carry a memo — derive a minimal
            # synthesizer slice from response_text so the frontend can
            # still render something. The downstream phases will return
            # empty slices and the UI degrades cleanly.
            "verdict_line": "",
            "verdict_body": result.response_text or "",
            "confidence":   "moderate",
        }
        next_phase = "opinion" if result.memo else None

        # Cache for subsequent phase calls.
        _cache_segments(
            memo_id,
            speech_input=speech_input,
            full_memo=full_memo,
            question=question,
            thread_id=memo_id,
            extra={
                "route":            result.route.value,
                "response_text":    result.response_text,
                "perspectives":     perspectives,
                "escalation_offer": result.escalation_offer,
            },
        )

        payload = {
            "phase":           "synthesizer",
            "memo_id":         memo_id,
            "thread_id":       memo_id,
            "route":           result.route.value,
            "response_text":   result.response_text,
            "perspectives":    perspectives,
            "formation_plan":  formation_plan_dict,
            "actual_iterations": actual_iterations,
            "actual_active_domains": actual_active_domains,
            "escalation_offer":  result.escalation_offer,
            "missing_capability_offers": result.missing_capability_offers,
            "capabilities":      result.capability_state,
            "triage": {
                "route": result.triage_result.route.value,
                "recommended_effort": result.triage_result.recommended_effort.value,
                "interrupt": result.triage_result.interrupt,
                "risk_flags": result.triage_result.risk_flags,
                "mcps_needed": [
                    {"name": m.name, "why": m.why, "required": m.required}
                    for m in result.triage_result.mcps_needed
                ],
                "why": result.triage_result.why,
                "classifier_mode": result.triage_result.classifier_mode,
            },
            "budget":          result.budget_summary,
            "segment":         syn_slice,
            # Full memo also surfaced — the frontend uses it for the
            # Map Room link + so opinion/prospects rehydrate from this
            # response if the user navigates away. The streaming code
            # path slices into it; the rehydration code path consumes
            # it whole. Same wire shape as /api/v2/trace.memo.
            "memo":            result.memo,
            "regenerated":     False,
            "next_phase":      next_phase,
            "done":            next_phase is None,
            "debug": {
                **result.debug,
                "request_id": request_id,
                "total_ms":   round(total_ms),
                "web_search": web_search_meta,
            },
        }

        # Mirror /api/v2/trace's memo cache + persistence so the Map Room
        # and history sidebar see this trace exactly as they would on a
        # non-segmented run.
        if result.memo:
            _cache_memo(memo_id, question=question, payload={
                **payload,
                # Cache uses the same key names as /api/v2/trace so the
                # legacy GET /api/v2/thread/<id> consumer keeps working.
                "response_text": result.response_text,
            })

        try:
            _persist_iteration(
                request_id=request_id,
                question=question,
                dispatch_result=result,
                triage=payload["triage"],
                cached_memo=result.memo,
                user_id=body.get("user_id"),
                project_id=project_id,
                workspace_id=body.get("workspace_id"),
                surface_id=body.get("surface_id"),
                parent_thread_id=body.get("parent_thread_id"),
                effort_picked=user_effort.value,
                started_at=request_start,
                web_search_meta=web_search_meta,
                title_question=original_question,
            )
        except Exception as e:
            log.warning(
                "[%s] thread persistence enqueue failed (non-fatal): %s",
                request_id, e,
            )

        return JSONResponse(payload)

    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] /api/v2/trace/segment(synthesizer) failed", request_id)
        return JSONResponse(
            {
                "error":      "Internal server error",
                "message":    str(exc),
                "request_id": request_id,
            },
            status_code=500,
        )


@app.post("/api/v2/dispatch/preview")
async def dispatch_preview_endpoint(request: Request) -> JSONResponse:
    """
    Pre-flight dispatch preview — runs ONLY the cheap upstream classifier(s)
    (triage + formation router for DEEP) and returns the decision + a
    deterministic cost estimate. Does NOT execute the engine.

    Use this BEFORE /api/v2/trace when you want to:
      - Show the user expected cost before running expensive analysis
      - Render the active-agent set in the UI without fanning out
      - Decide whether to confirm/cancel based on cost or risk flags

    Cost of this endpoint itself:
      - TRIVIAL / DIRECT / DIRECT_PLUS routes: 1 triage call (~$0.0002)
      - DEEP route: triage + router (~$0.0021 total)

    Body:
      {
        "question": "...",
        "effort":   "low" | "medium" | "high" | "auto"  (optional)
      }

    Response (200):
      {
        "route":               "trivial" | "direct" | "direct_plus" | "deep",
        "recommended_effort":  "low" | "medium" | "high" | "auto",
        "triage_why":          "one-sentence rationale",
        "risk_flags":          ["irreversible_action", ...],
        "interrupt":           bool,
        "mcps_needed":         [{"name", "why", "required"}, ...],
        "formation_plan":      {...} | null,    # null when route != "deep"
        "cost_breakdown": {
          "total_usd":           float,
          "estimated_llm_calls": int,
          "estimated_seconds":   float,
          "phases": {
            "triage":       float,
            "router":       float,
            "direct":       float,
            "domain_calls": float,
            "ke_calls":     float,
            "speech":       float
          }
        },
        "classifier_mode":     "mock" | "live" | ...
      }
    """
    request_id = f"prv-{int(time.time() * 1000)}"
    request_start = time.time()

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "Field 'question' is required"}, status_code=400)
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse(
            {"error": f"Question too long. Max {MAX_QUESTION_CHARS} characters."},
            status_code=400,
        )

    user_effort = normalize_effort(body.get("effort", DEFAULT_EFFORT))

    log.info(
        "[%s] /api/v2/dispatch/preview start | qlen=%d | effort=%s",
        request_id, len(question), user_effort.value,
    )

    try:
        client = LLMClient(mode=(ClientMode.LIVE if _has_live_key() else ClientMode.MOCK))
        preview = await preview_dispatch(
            text=question,
            client=client,
            user_effort=user_effort,
        )
        total_ms = (time.time() - request_start) * 1000

        log.info(
            "[%s] /api/v2/dispatch/preview done | route=%s | est_cost=$%.4f | calls=%d | total_ms=%d",
            request_id, preview.route,
            preview.cost_breakdown.total_usd,
            preview.cost_breakdown.estimated_llm_calls,
            int(total_ms),
        )

        payload = preview_to_dict(preview)
        payload["debug"] = {
            "request_id": request_id,
            "total_ms": round(total_ms),
        }
        return JSONResponse(payload)

    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] /api/v2/dispatch/preview failed", request_id)
        return JSONResponse(
            {
                "error": "Internal server error",
                "message": str(exc),
                "request_id": request_id,
            },
            status_code=500,
        )


@app.post("/api/v2/trace/resume")
async def trace_v2_resume(request: Request) -> JSONResponse:
    """
    Resume a request after the user has answered an escalation_offer.

    The frontend flow:
      1. POST /api/v2/trace with policy="ask"  → may return escalation_offer
      2. Show popup to user: "Extend to high effort?"
      3. User picks accept or decline
      4. POST /api/v2/trace/resume with the chosen effort and ORIGINAL question

    Request body:
      {
        "question":        "...",                  # required (the original)
        "accepted_effort": "low|medium|high",      # required (what user chose)
        "caps": { ... }                            # optional, same shape as /trace
      }

    The endpoint forces policy="strict" — no second offer can be generated.
    Response shape matches /api/v2/trace.
    """
    request_id = f"req-{int(time.time() * 1000)}"
    request_start = time.time()

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    question = str(body.get("question", "")).strip()
    if not question:
        return JSONResponse({"error": "Field 'question' is required"}, status_code=400)
    if len(question) > MAX_QUESTION_CHARS:
        return JSONResponse(
            {"error": f"Question too long. Max {MAX_QUESTION_CHARS} characters."},
            status_code=400,
        )

    accepted_raw = body.get("accepted_effort")
    if accepted_raw is None:
        return JSONResponse(
            {"error": "Field 'accepted_effort' is required (low|medium|high)"},
            status_code=400,
        )
    accepted_str = str(accepted_raw).strip().lower()
    if accepted_str not in ("low", "medium", "high", "auto"):
        return JSONResponse(
            {"error": "accepted_effort must be one of: low | medium | high | auto"},
            status_code=400,
        )
    accepted_effort = normalize_effort(accepted_str)

    caps_in = body.get("caps") or {}
    try:
        caps = BudgetCaps(
            max_iterations=int(caps_in.get("max_iterations", 12)),
            max_wall_time_sec=float(caps_in.get("max_wall_time_sec", 720.0)),
            max_cost_usd=float(caps_in.get("max_cost_usd", 1.00)),
            max_mcp_calls=int(caps_in.get("max_mcp_calls", 20)),
        )
    except (TypeError, ValueError):
        return JSONResponse({"error": "caps fields must be numeric"}, status_code=400)

    session_id = body.get("session_id")
    project_id = body.get("project_id")
    if session_id is not None:
        session_id = str(session_id).strip() or None
    if project_id is not None:
        project_id = str(project_id).strip() or None

    log.info(
        "[%s] /api/v2/trace/resume start | qlen=%d | accepted_effort=%s | session=%s",
        request_id, len(question), accepted_effort.value, session_id or "(none)",
    )

    try:
        client = LLMClient(mode=(ClientMode.LIVE if _has_live_key() else ClientMode.MOCK))
        conv_store = _make_conv_store(project_id) if session_id else None
        result: DispatchResult = await resume_with_choice(
            text=question,
            client=client,
            accepted_effort=accepted_effort,
            caps=caps,
            conversation_store=conv_store,
            session_id=session_id,
        )
        total_ms = (time.time() - request_start) * 1000

        log.info(
            "[%s] /api/v2/trace/resume done | route=%s | total_ms=%d | cost=$%.4f",
            request_id, result.route.value, int(total_ms),
            result.budget_summary.get("cost_usd", 0.0),
        )

        # Per-request LLM-call breakdown (same observability hook as /trace).
        try:
            client.summarize_calls(request_id=request_id)
        except Exception:
            log.exception("[%s] summarize_calls failed (non-fatal)", request_id)

        # Mirror /api/v2/trace's response shape so the UI's runResume()
        # validator never falls over on a field that exists on /trace
        # but was forgotten here. (Skipping perspectives previously
        # caused the frontend to throw SHAPE after a 10-minute run.)
        formation_plan_dict = None
        actual_iterations = None
        actual_active_domains: list[str] = []
        if result.engine_result is not None:
            formation_plan_dict = serialize_formation_plan(
                result.engine_result.formation_plan
            )
            ch = result.engine_result.convergence_history
            actual_iterations = ch.total_iterations or len(ch.snapshots)
            actual_active_domains = [
                d.value for d in result.engine_result.formation_plan.active_domains
            ]
        perspectives = serialize_perspectives(result.engine_result)

        payload = {
            "route": result.route.value,
            "response_text": result.response_text,
            "triage": {
                "route": result.triage_result.route.value,
                "recommended_effort": result.triage_result.recommended_effort.value,
                "interrupt": result.triage_result.interrupt,
                "risk_flags": result.triage_result.risk_flags,
                "mcps_needed": [
                    {"name": m.name, "why": m.why, "required": m.required}
                    for m in result.triage_result.mcps_needed
                ],
                "why": result.triage_result.why,
                "classifier_mode": result.triage_result.classifier_mode,
            },
            "formation_plan": formation_plan_dict,
            "actual_iterations": actual_iterations,
            "actual_active_domains": actual_active_domains,
            "perspectives": perspectives,
            "memo": result.memo,
            "thread_id": request_id,
            "budget": result.budget_summary,
            "capabilities": result.capability_state,
            "escalation_offer": result.escalation_offer,  # always None — policy=strict
            "missing_capability_offers": result.missing_capability_offers,
            "debug": {
                **result.debug,
                "request_id": request_id,
                "total_ms": round(total_ms),
                "resumed_at_effort": accepted_effort.value,
            },
        }
        if result.memo:
            _cache_memo(request_id, question=question, payload=payload)
        return JSONResponse(payload)

    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] /api/v2/trace/resume failed", request_id)
        return JSONResponse(
            {
                "error": "Internal server error",
                "message": str(exc),
                "request_id": request_id,
            },
            status_code=500,
        )


# ── Map Room — browser-tab fetch endpoint ─────────────────────────────────

@app.get("/api/v2/thread/{thread_id}")
async def get_thread_memo(thread_id: str) -> JSONResponse:
    """
    Fetch a cached memo by thread_id for the browser-tab Map Room.

    The extension UI hands this URL to the user as "Open Map Room ↗"; the
    browser tab loads /map/<thread_id> in the web app, which calls this
    endpoint to hydrate the visual study view.

    Returns the same payload shape /api/v2/trace returned for that thread,
    minus internal cache bookkeeping. 404 when the thread has expired or
    was never DEEP (trivial/direct routes don't produce a Map Room).
    """
    entry = _get_cached_memo(thread_id)
    if entry is None:
        return JSONResponse(
            {"error": "thread_not_found", "thread_id": thread_id},
            status_code=404,
        )
    return JSONResponse({
        "thread_id": thread_id,
        "question":  entry["question"],
        "payload":   entry["payload"],
    })


# ── Conversation endpoints (sessions, alerts, pinning, tree views) ────────

def _fp_to_dict(fp) -> dict | None:
    """dataclass → plain dict for JSON serialization."""
    if fp is None:
        return None
    from dataclasses import asdict
    return asdict(fp)


@app.post("/api/v2/conversation/session/start")
async def conv_session_start(request: Request) -> JSONResponse:
    """
    Create a new session. The frontend calls this once per conversation
    thread and then passes the returned session_id in subsequent
    /api/v2/trace calls so iterations get recorded.

    Body: { "project_id": "...", "title": "optional" }
    Response: full Session dict (id, started_at, expires_at, ...).
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    project_id = body.get("project_id")
    if project_id is not None:
        project_id = str(project_id).strip() or None
    title = str(body.get("title", "")).strip()

    cs = _make_conv_store(project_id)
    sess = await cs.start_session(title=title)
    return JSONResponse(_fp_to_dict(sess) or {})


@app.post("/api/v2/conversation/session/end")
async def conv_session_end(request: Request) -> JSONResponse:
    """
    Mark a session as ended. Optional — sessions auto-expire via TTL even
    without explicit end. Use this when the frontend knows the user is
    done (closed the chat, switched repos, etc.).

    Body: { "project_id": "...", "session_id": "..." }
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    project_id = body.get("project_id")
    if project_id is not None:
        project_id = str(project_id).strip() or None
    session_id = str(body.get("session_id", "")).strip()
    if not session_id:
        return JSONResponse({"error": "Field 'session_id' is required"}, status_code=400)

    cs = _make_conv_store(project_id)
    sess = await cs.end_session(session_id)
    if sess is None:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return JSONResponse(_fp_to_dict(sess) or {})


@app.get("/api/v2/conversation/session/{session_id}/tree")
async def conv_session_tree(session_id: str, request: Request) -> JSONResponse:
    """
    Render-ready tree view of a session — iterations + turning points
    nested under it, plus all decision links involving the session's
    decisions. The frontend uses this to draw the conversation timeline.
    """
    project_id = request.query_params.get("project_id")
    cs = _make_conv_store(project_id)
    tree = await cs.get_session_tree(session_id)
    return JSONResponse({
        "session": _fp_to_dict(tree["session"]),
        "iterations": [
            {
                "iteration": _fp_to_dict(block["iteration"]),
                "turning_points": [_fp_to_dict(tp) for tp in block["turning_points"]],
            }
            for block in tree["iterations"]
        ],
        "decision_links": [_fp_to_dict(link) for link in tree["decision_links"]],
    })


@app.get("/api/v2/conversation/alerts")
async def conv_alerts(request: Request) -> JSONResponse:
    """
    List entities approaching auto-deletion. Frontend polls this (or
    subscribes to a future SSE/WebSocket variant) to show notifications
    at 15 / 7 / 3-day thresholds + the expired tier.

    Query params:
        project_id: scope to one project (optional; omit for admin view)
        global=true: walk every project (admin/dashboard)
    """
    project_id = request.query_params.get("project_id")
    global_flag = request.query_params.get("global", "").lower() in ("true", "1", "yes")
    cs = _make_conv_store(project_id)
    alerts = await cs.get_expiry_alerts(project_only=not global_flag)

    from dataclasses import asdict
    return JSONResponse({
        "alerts": [asdict(a) for a in alerts],
        "count": len(alerts),
        "tiers_summary": {
            "expired":  sum(1 for a in alerts if a.tier == "expired"),
            "3_days":   sum(1 for a in alerts if a.tier == "3_days"),
            "7_days":   sum(1 for a in alerts if a.tier == "7_days"),
            "15_days":  sum(1 for a in alerts if a.tier == "15_days"),
        },
    })


@app.post("/api/v2/conversation/pin")
async def conv_pin(request: Request) -> JSONResponse:
    """
    Pin an entity to forever (expires_at = None) — opts out of TTL.
    Body: { "project_id": "...", "entity_type": "...", "entity_id": "...", "pin": true|false }
    pin=true   → call pin()
    pin=false  → call unpin() (re-applies the default TTL from now)
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    project_id = body.get("project_id")
    if project_id is not None:
        project_id = str(project_id).strip() or None
    entity_type = str(body.get("entity_type", "")).strip()
    entity_id = str(body.get("entity_id", "")).strip()
    pin = bool(body.get("pin", True))

    if not entity_type or not entity_id:
        return JSONResponse(
            {"error": "Fields 'entity_type' and 'entity_id' are required"},
            status_code=400,
        )

    cs = _make_conv_store(project_id)
    try:
        if pin:
            ok = await cs.pin(entity_type, entity_id)
        else:
            ok = await cs.unpin(entity_type, entity_id)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    if not ok:
        return JSONResponse(
            {"error": f"{entity_type} {entity_id!r} not found"},
            status_code=404,
        )
    return JSONResponse({"ok": True, "pinned": pin})


@app.post("/api/v2/conversation/sweep")
async def conv_sweep(request: Request) -> JSONResponse:
    """
    Manually trigger the conversation-store sweep.

    Useful for admin/debug, and for the background scheduler to delegate to
    a single code path. Body is optional:
        { "project_id": "...", "now": <unix-timestamp> }

    `now` defaults to current time; pass a custom one to simulate "as of
    timestamp X" for testing. project_id is ignored (sweep walks every
    project — TTL is global) but accepted for shape symmetry with the
    other endpoints.

    Returns per-bucket deletion counts.
    """
    try:
        body = await request.json() if (await request.body()) else {}
    except (json.JSONDecodeError, ValueError):
        body = {}

    now_override = body.get("now")
    if now_override is not None:
        try:
            now_override = float(now_override)
        except (TypeError, ValueError):
            return JSONResponse({"error": "'now' must be a number"}, status_code=400)

    cs = _make_conv_store(None)
    counts = await cs.sweep_expired(now=now_override)
    total = sum(counts.values())
    log.info("[manual-sweep] removed %d entries: %s", total, counts)
    return JSONResponse({
        "ok": True,
        "swept": counts,
        "total": total,
    })


async def _run_sweep_loop() -> None:
    """
    Background loop. Sleeps for SWEEP_INTERVAL_SECONDS, then calls
    sweep_expired on the shared store. Logs results. Never propagates
    exceptions — a crash here must NOT take down the server. Continues
    after any individual sweep failure.

    Cancellation: asyncio.CancelledError is re-raised so the task tree
    unwinds cleanly on shutdown.
    """
    log.info(
        "background sweep enabled (interval=%ds, store_path=%s)",
        SWEEP_INTERVAL_SECONDS,
        CONVERSATION_STORE_PATH or "(in-memory)",
    )
    while True:
        try:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            cs = _make_conv_store(None)
            counts = await cs.sweep_expired()
            total = sum(counts.values())
            if total > 0:
                log.info("[bg-sweep] removed %d expired entries: %s", total, counts)
            else:
                log.debug("[bg-sweep] no entries to remove")
        except asyncio.CancelledError:
            log.info("background sweep cancelled")
            raise
        except Exception as e:  # noqa: BLE001
            # Never crash the loop on a single failure. Log and continue.
            log.error("[bg-sweep] error (continuing): %s", e)


@app.on_event("startup")
async def _start_sweep_task() -> None:
    """Spawn the background sweep loop when the server starts."""
    global _sweep_task
    if not SWEEP_ENABLED:
        log.info("background sweep DISABLED (LORA_SWEEP_ENABLED=0)")
        return
    _sweep_task = asyncio.create_task(_run_sweep_loop())


@app.on_event("shutdown")
async def _stop_sweep_task() -> None:
    """Cancel the sweep task cleanly on shutdown."""
    global _sweep_task
    if _sweep_task is None:
        return
    _sweep_task.cancel()
    try:
        await _sweep_task
    except asyncio.CancelledError:
        pass
    _sweep_task = None


@app.on_event("shutdown")
async def _close_redis_client() -> None:
    """Close the Redis client cleanly so connections drain on shutdown."""
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        return
    try:
        # redis>=5 exposes aclose() (Redis 6 will drop the older close())
        if hasattr(_REDIS_CLIENT, "aclose"):
            await _REDIS_CLIENT.aclose()
        elif hasattr(_REDIS_CLIENT, "close"):
            await _REDIS_CLIENT.close()
    except Exception as e:
        log.warning("redis close error (ignored): %s", e)
    _REDIS_CLIENT = None


# ── Decision Trace memory recall (Phase 4) ─────────────────────────────────
# Builds a "PRIOR MEMORY" directive for each substantive trace request,
# injecting cross-thread / cross-platform Decision Trace context into the
# LLM prompt. Lazy-initialized on first request — no startup cost when
# the feature is disabled.
#
# Disable via LORA_MEMORY_RECALL_ENABLED=0 (e.g., dev without embedder).
# When disabled, build_memory_directive returns "" and dispatch is
# unchanged.

MEMORY_RECALL_ENABLED = os.environ.get("LORA_MEMORY_RECALL_ENABLED", "1") not in ("0", "false", "no")
_MEMORY_RETRIEVER: "MemoryRetriever | None" = None


async def _get_memory_retriever():
    """Lazy-build the retriever sharing the persistence layer's driver.

    Returns None if Neo4j isn't the active backend, the persistence
    layer hasn't initialized, or any error occurs during construction.
    Cached on first successful build."""
    global _MEMORY_RETRIEVER
    if _MEMORY_RETRIEVER is not None:
        return _MEMORY_RETRIEVER
    if not MEMORY_RECALL_ENABLED:
        return None
    try:
        from src.bridge.neo4j_backend import Neo4jThreadStore
        from src.bridge.thread_persistence import _ensure_initialized
        store, _, _ = await _ensure_initialized()
    except Exception as e:
        log.debug("memory retriever: persistence init failed (%s)", e)
        return None
    if not isinstance(store, Neo4jThreadStore):
        log.debug("memory retriever: store is %s, not Neo4j — disabled", type(store).__name__)
        return None
    _MEMORY_RETRIEVER = MemoryRetriever(store, GeminiEmbeddingService())
    log.info("memory retriever: ready")
    return _MEMORY_RETRIEVER


async def _build_memory_for_request(*, question: str, body: dict, parent_thread_id: str | None) -> str:
    """Build the PRIOR MEMORY directive for one trace request. Returns
    empty string on any failure — never blocks the dispatch path."""
    if not MEMORY_RECALL_ENABLED:
        return ""
    user_id = body.get("user_id") if isinstance(body, dict) else None
    if not user_id:
        return ""
    retriever = await _get_memory_retriever()
    if retriever is None:
        return ""
    try:
        return await build_memory_directive(
            question=question,
            retriever=retriever,
            user_id=user_id,
            thread_id=parent_thread_id,
            project_id=body.get("project_id"),
            workspace_id=body.get("workspace_id"),
            surface_id=body.get("surface_id"),
        )
    except Exception as e:
        log.warning("memory directive build failed (continuing): %s", e)
        return ""


# ── Decision Trace background sweeper (Phase 3) ────────────────────────────
# Auto-structures iterations 30 minutes after they go idle. Calls
# InlineClassifier per iteration, writes typed event nodes via
# Neo4jDecisionTraceWriter, stamps structured_at last. Restart-safe
# (Neo4j is the source of truth — no in-memory timers).
#
# Disable for dev / specific deploys via LORA_DT_SWEEP_ENABLED=0.

DT_SWEEP_ENABLED = os.environ.get("LORA_DT_SWEEP_ENABLED", "1") not in ("0", "false", "no")
DT_IDLE_SEC = int(os.environ.get("LORA_DT_IDLE_SEC", "1800"))            # 30 min default
DT_BATCH = int(os.environ.get("LORA_DT_BATCH", "50"))
DT_INTERVAL_SEC = int(os.environ.get("LORA_DT_INTERVAL_SEC", "300"))     # 5 min default

_dt_sweep_task: "asyncio.Task | None" = None
_dt_sweeper: "object | None" = None     # DecisionTraceSweeper, lazily imported
_dt_stop_event: "asyncio.Event | None" = None


@app.on_event("startup")
async def _start_dt_sweep_task() -> None:
    """Spawn the Decision Trace sweeper if Neo4j is the active backend.

    Strict guards:
      - LORA_DT_SWEEP_ENABLED must not be 0
      - ThreadStore must be a Neo4jThreadStore (the sweeper is Neo4j-bound)
      - persistence layer must initialize cleanly

    Any failure logs and disables the sweeper for this process lifetime —
    user-facing trace endpoints stay functional regardless."""
    global _dt_sweep_task, _dt_sweeper, _dt_stop_event
    if not DT_SWEEP_ENABLED:
        log.info("Decision Trace sweeper DISABLED (LORA_DT_SWEEP_ENABLED=0)")
        return

    # Lazy imports — keep the sweeper module out of the cold-start path
    # when it's not enabled.
    try:
        from src.bridge.decision_trace_sweeper import build_sweeper
        from src.bridge.neo4j_backend import Neo4jThreadStore
        from src.bridge.thread_persistence import _ensure_initialized
    except Exception as e:
        log.warning("DT sweeper: import failed (%s) — disabled", e)
        return

    try:
        store, _, _ = await _ensure_initialized()
    except Exception as e:
        log.warning("DT sweeper: persistence init failed (%s) — disabled", e)
        return

    if not isinstance(store, Neo4jThreadStore):
        log.info(
            "DT sweeper: ThreadStore is %s (not Neo4j) — sweeper requires Neo4j, skipping",
            type(store).__name__,
        )
        return

    _dt_sweeper = build_sweeper(
        store,
        idle_threshold_sec=DT_IDLE_SEC,
        batch_size=DT_BATCH,
        interval_sec=DT_INTERVAL_SEC,
    )
    _dt_stop_event = asyncio.Event()
    _dt_sweep_task = asyncio.create_task(_dt_sweeper.run(_dt_stop_event))
    log.info(
        "DT sweeper: scheduled (idle=%ds batch=%d interval=%ds)",
        DT_IDLE_SEC, DT_BATCH, DT_INTERVAL_SEC,
    )


@app.on_event("shutdown")
async def _stop_dt_sweep_task() -> None:
    """Cancel the Decision Trace sweeper cleanly on shutdown.

    Sequence:
      1. Signal stop via the sweeper's event (lets the current sleep wake)
      2. Wait up to 10s for the task to exit cleanly
      3. Hard-cancel if it doesn't
    """
    global _dt_sweep_task, _dt_sweeper, _dt_stop_event
    if _dt_sweeper is not None:
        try:
            _dt_sweeper.stop()
        except Exception:
            pass
    if _dt_sweep_task is None:
        return
    try:
        await asyncio.wait_for(_dt_sweep_task, timeout=10.0)
    except asyncio.TimeoutError:
        _dt_sweep_task.cancel()
        try:
            await _dt_sweep_task
        except (asyncio.CancelledError, Exception):
            pass
    except asyncio.CancelledError:
        pass
    _dt_sweep_task = None
    _dt_sweeper = None
    _dt_stop_event = None


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Constellax Reasoning Engine starting on http://%s:%d", HOST, PORT)
    log.info("CORS origins: %s", CORS_ORIGINS)
    log.info("Mode: %s", "LIVE (OpenRouter)" if _has_live_key() else "MOCK")
    log.info("Default effort: %s (%d iterations)", DEFAULT_EFFORT.value, iterations_for(DEFAULT_EFFORT))
    log.info(
        "ConversationStore: %s",
        type(_CONV_BACKEND_ACTIVE).__name__ if _CONV_BACKEND_ACTIVE is not None
        else ("JSON file" if CONVERSATION_STORE_PATH else "in-memory only"),
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
