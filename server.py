"""
LoRa Deep Reasoning Engine — FastAPI server.

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

from src.core.types import Direction, FrameworkID, Problem, Variable
from src.llm.client import ClientMode, LLMClient
from src.llm.engine import run_async_formation
from src.llm.speech import extract_speech_input, generate_speech


# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("lora.server")


# ── Config from environment ────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", "8100"))
HOST = os.environ.get("HOST", "0.0.0.0")
CORS_ORIGINS_ENV = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS_ENV.split(",") if o.strip()]

DEFAULT_MAX_ITERATIONS = int(os.environ.get("DEFAULT_MAX_ITERATIONS", "2"))
MAX_PHASE2_ITERATIONS = int(os.environ.get("MAX_PHASE2_ITERATIONS", "6"))
MAX_QUESTION_CHARS = int(os.environ.get("MAX_QUESTION_CHARS", "8000"))
MAX_PHASE1_SUMMARY_CHARS = int(os.environ.get("MAX_PHASE1_SUMMARY_CHARS", "20000"))


# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="LoRa Deep Reasoning Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

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
        "service": "LoRa Deep Reasoning Engine",
        "version": "2.0.0",
        "domains": 5,
        "concepts": 63,
    })


@app.get("/health")
async def health() -> dict:
    """Liveness probe for load balancers and orchestrators."""
    return {
        "status": "ok",
        "mode": "live" if os.environ.get("ANTHROPIC_API_KEY") else "mock",
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

    try:
        max_iterations = int(body.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    except (TypeError, ValueError):
        return JSONResponse({"error": "max_iterations must be an integer"}, status_code=400)
    max_iterations = max(1, min(max_iterations, MAX_PHASE2_ITERATIONS * 2))
    is_phase_one = max_iterations <= DEFAULT_MAX_ITERATIONS

    phase1_summary = str(body.get("phase1_summary", ""))
    if len(phase1_summary) > MAX_PHASE1_SUMMARY_CHARS:
        phase1_summary = phase1_summary[:MAX_PHASE1_SUMMARY_CHARS]

    log.info(
        "[%s] /api/trace start | qlen=%d | iters=%d | phase1=%s",
        request_id, len(question), max_iterations, bool(phase1_summary),
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
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        mode = ClientMode.LIVE if has_key else ClientMode.MOCK
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


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("LoRa Deep Reasoning Engine starting on http://%s:%d", HOST, PORT)
    log.info("CORS origins: %s", CORS_ORIGINS)
    log.info("Mode: %s", "LIVE (Sonnet)" if os.environ.get("ANTHROPIC_API_KEY") else "MOCK")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
