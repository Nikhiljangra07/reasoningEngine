"""
LoRa Deep Reasoning Engine — API Server for Live Visualization.

Produces a timeline-based trace object that the taoist.html visualizer
animates step-by-step, showing every concept firing, every bridge active,
every Ke challenge, every perspective born, every funnel pass.

Run: python server.py
Then open: ~/Desktop/lora-brain-viz/taoist.html
"""

import asyncio
import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Load .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key] = val

from src.core.types import Direction, Domain, FrameworkID, Problem, Variable
from src.llm.client import LLMClient, ClientMode
from src.llm.engine import run_async_formation
from src.llm.speech import generate_speech, extract_speech_input

app = FastAPI(title="LoRa Deep Reasoning Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve the web UI
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

# Trace collector — captures every event with timestamps
trace_events = []
trace_start = 0


def emit(event_type, data):
    trace_events.append({
        "type": event_type,
        "data": data,
        "t": (time.time() - trace_start) * 1000,  # ms since start
    })


def parse_problem(text):
    variables = []
    sentences = text.replace(".", ". ").split(". ")
    neg = ["but","struggle","fight","doubt","fear","worried","stuck","hate","dread","can't","don't","terrified","frustrated","unfulfilled","overwhelm"]
    pos = ["love","passionate","dream","want","excited","enjoy","growing","opportunity","happy"]
    for i, s in enumerate(sentences):
        s = s.strip()
        if not s or len(s) < 10: continue
        lo = s.lower()
        nc = sum(1 for w in neg if w in lo)
        pc = sum(1 for w in pos if w in lo)
        d = Direction.NEGATIVE if nc > pc else Direction.POSITIVE if pc > nc else Direction.NEUTRAL
        m = 0.85 if any(w in lo for w in ["every","always","never","completely"]) else 0.6
        variables.append(Variable(name=f"v{i}", description=s[:200], magnitude=m, direction=d, confidence=0.8, source_framework=FrameworkID.FIRST_PRINCIPLES, is_user_stated=True))
    return Problem(statement=text, variables=variables[:8])


@app.get("/")
async def root():
    """Serve the web UI."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "LoRa v2", "domains": 5, "concepts": 63}


@app.post("/api/trace")
async def trace(request: Request):
    """Run engine and return a full timeline trace for the visualizer."""
    global trace_events, trace_start
    body = await request.json()
    question = body.get("question", "")
    max_iterations = min(int(body.get("max_iterations", 2)), 12)
    is_phase_one = max_iterations <= 2
    phase1_summary = body.get("phase1_summary", "")
    if not question:
        return JSONResponse({"error": "No question"}, 400)

    trace_events = []
    trace_start = time.time()

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    mode = ClientMode.LIVE if has_key else ClientMode.MOCK
    client = LLMClient(mode=mode)

    # Instrument the client to capture every call
    original_call = client.call

    async def instrumented_call(system_prompt, user_message, domain, concept, **kwargs):
        t0 = time.time()
        emit("call_start", {"domain": domain, "concept": concept})
        result = await original_call(system_prompt, user_message, domain, concept, **kwargs)
        elapsed = (time.time() - t0) * 1000
        # Parse findings from response
        findings = []
        try:
            import json as j
            content = result.content.strip()
            if content.startswith("```"): content = "\n".join(l for l in content.split("\n") if not l.strip().startswith("```"))
            data = j.loads(content)
            for f in data.get("findings", []):
                findings.append({
                    "name": f.get("name", ""),
                    "type": f.get("type", ""),
                    "description": f.get("description", "")[:120],
                    "magnitude": f.get("magnitude", 0.5),
                    "direction": f.get("direction", "neutral"),
                    "confidence": f.get("confidence", 0.5),
                })
        except: pass

        emit("call_done", {
            "domain": domain, "concept": concept,
            "success": result.success, "elapsed_ms": round(elapsed),
            "tokens": result.input_tokens + result.output_tokens,
            "findings": findings,
            "preview": result.content[:300] if result.success else (result.error or ""),
        })
        return result

    client.call = instrumented_call

    problem = parse_problem(question)

    # Phase 2: inject Phase 1 findings as context so the engine goes DEEPER not sideways
    if phase1_summary:
        problem.context = (
            "PHASE 2 — DEEPER ANALYSIS. The user has already seen Phase 1 findings below. "
            "Do NOT repeat the same analysis. Go deeper. Challenge Phase 1's conclusions. "
            "Find what Phase 1 missed. Surface second-order effects and hidden variables "
            "that only emerge with more iterations.\n\n"
            f"PHASE 1 FINDINGS (already delivered to user):\n{phase1_summary}"
        )

    emit("stage", {"stage": 1, "name": "Chemistry Reads"})

    engine_result = await run_async_formation(problem, client, max_iterations=max_iterations)

    # Speech — Phase 2 gets full analysis mode (500 words, no dig deeper)
    speech_input = extract_speech_input(
        engine_result, question,
        is_phase_one=is_phase_one,
        estimated_additional_credits=15.0 if is_phase_one else 0,
    )
    emit("stage", {"stage": 7, "name": "Speech Module"})
    speech_result = await generate_speech(client, speech_input)

    total_ms = (time.time() - trace_start) * 1000

    # Build trace object
    # Domains
    domain_data = {}
    for dom, out in engine_result.domain_outputs.items():
        perspectives = []
        for p in out.perspectives:
            pvars = []
            for v in p.variables_found:
                pvars.append({"name": v.name, "desc": v.description[:80], "mag": v.magnitude, "dir": v.direction.value, "conf": v.confidence, "hidden": v.is_hidden})
            perspectives.append({"framework": p.framework.value, "weight": p.weight, "variables": pvars})
        roots = []
        for r in out.root_causes:
            roots.append({"name": r.variable.name, "desc": r.variable.description[:120], "confidence": r.confidence, "bias": r.bias_that_hid_it or "", "hidden": r.variable.is_hidden})
        domain_data[dom.value] = {"perspectives": perspectives, "root_causes": roots, "raw_preview": out.raw_analysis[:400]}

    # Ke
    ke_data = []
    for ke in engine_result.ke_results:
        ke_data.append({"challenger": ke.challenger_domain.value, "target": ke.target_domain.value, "scrutiny": ke.scrutiny_score, "contradictions": ke.contradictions[:3], "flags": ke.flags[:3]})

    # Funnel
    funnel_data = []
    for f in engine_result.funnel_history:
        funnel_data.append({"kept": f.variables_kept, "cached": f.variables_cached, "needs_work": f.variables_needing_work, "stable": f.variables_stable})

    # Convergence
    conv_data = []
    for s in engine_result.convergence_history.snapshots:
        conv_data.append({"iter": s.iteration, "gibbs": s.gibbs_energy, "converged": s.is_converged, "posterior_delta": s.posterior_delta, "new_vars": s.new_variables_count, "ke_scrutiny": s.avg_ke_scrutiny})

    # Trajectories
    traj_data = []
    for t in engine_result.trajectories[:4]:
        traj_data.append({"name": t.root_cause.variable.name, "desc": t.root_cause.variable.description[:200], "confidence": t.confidence, "hidden": t.root_cause.variable.is_hidden, "bias": t.root_cause.bias_that_hid_it or ""})

    summary = client.get_call_summary()

    return JSONResponse({
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


if __name__ == "__main__":
    print("\n  LoRa Deep Reasoning Engine")
    print("  http://localhost:8100")
    print("  Open in your browser to use the UI\n")
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="warning")
