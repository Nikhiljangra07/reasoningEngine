"""
run_autonomous_pipeline.py — THE AUTONOMOUS LEVEL-1 PIPELINE (live loop controller).

Self-driving multi-cycle loop. It wanders, audits, checks the checkpoint, and
re-dispatches goal-free leads until the cushion's question is covered or the
4-cycle cap is hit. Composes the validated organs on top of the EXISTING wander
machinery — it does NOT modify the single-pass collision pipeline.

PER CYCLE:
  1. wander            cycle 1 → main cushion (anchor only, no question);
                       cycle N>1 → goal-free lead(s) from the prior cycle
  2. build_dossier     → cards
  3. halo audit (GA)   → blind spots (the auditor lens — contributes to direction)
  4. coverage (CHECK)  → D_t + open angles on the ACCUMULATED cards (the checkpoint)
  5. signals           novelty (new angles), change (ΔD_t)
  6. HALT?             (coverage-complete ∧ saturated ∧ settled) OR cycle == CAP
  7. else dispatch     gap pool = open angles + GA blind spots → pick top (halo
                       severity) → translate to goal-free lead → next wave

SAFETY (be safe + meticulous):
  - CONSTELLAX_GOVERNOR=1 required (fail-closed; no accidental spend).
  - Waves bounded: soft budget + hard asyncio ceiling (the reliability fix).
  - 4-cycle hard cap; AUTON_MAX_CYCLES can LOWER it for cheap testing.
  - AUTON_PLAN_ONLY=1 validates wiring/config and exits before any wave (~$0).
  - DeepSeek+Haiku wander, NO Sonnet in digs (provider_map fix). Judges (halo GA
    + coverage checkpoint) run on DeepSeek too — deliberate, "cheap, validated"
    (coverage_scorer.py:37, halo_auditor.py:70). Sonnet touches ONLY the one-time
    cushion intake (compose_cushion → cushion_extraction), never wander or judge.
  - Every dispatched lead passes the chaos-law gate (translator); fail-closed.
  - Per-cycle cost logged.

Usage:
  AUTON_PLAN_ONLY=1 PYTHONPATH=.:scripts python3 scripts/run_autonomous_pipeline.py   # dry, ~$0
  CONSTELLAX_GOVERNOR=1 AUTON_MAX_CYCLES=2 PYTHONPATH=.:scripts python3 scripts/run_autonomous_pipeline.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")
for noisy in ("httpx", "google_genai", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

import run_fable_sorter_6agents as base
from src.llm.client import ClientMode, LLMClient
from src.wandering.cushion import CushionInput, CushionField, SkipReason
from src.wandering.composer import compose_cushion
from src.wandering.runtime import WanderingConfig, run_wandering_session, WanderingProgress, SessionResult
from src.wandering.fetcher import web_search_fetcher
from src.wandering.dossier import build_dossier
from src.wandering.halo_auditor import audit_cards
from src.wandering.coverage_scorer import parse_required_angles, score_coverage
from src.wandering.lead_translator import translate_gap
from src.wandering.master_synthesizer import master_synthesize
from src.wandering.formalizer import formalize_blends, render_markdown
from src.wandering.auton_memory import AutonMemory
from src.wandering.drift_checker import check_cycle_drift

# --- knobs ---
CYCLE_CAP = int(os.environ.get("AUTON_MAX_CYCLES", "4"))     # hard cap (≤4 by spec)
# Cycle-1 wave size. DEFAULT (env unset): one root agent per parsed sub-question —
# dynamic, no hard cap (3, 5, 10… whatever the cushion question yields). Grounded in
# Anthropic's multi-agent guidance: one agent per clearly-divided responsibility, and
# their documented failure mode was OVER-spawning — so we size to real work-items, not
# a fixed 7. Override with AUTON_MAIN_AGENTS=<N> to force a count.
_MAIN_AGENTS_OVERRIDE = os.environ.get("AUTON_MAIN_AGENTS")  # None → size to len(angles)
# Phase 6: dispatch wave size = clamp(distinct fresh gaps, 1, DISPATCH_CAP); 0 → halt.
# Grounded in Anthropic's multi-agent guidance (one agent per work-item; over-spawning
# is the failure mode) + the self-consistency plateau (~3-10, decline past 15).
DISPATCH_CAP = int(os.environ.get("AUTON_DISPATCH_CAP", "8"))
# Token BREATHING ROOM per agent. The smoke ran 15k (card-starved + DeepSeek JSON
# truncations); 40k (the absolute_chaos mode default) lets agents do more digs and
# emit complete JSON. Raise further for a richer run.
TOKENS_PER_AGENT = int(os.environ.get("AUTON_TOKENS_PER_AGENT", "40000"))
# HARD total-spend backstop. Checked against the LLM client's TRUE cumulative cost
# (not the wave-only undercount) at the top of every cycle and before the blender.
RUN_BUDGET_USD = float(os.environ.get("AUTON_RUN_BUDGET_USD", "50"))

# NO-TIME-LIMIT mode (Nikhil's directive 2026-06-17): let each wave run to
# NATURAL completion — agents self-terminate on their token/step budget, or the
# governor CLOSEs on convergence ("skeleton built"). No artificial soft-halt
# clock. A generous hard ceiling stays purely as an anti-hang backstop, and even
# it is LOSSLESS (cancel + harvest already-finalized reports). Set on by default
# here because run #1 spent money and got nothing — this time we let it finish.
NO_TIME_LIMIT = os.environ.get("AUTON_NO_TIME_LIMIT", "1") == "1"

# Bounded mode (NO_TIME_LIMIT=0): soft budget → graceful halt, grace, hard ceiling.
WAVE_SOFT_S = int(os.environ.get("AUTON_WAVE_SOFT_MIN", "6")) * 60
WAVE_GRACE_S = int(os.environ.get("AUTON_WAVE_GRACE_MIN", "3")) * 60
# Anti-hang backstop. Generous in no-limit mode (40m) so it ~never bites a
# legitimately-long wander; tight (9m) in bounded mode.
WAVE_CEILING_S = int(os.environ.get(
    "AUTON_WAVE_HARD_MIN", "40" if NO_TIME_LIMIT else "9")) * 60

TH_NOVELTY, TH_COVERAGE, TH_CHANGE = 1, 0.95, 0.05

# Phase 1 (cross-family coverage): the halt-gate judge runs on a DIFFERENT model
# family than the wander (DeepSeek/Haiku) and the halo (Sonnet). Intrinsic self-
# correction fails (Huang et al. ICLR 2024) — a model judging its own family's
# findings carries self-preference bias. Gemini keeps the coverage checkpoint
# lineage-independent from BOTH the producers and the halo. Routes via OpenRouter
# (coverage_scorer posts the slug directly). Override with AUTON_COVERAGE_MODEL.
COVERAGE_MODEL = os.environ.get("AUTON_COVERAGE_MODEL", "google/gemini-2.5-flash")

# Phase 4 (blender / synthesis). The missing stage that made run-#2 score D_t=0:
# raw goal-free cards DIAGNOSE but don't PROPOSE. The blender (Opus+GPT, 4-round
# master synthesis) fuses accumulated cards into proposals; coverage then scores
# the PROPOSALS (the true D_t), not raw cards. Runs ONCE after the loop (not per
# cycle — that'd be N× Opus) under a hard cost cap. Gate off with AUTON_BLEND=0.
AUTON_BLEND = os.environ.get("AUTON_BLEND", "1") == "1"
# Default raised $4 → $18 after the 2026-06-18 run: the blender is 4 rounds × 2
# seats, and with a full card set each Opus call is ~$3.5, so $4 tripped one call
# into R2 and produced ZERO fusions. $18 leaves room for all rounds; the deduped/
# capped card feed below keeps real spend ~$3-5. Still env-overridable.
BLENDER_COST_CAP = float(os.environ.get("AUTON_BLENDER_COST_CAP", "18.0"))
# Cap the cards handed to the blender. 218 FULL cards in one Opus prompt truncated
# its draft JSON (recovered only 4 of N) and made each call expensive. ~120 keeps
# the prompt clean and the per-call cost low. 0 = no cap (feed everything).
BLEND_CARD_CAP = int(os.environ.get("AUTON_BLEND_CARD_CAP", "120"))

# Phase 5 (R1 formalization). DeepSeek-R1 grounds each blend into testable math —
# now a BUILT-IN pipeline stage, not a separate script run. R1 is junior to the
# blender (reads blends, never rewrites them) and returns a separate report, so
# it's additive/non-invasive. Our blends are MasterFusionReports; the formalizer
# expects the collision Blend dict shape → adapter below. Gate off with AUTON_R1=0.
AUTON_R1 = os.environ.get("AUTON_R1", "1") == "1"
PLAN_ONLY = os.environ.get("AUTON_PLAN_ONLY", "0") == "1"
# Pre-flight: launder the sub-questions into goal-free leads and STOP (translator
# calls only, ~$0.02). Lets us see the clean-vs-fallback split before a full run.
TRANSLATE_ONLY = os.environ.get("AUTON_TRANSLATE_ONLY", "0") == "1"
# Phase 4: cross-cycle / cross-run memory (Neo4j, fail-open). ON by default; the
# pipeline runs identically if Neo4j is unreachable (JSON stays source of truth).
# AUTON_MEMORY=0 disables it; AUTON_MEMORY_FRESH=1 records but ignores PRIOR memory.
AUTON_MEMORY = os.environ.get("AUTON_MEMORY", "1") == "1"
AUTON_MEMORY_FRESH = os.environ.get("AUTON_MEMORY_FRESH", "0") == "1"
# Phase 5: cycle-trajectory shepherd (Sonnet sensor). ADVISORY only — surfaces
# on_track/circling/drifting + a refocus nudge each cycle; informs the dispatcher
# (next phase), never halts. One Sonnet call/cycle, fail-open. AUTON_DRIFT=0 off.
AUTON_DRIFT = os.environ.get("AUTON_DRIFT", "1") == "1"

# Observability root. Every wave streams a per-call jsonl (model/tokens/cost/ms)
# and a governance.json (edges/probes/rounds/decisions); every cycle writes a
# cycle_N.json; the run writes manifest.json. Set via AUTON_RUN_DIR or defaulted.
RUN_DIR = Path(os.environ.get(
    "AUTON_RUN_DIR",
    str(REPO_ROOT / "runs" / "auton-c4" / time.strftime("run-%Y%m%d-%H%M%S"))))

log = logging.getLogger("autonomous")

# Line-buffer stdout so prints FLUSH live even under nohup/redirect (run #1 was
# dark because banners buffered until exit). Belt-and-suspenders with `python -u`.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _p(msg: str = "") -> None:
    """Flushed print — never let a progress line sit in a buffer."""
    print(msg, flush=True)


def _read_governance(wave_dir: Path) -> dict:
    """The governor's own record beside the wave's call log: edges, probes,
    emergence-rate rounds, decisions, final action. This is the skeleton organ's
    black box — empty {} if the wave never wrote one (e.g. hard-cancel)."""
    f = wave_dir / "governance.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}
    return {}


# Price per 1M tokens (input, output), mirrored from provider_map.py. The
# CallTracker jsonl records tokens + model but NOT cost, so we compute it here.
_PRICE = {
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-haiku-4-5":  (1.00,  5.00),
    "deepseek/deepseek-v4-pro":    (1.74,  3.48),
    "google/gemini-2.5-flash":     (0.30,  2.50),
    "google/gemini-2.5-flash-lite":(0.10,  0.40),
}


def _wave_cost(wave_dir: Path) -> tuple[float, int, dict]:
    """Cost + tokens from the wave's per-call jsonl, tallied by model so the
    Sonnet/DeepSeek/Haiku split is visible at a glance. CallTracker jsonl has no
    cost field, so cost = tokens × price (provider_map mirror); falls back to a
    `cost_usd` field if a future schema adds one. Unknown models priced at 0 and
    surfaced by name so a missing price is obvious, not silently $0."""
    f = wave_dir / "calls.jsonl"
    total, toks, by_model = 0.0, 0, {}
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            m = d.get("model_actually_used") or d.get("model") or "?"
            itok = int(d.get("input_tokens") or d.get("in_tok") or 0)
            otok = int(d.get("output_tokens") or d.get("out_tok") or 0)
            if "cost_usd" in d:
                c = float(d.get("cost_usd") or 0.0)
            else:
                pin, pout = _PRICE.get(m, (0.0, 0.0))
                c = itok / 1e6 * pin + otok / 1e6 * pout
            total += c
            toks += itok + otok
            by_model[m] = round(by_model.get(m, 0.0) + c, 4)
    return round(total, 4), toks, by_model


def _seed_cushion_input(base_ci: CushionInput, lead: str) -> CushionInput:
    """A per-agent cushion INPUT: the FULL base cushion (problem/context/vision/
    hunches kept) with a goal-free laundered lead appended to the problem as a
    'specific territory' line — which steers that agent's search (query_seed =
    problem.content). Question excluded (chaos law). Used by BOTH cycle-1 sub-
    question seeding AND dispatch waves, so dispatch agents anchor on the cushion
    instead of a bare lead."""
    return CushionInput(
        problem=CushionField(
            name="problem",
            content=(base_ci.problem.content
                     + "\n\nSPECIFIC TERRITORY (one facet to explore): " + lead)),
        context=base_ci.context, vision=base_ci.vision, hunches=base_ci.hunches,
        question=CushionField(name="question", content=""),  # chaos law: never seed the question
    )


def _gap_key(text: str) -> str:
    """Normalized dedup key for a gap — lowercased alphanumerics, collapsed, first
    80 chars. Conservative (near-exact) so the dispatcher only drops a gap it has
    genuinely chased before, never a merely-similar live one."""
    import re
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()[:80]


async def _seed_inputs_from_angles(angles: list[str], base_ci: CushionInput) -> list:
    """Phase 3 — chaos-law-safe per-agent seeding for cycle 1. Launder each
    sub-question into a goal-free declarative lead (translate_gap → leak_check,
    fail-closed), then fold it into a per-agent cushion INPUT via _seed_cushion_input.
    The question is NEVER seeded (chaos law). An angle that won't launder clean →
    None → that agent wanders the plain base cushion. Returns a list 1:1 to `angles`."""
    seeds: list = []
    for i, ang in enumerate(angles, 1):
        res = await translate_gap(ang)
        if res.clean and res.lead:
            seeds.append(_seed_cushion_input(base_ci, res.lead))
            _p(f"    seed Q{i}: laundered OK → {res.lead[:66]}")
        else:
            seeds.append(None)
            why = "; ".join(res.reasons) or res.error or "no clean lead"
            _p(f"    seed Q{i}: NOT clean ({why}) → falls back to base cushion")
    return seeds


def _cards_text(cards) -> str:
    return "\n".join(f"[{getattr(c,'report_id','?')}] {getattr(c,'bridge','')[:240]}" for c in cards) or "(no cards)"


def _request_halt(prog) -> bool:
    """Ask the wander to stop GRACEFULLY via the existing governor_halt flag.
    Agents check it at the top of their loop (agent.py:1557) and exit after the
    in-flight dig, so run_wandering_session returns a COMPLETE SessionResult —
    no work discarded. Returns True if a shared session_state was reachable."""
    for a in getattr(prog, "agents", []):
        ss = getattr(a, "session_state", None)
        if ss is not None:
            try:
                ss.governor_halt = True
                ss.governor_halt_reason = "controller_budget"
                return True
            except Exception:
                return False
    return False


def _harvest_partial(prog, cushion, cfg, t0: float, label: str):
    """Last-resort harvest when even the graceful halt overran the hard ceiling.
    Pull the reports agents have ALREADY finalized off the live progress handle
    and wrap them in a SessionResult, so the cycle uses real findings instead of
    throwing away completed dig work (the bug that voided run #1). None if empty."""
    agents_live = list(getattr(prog, "agents", []))
    reports = [r for a in agents_live for r in getattr(a, "reports", [])]
    traces = [a.trace for a in agents_live if getattr(a, "trace", None) is not None]
    ss = next((a.session_state for a in agents_live
               if getattr(a, "session_state", None) is not None), None)
    log.warning("[wander] %s hard ceiling — harvested %d finalized report(s) from %d agent(s) "
                "(partial, NOT discarded)", label, len(reports), len(agents_live))
    if not reports:
        return None
    return SessionResult(
        session_id=cfg.session_id or "auton", mode=cfg.mode, cushion=cushion, config=cfg,
        reports=reports, traces=traces,
        total_tokens_spent=int(getattr(prog, "tokens_used", 0)),
        elapsed_seconds=time.time() - t0, ended_at=time.time(), session_state=ss,
        expected_agent_count=cfg.agents or len(agents_live),
    )


async def _wander(client, cushion_input, agents: int, label: str, wave_dir: Path,
                  per_agent_inputs=None):
    """One wave. In NO_TIME_LIMIT mode it runs to NATURAL completion (agents
    exhaust their token/step budget or the governor CLOSEs on convergence) with
    only a generous anti-hang ceiling. In bounded mode: soft graceful-halt →
    grace → hard ceiling. BOTH modes are LOSSLESS — a forced stop cancels but
    harvests already-finalized reports (never voids the wave, the run-#1 bug).
    Streams a per-call jsonl + governance.json into wave_dir. Returns a
    SessionResult or None."""
    wave_dir.mkdir(parents=True, exist_ok=True)
    cushion = await compose_cushion(cushion_input, client, session_id="auton", auto_enrich=False)
    # Phase 3: per-agent seeding. Compose the base cushion ONCE, then give each
    # agent a shallow copy whose SEARCH anchor (raw_input.problem → query_seed in
    # agent.py) is its laundered sub-question territory. The graph (Essence/Mechanism)
    # is shared → coherent matching + intact governor. None entry → shared base lens.
    per_agent_cushions = None
    if per_agent_inputs:
        import dataclasses
        per_agent_cushions = [
            cushion if inp is None
            else dataclasses.replace(
                cushion, raw_input=dataclasses.replace(cushion.raw_input, problem=inp.problem))
            for inp in per_agent_inputs
        ]
        seeded_n = sum(1 for inp in per_agent_inputs if inp is not None)
        log.info("[wander] %s — per-agent seeding: %d/%d agents on a laundered sub-question lens",
                 label, seeded_n, len(per_agent_cushions))
    # In no-limit mode the per-agent TIME budget is set to the anti-hang ceiling
    # so TOKENS/steps (not a clock) bound each agent — letting the skeleton form.
    agent_time = WAVE_CEILING_S if NO_TIME_LIMIT else WAVE_SOFT_S
    cfg = WanderingConfig(mode=base._MODE_MAP["absolute_chaos"], agents=agents,
                          time_budget_seconds=agent_time, tokens_per_agent=TOKENS_PER_AGENT,
                          # All ROOT agents = DeepSeek-v4-pro (1-tuple → round-robins
                          # uniform regardless of wave size). Sub-agents → Haiku (phase 2).
                          # Was None → the 5×DeepSeek+5×Haiku mode default (mixed roots).
                          model_mix=("deepseek/deepseek-v4-pro",), session_id="auton",
                          per_agent_cushions=per_agent_cushions,
                          call_log_path=str(wave_dir / "calls.jsonl"))  # per-call jsonl + governance.json
    prog = WanderingProgress()
    t0 = time.time()
    task = asyncio.create_task(
        run_wandering_session(cushion, cfg, client, fetcher=web_search_fetcher, progress=prog))

    if NO_TIME_LIMIT:
        log.info("[wander] %s — %d agents, NO TIME LIMIT (natural completion / governor CLOSE; "
                 "anti-hang backstop %dm, lossless)", label, agents, WAVE_CEILING_S // 60)
        done, _ = await asyncio.wait({task}, timeout=WAVE_CEILING_S)
    else:
        log.info("[wander] %s — %d agents, soft %dm / grace %dm / hard %dm", label, agents,
                 WAVE_SOFT_S // 60, WAVE_GRACE_S // 60, WAVE_CEILING_S // 60)
        done, _ = await asyncio.wait({task}, timeout=WAVE_SOFT_S)
        if task not in done:
            if _request_halt(prog):
                log.info("[wander] %s soft %dm hit — requested graceful halt, finishing in-flight digs…",
                         label, WAVE_SOFT_S // 60)
            else:
                log.info("[wander] %s soft %dm hit — no session_state yet to halt; waiting grace",
                         label, WAVE_SOFT_S // 60)
            done, _ = await asyncio.wait({task}, timeout=WAVE_GRACE_S)

    # returned (naturally / governor CLOSE / graceful halt) → real, COMPLETE result
    if task in done:
        try:
            return task.result()
        except asyncio.CancelledError:
            return None
        except Exception as e:  # wander itself raised — salvage whatever finalized
            log.warning("[wander] %s task raised (%s) — harvesting partial", label, e)
            return _harvest_partial(prog, cushion, cfg, t0, label)
    # still running at the anti-hang ceiling → cancel, harvest what's finalized
    log.warning("[wander] %s hit anti-hang ceiling %dm — cancelling + harvesting (not a silent hang)",
                label, WAVE_CEILING_S // 60)
    task.cancel()
    try:
        await task
    except BaseException:
        pass
    return _harvest_partial(prog, cushion, cfg, t0, label)


async def main() -> None:
    if not PLAN_ONLY and os.environ.get("CONSTELLAX_GOVERNOR", "0") != "1":
        raise SystemExit("REFUSED: set CONSTELLAX_GOVERNOR=1 to run live (or AUTON_PLAN_ONLY=1 for a dry check).")

    # The governor is fed ONLY by the noticeboard (agents call post_notice ->
    # governor.observe). Without WANDER_AGENT_NOTICEBOARD=1 the governor starves:
    # probes=0, edges=0, no skeleton. The first live run (2026-06-17) hit exactly
    # this — governor attached but unfed. Force the feed on whenever we run live.
    if os.environ.get("WANDER_AGENT_NOTICEBOARD", "0") != "1":
        os.environ["WANDER_AGENT_NOTICEBOARD"] = "1"
        log.info("[config] forced WANDER_AGENT_NOTICEBOARD=1 (governor needs the noticeboard feed)")

    ci_main = base._build_cushion_input()
    question = ci_main.question.content
    angles = parse_required_angles(question)
    # Cycle-1 size = one root agent per sub-question (dynamic), unless overridden.
    # max(1, …) guards an unparseable question (parse falls back to 1 angle anyway).
    main_agents = int(_MAIN_AGENTS_OVERRIDE) if _MAIN_AGENTS_OVERRIDE else max(1, len(angles))
    cushion_text = "PROBLEM:\n{}\n\nVISION:\n{}\n\nHUNCHES:\n{}".format(
        ci_main.problem.content, ci_main.vision.content, ci_main.hunches.content)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _p("=" * 80)
    _p("AUTONOMOUS LEVEL-1 PIPELINE")
    _p("=" * 80)
    _p(f"  checkpoint: {len(angles)} required angles  | cycle cap: {CYCLE_CAP}")
    _p(f"  cycle-1 agents: {main_agents}"
       + ("" if _MAIN_AGENTS_OVERRIDE else f" (= 1 per sub-question, dynamic)"))
    _dom = os.environ.get("WANDER_SEED_DOMAINS", "").strip()
    _p(f"  domains: {'OPEN — full palette (~60 domains)' if not _dom else 'SCOPED → ' + _dom}")
    _p(f"  wander:  DeepSeek-v4-pro (every main) + Haiku (sub-agent nuance) — no Sonnet in digs")
    _p(f"  judges:  halo auditor = Sonnet 4-6  |  coverage checkpoint = {COVERAGE_MODEL} (cross-family)")
    _p(f"  shepherd: cycle-drift sensor {'ON' if AUTON_DRIFT else 'OFF'} (Sonnet, advisory — guides, never halts)")
    _p(f"  Sonnet:  halo + cushion intake + shepherd. (governor stays the single halt authority)")
    _p(f"  synthesis: blender {'ON' if AUTON_BLEND else 'OFF'} (Opus 4-8 + R1 critic, cap ${BLENDER_COST_CAP}, "
       f"GOAL-AWARE — sees pursuit + checkpoint) → coverage on PROPOSALS")
    _p(f"  formalize: R1 {'ON' if AUTON_R1 else 'OFF'} (DeepSeek-R1, blends → math, built-in stage)")
    if NO_TIME_LIMIT:
        _p(f"  bounding: NO TIME LIMIT — natural completion / governor CLOSE; "
           f"anti-hang backstop {WAVE_CEILING_S//60}m (lossless harvest)")
    else:
        _p(f"  bounding: soft {WAVE_SOFT_S//60}m graceful-halt / grace {WAVE_GRACE_S//60}m / "
           f"hard {WAVE_CEILING_S//60}m (lossless)")
    _p(f"  observability → {RUN_DIR}")
    for i, a in enumerate(angles, 1):
        _p(f"    Q{i}: {a[:72]}")
    if PLAN_ONLY:
        _p("\n[PLAN-ONLY] wiring validated, config printed. Exiting before any wave (~$0).")
        return

    # Phase 3: launder the sub-questions into per-agent seeded cushion inputs
    # (chaos-law gated). Done ONCE up front; cycle 1 sends one agent per laundered
    # lead. Aligned/padded to main_agents (extra agents → base cushion fallback).
    _p("\n[seeding] laundering sub-questions → goal-free per-agent leads…")
    seed_inputs = await _seed_inputs_from_angles(angles, ci_main)
    per_agent_main = (seed_inputs + [None] * main_agents)[:main_agents]
    if TRANSLATE_ONLY:
        clean_n = sum(1 for s in per_agent_main if s is not None)
        _p(f"\n[TRANSLATE-ONLY] {clean_n}/{main_agents} cycle-1 agents seeded clean. "
           f"Exiting before any wander (translator cost only).")
        return

    client = LLMClient(mode=ClientMode.LIVE)

    # Phase 4: cross-cycle / cross-run memory (Neo4j, fail-open). run_id from the
    # run dir; cushion_hash keys cross-run recall. No-op instance if disabled or
    # Neo4j is unreachable — the run never depends on it.
    import hashlib
    run_id = RUN_DIR.name
    cushion_hash = hashlib.sha256(
        (ci_main.problem.content + "||" + question).encode()).hexdigest()[:16]
    mem = AutonMemory.from_env(fresh=AUTON_MEMORY_FRESH) if AUTON_MEMORY else AutonMemory(None, "neo4j")
    if AUTON_MEMORY:
        await mem.connect_and_init()
    _p(f"  memory: {'Neo4j ON' if mem.enabled else 'OFF / no-op (JSON is source of truth)'}"
       + (f"  (fresh={AUTON_MEMORY_FRESH}, cushion={cushion_hash})" if mem.enabled else ""))

    all_cards: list = []
    covered: set = set()
    prev_d = 0.0
    leads: list[str] = []          # leads to wander next cycle ([] → cycle 1 uses main cushion)
    dispatched_keys: set = set()   # Phase 6: within-run gap dedup (never re-chase a gap)
    t_start = time.time()
    manifest: dict = {"cushion": "C4", "angles": angles, "cycles": [],
                      "run_dir": str(RUN_DIR), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    run_cost = 0.0

    def _flush_manifest():
        (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    for cycle in range(1, CYCLE_CAP + 1):
        # Hard budget backstop — TRUE cumulative spend (not the wave-only undercount).
        true_cost = client.get_total_cost_estimate()
        if true_cost >= RUN_BUDGET_USD:
            _p(f"\n[BUDGET] ✋ ${true_cost:.2f} ≥ ${RUN_BUDGET_USD:.0f} cap — halting before cycle {cycle}")
            break
        cyc_dir = RUN_DIR / f"cycle-{cycle}"
        cyc_dir.mkdir(parents=True, exist_ok=True)
        _p("\n" + "=" * 80 + f"\nCYCLE {cycle}  (spent ${true_cost:.2f} / ${RUN_BUDGET_USD:.0f})\n" + "=" * 80)
        cyc_rec: dict = {"cycle": cycle, "waves": []}
        cyc_cost = 0.0

        # 1. wander — BOTH cycle 1 and dispatch cycles are now ONE wave with
        #    per-agent seeded cushions (shared governor → skeleton forms across the
        #    wave). Cycle 1 seeds from sub-questions; later cycles seed from the
        #    prioritized gap leads, each cushion-anchored via _seed_cushion_input.
        wave_specs = ([("main", ci_main, main_agents,
                        "cycle-1: 1 agent per laundered sub-question", per_agent_main)] if cycle == 1
                      else [("dispatch", ci_main, len(leads), f"dispatch: {len(leads)} gap(s)",
                             [_seed_cushion_input(ci_main, ld) for ld in leads])])
        sessions = []
        cycle_govs: list = []          # this cycle's per-wave governor records (skeleton-gaps)
        for wtag, wci, wagents, wlabel, wpai in wave_specs:
            wave_dir = cyc_dir / f"wave-{wtag}"
            s = await _wander(client, wci, wagents, wlabel, wave_dir, per_agent_inputs=wpai)
            # governor black box + cost, surfaced live
            gov = _read_governance(wave_dir)
            cycle_govs.append(gov)
            wcost, wtoks, wbymodel = _wave_cost(wave_dir)
            cyc_cost += wcost
            nrep = len(getattr(s, "reports", []) or []) if s else 0
            _p(f"  [wave {wtag}] {nrep} report(s) | "
               f"governor: edges={gov.get('edges_found','?')} probes={gov.get('probes_used','?')} "
               f"rounds={gov.get('emergence_rate_series','?')} final={gov.get('final_action','?')} | "
               f"${wcost:.4f} {wbymodel}")
            cyc_rec["waves"].append({"tag": wtag, "label": wlabel, "agents": wagents,
                                     "reports": nrep, "cost_usd": wcost, "tokens": wtoks,
                                     "cost_by_model": wbymodel, "governor": gov})
            if s:
                sessions.append(s)

        # 2. dossier → cards (accumulate)
        new_cards = 0
        for s in sessions:
            dossier = await build_dossier(s, client, run_master_synthesizer=False,
                                          pipeline_mode="sorter", verify_web=False)
            cs = dossier.all_cards()
            all_cards.extend(cs); new_cards += len(cs)
        _p(f"  [cards] +{new_cards} this cycle, {len(all_cards)} total")
        cyc_rec["cards_new"] = new_cards
        cyc_rec["cards_total"] = len(all_cards)

        # Persist the FULL ArticulatedCards to disk (cumulative, overwritten each
        # cycle → always complete up to the last finished cycle). The 2026-06-18
        # run only kept the 3-field card_bridges below; the full cards (spark/use/
        # limit/domain/confidence/citations) lived in memory and were LOST on exit,
        # forcing a partial-card re-blend. cards.json now lets any re-blank or post-
        # hoc analysis reload complete cards. Fail-open: never blocks/crashes the run.
        try:
            (RUN_DIR / "cards.json").write_text(json.dumps(
                [c.to_dict() for c in all_cards], indent=2, default=str))
        except Exception as e:
            log.warning("[cards] full-card persist failed (ignored): %s", e)

        # Phase 4: persist THIS cycle's new cards to Neo4j (fail-open + timeout-
        # bounded inside record_cycle; never blocks or crashes the run).
        cycle_cards = all_cards[-new_cards:] if new_cards else []
        if mem.enabled and cycle_cards:
            wrote = await mem.record_cycle(run_id=run_id, cushion_hash=cushion_hash,
                                           cycle=cycle, cards=cycle_cards)
            cyc_rec["memory_written"] = wrote
            _p(f"  [memory] Neo4j ← {wrote} card(s) this cycle")
        cyc_rec["card_bridges"] = [
            {"report_id": getattr(c, "report_id", "?"),
             "source_shape": getattr(c, "source_shape", "?"),
             "bridge": (getattr(c, "bridge", "") or "")[:400]} for c in all_cards]

        # 3. halo (GA) blind spots — Sonnet auditor lens
        ga = await audit_cards(cushion=cushion_text, cards=all_cards, client=client)
        ga_spots = [(b.blind_spot, b.severity, b.suggested_angle) for b in ga.blind_spots]
        _p(f"  [halo-GA · Sonnet] {len(ga_spots)} blind spots:")
        for bs, sev, ang in ga_spots:
            _p(f"      ({sev}) {bs[:90]}  →  {(ang or '')[:70]}")
        cyc_rec["halo_blind_spots"] = [{"blind_spot": bs, "severity": sev, "suggested_angle": ang}
                                       for bs, sev, ang in ga_spots]

        # 4. coverage CHECKPOINT on accumulated cards
        cov = await score_coverage(angles, _cards_text(all_cards), model=COVERAGE_MODEL)
        d_t = cov.d_t
        now_cov = {p["idx"] for p in cov.per_angle if p["coverage"] == "covered"}
        open_angles = [(p["idx"], p["angle"]) for p in cov.per_angle if p["coverage"] != "covered"]
        _p(f"  [checkpoint] D_t={d_t}  covered={sorted(now_cov)}  open={[i for i,_ in open_angles]}")
        for p in cov.per_angle:
            _p(f"      Q{p['idx']}: {p['coverage']:9} — {(p.get('by') or '')[:80]}")
        cyc_rec["coverage"] = {"d_t": d_t, "per_angle": cov.per_angle}

        # 5. signals
        n_t = len(now_cov - covered)
        c_t = round(abs(d_t - prev_d), 3)
        covered |= now_cov; prev_d = d_t
        _p(f"  [signals] novelty={n_t} change={c_t}")
        cyc_rec["signals"] = {"novelty": n_t, "change": c_t}

        # 5b. shepherd — cycle-trajectory drift sensor. ADVISORY: surfaces the
        #     loop's direction (on_track/circling/drifting) + a refocus nudge for
        #     the dispatcher (next phase). Never halts. Sonnet, fail-open.
        if AUTON_DRIFT:
            new_bridges = [(getattr(c, "bridge", "") or "") for c in cycle_cards]
            prior_bridges = [(getattr(c, "bridge", "") or "")
                             for c in all_cards[:len(all_cards) - new_cards]]
            drift = await check_cycle_drift(
                cushion_problem=ci_main.problem.content, cushion_question=question,
                cycle=cycle, new_card_bridges=new_bridges, prior_card_bridges=prior_bridges,
                open_angles=[a for _, a in open_angles], client=client)
            cyc_cost += drift.cost_usd
            icon = {"on_track": "✓", "circling": "↻", "drifting": "⤳"}.get(drift.status, "?")
            _p(f"  [shepherd · Sonnet] {icon} {drift.status}  momentum={drift.momentum:.2f}"
               + (f"  refocus→ {drift.refocus[:64]}" if drift.refocus else "")
               + ("" if drift.ok else "  (fail-open default)"))
            if drift.rationale:
                _p(f"      {drift.rationale[:100]}")
            cyc_rec["shepherd"] = drift.to_dict()

        # 6. DISPATCH (Phase 6) — the Shepherd-dispatcher. Fuse FIVE signals into
        #    one PRIORITIZED gap list, drop already-chased gaps (within-run + Neo4j
        #    cross-run), launder each through the chaos gate (fail-closed), and size
        #    = clamp(distinct fresh gaps, 1, CAP). Priority: structural holes first
        #    (governor skeleton-gaps + high-sev halo) → open coverage angles → rest
        #    halo → shepherd refocus (tiebreak). Neo4j memory FILTERS, never ranks.
        sev_rank = {"high": 0, "medium": 1, "low": 2}
        ga_sorted = sorted(ga_spots, key=lambda x: sev_rank.get(x[1], 1))
        skeleton_gaps: list = []
        for g in cycle_govs:
            skeleton_gaps += ((g.get("skeleton_gaps") or {}).get("isolated_findings") or [])
        high_halo = [bs for bs, sev, _ in ga_sorted if sev == "high"]
        rest_halo = [bs for bs, sev, _ in ga_sorted if sev != "high"]
        open_texts = [a for _, a in open_angles]
        refocus = [drift.refocus] if (AUTON_DRIFT and getattr(drift, "refocus", "")) else []
        ranked_gaps = skeleton_gaps + high_halo + open_texts + rest_halo + refocus

        # cross-run dedup keys (Neo4j, fail-open) ∪ within-run keys
        prior_keys = await mem.recall_gap_keys(cushion_hash=cushion_hash, exclude_run_id=run_id)
        seen_keys = set(dispatched_keys) | set(prior_keys)
        cyc_rec["dispatch_signals"] = {
            "skeleton_gaps": len(skeleton_gaps), "high_halo": len(high_halo),
            "open_angles": len(open_texts), "rest_halo": len(rest_halo),
            "shepherd_refocus": len(refocus), "prior_keys_known": len(prior_keys)}
        _p(f"  [dispatch] signals: skeleton={len(skeleton_gaps)} high-halo={len(high_halo)} "
           f"open-angles={len(open_texts)} rest-halo={len(rest_halo)} refocus={len(refocus)} "
           f"| {len(prior_keys)} prior gap(s) known")

        leads = []
        chased_this_cycle: list = []   # (key, original_text) to persist cross-run
        for gap in ranked_gaps:
            if len(leads) >= DISPATCH_CAP:
                break
            k = _gap_key(gap)
            if not k or k in seen_keys:
                continue                      # already chased (this run or prior) → skip
            seen_keys.add(k)
            tr = await translate_gap(gap)
            if tr.clean:
                leads.append(tr.lead)
                dispatched_keys.add(k)
                chased_this_cycle.append((k, gap))
            else:
                log.warning("[translate] skipped a gap (fail-closed leak guard): %s", tr.reasons)
        if mem.enabled and chased_this_cycle:
            await mem.record_gaps(run_id=run_id, cushion_hash=cushion_hash,
                                  cycle=cycle, gaps=chased_this_cycle)
        cyc_rec["dispatch_leads"] = list(leads)

        # 7. cost
        run_cost += cyc_cost
        cyc_rec["cost_usd"] = round(cyc_cost, 4)
        _p(f"  [cost] cycle ${cyc_cost:.4f} | run ${run_cost:.4f}")

        # 8. HALT? coverage-complete & settled, OR cycle cap, OR no FRESH gaps left
        #    (skeleton converged / territory exhausted → nothing new to dispatch).
        halt_reason = None
        if (n_t <= TH_NOVELTY and d_t >= TH_COVERAGE and c_t <= TH_CHANGE):
            halt_reason = "checkpoint complete & settled — question covered"
        elif cycle == CYCLE_CAP:
            halt_reason = f"cycle cap {CYCLE_CAP} reached (open: {[i for i,_ in open_angles]})"
        elif not leads:
            halt_reason = "no fresh gaps to chase — skeleton converged / territory exhausted"
        if halt_reason:
            cyc_rec["halt"] = halt_reason
        (cyc_dir / "cycle.json").write_text(json.dumps(cyc_rec, indent=2, default=str))
        manifest["cycles"].append(cyc_rec); _flush_manifest()
        if halt_reason:
            _p(f"  [HALT] ✋ {halt_reason}")
            break
        _p(f"  [dispatch] next cycle: {len(leads)} agent(s), one per fresh gap "
           f"(clamp 1..{DISPATCH_CAP}):")
        for ld in leads:
            _p(f"      → {ld[:100]}")

    # ===== SYNTHESIS — the blender: accumulated cards → PROPOSALS (fixes D_t=0) =====
    # Run ONCE on all cards. Coverage on the BLENDS is the true D_t (raw-card D_t
    # only ever measured diagnosis, not proposals). Cost-capped + non-fatal.
    blends: list = []
    blend_d_t = None
    _pre_blend_cost = client.get_total_cost_estimate()
    if AUTON_BLEND and all_cards and (_pre_blend_cost + BLENDER_COST_CAP) > RUN_BUDGET_USD:
        _p(f"\n[BUDGET] skipping blender — ${_pre_blend_cost:.2f} + blender cap ${BLENDER_COST_CAP} "
           f"would exceed ${RUN_BUDGET_USD:.0f}. Cards/coverage preserved; no synthesis.")
    elif AUTON_BLEND and all_cards:
        # Cap the cards fed to the blender (full set can truncate Opus draft JSON +
        # spike per-call cost). Keep the MOST RECENT cards — later cycles are the
        # dispatcher's targeted, gap-closing wanders. 0 = no cap.
        blend_cards = all_cards[-BLEND_CARD_CAP:] if BLEND_CARD_CAP > 0 else all_cards
        _p("\n" + "=" * 80 + f"\nSYNTHESIS — blender fusing {len(blend_cards)} of {len(all_cards)} cards "
           f"→ proposals (cap ${BLENDER_COST_CAP})\n" + "=" * 80)
        try:
            # GOAL-AWARE synthesis: hand the blender the cushion so it sees the pursuit,
            # the CHECKPOINT question, and the constellation — and SELECTS/fuses the cards
            # that advance the goal, instead of a goal-blind fusion. The chaos law protects
            # the WANDER (already done), not this final synthesizer. compose once (~$0.05);
            # raw_input carries the question even though graph extraction omits it.
            blend_cushion = await compose_cushion(ci_main, client, session_id="auton-blend",
                                                  auto_enrich=False)
            ms = await master_synthesize(cushion=blend_cushion, cards=blend_cards, synthesis_map=None,
                                         client=client, cost_ceiling_usd=BLENDER_COST_CAP)
            blends = list(getattr(ms, "master_fusions", []) or [])
            _p(f"  [blends] {len(blends)} fusion(s):")
            for i, b in enumerate(blends, 1):
                _p(f"    B{i} [{getattr(b, 'agreement_status', '?')}] {(getattr(b, 'title', '') or '')[:84]}")
            blend_text = "\n".join(
                f"[B{i}] {getattr(b, 'title', '')}: {getattr(b, 'claim', '')} — "
                f"{(getattr(b, 'reasoning', '') or '')[:300]}"
                for i, b in enumerate(blends, 1) if getattr(b, "claim", ""))
            if blend_text.strip():
                bc = await score_coverage(angles, blend_text, model=COVERAGE_MODEL)
                blend_d_t = bc.d_t
                _p(f"  [coverage·BLENDS] D_t={bc.d_t}  (raw-card D_t was {prev_d})")
                for p in bc.per_angle:
                    _p(f"      Q{p['idx']}: {p['coverage']:9} — {(p.get('by') or '')[:80]}")
                manifest["synthesis_coverage"] = {"d_t": bc.d_t, "per_angle": bc.per_angle}
        except Exception as e:
            _p(f"  [blends] synthesis failed (non-fatal): {type(e).__name__}: {e}")
        try:
            (RUN_DIR / "blends.json").write_text(json.dumps(
                [b.to_dict() for b in blends], indent=2, default=str))
        except Exception as e:
            log.warning("[blends] persist failed (ignored): %s", e)
        manifest["synthesis"] = {"n_blends": len(blends), "blend_d_t": blend_d_t}
        _flush_manifest()

    # ===== R1 FORMALIZATION — built-in stage: blends → testable math =====
    # Adapter: our MasterFusionReports → the collision Blend dict shape the
    # formalizer's _blend_text expects (thesis←claim, mechanism←reasoning,
    # structure←title, source methods enriched from all_cards by report_id).
    n_formalized = 0
    if AUTON_R1 and blends:
        _p("\n" + "=" * 80 + f"\nR1 FORMALIZATION — DeepSeek-R1 grounding {len(blends)} blend(s) "
           f"into math\n" + "=" * 80)
        shape_by_id = {getattr(c, "report_id", ""): getattr(c, "source_shape", "") for c in all_cards}

        def _fusion_to_blend_dict(b, i):
            src = []
            for c in (getattr(b, "citations", []) or []):
                rid = getattr(c, "report_id", "") or (c.get("report_id") if isinstance(c, dict) else "")
                src.append({"report_id": rid, "source_shape": shape_by_id.get(rid, "")})
            return {"blend_id": f"B{i}", "thesis": getattr(b, "claim", "") or "",
                    "mechanism": getattr(b, "reasoning", "") or "",
                    "emergent_structure": getattr(b, "title", "") or "",
                    "advances_cushion": getattr(b, "limit", "") or "",
                    "selection": {"tension": ""}, "source_cards": src}

        blend_dicts = [_fusion_to_blend_dict(b, i) for i, b in enumerate(blends, 1)
                       if getattr(b, "claim", "")]
        try:
            rep = await formalize_blends(
                blend_dicts,
                on_progress=lambda ev, d: _p(f"    R1 {d.get('blend_id')}: "
                                             f"formalizable={d.get('formalizable')} ok={d.get('ok')} "
                                             f"${d.get('cost', 0)}"))
            n_formalized = len(getattr(rep, "formalizations", []) or [])
            try:
                (RUN_DIR / "formalize.json").write_text(json.dumps(rep.to_dict(), indent=2, default=str))
                (RUN_DIR / "formalize.md").write_text(render_markdown(rep))
            except Exception as e:
                log.warning("[r1] artifact write failed (ignored): %s", e)
            _p(f"  [R1] {n_formalized} formalization(s), ${getattr(rep, 'total_cost_usd', 0.0):.4f} "
               f"→ formalize.json + formalize.md")
            manifest["formalization"] = {"n": n_formalized,
                                         "cost_usd": round(getattr(rep, "total_cost_usd", 0.0), 4)}
            _flush_manifest()
        except Exception as e:
            _p(f"  [R1] formalization failed (non-fatal): {type(e).__name__}: {e}")

    true_total = client.get_total_cost_estimate()
    manifest["terminal"] = {"d_t": prev_d, "blend_d_t": blend_d_t, "covered": sorted(covered),
                            "cards": len(all_cards), "blends": len(blends),
                            "formalized": n_formalized,
                            "cost_usd_wave_tracked": round(run_cost, 4),
                            "cost_usd_true": round(true_total, 4),
                            "budget_usd": RUN_BUDGET_USD, "elapsed_s": round(time.time() - t_start)}
    _flush_manifest()
    _p("\n" + "=" * 80)
    _p(f"TERMINAL: raw-card D_t={prev_d}  blend D_t={blend_d_t}  blends={len(blends)}  "
       f"formalized={n_formalized}  covered={sorted(covered)}  cards={len(all_cards)}  "
       f"TRUE cost=${true_total:.2f} / ${RUN_BUDGET_USD:.0f}  (wave-tracked ${run_cost:.2f})  "
       f"elapsed={time.time()-t_start:.0f}s")
    _p(f"artifacts → {RUN_DIR}/  (manifest.json, blends.json, formalize.json/.md, "
       f"per-cycle cycle.json, per-wave calls.jsonl/governance.json)")
    _p("=" * 80)
    await mem.close()


if __name__ == "__main__":
    asyncio.run(main())
