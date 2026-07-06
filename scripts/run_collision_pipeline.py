"""
run_collision_pipeline.py — the FULL collision pipeline, end to end.

cushion → wander → verified-sort (+ web) → blend (+ discovery_path genealogy)
→ drift-check → blend-verify (+ web; known / adjacent / novel / flawed).

A HALO AUDITOR (blend-03 Phase 1, OBSERVER) sits on top and audits each layer
for blind spots as its artifact appears — step by step: the cushion (before
any work), then the cards (after the wander), then the blends (after the
collision). It writes blind spots down and acts on NOTHING; it is fail-open,
so a halo failure can never crash the paid pipeline below it.

Writes a single auditable `run_record.json` (the documentation environment,
with a `trace` index for reverse-engineering each discovery), plus `audit.json`
(the halo's blind spots) and per-stage artifacts, into runs/r-collision/<ts>/.
A readable halo markdown also lands in ~/Downloads.

Models, domains, mode all come from scripts/control_room.py — every seat is
passed its control-room model EXPLICITLY (no reliance on module defaults).
As configured 2026-06-14: wander / sort / blend / blend-verify on DeepSeek V4
Pro (cheap workhorse); drift-check + halo on Sonnet 4.6 (the quality eyes).
The cushion (Nikhil's real pursuit/vision/hunches) is reused from
run_fable_sorter_6agents.

LIVE — this spends. Expect MEANINGFULLY less than the old all-Sonnet/Opus
$5 run: the two heavy spenders (wander, blend) are now DeepSeek; only
drift-check + the 3 halo calls are Sonnet, all small. Plus web-search API
calls (Exa/Tavily). Do NOT run casually.

Usage:
    cd /Users/nikhil/Desktop/reasoningEngine
    python scripts/run_collision_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import control_room
# Reuse the cushion text + JSON helpers + mode map from the sorter runner (DRY).
import run_fable_sorter_6agents as base

from src.llm.client import ClientMode, LLMClient
from src.wandering.collision_pipeline import (
    CollisionProgress,
    build_run_record,
    run_collision_pipeline,
)
from src.wandering.composer import compose_cushion
from src.wandering.dossier import build_dossier
from src.wandering.fetcher import search_chain, web_search_fetcher
from src.wandering.master_sorter import SortedReport
from src.wandering.halo_auditor import (
    AuditReport,
    LayerAudit,
    audit_blends,
    audit_cards,
    audit_cushion,
)
from src.wandering.quality_ranker import rank_blends
from src.wandering.runtime import WanderingConfig, run_wandering_session

# Sibling script — renders ONE consolidated readable report (run + halo audit).
import render_run_report

AGENT_COUNT    = control_room.WANDER_AGENTS
MODEL_MIX      = (control_room.WANDER_MODEL,) * AGENT_COUNT
WANDERING_MODE = base._MODE_MAP[control_room.WANDER_MODE]
TIME_BUDGET_S  = 30 * 60
TOKENS_PER_AGENT = 30_000
OUTPUT_ROOT    = REPO_ROOT / "runs" / "r-collision"


async def run() -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    log_handler = logging.FileHandler(out_dir / "run.log")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[log_handler, logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("run_collision")

    log.info("=" * 70)
    log.info("CONSTELLAX COLLISION PIPELINE — FULL RUN %s", timestamp)
    log.info("=" * 70)

    cr = control_room.as_dict()
    log.info("CONTROL ROOM: %s", cr)
    seed_env = control_room.seed_domains_env()
    if seed_env:
        os.environ["WANDER_SEED_DOMAINS"] = seed_env
        log.info("Domain narrowing ACTIVE: %s", seed_env)
    else:
        os.environ.pop("WANDER_SEED_DOMAINS", None)

    # Contribution board (additive, positive-sum dig directive). Off by default.
    os.environ["WANDER_CONTRIBUTION_BOARD"] = "1" if getattr(control_room, "CONTRIBUTION_BOARD", False) else "0"
    log.info("Contribution board: %s", "ON" if getattr(control_room, "CONTRIBUTION_BOARD", False) else "off")

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        git_sha, git_branch = "(unavailable)", "(unavailable)"

    cushion_input = base._build_cushion_input()
    base._write_json(out_dir / "cushion_input.json", {
        "problem":  cushion_input.problem.content,
        "question": cushion_input.question.content,
        "vision":   cushion_input.vision.content,
        "hunches":  cushion_input.hunches.content,
    })

    # Cushion text the halo auditor + judges read — same shape as the post-hoc
    # loaders (run_halo_audit._load / run_quality_ranker._load) so inline
    # blind-spot quality matches the validated post-hoc runs exactly. QUESTION
    # leads after PROBLEM — it is the checkpoint every judge measures against.
    cushion_text = "PROBLEM:\n{}\n\nQUESTION:\n{}\n\nVISION:\n{}\n\nHUNCHES:\n{}".format(
        cushion_input.problem.content,
        cushion_input.question.content,
        cushion_input.vision.content,
        cushion_input.hunches.content,
    )

    client = LLMClient(mode=ClientMode.LIVE)

    # ---- Halo auditor (blend-03 Phase 1, OBSERVER) ---------------------
    # Sits on top of the pipeline; audits each layer for blind spots as its
    # artifact appears (cushion → cards → blends), step by step. OBSERVE-ONLY:
    # it writes blind spots down and acts on nothing. FAIL-OPEN: a halo failure
    # is caught here and can NEVER take down the paid pipeline below it.
    halo_model = control_room.AUDITOR_MODEL
    halo_layers: dict[str, LayerAudit] = {}

    async def _safe_halo(layer: str, make_coro) -> LayerAudit:
        h0 = time.time()
        try:
            audit = await make_coro()
        except Exception as e:  # the halo must never crash the paid run
            log.warning("[halo] %s audit raised (ignored): %s", layer, e)
            audit = LayerAudit(layer=layer, ok=False,
                               note=f"raised: {type(e).__name__}: {e}")
        halo_layers[layer] = audit
        log.info("[halo] %s — %d blind spots, $%.4f, %.1fs (ok=%s)",
                 layer, len(audit.blind_spots), audit.cost_usd,
                 time.time() - h0, audit.ok)
        return audit

    # ---- Phase 1: cushion ----------------------------------------------
    t = time.time()
    log.info("[1] composing cushion…")
    cushion = await compose_cushion(cushion_input, client, user_id=None,
                                    session_id=f"r-collision-{timestamp}", auto_enrich=False)
    cushion_s = time.time() - t
    try:
        base._write_json(out_dir / "cushion.json", cushion)
    except Exception as e:
        base._write_json(out_dir / "cushion.json", {"error": str(e)})

    # Halo checkpoint 1 — audit the QUESTION ITSELF, before any work.
    await _safe_halo("cushion", lambda: audit_cushion(
        cushion=cushion_text, client=client, model=halo_model))

    # ---- Phase 2: wander -----------------------------------------------
    config = WanderingConfig(mode=WANDERING_MODE, agents=AGENT_COUNT,
                             time_budget_seconds=TIME_BUDGET_S, tokens_per_agent=TOKENS_PER_AGENT,
                             model_mix=MODEL_MIX, session_id=f"r-collision-{timestamp}")
    t = time.time()
    log.info("[2] wandering (%d agents, %s)…", AGENT_COUNT, WANDERING_MODE.value)
    session = await run_wandering_session(cushion, config, client, fetcher=web_search_fetcher)
    wander_s = time.time() - t
    log.info("[2] wander done — %d reports, %.0fs", session.report_count(), wander_s)
    try:
        base._write_json(out_dir / "session.json", session)
    except Exception as e:
        base._write_json(out_dir / "session.json", {"error": str(e), "reports": session.report_count()})

    # ---- Phase 3: dossier + VERIFIED sort (stages 1-2) -----------------
    t = time.time()
    log.info("[3] dossier (sort removed — question-aware blender validated)…")
    # SORT REMOVED 2026-06-15: the known/unplaced/invalid web-sort cost ~15 min
    # (a 37-card web crawl) and produced bins the question-aware blender no longer
    # needs — validated that the strengthened blender recovers the binned baseline
    # without it (drift 0, 3 novel + 1 adjacent). run_master_synthesizer=False
    # skips the entire sorter+web block; the high/medium/low bands are built before
    # it (dossier.py Step 4) and are unaffected, so all_cards() is identical to the
    # sorted path. The blender sees every card as "unsorted" via the empty
    # SortedReport() fed to the collision below.
    dossier = await build_dossier(
        session, client,
        run_master_synthesizer=False, pipeline_mode="sorter",
        verify_web=False,
    )
    dossier_s = time.time() - t
    base._write_json(out_dir / "dossier.json", dossier)
    if dossier.master_sorted is not None:
        base._write_json(out_dir / "sorted.json", dossier.master_sorted)
        log.info("[3] sort: known=%d invalid=%d unplaced=%d",
                 len(dossier.master_sorted.known), len(dossier.master_sorted.invalid),
                 len(dossier.master_sorted.unplaced))

    # Sort intentionally removed — master_sorted is None by design; the collision
    # runs on the full unsorted card stack (empty SortedReport below).
    log.info("[3] sort skipped (removed) — blender runs on the full unsorted stack")

    # Halo checkpoint 2 — audit the WANDER's coverage (cards now exist).
    await _safe_halo("cards", lambda: audit_cards(
        cushion=cushion_text, cards=dossier.all_cards(),
        client=client, model=halo_model))

    # ---- Phase 4: collision (stages 3-5) -------------------------------
    t = time.time()
    log.info("[4] collision: blend → drift-check → blend-verify…")
    collision = await run_collision_pipeline(
        cushion=cushion, cards=dossier.all_cards(), sorted_report=SortedReport(),
        client=client,
        blender_model=control_room.BLENDER_MODEL,
        drift_model=control_room.DRIFT_CHECKER_MODEL,
        query_model=control_room.SORTER_MODEL,
        verify_model=control_room.SORTER_MODEL,
        search_fn=search_chain,
        progress=CollisionProgress(),
    )
    collision_s = time.time() - t
    base._write_json(out_dir / "collision.json", collision)

    ver = collision.verification
    log.info("[4] collision done — blends=%d quarantined=%d | known=%d adjacent=%d novel=%d flawed=%d",
             len(collision.blends.blends) if collision.blends else 0,
             len(collision.quarantined_blend_ids),
             len(ver.known) if ver else 0, len(ver.adjacent) if ver else 0,
             len(ver.novel) if ver else 0, len(ver.flawed) if ver else 0)

    # Halo checkpoint 3 — audit the BLENDS (the holes their lanes leave).
    _blends_for_halo = collision.blends.blends if collision.blends else []
    await _safe_halo("blends", lambda: audit_blends(
        cushion=cushion_text, blends=_blends_for_halo,
        client=client, model=halo_model))

    # ---- Phase 5: unified run record (documentation environment) -------
    run_record = build_run_record(
        cushion_problem=cushion_input.problem.content,
        dossier_dict=json.loads(json.dumps(dossier, default=base._json_default)),
        collision=collision,
    )
    base._write_json(out_dir / "run_record.json", run_record)

    # ---- Halo report — assemble the 3 checkpoints, persist, surface ----
    halo_report = AuditReport(
        cushion_audit=halo_layers.get("cushion"),
        cards_audit=halo_layers.get("cards"),
        blends_audit=halo_layers.get("blends"),
        model=halo_model,
        total_cost_usd=sum(a.cost_usd for a in halo_layers.values()),
    )
    base._write_json(out_dir / "audit.json", halo_report.to_dict())
    log.info("[halo] total — %d blind spots across %d layers, $%.4f",
             len(halo_report.all_blind_spots()), len(halo_layers),
             halo_report.total_cost_usd)

    # ---- Final alignment pass (quality_ranker, stage-1 closer) ---------
    # Ranks the verified blends by advancement toward the cushion (primary) +
    # gap-coverage (secondary), protecting new-gap openers. RANKS, never
    # deletes. Fail-soft: a failure here never crashes the run.
    quality = None
    try:
        _ver = collision.verification
        _novelty: dict[str, str] = {}
        if _ver is not None:
            for _bn, _items in (("known", _ver.known), ("adjacent", _ver.adjacent),
                                ("novel", _ver.novel), ("flawed", _ver.flawed)):
                for _v in _items:
                    _novelty[_v.blend_id] = _bn
        quality = await rank_blends(
            cushion=cushion_text,
            blends=collision.blends.blends if collision.blends else [],
            blind_spots=halo_report.all_blind_spots(),
            novelty_by_id=_novelty,
            client=client,
            model=control_room.RANKER_MODEL,
        )
        base._write_json(out_dir / "quality.json", quality.to_dict())
        log.info("[quality] ranked %d blends, $%.4f (top: %s)",
                 len(quality.ranked), quality.total_cost_usd,
                 quality.ranked[0].blend_id if quality.ranked else "—")
    except Exception as e:
        log.warning("[quality] ranking failed (ignored): %s", e)

    # ONE consolidated readable report (run + halo + ranking) after run_meta.
    report_md_path = Path.home() / "Downloads" / f"constellax_collision_{timestamp}.md"

    meta = {
        "timestamp": timestamp, "git_sha": git_sha, "git_branch": git_branch,
        "control_room": cr,
        "durations_seconds": {
            "cushion": round(cushion_s, 2), "wander": round(wander_s, 2),
            "dossier_sort": round(dossier_s, 2), "collision": round(collision_s, 2),
        },
        "sort": None if dossier.master_sorted is None else {
            "known": len(dossier.master_sorted.known),
            "invalid": len(dossier.master_sorted.invalid),
            "unplaced": len(dossier.master_sorted.unplaced),
        },
        "collision": {
            "blends": len(collision.blends.blends) if collision.blends else 0,
            "quarantined": len(collision.quarantined_blend_ids),
            "bins": {
                "known": len(ver.known) if ver else 0,
                "adjacent": len(ver.adjacent) if ver else 0,
                "novel": len(ver.novel) if ver else 0,
                "flawed": len(ver.flawed) if ver else 0,
            },
            "stage_costs": collision.stage_costs,
            "collision_cost_usd": round(collision.total_cost_usd, 4),
        },
        "halo": {
            "model": halo_model,
            "blind_spots": {
                layer: len(halo_layers[layer].blind_spots)
                for layer in ("cushion", "cards", "blends") if layer in halo_layers
            },
            "ok": {layer: halo_layers[layer].ok for layer in halo_layers},
            "total_cost_usd": round(halo_report.total_cost_usd, 4),
            "report_md": str(report_md_path),
        },
        "quality": {
            "model": control_room.RANKER_MODEL,
            "ranked": len(quality.ranked) if quality else 0,
            "top_blend": quality.ranked[0].blend_id if (quality and quality.ranked) else None,
            "total_cost_usd": round(quality.total_cost_usd, 4) if quality else 0.0,
        },
        "output_dir": str(out_dir),
    }
    base._write_json(out_dir / "run_meta.json", meta)

    # ---- ONE consolidated readable .md (run report + halo audit) -------
    try:
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_md_path.write_text(render_run_report.render(out_dir))
        log.info("[report] consolidated readable report -> %s", report_md_path)
    except Exception as e:
        log.warning("[report] render failed (ignored): %s", e)

    log.info("=" * 70)
    log.info("COLLISION RUN COMPLETE — artifacts in %s", out_dir)
    log.info("=" * 70)
    return meta


if __name__ == "__main__":
    result = asyncio.run(run())
    print("\nCOLLISION RUN META:")
    print(json.dumps(result, indent=2, default=base._json_default))
