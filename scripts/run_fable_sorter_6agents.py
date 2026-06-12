"""
run_fable_sorter_6agents.py — Wandering Room session with master_sorter tier.

ONE wandering session (6 parallel Sonnet 4.6 agents) → one Fable 5 sort pass.
Synthesizer NOT run; this is the sorter-only probe.

Inputs are HARD-CODED at the top of this file — the user's actual pursuit /
vision / hunches as told during the conversation that authored this script
(2026-06-12). Cushion intake is verbatim (transcription artifacts like
"uh", duplicated-words, and the misspelling "Heinzberg" → "Heisenberg"
have been removed; substance untouched).

Outputs land in runs/r-fable-sorter-6agents/<timestamp>/ with one file
per artifact. Tracked in git. The /tmp wipe will never lose another run.

Usage:
    cd /Users/nikhil/Desktop/reasoningEngine
    source .venv/bin/activate
    python scripts/run_fable_sorter_6agents.py
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
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Make repo root importable so this script can be run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

from src.llm.client import LLMClient, ClientMode
from src.wandering.composer import compose_cushion
from src.wandering.cushion import CushionField, CushionInput, SkipReason
from src.wandering.dossier import build_dossier
from src.wandering.runtime import (
    WanderingConfig,
    WanderingMode,
    run_wandering_session,
)


# ---------------------------------------------------------------------------
# Run config — locked at script authorship time
# ---------------------------------------------------------------------------

AGENT_COUNT      = 6
MODEL_MIX        = ("anthropic/claude-sonnet-4-6",) * AGENT_COUNT
WANDERING_MODE   = WanderingMode.MULTI_PENDULUM
TIME_BUDGET_S    = 30 * 60                # 30 min default
TOKENS_PER_AGENT = 30_000
SORT_COST_CAP    = 8.00                   # $ ceiling for the master_sort pass
OUTPUT_ROOT      = REPO_ROOT / "runs" / "r-fable-sorter-6agents"


# ---------------------------------------------------------------------------
# THE QUESTION — Nikhil's real pursuit + vision + hunches
# ---------------------------------------------------------------------------
# Pass-through. Voice preserved. Light de-transcription only (filler removed,
# duplicate words collapsed, "Heinzberg" → "Heisenberg"). Substance intact.

PURSUIT_TEXT = (
    "I'm currently working on a system that helps people advance their concepts "
    "or the work they're currently working on. If there's a case or research "
    "that's already published, based on that inheritance, based on that work, we "
    "can advance it. If I'm the one who didn't want to be at the same level and "
    "I'm just having some similar inspirations and I wanted to take that method "
    "or research paper or concept to a next level — how can I do this? I'm "
    "making the system like that. Or I'm having a whole new concept: if there's "
    "a system I'm building and I wanted to make a feature out of it and I'm "
    "having some vague inspirations of that system but I don't know how to "
    "articulate it, I don't know how to make it live — there's a system that "
    "can do that part of it."
)

VISION_TEXT = (
    "If a person tells the inspiration, the pursuit, and any unfinished work or "
    "unfinished threads in their mind — some of the big inspirations — to the "
    "model, then the model has something to anchor against. These three "
    "information points — the current pursuit (whether half-baked or not), the "
    "undone inspirations, and the vision the user is actually visualizing in "
    "their brain — can give enough data to the model to keep intact its score "
    "and keep wandering around through the Internet to find native and "
    "metaphorical work in the respective domains, whether it's math, physics, "
    "psychology, or even theater, literature, anything."
)

HUNCHES_TEXT = (
    "Based on what I told you about this system, you can't be working on it "
    "without some theoretical seeds. I stumbled into many concepts — Heisenberg, "
    "Markov chain, chaos theory. Chaos theory is what lets the model wander the "
    "Internet unhindered, like total chaos. There's no prediction of where the "
    "model will go next, but we can track it. But that track is the past, not "
    "the future. We cannot predict where it will end up. That is the "
    "spontaneousness I want to give to the model. The Markov chain also plays "
    "an important role in my thinking. The same goes with the Heisenberg "
    "principle — it doesn't provide a clear understanding, but a half-"
    "understanding. Just a structure. Just a skeleton."
)


def _build_cushion_input() -> CushionInput:
    """Build the 4-field intake from the user's 3 components.

    Mapping:
      pursuit  → problem      (the actual problem — concrete description)
      vision   → vision       (future trajectory)
      hunches  → current_map  (initial inspirations + related domains)
      (none)   → context      (skipped intentionally; user didn't provide one)
    """
    return CushionInput(
        problem=CushionField(name="problem", content=PURSUIT_TEXT),
        vision=CushionField(name="vision", content=VISION_TEXT),
        current_map=CushionField(name="current_map", content=HUNCHES_TEXT),
        context=CushionField(
            name="context", content="", skip_reason=SkipReason.SKIPPED_AFTER_PROMPT,
        ),
    )


# ---------------------------------------------------------------------------
# JSON serialization helpers — robust against dataclasses, enums, sets, etc.
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback for objects json doesn't know how to render."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # Last resort — let str() try
    return str(obj)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, default=_json_default, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run() -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging — file handler + console
    log_path = out_dir / "run.log"
    log_handler = logging.FileHandler(log_path)
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s"
    ))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[log_handler, logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("run_fable_sorter")

    log.info("=" * 70)
    log.info("FABLE 5 SORTER — 6 SONNET 4.6 AGENTS — RUN %s", timestamp)
    log.info("=" * 70)
    log.info("Output directory: %s", out_dir)

    # Git revision for the run_meta — so we know exactly which code shipped
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        git_sha, git_branch = "(unavailable)", "(unavailable)"

    cushion_input = _build_cushion_input()
    _write_json(out_dir / "cushion_input.json", {
        "problem":     cushion_input.problem.content,
        "context":     cushion_input.context.content,
        "vision":      cushion_input.vision.content,
        "current_map": cushion_input.current_map.content,
    })
    log.info("Cushion input persisted (cushion_input.json)")

    client = LLMClient(mode=ClientMode.LIVE)

    # ---- Phase 1: compose cushion --------------------------------------
    t_phase = time.time()
    log.info("[Phase 1] composing cushion (Sonnet extraction + Gemini embeddings)…")
    try:
        cushion = await compose_cushion(
            cushion_input,
            client,
            user_id=None,
            session_id=f"r-fable-sorter-{timestamp}",
            auto_enrich=False,            # no project memory injection
        )
    except Exception as e:
        log.error("compose_cushion failed: %s\n%s", e, traceback.format_exc())
        raise
    cushion_duration = time.time() - t_phase
    log.info("[Phase 1] cushion composed in %.1fs", cushion_duration)
    try:
        _write_json(out_dir / "cushion.json", cushion)
    except Exception as e:
        log.warning("cushion.json dump failed: %s — writing minimal shape", e)
        _write_json(out_dir / "cushion.json", {"error": str(e), "type": type(cushion).__name__})

    # ---- Phase 2: wandering session ------------------------------------
    config = WanderingConfig(
        mode=WANDERING_MODE,
        agents=AGENT_COUNT,
        time_budget_seconds=TIME_BUDGET_S,
        tokens_per_agent=TOKENS_PER_AGENT,
        model_mix=MODEL_MIX,
        session_id=f"r-fable-sorter-{timestamp}",
    )
    t_phase = time.time()
    log.info("[Phase 2] running %d-agent wander (%s, %.0fs budget)…",
             AGENT_COUNT, WANDERING_MODE.value, TIME_BUDGET_S)
    try:
        session = await run_wandering_session(cushion, config, client)
    except Exception as e:
        log.error("run_wandering_session failed: %s\n%s", e, traceback.format_exc())
        raise
    wander_duration = time.time() - t_phase
    log.info("[Phase 2] wander complete in %.1fs — %d reports / %d traces, %d tokens",
             wander_duration, session.report_count(), session.agent_count(),
             session.total_tokens_spent)

    # Cohort integrity — surface (don't gate) before paying for sort
    ok, problems = session.validate_cohort_integrity()
    if not ok:
        log.warning("cohort integrity problems detected: %s", problems)
    else:
        log.info("cohort integrity OK")

    try:
        _write_json(out_dir / "session.json", session)
    except Exception as e:
        log.warning("session.json dump failed: %s — minimal shape", e)
        _write_json(out_dir / "session.json", {
            "error":         str(e),
            "report_count":  session.report_count(),
            "agent_count":   session.agent_count(),
            "tokens_spent":  session.total_tokens_spent,
        })

    # ---- Phase 3: build dossier + master_sorter ------------------------
    t_phase = time.time()
    log.info("[Phase 3] building dossier + Fable 5 sort…")
    try:
        dossier = await build_dossier(
            session, client,
            run_master_synthesizer=True,
            pipeline_mode="sorter",
            master_synth_cost_ceiling_usd=SORT_COST_CAP,
        )
    except Exception as e:
        log.error("build_dossier failed: %s\n%s", e, traceback.format_exc())
        raise
    dossier_duration = time.time() - t_phase
    log.info("[Phase 3] dossier complete in %.1fs", dossier_duration)

    try:
        _write_json(out_dir / "dossier.json", dossier)
    except Exception as e:
        log.warning("dossier.json dump failed: %s", e)
        _write_json(out_dir / "dossier.json", {"error": str(e)})

    # Surface just the SortedReport for fast comparison
    if dossier.master_sorted is not None:
        _write_json(out_dir / "sorted.json", dossier.master_sorted)
        log.info(
            "sort buckets: known=%d invalid=%d unplaced=%d demotions=%d dropped=%d",
            len(dossier.master_sorted.known),
            len(dossier.master_sorted.invalid),
            len(dossier.master_sorted.unplaced),
            len(dossier.master_sorted.parser_demotions),
            len(dossier.master_sorted.dropped_report_ids),
        )
    else:
        log.warning("master_sorted is None — sort did not produce a report")

    # ---- Phase 4: run_meta ---------------------------------------------
    run_meta = {
        "timestamp":            timestamp,
        "git_sha":              git_sha,
        "git_branch":           git_branch,
        "wandering": {
            "mode":              WANDERING_MODE.value,
            "agent_count":       AGENT_COUNT,
            "model_mix":         list(MODEL_MIX),
            "time_budget_s":     TIME_BUDGET_S,
            "tokens_per_agent":  TOKENS_PER_AGENT,
            "session_token_cap": config.session_token_cap,
        },
        "master_tier": {
            "pipeline_mode":     "sorter",
            "sorter_model":      "anthropic/claude-fable-5",
            "cost_ceiling_usd":  SORT_COST_CAP,
        },
        "durations_seconds": {
            "cushion_compose": round(cushion_duration, 2),
            "wandering":       round(wander_duration, 2),
            "dossier_build":   round(dossier_duration, 2),
            "total":           round(
                cushion_duration + wander_duration + dossier_duration, 2
            ),
        },
        "cohort_integrity_ok": ok,
        "cohort_integrity_problems": problems,
        "session_metrics": {
            "report_count":     session.report_count(),
            "agent_count":      session.agent_count(),
            "tokens_spent":     session.total_tokens_spent,
            "expected_agents":  session.expected_agent_count,
            "agent_errors":     {k: v.get("exc_type", "?")
                                 for k, v in session.agent_errors.items()},
        },
        "sort_metrics": (
            {
                "known":            len(dossier.master_sorted.known),
                "invalid":          len(dossier.master_sorted.invalid),
                "unplaced":         len(dossier.master_sorted.unplaced),
                "parser_demotions": len(dossier.master_sorted.parser_demotions),
                "dropped":          len(dossier.master_sorted.dropped_report_ids),
                "total_cost_usd":   round(dossier.master_sorted.total_cost_usd, 4),
                "truncated_by_budget": dossier.master_sorted.truncated_by_budget,
            }
            if dossier.master_sorted is not None else None
        ),
        "output_dir": str(out_dir),
    }
    _write_json(out_dir / "run_meta.json", run_meta)

    log.info("=" * 70)
    log.info("RUN COMPLETE — artifacts in %s", out_dir)
    log.info("=" * 70)

    return run_meta


if __name__ == "__main__":
    meta = asyncio.run(run())
    print("\nRUN META SUMMARY:")
    print(json.dumps(meta, indent=2, default=_json_default))
