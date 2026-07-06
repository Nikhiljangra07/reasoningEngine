"""
sketch_dispatch_live.py — STEP 2: ONE LIVE TEST WAVE (the first real spend).

Proves the dispatch organ end-to-end on the REAL wander:
    open gap (Q2)  →  translator → goal-free lead  →  compose_cushion(problem=lead)
    →  run_wandering_session (ONE small wave)  →  real findings  →  coverage re-score

It does NOT edit runtime.py or the production loop — it calls the real wander
machinery from a standalone harness. Deletable, reversible.

SAFETY (fail-closed at every step):
  1. Refuses to fire unless CONSTELLAX_GOVERNOR=1 (no accidental spend).
  2. The wander anchors on the goal-free LEAD; the question is excluded TWICE —
     the translator strips it, and CushionInput.fields() omits `question` by design.
  3. If the translated lead is not chaos-law clean → ABORT before any wave.
  4. ONE small wave (2 agents, DeepSeek+Haiku). Real cost ~$1-2, surfaced at the end.

Usage: CONSTELLAX_GOVERNOR=1 PYTHONPATH=.:scripts python3 scripts/sketch_dispatch_live.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Progress visibility: surface the wander's per-agent/per-dig logs so we can SEE it
# advance (distinguishes "slow" from "hung"). Standalone — no src/ change.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")
for noisy in ("httpx", "google_genai", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import run_fable_sorter_6agents as base   # for _MODE_MAP
from src.llm.client import ClientMode, LLMClient
from src.wandering.cushion import CushionInput, CushionField, SkipReason
from src.wandering.composer import compose_cushion
from src.wandering.runtime import WanderingConfig, run_wandering_session
from src.wandering.fetcher import web_search_fetcher
from src.wandering.lead_translator import translate_gap, leak_check
from src.wandering.coverage_scorer import parse_required_angles, score_coverage

BATCH1 = REPO_ROOT / "runs/r-collision/20260616-171733"
OPEN_GAP = "how to decide how many agents to commit to a given gap, and when"  # Q2
WAVE_AGENTS = 1                                  # minimal probe — cheap + fast
WAVE_MIX = ("deepseek/deepseek-v4-pro",)
WAVE_BUDGET_S = 5 * 60                            # soft budget (checked between steps)
WAVE_HARD_CEILING_S = 9 * 60                      # hard ceiling — wave CANNOT exceed this


def _findings_text(reports: list) -> str:
    out = []
    for r in reports:
        es = getattr(r, "exploration_summary", "") or ""
        adv = getattr(r, "advancement", "") or ""
        dom = getattr(r, "domain_explored", "?")
        out.append(f"[{dom}] {es} {adv}".strip())
    return "\n\n".join(out)


async def main() -> None:
    if os.environ.get("CONSTELLAX_GOVERNOR", "0") != "1":
        raise SystemExit("REFUSED: set CONSTELLAX_GOVERNOR=1 to fire a live wave (fail-closed).")

    print("=" * 78)
    print("STEP 2 — ONE LIVE TEST WAVE (dispatch organ, end-to-end)")
    print("=" * 78)

    # 1. translate the open gap → goal-free lead, GATED (fail-closed)
    print(f"\n[1] open gap (Q2): {OPEN_GAP}")
    tr = await translate_gap(OPEN_GAP)
    if tr.error:
        raise SystemExit(f"ABORT: translator error: {tr.error}")
    if not tr.clean:
        raise SystemExit(f"ABORT (fail-closed): lead leaked the question → {tr.reasons}")
    # re-verify the gate right here, defense in depth
    clean, reasons = leak_check(tr.lead, OPEN_GAP)
    if not clean:
        raise SystemExit(f"ABORT (fail-closed): re-check failed → {reasons}")
    print(f"[1] goal-free lead (chaos-law CLEAN, {tr.attempts} attempt/s):\n    {tr.lead}")

    # 2. build a cushion whose ANCHOR is the lead; question stays empty (and is
    #    excluded from fields() anyway). No memory enrichment (isolated).
    client = LLMClient(mode=ClientMode.LIVE)
    ci = CushionInput(
        problem=CushionField(name="problem", content=tr.lead),
        context=CushionField(name="context", content="", skip_reason=SkipReason.SKIPPED_AFTER_PROMPT),
        vision=CushionField(name="vision", content="", skip_reason=SkipReason.SKIPPED_AFTER_PROMPT),
        hunches=CushionField(name="hunches", content="", skip_reason=SkipReason.SKIPPED_AFTER_PROMPT),
        question=CushionField(name="question", content=""),   # empty + excluded by design
    )
    print("\n[2] composing cushion from the lead (anchor = territory, no question)…")
    cushion = await compose_cushion(ci, client, session_id="dispatch-test", auto_enrich=False)

    # 3. fire ONE small wave
    config = WanderingConfig(
        mode=base._MODE_MAP["absolute_chaos"], agents=WAVE_AGENTS,
        time_budget_seconds=WAVE_BUDGET_S, tokens_per_agent=15_000,
        model_mix=WAVE_MIX, session_id="dispatch-test",
    )
    print(f"[3] dispatching ONE wave: {WAVE_AGENTS} agent, budget {WAVE_BUDGET_S//60}m, "
          f"hard ceiling {WAVE_HARD_CEILING_S//60}m…")
    t0 = time.time()
    try:
        session = await asyncio.wait_for(
            run_wandering_session(cushion, config, client, fetcher=web_search_fetcher),
            timeout=WAVE_HARD_CEILING_S,
        )
        reports = list(getattr(session, "reports", []) or [])
        print(f"[3] wave returned {len(reports)} findings in {time.time()-t0:.0f}s")
    except asyncio.TimeoutError:
        print(f"[3] HARD CEILING HIT at {WAVE_HARD_CEILING_S//60}m — wave exceeded bound. "
              f"This is the reliability backstop firing (bounded, not a silent hang).")
        return

    # 4. feed findings back to the coverage scorer (the loop closes here)
    angles = parse_required_angles(json.loads((BATCH1 / "cushion_input.json").read_text())["question"])
    fresh = _findings_text(reports)
    print("\n[4] re-scoring coverage on the dispatched wave's findings…")
    res = await score_coverage(angles, fresh)
    if res.error:
        print(f"    coverage error: {res.error}")
    else:
        print(f"    D_t (this wave alone) = {res.d_t}")
        for p in res.per_angle:
            print(f"      Q{p['idx']}: {p['coverage']}")
        q2 = next((p for p in res.per_angle if p["idx"] == 2), None)
        print(f"\n    Q2 (the gap we dispatched for): "
              f"{q2['coverage'] if q2 else '?'} — {(q2 or {}).get('by','')[:80]}")

    # dump findings for inspection
    out = REPO_ROOT / "runs" / "dispatch-test-findings.json"
    out.write_text(json.dumps([{
        "domain": getattr(r, "domain_explored", "?"),
        "exploration_summary": getattr(r, "exploration_summary", ""),
        "advancement": getattr(r, "advancement", ""),
        "what_does_not_map": getattr(r, "what_does_not_map", ""),
    } for r in reports], indent=2))
    print(f"\n[done] findings → {out.name}")
    print("Loop closed live: dispatch(goal-free lead) → real findings → coverage re-score.")


if __name__ == "__main__":
    asyncio.run(main())
