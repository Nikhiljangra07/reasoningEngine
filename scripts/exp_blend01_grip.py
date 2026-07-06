"""
exp_blend01_grip.py — falsifiable test of blend-01 (counterfactual-step grip).

blend-01's premise: at a wander step, the divergence between the anchored
(cushion-in-context) read and the null (no-cushion) read is "the anchor's
grip" — high grip = anchor-driven advancement, low grip = generic retrieval
the agent would have done anyway.

This is a CAD-style measurement (Shi 2023: contrast with-context vs
without-context) applied per CARD on a saved collision run. For each card we
produce an anchored read and a null read of the SAME source, embed both, and
take grip = cosine distance. Same source text in both reads; only the cushion
presence differs — so the divergence isolates the cushion's effect.

Two falsifiable predictions:
  (P1) generic-OVERLAP cards (same prior_work matched by >=2 cards) score
       LOWER grip than rare cards  -> grip flags degeneracy.
  (P2) cards that FED a NOVEL blend score HIGHER grip than the rest
       -> grip predicts advancement-grade material (the quality-info insight).

Read-only on the pipeline. Spends ~$0.25 (Sonnet reads + embeddings).
Writes results into the run dir; prints a verdict.

Usage:
    PYTHONPATH=. python scripts/exp_blend01_grip.py [run_dir]
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import control_room
from src.llm.client import ClientMode, LLMClient

READ_MODEL = control_room.WANDER_MODEL        # match the wander's eyes (Sonnet)
CONCURRENCY = 5


def _cos_dist(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - dot / (na * nb)


def _load(run_dir: Path):
    rec = json.loads((run_dir / "run_record.json").read_text())
    cushion = rec.get("cushion", {}).get("problem", "")
    ms = (rec.get("stage_1_2_cards_and_sort", {}) or {}).get("master_sorted", {}) or {}
    cards = []
    for it in ms.get("known", []) or []:
        c = it.get("card", {})
        cards.append({
            "report_id":  c.get("report_id", ""),
            "source":     (c.get("source_shape", "") + " — " + c.get("spark", "")).strip(" —"),
            "prior_work": it.get("prior_work_name", ""),
        })
    # overlap clusters: prior_work matched by >=2 cards
    from collections import Counter
    pw = Counter(c["prior_work"] for c in cards if c["prior_work"])
    overlap_pw = {k for k, n in pw.items() if n >= 2}
    for c in cards:
        c["overlap"] = c["prior_work"] in overlap_pw
    # which cards fed a NOVEL blend
    novel_feeders = set()
    for t in rec.get("trace", []):
        if t.get("novelty_bin") == "novel":
            novel_feeders.update(t.get("source_card_ids", []))
    for c in cards:
        c["fed_novel"] = c["report_id"] in novel_feeders
    return cushion, cards


async def _grip(client, cushion, card, sem):
    anchored_sys = "You are a research scout. Be concise: 2-3 sentences only."
    anchored = (f"PROBLEM you are advancing:\n{cushion[:900]}\n\n"
                f"A wandering agent found this source:\n{card['source']}\n\n"
                f"In 2-3 sentences: what here is most valuable for advancing the PROBLEM, "
                f"and what would you explore next from here?")
    null = (f"A researcher found this source:\n{card['source']}\n\n"
            f"In 2-3 sentences: what is most interesting here, and what would you "
            f"explore next from here?")
    async with sem:
        ra = await client.call(system_prompt=anchored_sys, user_message=anchored,
                               domain="exp", concept="anchored_read", model=READ_MODEL,
                               max_tokens=300, temperature=0.3)
        rn = await client.call(system_prompt=anchored_sys, user_message=null,
                               domain="exp", concept="null_read", model=READ_MODEL,
                               max_tokens=300, temperature=0.3)
        ea = await client.embed(ra.content or " ")
        en = await client.embed(rn.content or " ")
    return _cos_dist(ea, en)


async def run(run_dir: Path) -> dict:
    cushion, cards = _load(run_dir)
    if not cards:
        print("no cards"); return {}
    client = LLMClient(mode=ClientMode.LIVE)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    grips = await asyncio.gather(*[_grip(client, cushion, c, sem) for c in cards])
    for c, g in zip(cards, grips):
        c["grip"] = round(g, 4)
    elapsed = time.time() - t0

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    overlap = [c["grip"] for c in cards if c["overlap"]]
    rare    = [c["grip"] for c in cards if not c["overlap"]]
    feeders = [c["grip"] for c in cards if c["fed_novel"]]
    others  = [c["grip"] for c in cards if not c["fed_novel"]]

    cost = client.get_total_cost_estimate()
    summary = {
        "run_dir": str(run_dir), "cards": len(cards), "read_model": READ_MODEL,
        "elapsed_s": round(elapsed, 1), "cost_usd": round(cost, 4),
        "grip_overall_mean": round(mean([c["grip"] for c in cards]), 4),
        "P1_overlap_vs_rare": {
            "overlap_n": len(overlap), "overlap_mean_grip": round(mean(overlap), 4),
            "rare_n": len(rare), "rare_mean_grip": round(mean(rare), 4),
            "prediction_holds": mean(overlap) < mean(rare) if overlap and rare else None,
        },
        "P2_novelfeeders_vs_rest": {
            "feeders_n": len(feeders), "feeders_mean_grip": round(mean(feeders), 4),
            "rest_n": len(others), "rest_mean_grip": round(mean(others), 4),
            "prediction_holds": mean(feeders) > mean(others) if feeders and others else None,
        },
        "cards_sorted_by_grip": sorted(
            [{"report_id": c["report_id"], "grip": c["grip"], "overlap": c["overlap"],
              "fed_novel": c["fed_novel"], "prior_work": c["prior_work"][:45]} for c in cards],
            key=lambda x: x["grip"]),
    }
    out = run_dir / "exp_blend01_grip.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "cards_sorted_by_grip"}, indent=2))
    print(f"\nfull results: {out}")
    return summary


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
        sorted((REPO_ROOT / "runs" / "r-collision").glob("*/"))[-1]
    asyncio.run(run(rd))
