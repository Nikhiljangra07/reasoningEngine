"""
sketch_tempo_video.py — TEMPO VIDEO (test harness, NOT wired into the pipeline).

Reveals the 37 cards in ROUNDS (proxy ordering), runs the mini-blender each round
on new candidate pairs, builds the EMERGENCE-RATE curve over rounds, and computes
blend-01's STERILITY doubt meter (rate positive but DECELERATING: r>0, ṙ<0, r̈≤0).

TESTS: does the rate/doubt MACHINERY produce a readable rise-then-fall curve on
real data revealed over time, and does sterility fire sensibly as it declines?

HONEST SCOPE:
  • rounds are a PROXY ordering (cards have no true arrival time) — so this tests the
    MACHINERY on a plausible ordering, not true live motion.
  • per-round checks are BOUNDED (new cards vs the skeleton frontier + within-batch,
    capped) — which is ALSO the real live-governor behavior (new findings attach to the
    growing skeleton; you don't re-check all history) and keeps cost ~$0.3.
  • blend-04's divergence trend Δd is a SEPARATE signal (needs embeddings) — not here.

mini-blender = qwen/qwen3.6-35b-a3b (validated B+). Standalone, touches nothing live.

Usage: PYTHONPATH=. python scripts/sketch_tempo_video.py runs/r-collision/20260615-212736
"""
from __future__ import annotations

import asyncio
import json
import sys
from itertools import combinations
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sketch_miniblender_validate import SYSTEM, URL, HEADERS, MODEL, _card_text, _parse
from sketch_governance_controller import sterility_series

BATCH = 4          # cards revealed per round
CAP_PER_ROUND = 12  # max mini-blender calls per round (bounds cost; also live behavior)
CONCURRENCY = 8


async def _probe(client: httpx.AsyncClient, sem: asyncio.Semaphore, a: str, b: str) -> tuple[str, float]:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Finding A:\n{a}\n\nFinding B:\n{b}"}],
        "temperature": 0.0, "max_tokens": 2000, "usage": {"include": True},
    }
    async with sem:
        for attempt in range(2):
            try:
                r = await client.post(URL, headers=HEADERS, json=body, timeout=90.0)
                r.raise_for_status()
                d = r.json()
                cost = float((d.get("usage") or {}).get("cost", 0.0) or 0.0)
                content = (d["choices"][0]["message"].get("content") or "").strip()
                if content:
                    return _parse(content).get("relation", "?"), cost
            except Exception:
                if attempt == 1:
                    return "ERR", 0.0
    return "EMPTY", 0.0


async def main(run_dir: Path) -> None:
    dos = json.loads((run_dir / "dossier.json").read_text())
    cards = (dos.get("high") or []) + (dos.get("medium") or []) + (dos.get("low") or [])
    by_id = {c["report_id"]: c for c in cards}
    order = [c["report_id"] for c in cards]   # proxy arrival order = dossier order

    edges: set[frozenset] = set()        # emergence edges found so far
    frontier: list[str] = []             # cards with ≥1 emergence edge (recent last)
    revealed: list[str] = []
    checked: set[frozenset] = set()
    rounds = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]

    print(f"TEMPO VIDEO · mini-blender={MODEL} · {len(order)} cards in {len(rounds)} rounds (proxy order)\n")
    r_series: list[int] = []
    total_cost = 0.0

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(CONCURRENCY)
        for ridx, batch in enumerate(rounds, 1):
            # candidate pairs this round: within-batch first, then new×frontier (recent first)
            cands: list[tuple[str, str]] = []
            for a, b in combinations(batch, 2):
                cands.append((a, b))
            for nc in batch:
                for fc in reversed(frontier):
                    cands.append((nc, fc))
            # dedupe + drop already-checked, cap
            seen = set()
            picked = []
            for a, b in cands:
                k = frozenset((a, b))
                if a == b or k in checked or k in seen:
                    continue
                seen.add(k)
                picked.append((a, b))
                if len(picked) >= CAP_PER_ROUND:
                    break

            results = await asyncio.gather(*[
                _probe(client, sem, _card_text(by_id[a]), _card_text(by_id[b])) for a, b in picked])

            new_em = 0
            for (a, b), (rel, cost) in zip(picked, results):
                total_cost += cost
                checked.add(frozenset((a, b)))
                if rel == "emergence":
                    edges.add(frozenset((a, b)))
                    new_em += 1
                    for x in (a, b):
                        if x in frontier:
                            frontier.remove(x)
                        frontier.append(x)
            revealed.extend(batch)
            r_series.append(new_em)
            bar = "█" * new_em
            print(f"  round {ridx:2}: +{len(batch)} cards | checked {len(picked):2} pairs | "
                  f"emergence r={new_em}  {bar}")

    # --- blend-01 doubt meter WITH K-consecutive hysteresis ----------------
    K = 2
    print(f"\n  emergence-rate series r_t = {r_series}")
    series = sterility_series(r_series, k=K)
    print(f"\n  STERILITY with K={K}-consecutive hysteresis (transient dip vs CONFIRMED death):")
    raw_rounds, confirmed_rounds = [], []
    for s in series:
        if s["rddot"] is None:
            continue
        if s["raw_sterile"]:
            raw_rounds.append(s["round"])
        if s["confirmed"]:
            confirmed_rounds.append(s["round"])
        tag = ("CONFIRMED → re-inject" if s["confirmed"]
               else "raw dip (filtered)" if s["raw_sterile"] else "fertile")
        print(f"    round {s['round']:2}: r={s['r']} ṙ={s['rdot']:+d} r̈={s['rddot']:+d} "
              f"streak={s['streak']}  -> {tag}")

    peak = r_series.index(max(r_series)) + 1 if r_series else 0
    print(f"\n  edges found: {len(edges)} | peak rate at round {peak}")
    print(f"  raw sterile rounds:      {raw_rounds or 'none'}")
    print(f"  CONFIRMED (held K={K}):     {confirmed_rounds or 'none'}   <- the actionable re-inject signal")
    print(f"  total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass a run dir, e.g. runs/r-collision/20260615-212736")
    asyncio.run(main(rd))
