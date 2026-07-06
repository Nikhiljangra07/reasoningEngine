"""
sketch_tempo_liveorder.py — TEMPO VIDEO on TRUE arrival order (no fresh wander).

The proxy tempo video revealed cards in dossier (confidence) order, which front-loads
the rise. session.json's noticeboard records the 37 findings WITH real timestamps — the
actual order agents produced them. This replays them in TRUE arrival order through the
mini-blender, removing the confidence-front-loading confound.

No fresh wander spend — reuses the existing run's real arrival timestamps. Mini-blender
calls only (~$0.2). Reuses the validated probe + the de-twitched doubt meter. Standalone.

Usage: PYTHONPATH=scripts python scripts/sketch_tempo_liveorder.py runs/r-collision/20260615-212736
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

from sketch_tempo_video import _probe, BATCH, CAP_PER_ROUND, CONCURRENCY
from sketch_governance_controller import sterility_series


def _nb_text(e: dict) -> str:
    return (f"[{e.get('domain','?')}] {e.get('summary','')} | principle: {e.get('principle','')} "
            f"| direction: {e.get('direction','')}")[:700]


async def main(run_dir: Path) -> None:
    s = json.loads((run_dir / "session.json").read_text())
    nb = sorted(s["session_state"]["noticeboard"], key=lambda e: e.get("timestamp", 0))  # TRUE order
    items = [(f"nb-{i:02d}", _nb_text(e)) for i, e in enumerate(nb)]
    text_by_id = dict(items)
    order = [i for i, _ in items]

    span = nb[-1].get("timestamp", 0) - nb[0].get("timestamp", 0)
    print(f"TEMPO VIDEO (TRUE arrival order) · {len(order)} findings · mini-blender")
    print(f"arrival span: {span:.0f}s of real wander time\n")

    edges: set[frozenset] = set()
    frontier: list[str] = []
    checked: set[frozenset] = set()
    rounds = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    r_series: list[int] = []
    total_cost = 0.0

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(CONCURRENCY)
        for ridx, batch in enumerate(rounds, 1):
            cands = [(a, b) for a, b in combinations(batch, 2)]
            for nc in batch:
                for fc in reversed(frontier):
                    cands.append((nc, fc))
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
                _probe(client, sem, text_by_id[a], text_by_id[b]) for a, b in picked])
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
            r_series.append(new_em)
            print(f"  round {ridx:2}: +{len(batch)} findings | checked {len(picked):2} | "
                  f"emergence r={new_em}  {'█' * new_em}")

    print(f"\n  emergence-rate series r_t = {r_series}")
    series = sterility_series(r_series, k=2)
    raw = [s["round"] for s in series if s["raw_sterile"]]
    conf = [s["round"] for s in series if s["confirmed"]]
    peak = r_series.index(max(r_series)) + 1 if r_series else 0
    print("\n  STERILITY (K=2 hysteresis):")
    for s in series:
        if s["rddot"] is None:
            continue
        tag = ("CONFIRMED → re-inject" if s["confirmed"]
               else "raw dip (filtered)" if s["raw_sterile"] else "fertile")
        print(f"    round {s['round']:2}: r={s['r']} ṙ={s['rdot']:+d} r̈={s['rddot']:+d} "
              f"streak={s['streak']}  -> {tag}")
    print(f"\n  edges found: {len(edges)} | peak rate at round {peak}")
    print(f"  raw sterile rounds:   {raw or 'none'}")
    print(f"  CONFIRMED (held K=2):  {conf or 'none'}   <- the actionable re-inject signal")
    print(f"  total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass a run dir, e.g. runs/r-collision/20260615-212736")
    asyncio.run(main(rd))
