"""
sketch_miniblender_kimi_fast.py — Kimi K2.6 mini-blender smoke test, CONCURRENT.

Same gold edges / prompt / scoring as sketch_miniblender_validate.py, but async
with a hard per-call timeout so a slow Kimi call (measured ~19s, high variance)
can't hang the whole run. Concurrency bounds wall time to ~2-3 min.

Usage: PYTHONPATH=. python scripts/sketch_miniblender_kimi_fast.py runs/r-collision/20260615-212736
"""
from __future__ import annotations

import asyncio
import json
import sys
from itertools import combinations
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sketch_miniblender_validate import HEADERS, SYSTEM, URL, _card_text, _parse

MODEL = "moonshotai/kimi-k2.6"
PER_CALL_TIMEOUT = 45.0
CONCURRENCY = 6


async def _probe(client, sem, a: str, b: str) -> tuple[dict, float]:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Finding A:\n{a}\n\nFinding B:\n{b}"}],
        "temperature": 0.0, "max_tokens": 2000, "usage": {"include": True},
    }
    async with sem:
        for attempt in range(2):
            try:
                r = await client.post(URL, headers=HEADERS, json=body, timeout=PER_CALL_TIMEOUT)
                r.raise_for_status()
                d = r.json()
                cost = float((d.get("usage") or {}).get("cost", 0.0) or 0.0)
                content = (d["choices"][0]["message"].get("content") or "").strip()
                if content:
                    return _parse(content), cost
            except Exception as e:
                if attempt == 1:
                    return {"relation": f"ERR:{type(e).__name__}", "confidence": 0.0}, 0.0
    return {"relation": "EMPTY", "confidence": 0.0}, 0.0


async def main(run_dir: Path) -> None:
    col = json.loads((run_dir / "collision.json").read_text())
    dos = json.loads((run_dir / "dossier.json").read_text())
    cards = (dos.get("high") or []) + (dos.get("medium") or []) + (dos.get("low") or [])
    by_id = {c["report_id"]: c for c in cards}
    blends = col["blends"]["blends"]

    positives, blended = [], set()
    for b in blends:
        sids = b.get("source_card_ids", [])
        blended.update(sids)
        for a, c in combinations(sids, 2):
            positives.append((a, c))
    outliers = [c["report_id"] for c in cards if c["report_id"] not in blended]
    negatives = [(outliers[i], outliers[i + 1]) for i in range(0, len(outliers) - 1, 2)][:len(positives)]

    print(f"mini-blender = {MODEL}  (concurrent smoke test)", flush=True)
    print(f"POSITIVES: {len(positives)}  NEGATIVES: {len(negatives)}  total probes: {len(positives)+len(negatives)}\n", flush=True)

    total = {"cost": 0.0}

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def run(pairs, label, expect):
            results = await asyncio.gather(*[
                _probe(client, sem, _card_text(by_id[a]), _card_text(by_id[c])) for a, c in pairs])
            hits = 0
            print(f"[{label}] expect '{expect}':", flush=True)
            for (a, c), (res, cost) in zip(pairs, results):
                total["cost"] += cost
                rel = res.get("relation", "?")
                ok = (rel == expect)
                hits += ok
                ra = '-'.join(a.split('-')[1:]); rc = '-'.join(c.split('-')[1:])
                print(f"    {'OK' if ok else ' .'} {ra:>11}<->{rc:<11}  {rel:<14} conf={res.get('confidence','?')}", flush=True)
            print(f"  => {hits}/{len(pairs)} = {100*hits/max(1,len(pairs)):.0f}%\n", flush=True)
            return hits

        pos = await run(positives, "POSITIVE", "emergence")
        neg = await run(negatives, "NEGATIVE", "unrelated")

    print("=== VERDICT (Kimi K2.6) ===", flush=True)
    print(f"  POSITIVE recall:    {pos}/{len(positives)} = {100*pos/max(1,len(positives)):.0f}%   <- strong signal", flush=True)
    print(f"  NEGATIVE specificity: {neg}/{len(negatives)} = {100*neg/max(1,len(negatives)):.0f}%", flush=True)
    print(f"  cost: ${total['cost']:.4f}", flush=True)
    print("  --- baseline qwen/qwen3.6-35b-a3b: POSITIVE 8/12=67%  NEGATIVE 5/12=42%  $0.0517  ~5s/probe ---", flush=True)
    print("  Kimi K2.6 measured latency: ~19s/probe (single timed probe) — 4x slower than qwen.", flush=True)


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass a run dir")
    asyncio.run(main(rd))
