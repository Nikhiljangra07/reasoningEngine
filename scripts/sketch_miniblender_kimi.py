"""
sketch_miniblender_kimi.py — SMOKE TEST: Kimi K2.6 as the mini-blender edge detector.

Same gold standard, same prompt, same scoring as sketch_miniblender_validate.py
(qwen/qwen3.6-35b-a3b → 8/12 positive recall, 5/12 negative, $0.05). Only the
model changes, so the result is apples-to-apples against the existing mini-blender.

Reuses the validated machinery (SYSTEM prompt, gold-edge construction, scoring)
verbatim — only MODEL + the probe body differ.

Usage: PYTHONPATH=. python scripts/sketch_miniblender_kimi.py runs/r-collision/20260615-212736
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json

from scripts.sketch_miniblender_validate import HEADERS, SYSTEM, URL, _card_text, _parse

MODEL = "moonshotai/kimi-k2.6"


def _probe(client: httpx.Client, a: str, b: str) -> tuple[dict, float]:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Finding A:\n{a}\n\nFinding B:\n{b}"}],
        "temperature": 0.0,
        "max_tokens": 2000,
        "usage": {"include": True},
    }
    last_cost = 0.0
    for attempt in range(2):
        try:
            r = client.post(URL, headers=HEADERS, json=body, timeout=90.0)
            r.raise_for_status()
            d = r.json()
            last_cost = float((d.get("usage") or {}).get("cost", 0.0) or 0.0)
            content = (d["choices"][0]["message"].get("content") or "").strip()
            if content:
                return _parse(content), last_cost
        except Exception as e:
            if attempt == 1:
                return {"relation": f"ERR:{type(e).__name__}", "confidence": 0.0}, last_cost
    return {"relation": "EMPTY", "confidence": 0.0}, last_cost


def main(run_dir: Path) -> None:
    col = json.loads((run_dir / "collision.json").read_text())
    dos = json.loads((run_dir / "dossier.json").read_text())
    cards = (dos.get("high") or []) + (dos.get("medium") or []) + (dos.get("low") or [])
    by_id = {c["report_id"]: c for c in cards}
    blends = col["blends"]["blends"]

    positives = []
    blended = set()
    for b in blends:
        sids = b.get("source_card_ids", [])
        blended.update(sids)
        for a, c in combinations(sids, 2):
            positives.append((a, c))
    outliers = [c["report_id"] for c in cards if c["report_id"] not in blended]
    negatives = [(outliers[i], outliers[i + 1]) for i in range(0, len(outliers) - 1, 2)][:len(positives)]

    print(f"mini-blender = {MODEL}  (smoke test vs qwen/qwen3.6-35b-a3b baseline)")
    print(f"POSITIVES (Opus-blended pairs): {len(positives)}   NEGATIVES (outlier pairs): {len(negatives)}")
    print(f"total probes: {len(positives) + len(negatives)}\n")

    total_cost = 0.0
    with httpx.Client() as client:
        def run(pairs, label, expect):
            nonlocal total_cost
            hits = 0
            rows = []
            for a, c in pairs:
                res, cost = _probe(client, _card_text(by_id[a]), _card_text(by_id[c]))
                total_cost += cost
                rel = res.get("relation", "?")
                conf = res.get("confidence", "?")
                ok = (rel == expect)
                hits += ok
                rows.append(('-'.join(a.split('-')[1:]), '-'.join(c.split('-')[1:]), rel, conf, ok))
            print(f"[{label}] expect '{expect}': {hits}/{len(pairs)} = {100*hits/max(1,len(pairs)):.0f}%")
            for ra, rc, rel, conf, ok in rows:
                print(f"    {('OK' if ok else ' .')} {ra:>11}<->{rc:<11}  {rel:<13} conf={conf}")
            return hits

        pos_hits = run(positives, "POSITIVE", "emergence")
        print()
        neg_hits = run(negatives, "NEGATIVE", "unrelated")

    print("\n=== VERDICT (Kimi K2.6) ===")
    print(f"  POSITIVE recall (Opus edges called emergence):  {pos_hits}/{len(positives)} = "
          f"{100*pos_hits/max(1,len(positives)):.0f}%   <- the STRONG signal")
    print(f"  NEGATIVE specificity (outliers called unrelated): {neg_hits}/{len(negatives)} = "
          f"{100*neg_hits/max(1,len(negatives)):.0f}%")
    print(f"  total cost: ${total_cost:.4f}")
    print("\n  baseline qwen/qwen3.6-35b-a3b: POSITIVE 8/12=67%  NEGATIVE 5/12=42%  $0.0517")


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass a run dir, e.g. runs/r-collision/20260615-212736")
    main(rd)
