"""
sketch_miniblender_validate.py — VALIDATE the live mini-blender (Qwen) against
the blender's gold-standard edges. The prerequisite test: everything downstream
(the tempo video, the live governor) rides on this edge detector being trustworthy.

Gold standard = Opus's blends (run 20260615-212736):
  POSITIVES = the 12 within-blend card pairs  (Opus says these cross-connect → expect "emergence")
  NEGATIVES = ~12 outlier pairs Opus blended with nothing (expect "unrelated")

The mini-blender = qwen/qwen3.6-35b-a3b via OpenRouter (open weights, API-served,
matches the pipeline's R1/V4-Pro pattern). ~24 calls, cents. Standalone, touches
nothing live.

Honest read of "agreement": POSITIVE recall (Opus-blended pairs called emergence) is
the STRONG signal. A NEGATIVE called emergence is SOFT — Opus only made 4 blends, so a
real connection it skipped could legitimately read emergence. So weigh positive recall.

Usage: PYTHONPATH=. python scripts/sketch_miniblender_validate.py runs/r-collision/20260615-212736
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

MODEL = "qwen/qwen3.6-35b-a3b"
URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
           "Content-Type": "application/json"}

SYSTEM = (
    "You are a STRICT edge detector in a reasoning engine. Given two research findings, "
    "classify their structural relation. Be CONSERVATIVE — most random pairs are unrelated.\n"
    '- "emergence": ONLY if you can NAME a specific shared deep mechanism or tension that '
    "bridges them across different surfaces — a genuine cross-connection that creates something "
    "neither states alone. The bar is HIGH; if you cannot name the precise shared structure, do not use this.\n"
    '- "reinforcement": they restate the SAME idea — near-duplicate, same point.\n'
    '- "unrelated": no specific structural connection. THIS IS THE DEFAULT. If the link is '
    "vague, generic, or merely topical (same broad subject), answer unrelated.\n"
    "Judge STRUCTURE, not topic overlap. When unsure, answer unrelated.\n"
    'Output ONLY compact JSON: {"relation":"emergence|reinforcement|unrelated","confidence":0.0-1.0}'
)


def _card_text(c: dict) -> str:
    dom = c.get("domain", "") or "?"
    return (f"[{dom}] {c.get('spark','')} | bridge: {c.get('bridge','')} "
            f"| use: {c.get('use','')}")[:700]


def _parse(txt: str) -> dict:
    s = txt.strip()
    if "{" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    try:
        return json.loads(s)
    except Exception:
        return {"relation": "PARSE_FAIL", "confidence": 0.0}


def _probe(client: httpx.Client, a: str, b: str) -> tuple[dict, float]:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Finding A:\n{a}\n\nFinding B:\n{b}"}],
        "temperature": 0.0,
        "max_tokens": 2000,   # qwen3.6-35b-a3b is a THINKING model — generous room for reasoning + the JSON
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

    # POSITIVES: within-blend pairs
    positives = []
    blended = set()
    for b in blends:
        sids = b.get("source_card_ids", [])
        blended.update(sids)
        for a, c in combinations(sids, 2):
            positives.append((a, c))
    # NEGATIVES: outlier pairs (cards Opus blended with nothing), deterministic
    outliers = [c["report_id"] for c in cards if c["report_id"] not in blended]
    negatives = [(outliers[i], outliers[i + 1]) for i in range(0, len(outliers) - 1, 2)][:len(positives)]

    print(f"mini-blender = {MODEL}")
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
                print(f"    {('✓' if ok else '·')} {ra:>11}↔{rc:<11}  {rel:<13} conf={conf}")
            return hits

        pos_hits = run(positives, "POSITIVE", "emergence")
        print()
        neg_hits = run(negatives, "NEGATIVE", "unrelated")

    print("\n=== VERDICT ===")
    print(f"  POSITIVE recall (Opus edges called emergence):  {pos_hits}/{len(positives)} = "
          f"{100*pos_hits/max(1,len(positives)):.0f}%   <- the STRONG signal")
    print(f"  NEGATIVE specificity (outliers called unrelated): {neg_hits}/{len(negatives)} = "
          f"{100*neg_hits/max(1,len(negatives)):.0f}%   <- soft (Opus skipped real links too)")
    print(f"  total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass a run dir, e.g. runs/r-collision/20260615-212736")
    main(rd)
