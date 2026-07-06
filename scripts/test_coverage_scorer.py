"""
test_coverage_scorer.py — validate ORGAN 2 (coverage scorer) on frozen data.

Runs the coverage scorer on BOTH Cushion-3 batches and checks it reproduces the
independent strict hand-grade:
  • batch-1 (20260616-171733): Q1✓ Q2✗ Q3✓ Q4~ Q5✓  → D_t ≈ 0.70
  • batch-2 (20260616-221321): all five covered        → D_t ≈ 1.00
If the automated scorer matches the hand-grade, the load-bearing organ is validated.
Flow-not-judge: it must score PRESENCE, never quality.

Usage: PYTHONPATH=.:scripts python3 scripts/test_coverage_scorer.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

from src.wandering.coverage_scorer import parse_required_angles, score_coverage

BATCHES = {
    "batch-1": ("runs/r-collision/20260616-171733", 0.70, {"Q2": "uncovered", "Q4": "partial"}),
    "batch-2": ("runs/r-collision/20260616-221321", 1.00, {}),
}


def _findings_text(run_dir: Path) -> str:
    c = json.loads((run_dir / "collision.json").read_text())
    bl = c.get("blends", {})
    bl = bl.get("blends", bl) if isinstance(bl, dict) else bl
    lines = []
    for b in (bl if isinstance(bl, list) else []):
        lines.append(f"[{b.get('blend_id')}] {b.get('thesis','')}")
    return "\n\n".join(lines)


async def main() -> None:
    # required angles from the (identical) cushion question
    q = json.loads((REPO_ROOT / BATCHES["batch-1"][0] / "cushion_input.json").read_text())["question"]
    angles = parse_required_angles(q)
    print(f"Parsed {len(angles)} required angles from the cushion:")
    for i, a in enumerate(angles, 1):
        print(f"  Q{i}: {a[:78]}")
    print("=" * 78)

    for name, (rd, expected_dt, expected_open) in BATCHES.items():
        run_dir = REPO_ROOT / rd
        res = await score_coverage(angles, _findings_text(run_dir))
        if res.error:
            print(f"\n[{name}] ERROR: {res.error}")
            continue
        print(f"\n[{name}]  D_t = {res.d_t}   (hand-grade expected ≈ {expected_dt})")
        for p in res.per_angle:
            flag = "" if p["coverage"] == "covered" else "  <-- OPEN"
            print(f"   Q{p['idx']}: {p['coverage']:<9} {flag}  ({str(p['by'])[:60]})")
        # check against hand-grade expectations
        mismatches = []
        for p in res.per_angle:
            exp = expected_open.get(f"Q{p['idx']}")
            if exp and p["coverage"] != exp:
                mismatches.append(f"Q{p['idx']} got {p['coverage']}, hand-grade said {exp}")
        dt_ok = abs(res.d_t - expected_dt) <= 0.12
        print(f"   D_t within ±0.12 of hand-grade: {'YES' if dt_ok else 'NO'}")
        if mismatches:
            print(f"   per-angle mismatches: {mismatches}")
        else:
            print(f"   per-angle: matches hand-grade on the load-bearing angles")


if __name__ == "__main__":
    asyncio.run(main())
