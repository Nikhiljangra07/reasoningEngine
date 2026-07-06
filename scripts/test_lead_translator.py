"""
test_lead_translator.py — validate the CHAOS-LAW organ two ways:

  PART A (free, no LLM): adversarial test of the leak-test GATE itself. Feed it known
    LEAKING leads (verbatim question, interrogative, goal-imperative) and known-CLEAN
    leads. The gate MUST flag every leak and pass every clean lead. If the gate is
    blind, nothing downstream is safe — so this is tested first.

  PART B (~$0.02): translate all 5 real Cushion-3 sub-questions into goal-free leads
    and confirm every one passes the gate — i.e. the live wander would receive only
    question-free territory.

Usage: PYTHONPATH=.:scripts python3 scripts/test_lead_translator.py
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

from src.wandering.lead_translator import translate_gap, leak_check
from src.wandering.coverage_scorer import parse_required_angles

BATCH1 = REPO_ROOT / "runs/r-collision/20260616-171733"

# A real gap to test the gate against.
GAP = "how to decide how many agents to commit to a given gap, and when"

# (label, candidate lead, must_be_clean)
GATE_CASES = [
    ("verbatim question",      GAP, False),
    ("interrogative restate",  "Consider how many agents to commit and when to commit them.", False),
    ("goal-imperative",        "Find the rule that determines the number of agents per gap.", False),
    ("question mark",          "Territory of effort allocation. How much to commit?", False),
    ("clean territory lead",   "Systems that distribute a bounded pool of effort across competing "
                               "sites under diminishing returns — foragers across patches, schedulers "
                               "across jobs, capital across positions — where the distribution shifts "
                               "as each site's yield and signs of exhaustion change.", True),
]


def part_a() -> bool:
    print("=" * 78)
    print("PART A — adversarial test of the leak-test GATE (no LLM)")
    print("=" * 78)
    ok = True
    for label, lead, must_clean in GATE_CASES:
        clean, reasons = leak_check(lead, GAP)
        verdict = "PASS-clean" if clean else f"FLAGGED ({'; '.join(reasons)})"
        correct = (clean == must_clean)
        ok = ok and correct
        mark = "✓" if correct else "✗ GATE WRONG"
        print(f"  {mark}  [{label:22}] expected {'clean' if must_clean else 'leak'} → {verdict}")
    print(f"\n  GATE {'VALID — catches every leak, passes clean' if ok else 'BROKEN — do not trust'}")
    return ok


async def part_b() -> None:
    print("\n" + "=" * 78)
    print("PART B — translate the 5 real gaps; every lead must pass the gate")
    print("=" * 78)
    q = json.loads((BATCH1 / "cushion_input.json").read_text())["question"]
    angles = parse_required_angles(q)
    all_clean = True
    for i, angle in enumerate(angles, 1):
        res = await translate_gap(angle)
        if res.error:
            print(f"\n  Q{i}: ERROR {res.error}"); all_clean = False; continue
        all_clean = all_clean and res.clean
        mark = "✓ CLEAN" if res.clean else f"✗ LEAK ({'; '.join(res.reasons)})"
        print(f"\n  Q{i} gap: {angle[:70]}")
        print(f"     lead ({res.attempts} attempt/s): {res.lead[:300]}")
        print(f"     chaos-law: {mark}")
    print("\n" + "=" * 78)
    print(f"ALL 5 LEADS CHAOS-LAW CLEAN: {'YES — safe to dispatch' if all_clean else 'NO — fail-closed, do not dispatch'}")


async def main() -> None:
    gate_ok = part_a()
    if not gate_ok:
        print("\nABORT: gate is broken; not translating real gaps.")
        return
    await part_b()


if __name__ == "__main__":
    asyncio.run(main())
