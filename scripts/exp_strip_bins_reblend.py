"""
exp_strip_bins_reblend.py — SINGLE-VARIABLE TEST: does Opus actually USE the
sort bins, or does the cushion + tension dominate and the known/unplaced/invalid
labels just sit in the prompt unused?

Re-blends THIS run's EXACT 37 cards with the sorter's distribution STRIPPED:
every card shown to Opus as "unsorted", and ZERO invalid-exclusion. Then runs
the SAME drift-check + blend-verify downstream, so the novelty verdict
(known/adjacent/novel/flawed) is directly comparable to the binned original.

Reuses the FROZEN cards from dossier.json — no re-wander, no re-articulate, no
re-sort — so the ONLY variable vs the original collision is: bins removed.

Throwaway experiment. Writes to <run_dir>/_strip_bins_test/ ONLY. Never touches
the original run's files.

Usage:
    PYTHONPATH=. python scripts/exp_strip_bins_reblend.py runs/r-collision/20260615-212736
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import control_room
import run_fable_sorter_6agents as base  # reuse the proven _write_json serializer
from src.llm.client import ClientMode, LLMClient
from src.wandering.fetcher import search_chain
from src.wandering.master_sorter import SortedReport
from src.wandering.collision_pipeline import run_collision_pipeline, CollisionProgress


# --- card shim: EXACTLY the 7 attrs the blend path reads -------------------
# CardSnapshot.from_card + _build_blend_payload touch only:
#   report_id, agent_id, spark, bridge, source_shape, use, limit.
# Nothing else is read anywhere downstream (traced 2026-06-15).
def _card(d: dict) -> SimpleNamespace:
    return SimpleNamespace(
        report_id    = d["report_id"],
        agent_id     = d.get("agent_id", "") or "",
        spark        = d["spark"],
        bridge       = d["bridge"],
        source_shape = d["source_shape"],
        use          = d["use"],
        limit        = d["limit"],
    )


def _cushion(problem: str, question: str) -> SimpleNamespace:
    # blend reads .raw_input.problem.content; drift reads problem + question.
    return SimpleNamespace(raw_input=SimpleNamespace(
        problem  = SimpleNamespace(content=problem),
        question = SimpleNamespace(content=question),
    ))


async def main(run_dir: Path, label: str = "") -> None:
    dossier = json.loads((run_dir / "dossier.json").read_text())
    cards_raw = ((dossier.get("high") or [])
                 + (dossier.get("medium") or [])
                 + (dossier.get("low") or []))      # == dossier.all_cards() order
    cards = [_card(c) for c in cards_raw]

    ci = json.loads((run_dir / "cushion_input.json").read_text())
    cushion = _cushion(ci["problem"], ci["question"])

    print(f"STRIP-BINS RE-BLEND · {len(cards)} cards (FULL stack, no bins, no invalid filter)")
    print(f"blender={control_room.BLENDER_MODEL}  drift={control_room.DRIFT_CHECKER_MODEL}  "
          f"verify={control_room.SORTER_MODEL}")
    print(f"source: {run_dir}\n")

    client = LLMClient(mode=ClientMode.LIVE)
    t0 = time.time()
    report = await run_collision_pipeline(
        cushion=cushion,
        cards=cards,
        sorted_report=SortedReport(),   # <<< THE ONLY CHANGE: empty -> bins {}, invalid set()
        client=client,
        blender_model=control_room.BLENDER_MODEL,
        drift_model=control_room.DRIFT_CHECKER_MODEL,
        query_model=control_room.SORTER_MODEL,
        verify_model=control_room.SORTER_MODEL,
        search_fn=search_chain,
        progress=CollisionProgress(),
    )
    elapsed = time.time() - t0

    out_dir = run_dir / "_strip_bins_test"
    out_dir.mkdir(exist_ok=True)
    fname = f"collision_nobins{('_' + label) if label else ''}.json"
    base._write_json(out_dir / fname, report)

    v = report.verification
    blends = report.blends.blends if report.blends else []
    print("\n=== STRIP-BINS RESULT (no sorter) ===")
    print(f"blends={len(blends)}  quarantined={len(report.quarantined_blend_ids)}  "
          f"| known={len(v.known) if v else 0} adjacent={len(v.adjacent) if v else 0} "
          f"novel={len(v.novel) if v else 0} flawed={len(v.flawed) if v else 0}")
    print(f"cost=${report.total_cost_usd:.4f}  elapsed={elapsed:.1f}s")
    print(f"stage_costs={report.stage_costs}")
    for b in blends:
        q = "  [QUARANTINED]" if b.blend_id in report.quarantined_blend_ids else ""
        print(f"  {b.blend_id}: {b.thesis[:120]}{q}")
    print(f"\nwrote -> {out_dir / fname}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    label = ""
    if "--label" in argv:
        i = argv.index("--label")
        label = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    rd = Path(argv[0]).resolve() if argv else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass the run dir, e.g. runs/r-collision/20260615-212736")
    asyncio.run(main(rd, label))
