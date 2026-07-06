"""
run_quality_ranker.py — run the FINAL alignment pass over a SAVED run.

Loads a runs/r-collision/<ts>/ run, reconstructs cushion + verified blends +
all halo blind spots + novelty bins, and ranks the blends by advancement
toward the cushion (primary) + gap-coverage (secondary). RANKS, never deletes.

Writes quality.json into the run dir and re-renders the consolidated readable
report (so the Quality Ranking table appears in the one .md).

LIVE — one cheap LLM call (~$0.05 on Sonnet). Model = control_room.RANKER_MODEL.

Usage:
    PYTHONPATH=. python scripts/run_quality_ranker.py [run_dir]
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

import control_room
import render_run_report
from src.llm.client import ClientMode, LLMClient
from src.wandering.blender import Blend
from src.wandering.halo_auditor import BlindSpot
from src.wandering.quality_ranker import rank_blends


def _load(run_dir: Path):
    rec = json.loads((run_dir / "run_record.json").read_text())
    ci_path = run_dir / "cushion_input.json"
    cushion = ""
    if ci_path.exists():
        ci = json.loads(ci_path.read_text())
        # New format leads QUESTION after PROBLEM. Accepts legacy runs:
        # 'hunches' falls back to 'current_map'; 'question' empty on old runs.
        cushion = "PROBLEM:\n{}\n\nQUESTION:\n{}\n\nVISION:\n{}\n\nHUNCHES:\n{}".format(
            ci.get("problem", ""), ci.get("question", ""), ci.get("vision", ""),
            ci.get("hunches") or ci.get("current_map", ""))

    blends = []
    for b in (rec.get("stage_3_blends", {}) or {}).get("blends", []) or []:
        blends.append(Blend(blend_id=b.get("blend_id", ""), source_card_ids=b.get("source_card_ids", []),
                            thesis=b.get("thesis", ""), mechanism=b.get("mechanism", ""),
                            emergent_structure=b.get("emergent_structure", ""),
                            advances_cushion=b.get("advances_cushion", "")))

    novelty = {}
    ver = rec.get("stage_5_verification", {}) or {}
    for bn in ("known", "adjacent", "novel", "flawed"):
        for v in ver.get(bn, []) or []:
            novelty[v.get("blend_id", "")] = bn

    spots = []
    a_path = run_dir / "audit.json"
    if a_path.exists():
        a = json.loads(a_path.read_text())
        for key in ("cushion_audit", "cards_audit", "blends_audit"):
            la = a.get(key) or {}
            for s in la.get("blind_spots", []) or []:
                spots.append(BlindSpot(layer=s.get("layer", ""), blind_spot=s.get("blind_spot", ""),
                                       why_it_matters=s.get("why_it_matters", ""),
                                       severity=s.get("severity", "medium")))
    return cushion, blends, spots, novelty


async def run(run_dir: Path) -> dict:
    cushion, blends, spots, novelty = _load(run_dir)
    print(f"Ranking {len(blends)} blends against cushion + {len(spots)} blind spots "
          f"({control_room.RANKER_MODEL})")
    client = LLMClient(mode=ClientMode.LIVE)
    report = await rank_blends(cushion=cushion, blends=blends, blind_spots=spots,
                               novelty_by_id=novelty, client=client,
                               model=control_room.RANKER_MODEL)
    (run_dir / "quality.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

    # Re-render the consolidated report so the Quality Ranking table appears.
    md = Path.home() / "Downloads" / f"constellax_collision_{run_dir.name}.md"
    try:
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_run_report.render(run_dir))
    except Exception as e:
        print(f"(re-render failed, quality.json still written: {e})")

    print("\nRANKING:")
    for r in report.ranked:
        gaps = ", ".join(r.blind_spots_addressed) or "—"
        print(f"  #{r.rank}  {r.blend_id}  adv={r.advancement:.2f}  [{r.novelty_bin}]  "
              f"gaps={gaps}  new_gap={'yes' if r.opens_new_gap else 'no'}")
        if r.advancement_note:
            print(f"        {r.advancement_note[:110]}")
    print(f"\ncost ${report.total_cost_usd:.4f} · quality.json + re-rendered {md}")
    return report.to_dict()


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
        sorted((REPO_ROOT / "runs" / "r-collision").glob("*/"))[-1]
    asyncio.run(run(rd))
