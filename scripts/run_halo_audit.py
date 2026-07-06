"""
run_halo_audit.py — run the HALO AUDITOR (blend-03 Phase 1) over a saved run.

Loads a runs/r-collision/<ts>/ run, reconstructs the cushion + cards + blends,
and audits all three layers for blind spots + slack. OBSERVER ONLY — writes
the blind spots down, acts on nothing. This is the "first, see what blind
spots it finds" step.

Writes audit.json into the run dir and a readable audit markdown to
~/Downloads/constellax_halo_audit_<ts>.md.

Model comes from control_room.AUDITOR_MODEL. LIVE — spends (~$0.02 on
DeepSeek; more if you set the auditor to Sonnet).

Usage:
    PYTHONPATH=. python scripts/run_halo_audit.py [run_dir]
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
from src.llm.client import ClientMode, LLMClient
from src.wandering.articulate import ArticulatedCard
from src.wandering.blender import Blend
from src.wandering.halo_auditor import run_halo_audit
from src.wandering.report import Confidence


def _load(run_dir: Path):
    rec = json.loads((run_dir / "run_record.json").read_text())
    # rich cushion text from cushion_input.json (problem + vision + map)
    cushion = rec.get("cushion", {}).get("problem", "")
    ci_path = run_dir / "cushion_input.json"
    if ci_path.exists():
        ci = json.loads(ci_path.read_text())
        # New format leads QUESTION after PROBLEM. Accepts legacy runs:
        # 'hunches' falls back to 'current_map'; 'question' empty on old runs.
        cushion = "PROBLEM:\n{}\n\nQUESTION:\n{}\n\nVISION:\n{}\n\nHUNCHES:\n{}".format(
            ci.get("problem", ""), ci.get("question", ""), ci.get("vision", ""),
            ci.get("hunches") or ci.get("current_map", ""))

    ms = (rec.get("stage_1_2_cards_and_sort", {}) or {}).get("master_sorted", {}) or {}
    cards = []
    for bn in ("known", "invalid", "unplaced"):
        for it in ms.get(bn, []) or []:
            c = it.get("card", {})
            if c.get("report_id"):
                cards.append(ArticulatedCard(
                    report_id=c["report_id"], spark=c.get("spark", ""),
                    source_shape=c.get("source_shape", ""), bridge=c.get("bridge", ""),
                    use=c.get("use", ""), limit=c.get("limit", ""), confidence=Confidence.MEDIUM))

    blends = []
    for b in (rec.get("stage_3_blends", {}) or {}).get("blends", []) or []:
        blends.append(Blend(blend_id=b.get("blend_id", ""), source_card_ids=b.get("source_card_ids", []),
                            thesis=b.get("thesis", ""), mechanism=b.get("mechanism", ""),
                            emergent_structure=b.get("emergent_structure", "")))
    return cushion, cards, blends


def _md(report) -> str:
    out = ["# Halo Audit — blind spots (observer only)\n",
           f"**Auditor model:** {report.model} · **total cost:** ${report.total_cost_usd:.4f}\n",
           "_Phase 1: these are written down, acted on by nothing. The human judges whether they're worth a Phase-2 commander._\n"]
    for layer, audit in (("Cushion (the question itself)", report.cushion_audit),
                         ("Cards (the wander's coverage)", report.cards_audit),
                         ("Blends (the lanes' holes)", report.blends_audit)):
        if audit is None:
            continue
        out.append(f"\n## {layer}\n")
        if not audit.blind_spots:
            out.append("_(no blind spots returned)_\n")
        for i, b in enumerate(audit.blind_spots, 1):
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(b.severity, "⚪")
            out.append(f"### {i}. {icon} {b.severity.upper()} — {b.blind_spot}\n")
            out.append(f"**Why it matters:** {b.why_it_matters}\n")
            if b.suggested_angle:
                out.append(f"**Suggested angle (for a future commander):** {b.suggested_angle}\n")
    return "\n".join(out)


async def run(run_dir: Path) -> dict:
    cushion, cards, blends = _load(run_dir)
    print(f"Auditing: cushion ({len(cushion)} chars), {len(cards)} cards, {len(blends)} blends")
    print(f"Auditor model: {control_room.AUDITOR_MODEL}")
    client = LLMClient(mode=ClientMode.LIVE)
    report = await run_halo_audit(cushion=cushion, cards=cards, blends=blends,
                                  client=client, model=control_room.AUDITOR_MODEL)

    (run_dir / "audit.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    md_path = Path.home() / "Downloads" / f"constellax_halo_audit_{run_dir.name}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_md(report))

    meta = {
        "run_dir": str(run_dir), "model": report.model,
        "total_cost_usd": round(report.total_cost_usd, 4),
        "blind_spots": {
            "cushion": len(report.cushion_audit.blind_spots) if report.cushion_audit else 0,
            "cards":   len(report.cards_audit.blind_spots) if report.cards_audit else 0,
            "blends":  len(report.blends_audit.blind_spots) if report.blends_audit else 0,
        },
        "report_md": str(md_path),
    }
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
        sorted((REPO_ROOT / "runs" / "r-collision").glob("*/"))[-1]
    asyncio.run(run(rd))
