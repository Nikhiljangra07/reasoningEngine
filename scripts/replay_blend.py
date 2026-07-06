"""
replay_blend.py — Run the BLENDER on a verified-sort's cards.

Brick 2 validation. Loads a verified-<ts>/sorted.json (cards already binned
known / unplaced, with the web evidence behind them), reconstructs the cards
WITH their bin labels, and runs the Opus 4.8 blender on them. INVALID cards
are excluded — they're dirt.

The point: does the blender BLEND (produce a concept with emergent structure
that's in neither source card) or MERGE (list two cards)? Every blend's
emergent_structure is printed so you can judge.

Output lands in <verified_dir>/blend-<YYYYMMDD-HHMMSS>/{blends.json, blend_meta.json, blend.log}.
Touches nothing existing.

Usage:
    python scripts/replay_blend.py <verified_sort_dir>
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import control_room

from src.llm.client import LLMClient, ClientMode
from src.wandering.articulate import ArticulatedCard
from src.wandering.blender import BlendProgress, blend_cards
from src.wandering.report import Confidence


class _Problem:
    def __init__(self, content): self.content = content
class _RawInput:
    def __init__(self, content): self.problem = _Problem(content)
class _CushionShim:
    def __init__(self, content): self.raw_input = _RawInput(content)


def _load_problem_text(run_dir: Path) -> str:
    for name in ("cushion_input.json", "cushion.json"):
        p = run_dir / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for path in (("problem", "content"), ("raw_input", "problem", "content"), ("problem",)):
            cur = data
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, str) and cur.strip():
                return cur
    return ""


def _card_from_snapshot(snap: dict) -> ArticulatedCard:
    return ArticulatedCard(
        report_id    = snap.get("report_id", ""),
        spark        = snap.get("spark", ""),
        source_shape = snap.get("source_shape", ""),
        bridge       = snap.get("bridge", ""),
        use          = snap.get("use", ""),
        limit        = snap.get("limit", ""),
        confidence   = Confidence.MEDIUM,
        agent_id     = snap.get("agent_id", ""),
    )


def _load_cards_and_bins(sorted_path: Path):
    """Reconstruct cards + report_id->bin from a verified sorted.json.
    Excludes INVALID (dirt)."""
    s = json.loads(sorted_path.read_text())
    cards: list[ArticulatedCard] = []
    bins: dict[str, str] = {}
    for bin_name in ("known", "unplaced"):
        for item in s.get(bin_name, []) or []:
            snap = item.get("card") or {}
            rid = snap.get("report_id")
            if not rid:
                continue
            cards.append(_card_from_snapshot(snap))
            bins[rid] = bin_name
    return cards, bins


async def run(verified_dir: Path) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = verified_dir / f"blend-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    h = logging.FileHandler(out_dir / "blend.log")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[h, logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("replay_blend")

    sorted_path = verified_dir / "sorted.json"
    if not sorted_path.exists():
        log.error("No sorted.json in %s", verified_dir)
        return {"error": "no_sorted_json"}

    cards, bins = _load_cards_and_bins(sorted_path)
    log.info("Loaded %d cards (known=%d unplaced=%d) — invalid excluded",
             len(cards), sum(1 for b in bins.values() if b == "known"),
             sum(1 for b in bins.values() if b == "unplaced"))
    if len(cards) < 2:
        return {"error": "too_few_cards"}

    # cushion problem lives in the grandparent run dir (verified-* is nested)
    run_dir = verified_dir.parent
    problem = _load_problem_text(run_dir)
    cushion = _CushionShim(problem) if problem else None
    model = control_room.BLENDER_MODEL
    log.info("Blender model: %s | cushion problem chars: %d", model, len(problem))

    client = LLMClient(mode=ClientMode.LIVE)
    progress = BlendProgress()

    t0 = time.time()
    batch = await blend_cards(
        cushion=cushion, cards=cards, bins_by_id=bins,
        client=client, progress=progress, model=model,
    )
    elapsed = time.time() - t0

    (out_dir / "blends.json").write_text(json.dumps(batch.to_dict(), indent=2, ensure_ascii=False))

    meta = {
        "timestamp":        timestamp,
        "verified_dir":     str(verified_dir),
        "input_card_count": batch.input_card_count,
        "model":            model,
        "blend_count":      len(batch.blends),
        "parser_notes":     len(batch.parser_notes),
        "duration_seconds": round(elapsed, 2),
        "llm_cost_usd":     round(batch.total_cost_usd, 4),
        "truncated":        batch.truncated_by_budget,
    }
    (out_dir / "blend_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    log.info("Done in %.1fs | %d blends | $%.4f", elapsed, len(batch.blends), batch.total_cost_usd)
    for b in batch.blends:
        log.info("  [%s] cards=%s conf=%.2f", b.blend_id, b.source_card_ids, b.confidence)
        log.info("       thesis: %s", b.thesis[:140])
        log.info("       emergent: %s", b.emergent_structure[:140])
        log.info("       discovery_path: %s", b.selection.discovery_path[:160])
    if batch.parser_notes:
        log.info("  parser notes: %s", json.dumps(batch.parser_notes))
    return meta


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    d = Path(sys.argv[1]).resolve()
    if not d.is_dir():
        print(f"Not a directory: {d}")
        sys.exit(2)
    result = asyncio.run(run(d))
    print("\nBLEND META:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
