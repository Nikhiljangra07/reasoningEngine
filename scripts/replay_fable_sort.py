"""
replay_fable_sort.py — Replay the master_sorter on a SAVED run's cards.

Loads the dossier.json from an existing runs/r-fable-sorter-6agents/<ts>/
and re-runs ONLY the master_sort pass on the same cards. Skips the
~$0.70 wander layer entirely.

Use this to iterate on sorter prompt / effort / max_tokens without
re-running the wandering session every time.

Usage:
    python scripts/replay_fable_sort.py <source_run_dir>

Output lands in <source_run_dir>/replay-<YYYYMMDD-HHMMSS>/{sorted.json, replay_meta.json, replay.log}.
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

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

from src.llm.client import LLMClient, ClientMode
from src.wandering.articulate import ArticulatedCard
from src.wandering.master_sorter import master_sort, MasterSortProgress
from src.wandering.report import Confidence


def _card_from_dict(d: dict) -> ArticulatedCard:
    """Reconstruct an ArticulatedCard from a dossier.json card dict.

    Only the fields master_sorter actually consumes are required; the rest
    get harmless defaults.
    """
    conf_raw = d.get("confidence", "LOW")
    try:
        confidence = Confidence(conf_raw) if isinstance(conf_raw, str) else Confidence.LOW
    except ValueError:
        confidence = Confidence.LOW
    return ArticulatedCard(
        report_id    = d.get("report_id", ""),
        spark        = d.get("spark", ""),
        source_shape = d.get("source_shape", ""),
        bridge       = d.get("bridge", ""),
        use          = d.get("use", ""),
        limit        = d.get("limit", ""),
        confidence   = confidence,
        agent_id     = d.get("agent_id", ""),
    )


async def replay(source_dir: Path) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = source_dir / f"replay-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_handler = logging.FileHandler(out_dir / "replay.log")
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s"
    ))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[log_handler, logging.StreamHandler(sys.stdout)],
        force=True,
    )
    log = logging.getLogger("replay_fable_sort")
    log.info("Replay source: %s", source_dir)
    log.info("Replay output: %s", out_dir)

    dossier_path = source_dir / "dossier.json"
    if not dossier_path.exists():
        raise FileNotFoundError(f"dossier.json not in {source_dir}")
    dossier = json.loads(dossier_path.read_text())

    # All cards from the three bands
    card_dicts = (dossier.get("high", []) + dossier.get("medium", []) + dossier.get("low", []))
    cards = [_card_from_dict(d) for d in card_dicts]
    log.info("Loaded %d cards from dossier", len(cards))
    if not cards:
        log.error("No cards in source dossier; aborting")
        return {"error": "no_cards"}

    client = LLMClient(mode=ClientMode.LIVE)
    progress = MasterSortProgress()

    t0 = time.time()
    sorted_report = await master_sort(
        cushion=None,             # replay does not need cushion context
        cards=cards,
        synthesis_map=None,
        client=client,
        progress=progress,
    )
    elapsed = time.time() - t0

    (out_dir / "sorted.json").write_text(
        json.dumps(sorted_report.to_dict(), indent=2, ensure_ascii=False)
    )

    meta = {
        "timestamp":          timestamp,
        "source_run":         str(source_dir),
        "card_count":         len(cards),
        "duration_seconds":   round(elapsed, 2),
        "buckets": {
            "known":            len(sorted_report.known),
            "invalid":          len(sorted_report.invalid),
            "unplaced":         len(sorted_report.unplaced),
            "parser_demotions": len(sorted_report.parser_demotions),
            "dropped":          len(sorted_report.dropped_report_ids),
        },
        "total_cost_usd":     round(sorted_report.total_cost_usd, 4),
        "truncated_by_budget": sorted_report.truncated_by_budget,
        "dropped_report_ids": sorted_report.dropped_report_ids,
        "progress_events":    [e["name"] for e in progress.events],
    }
    (out_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2))

    log.info("Replay complete in %.1fs", elapsed)
    log.info("Buckets: known=%d invalid=%d unplaced=%d demotions=%d dropped=%d",
             len(sorted_report.known), len(sorted_report.invalid),
             len(sorted_report.unplaced), len(sorted_report.parser_demotions),
             len(sorted_report.dropped_report_ids))
    log.info("Cost: $%.4f", sorted_report.total_cost_usd)
    return meta


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1]).resolve()
    if not src.is_dir():
        print(f"Not a directory: {src}")
        sys.exit(2)
    result = asyncio.run(replay(src))
    print("\nREPLAY META:")
    print(json.dumps(result, indent=2))
