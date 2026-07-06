"""
replay_verified_sort.py — Replay the WEB-VERIFIED sorter on a saved run.

Brick 1 validation. Loads a saved run's cards, gathers REAL web evidence
for each (sorter_verify.gather_evidence → Exa/Tavily/DDG), then runs the
verified master_sort against that evidence. The point: a card whose prior
work was published after the model's training cutoff (e.g. FERMAT,
arXiv 2511.14778) should move out of `unplaced` into `known` — because the
sorter can now SEE the paper in the evidence, not just guess from memory.

It does NOT touch the source run's sorted.json or any prior report. Output
lands in a fresh <source_run_dir>/verified-<YYYYMMDD-HHMMSS>/ subdir, and
the meta prints a bin-MOVEMENT diff vs the saved memory-only sort.

Models come from scripts/control_room.py (SORTER_MODEL + the verify query
model). LIVE — spends on one extraction call, N searches, one sort call.

Usage:
    python scripts/replay_verified_sort.py <source_run_dir>
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

import control_room  # the single config surface

from src.llm.client import LLMClient, ClientMode
from src.wandering.articulate import ArticulatedCard
from src.wandering.fetcher import search_chain
from src.wandering.master_sorter import master_sort, MasterSortProgress
from src.wandering.report import Confidence
from src.wandering.sorter_verify import DEFAULT_QUERY_MODEL, gather_evidence


# A minimal cushion stand-in exposing only what the sorter reads:
# cushion.raw_input.problem.content. Reconstructing the full CushionGraph
# from JSON isn't needed just to give the query extractor + sorter the
# problem anchor.
class _Problem:
    def __init__(self, content: str):
        self.content = content


class _RawInput:
    def __init__(self, content: str):
        self.problem = _Problem(content)


class _CushionShim:
    def __init__(self, content: str):
        self.raw_input = _RawInput(content)


def _load_problem_text(source_dir: Path) -> str:
    """Best-effort pull of the cushion problem string from the saved run."""
    for name in ("cushion_input.json", "cushion.json"):
        p = source_dir / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        # Try a few plausible shapes without assuming one.
        for path in (
            ("problem", "content"),
            ("raw_input", "problem", "content"),
            ("problem",),
        ):
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


def _card_from_dict(d: dict) -> ArticulatedCard:
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


def _bin_map_from_sorted(sorted_dict: dict) -> dict[str, str]:
    """report_id -> bin name, from a saved sorted.json."""
    out: dict[str, str] = {}
    for bin_name in ("known", "invalid", "unplaced"):
        for item in sorted_dict.get(bin_name, []) or []:
            rid = (item.get("card") or {}).get("report_id") or item.get("report_id")
            if rid:
                out[str(rid)] = bin_name
    return out


def _bin_map_from_report(report) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in report.known:
        out[item.card.report_id] = "known"
    for item in report.invalid:
        out[item.card.report_id] = "invalid"
    for item in report.unplaced:
        out[item.card.report_id] = "unplaced"
    return out


async def replay(source_dir: Path) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = source_dir / f"verified-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_handler = logging.FileHandler(out_dir / "verified.log")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    logging.basicConfig(level=logging.INFO,
                        handlers=[log_handler, logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("replay_verified_sort")
    log.info("Verified-sort replay source: %s", source_dir)
    log.info("Output: %s", out_dir)

    dossier = json.loads((source_dir / "dossier.json").read_text())
    card_dicts = dossier.get("high", []) + dossier.get("medium", []) + dossier.get("low", [])
    cards = [_card_from_dict(d) for d in card_dicts]
    if not cards:
        log.error("No cards in source dossier; aborting")
        return {"error": "no_cards"}
    log.info("Loaded %d cards", len(cards))

    problem = _load_problem_text(source_dir)
    cushion = _CushionShim(problem) if problem else None
    query_model = DEFAULT_QUERY_MODEL
    sort_model  = control_room.SORTER_MODEL
    log.info("Query model: %s | Sort model: %s", query_model, sort_model)

    client = LLMClient(mode=ClientMode.LIVE)

    # Phase 1+2 — gather real web evidence.
    t0 = time.time()
    ledger = await gather_evidence(
        cushion=cushion, cards=cards, client=client,
        query_model=query_model, search_fn=search_chain,
    )
    log.info("Evidence: %d queries, %d hits, %d search errors, %.0fms",
             ledger.total_queries, ledger.total_hits, len(ledger.search_errors), ledger.elapsed_ms)

    # Phase 3 — verified sort.
    progress = MasterSortProgress()
    report = await master_sort(
        cushion=cushion, cards=cards, synthesis_map=None,
        client=client, progress=progress,
        fable_model=sort_model, web_evidence=ledger,
    )
    elapsed = time.time() - t0

    (out_dir / "sorted.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    (out_dir / "evidence.json").write_text(json.dumps(ledger.to_dict(), indent=2, ensure_ascii=False))

    # Bin-movement diff vs the saved memory-only sort.
    old_path = source_dir / "sorted.json"
    moves: list[dict] = []
    if old_path.exists():
        old_bins = _bin_map_from_sorted(json.loads(old_path.read_text()))
        new_bins = _bin_map_from_report(report)
        for rid, new_bin in new_bins.items():
            old_bin = old_bins.get(rid, "(absent)")
            if old_bin != new_bin:
                moves.append({"report_id": rid, "from": old_bin, "to": new_bin})

    # search cost is provider-side (Exa/Tavily); only LLM cost is tracked here.
    meta = {
        "timestamp":         timestamp,
        "source_run":        str(source_dir),
        "card_count":        len(cards),
        "query_model":       query_model,
        "sort_model":        sort_model,
        "duration_seconds":  round(elapsed, 2),
        "evidence": {
            "total_queries":  ledger.total_queries,
            "total_hits":     ledger.total_hits,
            "search_errors":  len(ledger.search_errors),
            "extraction_ok":  ledger.extraction_ok,
        },
        "buckets": {
            "known":    len(report.known),
            "invalid":  len(report.invalid),
            "unplaced": len(report.unplaced),
            "demotions": len(report.parser_demotions),
            "dropped":  len(report.dropped_report_ids),
        },
        "llm_cost_usd":      round(report.total_cost_usd + ledger.extraction_cost_usd, 4),
        "bin_movements":     moves,
    }
    (out_dir / "verified_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    log.info("Done in %.1fs | known=%d invalid=%d unplaced=%d | %d cards moved bins",
             elapsed, len(report.known), len(report.invalid), len(report.unplaced), len(moves))
    for m in moves:
        log.info("  MOVED %s: %s -> %s", m["report_id"], m["from"], m["to"])
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
    print("\nVERIFIED-SORT META:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
