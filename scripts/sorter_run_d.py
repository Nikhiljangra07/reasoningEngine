"""
sorter_run_d.py — Run the D variant: Opus 4.8 + TIGHTENED prompt.

D uses the CURRENT master_sorter._DOCTRINE_PREAMBLE (commit 1097b92,
the surface-match + factual-sweep + confidence-rubric tightening).
Same 9 cards, same model (Opus 4.8), same effort/max_tokens as the
A/B/C compare so the four-way is apples-to-apples.

Writes to the SAME abc-<timestamp> dir as the A/B/C run if given, or a
fresh d-<timestamp> dir. Never touches the Fable 5 replay.

Usage:
    python scripts/sorter_run_d.py <source_run_dir>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import anthropic

# Reuse the A/B/C helpers so the payload + parse logic is identical.
from scripts.sorter_ab_compare import (
    OPUS_MODEL,
    MAX_TOKENS,
    EFFORT,
    _build_payload,
    _load_cards,
    _call_opus,
    _parse_to_buckets,
)


async def run(source_dir: Path) -> dict:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = source_dir / f"d-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = _load_cards(source_dir)
    print(f"Loaded {len(cards)} cards from {source_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Model: {OPUS_MODEL}  effort: {EFFORT}  max_tokens: {MAX_TOKENS}")

    # D uses the CURRENT (tightened) doctrine preamble from HEAD.
    from src.wandering.master_sorter import _DOCTRINE_PREAMBLE as TIGHTENED_PREAMBLE
    print(f"Tightened preamble: {len(TIGHTENED_PREAMBLE)} chars")

    api_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n[D] Opus 4.8 + TIGHTENED prompt — binding surface-match + factual-sweep…")
    payload = _build_payload(
        cards,
        why_unplaced_form=(
            "<REQUIRED, single neutral technical clause naming the "
            "match-impossibility; no speculation about novelty, value, "
            "plausibility, or potential>"
        ),
    )
    text_d, meta_d = await _call_opus(api_client, TIGHTENED_PREAMBLE, payload)
    parsed_d = _parse_to_buckets(text_d)
    (out_dir / "opus_tightened.json").write_text(json.dumps({
        "meta":   meta_d,
        "parsed": parsed_d,
        "raw":    text_d,
    }, indent=2, ensure_ascii=False))

    print(f"  buckets: known={parsed_d.get('known_count','ERR')} "
          f"invalid={parsed_d.get('invalid_count','ERR')} "
          f"unplaced={parsed_d.get('unplaced_count','ERR')}")
    print(f"  cost: ${meta_d['cost_usd']}  time: {meta_d['elapsed_ms']/1000:.1f}s  "
          f"blocks: {meta_d['blocks']}")

    summary = {
        "timestamp": ts,
        "variant":   "D_opus_tightened",
        "model":     OPUS_MODEL,
        "effort":    EFFORT,
        "buckets": {
            "known":    parsed_d.get("known_count", "ERR"),
            "invalid":  parsed_d.get("invalid_count", "ERR"),
            "unplaced": parsed_d.get("unplaced_count", "ERR"),
        },
        "cost": meta_d["cost_usd"],
        "ms":   meta_d["elapsed_ms"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1]).resolve()
    if not src.is_dir():
        print(f"Not a directory: {src}")
        sys.exit(2)
    result = asyncio.run(run(src))
    print("\n=== D SUMMARY ===")
    print(json.dumps(result, indent=2))
