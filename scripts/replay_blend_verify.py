"""
replay_blend_verify.py — Run BLEND VERIFICATION (stage 5) on a blend batch.

Brick 4 validation. Loads a blend-<ts>/blends.json, reconstructs the blends,
and runs the web-verified 4-bin sorter on them: known / adjacent / novel /
flawed. Shows whether the blender's "new" concepts are actually new, only
partially new (adjacent — the 4th bin), or already published.

Output: <blend_dir>/verify-<YYYYMMDD-HHMMSS>/{blend_verification.json, verify_meta.json}.

Usage:
    python scripts/replay_blend_verify.py <blend_dir>
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
from src.wandering.blender import Blend
from src.wandering.blend_verify import verify_blends


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


def _blend_from_dict(d: dict) -> Blend:
    return Blend(
        blend_id=d.get("blend_id", ""),
        source_card_ids=list(d.get("source_card_ids", [])),
        thesis=d.get("thesis", ""),
        mechanism=d.get("mechanism", ""),
        emergent_structure=d.get("emergent_structure", ""),
        advances_cushion=d.get("advances_cushion", ""),
    )


async def run(blend_dir: Path) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = blend_dir / f"verify-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("replay_blend_verify")

    blends_path = blend_dir / "blends.json"
    if not blends_path.exists():
        log.error("No blends.json in %s", blend_dir)
        return {"error": "no_blends_json"}
    blends = [_blend_from_dict(b) for b in json.loads(blends_path.read_text()).get("blends", [])]
    if not blends:
        return {"error": "no_blends"}
    log.info("Loaded %d blends", len(blends))

    run_dir = blend_dir.parent.parent
    problem = _load_problem_text(run_dir)
    cushion = _CushionShim(problem) if problem else None
    sort_model = control_room.SORTER_MODEL
    log.info("Verify model: %s | cushion chars: %d", sort_model, len(problem))

    client = LLMClient(mode=ClientMode.LIVE)
    t0 = time.time()
    report = await verify_blends(
        cushion=cushion, blends=blends, client=client,
        query_model=sort_model, verify_model=sort_model,
    )
    elapsed = time.time() - t0

    (out_dir / "blend_verification.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    ev = report.evidence
    meta = {
        "timestamp":       timestamp,
        "blend_dir":       str(blend_dir),
        "verify_model":    sort_model,
        "blend_count":     report.input_blend_count,
        "bins": {
            "known":    len(report.known),
            "adjacent": len(report.adjacent),
            "novel":    len(report.novel),
            "flawed":   len(report.flawed),
        },
        "evidence": {
            "queries": ev.total_queries if ev else 0,
            "hits":    ev.total_hits if ev else 0,
            "errors":  len(ev.search_errors) if ev else 0,
        },
        "parser_notes":    report.parser_notes,
        "duration_seconds": round(elapsed, 2),
        "llm_cost_usd":    round(report.total_cost_usd, 4),
    }
    (out_dir / "verify_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    log.info("Done in %.1fs | known=%d adjacent=%d novel=%d flawed=%d | $%.4f",
             elapsed, len(report.known), len(report.adjacent), len(report.novel),
             len(report.flawed), report.total_cost_usd)
    for bin_name, items in (("KNOWN", report.known), ("ADJACENT", report.adjacent),
                            ("NOVEL", report.novel), ("FLAWED", report.flawed)):
        for vb in items:
            log.info("  [%s] %s conf=%.2f", bin_name, vb.blend_id, vb.confidence)
            if vb.resemblance:
                log.info("       resembles: %s", vb.resemblance[:140])
            if vb.still_new:
                log.info("       still new: %s", vb.still_new[:140])
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
    print("\nBLEND-VERIFY META:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
