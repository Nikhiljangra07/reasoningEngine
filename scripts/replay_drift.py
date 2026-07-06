"""
replay_drift.py — Run the DRIFT-CHECKER on a blend batch.

Brick 3 validation + DeepSeek-route derisk. Loads a blend-<ts>/blends.json,
reconstructs the blends, and runs the DeepSeek V4 Pro supervisor on them to
judge directional fidelity to the cushion. Proves the new provider lineage
(DeepSeek via OpenRouter) responds, and that the supervisor stays out of the
way on already-anchored blends (doesn't false-flag).

Output: <blend_dir>/drift-<YYYYMMDD-HHMMSS>/{drift.json, drift_meta.json}.

Usage:
    python scripts/replay_drift.py <blend_dir>
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
from src.wandering.drift_checker import DriftProgress, check_drift


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
        advances_cushion=d.get("advances_cushion", ""),
        emergent_structure=d.get("emergent_structure", ""),
    )


async def run(blend_dir: Path) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = blend_dir / f"drift-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("replay_drift")

    blends_path = blend_dir / "blends.json"
    if not blends_path.exists():
        log.error("No blends.json in %s", blend_dir)
        return {"error": "no_blends_json"}
    batch = json.loads(blends_path.read_text())
    blends = [_blend_from_dict(b) for b in batch.get("blends", [])]
    if not blends:
        return {"error": "no_blends"}
    log.info("Loaded %d blends", len(blends))

    # cushion lives in the run dir: blend-* nested in verified-* nested in run dir
    run_dir = blend_dir.parent.parent
    problem = _load_problem_text(run_dir)
    cushion = _CushionShim(problem) if problem else None
    model = control_room.DRIFT_CHECKER_MODEL
    log.info("Drift-checker model: %s | cushion chars: %d", model, len(problem))

    client = LLMClient(mode=ClientMode.LIVE)
    progress = DriftProgress()

    t0 = time.time()
    report = await check_drift(cushion=cushion, blends=blends, client=client, progress=progress, model=model)
    elapsed = time.time() - t0

    (out_dir / "drift.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    meta = {
        "timestamp":       timestamp,
        "blend_dir":       str(blend_dir),
        "model":           model,
        "blend_count":     len(blends),
        "on_course":       len(report.on_course_ids),
        "drifting":        len(report.drifting_ids),
        "drifting_ids":    report.drifting_ids,
        "duration_seconds": round(elapsed, 2),
        "llm_cost_usd":    round(report.total_cost_usd, 4),
        "parser_notes":    report.parser_notes,
    }
    (out_dir / "drift_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    log.info("Done in %.1fs | on_course=%d drifting=%d | $%.4f",
             elapsed, len(report.on_course_ids), len(report.drifting_ids), report.total_cost_usd)
    for v in report.verdicts:
        tag = "ON-COURSE" if v.on_course else "DRIFT"
        log.info("  [%s] %s resonance=%.2f %s", v.blend_id, tag, v.resonance,
                 ("→ " + v.redirect) if not v.on_course else "")
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
    print("\nDRIFT META:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
