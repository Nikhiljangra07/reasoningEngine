"""
run_formalize.py — the FORMALIZE stage: DeepSeek R1 renders each finished blend
into testable mathematics. Reads a blender output (blends.json), writes
formalize.json + formalize.md beside it.

A back-half stage runner, same shape as run_quality_ranker.py: point it at a
saved run, it ADDS its output — formalizes nothing destructively, never touches
drift/verify/rank. R1 is the JUNIOR formalizer (seat: src/wandering/formalizer.py):
Opus makes the blend, R1 only formalizes it.

LIVE — small spend (~$0.02/blend; R1 reasons verbosely so OUTPUT dominates,
~3 min/blend). Model = control_room.R1_FORMALIZE_MODEL. Needs OPENROUTER_API_KEY.

Usage:
    PYTHONPATH=. python scripts/run_formalize.py [blends.json | run_dir] [--blend blend-02]
    # default: the most recently modified blends.json under runs/
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import control_room
from src.wandering.formalizer import formalize_blends, render_markdown


def _resolve_blends_path(arg: str | None) -> Path:
    """A blends.json, OR a dir containing one (deepest/newest), OR — default —
    the most recently modified blends.json anywhere under runs/."""
    if arg:
        p = Path(arg).resolve()
        if p.is_dir():
            hits = sorted(p.rglob("blends.json"), key=lambda x: x.stat().st_mtime)
            if not hits:
                raise SystemExit(f"no blends.json under {p}")
            return hits[-1]
        return p
    hits = sorted((REPO_ROOT / "runs").rglob("blends.json"), key=lambda x: x.stat().st_mtime)
    if not hits:
        raise SystemExit("no blends.json found under runs/ — pass a path explicitly")
    return hits[-1]


def _has_falsifier(sec: dict) -> bool:
    return bool(re.search(r"falsif\w*\s*:\s*\S",
                          (sec.get("test", "") or "").replace("*", ""), re.I))


def _extract_blends(data) -> list:
    """Accept the shapes the back-half emits: a bare list; a BlendBatch dict
    (blends.json — top-level 'blends' is a list); a CollisionReport dict
    (collision.json — 'blends' is a NESTED BlendBatch); or a run_record
    (stage_3_blends)."""
    if isinstance(data, list):
        return data
    b = data.get("blends")
    if isinstance(b, list):
        return b
    if isinstance(b, dict) and isinstance(b.get("blends"), list):
        return b["blends"]
    s3 = data.get("stage_3_blends")
    if isinstance(s3, dict) and isinstance(s3.get("blends"), list):
        return s3["blends"]
    return []


async def run(blends_path: Path, only: str | None) -> None:
    data = json.loads(blends_path.read_text())
    blends = _extract_blends(data)
    if only:
        blends = [b for b in blends if b.get("blend_id") == only]
    if not blends:
        raise SystemExit(f"no blends to formalize (path={blends_path}, filter={only})")

    print(f"Formalizing {len(blends)} blend(s) · model={control_room.R1_FORMALIZE_MODEL}")
    print(f"source: {blends_path}\n")

    report = await formalize_blends(
        blends,
        model=control_room.R1_FORMALIZE_MODEL,
        on_progress=lambda n, p: print(
            f"  … {p['blend_id']}: {p['formalizable'] or 'FAILED'} (${p['cost']:.4f})"),
    )

    out_dir = blends_path.parent
    (out_dir / "formalize.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    (out_dir / "formalize.md").write_text(render_markdown(report))

    print("\nFORMALIZE:")
    for f in report.formalizations:
        flag = "" if f.ok else "  ⚠ failed"
        print(f"  {f.blend_id}: formalizable={(f.formalizable or '?'):<7} "
              f"conf={str(f.sections.get('confidence', '?'))[:4]:<4} "
              f"falsifier={'yes' if _has_falsifier(f.sections) else 'NO':<3} "
              f"${f.cost_usd:.4f}{flag}")
    print(f"\ntotal ${report.total_cost_usd:.4f} · formalize.json + formalize.md -> {out_dir}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    only = None
    if "--blend" in argv:
        i = argv.index("--blend")
        only = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    asyncio.run(run(_resolve_blends_path(argv[0] if argv else None), only))
