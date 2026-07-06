"""
reblend_run.py — RE-RUN THE BLENDER on an already-completed autonomous run.

Why this exists: the 2026-06-18 live run (run-20260618-044258) wandered 4 cycles
and produced 218 cards (raw-card D_t=0.8), but the blender (master_synthesize)
emitted ZERO fusions — it tripped the $4 cost ceiling one Opus call into R2
(R1_draft alone cost $3.55 with 218 cards). The full ArticulatedCard objects were
in-memory only; what persisted to disk is the 3-field bridge subset
(report_id, source_shape, bridge) in cycle-4/cycle.json (cumulative → all 218).

This script reconstructs ArticulatedCards from those bridges (spark/use/limit are
unrecoverable → left empty; source_shape + bridge carry the substance), dedups the
shepherd-flagged restatements, and re-runs the GOAL-AWARE blender → coverage → R1
with a sane cost ceiling and headroom. NO re-wander — the $24 wander is reused.

Mirrors the pipeline's blender + R1 path exactly (run_autonomous_pipeline.py).

Env:
  REBLEND_RUN_DIR   (required) path to the completed run dir
  REBLEND_CAP       blender cost ceiling USD (default 18)
  REBLEND_CARD_CAP  max cards fed to blender after dedup (default 120)
  CONSTELLAX_CUSHION  which cushion built the run (default 6)
  AUTON_COVERAGE_MODEL  cross-family coverage judge (default google/gemini-2.5-flash)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")
for noisy in ("httpx", "google_genai", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

os.environ.setdefault("CONSTELLAX_CUSHION", "6")

import run_fable_sorter_6agents as base
from src.llm.client import ClientMode, LLMClient
from src.wandering.composer import compose_cushion
from src.wandering.articulate import ArticulatedCard, Confidence
from src.wandering.coverage_scorer import parse_required_angles, score_coverage
from src.wandering.master_synthesizer import master_synthesize
from src.wandering.formalizer import formalize_blends, render_markdown

RUN_DIR = Path(os.environ["REBLEND_RUN_DIR"])
CEIL = float(os.environ.get("REBLEND_CAP", "18"))
CARD_CAP = int(os.environ.get("REBLEND_CARD_CAP", "120"))
COVERAGE_MODEL = os.environ.get("AUTON_COVERAGE_MODEL", "google/gemini-2.5-flash")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def _p(s: str = "") -> None:
    print(s, flush=True)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()[:120]


def _load_cards() -> list[ArticulatedCard]:
    # Preferred path: full cards persisted by the pipeline (cards.json, added after
    # the 2026-06-18 run). These carry ALL fields (spark/use/limit/domain/confidence/
    # citations) — a complete, lossless re-blend.
    full = RUN_DIR / "cards.json"
    if full.exists():
        rows = json.load(open(full))
        seen: set[str] = set()
        cards: list[ArticulatedCard] = []
        dropped = 0
        for r in rows:
            key = _norm(r.get("source_shape", ""))
            if not key or key in seen:
                dropped += 1
                continue
            seen.add(key)
            conf = r.get("confidence", "medium")
            try:
                conf_enum = Confidence(conf)
            except Exception:
                conf_enum = Confidence.MEDIUM
            cards.append(ArticulatedCard(
                report_id=r.get("report_id", ""), spark=r.get("spark", ""),
                source_shape=r.get("source_shape", ""), bridge=r.get("bridge", ""),
                use=r.get("use", ""), limit=r.get("limit", ""), confidence=conf_enum,
                confidence_detail=r.get("confidence_detail", ""),
                agent_id=r.get("agent_id", ""), domain=r.get("domain", ""),
                citations=list(r.get("citations", []) or []),
                match_strength=float(r.get("match_strength", 0.0) or 0.0)))
        _p(f"  loaded {len(rows)} FULL cards from cards.json → {len(cards)} unique "
           f"({dropped} dropped as dup/empty); feeding top {min(len(cards), CARD_CAP)}")
        return cards[:CARD_CAP]

    # Fallback (pre-cards.json runs, e.g. run-20260618-044258): reconstruct from the
    # 3-field bridges. cycle-4 card_bridges is CUMULATIVE → holds every card.
    cyc4 = RUN_DIR / "cycle-4" / "cycle.json"
    if not cyc4.exists():
        # fall back to the highest-numbered cycle present
        cands = sorted(RUN_DIR.glob("cycle-*/cycle.json"))
        cyc4 = cands[-1]
    bridges = json.load(open(cyc4))["card_bridges"]
    seen: set[str] = set()
    cards: list[ArticulatedCard] = []
    dropped = 0
    for b in bridges:
        rid = b.get("report_id", "")
        ss = b.get("source_shape", "")
        br = b.get("bridge", "")
        key = _norm(ss)
        if not key or key in seen:        # drop empties + shepherd-flagged restatements
            dropped += 1
            continue
        seen.add(key)
        parts = rid.split("-")
        agent_id = parts[1] if len(parts) > 1 else ""
        cards.append(ArticulatedCard(
            report_id=rid, spark="", source_shape=ss, bridge=br,
            use="", limit="", confidence=Confidence.MEDIUM,
            confidence_detail="", agent_id=agent_id, domain=""))
    _p(f"  loaded {len(bridges)} bridges → {len(cards)} unique cards "
       f"({dropped} dropped as dup/empty); feeding top {min(len(cards), CARD_CAP)}")
    return cards[:CARD_CAP]


async def main() -> None:
    _p("=" * 80)
    _p("RE-BLEND — reusing existing wander cards, fresh synthesis")
    _p("=" * 80)
    _p(f"  run dir: {RUN_DIR.name}")
    _p(f"  blender cost ceiling: ${CEIL}  | card cap: {CARD_CAP}")

    ci_main = base._build_cushion_input()
    question = ci_main.question.content
    angles = parse_required_angles(question)
    _p(f"  checkpoint: {len(angles)} required angles")

    cards = _load_cards()
    if not cards:
        _p("  [abort] no cards recovered")
        return

    client = LLMClient(mode=ClientMode.LIVE)

    # GOAL-AWARE: hand the blender the cushion (pursuit + checkpoint question +
    # constellation) so it SELECTS/fuses cards that advance the goal.
    _p("\n  composing goal-aware cushion…")
    blend_cushion = await compose_cushion(ci_main, client, session_id="reblend",
                                          auto_enrich=False)

    _p(f"\n  blending {len(cards)} cards (Opus 4-8 + R1 critic, ceiling ${CEIL})…")
    ms = await master_synthesize(cushion=blend_cushion, cards=cards, synthesis_map=None,
                                 client=client, cost_ceiling_usd=CEIL)
    blends = list(getattr(ms, "master_fusions", []) or [])
    _p(f"\n  [blends] {len(blends)} fusion(s)  (synth cost ${getattr(ms, 'total_cost_usd', 0.0):.2f}"
       f"{', BUDGET-TRUNCATED' if getattr(ms, 'truncated_by_budget', False) else ''}):")
    for i, b in enumerate(blends, 1):
        _p(f"    B{i} [{getattr(b, 'agreement_status', '?')}] {(getattr(b, 'title', '') or '')[:90]}")

    # coverage on the PROPOSALS = the true D_t
    blend_d_t = None
    blend_text = "\n".join(
        f"[B{i}] {getattr(b, 'title', '')}: {getattr(b, 'claim', '')} — "
        f"{(getattr(b, 'reasoning', '') or '')[:300]}"
        for i, b in enumerate(blends, 1) if getattr(b, "claim", ""))
    if blend_text.strip():
        bc = await score_coverage(angles, blend_text, model=COVERAGE_MODEL)
        blend_d_t = bc.d_t
        _p(f"\n  [coverage·BLENDS] D_t={bc.d_t}  (raw-card D_t was 0.8)")
        for pa in bc.per_angle:
            _p(f"      Q{pa['idx']}: {pa['coverage']:9} — {(pa.get('by') or '')[:80]}")

    try:
        (RUN_DIR / "blends.json").write_text(json.dumps(
            [b.to_dict() for b in blends], indent=2, default=str))
        _p(f"\n  wrote {RUN_DIR / 'blends.json'}")
    except Exception as e:
        _p(f"  [blends] persist failed: {e}")

    # ===== R1 FORMALIZATION — blends → testable math (same adapter as pipeline) =====
    n_formalized = 0
    if blends:
        _p("\n" + "=" * 80 + f"\nR1 FORMALIZATION — DeepSeek-R1 grounding {len(blends)} blend(s)\n" + "=" * 80)
        shape_by_id = {c.report_id: c.source_shape for c in cards}

        def _fusion_to_blend_dict(b, i):
            src = []
            for c in (getattr(b, "citations", []) or []):
                rid = getattr(c, "report_id", "") or (c.get("report_id") if isinstance(c, dict) else "")
                src.append({"report_id": rid, "source_shape": shape_by_id.get(rid, "")})
            return {"blend_id": f"B{i}", "thesis": getattr(b, "claim", "") or "",
                    "mechanism": getattr(b, "reasoning", "") or "",
                    "emergent_structure": getattr(b, "title", "") or "",
                    "advances_cushion": getattr(b, "limit", "") or "",
                    "selection": {"tension": ""}, "source_cards": src}

        blend_dicts = [_fusion_to_blend_dict(b, i) for i, b in enumerate(blends, 1)
                       if getattr(b, "claim", "")]
        try:
            rep = await formalize_blends(
                blend_dicts,
                on_progress=lambda ev, d: _p(f"    R1 {d.get('blend_id')}: "
                                             f"formalizable={d.get('formalizable')} ok={d.get('ok')} "
                                             f"${d.get('cost', 0)}"))
            n_formalized = len(getattr(rep, "formalizations", []) or [])
            (RUN_DIR / "formalize.json").write_text(json.dumps(rep.to_dict(), indent=2, default=str))
            (RUN_DIR / "formalize.md").write_text(render_markdown(rep))
            _p(f"\n  [R1] {n_formalized} formalization(s), ${getattr(rep, 'total_cost_usd', 0.0):.4f} "
               f"→ formalize.json + formalize.md")
        except Exception as e:
            _p(f"  [R1] formalization failed (non-fatal): {type(e).__name__}: {e}")

    true_total = client.get_total_cost_estimate()
    _p("\n" + "=" * 80)
    _p(f"RE-BLEND DONE: blends={len(blends)}  blend_D_t={blend_d_t}  formalized={n_formalized}  "
       f"reblend cost=${true_total:.2f}")
    _p(f"artifacts → {RUN_DIR}/ (blends.json, formalize.json, formalize.md)")
    _p("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
