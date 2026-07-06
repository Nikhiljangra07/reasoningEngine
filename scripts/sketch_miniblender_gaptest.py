"""
sketch_miniblender_gaptest.py — SMOKE TEST: can the mini-blender (DeepSeek V4 Pro)
do GAP-NAMING, not just edge-detection? And does it CONVERGE with the Halo auditor?

This tests the capability I flagged as unvalidated: edge-detection ("does A fit B?")
is a closed task; gap-naming ("what piece is MISSING?") is open-ended generation.
There is NO ground truth for "the correct missing pieces" — so this measures the
ONLY thing that's measurable: do two independent detectors converge?

  detector 1 (mini-blender, DeepSeek): sees ONLY the findings (question-BLIND),
    names structural gaps — what bridge/piece is absent + what domain it lives in.
  detector 2 (Halo auditor, audit.json): saw the findings AND the question,
    flagged territory gaps ("no card addresses X").

Convergence = the user's hypothesis: pieces the mini-blender says are missing
should point toward the same territory the auditor flagged. Divergence is NOT
noise — auditor-only gaps are territory the mini-blender is structurally blind to.

CHAOS LAW: the mini-blender is given findings only, never the question.

Usage: PYTHONPATH=. python scripts/sketch_miniblender_gaptest.py runs/r-collision/20260615-212736
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sketch_miniblender_validate import HEADERS, URL, _card_text

MODEL = "deepseek/deepseek-v4-pro"

GAP_SYSTEM = (
    "You are a STRUCTURAL GAP detector in a reasoning engine. You are given a set of "
    "research findings — each a node in a knowledge skeleton being assembled. Some nodes "
    "share deep mechanisms and connect; some clusters stand alone or dangle.\n"
    "Your job: identify the MISSING PIECES — findings that are ABSENT but would either "
    "(a) BRIDGE two otherwise-disconnected clusters, or (b) COMPLETE a partial structure.\n"
    "You have NO goal and NO question — judge ONLY from the structure of what is present. "
    "Do not invent a topic; reason from what the findings themselves leave unconnected.\n"
    "For each gap name: 'missing' (what piece/bridge is absent, one sentence), 'links' "
    "(which existing findings it would connect, by their bracket domain), and 'domain' "
    "(the territory a searcher should walk into to find it, 2-4 words).\n"
    'Output ONLY compact JSON: {"gaps":[{"missing":"...","links":"...","domain":"..."}]} '
    "with at most 6 gaps, the most structurally load-bearing first."
)


def _post(body: dict) -> tuple[str, float]:
    for attempt in range(2):
        try:
            r = httpx.post(URL, headers=HEADERS, json=body, timeout=90.0)
            r.raise_for_status()
            d = r.json()
            cost = float((d.get("usage") or {}).get("cost", 0.0) or 0.0)
            content = (d["choices"][0]["message"].get("content") or "").strip()
            if content:
                return content, cost
        except Exception as e:
            if attempt == 1:
                return f"ERR:{type(e).__name__}", 0.0
    return "EMPTY", 0.0


def _parse_json(txt: str) -> dict:
    s = txt.strip()
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    try:
        return json.loads(s)
    except Exception:
        return {}


def main(run_dir: Path) -> None:
    dos = json.loads((run_dir / "dossier.json").read_text())
    audit = json.loads((run_dir / "audit.json").read_text())
    cards = (dos.get("high") or []) + (dos.get("medium") or []) + (dos.get("low") or [])

    # detector 2: the auditor's CARDS-layer blind spots (territory gaps) — the reference
    auditor_gaps = [b["blind_spot"] for b in audit.get("cards_audit", {}).get("blind_spots", [])]

    # detector 1: mini-blender, question-BLIND, names structural gaps
    findings_block = "\n".join(f"{i+1}. {_card_text(c)}" for i, c in enumerate(cards))
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": GAP_SYSTEM},
                     {"role": "user", "content": f"The {len(cards)} findings:\n{findings_block}"}],
        "temperature": 0.2, "max_tokens": 8000, "usage": {"include": True},
    }
    print(f"gap-naming smoke test · mini-blender = {MODEL} (question-BLIND)\n", flush=True)
    raw, cost = _post(body)
    mb = _parse_json(raw).get("gaps", [])

    print("=" * 78)
    print("DETECTOR 1 — MINI-BLENDER structural gaps (saw findings only, NOT the question):")
    print("=" * 78)
    if not mb:
        print("  (no parseable gaps — capability FAIL)\n  raw:", raw[:300])
    for i, g in enumerate(mb, 1):
        print(f"  G{i}. [{g.get('domain','?')}] {g.get('missing','')}")
        print(f"       links: {g.get('links','')}")

    print("\n" + "=" * 78)
    print("DETECTOR 2 — HALO AUDITOR territory gaps (saw findings AND the question):")
    print("=" * 78)
    for i, a in enumerate(auditor_gaps, 1):
        print(f"  A{i}. {a[:200]}")

    # convergence: ask DeepSeek to match — for each auditor gap, does any mini-blender gap cover it?
    match_sys = (
        "For each AUDITOR gap, decide if ANY of the MINI-BLENDER gaps points at the same "
        "missing territory (even loosely — same underlying hole, not exact words). "
        'Output ONLY JSON: {"matches":[{"auditor":1,"covered_by":"G3"|null,"why":"..."}]}'
    )
    mb_txt = "\n".join(f"G{i+1}. [{g.get('domain','?')}] {g.get('missing','')}" for i, g in enumerate(mb))
    au_txt = "\n".join(f"A{i+1}. {a[:200]}" for i, a in enumerate(auditor_gaps))
    mbody = {
        "model": MODEL,
        "messages": [{"role": "system", "content": match_sys},
                     {"role": "user", "content": f"MINI-BLENDER gaps:\n{mb_txt}\n\nAUDITOR gaps:\n{au_txt}"}],
        "temperature": 0.0, "max_tokens": 8000, "usage": {"include": True},
    }
    mraw, mcost = _post(mbody)
    matches = _parse_json(mraw).get("matches", [])

    print("\n" + "=" * 78)
    print("CONVERGENCE (do the two detectors point at the same territory?):")
    print("=" * 78)
    covered = 0
    for m in matches:
        cov = m.get("covered_by")
        if cov:
            covered += 1
        tag = f"<- {cov}" if cov else "(auditor-ONLY — mini-blender blind to this territory)"
        ai = m.get("auditor", "?")
        print(f"  A{ai}: {tag}")
        if m.get("why"):
            print(f"        {m['why'][:140]}")

    n = len(auditor_gaps) or 1
    print(f"\n  CONVERGED: {covered}/{len(auditor_gaps)} auditor gaps corroborated by a mini-blender gap")
    print(f"  AUDITOR-ONLY (territory the mini-blender can't see): {len(auditor_gaps)-covered}/{len(auditor_gaps)}")
    print(f"  mini-blender produced {len(mb)} structural gaps total")
    print(f"  cost: ${cost+mcost:.4f}")


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass a run dir")
    main(rd)
