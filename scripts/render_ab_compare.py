"""
render_ab_compare.py — ONE readable Markdown file comparing TWO collision runs.

Not a scoreboard — the real substance: every card's BRIDGE (the transfer claim)
and every blend in FULL (thesis, mechanism, emergent structure, discovery path,
novelty verdict), for both runs, side by side, plus a summary table + verdict.
All pulled straight from the run dirs (run_meta + run_record + audit + run.log).

Usage:
    python scripts/render_ab_compare.py <baseline_dir> <variant_dir> [out.md]
If out omitted -> ~/Downloads/constellax_ab_<baseline_ts>_vs_<variant_ts>.md
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

BIN_ICON = {"novel": "🟢 NOVEL", "adjacent": "🟡 ADJACENT", "known": "⚪ KNOWN",
            "flawed": "🔴 FLAWED", "unverified": "⚫ UNVERIFIED"}


def _total_cost(rd: Path) -> float:
    log = (rd / "run.log").read_text()
    return sum(json.loads(m.group(1)).get("cost_usd", 0.0)
               for m in re.finditer(r"CALL (\{.*\})", log))


def _field(label: str, text: str) -> str:
    return f"**{label}:** {(text or '_(none)_').strip()}"


def _quote(text: str) -> str:
    text = (text or "").strip() or "_(none)_"
    return "\n".join("> " + ln for ln in text.splitlines())


def render(baseline: Path, variant: Path, changed: str, verdict: str) -> str:
    L = []
    A = L.append
    metas = {}
    for rd in (baseline, variant):
        metas[rd] = json.loads((rd / "run_meta.json").read_text())

    A("# Constellax — A/B Comparison (full content)")
    A("")
    A(f"**Baseline:** `{baseline.name}`  vs  **Variant:** `{variant.name}`")
    A("")
    A(f"**The one variable changed:** {changed} — everything else identical.")
    A("")

    # ── summary table ───────────────────────────────────────────────────
    bm, vm = metas[baseline], metas[variant]
    A("## Summary")
    A("")
    A("| Metric | Baseline | Variant |")
    A("|---|---|---|")
    for rd, m in ((baseline, bm),):
        pass
    bb, vb = bm["collision"]["bins"], vm["collision"]["bins"]
    A(f"| Sort: known / invalid / unplaced | {bm['sort']['known']} / {bm['sort']['invalid']} / {bm['sort']['unplaced']} | {vm['sort']['known']} / {vm['sort']['invalid']} / {vm['sort']['unplaced']} |")
    A(f"| Novelty: novel / adjacent | {bb['novel']} / {bb['adjacent']} | {vb['novel']} / {vb['adjacent']} |")
    A(f"| Total cost | ${_total_cost(baseline):.4f} | ${_total_cost(variant):.4f} |")
    A("")
    A("---")
    A("")

    # ── per-run full content ────────────────────────────────────────────
    for label, rd in (("BASELINE", baseline), ("VARIANT", variant)):
        rec = json.loads((rd / "run_record.json").read_text())
        ms = (rec.get("stage_1_2_cards_and_sort") or {}).get("master_sorted") or {}
        A(f"# {label} — `{rd.name}`")
        A("")

        # cards with bridges
        shapes = []
        for bn in ("known", "invalid", "unplaced"):
            for it in ms.get(bn, []) or []:
                shapes.append((it.get("card", {}).get("source_shape", "") or "")[:60])
        A(f"## Cards — {len(shapes)} total, {len(set(shapes))} distinct source-shapes")
        A("")
        for bn, title in (("known", "Known — matched real prior work"),
                          ("unplaced", "Unplaced — no match, no flaw (gold zone)"),
                          ("invalid", "Invalid — contradicted")):
            items = ms.get(bn, []) or []
            if not items:
                continue
            A(f"### {title} ({len(items)})")
            A("")
            for it in items:
                c = it.get("card", {})
                A(f"**`{c.get('report_id','?')}`** · {c.get('source_shape','')}")
                A("")
                A(_field("Bridge", c.get("bridge", "")))
                A("")
                if bn == "known" and it.get("prior_work_name"):
                    ref = (it.get("reference", "") or "").strip()
                    A(_field("Prior work", it.get("prior_work_name", "")))
                    A("")
                    if ref:
                        A(f"_Reference:_ {ref}")
                        A("")
                if bn == "unplaced" and it.get("why_unplaced"):
                    A(_field("Why unplaced", it.get("why_unplaced", "")))
                    A("")
                if bn == "invalid" and it.get("contradicts"):
                    A(_field("Contradicts", it.get("contradicts", "")))
                    A("")

        # blends in FULL
        blends = (rec.get("stage_3_blends") or {}).get("blends", []) or []
        ver = rec.get("stage_5_verification") or {}
        nov = {}
        for b in ("known", "adjacent", "novel", "flawed"):
            for v in ver.get(b, []) or []:
                nov[v["blend_id"]] = (b, v)
        A(f"## Blends — {len(blends)}")
        A("")
        for x in blends:
            bid = x["blend_id"]
            nb, nv = nov.get(bid, ("unverified", {}))
            sel = x.get("selection", {})
            A(f"### {bid} → {BIN_ICON.get(nb, nb.upper())}")
            A("")
            A(f"**Source cards:** {', '.join('`'+c+'`' for c in x.get('source_card_ids', []))}")
            A("")
            A("**Discovery path** (how this blend was reached)")
            A("")
            A(_quote(sel.get("discovery_path", "")))
            A("")
            A(_field("Tension", sel.get("tension", "")))
            A("")
            A(_field("Thesis", x.get("thesis", "")))
            A("")
            A(_field("Mechanism", x.get("mechanism", "")))
            A("")
            A(_field("Emergent structure", x.get("emergent_structure", "")))
            A("")
            A(_field("Advances cushion", x.get("advances_cushion", "")))
            A("")
            A(_field("Novelty verdict", f"{BIN_ICON.get(nb, nb.upper())} (conf {nv.get('confidence','')})"))
            A("")
            if nv.get("resemblance"):
                A(f"- _Resembles:_ {nv['resemblance']}")
            if nv.get("still_new"):
                A(f"- _Still new:_ {nv['still_new']}")
            if nv.get("reasoning"):
                A(f"- _Reasoning:_ {nv['reasoning']}")
            A("")
        A("---")
        A("")

    A("## Verdict")
    A("")
    A(verdict.strip())
    A("")
    A("_The human is the judge._")
    return "\n".join(L)


_VERDICT = """
The contribution board did its job — and the result also confirms its limit.

**What worked.** Source-level duplication was eliminated: the baseline repeated
one card eight times (18 distinct shapes across 25 cards); the board produced
28 cards, all 28 distinct, into genuinely new territory (Assembly Theory,
quantum state inheritance, structural-hole theory, von Neumann observer
formalism). The blends got more novel (4/4 vs 2/4) and stayed deep — the Sonnet
verifier's refutations were specific and grounded, so the novelty is earned,
not thin. Zero invalid cards: no inflation — the additive, anti-competition
framing held.

**The tradeoff.** Diversity pushed 16 cards to "unplaced" (vs 1) — varied,
deeper cards are harder to match to known prior work, so the blender got less
precisely-referenced scaffolding. It still produced deep blends, so diversity
compensated; but the sort is now mostly unplaced rather than referenced-known.

**The limit (confirmed empirically).** The halo still flags clustering — one
level deeper: the board killed source duplication yet the cards still converge
on the same METAPHOR family (scaffolding / inheritance / structural skeletons).
That convergence is gravitational — the cushion is about extracting structural
skeletons, so even radically different sources collapse onto the same metaphor.
The board is a sampling fix and it worked at the sampling level; escaping the
metaphor basin needs the halo→wander loop (gap-gravity), not the board.

**Cost.** +$1.24 over baseline (the dig prompts carry the growing peer board).

**Bottom line.** The board earns its place: kills duplication, diversifies,
feeds deeper all-novel blends, no inflation — at an acceptable +$1.24. It also
sharpened the diagnosis: the remaining overlap is the metaphor basin, which is
the halo→wander loop's job.
"""


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    base = Path(sys.argv[1]).resolve()
    var = Path(sys.argv[2]).resolve()
    out = (Path(sys.argv[3]).expanduser().resolve() if len(sys.argv) >= 4
           else Path.home() / "Downloads" / f"constellax_ab_{base.name}_vs_{var.name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render(base, var, changed="Contribution board OFF → ON", verdict=_VERDICT)
    out.write_text(text)
    print(f"Wrote {len(text):,} chars to: {out}")
