"""
render_run_report.py — turn a collision run's JSON into a readable MARKDOWN report.

Reads a runs/r-collision/<ts>/ directory (run_record.json + cushion_input.json
+ run_meta.json) and emits a single Markdown .md with everything in reading
order: config → cushion → cards+sort → blends (full provenance, discovery_path,
novelty verdicts) → trace index.

Usage:
    python scripts/render_run_report.py <run_dir> [output.md]
If output omitted, writes to ~/Downloads/constellax_collision_<ts>.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _short(model: str) -> str:
    return {
        "anthropic/claude-sonnet-4-6": "Sonnet 4.6",
        "anthropic/claude-opus-4-8":   "Opus 4.8",
        "anthropic/claude-haiku-4-5":  "Haiku 4.5",
        "deepseek/deepseek-v4-pro":    "DeepSeek V4 Pro",
    }.get(model, model)


def _quote(text: str) -> str:
    """Render text as a markdown blockquote (each line prefixed with > )."""
    text = (text or "").strip() or "_(none)_"
    return "\n".join("> " + ln for ln in text.splitlines()) or "> _(none)_"


def _field(label: str, text: str) -> str:
    return f"**{label}:** {(text or '_(none)_').strip()}"


BIN_ICON = {"novel": "🟢 NOVEL", "adjacent": "🟡 ADJACENT",
            "known": "⚪ KNOWN", "flawed": "🔴 FLAWED", "unverified": "⚫ UNVERIFIED"}


def render(run_dir: Path) -> str:
    rec = json.loads((run_dir / "run_record.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}
    try:
        ci = json.loads((run_dir / "cushion_input.json").read_text())
    except Exception:
        ci = {}

    out: list[str] = []
    A = out.append
    cr = meta.get("control_room", {})

    # ── header ──────────────────────────────────────────────────────────
    A("# Constellax Collision Pipeline — Run Report")
    A("")
    A(f"**Run:** `{meta.get('timestamp', run_dir.name)}` · "
      f"**Branch:** `{meta.get('git_branch','?')}` (git `{str(meta.get('git_sha',''))[:10]}`)")
    A("")
    A(f"**Domains:** {', '.join(cr.get('WANDER_DOMAINS', []))} · "
      f"**Mode:** {cr.get('WANDER_MODE','?')} · **Agents:** {cr.get('WANDER_AGENTS','?')}")
    A("")
    A("| Stage | Model |")
    A("|---|---|")
    A(f"| Wander | {_short(cr.get('WANDER_MODEL',''))} |")
    A(f"| Verified sort | {_short(cr.get('SORTER_MODEL',''))} + web |")
    A(f"| Blend | {_short(cr.get('BLENDER_MODEL',''))} |")
    A(f"| Drift-check | {_short(cr.get('DRIFT_CHECKER_MODEL',''))} |")
    A(f"| Blend-verify | {_short(cr.get('SORTER_MODEL',''))} + web |")
    A("")
    d = meta.get("durations_seconds", {})
    if d:
        A(f"**Durations:** cushion {d.get('cushion',0):.0f}s · wander {d.get('wander',0):.0f}s · "
          f"dossier+sort {d.get('dossier_sort',0):.0f}s · collision {d.get('collision',0):.0f}s")
        A("")
    col = meta.get("collision", {})
    sc = col.get("stage_costs", {})
    A(f"**Collision cost:** ${col.get('collision_cost_usd',0):.4f} "
      f"(blend ${sc.get('blend','?')} / drift ${sc.get('drift','?')} / verify ${sc.get('verify','?')})")
    A("")
    A("---")
    A("")

    # ── cushion ─────────────────────────────────────────────────────────
    A("## The Cushion — what everything anchors to")
    A("")
    for key, title in (("problem","Problem (pursuit)"), ("question","Question (checkpoint)"),
                       ("vision","Vision"), ("hunches","Hunches")):
        # Legacy runs stored hunches under 'current_map'.
        val = ci.get(key) or (ci.get("current_map") if key == "hunches" else "")
        if val:
            A(f"### {title}")
            A("")
            A(_quote(val))
            A("")
    A("---")
    A("")

    # ── stage 1-2: cards + sort ─────────────────────────────────────────
    dossier = rec.get("stage_1_2_cards_and_sort", {}) or {}
    ms = dossier.get("master_sorted", {}) or {}
    A("## Stages 1–2 · Wander → Verified Web-Sort")
    A("")
    A(f"**known {len(ms.get('known',[]))} · invalid {len(ms.get('invalid',[]))} · "
      f"unplaced {len(ms.get('unplaced',[]))}**")
    A("")
    for bin_name, title in (("known","Known — matched real prior work"),
                            ("invalid","Invalid — contradicted"),
                            ("unplaced","Unplaced — no match, no flaw (gold zone)")):
        items = ms.get(bin_name, []) or []
        if not items:
            continue
        A(f"### {title} ({len(items)})")
        A("")
        for it in items:
            card = it.get("card", {})
            rid = card.get("report_id","?")
            conf = it.get("confidence","")
            A(f"#### `{rid}` · conf {conf}")
            A("")
            A(_field("Bridge", card.get("bridge","")))
            A("")
            if bin_name == "known":
                ref = (it.get("reference","") or "").strip()
                prior = it.get("prior_work_name","")
                A(_field("Prior work", prior))
                A("")
                if ref.startswith("http"):
                    A(f"**Reference:** [{ref}]({ref})")
                else:
                    A(_field("Reference", ref))
                A("")
            if bin_name == "invalid":
                A(_field("Contradicts", it.get("contradicts","")))
                A("")
            if bin_name == "unplaced":
                A(_field("Why unplaced", it.get("why_unplaced","")))
                A("")
    A("---")
    A("")

    # ── stage 3-5: blends ───────────────────────────────────────────────
    batch = rec.get("stage_3_blends", {}) or {}
    blends = batch.get("blends", []) or []
    drift = rec.get("stage_4_drift", {}) or {}
    drift_by = {v["blend_id"]: v for v in drift.get("verdicts", [])}
    ver = rec.get("stage_5_verification", {}) or {}
    novelty_by = {}
    for bn in ("known","adjacent","novel","flawed"):
        for v in ver.get(bn, []) or []:
            novelty_by[v["blend_id"]] = (bn, v)
    quarantined = set(rec.get("quarantined_blend_ids", []))

    A("## Stages 3–5 · Blend → Drift-check → Novelty verify")
    A("")
    A(f"**{len(blends)} blends · {len(quarantined)} quarantined · "
      f"known {len(ver.get('known',[]))} / adjacent {len(ver.get('adjacent',[]))} / "
      f"novel {len(ver.get('novel',[]))} / flawed {len(ver.get('flawed',[]))}**")
    A("")

    for b in blends:
        bid = b.get("blend_id","?")
        nbin, nver = novelty_by.get(bid, ("unverified", {}))
        dv = drift_by.get(bid, {})
        sel = b.get("selection", {})
        tag = " · **QUARANTINED**" if bid in quarantined else ""
        A(f"### {bid} → {BIN_ICON.get(nbin, nbin.upper())}{tag}")
        A("")
        cards = ", ".join(f"`{c}`" for c in b.get("source_card_ids", []))
        A(f"**Source cards:** {cards}")
        A("")
        A(f"**Drift:** on-course `{dv.get('on_course','?')}` · resonance `{dv.get('resonance','?')}`")
        A("")
        A("**Discovery path** (genealogy — how this blend was reached)")
        A("")
        A(_quote(sel.get("discovery_path","")))
        A("")
        A(_field("Why these cards", sel.get("why_these_cards","")))
        A("")
        A(_field("Spark", sel.get("spark","")))
        A("")
        A(_field("Tension", sel.get("tension","")))
        A("")
        A(_field("Thesis", b.get("thesis","")))
        A("")
        A(_field("Mechanism", b.get("mechanism","")))
        A("")
        A(_field("Emergent structure", b.get("emergent_structure","")))
        A("")
        A(_field("Advances cushion", b.get("advances_cushion","")))
        A("")
        A(f"**Novelty verdict:** {BIN_ICON.get(nbin, nbin.upper())} (conf {nver.get('confidence','')})")
        A("")
        if nver.get("resemblance"):
            A(f"- _Resembles:_ {nver['resemblance']}")
        if nver.get("still_new"):
            A(f"- _Still new:_ {nver['still_new']}")
        if nver.get("reasoning"):
            A(f"- _Reasoning:_ {nver['reasoning']}")
        refs = nver.get("references", []) or []
        for r in refs:
            url = r.get("url","")
            title = r.get("title","") or url
            if url.startswith("http"):
                A(f"- _Reference:_ [{title}]({url})")
            elif title:
                A(f"- _Reference:_ {title}")
        A("")
        A("---")
        A("")

    # ── trace ───────────────────────────────────────────────────────────
    A("## Trace index — reverse-engineering map")
    A("")
    A("| Blend | Bin | Source cards (sort bin) |")
    A("|---|---|---|")
    for t in rec.get("trace", []):
        bins = t.get("source_card_bins", {})
        cards = ", ".join(f"`{c}` ({bins.get(c,'?')})" for c in t.get("source_card_ids", []))
        A(f"| {t.get('blend_id')} | {t.get('novelty_bin','?').upper()} | {cards} |")
    A("")
    A("---")
    A("")

    # ── quality ranking — final alignment pass ──────────────────────────
    q_path = run_dir / "quality.json"
    if q_path.exists():
        try:
            q = json.loads(q_path.read_text())
        except Exception:
            q = None
        if q and q.get("ranked"):
            A("## Quality Ranking — blends by advancement toward the cushion")
            A("")
            A(f"_Judge: {_short(q.get('model',''))} · cost ${q.get('total_cost_usd', 0):.4f}. "
              f"Ranked by advancement toward the cushion **alone**. The gaps each blend "
              f"resolves and any new gap it opens are surfaced beside the rank as a **map**, "
              f"never scored — severity is never weighted, because flaws are load-bearing and "
              f"which ones to solve is yours. Near-equal advancement **shares a rank (tie)** — "
              f"break those ties with the flaw-map. RANKED — nothing deleted._")
            A("")
            A("| Rank | Blend | Advancement | Novelty | Gaps resolved (map) | Opens new gap |")
            A("|---|---|---|---|---|---|")
            for r in q["ranked"]:
                gaps = ", ".join(r.get("blind_spots_addressed", []) or []) or "—"
                newg = (r.get("opens_new_gap", "") or "—")[:48]
                rank_disp = f"{r.get('rank')}" + (" (tie)" if r.get("tied") else "")
                A(f"| {rank_disp} | {r.get('blend_id')} | {r.get('advancement', 0):.2f} | "
                  f"{r.get('novelty_bin', '')} | {gaps} | {newg} |")
            A("")
            for r in q["ranked"]:
                note = (r.get("advancement_note", "") or "").strip()
                rat = (r.get("rationale", "") or "").strip()
                if note or rat:
                    tie = " · tied" if r.get("tied") else ""
                    A(f"- **{r.get('blend_id')}** (rank {r.get('rank')}{tie}): {note}"
                      + (f" — _{rat}_" if rat else ""))
            A("")
            A("---")
            A("")

    # ── halo audit — blind spots (observer layer) ───────────────────────
    audit_path = run_dir / "audit.json"
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text())
        except Exception:
            audit = None
        if audit:
            A("## Halo Audit — blind spots (observer only)")
            A("")
            A(f"_Auditor: {_short(audit.get('model',''))} · "
              f"cost ${audit.get('total_cost_usd', 0):.4f}. Sits on top of the pipeline and "
              f"audits each layer for what's missing — observer only, acts on nothing._")
            A("")
            sev_icon = {"high": "🔴 HIGH", "medium": "🟠 MEDIUM", "low": "🟡 LOW"}
            for key, title in (
                ("cushion_audit", "Cushion — the question itself"),
                ("cards_audit",   "Cards — the wander's coverage"),
                ("blends_audit",  "Blends — the lanes' holes"),
            ):
                la = audit.get(key)
                if not la:
                    continue
                A(f"### {title}")
                A("")
                spots = la.get("blind_spots", []) or []
                if not spots:
                    A("_(none returned)_")
                    A("")
                for i, b in enumerate(spots, 1):
                    tag = sev_icon.get(b.get("severity", ""), "⚪")
                    A(f"{i}. **{tag}** — {b.get('blind_spot', '')}")
                    if b.get("why_it_matters"):
                        A(f"   - _Why it matters:_ {b['why_it_matters']}")
                    if b.get("suggested_angle"):
                        A(f"   - _Angle (for a future commander):_ {b['suggested_angle']}")
                A("")
            A("---")
            A("")

    A("_The human is the judge._")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    run_dir = Path(sys.argv[1]).resolve()
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2]).expanduser().resolve()
    else:
        out_path = Path.home() / "Downloads" / f"constellax_collision_{run_dir.name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = render(run_dir)
    out_path.write_text(text)
    print(f"Wrote {len(text):,} chars to: {out_path}")
