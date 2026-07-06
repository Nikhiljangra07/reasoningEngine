"""
probe_dig_models.py — CONTROLLED dig-quality A/B: Sonnet vs DeepSeek vs Haiku.

WHY: we removed Sonnet from the wander dig mix (now DeepSeek + Haiku). The open
question is whether DeepSeek+Haiku hold the *nuance* Sonnet gave on a dig. A fresh
$4 wander would answer it only as a confounded smell test (chaos variance). This
answers it as a true controlled A/B for cents: SAME input, three models, side by
side — you judge the nuance (flow-not-judge: this surfaces, Nikhil decides).

WHAT IS FAITHFUL vs APPROXIMATED (honest):
  • anchor          REAL  — rebuilt from Cushion 3 session.json cushion (problem/
                            context/vision/hunches + 3 layer summaries). The QUESTION
                            is EXCLUDED — chaos law: the judge-checkpoint never leaks
                            into the wander anchor.
  • iteration-1     REAL  — the actual four-field dig reports saved in the run.
  • the REVISE step REAL  — verbatim _DIG_REVISE_SYSTEM_PROMPT (mirrored from
                            agent.py:746), the exact prompt the wander uses.
  • the critique    SYNTH — one fixed, neutral critique applied IDENTICALLY to all
                            three models (fair — it's a constant, not a per-model var).
  • identity wrap   OMIT  — compose_system_prompt's persona preamble is dropped; it
                            is identical across models, so it cannot bias a relative
                            comparison. The dig reasoning prompt is the controlled var.
  • the FIND step   N/A   — not replayable: the source ARTICLES the digs read were
                            never persisted (only outputs were). We test REVISE, whose
                            inputs we have, and whose job IS nuance preservation.

SAFE: standalone. Reads one frozen run; writes only a markdown report. Touches no
src/, no pipeline state, nothing committed.

Usage: PYTHONPATH=. python3 scripts/probe_dig_models.py runs/r-collision/20260616-171733
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sketch_miniblender_gaptest import _post, _parse_json

# The three models under test. baseline = what we removed; the other two = the
# new wander dig mix we want to prove holds the nuance.
MODELS = [
    ("SONNET (baseline, removed)", "anthropic/claude-sonnet-4-6"),
    ("DEEPSEEK (new workhorse)",   "deepseek/deepseek-v4-pro"),
    ("HAIKU (new nuance)",         "anthropic/claude-haiku-4-5"),
]

N_CONTEXTS = 6  # ~18 small calls total → cents

# Verbatim from src/wandering/agent.py:746 (_DIG_REVISE_SYSTEM_PROMPT). Mirrored
# here to avoid importing agent.py's heavy module-level deps. Keep in sync.
DIG_REVISE_SYSTEM = """\
You are the same wandering agent. You just produced ITERATION 1 of this
dig. A self-critique just ran on it. Now you revise.

YOU HAVE:
  - Your iteration 1 work (the four-field report you just wrote)
  - The self-critique's verdict + any red-flagged questions + summary
  - The same article and problem map you read in iteration 1

YOUR JOB — revise. Do these four things in order:

  1. READ THE CRITIQUE HONESTLY. What did it flag? What did it confirm?
  2. REVISE iteration 1's content. Keep what's solid, fix what's weak.
  3. EXTEND where the critique opened a new thread.
  4. STRENGTHEN the cross-world analogy if iteration 1's was thin. Do
     genuine second-pass thinking, NOT rephrasing.

CRITICAL — PRESERVE THE VOCABULARY OF ITERATION 1.
  If iteration 1 used unusual, specific, or cross-domain language — terms
  like "attractor hazard", "negative topology", "crystallization from
  collision" — KEEP THEM unless structurally wrong. Do NOT sand iteration 1's
  prose down into safer, more conventional phrasing. The unusual vocabulary IS
  the signal of cross-domain analogical work. Polished prose with generic
  vocabulary is the failure mode — not the success mode.

HONEST DISAGREEMENT IS ALLOWED. If iteration 1 was right and the critique was
off-base, write iteration 2 essentially identical and say so in next_lead.

OUTPUT FORMAT — return ONE JSON object with the four standard fields:
{
  "exploration_summary": "<2-3 sentences>",
  "advancement": "<1-2 sentences>",
  "what_does_not_map": "<1-2 sentences; MANDATORY>",
  "next_lead": "<optional>"
}
No prose preamble. No code fences. Just JSON.
"""

# One neutral critique, IDENTICAL for all three models (the controlled constant).
FIXED_CRITIQUE = (
    "SELF-CRITIQUE: The cross-world analogy could be sharper and more concrete. "
    "Confirm what_does_not_map names a specific mismatch, not a generic hedge. "
    "Preserve any unusual cross-domain vocabulary; do not generalize it away."
)

# CANDIDATE STRENGTHENING (A/B tested before any merge into agent.py). Targets the
# two failure modes the baseline probe surfaced in DeepSeek + Haiku — NOT the
# vocabulary block (which already holds). Appended to the revise system prompt only
# when PROBE_STRENGTHEN=1. Sonnet runs it too, as a no-regression control.
STRENGTHEN_BLOCK = """\

NAME THE SHARPEST JOINT — DO NOT LABEL IT, DO NOT REUSE BOILERPLATE.
  Your `advancement` must name the ONE point where THIS source and THIS anchor
  correspond most tightly, in the form:
    "<a specific feature of the source in front of you> IS <a specific anchor
     mechanism> — and this predicts <a concrete consequence for the design>."
  Use the actual vocabulary of THIS source and THIS anchor. Never a stock phrase.
  TEST: if your advancement sentence could be pasted onto a DIFFERENT source
  unchanged, it is boilerplate, not a joint — rewrite it with this source's terms.
    WEAK (a label): "a proven design pattern for holding the divergence-convergence
      tension", "a mechanism for dual-signal fusion", "addresses the allocation
      problem". The fix is NOT to swap in a different stock phrase — it is to name
      what, in THIS source, corresponds to what, in THIS anchor, and what that buys.

CALIBRATE — DO NOT INFLATE, DO NOT FORCE THE JOINT.
  One concrete, defensible correspondence beats three loose ones. If a mapping is
  partial, name which part holds and which part breaks. If THIS source does not
  actually contain a tight joint, say so plainly — a forced "X IS Y" on a source
  that doesn't support it is WORSE than an honest "the tightest correspondence
  here is only partial, because…". Never round a partial match up to a clean one.
"""


def build_anchor(cushion: dict) -> str:
    """Rebuild the wander anchor from the saved cushion — QUESTION EXCLUDED (chaos law)."""
    ri = cushion.get("raw_input", {})
    def g(d, k):
        v = d.get(k)
        return (v.get("content") if isinstance(v, dict) else v) or ""
    parts = ["# ANCHOR (the user's problem map — background, not a question to answer)"]
    for label, key in (("PROBLEM", "problem"), ("CONTEXT", "context"),
                       ("VISION", "vision"), ("HUNCHES", "hunches")):
        t = g(ri, key)
        if t:
            parts.append(f"## {label}\n{t[:600]}")
    for layer in ("actual", "essence", "mechanism"):
        s = (cushion.get(layer) or {}).get("summary", "")
        if s:
            parts.append(f"## LAYER {layer.upper()}\n{s[:500]}")
    return "\n\n".join(parts)


def pick_contexts(reports: list[dict], n: int) -> list[dict]:
    """Richest reports = best dig material to revise."""
    def richness(r):
        return len(r.get("exploration_summary", "")) + len(r.get("advancement", "")) \
            + len(r.get("what_does_not_map", "")) + len(r.get("next_lead", ""))
    return sorted(reports, key=richness, reverse=True)[:n]


def iter1_block(r: dict) -> str:
    return json.dumps({
        "exploration_summary": r.get("exploration_summary", ""),
        "advancement": r.get("advancement", ""),
        "what_does_not_map": r.get("what_does_not_map", ""),
        "next_lead": r.get("next_lead", ""),
    }, indent=2)


STRENGTHEN = os.environ.get("PROBE_STRENGTHEN", "0") == "1"


def revise(model: str, anchor: str, r: dict) -> tuple[dict, float]:
    system = DIG_REVISE_SYSTEM + (STRENGTHEN_BLOCK if STRENGTHEN else "")
    user = (
        f"{anchor}\n\n"
        f"# SOURCE DOMAIN EXPLORED\n{r.get('domain_explored','?')} — "
        f"{r.get('anchor_summary','')}\n\n"
        f"# YOUR ITERATION 1 REPORT\n{iter1_block(r)}\n\n"
        f"# {FIXED_CRITIQUE}\n\n"
        "Revise iteration 1 per the four steps in your instructions. Return the "
        "four-field JSON only."
    )
    raw, cost = _post({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.3, "max_tokens": 4000, "usage": {"include": True},
    })
    return _parse_json(raw) or {"_raw": raw[:300]}, cost


def main(run_dir: Path) -> None:
    s = json.loads((run_dir / "session.json").read_text())
    anchor = build_anchor(s.get("cushion", {}))
    contexts = pick_contexts(s.get("reports", []), N_CONTEXTS)
    mode = "STRENGTHENED prompt" if STRENGTHEN else "BASELINE prompt"
    print(f"[probe] {mode} — {len(contexts)} real dig contexts × {len(MODELS)} models "
          f"= {len(contexts)*len(MODELS)} calls (revise step)")

    out_md = [f"# Dig-quality A/B — Sonnet vs DeepSeek vs Haiku (revise step) — {mode}",
              f"\nSource run: `{run_dir.name}` · {len(contexts)} real contexts · "
              "same input, three models. **You judge the nuance.**\n",
              "Read each block for: did it PRESERVE the cross-domain vocabulary and "
              "sharpen the analogy (good), or SAND it into generic prose (the failure "
              "mode the prompt warns against)?\n"]
    total = 0.0
    for i, r in enumerate(contexts, 1):
        print(f"\n[{i}/{len(contexts)}] {r.get('domain_explored','?')} "
              f"({r.get('report_id','?')})")
        out_md.append(f"\n---\n\n## Context {i} — domain: "
                      f"`{r.get('domain_explored','?')}`  (report {r.get('report_id','?')})")
        out_md.append(f"\n**Original iteration-1 (what's being revised):**\n"
                      f"> {r.get('exploration_summary','')[:400]}\n")
        for label, model in MODELS:
            rev, cost = revise(model, anchor, r)
            total += cost
            print(f"    {label:28} ${cost:.4f}")
            es = rev.get("exploration_summary", rev.get("_raw", "—"))
            adv = rev.get("advancement", "")
            wdm = rev.get("what_does_not_map", "")
            out_md.append(
                f"\n### {label}  (${cost:.4f})\n"
                f"- **exploration:** {es}\n"
                f"- **advancement:** {adv}\n"
                f"- **what_does_not_map:** {wdm}\n")
    out_md.append(f"\n---\n\n**Total cost: ${total:.4f}** · "
                  f"{len(contexts)*len(MODELS)} calls\n")

    report = run_dir / ("dig_model_ab_strong.md" if STRENGTHEN else "dig_model_ab.md")
    report.write_text("\n".join(out_md))
    print(f"\n[probe] total ${total:.4f} → {report}")
    print("[probe] read it side by side — you judge whether DeepSeek+Haiku hold "
          "Sonnet's nuance.")


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass the Cushion 3 run dir, e.g. runs/r-collision/20260616-171733")
    main(rd)
