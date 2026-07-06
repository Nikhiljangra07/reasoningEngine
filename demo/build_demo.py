"""
build_demo.py — assemble the Constellax demo data from the REAL run artifacts.

ISOLATED + SAFE: reads only saved run files; writes a single data.js the static
demo page loads. Touches no production code, no server, no pipeline. Every figure
that can come from disk does; curated analysis text (Ψ, toy results, novelty) is
verified against RESEARCH_CASE_STUDY.md / psi-toy/RESULTS.md.

Run:  python3 demo/build_demo.py   ->  writes demo/data.js
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs" / "auton-c4" / "run-20260618-044258"
OUT = Path(__file__).resolve().parent / "data.js"


def _load_cycles() -> list[dict]:
    cycles = []
    for n in (1, 2, 3, 4):
        d = json.load(open(RUN / f"cycle-{n}" / "cycle.json"))
        shep = d.get("shepherd") or {}
        cycles.append({
            "n": n,
            "cards_total": d.get("cards_total"),
            "d_t": d.get("coverage", {}).get("d_t"),
            "shepherd": shep.get("status") or shep.get("verdict") or "—",
            "refocus": (shep.get("refocus") or "")[:140],
            "halt": d.get("halt") or "",
        })
    return cycles


def _load_blends() -> list[dict]:
    blends = []
    for i, b in enumerate(json.load(open(RUN / "blends.json")), 1):
        blends.append({
            "id": f"B{i}",
            "status": b.get("agreement_status", ""),
            "title": b.get("title", ""),
            "claim": (b.get("claim") or "")[:340],
        })
    return blends


# Curated standout cards — selected by distinctive content (report_ids are not
# globally unique across cycles, so we match on source_shape text, which is safe).
_CARD_PICKS = [
    ("Stoic philosophy", "Philosophy", r"\bstoic"),
    ("Buddhist dependent origination", "Philosophy", r"dependent origination|prat[iī]tyasamutp[aā]da"),
    ("Collective agency as causal abstraction", "Causal theory", r"collective agency.*causal abstraction|causal abstraction.*collective"),
    ("SWARM economic sim — acceptance decline", "Economics", r"SWARM"),
    ("CoT monitoring is worse with more signal", "AI safety", r"chain-of-thought.*worse|full.*trajectory.*worse|extract-and-evaluate"),
    ("EU AI Act Art.14 — oversight recursion", "Law/Governance", r"Article 14|rubber-stamp"),
]


def _load_cards() -> list[dict]:
    bridges = json.load(open(RUN / "cycle-4" / "cycle.json"))["card_bridges"]
    out = []
    for label, domain, pat in _CARD_PICKS:
        rx = re.compile(pat, re.I)
        match = next((b for b in bridges if rx.search(b.get("source_shape", ""))), None)
        if match:
            out.append({
                "label": label,
                "domain": domain,
                "shape": match["source_shape"][:300],
                "bridge": (match.get("bridge") or "")[:200],
            })
    return out


# Verified analysis content (matches RESEARCH_CASE_STUDY.md + psi-toy/RESULTS.md).
CRITERION = {
    "name": "Ψ  —  deconfounded residual dependence",
    "formula": "Ψ = I(A₁ ; A₂ | C)   —   conditional mutual information after subtracting common causes C",
    "decision": "Ψ ≈ 0  ⇒  correlation (common cause).   Ψ > 0  ⇒  coordination (mutual influence).",
    "anchor": "Ψ_obs ≤ Ψ_causal (interventional influence); equality iff C is back-door complete — so Ψ is the identifiable lower bound on true coordination.",
}

VALIDATION = {
    "regimes": [
        {"name": "A — coordinated (real coupling)", "psi": "0.345", "naive": "0.95", "truth": "coordinated"},
        {"name": "B — common-caused, independent", "psi": "0.0001", "naive": "0.80", "truth": "correlation"},
        {"name": "C — covert coordination (hidden channel)", "psi": "0.326", "naive": "0.86", "truth": "coordinated"},
        {"name": "D — unobserved benign cause", "psi": "0.328", "naive": "0.86", "truth": "correlation"},
    ],
    "win": "Naive correlation is high in BOTH A and B (0.95, 0.80) — it false-positives the innocent regime. Ψ separates them: 0.345 vs 0.0001.",
    "ceiling": "Ψ_C ≈ Ψ_D: from outside, covert coordination is mathematically indistinguishable from an unmeasured common cause. Ψ is a screen with a hard identifiability ceiling, not an oracle.",
    "note": "Estimator self-validated against closed-form Ψ to < 0.002 in every regime before any verdict.",
}

NOVELTY = [
    {"output": "B1 — interventional separator", "verdict": "Known", "why": "causal-influence-via-intervention is established (SCIC 2312.09539; causal-incentives literature)."},
    {"output": "B3 — more observation = more attack surface", "verdict": "Known", "why": "CoT Red-Handed (2505.23575) publishes the core finding directly."},
    {"output": "B4 — order parameter", "verdict": "Partial", "why": "stat-mech order parameters exist; self-disqualified by its coupled-OR-shocked limit."},
    {"output": "B2 — channel-free residual after deconfounding", "verdict": "Survives", "why": "Colosseum (2602.15198) needs a known optimum & cannot isolate shared-origin correlation; 2601.00360 names the gap as open with no such framework."},
]

HONEST = {
    "shows": [
        "Ran end-to-end on a real published open problem (cushion → 4 cycles → 218 cards → blends → Ψ → testable math).",
        "Did not hallucinate — re-derived real, literature-consistent results.",
        "Produced a validated criterion + a sharp impossibility boundary (the identifiability ceiling).",
        "Cross-domain wandering surfaced analogies (Stoicism, Buddhism, stat-mech) a single-shot answer discards.",
    ],
    "doesnt": [
        "Did NOT solve the paper's problem — Ψ is a screen with an identifiability ceiling, not a deployable detector.",
        "Result is KNOWN, not novel — a single GPT-5 prompt matched it for ~2¢ vs. 4h / $24.",
        "The wander clusters on topic (48 of 218 cards re-found the same source) — divergence underperforms.",
        "What it truly demonstrates is the system + the honesty of the evaluation, not a discovery.",
    ],
}

LINKS = [
    {"label": "Source paper — Multi-Agent Risks from Advanced AI (Hammond et al., 2025)", "url": "https://arxiv.org/abs/2502.14143"},
    {"label": "Colosseum — auditing collusion", "url": "https://arxiv.org/abs/2602.15198"},
    {"label": "Mapping Human Anti-collusion Mechanisms", "url": "https://arxiv.org/abs/2601.00360"},
    {"label": "CoT Red-Handed", "url": "https://arxiv.org/abs/2505.23575"},
    {"label": "Secret Collusion among AI Agents", "url": "https://arxiv.org/abs/2402.07510"},
]

PROBLEM = {
    "title": "Distinguishing genuine coordination from mere correlation among AI agents — from outside",
    "paper": "Multi-Agent Risks from Advanced AI (Cooperative AI Foundation, 2025)",
    "paper_url": "https://arxiv.org/abs/2502.14143",
    "angles": [
        "What observable signature separates coordination from shared-cause correlation?",
        "Is 'intent' the wrong criterion — replace it with an interventional test?",
        "How to detect emergent coordination with no message to intercept?",
        "What to rely on when the channel is covert / undetectable in principle?",
        "What governs the correlation↔coordination boundary — a measurable order parameter?",
    ],
}


def main() -> None:
    data = {
        "problem": PROBLEM,
        "cycles": _load_cycles(),
        "cards": _load_cards(),
        "blends": _load_blends(),
        "criterion": CRITERION,
        "validation": VALIDATION,
        "novelty": NOVELTY,
        "honest": HONEST,
        "links": LINKS,
        "stats": {"cost_usd": 24.35, "time": "4h 14m", "cards": 218, "cycles": 4,
                  "reblend_usd": 3.06, "blend_d_t": 0.8},
    }
    OUT.write_text("// AUTO-GENERATED from real run artifacts by build_demo.py — do not hand-edit.\n"
                   "window.RUN_DATA = " + json.dumps(data, indent=2) + ";\n")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")
    print(f"  cycles={len(data['cycles'])} cards={len(data['cards'])} blends={len(data['blends'])} "
          f"novelty={len(data['novelty'])}")


if __name__ == "__main__":
    main()
