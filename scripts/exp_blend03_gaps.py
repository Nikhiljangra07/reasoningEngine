"""
exp_blend03_gaps.py — falsifiable test of blend-03 (absence-coded anchor).

blend-03's premise: anchor the wander to the inherited work's IMPLIED-BUT-
UNFILLED gaps (absence-shape) instead of its present content, and the search
gets pulled PAST adjacency toward genuinely new territory.

Its load-bearing risk: gap-extraction may be as hard as the original problem.
So this experiment tests TWO things on the saved collision run:

  (P1) GAP QUALITY — extract implied-gaps from the inherited work the wander
       surfaced, then a SKEPTICAL verifier rates each: genuinely implied?
       genuinely unfilled? non-trivial? If the gaps are mostly trivial or
       already-filled, blend-03's fuel is bad and it dies here.

  (P2) OFF-CLUSTER — search each gap, embed the hits, and measure their
       distance from the existing 24-card cluster centroid AGAINST a
       present-content control search. blend-03 predicts gap-hits land
       FARTHER from the cluster than control-hits (it escapes adjacency).
       If gap-hits sit on the cluster, blend-03 doesn't escape.

Read-only on the pipeline. Spends ~$0.20 (Sonnet extract+verify + searches +
embeddings). Writes results into the run dir; prints a verdict.

Usage:
    PYTHONPATH=. python scripts/exp_blend03_gaps.py [run_dir]
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import control_room
from src.llm.client import ClientMode, LLMClient
from src.wandering.fetcher import search_chain
from src.wandering.master_synthesizer import _parse_json_safely

MODEL = control_room.WANDER_MODEL       # Sonnet — match the wander's eyes
N_GAPS = 6
QUERIES_PER_GAP = 2
HITS_PER_QUERY = 5
CONCURRENCY = 5


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _centroid(vecs):
    if not vecs:
        return []
    n = len(vecs[0])
    c = [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]
    return c


def _load(run_dir: Path):
    rec = json.loads((run_dir / "run_record.json").read_text())
    cushion = rec.get("cushion", {}).get("problem", "")
    ms = (rec.get("stage_1_2_cards_and_sort", {}) or {}).get("master_sorted", {}) or {}
    cards, priors = [], []
    for it in ms.get("known", []) or []:
        c = it.get("card", {})
        cards.append((c.get("source_shape", "") + " — " + c.get("spark", "") + " " + c.get("bridge", "")).strip())
        if it.get("prior_work_name"):
            priors.append(it["prior_work_name"])
    # distinct inherited works
    seen, distinct = set(), []
    for p in priors:
        key = p.split(":")[0].strip().lower()
        if key not in seen:
            seen.add(key); distinct.append(p)
    return cushion, cards, distinct


_EXTRACT_SYS = "You identify structural gaps. Output ONLY JSON."
_EXTRACT = """\
A user is building this concept (their CUSHION):
{cushion}

A search surfaced this body of INHERITED / published work around it:
{inherited}

Identify {n} structurally IMPLIED-BUT-UNFILLED gaps: things this body of work
(taken together with the user's concept) PRESUPPOSES or points toward, but
that no current work actually resolves. A gap is the NEGATIVE SPACE — the slot
the existing structure implies must exist but never fills. Not "more research
needed"; a specific structural absence.

Output JSON only:
{{"gaps": [{{"gap": "<crisp statement of the specific absence>",
            "why_implied": "<what in the work implies this slot must exist>",
            "search_queries": ["<query to find work filling this gap>", "<alt phrasing>"]}}]}}
"""

_VERIFY_SYS = "You are a ruthless skeptic. A gap is GUILTY of being trivial or already-filled until proven otherwise. Output ONLY JSON."
_VERIFY = """\
Here are proposed structural gaps in a body of work. For EACH, judge honestly:
  - implied:   is it genuinely IMPLIED by the work, or invented/unrelated? (0..1)
  - unfilled:  is it genuinely UNFILLED, or does work you know already solve it? (0..1)
  - nontrivial: is it a DEEP structural gap, or an obvious/minor one? (0..1)
A real gap scores high on all three. Default low when unsure.

GAPS:
{gaps}

Output JSON only:
{{"verdicts": [{{"gap": "<copy first ~8 words>", "implied": <f>, "unfilled": <f>, "nontrivial": <f>,
               "real_gap_score": <f, the min-ish of the three>, "note": "<one clause>"}}]}}
"""


async def run(run_dir: Path) -> dict:
    cushion, card_texts, inherited = _load(run_dir)
    if not card_texts:
        print("no cards"); return {}
    client = LLMClient(mode=ClientMode.LIVE)
    t0 = time.time()

    # ── P1a: extract gaps ───────────────────────────────────────────────
    inh = "\n".join(f"- {p}" for p in inherited[:12]) or "(none named)"
    r = await client.call(system_prompt=_EXTRACT_SYS,
                          user_message=_EXTRACT.format(cushion=cushion[:1100], inherited=inh, n=N_GAPS),
                          domain="exp", concept="extract_gaps", model=MODEL,
                          max_tokens=2000, temperature=0.4)
    gaps = (_parse_json_safely(r.content, default={}) or {}).get("gaps", []) or []
    gaps = [g for g in gaps if isinstance(g, dict) and g.get("gap")][:N_GAPS]

    # ── P1b: skeptically verify gap quality ─────────────────────────────
    gaps_blob = "\n".join(f"{i+1}. {g['gap']}" for i, g in enumerate(gaps))
    rv = await client.call(system_prompt=_VERIFY_SYS,
                           user_message=_VERIFY.format(gaps=gaps_blob),
                           domain="exp", concept="verify_gaps", model=MODEL,
                           max_tokens=2000, temperature=0.2)
    verdicts = (_parse_json_safely(rv.content, default={}) or {}).get("verdicts", []) or []
    scores = [float(v.get("real_gap_score", 0) or 0) for v in verdicts if isinstance(v, dict)]
    gap_quality = sum(scores) / len(scores) if scores else 0.0

    # ── P2: searches — gap-anchored vs present-content control ──────────
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _search(q):
        async with sem:
            try:
                res = await search_chain(q)
            except Exception:
                return []
            return [(h.snippet or h.title or "") for h in (res.hits or [])[:HITS_PER_QUERY]]

    gap_queries = [q for g in gaps for q in (g.get("search_queries") or [])][: N_GAPS * QUERIES_PER_GAP]
    # present-content control: search the dominant inherited works directly
    control_queries = [f"{p}" for p in inherited[:4]] + [cushion[:120]]

    gap_hit_lists = await asyncio.gather(*[_search(q) for q in gap_queries])
    ctl_hit_lists = await asyncio.gather(*[_search(q) for q in control_queries])
    gap_hits = [h for lst in gap_hit_lists for h in lst if h]
    ctl_hits = [h for lst in ctl_hit_lists for h in lst if h]

    # ── embed cluster + hits, measure distance-to-centroid ──────────────
    async def _embed_many(texts):
        out = []
        for t in texts:
            async with sem:
                out.append(await client.embed(t[:1500] or " "))
        return out

    card_vecs = await _embed_many(card_texts)
    centroid = _centroid(card_vecs)
    intra = [1 - _cos(v, centroid) for v in card_vecs]               # cluster's own spread
    gap_vecs = await _embed_many(gap_hits)
    ctl_vecs = await _embed_many(ctl_hits)
    gap_d = [1 - _cos(v, centroid) for v in gap_vecs]
    ctl_d = [1 - _cos(v, centroid) for v in ctl_vecs]

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    elapsed = time.time() - t0
    cost = client.get_total_cost_estimate()

    summary = {
        "run_dir": str(run_dir), "model": MODEL,
        "elapsed_s": round(elapsed, 1), "cost_usd": round(cost, 4),
        "inherited_works": inherited[:12],
        "gaps": [{"gap": g.get("gap"), "why_implied": g.get("why_implied")} for g in gaps],
        "gap_verdicts": verdicts,
        "P1_gap_quality": {
            "mean_real_gap_score": round(gap_quality, 3),
            "verdict": ("STRONG" if gap_quality >= 0.6 else "WEAK" if gap_quality >= 0.4 else "POOR"),
        },
        "P2_off_cluster": {
            "cluster_intra_spread": round(mean(intra), 4),
            "control_dist_to_cluster": round(mean(ctl_d), 4), "control_hits": len(ctl_hits),
            "gap_dist_to_cluster": round(mean(gap_d), 4), "gap_hits": len(gap_hits),
            "gap_pushes_past_adjacency": (mean(gap_d) > mean(ctl_d)) if gap_d and ctl_d else None,
            "margin": round(mean(gap_d) - mean(ctl_d), 4) if gap_d and ctl_d else None,
        },
    }
    out = run_dir / "exp_blend03_gaps.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    printable = {k: v for k, v in summary.items() if k not in ("gap_verdicts",)}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"\nfull results (incl. skeptic verdicts): {out}")
    return summary


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
        sorted((REPO_ROOT / "runs" / "r-collision").glob("*/"))[-1]
    asyncio.run(run(rd))
