"""
Blend verification — the sorter's SECOND seat (pipeline stage 5).

The sorter sits in two places. In stage 2 it verifies the raw wander cards
("is this real?"). Here, in stage 5, it verifies the BLENDER's output with
the same aggressive web search, but it asks a different question:

    Does this blend already exist — even partially, even something that
    merely RESEMBLES it?

Because assuming a blend is original just because WE made it is the fool's
error. Only after an aggressive sweep finds nothing resembling it does a
blend earn the novelty bin.

THE FOUR BINS (for blends)
--------------------------
- known    — the whole blended concept already exists on the web. Cite it.
- adjacent — the 4th bin. Components, or a close cousin, already exist, but
             not this exact combination. The blend is NOT fully novel; name
             what it resembles AND what part still looks new.
- novel    — the aggressive sweep found nothing that does this. Genuine
             candidate gold. The bar is HIGH: any real resemblance demotes
             it to `adjacent`.
- flawed   — the evidence contradicts a factual premise the blend rests on
             (rare; coherence is mostly vetted upstream by the blender +
             drift-checker, but a blend can still rest on a bad fact).

Reuses the same search machinery as stage 2 (sorter_verify: Exa/Tavily/DDG
via search_chain, EvidenceLedger trail). Both LLM calls compose at the call
site, so no identity-exempt entries are needed.

ISOLATION
---------
Imports cushion + Blend type + the sorter_verify search primitives +
LLMClient + pricing + json helpers. Makes web calls (verification) but NO
blends and NO bin decisions of its own beyond classification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum

from src.identity import compose_system_prompt
from src.llm.client import LLMClient
from src.llm.provider_map import get_pricing
from src.wandering.blender import Blend
from src.wandering.cushion import CushionGraph
from src.wandering.fetcher import search_chain
from src.wandering.master_synthesizer import _parse_json_safely
from src.wandering.sorter_verify import (
    MAX_HITS_PER_QUERY,
    SEARCH_CONCURRENCY,
    CardEvidence,
    EvidenceLedger,
    SearchFn,
    _run_one_search,
)

log = logging.getLogger("constellax.wandering.blend_verify")


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

DEFAULT_QUERY_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_VERIFY_MODEL = "anthropic/claude-sonnet-4-6"

MAX_QUERIES_PER_BLEND = 3
MAX_TOKENS_QUERIES = 4096
MAX_TOKENS_VERIFY = 8192
QUERY_TEMPERATURE = 0.2
VERIFY_TEMPERATURE = 0.1

DEFAULT_COST_CEILING_USD = 8.00

BLEND_VERIFY_DOMAIN = "blend_verify"


class BlendBin(str, Enum):
    KNOWN    = "known"
    ADJACENT = "adjacent"
    NOVEL    = "novel"
    FLAWED   = "flawed"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VerifiedBlend:
    """One blend's verification verdict, grounded in web evidence."""
    blend_id:    str
    bin:         str                  # BlendBin value
    references:  list[dict] = field(default_factory=list)  # [{title,url}] from the evidence
    resemblance: str        = ""      # what it matches (known) / resembles (adjacent)
    still_new:   str        = ""      # for adjacent: the part that still looks new
    reasoning:   str        = ""
    confidence:  float      = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlendVerificationReport:
    known:               list[VerifiedBlend] = field(default_factory=list)
    adjacent:            list[VerifiedBlend] = field(default_factory=list)
    novel:               list[VerifiedBlend] = field(default_factory=list)
    flawed:              list[VerifiedBlend] = field(default_factory=list)
    evidence:            EvidenceLedger | None = None
    total_cost_usd:      float       = 0.0
    cost_ceiling_usd:    float       = DEFAULT_COST_CEILING_USD
    truncated_by_budget: bool        = False
    parser_notes:        list[dict]  = field(default_factory=list)
    call_log:            list[dict]  = field(default_factory=list)
    input_blend_count:   int         = 0
    verify_model:        str         = ""

    def to_dict(self) -> dict:
        return {
            "known":               [b.to_dict() for b in self.known],
            "adjacent":            [b.to_dict() for b in self.adjacent],
            "novel":               [b.to_dict() for b in self.novel],
            "flawed":              [b.to_dict() for b in self.flawed],
            "evidence":            self.evidence.to_dict() if self.evidence is not None else None,
            "total_cost_usd":      round(self.total_cost_usd, 4),
            "cost_ceiling_usd":    self.cost_ceiling_usd,
            "truncated_by_budget": self.truncated_by_budget,
            "parser_notes":        list(self.parser_notes),
            "call_log":            list(self.call_log),
            "input_blend_count":   self.input_blend_count,
            "verify_model":        self.verify_model,
        }


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    in_price, out_price = get_pricing(model)
    return (in_tok or 0) / 1_000_000 * in_price + (out_tok or 0) / 1_000_000 * out_price


# ---------------------------------------------------------------------------
# Phase 1 — query extraction (blend-shaped)
# ---------------------------------------------------------------------------


_BLEND_QUERY_DOCTRINE = """\
You are a verification scout. For each BLEND (a candidate new concept built
by colliding research cards), write web-search queries that would reveal
whether this concept — or anything that RESEMBLES it — already exists.

Each blend has:
  - thesis: the new concept as a claim
  - mechanism: how it works
  - emergent_structure: what's claimed to be new (in neither source card)

For EACH blend, write up to %(max_q)d aggressive queries:
  1. EXACT-MATCH query — search the blend's specific mechanism/thesis as if
     looking for a paper that already proposes exactly this.
  2. RESEMBLANCE query — search the nearest neighbor: a method that does
     something structurally similar, even in another field. We want to catch
     "someone already did a cousin of this."
  3. COMPONENT query — search whether the emergent_structure (the claimed-new
     part) exists on its own anywhere.

Be specific. Search the STRUCTURE, not the buzzwords. Output a single JSON
object, nothing around it:
{ "queries": { "<blend_id>": ["q1","q2","q3"], ... } }
Every blend_id MUST appear. Output ONLY the JSON.
"""


async def _extract_blend_queries(
    *, blends: list[Blend], client: LLMClient, model: str, ledger: EvidenceLedger,
    max_q: int,
) -> dict[str, list[str]]:
    system = compose_system_prompt(_BLEND_QUERY_DOCTRINE % {"max_q": max_q}, mode="blend_verify")
    payload = json.dumps({
        "blend_count": len(blends),
        "blends": [
            {"blend_id": b.blend_id, "thesis": b.thesis,
             "mechanism": b.mechanism, "emergent_structure": b.emergent_structure}
            for b in blends
        ],
        "instruction": "Write resemblance-hunting queries for EVERY blend. JSON only.",
    }, ensure_ascii=False, indent=2)

    t0 = time.time()
    resp = await client.call(
        system_prompt=system, user_message=payload,
        domain=BLEND_VERIFY_DOMAIN, concept="verify_queries",
        model=model, max_tokens=MAX_TOKENS_QUERIES, temperature=QUERY_TEMPERATURE,
    )
    cost = _cost(model, resp.input_tokens, resp.output_tokens)
    ledger.extraction_cost_usd += cost
    ledger.call_log.append({
        "phase": "blend_extract_queries", "model": model,
        "in_tok": resp.input_tokens, "out_tok": resp.output_tokens,
        "cost_usd": round(cost, 4), "ms": round((time.time() - t0) * 1000, 1),
        "ok": resp.success, "err": (resp.error or "")[:200] if not resp.success else "",
    })
    if not resp.success:
        return {}
    parsed = _parse_json_safely(resp.content, default={})
    if not isinstance(parsed, dict):
        return {}
    raw_map = parsed.get("queries", {})
    if not isinstance(raw_map, dict):
        return {}
    valid = {b.blend_id for b in blends}
    out: dict[str, list[str]] = {}
    for bid, qlist in raw_map.items():
        bid = str(bid)
        if bid not in valid or not isinstance(qlist, list):
            continue
        cleaned = [str(q).strip() for q in qlist if str(q).strip()][:max_q]
        if cleaned:
            out[bid] = cleaned
    ledger.extraction_ok = True
    return out


def _heuristic_blend_query(b: Blend) -> str:
    return (b.thesis or b.emergent_structure or b.mechanism or "")[:200]


async def gather_blend_evidence(
    *, blends: list[Blend], client: LLMClient,
    query_model: str = DEFAULT_QUERY_MODEL,
    max_queries_per_blend: int = MAX_QUERIES_PER_BLEND,
    hits_per_query: int = MAX_HITS_PER_QUERY,
    concurrency: int = SEARCH_CONCURRENCY,
    search_fn: SearchFn | None = None,
) -> EvidenceLedger:
    """Gather resemblance evidence for every blend. Same engine as the raw-
    card verifier, blend-shaped queries. Every blend is searched (heuristic
    fallback guarantees it). Never raises."""
    ledger = EvidenceLedger(query_model=query_model)
    if not blends:
        return ledger
    search_fn = search_fn or search_chain
    t_start = time.time()

    try:
        query_map = await _extract_blend_queries(
            blends=blends, client=client, model=query_model, ledger=ledger,
            max_q=max_queries_per_blend,
        )
    except Exception as e:
        log.warning("blend query extraction raised, using heuristics: %s", e)
        query_map = {}

    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = []
    for b in blends:
        ev = CardEvidence(report_id=b.blend_id)
        queries = query_map.get(b.blend_id) or [_heuristic_blend_query(b)]
        queries = [q for q in queries if q][:max_queries_per_blend]
        if not query_map.get(b.blend_id):
            ev.note = "heuristic fallback query (extractor gave none)"
        ev.queries = queries
        ledger.per_card[b.blend_id] = ev
        for q in queries:
            tasks.append(_run_one_search(b.blend_id, q, search_fn, sem, hits_per_query))

    results = await asyncio.gather(*tasks) if tasks else []
    for blend_id, query, hits, error in results:
        ev = ledger.per_card.get(blend_id)
        if ev is None:
            continue
        ev.searched = True
        ledger.total_queries += 1
        if hits:
            ev.hits.extend(hits)
            ledger.total_hits += len(hits)
        if error:
            ledger.search_errors.append({"blend_id": blend_id, "query": query, "error": error})

    ledger.elapsed_ms = (time.time() - t_start) * 1000
    log.info("[blend_verify] evidence: %d blends, %d queries, %d hits, %d errors, %.0fms",
             len(blends), ledger.total_queries, ledger.total_hits, len(ledger.search_errors), ledger.elapsed_ms)
    return ledger


# ---------------------------------------------------------------------------
# Phase 2 — the 4-bin verdict
# ---------------------------------------------------------------------------


_BLEND_VERIFY_DOCTRINE = """\
You are a novelty auditor, and your discipline is BALANCE. Tension and
friction are what give birth to new theory — so your job is to find where a
blend genuinely departs from the known, NOT to smooth that departure away
for the comfort of certainty. Two failures are equally bad, and you must
avoid BOTH:

  - TOO SMOOTH (over-certain): collapsing a genuinely new move into "already
    known/adjacent" because its PARTS resemble existing work. Every new idea
    is assembled from existing components — if shared components were enough
    to deny novelty, NOTHING could ever be new. Reaching for "this basically
    exists" to resolve the discomfort of "this might be new" is the cardinal
    sin of this role.
  - TOO CHAOTIC (over-uncertain): calling a blend novel when the evidence
    actually shows the SAME move already exists. Inflating novelty without
    grounding is equally a failure.

Novelty lives in the TRANSFER, not the components. The question is never
"do the parts exist somewhere?" — they always do. The question is: does any
hit perform THIS SPECIFIC MOVE — the same transfer, the same combination,
applied the same way?

The friction between "the parts resemble prior work" AND "the move itself is
unprecedented" is the SIGNAL, not noise to be resolved. Hold it. Weigh it
honestly. Report both sides — and let the bin reflect the MOVE, not the
ingredients.

For each blend you have REAL web hits (evidence: queries run + results). Bin
against the EVIDENCE using the same-MOVE test below.

KNOWN — a hit performs essentially THIS SAME move: same transfer, same
  combination, same application. The concept already exists; someone built
  it. Cite the hit's url in `references`, name what it duplicates in
  `resemblance`.

ADJACENT — a hit performs a genuinely CLOSE COUSIN of the same move: not
  merely sharing a component or a domain, but doing nearly the same transfer,
  such that this blend is a SMALL STEP from it. Name the cousin (with url) in
  `resemblance` AND the real difference in `still_new`. ADJACENT requires
  same-transfer PROXIMITY — component overlap alone does NOT qualify.

NOVEL — components may well exist (they always do), but NO hit performs THIS
  transfer / THIS combination / THIS application. The specific MOVE is
  unprecedented in the evidence. This is candidate gold. Be honest: name the
  related-but-different work in `resemblance` (what's nearby) AND the
  unprecedented move in `still_new`. A blend whose ingredients appear in
  other domains but whose central transfer no hit performs IS NOVEL — that
  is precisely the friction you must NOT smooth away. Thin/empty hits → NOVEL
  with a note that evidence was thin.

FLAWED — the evidence CONTRADICTS a factual premise the blend rests on (a
  named method that doesn't work as claimed, a false attribution). Name the
  contradiction in `reasoning`. Sparingly — coherence was vetted upstream.

THE TEST that separates ADJACENT from NOVEL — apply it to every blend:
  "Does a hit do THE SAME MOVE, or only use the same INGREDIENTS?"
    same move / near-twin       → KNOWN or ADJACENT
    same ingredients, new move  → NOVEL
  Component resemblance in an adjacent domain is NOT grounds to deny novelty.
  Equally, do not call it novel if a hit truly performs this move.

RULES:
  - Every `references` url must be COPIED from the evidence hits. Never invent.
  - Do NOT default to any bin. EARN each placement from the same-move test —
    including NOVEL when the evidence earns it.
  - One blend, one bin.

OUTPUT FORMAT: a single JSON object with one array `verdicts`, each entry in
the schema in the user message. Output ONLY the JSON.
"""


def _build_verify_payload(cushion, blends: list[Blend], evidence: EvidenceLedger) -> str:
    problem = ""
    if cushion is not None and getattr(cushion, "raw_input", None) is not None:
        problem = cushion.raw_input.problem.content[:600]

    blocks = []
    for b in blends:
        ev = evidence.evidence_for(b.blend_id)
        ev_block = None
        if ev is not None:
            ev_block = {
                "queries_run": list(ev.queries),
                "hits": [{"title": h.title, "url": h.url, "snippet": h.snippet} for h in ev.hits],
            }
        blocks.append({
            "blend_id": b.blend_id,
            "thesis": b.thesis,
            "mechanism": b.mechanism,
            "emergent_structure": b.emergent_structure,
            "evidence": ev_block,
        })

    schema = {"verdicts": [{
        "blend_id":    "<copy>",
        "bin":         "<known|adjacent|novel|flawed>",
        "references":  [{"title": "<from a hit>", "url": "<COPIED from a hit>"}],
        "resemblance": "<what it matches/resembles; '' for novel>",
        "still_new":   "<for adjacent: the part still unprecedented; else ''>",
        "reasoning":   "<one or two sentences grounded in the evidence>",
        "confidence":  "<float 0..1>",
    }]}
    return json.dumps({
        "cushion_problem": problem,
        "blend_count": len(blends),
        "blends": blocks,
        "output_schema": schema,
        "instruction": (
            "Bin every blend against its evidence using the same-MOVE test: a "
            "blend is KNOWN/ADJACENT only if a hit performs the same transfer, "
            "not merely shares components. Components existing in other domains "
            "does NOT deny novelty — if no hit performs THIS move, it is NOVEL. "
            "Hold the friction; do not smooth it away. references must be copied "
            "from hits. JSON only."
        ),
    }, ensure_ascii=False, indent=2)


def _parse_verify(raw: str, blends: list[Blend], report: BlendVerificationReport) -> None:
    by_id = {b.blend_id for b in blends}
    parsed = _parse_json_safely(raw, default={})
    if not isinstance(parsed, dict):
        report.parser_notes.append({"reason": "top_level_not_dict"})
        return
    raw_v = parsed.get("verdicts", []) or []
    if not isinstance(raw_v, list):
        report.parser_notes.append({"reason": "verdicts_not_list"})
        return

    bins = {
        "known": report.known, "adjacent": report.adjacent,
        "novel": report.novel, "flawed": report.flawed,
    }
    seen: set[str] = set()
    for entry in raw_v:
        if not isinstance(entry, dict):
            continue
        bid = str(entry.get("blend_id", ""))
        if bid not in by_id or bid in seen:
            report.parser_notes.append({"reason": "unknown_or_dup_blend_id", "blend_id": bid})
            continue
        bin_name = str(entry.get("bin", "")).strip().lower()
        if bin_name not in bins:
            report.parser_notes.append({"reason": "bad_bin_defaulted_adjacent", "blend_id": bid, "bin": bin_name})
            bin_name = "adjacent"

        refs_raw = entry.get("references", []) or []
        refs = [
            {"title": str(r.get("title", "")), "url": str(r.get("url", ""))}
            for r in refs_raw if isinstance(r, dict)
        ]
        # known/adjacent claim prior art → must cite an evidence url, else demote to novel-with-note
        if bin_name in ("known", "adjacent") and not any(r["url"] for r in refs):
            report.parser_notes.append({"reason": "claim_without_reference_demoted_to_novel", "blend_id": bid, "claimed_bin": bin_name})
            bin_name = "novel"

        vb = VerifiedBlend(
            blend_id=bid,
            bin=bin_name,
            references=refs,
            resemblance=str(entry.get("resemblance", "")),
            still_new=str(entry.get("still_new", "")),
            reasoning=str(entry.get("reasoning", "")),
            confidence=float(entry.get("confidence", 0.0) or 0.0),
        )
        bins[bin_name].append(vb)
        seen.add(bid)

    # Any blend the verifier skipped → record it (don't silently lose it).
    for b in blends:
        if b.blend_id not in seen:
            report.parser_notes.append({"reason": "blend_unverified", "blend_id": b.blend_id})


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def verify_blends(
    *,
    cushion:          CushionGraph | None,
    blends:           list[Blend],
    client:           LLMClient,
    query_model:      str = DEFAULT_QUERY_MODEL,
    verify_model:     str = DEFAULT_VERIFY_MODEL,
    search_fn:        SearchFn | None = None,
    cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
) -> BlendVerificationReport:
    """Verify blends against the live web into known / adjacent / novel / flawed.

    Gathers resemblance evidence (Exa/Tavily/DDG), then a single sorter pass
    bins each blend against it. Empty input → empty report (no calls).
    """
    report = BlendVerificationReport(cost_ceiling_usd=cost_ceiling_usd, verify_model=verify_model)
    report.input_blend_count = len(blends)
    if not blends:
        return report

    evidence = await gather_blend_evidence(
        blends=blends, client=client, query_model=query_model, search_fn=search_fn,
    )
    report.evidence = evidence
    report.total_cost_usd += evidence.extraction_cost_usd

    system = compose_system_prompt(_BLEND_VERIFY_DOCTRINE, mode="blend_verify")
    payload = _build_verify_payload(cushion, blends, evidence)

    t0 = time.time()
    resp = await client.call(
        system_prompt=system, user_message=payload,
        domain=BLEND_VERIFY_DOMAIN, concept="verdict",
        model=verify_model, max_tokens=MAX_TOKENS_VERIFY, temperature=VERIFY_TEMPERATURE,
    )
    cost = _cost(verify_model, resp.input_tokens, resp.output_tokens)
    report.total_cost_usd += cost
    report.call_log.append({
        "phase": "blend_verdict", "model": verify_model,
        "in_tok": resp.input_tokens, "out_tok": resp.output_tokens,
        "cost_usd": round(cost, 4), "ms": round((time.time() - t0) * 1000, 1),
        "ok": resp.success, "err": (resp.error or "")[:200] if not resp.success else "",
    })
    if report.total_cost_usd > cost_ceiling_usd:
        report.truncated_by_budget = True

    if resp.success:
        _parse_verify(resp.content, blends, report)
    else:
        report.parser_notes.append({"reason": "verdict_call_failed", "err": (resp.error or "")[:200]})
    return report


__all__ = (
    "DEFAULT_QUERY_MODEL",
    "DEFAULT_VERIFY_MODEL",
    "BlendBin",
    "VerifiedBlend",
    "BlendVerificationReport",
    "gather_blend_evidence",
    "verify_blends",
)
