"""
Sorter verification engine — gives the master_sorter REAL web search.

WHY THIS EXISTS
---------------
The bare master_sorter is BLIND, not lazy. It classifies cards against
the model's training memory alone. So a card whose bridge maps onto a
paper published AFTER the model's cutoff (e.g. FERMAT, arXiv 2511.14778,
Nov 2025) lands in `unplaced` — not because the sorter is careless, but
because it has never seen the paper. The fix is a TOOL, not a sterner
prompt: give the sorter the same web search the wanderers used.

WHAT IT DOES
------------
Two phases, both fully recorded so the human downstream can audit the
trace:

  Phase 1 — QUERY EXTRACTION (one LLM call)
    For every card, the model reads the bridge claim and emits a handful
    of aggressive search queries: queries that would surface ANY prior
    work performing the same transfer, plus queries that check the card's
    concrete factual claims (dates, attributions, "first to" assertions).

  Phase 2 — SEARCH (concurrent, deterministic)
    Each query runs through the SAME provider chain the wanderers use
    (Exa neural → Tavily → DuckDuckGo, via fetcher.search_chain). Every
    hit (title / url / snippet / provider) is captured against its card.
    Nothing is filtered by predicted quality — the sorter LLM judges the
    evidence in the next stage; this layer just gathers it.

The product is an `EvidenceLedger`: per-card queries + hits, plus totals
and a call log. It is handed to `master_sort(web_evidence=ledger)`, which
switches to its verified doctrine and bins each card against the REAL
evidence instead of memory.

DESIGN NOTES
------------
- search_fn is injectable (defaults to fetcher.search_chain) so tests run
  fully offline with a fake searcher and MOCK LLM client.
- Aggression by construction: if query extraction returns nothing for a
  card (or the extraction call fails entirely), a heuristic fallback query
  built from the card's source_shape + spark is searched anyway. EVERY
  card gets searched — no card escapes verification because the LLM
  hiccuped.
- This module makes ZERO binning decisions. It gathers; the sorter judges;
  the human decides. Three separated responsibilities.

ISOLATION
---------
Imports: dossier card type + cushion + LLMClient + provider_map pricing +
fetcher.search_chain + json helpers. Composes its own system prompt at the
call site (so the identity source-proof scan is satisfied without an
exempt-registry entry). No persistence, no binning.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable

from src.bridge.web_search import SearchResult
from src.identity import compose_system_prompt
from src.llm.client import LLMClient
from src.llm.provider_map import get_pricing
from src.wandering.articulate import ArticulatedCard
from src.wandering.cushion import CushionGraph
from src.wandering.fetcher import search_chain
from src.wandering.master_synthesizer import _parse_json_safely

log = logging.getLogger("constellax.wandering.sorter_verify")


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

#: Model that reads the cards and proposes verification queries. Sonnet
#: 4.6 by default — same seat as the sorter itself. Query extraction is a
#: light reading task; Sonnet is plenty and keeps the lineage uniform.
DEFAULT_QUERY_MODEL = "anthropic/claude-sonnet-4-6"

#: Max search queries the extractor may request per card. Three gives the
#: model room for: (1) the bridge-transfer prior-art query, (2) a factual-
#: claim check, (3) one alternative phrasing — without exploding the search
#: count (20 cards x 3 = 60 searches).
MAX_QUERIES_PER_CARD = 3

#: How many hits to keep per query. The sorter LLM reads these; more than
#: five per query bloats the verdict prompt with diminishing returns.
MAX_HITS_PER_QUERY = 5

#: Concurrent in-flight searches. Bounded so we don't hammer Exa/Tavily.
SEARCH_CONCURRENCY = 5

#: Output cap for the single query-extraction call.
MAX_TOKENS_QUERIES = 4096

#: Low temperature — query extraction is near-deterministic structuring,
#: not creative work. (Dropped for models that reject temperature.)
QUERY_TEMPERATURE = 0.2

#: Truncate each hit snippet to keep the downstream verdict prompt lean.
SNIPPET_CHAR_CAP = 400


# A search function: query string -> SearchResult. Injectable for tests.
SearchFn = Callable[[str], Awaitable[SearchResult]]


# ---------------------------------------------------------------------------
# Evidence dataclasses — the audit trail
# ---------------------------------------------------------------------------


@dataclass
class EvidenceHit:
    """One web result captured for one query."""
    query:    str
    title:    str
    url:      str
    snippet:  str
    provider: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CardEvidence:
    """Everything the verifier gathered for a single card.

    `searched` is True once at least one query ran (even if it returned
    nothing) — it distinguishes "searched, found nothing" (genuine
    non-placement signal) from "never searched" (a plumbing gap). The
    sorter treats the two very differently.
    """
    report_id: str
    queries:   list[str]        = field(default_factory=list)
    hits:      list[EvidenceHit] = field(default_factory=list)
    searched:  bool             = False
    note:      str              = ""

    @property
    def found_anything(self) -> bool:
        return bool(self.hits)

    def to_dict(self) -> dict:
        return {
            "report_id":      self.report_id,
            "queries":        list(self.queries),
            "hits":           [h.to_dict() for h in self.hits],
            "searched":       self.searched,
            "found_anything": self.found_anything,
            "note":           self.note,
        }


@dataclass
class EvidenceLedger:
    """The full verification trace for one sort pass.

    Handed to master_sort(web_evidence=...). Also serialized into the
    dossier so the human reads not just the bins but the EVIDENCE behind
    each placement: which queries ran, what the web returned.
    """
    per_card:            dict[str, CardEvidence] = field(default_factory=dict)
    query_model:         str            = ""
    total_queries:       int            = 0
    total_hits:          int            = 0
    extraction_cost_usd: float          = 0.0
    extraction_ok:       bool           = False
    search_errors:       list[dict]     = field(default_factory=list)
    call_log:            list[dict]     = field(default_factory=list)
    elapsed_ms:          float          = 0.0

    def evidence_for(self, report_id: str) -> CardEvidence | None:
        return self.per_card.get(report_id)

    def to_dict(self) -> dict:
        return {
            "per_card":            {rid: ev.to_dict() for rid, ev in self.per_card.items()},
            "query_model":         self.query_model,
            "total_queries":       self.total_queries,
            "total_hits":          self.total_hits,
            "extraction_cost_usd": round(self.extraction_cost_usd, 4),
            "extraction_ok":       self.extraction_ok,
            "search_errors":       list(self.search_errors),
            "call_log":            list(self.call_log),
            "elapsed_ms":          round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Phase 1 — query extraction
# ---------------------------------------------------------------------------


_QUERY_DOCTRINE = """\
You are a verification scout. Your ONLY job is to write web-search
queries that will let an auditor check whether each card's central
claim already exists on the internet, and whether its factual claims
hold up. You do not judge the cards. You do not bin them. You write
the search queries an aggressive fact-checker would run.

Each card has this shape:
  - spark        — the seed observation (often names a known concept)
  - source_shape — the source domain the analogy is drawn FROM
  - bridge       — THE CENTRAL TRANSFER CLAIM: how the source maps onto
                   the target problem. This is the thing to verify.
  - use          — the recommended action
  - limit        — where the analogy breaks

For EACH card, write up to %(max_q)d search queries. Spend them well:

  1. PRIOR-ART query — phrase a search that would surface any existing
     paper, framework, or theory that performs THE SAME TRANSFER the
     bridge asserts. Search the structural claim, not just the buzzword.
     Bad:  "Markov chains"            (too broad — the concept, not the claim)
     Good: "Markov chain model of conversational trust decay"   (the transfer)

  2. FACTUAL-CHECK query — if the card asserts any checkable fact (a
     date, an attribution, a "first to do X", a named result), write a
     query that would confirm or refute it.

  3. ALTERNATE-PHRASING query — one more angle on the prior-art search,
     worded differently, so a real match isn't missed on vocabulary alone.

Be aggressive and specific. Vague queries waste the search. If a card
makes no checkable factual claim, skip query 2 and use the slot for
another prior-art angle.

OUTPUT FORMAT — a single JSON object, nothing around it:
{
  "queries": {
    "<report_id>": ["query one", "query two", "query three"],
    "<report_id>": ["query one", "query two"]
  }
}

Every input card's report_id MUST appear as a key. Output ONLY the JSON.
"""


def _build_query_payload(
    cushion: CushionGraph | None,
    cards:   list[ArticulatedCard],
) -> str:
    """User-message payload for the query-extraction call."""
    import json as _json

    problem = ""
    if cushion is not None and cushion.raw_input is not None:
        problem = cushion.raw_input.problem.content[:500]

    card_blocks = [
        {
            "report_id":    c.report_id,
            "spark":        c.spark,
            "source_shape": c.source_shape,
            "bridge":       c.bridge,
            "use":          c.use,
            "limit":        c.limit,
        }
        for c in cards
    ]
    payload = {
        "problem_context": problem,
        "card_count":      len(cards),
        "cards":           card_blocks,
        "instruction": (
            "Write verification search queries for EVERY card. Output the "
            "JSON object only — no prose around it."
        ),
    }
    return _json.dumps(payload, ensure_ascii=False, indent=2)


def _heuristic_query(card: ArticulatedCard) -> str:
    """Fallback query when the extractor gives a card nothing.

    Built from the source domain + spark so the card still gets searched.
    Aggression guarantee: no card escapes verification on an LLM hiccup.
    """
    parts = [p.strip() for p in (card.source_shape, card.spark) if p and p.strip()]
    q = " ".join(parts)[:200]
    return q or (card.bridge or "")[:200]


async def extract_verification_queries(
    *,
    cushion: CushionGraph | None,
    cards:   list[ArticulatedCard],
    client:  LLMClient,
    model:   str,
    ledger:  EvidenceLedger,
    max_queries_per_card: int = MAX_QUERIES_PER_CARD,
) -> dict[str, list[str]]:
    """Phase 1: ask the model for per-card verification queries.

    Returns {report_id: [query, ...]}. On any failure (LLM error, bad
    JSON, missing keys) the affected cards simply get no queries here —
    the caller fills them with `_heuristic_query`. Records cost + a call
    log entry on `ledger`.
    """
    system = compose_system_prompt(
        _QUERY_DOCTRINE % {"max_q": max_queries_per_card},
        mode="master_sorter",
    )
    payload = _build_query_payload(cushion, cards)

    t0 = time.time()
    response = await client.call(
        system_prompt=system,
        user_message=payload,
        domain="master_sorter",
        concept="verify_queries",
        model=model,
        max_tokens=MAX_TOKENS_QUERIES,
        temperature=QUERY_TEMPERATURE,
    )
    elapsed = (time.time() - t0) * 1000

    in_price, out_price = get_pricing(model)
    cost = (
        (response.input_tokens  or 0) / 1_000_000 * in_price
        + (response.output_tokens or 0) / 1_000_000 * out_price
    )
    ledger.extraction_cost_usd += cost
    ledger.call_log.append({
        "phase":    "extract_queries",
        "model":    model,
        "in_tok":   response.input_tokens,
        "out_tok":  response.output_tokens,
        "cost_usd": round(cost, 4),
        "ms":       round(elapsed, 1),
        "ok":       response.success,
        "err":      (response.error or "")[:200] if not response.success else "",
    })

    if not response.success:
        log.warning("query extraction call failed: %s", response.error)
        return {}

    parsed = _parse_json_safely(response.content, default={})
    if not isinstance(parsed, dict):
        return {}
    raw_map = parsed.get("queries", {})
    if not isinstance(raw_map, dict):
        return {}

    valid_ids = {c.report_id for c in cards}
    out: dict[str, list[str]] = {}
    for rid, qlist in raw_map.items():
        rid = str(rid)
        if rid not in valid_ids or not isinstance(qlist, list):
            continue
        cleaned = [str(q).strip() for q in qlist if str(q).strip()][:max_queries_per_card]
        if cleaned:
            out[rid] = cleaned
    ledger.extraction_ok = True
    return out


# ---------------------------------------------------------------------------
# Phase 2 — search
# ---------------------------------------------------------------------------


async def _run_one_search(
    report_id: str,
    query:     str,
    search_fn: SearchFn,
    sem:       asyncio.Semaphore,
    hits_per_query: int,
) -> tuple[str, str, list[EvidenceHit], str | None]:
    """Run a single query. Returns (report_id, query, hits, error)."""
    async with sem:
        try:
            result: SearchResult = await search_fn(query)
        except Exception as e:  # pragma: no cover — search_chain is itself guarded
            return report_id, query, [], f"{type(e).__name__}: {e}"

    hits: list[EvidenceHit] = []
    for h in (result.hits or [])[:hits_per_query]:
        snippet = (h.snippet or "").strip()
        if len(snippet) > SNIPPET_CHAR_CAP:
            snippet = snippet[:SNIPPET_CHAR_CAP] + "…"
        hits.append(EvidenceHit(
            query=query,
            title=(h.title or "").strip(),
            url=(h.url or "").strip(),
            snippet=snippet,
            provider=result.provider or "",
        ))
    return report_id, query, hits, (result.error if not result.hits else None)


async def gather_evidence(
    *,
    cushion:   CushionGraph | None,
    cards:     list[ArticulatedCard],
    client:    LLMClient,
    query_model: str = DEFAULT_QUERY_MODEL,
    max_queries_per_card: int = MAX_QUERIES_PER_CARD,
    hits_per_query: int = MAX_HITS_PER_QUERY,
    concurrency: int = SEARCH_CONCURRENCY,
    search_fn: SearchFn | None = None,
) -> EvidenceLedger:
    """Gather web evidence for every card. Phase 1 (extract) + Phase 2
    (search), fully recorded.

    Empty input → empty ledger (no calls). Every returned card has a
    CardEvidence entry; `searched=True` once at least one query ran for
    it. Never raises — search failures are recorded on the ledger.
    """
    ledger = EvidenceLedger(query_model=query_model)
    if not cards:
        return ledger

    search_fn = search_fn or search_chain
    t_start = time.time()

    # Phase 1 — propose queries (best-effort; failures fall back below).
    try:
        query_map = await extract_verification_queries(
            cushion=cushion,
            cards=cards,
            client=client,
            model=query_model,
            ledger=ledger,
            max_queries_per_card=max_queries_per_card,
        )
    except Exception as e:
        log.warning("query extraction raised, falling back to heuristics: %s", e)
        query_map = {}

    # Seed per-card evidence; fill missing cards with a heuristic query so
    # EVERY card is searched.
    sem = asyncio.Semaphore(max(1, concurrency))
    tasks = []
    for c in cards:
        ev = CardEvidence(report_id=c.report_id)
        queries = query_map.get(c.report_id) or [_heuristic_query(c)]
        queries = [q for q in queries if q][:max_queries_per_card]
        if not query_map.get(c.report_id):
            ev.note = "heuristic fallback query (extractor gave none)"
        ev.queries = queries
        ledger.per_card[c.report_id] = ev
        for q in queries:
            tasks.append(_run_one_search(c.report_id, q, search_fn, sem, hits_per_query))

    # Phase 2 — run all searches concurrently.
    results = await asyncio.gather(*tasks) if tasks else []
    for report_id, query, hits, error in results:
        ev = ledger.per_card.get(report_id)
        if ev is None:  # pragma: no cover — every report_id was seeded above
            continue
        ev.searched = True
        ledger.total_queries += 1
        if hits:
            ev.hits.extend(hits)
            ledger.total_hits += len(hits)
        if error:
            ledger.search_errors.append({"report_id": report_id, "query": query, "error": error})

    ledger.elapsed_ms = (time.time() - t_start) * 1000
    log.info(
        "[sorter_verify] gathered evidence: %d cards, %d queries, %d hits, %d search errors, %.0fms",
        len(cards), ledger.total_queries, ledger.total_hits,
        len(ledger.search_errors), ledger.elapsed_ms,
    )
    return ledger


__all__ = (
    "EvidenceHit",
    "CardEvidence",
    "EvidenceLedger",
    "SearchFn",
    "DEFAULT_QUERY_MODEL",
    "MAX_QUERIES_PER_CARD",
    "MAX_HITS_PER_QUERY",
    "extract_verification_queries",
    "gather_evidence",
)
