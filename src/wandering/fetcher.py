"""
Real fetcher adapter — multi-provider, dedup-aware, trust-tiebreaker.

Wandering Room's tier-1 fetcher. One call = one content unit handed to
the matcher. Behavior:

  1. Honor the session's follow-on queue. If the agent's session has a
     queued URL (from a prior tier-2 read or Exa.findSimilar hop), use
     that as the seed instead of running a fresh search.

  2. Otherwise run a domain-biased search. Provider chain: Exa neural
     search (when EXA_API_KEY is set) → Tavily (when TAVILY_API_KEY is
     set) → DuckDuckGo HTML (free fallback).

  3. From the search result, pick the first hit whose URL has NOT been
     visited by any agent in this session. Mark it visited before
     returning so concurrent agents skip it.

  4. Stitch the hit's snippet (or full Exa text excerpt) into a
     FetchResult body. Tier-2 escalation, if it fires, will replace
     this body with a Jina Reader full-page extract — handled in
     agent.py, not here.

Per Law 4: read-only. No writes outside `session_state` (which is
in-memory per-wander). Per Law 1: NO ranking/filtering by predicted
quality. Dedup is anti-waste, not anti-chaos — we skip URLs we've already
read, not URLs we'd "rather not read." Trust scoring is a TIE-BREAKER
only — applied at confidence time, never at fetch time.

ISOLATION: imports web_search bridge (Tavily/DDG), exa_provider, and
session_state. No persistence, no LLM, no agent-loop knowledge.
"""

from __future__ import annotations

import logging
from typing import Any

from src.bridge.web_search import SearchResult, web_search
from src.wandering.agent import FetchResult
from src.wandering import exa_provider
from src.wandering.session_state import (
    FollowonItem,
    SessionState,
    normalize_url,
)


log = logging.getLogger("constellax.wandering.fetcher")


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


#: How many search hits to use when stitching together "content" for one
#: fetch call. Wandering Room treats one fetch as one content unit (which
#: the matcher analyzes). We collect a few top hits and bundle them so
#: the structural match has enough surface to evaluate.
DEFAULT_HITS_PER_FETCH = 3

#: Cap on body length per fetch. Wandering Room matches against structural
#: patterns; we don't need raw HTML dumps. Truncate to keep token costs sane.
DEFAULT_BODY_CHAR_CAP = 2500

#: How many hits to inspect when looking for an unvisited URL.
#: A search returns ~5 hits; we walk past visited ones until we find a
#: novel candidate. If all hits in a search are stale we accept the
#: top hit anyway (the matcher will probably no-match and the agent moves on).
DEDUP_LOOKBACK_HITS = 5

#: Exa: how many results to request for neural search at tier-1. Five is
#: the same as Tavily's default and matches our hit-stitching width.
EXA_NUM_RESULTS = 5


# ---------------------------------------------------------------------------
# Query helpers (unchanged from prior version, comments updated)
# ---------------------------------------------------------------------------


def _build_query_for_domain(domain: str, anchor_summary: str) -> str:
    """Build a search query that biases the search toward the agent's
    chosen domain while keeping the anchor visible.

    Per Law 1: we do NOT optimize the query to "find good matches" —
    the anchor goes in raw and the domain is appended as a hint. The
    chaos lives in WHERE we look (domain) not in HOW we phrase the query.
    """
    domain_clean = domain.replace("_", " ").strip()
    anchor_clean = anchor_summary.strip()[:200]
    if not anchor_clean:
        return domain_clean
    if not domain_clean:
        return anchor_clean
    return f"{anchor_clean} {domain_clean}"


# ---------------------------------------------------------------------------
# Tavily / DDG (existing path, lightly adapted)
# ---------------------------------------------------------------------------


def _pick_unvisited_hit(
    result: SearchResult,
    session_state: SessionState | None,
) -> tuple[int, str]:
    """Walk the search hits and return (index, url) of the first one we
    haven't visited this session.

    If `session_state` is None (callers running outside a session, e.g.
    legacy tests), behave as before — always pick hit[0].

    Returns (-1, "") if the result has no hits or all hits are exhausted.
    """
    if not result.hits:
        return -1, ""

    if session_state is None:
        return 0, result.hits[0].url or ""

    for i, hit in enumerate(result.hits[:DEDUP_LOOKBACK_HITS]):
        url = (hit.url or "").strip()
        if not url:
            continue
        if not session_state.has_visited(url):
            return i, url

    # All hits visited — fall back to the top hit anyway. The agent's
    # matcher will likely no-match (or match again on cached content,
    # which is fine — the LLM call is cheap). The trace records the
    # repeat so audits can see it.
    log.debug(
        "fetcher: all %d hits already visited this session; using top",
        len(result.hits[:DEDUP_LOOKBACK_HITS]),
    )
    return 0, result.hits[0].url or ""


def _stitch_from_hit(
    result: SearchResult,
    pick_index: int,
    char_cap: int = DEFAULT_BODY_CHAR_CAP,
) -> tuple[str, str, str]:
    """Combine search results around the picked hit into (title, url, body).

    Title = picked hit's title (or "Search: <query>").
    URL = picked hit's URL.
    Body = picked hit's snippet first, then up to DEFAULT_HITS_PER_FETCH-1
    additional snippets from neighbouring hits for structural context.
    """
    if not result.hits:
        return (f"Search: {result.query}", "", "(no results)")

    pick_index = max(0, min(pick_index, len(result.hits) - 1))
    primary = result.hits[pick_index]
    title = primary.title or f"Search: {result.query}"
    url = primary.url or ""

    pieces: list[str] = []
    # Primary hit first, then up to two more for surrounding context.
    ordered = [primary] + [
        h for i, h in enumerate(result.hits) if i != pick_index
    ][: DEFAULT_HITS_PER_FETCH - 1]
    for h in ordered:
        snippet = (h.snippet or "").strip()
        if not snippet:
            continue
        piece = f"[{h.title or 'untitled'}]\n{snippet}"
        if h.url:
            piece += f"\n(source: {h.url})"
        pieces.append(piece)

    body = "\n\n---\n\n".join(pieces)
    if len(body) > char_cap:
        body = body[:char_cap] + "...[truncated]"
    return title, url, body


# ---------------------------------------------------------------------------
# Exa first → Tavily/DDG fallback
# ---------------------------------------------------------------------------


def _exa_hits_to_search_result(
    exa_result: exa_provider.ExaResult,
) -> SearchResult:
    """Adapter — re-shape Exa's hits into a SearchResult so the rest of
    the fetcher pipeline (dedup, stitch) treats Exa output the same way.

    We use snippet from Exa's `text` field (their excerpt). Score is
    discarded at this layer — confidence assignment uses trust.py
    instead, which is uniform across providers.
    """
    from src.bridge.web_search import SearchHit

    hits = [
        SearchHit(title=h.title, snippet=h.text, url=h.url)
        for h in exa_result.hits
    ]
    return SearchResult(
        query=exa_result.query_or_url,
        hits=hits,
        provider="exa-neural",
        latency_ms=exa_result.latency_ms,
        error=exa_result.error,
    )


async def _search_with_chain(
    query: str,
) -> SearchResult:
    """Run the provider chain: Exa neural → Tavily/DDG via web_search.

    Returns the FIRST result that has hits. On total failure returns the
    last SearchResult (probably DDG's, with `error` set) — caller treats
    that as "no usable result."

    Exa is the wandering differentiator (neural search → cross-domain
    embedding-similar pages). Tavily is the well-rounded lexical baseline.
    DDG is the free key-less floor.
    """
    if exa_provider.is_available():
        exa_result = await exa_provider.search(
            query, num_results=EXA_NUM_RESULTS,
        )
        if exa_result.ok:
            return _exa_hits_to_search_result(exa_result)
        log.info("exa returned no hits (error=%s), falling through", exa_result.error)

    return await web_search(query)


# ---------------------------------------------------------------------------
# Follow-on queue path
# ---------------------------------------------------------------------------


async def _fetch_from_followon(
    item: FollowonItem,
    domain_hint: str,
) -> FetchResult:
    """Materialise a FollowonItem into a FetchResult.

    The follow-on queue holds URLs the agent (or a prior agent in the
    session) has already deemed promising. We don't run a new search —
    we go straight to the URL via tier-2 extraction (Jina Reader) so the
    matcher sees a richer body than a stitched snippet.

    On Jina failure, return a thin placeholder so the matcher still has
    something to score (it'll likely no-match, agent moves on). Per Law
    1 we don't try to be clever about retries.
    """
    from src.wandering.extractors import extract_url

    extract = await extract_url(item.url)
    if extract.ok:
        title = f"[followon · {item.origin}] {item.url}"
        return FetchResult(
            title=title,
            url=item.url,
            body=extract.body,
            domain_hint=domain_hint or item.origin,
        )

    log.info(
        "followon extract failed for %s (error=%s); returning thin fetch",
        item.url, extract.error,
    )
    return FetchResult(
        title=f"[followon · {item.origin}] {item.url}",
        url=item.url,
        body=f"(could not extract; error={extract.error}; origin={item.origin})",
        domain_hint=domain_hint or item.origin,
    )


# ---------------------------------------------------------------------------
# Public fetcher
# ---------------------------------------------------------------------------


async def web_search_fetcher(
    domain: str,
    query_hint: str,
    *,
    session_state: SessionState | None = None,
) -> FetchResult:
    """Fetch one content unit for the wandering agent.

    BEHAVIOR:
      - If session_state has a queued follow-on URL, materialise that
        first (via tier-2 extract).
      - Otherwise run the provider chain (Exa → Tavily → DDG), dedup
        against session_state.visited_urls, return the first unvisited
        hit stitched into a FetchResult.

    Defensive: every error path returns a FetchResult with an explanatory
    message — the matcher will score it as no-match and the agent moves
    on. NEVER raises.

    `session_state=None` is supported for legacy callers / tests so the
    old single-agent stub_fetcher pattern still works without breaking.
    """
    # 1. Follow-on queue takes precedence over search.
    if session_state is not None:
        item = await session_state.pop_followon()
        if item is not None:
            await session_state.mark_visited(item.url)
            return await _fetch_from_followon(item, domain_hint=domain)

    # 2. Standard search path.
    query = _build_query_for_domain(domain, query_hint)
    try:
        result = await _search_with_chain(query)
    except Exception as e:
        log.warning("provider chain failed for domain=%s: %s", domain, e)
        return FetchResult(
            title=f"[search error for {domain}]",
            url="",
            body=f"(provider chain error: {e})",
            domain_hint=domain,
        )

    if result.error and not result.hits:
        log.debug("provider chain returned error for %s: %s", domain, result.error)
        return FetchResult(
            title=f"[no results for {domain}]",
            url="",
            body=f"(no usable results: {result.error})",
            domain_hint=domain,
        )

    pick_index, pick_url = _pick_unvisited_hit(result, session_state)
    if pick_index < 0:
        return FetchResult(
            title=f"[no results for {domain}]",
            url="",
            body="(empty hit list)",
            domain_hint=domain,
        )

    if session_state is not None and pick_url:
        await session_state.mark_visited(pick_url)

    title, url, body = _stitch_from_hit(result, pick_index)
    return FetchResult(
        title=title,
        url=url,
        body=body,
        domain_hint=domain,
    )


__all__ = [
    "DEFAULT_HITS_PER_FETCH",
    "DEFAULT_BODY_CHAR_CAP",
    "DEDUP_LOOKBACK_HITS",
    "EXA_NUM_RESULTS",
    "web_search_fetcher",
]
