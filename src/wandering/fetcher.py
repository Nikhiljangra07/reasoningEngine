"""
Real fetcher adapter — wires Wandering Room's FetchFn to web_search.

Replaces the stub_fetcher with a real source that runs Tavily (or the
configured search provider chain). Per Law 4: read-only — no writes,
no scraping that bypasses robots/TOS handling that web_search already
does.

Wandering Room can also fetch from:
  - Tavily web search (this module)
  - Notion (deferred — Phase wiring when Notion MCP lands)
  - Project memory (handled by composer.fetch_memory_enrichment)
  - IDE files (deferred — VSCode shim)

ISOLATION: imports web_search bridge + FetchResult type. No persistence,
no LLM calls — purely a fetch adapter.
"""

from __future__ import annotations

import logging
from typing import Any

from src.bridge.web_search import SearchResult, web_search
from src.wandering.agent import FetchResult


log = logging.getLogger("constellax.wandering.fetcher")


#: How many search hits to use when stitching together "content" for one
#: fetch call. Wandering Room treats one fetch as one content unit (which
#: the matcher analyzes). We collect a few top hits and bundle them so
#: the structural match has enough surface to evaluate.
DEFAULT_HITS_PER_FETCH = 3

#: Cap on body length per fetch. Wandering Room matches against structural
#: patterns; we don't need raw HTML dumps. Truncate to keep token costs sane.
DEFAULT_BODY_CHAR_CAP = 2500


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


def _stitch_hits(result: SearchResult, char_cap: int = DEFAULT_BODY_CHAR_CAP) -> tuple[str, str, str]:
    """Combine a SearchResult into (title, url, body) for one FetchResult.

    Title = first hit's title (or "Search: <query>")
    URL = first hit's URL (the citation anchor)
    Body = concatenated snippets across hits, capped at char_cap
    """
    if not result.hits:
        return (f"Search: {result.query}", "", "(no results)")

    first = result.hits[0]
    title = first.title or f"Search: {result.query}"
    url = first.url or ""

    pieces: list[str] = []
    for h in result.hits[:DEFAULT_HITS_PER_FETCH]:
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


async def web_search_fetcher(domain: str, query_hint: str) -> FetchResult:
    """Fetch one content unit by running a web search for the anchor in
    the given domain.

    Defensive: if web_search fails or returns no provider, falls back to
    a FetchResult with an explanatory message. The agent's matcher will
    score this as no-match (filtered nodes will be 0), and the agent
    moves on without crashing.

    Wire this in by passing it as the `fetcher` kwarg to
    run_wandering_session(). The default in runtime.py is stub_fetcher;
    production callers swap in this one (or a project-memory fetcher).
    """
    query = _build_query_for_domain(domain, query_hint)
    try:
        result = await web_search(query)
    except Exception as e:
        log.warning("web_search_fetcher failed for domain=%s: %s", domain, e)
        return FetchResult(
            title=f"[search error for {domain}]",
            url="",
            body=f"(web_search error: {e})",
            domain_hint=domain,
        )

    if result.error:
        log.debug("web_search returned error for %s: %s", domain, result.error)
        return FetchResult(
            title=f"[no results for {domain}]",
            url="",
            body=f"(no usable results: {result.error})",
            domain_hint=domain,
        )

    title, url, body = _stitch_hits(result)
    return FetchResult(
        title=title,
        url=url,
        body=body,
        domain_hint=domain,
    )


__all__ = [
    "DEFAULT_HITS_PER_FETCH",
    "DEFAULT_BODY_CHAR_CAP",
    "web_search_fetcher",
]
