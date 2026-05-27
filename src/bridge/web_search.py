"""
web_search — pluggable web search for the reasoning engine.

THE PURPOSE
===========
The synthesizer's training data has a knowledge cutoff. When a user
asks about current events, live policy documents, or anything that
postdates training (or that the model never saw), the engine should
ground its answer in the live web instead of confabulating.

We do this with a small pre-trace augmentation: an LLM-as-router
decides whether to search and rewrites the query for search engines,
we fire the query, and we inject the top 3-5 results as a "WEB
CONTEXT" block at the top of the prompt. The synthesizer reads it as
authoritative context and cites accordingly.

PROVIDERS
=========
Two backends, same wire shape. The dispatcher (`web_search` below)
picks whichever is configured.

1. **Tavily** (default when `TAVILY_API_KEY` is set). LLM-optimized
   search — returns extracted answers + full page text on top results,
   which is the right shape for grounding a model. 1,000 queries/mo
   free, $30/mo for 4k, then $0.0075/query.

2. **DuckDuckGo HTML** (fallback when no key set). Free, key-less,
   no monthly cap. Quality is "Bing-via-DDG" — strong for well-known
   sites, snippets are shorter (~200 chars vs Tavily's full page).
   Zero new dependencies — httpx + regex.

The fallback chain matters: if Tavily is configured but the request
fails (network, rate limit, malformed response), we fall through to
DDG so the engine still has *some* live context.

SWAPPING IN AN OPEN-SOURCE / LOCAL PROVIDER
===========================================
The `_SearchProvider` protocol is intentionally tiny — a class with
one async method `search(query, max_results) -> SearchResult`. A new
provider only needs to implement that interface and register itself
in `_pick_provider()`. Candidates worth wiring next:

- **SearXNG** (self-hosted meta-search; aggregates Bing/Brave/etc.;
  fully open-source, zero cost, you run the instance)
- **DistilBERT-reranked SerpAPI** (paid Google + local reranker for
  retrieval-quality boost)
- **Local Wikipedia + arXiv via sentence-transformers retrieval**
  (no live web, but excellent for technical/scientific questions)

For now we ship Tavily + DDG; the rest plug in without touching
the trace endpoint.

SAFETY
======
Every search call returns a SearchResult — never raises. The caller
can rely on `web_search(q)` to either help or no-op; a search
failure must never break the trace.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import httpx

log = logging.getLogger("constellax.web_search")

# How long a cached result stays fresh. Web pages change but the
# expensive things (terms of service, policy docs, paper abstracts) are
# usually stable hour-to-hour. Five minutes is a reasonable compromise.
CACHE_TTL_SEC = 300
DEFAULT_TIMEOUT_SEC = 10.0
MAX_RESULTS_DEFAULT = 5

# ─── Provider configuration via env ──────────────────────────────────
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
TAVILY_URL = "https://api.tavily.com/search"

# DDG's HTML endpoint. Stable across the last several years; if it ever
# moves, swap the URL here and the rest of the module is unaffected.
DDG_URL = "https://html.duckduckgo.com/html/"
DDG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.5 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.8",
}


@dataclass
class SearchHit:
    """One result row from a search."""
    title:   str
    snippet: str
    url:     str

    def as_context_line(self) -> str:
        """Format as a single-bullet line for prompt injection."""
        return f"- {self.title}\n  {self.snippet}\n  ({self.url})"


@dataclass
class SearchResult:
    """Outcome of a search call. Never null — empty hits on failure."""
    query:    str
    hits:     list[SearchHit] = field(default_factory=list)
    cached:   bool = False
    error:    str | None = None
    latency_ms: int = 0
    provider: str = ""            # "tavily" | "duckduckgo" | ...
    answer:   str | None = None   # Tavily-only: pre-extracted direct answer

    @property
    def ok(self) -> bool:
        return self.hits and not self.error


class _SearchProvider(Protocol):
    """Minimal interface a backend implements. Any new provider plugs
    in here — same dispatcher, same SearchResult shape, no caller
    changes."""
    name: str
    async def search(self, query: str, max_results: int) -> SearchResult: ...


# ─── Tavily provider ─────────────────────────────────────────────────

class TavilyProvider:
    """Tavily — LLM-optimized search. Returns extracted answers + page
    excerpts in one call, which is the right shape for grounding. Free
    tier: 1000 queries/month. Falls back through to DDG if key absent."""
    name = "tavily"

    def __init__(self, api_key: str, timeout_sec: float = DEFAULT_TIMEOUT_SEC):
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    # Tavily's API caps query length at ~400 chars (free tier — longer
    # queries silently 400 or return empty). The search router can hand
    # us the raw user question (multi-paragraph reflective questions
    # are common in Constellax), so we trim defensively here before
    # the request leaves our process.
    _MAX_QUERY_CHARS = 380

    async def search(self, query: str, max_results: int) -> SearchResult:
        start = time.time()
        # Trim long queries at a word boundary to stay under Tavily's cap.
        if len(query) > self._MAX_QUERY_CHARS:
            cut = query[: self._MAX_QUERY_CHARS].rsplit(" ", 1)[0]
            query = cut if len(cut) > 0 else query[: self._MAX_QUERY_CHARS]
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                response = await client.post(
                    TAVILY_URL,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        # "basic" is plenty for the synthesizer; "advanced"
                        # doubles cost-tier without proportional quality
                        # gain for our use case.
                        "search_depth": "basic",
                        "max_results": max_results,
                        "include_answer": True,
                        # Snippets are enough — raw_content is huge and
                        # blows past prompt budgets on multi-result responses.
                        "include_raw_content": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException:
            return SearchResult(query=query, provider=self.name, error="timeout",
                                latency_ms=int((time.time() - start) * 1000))
        except httpx.HTTPStatusError as e:
            # 401 = bad key, 429 = rate limit, 5xx = upstream. Treat all
            # as soft-fail so the caller can fall through to DDG.
            return SearchResult(
                query=query, provider=self.name,
                error=f"http_{e.response.status_code}",
                latency_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            log.warning("tavily search failed: %s", e)
            return SearchResult(
                query=query, provider=self.name,
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.time() - start) * 1000),
            )

        hits: list[SearchHit] = []
        for r in (payload.get("results") or [])[:max_results]:
            title = (r.get("title") or "").strip()
            content = (r.get("content") or "").strip()
            url = (r.get("url") or "").strip()
            if not title or not url:
                continue
            hits.append(SearchHit(title=title, snippet=content, url=url))

        return SearchResult(
            query=query,
            hits=hits,
            provider=self.name,
            answer=(payload.get("answer") or None),
            latency_ms=int((time.time() - start) * 1000),
        )


# ─── DuckDuckGo provider (free, key-less, HTML scrape) ───────────────

class DuckDuckGoProvider:
    """DDG HTML endpoint. No key, no cost, no monthly cap. Always
    available as the floor of our fallback chain."""
    name = "duckduckgo"

    def __init__(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC):
        self.timeout_sec = timeout_sec

    async def search(self, query: str, max_results: int) -> SearchResult:
        start = time.time()
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_sec, headers=DDG_HEADERS,
            ) as client:
                response = await client.post(DDG_URL, data={"q": query})
                response.raise_for_status()
                body = response.text
        except httpx.TimeoutException:
            return SearchResult(query=query, provider=self.name, error="timeout",
                                latency_ms=int((time.time() - start) * 1000))
        except Exception as e:
            log.warning("ddg search failed: %s", e)
            return SearchResult(
                query=query, provider=self.name,
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.time() - start) * 1000),
            )

        hits = list(_parse_ddg_html(body))[:max_results]
        return SearchResult(
            query=query, hits=hits, provider=self.name,
            latency_ms=int((time.time() - start) * 1000),
        )


# ─── Cache + dispatcher ──────────────────────────────────────────────

_cache: dict[str, tuple[float, SearchResult]] = {}
_cache_lock = asyncio.Lock()

_tavily: TavilyProvider | None = TavilyProvider(TAVILY_API_KEY) if TAVILY_API_KEY else None
_ddg = DuckDuckGoProvider()


def _normalise(query: str) -> str:
    return " ".join(query.lower().split())


def active_provider_chain() -> list[str]:
    """For diagnostics — which providers will run, in order, on a search."""
    chain: list[str] = []
    if _tavily is not None: chain.append(_tavily.name)
    chain.append(_ddg.name)
    return chain


async def web_search(
    query: str,
    *,
    max_results: int = MAX_RESULTS_DEFAULT,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> SearchResult:
    """Run the configured provider chain. Returns the first successful
    result, or the last failed one if every provider misses.

    Tavily (when configured) runs first because its results are
    LLM-optimized (richer snippets, includes a pre-extracted answer).
    DDG runs as the floor — always available, free, no key needed."""
    q = (query or "").strip()
    if not q:
        return SearchResult(query=q, error="empty query")

    key = _normalise(q)
    now = time.time()

    async with _cache_lock:
        cached = _cache.get(key)
        if cached and (now - cached[0]) < CACHE_TTL_SEC:
            res = cached[1]
            return SearchResult(
                query=res.query,
                hits=list(res.hits[:max_results]),
                cached=True,
                latency_ms=res.latency_ms,
                provider=res.provider,
                answer=res.answer,
            )

    providers: list[_SearchProvider] = []
    if _tavily is not None: providers.append(_tavily)
    providers.append(_ddg)

    last_failure: SearchResult | None = None
    for p in providers:
        try:
            result = await asyncio.wait_for(
                p.search(q, max_results), timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            log.warning("provider %s timed out", p.name)
            last_failure = SearchResult(query=q, provider=p.name, error="timeout")
            continue
        if result.ok:
            async with _cache_lock:
                _cache[key] = (now, result)
                if len(_cache) > 200:
                    oldest_key = min(_cache.keys(), key=lambda k: _cache[k][0])
                    _cache.pop(oldest_key, None)
            return result
        last_failure = result
        log.info("provider %s returned no results (error=%s)", p.name, result.error)

    return last_failure or SearchResult(query=q, error="no providers available")


# ─── HTML parser (regex over DDG's stable structure) ─────────────────

# DDG result blocks look approximately like:
#   <div class="result">
#     <h2 class="result__title">
#       <a class="result__a" href="...">Title</a>
#     </h2>
#     <a class="result__snippet" href="...">Snippet text</a>
#   </div>
#
# We pull (title, href, snippet) per block. The HTML can drift; we tolerate
# missing snippets and malformed href attributes by skipping bad rows.

_RESULT_BLOCK_RE = re.compile(
    r'<a\s+[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?'
    r'(?:class="result__snippet"[^>]*>(.*?)</a>|class="result__snippet"\s*[^>]*>(.*?)</)',
    re.DOTALL | re.IGNORECASE,
)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# DDG wraps target URLs in their redirector; we unwrap.
_DDG_REDIR_RE = re.compile(r"^//duckduckgo\.com/l/\?uddg=([^&]+)")


def _clean_text(raw: str) -> str:
    if not raw:
        return ""
    no_tags = _TAG_STRIP_RE.sub("", raw)
    decoded = html.unescape(no_tags)
    return _WS_RE.sub(" ", decoded).strip()


def _unwrap_url(href: str) -> str:
    if not href:
        return ""
    m = _DDG_REDIR_RE.match(href)
    if m:
        from urllib.parse import unquote
        return unquote(m.group(1))
    # Some entries come back as full URLs already
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    return href


def _parse_ddg_html(body: str) -> Iterable[SearchHit]:
    for m in _RESULT_BLOCK_RE.finditer(body):
        href, title_raw, snip_a, snip_b = m.groups()
        snippet = _clean_text(snip_a or snip_b or "")
        title = _clean_text(title_raw or "")
        url = _unwrap_url(href)
        if not title or not url:
            continue
        yield SearchHit(title=title, snippet=snippet, url=url)


# ─── Heuristic: should we search? ─────────────────────────────────────

# Triggers — keep tight. False positives cost ~300ms per question (the
# search latency); false negatives mean the engine answers from training
# data when it shouldn't. We err toward false negatives (don't search
# when in doubt) so cheap questions stay cheap.
_NEEDS_SEARCH_PATTERNS = [
    # Currency / recency
    r"\b(current|latest|today|recent|recently|this (?:year|month|week)|2025|2026)\b",
    # Live / external documents
    r"\b(terms of (?:service|use)|privacy policy|policies|pricing|api docs?|documentation|changelog)\b",
    # Explicit search asks
    r"\b(search|google|look up|dig out|find out|find me|fetch|pull up)\b",
    # Named entities that change over time
    r"\b(release|launched?|announced|update[ds]?|version)\b",
    # URLs in question = obviously wants the live page
    r"https?://\S+",
]
_NEEDS_SEARCH_RE = re.compile("|".join(_NEEDS_SEARCH_PATTERNS), re.IGNORECASE)


def should_search(question: str) -> bool:
    """Heuristic: does this question benefit from live web data?

    Returns True when the question mentions current events, live
    documents, policy/ToS, explicit "search/look up" verbs, or
    contains a URL. Otherwise False — cheap questions stay cheap."""
    if not question or len(question.strip()) < 4:
        return False
    return bool(_NEEDS_SEARCH_RE.search(question))


# ─── Prompt augmentation block builder ────────────────────────────────

def format_web_context_block(result: SearchResult) -> str:
    """Render a SearchResult as a prompt-ready context block.

    The synthesizer sees this prepended to the user's question, framed
    explicitly as "consulted live sources" so the model treats it as
    authoritative for time-sensitive claims. Empty when no hits.

    When the provider (Tavily) returns a pre-extracted `answer`, it's
    rendered above the result list so the synthesizer sees it first —
    that field is the search provider's best attempt to directly
    answer the query and is the most useful single sentence."""
    if not result.ok:
        return ""
    lines = [
        "[WEB CONTEXT — live search results via "
        f"{result.provider or 'unknown'}, treat as authoritative for "
        "time-sensitive claims; cite URLs when quoting]",
        f"Query: {result.query}",
    ]
    if result.answer:
        lines.append("")
        lines.append(f"Synthesised answer (from provider): {result.answer}")
    lines.append("")
    for h in result.hits:
        lines.append(h.as_context_line())
    lines.append("")
    return "\n".join(lines)
