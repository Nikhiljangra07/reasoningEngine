"""
exa_provider — Exa search (neural + keyword) and /findSimilar.

Exa is the differentiator in Wandering Room's retrieval mesh. Tavily
returns lexically-matched pages; Exa's neural search returns pages whose
embeddings sit near the query in concept space, which is exactly the
cross-domain-analogy primitive wandering needs.

/findSimilar is the gold: give it a URL, get back URLs whose embeddings
are near it. This is "wander from THIS point along the embedding
manifold" — a different flavor of chaos than the domain seed list.

Per Law 1, neither call is used to OPTIMIZE the walk. They widen the
walkable space. Two flavors of chaos coexisting (lexical via Tavily,
embedding via Exa) is exactly what the multi-perspective wander wants.

API:
  POST https://api.exa.ai/search      — neural + keyword search
  POST https://api.exa.ai/findSimilar — URL → similar URLs

Auth: Bearer token via `EXA_API_KEY`. Without the key, every call
returns an empty result with error="no_api_key" — agents fall back
to Tavily.

Free tier: 1,000 requests/mo. Enough for ~20 ABSOLUTE_CHAOS wanders
at the planned per-agent call count. Above that, ~$0.005/request.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx


log = logging.getLogger("constellax.wandering.exa")


EXA_BASE = "https://api.exa.ai"
DEFAULT_TIMEOUT_SEC = 10.0
DEFAULT_NUM_RESULTS = 5

# Search type: "neural" leans on embedding proximity (best for
# cross-domain analogy), "keyword" is closer to traditional search,
# "auto" lets Exa pick. We default to "neural" because that's the
# wandering-specific advantage.
SearchType = Literal["neural", "keyword", "auto"]
DEFAULT_SEARCH_TYPE: SearchType = "neural"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ExaHit:
    """One result row from an Exa search or findSimilar call.

    Exa returns richer metadata than Tavily (score, published date,
    author). We keep score (used for sorting / trust adjustment) and the
    fields we know we'll use; the raw dict is discarded.
    """

    title: str
    url: str
    text: str = ""              # snippet/excerpt; can be empty
    score: float = 0.0          # Exa's relevance score (higher better)
    published_date: str = ""    # ISO date if available


@dataclass
class ExaResult:
    """Outcome of one Exa API call. Never raises; caller falls back.

    `kind` distinguishes search vs findSimilar for telemetry.
    `query_or_url` carries either the search query or the seed URL,
    for log/audit purposes (and for caching keys).
    """

    kind: Literal["search", "findSimilar"]
    query_or_url: str
    hits: list[ExaHit] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.hits) and self.error is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_key() -> str:
    """Resolved at call time so test code can monkey-patch the env."""
    return os.environ.get("EXA_API_KEY", "").strip()


def _headers() -> dict[str, str]:
    """Build Exa request headers. Authorization is required for all
    endpoints — without the key the caller should short-circuit before
    we get here."""
    key = _api_key()
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _parse_hits(payload: dict) -> list[ExaHit]:
    """Pull the hits array out of an Exa response.

    Exa nests results under `results`. Each item has the fields we care
    about plus extras (highlights, similar_url, etc.) we ignore. We
    tolerate missing fields gracefully — title and URL are required;
    everything else defaults.
    """
    out: list[ExaHit] = []
    for r in (payload.get("results") or []):
        url = (r.get("url") or "").strip()
        title = (r.get("title") or "").strip()
        if not url:
            continue
        # If title is missing, fall back to the URL host (better than
        # empty in the trace).
        if not title:
            try:
                from urllib.parse import urlsplit
                title = urlsplit(url).netloc or url
            except Exception:
                title = url
        text = (r.get("text") or r.get("snippet") or "").strip()
        score_raw = r.get("score")
        try:
            score = float(score_raw) if score_raw is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        published = (r.get("publishedDate") or r.get("published_date") or "").strip()
        out.append(ExaHit(
            title=title,
            url=url,
            text=text,
            score=score,
            published_date=published,
        ))
    return out


def is_available() -> bool:
    """True if an Exa API key is configured. Cheap diagnostic for the
    runtime to log on startup so we know which provider is active."""
    return bool(_api_key())


# ---------------------------------------------------------------------------
# /search — neural or keyword
# ---------------------------------------------------------------------------


async def search(
    query: str,
    *,
    num_results: int = DEFAULT_NUM_RESULTS,
    search_type: SearchType = DEFAULT_SEARCH_TYPE,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> ExaResult:
    """Run an Exa search and return up to `num_results` hits.

    `search_type="neural"` (default) ranks by embedding proximity to the
    query — best for wandering's cross-domain analogy use case.

    Returns an empty ExaResult with error="no_api_key" if EXA_API_KEY is
    not configured. Caller (the fetcher) should fall back to Tavily.
    """
    if not _api_key():
        return ExaResult(kind="search", query_or_url=query, error="no_api_key")
    if not query or not query.strip():
        return ExaResult(kind="search", query_or_url=query, error="empty_query")

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.post(
                f"{EXA_BASE}/search",
                headers=_headers(),
                json={
                    "query": query.strip(),
                    "numResults": num_results,
                    "type": search_type,
                    # Return the page text excerpt so we can stitch a
                    # tier-1 body without a second call.
                    "contents": {"text": True},
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException:
        return ExaResult(
            kind="search", query_or_url=query, error="timeout",
            latency_ms=int((time.time() - start) * 1000),
        )
    except httpx.HTTPStatusError as e:
        return ExaResult(
            kind="search", query_or_url=query,
            error=f"http_{e.response.status_code}",
            latency_ms=int((time.time() - start) * 1000),
        )
    except Exception as e:
        log.warning("exa search failed: %s", e)
        return ExaResult(
            kind="search", query_or_url=query,
            error=f"{type(e).__name__}: {e}",
            latency_ms=int((time.time() - start) * 1000),
        )

    hits = _parse_hits(payload)
    return ExaResult(
        kind="search",
        query_or_url=query,
        hits=hits,
        latency_ms=int((time.time() - start) * 1000),
    )


# ---------------------------------------------------------------------------
# /findSimilar — URL in → similar URLs out
# ---------------------------------------------------------------------------


async def find_similar(
    url: str,
    *,
    num_results: int = DEFAULT_NUM_RESULTS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    exclude_source_domain: bool = True,
) -> ExaResult:
    """Find pages embedding-similar to `url`.

    This is the chaos-hop primitive. Given a URL the agent already
    matched against the cushion, /findSimilar returns URLs whose
    embeddings sit near it in concept space — often from domains the
    agent's seed list never named.

    `exclude_source_domain` defaults True because for wandering we
    explicitly DON'T want to find more pages on the same site (that's
    the optimization mistake — see WANDERING_ROOM_DECISIONS.md D4).
    We want the hop, not the dive.
    """
    if not _api_key():
        return ExaResult(kind="findSimilar", query_or_url=url, error="no_api_key")
    if not url or not url.startswith(("http://", "https://")):
        return ExaResult(
            kind="findSimilar", query_or_url=url, error="invalid_url",
        )

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.post(
                f"{EXA_BASE}/findSimilar",
                headers=_headers(),
                json={
                    "url": url.strip(),
                    "numResults": num_results,
                    "excludeSourceDomain": exclude_source_domain,
                    # Page text excerpt is optional for findSimilar; the
                    # caller (agent.py) only needs the URLs to queue.
                    # We omit contents to keep latency low.
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException:
        return ExaResult(
            kind="findSimilar", query_or_url=url, error="timeout",
            latency_ms=int((time.time() - start) * 1000),
        )
    except httpx.HTTPStatusError as e:
        return ExaResult(
            kind="findSimilar", query_or_url=url,
            error=f"http_{e.response.status_code}",
            latency_ms=int((time.time() - start) * 1000),
        )
    except Exception as e:
        log.warning("exa findSimilar failed: %s", e)
        return ExaResult(
            kind="findSimilar", query_or_url=url,
            error=f"{type(e).__name__}: {e}",
            latency_ms=int((time.time() - start) * 1000),
        )

    hits = _parse_hits(payload)
    return ExaResult(
        kind="findSimilar",
        query_or_url=url,
        hits=hits,
        latency_ms=int((time.time() - start) * 1000),
    )


__all__ = [
    "EXA_BASE",
    "DEFAULT_TIMEOUT_SEC",
    "DEFAULT_NUM_RESULTS",
    "DEFAULT_SEARCH_TYPE",
    "SearchType",
    "ExaHit",
    "ExaResult",
    "is_available",
    "search",
    "find_similar",
]
