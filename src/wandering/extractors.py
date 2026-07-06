"""
extractors — URL → clean markdown body, gated to fire only when worth it.

Tier 1 in Wandering Room is the snippet sweep (stitched search hits,
~2500 chars total). Cheap. Most fetches end here.

Tier 2 is THIS module: fetch the actual page, strip nav/ads/footers,
return ~10-20k chars of clean markdown. Fires ONLY when tier-1 has
already produced signal (≥1 matched node) AND the structural confidence
is below the strong threshold — i.e., the borderline case where a fuller
body could tip the match.

The provider is Jina Reader: `GET https://r.jina.ai/<URL>` returns
markdown-ish text. Free at 20 RPM without an API key; faster + higher
limits with `JINA_API_KEY`. The endpoint already handles JS-rendered
pages and content extraction — we don't have to ship a headless browser.

WHY NOT FIRECRAWL/CRAWL4AI:
See WANDERING_ROOM_DECISIONS.md D4. Full-site crawl is the opposite of
what wandering wants (sparse cross-domain hops, not 200 pages from one
site). Jina's single-URL read is exactly the right primitive.

SAFETY:
Every call returns an `ExtractResult` — never raises. Caller can fall
back to tier-1 body on any failure. Adding tier-2 must never degrade
tier-1 performance.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import httpx


log = logging.getLogger("constellax.wandering.extractors")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


#: Jina Reader endpoint. The URL is appended verbatim: r.jina.ai/<URL>.
#: API key is optional — set JINA_API_KEY for higher rate limits.
JINA_READER_BASE = "https://r.jina.ai/"

#: Default timeout. Jina renders JS so first-byte can be slow (~3-8s).
#: 15s is generous-but-bounded; if Jina is degraded, tier-2 falls through
#: and tier-1 body is used.
DEFAULT_TIMEOUT_SEC = 15.0

#: Hard cap on extracted body length. ~20k chars is roughly 5k tokens for
#: the matcher — enough to capture the structural shape of long-form
#: writing without blowing past the matcher's prompt budget.
MAX_EXTRACTED_CHARS = 20_000

#: Tier-2 should only fire on borderline cases. These thresholds gate the
#: escalation decision (called from agent.py).
#:   - Minimum matched-node count to consider escalation (tier-1 must
#:     have at least SOME signal — escalating zero-match URLs is wasted
#:     budget).
#:   - Maximum essence/mechanism ratio that still qualifies as
#:     "borderline" — above this, tier-1 already captured the strong
#:     signal and a richer body is unlikely to materially improve it.
TIER2_MIN_MATCHED_NODES = 1
TIER2_ESCALATION_CEILING = 0.7  # essence or mechanism ratio above this → skip

#: Sampling rate for trusted-domain zero-match URLs. The normal gate
#: filters out anything with `matched_nodes < TIER2_MIN_MATCHED_NODES`,
#: but search snippets can be misleading — a great arxiv paper might
#: have an abstract that yields zero structural matches even when the
#: full body is full of them. This rule lets ~15% of zero-match URLs
#: from trusted domains (arxiv, wikipedia, *.edu, *.gov, etc.) through
#: to tier-2 anyway. Sampling is deterministic per URL via hash so the
#: same URL always gets the same decision across runs.
#:
#: 15% adds ~15% to Jina spend on trusted-zero-match URLs (a tiny
#: fraction of total tier-2 spend) in exchange for catching the
#: misleading-snippet case. Tunable — drop to 5% if Jina costs spike.
TIER2_TRUSTED_ZERO_MATCH_SAMPLE = 0.15


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ExtractResult:
    """Outcome of a tier-2 page read.

    `ok` is True when we got usable content. Caller checks `ok` before
    using `body`. On failure, `error` carries a short reason (network
    timeout, http_404, etc.) and `body` is empty — caller falls back
    to the tier-1 body it already has.

    `chars` is len(body) for telemetry (so we can see in counters whether
    tier-2 is producing materially larger bodies than tier-1's 2500 cap).
    """

    url: str
    body: str = ""
    chars: int = 0
    error: str | None = None
    latency_ms: int = 0
    ok: bool = False


# ---------------------------------------------------------------------------
# Tier-2 escalation gate
# ---------------------------------------------------------------------------


def should_escalate_to_tier2(
    total_matched_nodes: int,
    essence_ratio: float,
    mechanism_ratio: float,
    *,
    url: str = "",
) -> bool:
    """Decide whether a tier-1 result warrants the more expensive tier-2
    read. Pure function — no I/O. Caller wires this in agent.py.

    Two paths:

      MAIN — fires when:
        - tier-1 has signal (>=1 node matched), AND
        - structural match is borderline (essence and mechanism both
          below the strong-signal ceiling).
        The "essence OR mechanism >= 0.7" case already qualifies as
        HIGH confidence — re-reading the full page can't strengthen
        what's already strong, and the budget is better spent on a
        fresh URL.

      SAMPLING ESCAPE HATCH — fires when:
        - matched_nodes == 0 (no tier-1 signal at all), AND
        - the URL host is in the trust whitelist
          (`domain_weight(url) >= PROMOTION_THRESHOLD`), AND
        - a deterministic hash of the URL falls into the bottom 15%.
        This catches "bad snippet, good page" cases — an arxiv paper
        with a misleading abstract whose body would have produced a
        strong match.

    The `url` param is optional for backwards compatibility with
    callers that don't have a URL handy; the sampling rule simply
    doesn't fire in that case.
    """
    # MAIN path
    if total_matched_nodes >= TIER2_MIN_MATCHED_NODES:
        if essence_ratio >= TIER2_ESCALATION_CEILING:
            return False
        if mechanism_ratio >= TIER2_ESCALATION_CEILING:
            return False
        return True

    # SAMPLING escape hatch — only triggers on zero-match URLs from
    # trusted domains, and only on ~15% of them (deterministic per URL).
    if not url:
        return False
    # Lazy import to avoid circulars (trust imports from report; we live
    # parallel to both).
    from src.wandering.trust import domain_weight, PROMOTION_THRESHOLD
    if domain_weight(url) < PROMOTION_THRESHOLD:
        return False
    return _trusted_zero_match_sample(url) < TIER2_TRUSTED_ZERO_MATCH_SAMPLE


def _trusted_zero_match_sample(url: str) -> float:
    """Deterministic 0..1 sample value for a URL. Same URL always returns
    the same number, so the sampling decision is reproducible across
    runs and process restarts — important for debugging "why did this
    URL get extracted last week and not now?"

    Uses md5 for stability (Python's built-in hash() is randomized
    across processes for hash-flooding protection)."""
    import hashlib
    digest = hashlib.md5(url.encode("utf-8", errors="ignore")).hexdigest()
    # First 8 hex chars → uint32 → fraction in [0, 1).
    return int(digest[:8], 16) / 0x1_0000_0000


# ---------------------------------------------------------------------------
# Jina Reader extractor
# ---------------------------------------------------------------------------


def _build_jina_url(target_url: str) -> str:
    """Compose the Jina endpoint URL. Jina expects the full target URL
    appended verbatim (no encoding). Defensive: strip whitespace.
    """
    return JINA_READER_BASE + target_url.strip()


def _jina_headers() -> dict[str, str]:
    """Build request headers for Jina. Adds Authorization if key present;
    requests plain-text markdown response (default is also markdown but
    we make it explicit)."""
    headers = {
        "Accept": "text/plain, text/markdown",
        "User-Agent": "Constellax-Wandering/1.0",
        # Jina respects X-No-Cache to bypass its CDN — wandering values
        # freshness over cache-hit-rate (we're a low-volume caller).
        # Disabled by default; can enable per-call if needed.
    }
    api_key = os.environ.get("JINA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def extract_url(
    url: str,
    *,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    char_cap: int = MAX_EXTRACTED_CHARS,
) -> ExtractResult:
    """Fetch and clean a single URL via the appropriate extractor.

    Dispatching order (see WANDERING_ROOM_DECISIONS.md D5 and
    WANDERING_ROOM_FUTURE_WORK.md F1):
      1. PDF URLs go through pdf_extractor (Claude native PDF input)
         when ANTHROPIC_API_KEY is set — preserves LaTeX math, code
         blocks, figure captions.
      2. Everything else (including PDFs without an Anthropic key)
         goes through Jina Reader.
      3. If the PDF path fails for any reason, we fall back to Jina —
         degraded math fidelity is better than no extract.

    Returns ExtractResult — never raises. On any failure (timeout, 4xx,
    5xx, network, empty body) the caller (agent.py) falls back to
    tier-1 stitched snippets.

    Why the strict char cap: the matcher LLM is on Haiku for cost. A
    20k-char body is ~5k tokens; pushing past that buys little signal
    per token and squeezes the cushion + prompt scaffolding.
    """
    if not url or not url.startswith(("http://", "https://")):
        return ExtractResult(
            url=url, error="invalid_url", ok=False, latency_ms=0,
        )

    # PDF dispatch (F1) — only when the URL is clearly a PDF AND the
    # Anthropic-direct path is configured. Soft-fails through to Jina.
    if _is_pdf_url(url) and _pdf_extractor_available():
        pdf_result = await _try_pdf_extract(url, char_cap=char_cap)
        if pdf_result is not None:
            return pdf_result
        log.debug("pdf extract failed for %s; falling back to Jina", url)

    jina_url = _build_jina_url(url)
    start = time.time()

    try:
        async with httpx.AsyncClient(
            timeout=timeout_sec, headers=_jina_headers(),
        ) as client:
            response = await client.get(jina_url)
            response.raise_for_status()
            body = response.text
    except httpx.TimeoutException:
        return ExtractResult(
            url=url, error="timeout", ok=False,
            latency_ms=int((time.time() - start) * 1000),
        )
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        # 404/410/451 = page gone or blocked; 429 = rate limit; 5xx = upstream.
        # All map to soft-fail; agent uses tier-1 body.
        return ExtractResult(
            url=url, error=f"http_{status}", ok=False,
            latency_ms=int((time.time() - start) * 1000),
        )
    except Exception as e:
        log.warning("jina extract failed for %s: %s", url, e)
        return ExtractResult(
            url=url, error=f"{type(e).__name__}: {e}", ok=False,
            latency_ms=int((time.time() - start) * 1000),
        )

    body = (body or "").strip()
    if not body:
        return ExtractResult(
            url=url, error="empty_body", ok=False,
            latency_ms=int((time.time() - start) * 1000),
        )

    truncated = False
    if len(body) > char_cap:
        body = body[:char_cap] + "\n\n...[truncated by Constellax]"
        truncated = True

    if truncated:
        log.debug("jina extract truncated %s to %d chars", url, char_cap)

    return ExtractResult(
        url=url,
        body=body,
        chars=len(body),
        ok=True,
        latency_ms=int((time.time() - start) * 1000),
    )


# ---------------------------------------------------------------------------
# Link extraction from extracted markdown
# ---------------------------------------------------------------------------


def extract_links(markdown_body: str, max_links: int = 25) -> list[tuple[str, str]]:
    """Pull (anchor_text, url) tuples from a Jina markdown body.

    Used by the link follow-on pipeline (agent.py): we extract links,
    score them via the matcher LLM (anchor text + cushion), and queue
    the top 1-2 in the session's follow-on queue.

    Conservative regex — only matches well-formed `[text](url)` pairs.
    Strips URLs to absolute HTTP(S) only (relative links would need
    a base-URL anyway, and most Jina markdowns are pre-absolutized).

    `max_links` caps the parse to avoid burning matcher tokens on
    every footer/sidebar link of a long article. The agent later
    scores all of these but only queues 1-2.
    """
    import re

    if not markdown_body:
        return []

    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
    out: list[tuple[str, str]] = []
    seen_urls: set[str] = set()

    for m in pattern.finditer(markdown_body):
        text = (m.group(1) or "").strip()
        url = (m.group(2) or "").strip()
        if not text or not url:
            continue
        if url in seen_urls:
            continue
        # Drop obvious nav/social/tracker links by anchor text.
        if text.lower() in {
            "home", "menu", "search", "login", "sign in", "sign up",
            "subscribe", "share", "tweet", "facebook", "twitter",
            "linkedin", "email", "print", "copy link", "back to top",
        }:
            continue
        seen_urls.add(url)
        out.append((text, url))
        if len(out) >= max_links:
            break

    return out


def _is_pdf_url(url: str) -> bool:
    """Lazy local import — keeps pdf_extractor optional."""
    try:
        from src.wandering import pdf_extractor
        return pdf_extractor.is_pdf_url(url)
    except Exception:
        return False


def _pdf_extractor_available() -> bool:
    """Lazy local import — pdf_extractor only matters when ANTHROPIC_API_KEY
    is set. Without the key, this returns False and the dispatcher routes
    everything through Jina."""
    try:
        from src.wandering import pdf_extractor
        return pdf_extractor.is_available()
    except Exception:
        return False


async def _try_pdf_extract(
    url: str, *, char_cap: int = MAX_EXTRACTED_CHARS,
) -> "ExtractResult | None":
    """Call the PDF extractor. Returns an ExtractResult on success or
    None to signal the caller to fall back to Jina. The pdf_extractor
    has its own error reporting via `error` field, but we only return
    that result when ok=True — otherwise we let Jina have a turn at
    the same URL."""
    try:
        from src.wandering import pdf_extractor
        result = await pdf_extractor.extract_pdf_url(url)
    except Exception as e:
        log.debug("pdf_extractor crashed for %s: %s", url, e)
        return None
    if not result.ok or not result.body:
        return None
    body = result.body
    if len(body) > char_cap:
        body = body[:char_cap] + "\n\n...[truncated by Constellax]"
    return ExtractResult(
        url        = result.url,
        body       = body,
        chars      = len(body),
        ok         = True,
        latency_ms = result.latency_ms,
    )


__all__ = [
    "JINA_READER_BASE",
    "DEFAULT_TIMEOUT_SEC",
    "MAX_EXTRACTED_CHARS",
    "TIER2_MIN_MATCHED_NODES",
    "TIER2_ESCALATION_CEILING",
    "ExtractResult",
    "should_escalate_to_tier2",
    "extract_url",
    "extract_links",
]
