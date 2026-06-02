"""
Content fingerprinting — the bridge between raw web content and the cushion.

THE PURPOSE
===========
Wandering agents fetch raw prose from the internet (Wikipedia articles,
papers, blog posts, movie summaries — anything). The cushion is a
structured 3-layer graph of essence/mechanism patterns. Matching one
against the other is the load-bearing step of the Constellation
Interpreter.

The naive approach — embed raw prose directly against cushion essence
nodes — fails for cross-domain analogy. Off-the-shelf embedding models
(Gemini, OpenAI ada, Cohere) capture surface semantics, not structural
patterns. So "jazz improvisation" and "AI agent control" land far apart
in embedding space even though they're structurally identical at the
essence layer (bounded freedom).

The fix: BEFORE matching, run a cheap Haiku call that extracts 2-5
short "structural phrases" describing what the content is structurally
about. These phrases use domain-neutral language — language that would
plausibly appear in any content exhibiting the same pattern. Embed THOSE
phrases. Now both sides of the match (cushion nodes' embedding_text and
content fingerprints' phrases_combined) live in the SAME structural
language. Cosine similarity becomes meaningful for cross-domain.

THE CACHE
=========
The same Wikipedia article scanned by 50 sessions should fingerprint
ONCE. Cache key is SHA-256 of normalized content. Hits return the
cached fingerprint without a Haiku call or an embedding call —
typically <50ms. Misses pay the full ~1s for Haiku + ~300ms for embed.

Cache lives in Neo4j as (:ContentFingerprint) nodes with the
`content_fingerprint_hash_unique` constraint and the
`content_fingerprint_embedding_idx` vector index (both created in
Phase 1).

PURE vs CACHED
==============
This module exposes TWO functions:

  fingerprint_content(content, url, *, client, embedder)
    Pure function. Generates a fresh fingerprint via Haiku + embed.
    No I/O beyond LLM/embed calls. Useful for tests, isolated calls,
    and the cache-miss branch of get_or_create_fingerprint.

  get_or_create_fingerprint(content, url, *, client, embedder,
                            neo4j_driver, neo4j_database)
    Cached wrapper. Hits Neo4j first; on miss, calls
    fingerprint_content then persists. The wandering loop uses this.

FAILURE HANDLING
================
The matcher's vector channel is degradation-tolerant by design:
  - Haiku call fails -> ContentFingerprint with empty phrases tuple,
    embedding=None. The match still proceeds; the other 6 channels
    pick up the slack.
  - Embedding fails -> phrases populated, embedding=None. The
    vector channel skips this fingerprint; the rest still works.
  - Neo4j cache lookup fails -> fall through to fresh generation.
  - Persistence fails -> in-memory fingerprint still returned;
    next session will re-fingerprint the same content (cache miss).

Nothing raises. Nothing crashes the wander.

ISOLATION
=========
Imports llm.client (LLM seam) + bridge.embedding_service (Gemini).
Optionally takes a neo4j driver for the cache layer. Does NOT import
wandering.agent / wandering.runtime — fingerprinting is a LEAF
operation, not part of the wander orchestration.

Created 2026-06-01 as Phase 3 of the Constellation Interpreter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from src.bridge.embedding_service import EmbeddingResult, GeminiEmbeddingService
from src.identity import compose_system_prompt
from src.llm.client import LLMClient, LLMResponse


log = logging.getLogger("constellax.wandering.fingerprint")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Same Haiku route the legacy matcher used — the structural-pattern
#: judgment is a small, structured task that doesn't need Sonnet's depth.
FINGERPRINT_DOMAIN = "psychology"
FINGERPRINT_CONCEPT = "content_structural_fingerprint"

#: Min/max phrases per fingerprint. Below 2, the embedding has too
#: little signal; above 5, the phrases lose structural focus and
#: drift into surface paraphrase.
MIN_PHRASES = 2
MAX_PHRASES = 5

#: Cap on input content fed to Haiku. Anything longer is truncated.
#: The model only needs enough to recognise the structural pattern;
#: a full novel would cost tokens for diminishing returns.
MAX_CONTENT_CHARS = 8000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ContentFingerprint:
    """A structural fingerprint of one piece of fetched content.

    `content_hash` is the cache key — SHA-256 of normalized content.
    Same article hashed twice produces the same key, so caching works
    across sessions.

    `phrases` is a tuple of 2-5 essence-shaped structural phrases. Each
    phrase reads as prose, ~8-20 words, in domain-neutral language so
    cross-domain content with the same pattern produces similar phrases.

    `phrases_combined` is the canonical text that gets embedded — phrases
    joined with " | " separator. The matcher's vector channel computes
    cosine similarity between this and cushion nodes' embedding_text.

    `embedding` is the 1536-dim Gemini vector. None if embedding failed.

    `model_used` records which LLM produced the phrases — useful for
    debugging fingerprint quality differences across models.
    """

    content_hash:     str
    url:              str
    domain:           str
    phrases:          tuple[str, ...]
    phrases_combined: str
    embedding:        list[float] | None = None
    created_at:       float = 0.0
    model_used:       str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash":     self.content_hash,
            "url":              self.url,
            "domain":           self.domain,
            "phrases":          list(self.phrases),
            "phrases_combined": self.phrases_combined,
            "embedding":        list(self.embedding) if self.embedding is not None else None,
            "created_at":       self.created_at,
            "model_used":       self.model_used,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContentFingerprint":
        emb = payload.get("embedding")
        return cls(
            content_hash=str(payload.get("content_hash", "")),
            url=str(payload.get("url", "")),
            domain=str(payload.get("domain", "")),
            phrases=tuple(payload.get("phrases") or ()),
            phrases_combined=str(payload.get("phrases_combined", "")),
            embedding=list(emb) if emb else None,
            created_at=float(payload.get("created_at") or 0.0),
            model_used=str(payload.get("model_used") or ""),
        )


# ---------------------------------------------------------------------------
# Hashing + helpers
# ---------------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_content(content: str) -> str:
    """Collapse whitespace and lowercase — case- and spacing-insensitive
    cache key. Punctuation is preserved (changes meaning)."""
    return _WHITESPACE_RE.sub(" ", content.strip().lower())


def compute_content_hash(content: str) -> str:
    """SHA-256 short-hash of normalized content. Primary cache key.

    Returns 32-character hex digest, prefixed `cf_` so the id is
    visually distinguishable from cushion node ids (`cn_...`)."""
    norm = _normalize_content(content)
    return "cf_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def extract_domain(url: str) -> str:
    """Pull the registrable domain out of a URL.

    Best-effort; used by the matcher's evidence channel (Phase 4).
    Falls back to the raw netloc, then empty string."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url)
        netloc = (parsed.netloc or "").lower()
        # Strip leading www. — it's noise for source-credibility scoring.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Fingerprint extraction prompt
# ---------------------------------------------------------------------------


_FINGERPRINT_SYSTEM_PROMPT = """\
You are Constellax's structural fingerprint extractor.

You will be given a piece of CONTENT from some domain — could be a
Wikipedia article, a blog post, a movie scene, a paper, anything.

Your job: extract 2-5 short STRUCTURAL phrases that describe what this
content is structurally about, not topically.

# WHAT A STRUCTURAL PHRASE IS

A structural phrase captures dynamics, forces, constraints, cycles,
tensions, or causal mechanisms. It reads like prose, 8-20 words. Each
phrase is independent — a separate facet of the content's structure.

CRUCIAL: write each phrase in language that would naturally appear in
content from ANY domain exhibiting this structural pattern. If the
content is about jazz, your phrases should be domain-neutral enough
that they could also describe AI agents, kite flying, or political
movements exhibiting the same structure.

# EXAMPLES

Content: "A jazz pianist improvises within a fixed chord progression."
GOOD:
  - "creative output emerging within a fixed structural constraint"
  - "real-time choices governed by an underlying harmonic frame"
  - "freedom expressed through, not against, the boundary"
BAD (too topical — these would fail cross-domain matching):
  - "jazz improvisation techniques"
  - "chord progressions in modern music"
  - "piano performance"

Content: "Newton's laws of gravity attract masses across a vacuum."
GOOD:
  - "attractive force operating at a distance with no visible medium"
  - "magnitude proportional to mass, inversely to squared distance"
  - "universal law without exceptions or domain restrictions"
BAD:
  - "Newton's three laws of motion"
  - "gravitational physics fundamentals"

# WRITING DISCIPLINE

- 8-20 words per phrase. Phrases are independent statements, not
  parts of one paragraph.
- No domain-specific nouns when a structural noun suffices ("constraint"
  not "chord progression"; "node" not "musician").
- No fluff. Each word earns its place.
- 2-5 phrases per fingerprint. Quality over quantity.

# OUTPUT FORMAT

Return ONE valid JSON object:

{
  "phrases": [
    "phrase 1 here",
    "phrase 2 here",
    ...
  ]
}

No prose preamble. No code fences. JUST the JSON object.
"""


def _build_fingerprint_user_message(content: str, url: str, domain: str) -> str:
    """Compact user-message payload for the Haiku call."""
    blocks = []
    if url:
        blocks.append(f"# SOURCE\nURL: {url}\nDomain: {domain}")
    blocks.append("# CONTENT")
    blocks.append(content.strip()[:MAX_CONTENT_CHARS])
    blocks.append(
        "# TASK\n"
        "Extract 2-5 structural phrases per the system prompt's rules. "
        "Return JSON with a `phrases` array."
    )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def _extract_json_object(text: str) -> str:
    text = _strip_code_fences(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in fingerprint response")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    raise ValueError("unterminated JSON in fingerprint response")


def _parse_phrases(response_text: str) -> tuple[str, ...]:
    """Pull the phrase list out of Haiku's JSON response. Tolerant of
    extra keys, malformed entries, and outright junk — returns empty
    tuple on any parse failure so the caller can degrade gracefully."""
    try:
        json_text = _extract_json_object(response_text)
        payload = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("fingerprint response unparseable: %s", e)
        return ()

    if not isinstance(payload, dict):
        return ()
    phrases_raw = payload.get("phrases", [])
    if not isinstance(phrases_raw, list):
        return ()

    phrases: list[str] = []
    for p in phrases_raw:
        if not isinstance(p, (str, int, float)):
            continue
        s = str(p).strip()
        if s:
            phrases.append(s)
        if len(phrases) >= MAX_PHRASES:
            break

    if len(phrases) < MIN_PHRASES:
        log.debug("fingerprint produced %d phrases (need %d); leaving as-is",
                  len(phrases), MIN_PHRASES)
        # Caller decides whether to retry or accept a thin fingerprint.

    return tuple(phrases)


# ---------------------------------------------------------------------------
# Pure: fingerprint_content
# ---------------------------------------------------------------------------


async def fingerprint_content(
    content: str,
    url: str,
    *,
    client: LLMClient,
    embedder: GeminiEmbeddingService | None = None,
) -> ContentFingerprint:
    """Extract a structural fingerprint from raw content.

    Pure function — no Neo4j I/O. Used directly when you don't need
    caching (tests, one-off analysis) or as the cache-miss branch of
    `get_or_create_fingerprint`.

    Steps:
      1. Compute content_hash (cache key, even though this function
         doesn't use the cache itself — callers may pass the hash
         along for upstream caching)
      2. Extract domain from URL
      3. Call Haiku to produce 2-5 structural phrases
      4. Embed the joined phrases via Gemini
      5. Return ContentFingerprint

    Empty content -> ContentFingerprint with empty phrases + None
    embedding. The matcher will skip this fingerprint cleanly.
    """
    content_hash = compute_content_hash(content)
    domain = extract_domain(url)

    if not content or not content.strip():
        return ContentFingerprint(
            content_hash=content_hash, url=url, domain=domain,
            phrases=(), phrases_combined="",
            embedding=None, created_at=time.time(), model_used="",
        )

    user_message = _build_fingerprint_user_message(content, url, domain)

    response: LLMResponse | None = None
    try:
        response = await client.call(
            system_prompt=compose_system_prompt(
                _FINGERPRINT_SYSTEM_PROMPT, mode="content_fingerprint"),
            user_message=user_message,
            domain=FINGERPRINT_DOMAIN,
            concept=FINGERPRINT_CONCEPT,
        )
    except Exception as e:
        log.warning("fingerprint LLM call raised: %s", e)

    phrases: tuple[str, ...] = ()
    model_used = ""
    if response and response.success and response.content:
        phrases = _parse_phrases(response.content)
        model_used = response.model or ""
    elif response and not response.success:
        log.warning("fingerprint LLM call failed: %s",
                    response.error or "unknown")

    phrases_combined = " | ".join(phrases) if phrases else ""

    # Embed the combined phrases. If embedding fails (no API key,
    # network, etc.), leave embedding as None — matcher's vector
    # channel will skip this fingerprint; other channels still work.
    embedding: list[float] | None = None
    if phrases_combined:
        try:
            emb = embedder or GeminiEmbeddingService()
            result: EmbeddingResult = await emb.embed(phrases_combined)
            if result.success and result.vector:
                embedding = result.vector
            else:
                log.warning("fingerprint embed failed: %s",
                            result.error or "no_vector")
        except Exception as e:
            log.warning("fingerprint embed raised: %s", e)

    return ContentFingerprint(
        content_hash=content_hash,
        url=url,
        domain=domain,
        phrases=phrases,
        phrases_combined=phrases_combined,
        embedding=embedding,
        created_at=time.time(),
        model_used=model_used,
    )


# ---------------------------------------------------------------------------
# Cached: get_or_create_fingerprint
# ---------------------------------------------------------------------------


async def _lookup_fingerprint(
    content_hash: str,
    driver: Any,
    database: str,
) -> ContentFingerprint | None:
    """Read a fingerprint from Neo4j by content_hash. Returns None on
    miss or any failure (network, parse, missing fields)."""
    cypher = """
    MATCH (f:ContentFingerprint {content_hash: $hash})
    RETURN f.content_hash AS content_hash,
           f.url AS url,
           f.domain AS domain,
           f.phrases AS phrases,
           f.phrases_combined AS phrases_combined,
           f.embedding AS embedding,
           f.created_at AS created_at,
           f.model_used AS model_used
    LIMIT 1
    """
    try:
        async with driver.session(database=database) as sess:
            result = await sess.run(cypher, hash=content_hash)
            rec = await result.single()
    except Exception as e:
        log.debug("fingerprint cache lookup failed: %s", e)
        return None

    if rec is None:
        return None

    return ContentFingerprint(
        content_hash=rec.get("content_hash", content_hash),
        url=rec.get("url") or "",
        domain=rec.get("domain") or "",
        phrases=tuple(rec.get("phrases") or ()),
        phrases_combined=rec.get("phrases_combined") or "",
        embedding=list(rec.get("embedding")) if rec.get("embedding") else None,
        created_at=float(rec.get("created_at") or 0.0),
        model_used=rec.get("model_used") or "",
    )


async def _persist_fingerprint(
    fp: ContentFingerprint,
    driver: Any,
    database: str,
) -> None:
    """Upsert a fingerprint to Neo4j. Failures logged, never raised —
    the in-memory fingerprint is still usable downstream even if
    persistence missed."""
    cypher = """
    MERGE (f:ContentFingerprint {content_hash: $content_hash})
    SET f.url              = $url,
        f.domain           = $domain,
        f.phrases          = $phrases,
        f.phrases_combined = $phrases_combined,
        f.embedding        = $embedding,
        f.created_at       = $created_at,
        f.model_used       = $model_used
    """
    try:
        async with driver.session(database=database) as sess:
            await sess.run(
                cypher,
                content_hash=fp.content_hash,
                url=fp.url,
                domain=fp.domain,
                phrases=list(fp.phrases),
                phrases_combined=fp.phrases_combined,
                embedding=fp.embedding,
                created_at=fp.created_at,
                model_used=fp.model_used,
            )
    except Exception as e:
        log.warning("fingerprint persist failed for %s: %s",
                    fp.content_hash, e)


async def get_or_create_fingerprint(
    content: str,
    url: str,
    *,
    client: LLMClient,
    embedder: GeminiEmbeddingService | None = None,
    neo4j_driver: Any | None = None,
    neo4j_database: str = "neo4j",
    force_refresh: bool = False,
) -> ContentFingerprint:
    """Cached fingerprint extraction.

    Steps:
      1. Compute content_hash. If `neo4j_driver` is None, skip cache
         entirely and just call `fingerprint_content` — useful for
         tests and isolated calls.
      2. Look up `(:ContentFingerprint {content_hash})` in Neo4j.
         If hit and `force_refresh=False`, return it.
      3. On miss (or force refresh), call `fingerprint_content` to
         generate fresh. Persist before returning.

    Caching keys ONLY on content. URL is recorded but doesn't
    participate in the cache key — same content at two URLs hits the
    same cached fingerprint. The `url` field of the returned
    fingerprint reflects the URL passed to THIS call, not the URL of
    the original cached entry.
    """
    if neo4j_driver is None:
        # No cache available; fall through to pure generation.
        return await fingerprint_content(
            content, url, client=client, embedder=embedder,
        )

    content_hash = compute_content_hash(content)

    if not force_refresh:
        cached = await _lookup_fingerprint(content_hash, neo4j_driver, neo4j_database)
        if cached is not None and cached.phrases:
            # Recovered the cached entry. We DO refresh `url` to reflect
            # the current caller's source, but keep all other fields from
            # the cache (phrases, embedding, etc.).
            if url and not cached.url:
                cached.url = url
            log.debug("fingerprint cache hit: %s", content_hash)
            return cached

    fresh = await fingerprint_content(
        content, url, client=client, embedder=embedder,
    )
    # Only persist if there's something worth caching — empty phrases
    # means generation failed, no point caching a degraded entry.
    if fresh.phrases:
        await _persist_fingerprint(fresh, neo4j_driver, neo4j_database)
    return fresh


__all__ = [
    "MIN_PHRASES",
    "MAX_PHRASES",
    "MAX_CONTENT_CHARS",
    "FINGERPRINT_DOMAIN",
    "FINGERPRINT_CONCEPT",
    "ContentFingerprint",
    "compute_content_hash",
    "extract_domain",
    "fingerprint_content",
    "get_or_create_fingerprint",
]
