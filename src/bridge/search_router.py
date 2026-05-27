"""
search_router — LLM-as-router for adaptive web search.

THE PURPOSE
===========
The first version of web search used a regex heuristic to decide
when to fire a search. It's fast but dumb: it misses cases like
"what does my user research suggest about Gen-Z workflows" (which
benefits from search) and over-fires on cases that mention a
trigger word incidentally.

This module replaces the regex with a one-shot LLM classifier that:

  1. Decides: needs_search ∈ {true, false}
  2. Rewrites: refined_query — the actual phrase to search for
     (often very different from the user's question)
  3. Explains: reason — one sentence why, surfaced in the
     Reasoning Trace so the user sees the decision

We use the same Gemini 2.5 Flash that already powers metadata
extraction in the persistence layer — no new model, no new auth,
no new dependency. Cost: ~$0.00005 per question, latency ~200-400ms.

FAIL-SAFE
=========
If the LLM call fails (timeout, parse error, no API key), we fall
back to the regex heuristic from web_search.py. The trace never
breaks because the router failed — at worst we degrade to v1
behavior.

OPEN-SOURCE SWAP
================
The router is one async function with a typed return. Swapping in a
local classifier (DistilBERT, a small Llama, an embedding-similarity
scorer) means replacing `route_via_llm()` with the new implementation
and leaving the caller untouched. The function signature is
intentionally narrow so any binary "needs_search" classifier can plug
in here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass

from src.bridge.web_search import should_search as _regex_should_search

log = logging.getLogger("constellax.search_router")

ROUTER_MODEL = os.environ.get("SEARCH_ROUTER_MODEL", "gemini-2.5-flash")
# Bumped from 6s → 12s. Real-world Gemini 2.5 Flash latency on
# multi-paragraph questions runs 5-8s (the metadata_extraction call in
# the persistence layer regularly takes ~7s for similar input sizes).
# A tight timeout silently dropped good LLM-router decisions and forced
# the heuristic fallback, which historically leaked the raw question
# as the search query.
ROUTER_TIMEOUT_SEC = 12.0
MAX_QUESTION_CHARS = 4000  # safety bound on the prompt
# Cap on the heuristic-distilled fallback query. Tavily caps at ~400
# chars (and bills per token); DDG breaks on long queries too.
HEURISTIC_QUERY_MAX_CHARS = 160


@dataclass
class SearchDecision:
    """Outcome of routing a question to (maybe) web search."""
    needs_search:   bool
    refined_query:  str         # the actual query the search backend sees
    reason:         str         # one-sentence rationale (UI surfaces this)
    via:            str         # "llm" | "heuristic" | "fallback"
    latency_ms:     int = 0
    model:          str | None = None


# ─── Prompt ──────────────────────────────────────────────────────────

_ROUTER_PROMPT = """You decide whether a user's question needs live web search before an AI assistant answers it.

Return ONLY valid JSON with exactly these keys:
  needs_search:   boolean
  refined_query:  string  (the best search-engine query to find the answer; rewrite for search-engine grammar, not chat grammar)
  reason:         string  (one short sentence — why search or why not)

Decide TRUE when the question:
  - References current/recent events, releases, prices, policies, or news
  - Asks about specific external documents (terms of service, privacy policies, API docs)
  - Contains a URL the assistant would need to read
  - References named people / products / companies whose state changes over time
  - Asks "what does X say about Y" where X is an external source
  - Asks for citations, sources, or verification of a claim

Decide FALSE when the question:
  - Is conversational ("how are you", "thanks")
  - Asks for general explanation of a stable concept (transformers, sorting algorithms, philosophy)
  - Is a code/syntax question
  - Is reflective/personal ("should I quit my job") with no factual ground-truth requirement
  - Is so vague that no useful query exists

The refined_query should be 2-8 words optimized for search-engine ranking. Drop filler words, expand acronyms, prefer named entities.

Question:
{question}
"""


# ─── Public: route() ─────────────────────────────────────────────────

async def route(question: str) -> SearchDecision:
    """Decide whether to fire web search for this question.

    Tries the LLM router first; falls back to the regex heuristic on
    any failure. The caller always gets a SearchDecision."""
    start = time.time()
    q = (question or "").strip()[:MAX_QUESTION_CHARS]
    if not q:
        return SearchDecision(
            needs_search=False, refined_query="", reason="empty question",
            via="heuristic", latency_ms=0,
        )

    # LLM router (best path)
    try:
        decision = await _route_via_llm(q)
        if decision is not None:
            decision.latency_ms = int((time.time() - start) * 1000)
            return decision
    except Exception as e:
        log.warning("LLM router crashed (falling back to heuristic): %s", e)

    # Heuristic fallback. The LLM router couldn't produce a refined
    # query, so we distill one ourselves. NEVER pass the raw question
    # through — Tavily caps at ~400 chars, DDG breaks on long queries.
    needs = _regex_should_search(q)
    distilled = _distill_query(q) if needs else ""
    return SearchDecision(
        needs_search=needs,
        refined_query=distilled,
        reason=(
            f"regex heuristic matched a trigger word; "
            f"query distilled from raw question ({len(distilled)} chars)"
            if needs
            else "regex heuristic found no triggers"
        ),
        via="heuristic" if not _llm_available() else "fallback",
        latency_ms=int((time.time() - start) * 1000),
    )


def _llm_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


# ─── Heuristic query distiller (fallback only) ───────────────────────
# When the LLM router can't run (timeout, no key, parse fail), we used
# to pass the entire user question through as the search query. That
# routinely sent 2000+ char rambling prompts to Tavily (which caps at
# ~400 chars) and DDG (which then 302's into a rate-limit page).
# This distiller produces a search-engine-shaped query without any
# LLM call: strips fillers, drops conversational openers, picks the
# most informative chunk, hard-caps at HEURISTIC_QUERY_MAX_CHARS.

_FILLER_TOKENS = {
    "uh", "um", "like", "you", "know", "i", "mean", "actually",
    "basically", "literally", "so", "well", "okay", "ok", "yeah",
    "right", "kinda", "sorta", "just", "really", "very", "stuff",
    "thing", "things", "the", "a", "an", "is", "are", "was", "were",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "with",
    "we", "us", "our", "my", "me", "your", "their", "this", "that",
    "these", "those", "it", "be", "as", "by", "from", "have", "has",
    "had", "do", "does", "did", "can", "could", "should", "would",
    "might", "may", "will", "shall",
}

_CONVERSATIONAL_OPENERS = (
    r"^\s*(hey|hi|hello|yo|sup)[\.,!\s]+",
    r"^\s*(so|well|okay|ok|right)[\.,!\s]+",
    r"^\s*(tell me one thing|let me ask|let me think|i was thinking|"
    r"i wanna know|i want to know|i wanted to ask|i'm asking|"
    r"can you tell me|can you explain|please explain|please tell)[\.,!\s]+",
)


def _distill_query(question: str) -> str:
    """Best-effort heuristic distillation: long rambling question →
    short search-friendly query. No LLM call, fully deterministic.

    Strategy:
      1. Drop conversational openers (Hey, Tell me one thing, So,…)
      2. Pick the most information-dense sentence (highest ratio of
         non-filler content words to total words). Long rambling
         questions tend to have one substantive sentence buried in
         conversational filler; this surfaces it.
      3. Strip filler words, collapse whitespace.
      4. Hard-cap at HEURISTIC_QUERY_MAX_CHARS at a word boundary.

    Returns "" when nothing salvageable remains (caller should not
    fire search in that case).
    """
    if not question:
        return ""
    q = question.strip()
    # Drop conversational openers.
    for pat in _CONVERSATIONAL_OPENERS:
        q = re.sub(pat, "", q, count=1, flags=re.IGNORECASE)

    # Split into sentences (rough — we don't need real grammar parsing).
    sentences = [s.strip() for s in re.split(r"[.!?]+", q) if s.strip()]
    if not sentences:
        return ""

    def _density(s: str) -> float:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-']*", s.lower())
        if not words:
            return 0.0
        content = sum(1 for w in words if w not in _FILLER_TOKENS and len(w) > 2)
        return content / len(words)

    # Pick the densest sentence; ties broken by length (richer content).
    best = max(sentences, key=lambda s: (_density(s), len(s)))

    # Final cleanup: collapse whitespace, drop trailing punctuation.
    out = re.sub(r"\s+", " ", best).strip(" ,;:-—")

    # Hard cap at word boundary.
    if len(out) > HEURISTIC_QUERY_MAX_CHARS:
        cut = out[:HEURISTIC_QUERY_MAX_CHARS].rsplit(" ", 1)[0]
        out = cut if cut else out[:HEURISTIC_QUERY_MAX_CHARS]
    return out


async def _route_via_llm(question: str) -> SearchDecision | None:
    """One-shot Gemini Flash call. Returns None if the call can't run
    (no key, SDK missing) so the caller can fall through cleanly."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types as genai_types  # noqa: F401
    except ImportError:
        log.warning("google-genai SDK not available — router falling back")
        return None

    client = genai.Client(api_key=api_key)
    prompt = _ROUTER_PROMPT.format(question=question)

    call_start = time.time()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=ROUTER_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    # Keep output tiny — the router only emits three short fields.
                    "max_output_tokens": 256,
                    # Low temperature for consistent routing across re-runs.
                    "temperature": 0.0,
                },
            ),
            timeout=ROUTER_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        # Promoted to INFO-level so we can SEE when the timeout is the
        # cause (the trace reports "via=fallback" but doesn't tell us
        # which failure mode hit).
        elapsed = time.time() - call_start
        log.info(
            "search_router: LLM call TIMED OUT after %.1fs (limit=%ss) — "
            "falling back to heuristic distiller",
            elapsed, ROUTER_TIMEOUT_SEC,
        )
        return None
    except Exception as e:
        log.info(
            "search_router: LLM call FAILED (%s: %s) — falling back",
            type(e).__name__, e,
        )
        return None

    raw = getattr(response, "text", None) or ""
    parsed = _parse_router_json(raw)
    if parsed is None:
        log.info(
            "search_router: LLM emitted UNPARSEABLE JSON (raw[:200]=%r) — falling back",
            raw[:200],
        )
        return None

    return SearchDecision(
        needs_search=bool(parsed.get("needs_search", False)),
        refined_query=str(parsed.get("refined_query", "")).strip()[:200],
        reason=str(parsed.get("reason", "")).strip()[:280],
        via="llm",
        model=ROUTER_MODEL,
    )


_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_router_json(raw: str) -> dict | None:
    """Defensively extract the JSON object from the router's output."""
    if not raw:
        return None
    txt = raw.strip()
    # Strip ```json fences if present.
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict): return obj
    except json.JSONDecodeError:
        pass
    # Fall back to first {...} block in the raw text.
    m = _JSON_BLOCK_RE.search(txt)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
