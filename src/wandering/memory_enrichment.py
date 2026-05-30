"""
Memory enrichment — pull relevant project context from the user's memory graph.

When the user starts a Wandering Room session, the cushion they provide
in the form gets automatically enriched with relevant context from their
project memory (Neo4j graph) — recent threads, salient entities, ongoing
decisions. This is transparent (no explicit permission, it's the user's
own memory) and BEST-EFFORT (if memory is unavailable, the cushion is
built without it and the session still works).

Per Law 4: read-only. We never write to the user's memory from here.

Per the plan (V0.2, auto-enrichment correction): this is the wiring point
for the "Your Current Map" field's automatic dimension.

ISOLATION: imports thread_store + cushion types. Defensive everywhere —
any failure returns empty enrichment, never raises.
"""

from __future__ import annotations

import logging
from typing import Any

from src.wandering.cushion import CushionInput


import asyncio as _asyncio
import time as _time


# Per-user TTL cache. Wandering /brief is called repeatedly during a
# session (re-runs, dig-deeper, etc); without a cache each call would
# fetch the user's recent threads from Neo4j fresh. With a 60-second
# TTL the cushion composer reads warm memory, and the user's project
# state still feels "current" — refreshed automatically the next time
# the cache window expires.
_ENRICHMENT_TTL_SECONDS = 60
_ENRICHMENT_CACHE: dict[str, tuple[float, str]] = {}
_ENRICHMENT_LOCK = _asyncio.Lock()


log = logging.getLogger("constellax.wandering.memory_enrichment")


# How many recent threads to summarize into the enrichment block. Above
# 5 the context block grows large; below 3 we lose useful signal.
MAX_THREADS_FOR_ENRICHMENT = 5

# How many entities/tags to surface per thread.
MAX_ENTITIES_PER_THREAD = 3
MAX_TAGS_PER_THREAD = 3

# Cap on total enrichment text length — keeps the Sonnet extraction prompt
# from ballooning when the user has lots of recent activity.
MAX_ENRICHMENT_CHARS = 1500


def _summarize_thread(thread: Any) -> str:
    """Build a one-line summary of a thread for the enrichment block.

    Defensive — accepts anything ThreadRecord-shaped. Pulls title +
    entities + tags if present. Falls back to thread_id only when fields
    are missing.
    """
    title = (getattr(thread, "title", "") or "").strip()
    if not title:
        title = f"thread:{getattr(thread, 'id', 'unknown')}"

    bits: list[str] = [title]

    entities = getattr(thread, "entities", None) or []
    if entities:
        # entities may be a list of strings or a list of objects with .name
        entity_strs: list[str] = []
        for e in entities[:MAX_ENTITIES_PER_THREAD]:
            if isinstance(e, str):
                entity_strs.append(e)
            else:
                name = getattr(e, "name", None) or getattr(e, "text", None)
                if name:
                    entity_strs.append(str(name))
        if entity_strs:
            bits.append(f"entities: {', '.join(entity_strs)}")

    tags = getattr(thread, "tags", None) or []
    if tags:
        tag_strs = [str(t) for t in tags[:MAX_TAGS_PER_THREAD] if t]
        if tag_strs:
            bits.append(f"tags: {', '.join(tag_strs)}")

    return " — ".join(bits)


async def _try_build_thread_store() -> Any | None:
    """Build the thread store from env. Returns None on any failure.

    We defensively wrap the build so import-time failures don't crash
    the wandering composer. If the thread store can't be created
    (Neo4j not configured, Redis down, etc.), memory enrichment simply
    returns empty.
    """
    try:
        from src.bridge.thread_store import build_thread_store_from_env
        store = build_thread_store_from_env()
        return store
    except Exception as e:
        log.debug("thread_store unavailable for enrichment: %s", e)
        return None


async def fetch_memory_enrichment_real(user_id: str | None) -> str:
    """Pull relevant project memory and format as enrichment text.

    Returns an empty string on ANY failure or when no useful context is
    available. The caller (composer.compose_cushion) treats empty
    enrichment as "no extra context" — the Sonnet extraction handles it.

    Phase 0 used a no-op stub. This is the V1 real implementation.

    Future enhancement: use the user's brief text to embed-search for
    SIMILAR iterations (not just recent ones). For V1 we list the
    user's most recent threads — simpler, predictable, useful enough.
    """
    if not user_id:
        return ""

    async with _ENRICHMENT_LOCK:
        cached = _ENRICHMENT_CACHE.get(user_id)
        if cached is not None and (_time.time() - cached[0]) < _ENRICHMENT_TTL_SECONDS:
            return cached[1]

    store = await _try_build_thread_store()
    if store is None:
        async with _ENRICHMENT_LOCK:
            _ENRICHMENT_CACHE[user_id] = (_time.time(), "")
        return ""

    try:
        # Get the user's most recent threads. ThreadStore.list_threads
        # signature varies by backend; defensive call with safe defaults.
        list_fn = getattr(store, "list_threads", None)
        if list_fn is None:
            return ""

        threads = await list_fn(
            user_id=user_id,
            limit=MAX_THREADS_FOR_ENRICHMENT,
        )
    except Exception as e:
        log.debug("list_threads failed for user_id=%s: %s", user_id, e)
        async with _ENRICHMENT_LOCK:
            _ENRICHMENT_CACHE[user_id] = (_time.time(), "")
        return ""

    if not threads:
        async with _ENRICHMENT_LOCK:
            _ENRICHMENT_CACHE[user_id] = (_time.time(), "")
        return ""

    lines: list[str] = [
        "Recent project activity from this user's memory graph "
        "(use as background context, do not lead with it):"
    ]
    for t in threads[:MAX_THREADS_FOR_ENRICHMENT]:
        try:
            lines.append(f"- {_summarize_thread(t)}")
        except Exception as e:
            log.debug("thread summary failed: %s", e)
            continue

    enrichment = "\n".join(lines)
    if len(enrichment) > MAX_ENRICHMENT_CHARS:
        enrichment = enrichment[:MAX_ENRICHMENT_CHARS] + "\n[... truncated]"
    async with _ENRICHMENT_LOCK:
        _ENRICHMENT_CACHE[user_id] = (_time.time(), enrichment)
    return enrichment


async def enrich_cushion_input(
    input_data: CushionInput,
    user_id: str | None,
) -> None:
    """Mutates `input_data.memory_enrichment` in place with project context.

    Called by composer.compose_cushion when auto_enrich=True. The
    composer's existing behavior is preserved: if enrichment is already
    set (non-empty), we don't overwrite it.

    Why mutate vs return: the composer expects the enrichment to land
    on the CushionInput object directly so audit can show the user what
    context was injected. Returning the string would require composer
    changes that aren't worth the API churn.
    """
    if input_data.memory_enrichment.strip():
        return  # already set, respect it
    input_data.memory_enrichment = await fetch_memory_enrichment_real(user_id)


def clear_memory_enrichment_cache() -> None:
    """Drop the per-user TTL cache. Test-only helper."""
    _ENRICHMENT_CACHE.clear()


__all__ = [
    "MAX_THREADS_FOR_ENRICHMENT",
    "MAX_ENRICHMENT_CHARS",
    "fetch_memory_enrichment_real",
    "enrich_cushion_input",
    "clear_memory_enrichment_cache",
]
