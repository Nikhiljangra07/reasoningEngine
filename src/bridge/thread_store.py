"""
ThreadStore — repository-pattern interface for ThreadRecord + IterationRecord
persistence.

The application layer (server.py, dispatcher.py) ONLY talks to the ThreadStore
protocol. It never imports redis, FalkorDB, Postgres, or any storage client.
That means: when we switch backends (or add a vector DB for embeddings, or
fan out KV/vector/graph across three different stores), only this file and
the new implementation file change. Nothing else in the codebase moves.

CURRENT STATE
=============
Active backend: FalkorThreadStore (against constellax-falkor on port 6379,
Redis-protocol KV with optional graph layer via GRAPH.QUERY commands).
In-memory backend exists for tests.

KEY LAYOUT (Redis namespace)
============================
All keys live under `constellax:` for product isolation. The layout below
is the contract:

    constellax:thread:<thread_id>                STRING   — ThreadRecord JSON
    constellax:iteration:<iter_id>               STRING   — IterationRecord JSON
    constellax:thread:<thread_id>:iterations     LIST     — ordered iter_ids
    constellax:threads:by_user:<user_id>         ZSET     — score=created_at, member=thread_id
    constellax:threads:by_project:<project_id>   ZSET     — same
    constellax:threads:by_workspace:<workspace>  ZSET     — same
    constellax:threads:by_entity:<entity_name>   SET      — thread_ids mentioning an entity
    constellax:threads:by_tag:<tag>              SET      — thread_ids carrying a tag
    constellax:embedding:<iter_id>               STRING   — JSON list of floats
    constellax:embedding:index                   SET      — all iter_ids with embeddings (for brute-force scan)
    constellax:thread:<thread_id>:summary_lock   STRING   — soft lock to prevent concurrent summary regen

The store is conservative: writes are best-effort and idempotent. Read paths
never raise on missing data — they return None / empty list. The dispatcher
hook wraps `save_*` calls in try/except so persistence failures do not bubble
into the user-facing response.

SAFETY
======
- All writes pipeline-batched per save call (atomic at the Redis pipeline level).
- Reads guard against malformed JSON and missing keys.
- TTLs are NOT set today (we keep everything). When you want auto-expiry,
  configure it at the ThreadStore level — single file change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any, Protocol

from src.core.thread_types import (
    IterationRecord,
    ThreadRecord,
    OutcomeRecord,
)

log = logging.getLogger("constellax.thread_store")


# ─── Key naming helpers (keep all string formatting in one place) ─────

_PREFIX = "constellax"

def k_thread(thread_id: str) -> str:           return f"{_PREFIX}:thread:{thread_id}"
def k_iteration(iter_id: str) -> str:           return f"{_PREFIX}:iteration:{iter_id}"
def k_thread_iter_list(thread_id: str) -> str:  return f"{_PREFIX}:thread:{thread_id}:iterations"
def k_threads_by_user(user_id: str) -> str:     return f"{_PREFIX}:threads:by_user:{user_id}"
def k_threads_by_project(p: str) -> str:        return f"{_PREFIX}:threads:by_project:{p}"
def k_threads_by_workspace(w: str) -> str:      return f"{_PREFIX}:threads:by_workspace:{w}"
def k_threads_by_entity(e: str) -> str:         return f"{_PREFIX}:threads:by_entity:{e.lower()}"
def k_threads_by_tag(t: str) -> str:            return f"{_PREFIX}:threads:by_tag:{t.lower()}"
def k_embedding(iter_id: str) -> str:           return f"{_PREFIX}:embedding:{iter_id}"
def k_embedding_index() -> str:                 return f"{_PREFIX}:embedding:index"


GUEST_BUCKET = "guest"  # fallback user_id for anonymous threads


# ─── Protocol — the contract every backend must satisfy ───────────────

class ThreadStore(Protocol):
    """Abstract repository. Application code only talks to this interface."""

    async def save_thread(self, thread: ThreadRecord) -> None: ...

    async def get_thread(self, thread_id: str) -> ThreadRecord | None: ...

    async def list_threads(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThreadRecord]: ...

    async def save_iteration(self, iteration: IterationRecord) -> None: ...

    async def get_iteration(self, iter_id: str) -> IterationRecord | None: ...

    async def list_iterations_for_thread(self, thread_id: str) -> list[IterationRecord]: ...

    async def find_similar_iterations(
        self, embedding: list[float], k: int = 5, exclude_iter_id: str | None = None,
    ) -> list[tuple[float, IterationRecord]]: ...

    async def find_threads_mentioning_entity(self, entity_name: str) -> list[str]: ...
    async def find_threads_by_tag(self, tag: str) -> list[str]: ...

    async def attach_outcome(self, iter_id: str, outcome: OutcomeRecord) -> bool: ...

    async def delete_thread(self, thread_id: str) -> bool: ...


# ─── FalkorThreadStore — production implementation ─────────────────────

class FalkorThreadStore:
    """Redis-protocol implementation of ThreadStore, targeting FalkorDB.

    Holds a redis.asyncio.Redis client. Does NOT use FalkorDB's graph commands
    in this file — those live in a separate adapter that's wired up later.
    This file is plain KV + sorted-sets + sets, so it works equally against
    vanilla Redis if you ever switch."""

    def __init__(self, client: Any) -> None:
        self._r = client

    # ─── Threads ────────────────────────────────────────

    async def save_thread(self, thread: ThreadRecord) -> None:
        thread.updated_at = time.time()
        if not thread.created_at:
            thread.created_at = thread.updated_at
        payload = json.dumps(thread.to_payload())
        user_bucket = thread.user_id or GUEST_BUCKET

        pipe = self._r.pipeline(transaction=False)
        pipe.set(k_thread(thread.id), payload)
        pipe.zadd(k_threads_by_user(user_bucket), {thread.id: thread.created_at})
        if thread.project_id:
            pipe.zadd(k_threads_by_project(thread.project_id), {thread.id: thread.created_at})
        if thread.workspace_id:
            pipe.zadd(k_threads_by_workspace(thread.workspace_id), {thread.id: thread.created_at})
        for ent in thread.all_entities:
            pipe.sadd(k_threads_by_entity(ent), thread.id)
        for tag in thread.all_tags:
            pipe.sadd(k_threads_by_tag(tag), thread.id)
        await pipe.execute()

    async def get_thread(self, thread_id: str) -> ThreadRecord | None:
        raw = await self._r.get(k_thread(thread_id))
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("thread %s has malformed JSON; ignoring", thread_id)
            return None
        return ThreadRecord.from_payload(data)

    async def list_threads(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThreadRecord]:
        # Pick the most specific index available; fall back to user bucket.
        if workspace_id:
            zset_key = k_threads_by_workspace(workspace_id)
        elif project_id:
            zset_key = k_threads_by_project(project_id)
        else:
            zset_key = k_threads_by_user(user_id or GUEST_BUCKET)

        # ZSet is sorted by created_at; we want newest first → ZREVRANGE.
        thread_ids_raw = await self._r.zrevrange(zset_key, offset, offset + limit - 1)
        thread_ids = [tid.decode() if isinstance(tid, bytes) else tid for tid in (thread_ids_raw or [])]
        if not thread_ids:
            return []

        # MGET the records in parallel
        raws = await self._r.mget([k_thread(t) for t in thread_ids])
        out: list[ThreadRecord] = []
        for raw in raws:
            if not raw:
                continue
            try:
                out.append(ThreadRecord.from_payload(json.loads(raw)))
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("malformed thread record: %s", e)
        return out

    async def delete_thread(self, thread_id: str) -> bool:
        thread = await self.get_thread(thread_id)
        if not thread:
            return False
        # Source of truth for "what iterations belong to this thread" is the
        # iteration list key, NOT thread.iteration_ids — the list key gets
        # updated atomically on every save_iteration, the denormalized field
        # on the thread can drift if a thread is saved before its iterations.
        iter_ids_raw = await self._r.lrange(k_thread_iter_list(thread_id), 0, -1)
        iter_ids = [i.decode() if isinstance(i, bytes) else i for i in (iter_ids_raw or [])]

        user_bucket = thread.user_id or GUEST_BUCKET
        pipe = self._r.pipeline(transaction=False)
        pipe.delete(k_thread(thread_id))
        pipe.delete(k_thread_iter_list(thread_id))
        pipe.zrem(k_threads_by_user(user_bucket), thread_id)
        if thread.project_id:
            pipe.zrem(k_threads_by_project(thread.project_id), thread_id)
        if thread.workspace_id:
            pipe.zrem(k_threads_by_workspace(thread.workspace_id), thread_id)
        for ent in thread.all_entities:
            pipe.srem(k_threads_by_entity(ent), thread_id)
        for tag in thread.all_tags:
            pipe.srem(k_threads_by_tag(tag), thread_id)
        # Cascade-delete iterations + embeddings using the authoritative list
        for iter_id in iter_ids:
            pipe.delete(k_iteration(iter_id))
            pipe.delete(k_embedding(iter_id))
            pipe.srem(k_embedding_index(), iter_id)
        await pipe.execute()
        return True

    # ─── Iterations ────────────────────────────────────────

    async def save_iteration(self, iteration: IterationRecord) -> None:
        if not iteration.created_at:
            iteration.created_at = time.time()
        payload = json.dumps(iteration.to_payload())

        pipe = self._r.pipeline(transaction=False)
        pipe.set(k_iteration(iteration.id), payload)
        if iteration.thread_id:
            # rpush so order matches sequence_num for sequential turns
            pipe.rpush(k_thread_iter_list(iteration.thread_id), iteration.id)
        if iteration.embedding is not None:
            pipe.set(k_embedding(iteration.id), json.dumps(iteration.embedding))
            pipe.sadd(k_embedding_index(), iteration.id)
        await pipe.execute()

    async def get_iteration(self, iter_id: str) -> IterationRecord | None:
        raw = await self._r.get(k_iteration(iter_id))
        if not raw:
            return None
        try:
            return IterationRecord.from_payload(json.loads(raw))
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("iteration %s malformed: %s", iter_id, e)
            return None

    async def list_iterations_for_thread(self, thread_id: str) -> list[IterationRecord]:
        ids_raw = await self._r.lrange(k_thread_iter_list(thread_id), 0, -1)
        ids = [i.decode() if isinstance(i, bytes) else i for i in (ids_raw or [])]
        if not ids:
            return []
        raws = await self._r.mget([k_iteration(i) for i in ids])
        out: list[IterationRecord] = []
        for raw in raws:
            if not raw:
                continue
            try:
                out.append(IterationRecord.from_payload(json.loads(raw)))
            except (json.JSONDecodeError, ValueError) as e:
                log.warning("malformed iteration in thread %s: %s", thread_id, e)
        out.sort(key=lambda it: it.sequence_num)
        return out

    # ─── Outcome attachment ────────────────────────────────────────

    async def attach_outcome(self, iter_id: str, outcome: OutcomeRecord) -> bool:
        """Bolt an OutcomeRecord onto an existing iteration. Used by the
        POST /api/v2/iteration/<id>/outcome endpoint when user reports back."""
        existing = await self.get_iteration(iter_id)
        if not existing:
            return False
        existing.outcome_followup = outcome
        await self.save_iteration(existing)
        return True

    # ─── Similarity search (brute-force; suitable up to ~10k iterations) ────

    async def find_similar_iterations(
        self,
        embedding: list[float],
        k: int = 5,
        exclude_iter_id: str | None = None,
    ) -> list[tuple[float, IterationRecord]]:
        """Cosine-similarity scan over all stored embeddings.

        SCALE NOTE: brute-force is fine up to ~10k records. Past that, swap
        in FalkorDB's native vector index or move embeddings to a dedicated
        vector store (Chroma, Pinecone, pgvector). The application code never
        sees this change — only this method's body."""
        all_ids_raw = await self._r.smembers(k_embedding_index())
        all_ids = [i.decode() if isinstance(i, bytes) else i for i in (all_ids_raw or [])]
        if exclude_iter_id:
            all_ids = [i for i in all_ids if i != exclude_iter_id]
        if not all_ids:
            return []

        # Fetch all embeddings in one MGET, score, sort.
        emb_blobs = await self._r.mget([k_embedding(i) for i in all_ids])
        scored: list[tuple[float, str]] = []
        q_norm = _l2(embedding)
        if q_norm == 0:
            return []
        for i_id, blob in zip(all_ids, emb_blobs):
            if not blob:
                continue
            try:
                vec = json.loads(blob)
            except json.JSONDecodeError:
                continue
            n = _l2(vec)
            if n == 0 or len(vec) != len(embedding):
                continue
            score = _dot(embedding, vec) / (q_norm * n)
            scored.append((score, i_id))

        scored.sort(reverse=True, key=lambda t: t[0])
        top = scored[:k]
        if not top:
            return []

        # Hydrate the top records
        raws = await self._r.mget([k_iteration(i) for _, i in top])
        out: list[tuple[float, IterationRecord]] = []
        for (score, i_id), raw in zip(top, raws):
            if not raw:
                continue
            try:
                rec = IterationRecord.from_payload(json.loads(raw))
                out.append((score, rec))
            except (json.JSONDecodeError, ValueError):
                continue
        return out

    # ─── Entity / tag lookups ────────────────────────────────────────

    async def find_threads_mentioning_entity(self, entity_name: str) -> list[str]:
        ids = await self._r.smembers(k_threads_by_entity(entity_name))
        return [i.decode() if isinstance(i, bytes) else i for i in (ids or [])]

    async def find_threads_by_tag(self, tag: str) -> list[str]:
        ids = await self._r.smembers(k_threads_by_tag(tag))
        return [i.decode() if isinstance(i, bytes) else i for i in (ids or [])]


# ─── InMemoryThreadStore — for tests + offline dev ────────────────────

class InMemoryThreadStore:
    """In-process store. No persistence across restart. Used by tests
    and any context where the Redis client isn't available."""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadRecord] = {}
        self._iterations: dict[str, IterationRecord] = {}
        self._thread_iters: dict[str, list[str]] = {}
        self._by_user: dict[str, dict[str, float]] = {}
        self._by_project: dict[str, dict[str, float]] = {}
        self._by_workspace: dict[str, dict[str, float]] = {}
        self._by_entity: dict[str, set[str]] = {}
        self._by_tag: dict[str, set[str]] = {}
        self._embeddings: dict[str, list[float]] = {}

    async def save_thread(self, thread: ThreadRecord) -> None:
        thread.updated_at = time.time()
        if not thread.created_at:
            thread.created_at = thread.updated_at
        # Round-trip through JSON so in-memory store has the same semantics
        # as a persistent one (no leaked references to mutable objects).
        snap = ThreadRecord.from_payload(thread.to_payload())
        self._threads[thread.id] = snap
        ub = thread.user_id or GUEST_BUCKET
        self._by_user.setdefault(ub, {})[thread.id] = thread.created_at
        if thread.project_id:
            self._by_project.setdefault(thread.project_id, {})[thread.id] = thread.created_at
        if thread.workspace_id:
            self._by_workspace.setdefault(thread.workspace_id, {})[thread.id] = thread.created_at
        for ent in thread.all_entities:
            self._by_entity.setdefault(ent.lower(), set()).add(thread.id)
        for tag in thread.all_tags:
            self._by_tag.setdefault(tag.lower(), set()).add(thread.id)

    async def get_thread(self, thread_id: str) -> ThreadRecord | None:
        return self._threads.get(thread_id)

    async def list_threads(
        self, *, user_id=None, project_id=None, workspace_id=None, limit=50, offset=0,
    ) -> list[ThreadRecord]:
        if workspace_id:
            bucket = self._by_workspace.get(workspace_id, {})
        elif project_id:
            bucket = self._by_project.get(project_id, {})
        else:
            bucket = self._by_user.get(user_id or GUEST_BUCKET, {})
        items = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
        ids = [tid for tid, _ in items[offset:offset + limit]]
        return [self._threads[i] for i in ids if i in self._threads]

    async def delete_thread(self, thread_id: str) -> bool:
        thread = self._threads.pop(thread_id, None)
        if not thread:
            return False
        ub = thread.user_id or GUEST_BUCKET
        self._by_user.get(ub, {}).pop(thread_id, None)
        if thread.project_id:
            self._by_project.get(thread.project_id, {}).pop(thread_id, None)
        if thread.workspace_id:
            self._by_workspace.get(thread.workspace_id, {}).pop(thread_id, None)
        for s in self._by_entity.values(): s.discard(thread_id)
        for s in self._by_tag.values():    s.discard(thread_id)
        for iid in thread.iteration_ids:
            self._iterations.pop(iid, None)
            self._embeddings.pop(iid, None)
        self._thread_iters.pop(thread_id, None)
        return True

    async def save_iteration(self, iteration: IterationRecord) -> None:
        if not iteration.created_at:
            iteration.created_at = time.time()
        snap = IterationRecord.from_payload(iteration.to_payload())
        self._iterations[iteration.id] = snap
        self._thread_iters.setdefault(iteration.thread_id, []).append(iteration.id)
        if iteration.embedding is not None:
            self._embeddings[iteration.id] = list(iteration.embedding)

    async def get_iteration(self, iter_id: str) -> IterationRecord | None:
        return self._iterations.get(iter_id)

    async def list_iterations_for_thread(self, thread_id: str) -> list[IterationRecord]:
        ids = self._thread_iters.get(thread_id, [])
        out = [self._iterations[i] for i in ids if i in self._iterations]
        out.sort(key=lambda it: it.sequence_num)
        return out

    async def attach_outcome(self, iter_id: str, outcome: OutcomeRecord) -> bool:
        existing = self._iterations.get(iter_id)
        if not existing:
            return False
        existing.outcome_followup = outcome
        return True

    async def find_similar_iterations(
        self, embedding, k=5, exclude_iter_id=None,
    ):
        q_norm = _l2(embedding)
        if q_norm == 0:
            return []
        scored: list[tuple[float, str]] = []
        for i_id, vec in self._embeddings.items():
            if exclude_iter_id and i_id == exclude_iter_id: continue
            if len(vec) != len(embedding): continue
            n = _l2(vec)
            if n == 0: continue
            scored.append((_dot(embedding, vec) / (q_norm * n), i_id))
        scored.sort(reverse=True, key=lambda t: t[0])
        return [(s, self._iterations[i]) for s, i in scored[:k] if i in self._iterations]

    async def find_threads_mentioning_entity(self, entity_name: str) -> list[str]:
        return list(self._by_entity.get(entity_name.lower(), set()))

    async def find_threads_by_tag(self, tag: str) -> list[str]:
        return list(self._by_tag.get(tag.lower(), set()))


# ─── Math helpers (kept inline so this file has no external deps) ─────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _l2(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


# ─── Factory: pick a backend from env ───────────────────────────────────

def build_thread_store_from_env() -> ThreadStore | None:
    """Build a ThreadStore based on environment variables.

    Backend selection:
      CONSTELLAX_DB_BACKEND=neo4j   → Neo4jThreadStore (requires NEO4J_URI +
                                       NEO4J_PASSWORD; falls through to Falkor
                                       on missing creds with a LOUD warning)
      CONSTELLAX_DB_BACKEND=falkor  → FalkorThreadStore (default — requires
                                       CONSTELLAX_REDIS_URL)
      (unset or any other value)    → Falkor path (back-compat)

    Returns None if no backend can be built — caller falls back to
    InMemoryThreadStore.

    NOTE: when this returns a Neo4jThreadStore, the schema is NOT yet
    initialized. Callers should `await init_store_schema(store)` before
    first use. The thread_persistence singleton handles that step.
    """
    import os

    backend = os.environ.get("CONSTELLAX_DB_BACKEND", "falkor").strip().lower()

    if backend == "neo4j":
        # Lazy import keeps the redis-only deploy path from pulling neo4j.
        from src.bridge.neo4j_backend import build_neo4j_thread_store_from_env
        store = build_neo4j_thread_store_from_env()
        if store is not None:
            log.info("ThreadStore: Neo4j-backed (CONSTELLAX_DB_BACKEND=neo4j)")
            return store
        # Explicit-intent signal lost. Loud warning so this is discoverable
        # in logs; fall through to Falkor as the safest behavior (don't
        # take the live system down because of a missing env var).
        log.warning(
            "CONSTELLAX_DB_BACKEND=neo4j but NEO4J_URI/NEO4J_PASSWORD are not "
            "set — falling back to Falkor. Set the Neo4j env vars to complete "
            "the migration."
        )

    # Default / Falkor path (also the fallback for misconfigured Neo4j)
    url = os.environ.get("CONSTELLAX_REDIS_URL")
    if not url:
        return None
    try:
        import redis.asyncio as redis_async
        client = redis_async.from_url(url)
        log.info("ThreadStore: Falkor-backed (CONSTELLAX_REDIS_URL set)")
        return FalkorThreadStore(client)
    except Exception as e:
        log.warning("ThreadStore: failed to build Falkor client (%s); caller should fall back", e)
        return None


async def init_store_schema(store: ThreadStore) -> None:
    """Run any one-time schema initialization needed by the store.

    No-op for FalkorThreadStore / InMemoryThreadStore (no schema to set up).
    Neo4jThreadStore creates constraints + vector index on first call. Safe
    to call on every startup — all DDL uses IF NOT EXISTS."""
    # Lazy import to avoid forcing neo4j import on Falkor-only deploys.
    try:
        from src.bridge.neo4j_backend import Neo4jThreadStore, init_schema
    except ImportError:
        return
    if isinstance(store, Neo4jThreadStore):
        import os
        dim = int(os.environ.get("NEO4J_EMBEDDING_DIM", "1536"))
        database = os.environ.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
        try:
            await init_schema(store._driver, database=database, embedding_dim=dim)
        except Exception as e:
            # Schema init failure shouldn't take down the app — log and
            # let read/write methods surface specific errors when called.
            log.warning("Neo4j schema init failed (non-fatal): %s", e)
