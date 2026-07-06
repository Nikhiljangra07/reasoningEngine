"""
Neo4jThreadStore — Neo4j-backed implementation of the ThreadStore Protocol.

WHY THIS EXISTS
===============
The original FalkorThreadStore (src/bridge/thread_store.py) uses Redis
primitives (SET/MGET/ZADD/LRANGE/SADD) because FalkorDB speaks the Redis
protocol. That works, but uses a graph DB as a KV store.

Neo4j Aura is the production target. This file uses Cypher and models the
data as a real graph:

    (User)-[:OWNS]->(Thread)
    (Thread)-[:IN_PROJECT]->(Project)
    (Thread)-[:IN_WORKSPACE]->(Workspace)
    (Thread)-[:HAS_ITERATION {seq}]->(Iteration)
    (Thread)-[:HAS_ENTITY]->(Entity)       — aggregate-level mention
    (Thread)-[:HAS_TAG]->(Tag)
    (Iteration)-[:MENTIONS]->(Entity)      — per-iteration granular
    (Iteration)-[:TAGGED]->(Tag)
    (Iteration)-[:IN_DOMAIN]->(Domain)

Full ThreadRecord / IterationRecord blobs are stored as JSON in
`payload_json` properties on the respective nodes — so round-trip is
lossless and the graph structure is a queryable side-index over the data.

VECTOR SIMILARITY
=================
Neo4j 5.13+ has native vector indexes. find_similar_iterations uses
`db.index.vector.queryNodes` instead of the brute-force cosine loop
that thread_store.py runs in Python. This is a major performance win
for the memory recall path.

CONTRACT
========
This class implements the ThreadStore Protocol defined in thread_store.py.
Drop-in compatible: the factory picks Neo4j vs Falkor based on
`CONSTELLAX_DB_BACKEND` env var (added in a separate commit). Application
code never imports Neo4j directly — only the Protocol.

SAFETY
======
- Schema init is idempotent (`CREATE CONSTRAINT IF NOT EXISTS`).
- All writes are transactional — `session.execute_write` retries on
  transient errors.
- Reads return None / empty list on missing data, never raise.
- The driver is held at the store level; close on shutdown via
  `await store.close()`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from src.bridge.types import (
    CodeRef,
    DecisionAnchor,
    DecisionLink,
    Iteration as BridgeIteration,
    Session,
    TurningPoint,
)
from src.core.decision_trace_types import (
    Decision,
    DecisionTraceBundle,
    Insight,
    Question,
    Reference,
    SystemResponse,
    UserMessage,
)
from src.core.thread_types import (
    IterationRecord,
    OutcomeRecord,
    ThreadRecord,
)

log = logging.getLogger("constellax.neo4j_backend")


GUEST_BUCKET = "guest"
_DEFAULT_EMBEDDING_DIM = 1536  # matches OpenAI ada-002 / Gemini at output_dim=1536
_VECTOR_INDEX_NAME = "iteration_embedding_idx"
# Constellation Interpreter (2026-06-01) — vector indexes for the
# multi-channel matcher's cushion-node and content-fingerprint sides.
# Both share the iteration index's 1536-dim cosine contract.
_CUSHION_NODE_VECTOR_INDEX_NAME = "cushion_node_embedding_idx"
_CONTENT_FINGERPRINT_VECTOR_INDEX_NAME = "content_fingerprint_embedding_idx"
_UNSCOPED_PROJECT = "__unscoped__"  # marker for project_id=None in ConversationStore

# Bridge entity types — must match _ENTITY_TYPES tuple in redis_backend.py.
# These are the four conversation entities that the ConversationBackend
# protocol handles uniformly via (entity_type, entity_id) addressing.
_BRIDGE_ENTITY_TYPES = ("sessions", "iterations", "turning_points", "decision_links")

# Label per entity_type. We deliberately use ":BridgeIteration" rather than
# ":Iteration" so the bridge Iteration model never collides with the very
# different ThreadStore IterationRecord (also labeled :Iteration). Naming
# different things the same label inside one Neo4j database is a recipe for
# silent cross-contamination — the labels are the namespace.
_BRIDGE_LABEL_BY_TYPE = {
    "sessions":        "Session",
    "iterations":      "BridgeIteration",
    "turning_points":  "TurningPoint",
    "decision_links":  "DecisionLink",
}


# ─── Schema initialization (idempotent) ───────────────────────────────

_SCHEMA_STATEMENTS = [
    # ThreadStore schema
    "CREATE CONSTRAINT thread_id_unique IF NOT EXISTS FOR (t:Thread) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT iteration_id_unique IF NOT EXISTS FOR (i:Iteration) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
    "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT workspace_id_unique IF NOT EXISTS FOR (w:Workspace) REQUIRE w.id IS UNIQUE",
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS FOR (g:Tag) REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT domain_name_unique IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
    "CREATE INDEX thread_created_at IF NOT EXISTS FOR (t:Thread) ON (t.created_at)",
    "CREATE INDEX iteration_thread IF NOT EXISTS FOR (i:Iteration) ON (i.thread_id)",
    # Phase 1: Provenance + structured-memory sweep indexes.
    # iteration_workspace + iteration_surface support fast platform/surface
    # filtering in the cross-thread retriever (Phase 4). iteration_structured
    # supports the background sweeper's WHERE structured_at IS NULL scan.
    # The composite (structured_at, completed_at) accelerates the actual
    # sweeper query — Neo4j picks it for predicates on both columns.
    "CREATE INDEX iteration_workspace IF NOT EXISTS FOR (i:Iteration) ON (i.workspace_id)",
    "CREATE INDEX iteration_surface IF NOT EXISTS FOR (i:Iteration) ON (i.surface_id)",
    "CREATE INDEX iteration_structured IF NOT EXISTS FOR (i:Iteration) ON (i.structured_at)",
    "CREATE INDEX iteration_completed IF NOT EXISTS FOR (i:Iteration) ON (i.completed_at)",
    # Phase 2a: Decision Trace node uniqueness. Dual labels (:DecisionTrace:Type)
    # mean we also need to be careful when matching — see Neo4jDecisionTraceWriter
    # for the canonical query shape. Constraints are on the type label only;
    # the :DecisionTrace label is the source-namespace marker.
    "CREATE CONSTRAINT dt_user_message_id IF NOT EXISTS FOR (n:UserMessage) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dt_system_response_id IF NOT EXISTS FOR (n:SystemResponse) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dt_decision_id IF NOT EXISTS FOR (n:Decision) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dt_question_id IF NOT EXISTS FOR (n:Question) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dt_reference_id IF NOT EXISTS FOR (n:Reference) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT dt_insight_id IF NOT EXISTS FOR (n:Insight) REQUIRE n.id IS UNIQUE",
    # Filter indexes for cross-thread / cross-workspace retrieval. These get
    # added on every Decision Trace type since the retriever filters by these
    # fields. The thread_id index lets us pull "all events in this thread"
    # without a graph traversal through Iteration.
    "CREATE INDEX dt_user_message_thread IF NOT EXISTS FOR (n:UserMessage) ON (n.thread_id)",
    "CREATE INDEX dt_user_message_user IF NOT EXISTS FOR (n:UserMessage) ON (n.user_id)",
    "CREATE INDEX dt_system_response_thread IF NOT EXISTS FOR (n:SystemResponse) ON (n.thread_id)",
    "CREATE INDEX dt_system_response_user IF NOT EXISTS FOR (n:SystemResponse) ON (n.user_id)",
    "CREATE INDEX dt_decision_thread IF NOT EXISTS FOR (n:Decision) ON (n.thread_id)",
    "CREATE INDEX dt_decision_user IF NOT EXISTS FOR (n:Decision) ON (n.user_id)",
    "CREATE INDEX dt_decision_workspace IF NOT EXISTS FOR (n:Decision) ON (n.workspace_id)",
    "CREATE INDEX dt_decision_surface IF NOT EXISTS FOR (n:Decision) ON (n.surface_id)",
    "CREATE INDEX dt_question_thread IF NOT EXISTS FOR (n:Question) ON (n.thread_id)",
    "CREATE INDEX dt_question_resolved IF NOT EXISTS FOR (n:Question) ON (n.resolved)",
    "CREATE INDEX dt_reference_thread IF NOT EXISTS FOR (n:Reference) ON (n.thread_id)",
    "CREATE INDEX dt_reference_kind IF NOT EXISTS FOR (n:Reference) ON (n.kind)",
    "CREATE INDEX dt_insight_thread IF NOT EXISTS FOR (n:Insight) ON (n.thread_id)",
    # Bridge schema (DecisionAnchor + Conversation entities)
    "CREATE CONSTRAINT decision_anchor_id_unique IF NOT EXISTS FOR (d:DecisionAnchor) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT bridge_session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT bridge_iteration_id_unique IF NOT EXISTS FOR (i:BridgeIteration) REQUIRE i.id IS UNIQUE",
    "CREATE CONSTRAINT turning_point_id_unique IF NOT EXISTS FOR (t:TurningPoint) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT decision_link_id_unique IF NOT EXISTS FOR (l:DecisionLink) REQUIRE l.id IS UNIQUE",
    "CREATE INDEX decision_anchor_project IF NOT EXISTS FOR (d:DecisionAnchor) ON (d.project_id)",
    "CREATE INDEX decision_anchor_status IF NOT EXISTS FOR (d:DecisionAnchor) ON (d.status)",
    "CREATE INDEX bridge_session_project IF NOT EXISTS FOR (s:Session) ON (s.project_id)",
    "CREATE INDEX bridge_iteration_session IF NOT EXISTS FOR (i:BridgeIteration) ON (i.session_id)",
    # Constellation Interpreter (2026-06-01). CushionNode persists each
    # dual-artifact cushion entry (graph_meaning + retrieval_face +
    # embedding). ContentFingerprint persists Haiku-extracted structural
    # phrases per fetched content piece, keyed by content_hash for cache
    # lookup across sessions. Vector indexes for both are created
    # separately below (Neo4j 5.13+ vector DDL).
    "CREATE CONSTRAINT cushion_node_id_unique IF NOT EXISTS FOR (n:CushionNode) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT content_fingerprint_hash_unique IF NOT EXISTS FOR (f:ContentFingerprint) REQUIRE f.content_hash IS UNIQUE",
]


def _vector_index_statement(dim: int) -> str:
    """Vector index DDL. Neo4j 5.13+ native vector index for cosine
    similarity over Iteration.embedding."""
    return (
        f"CREATE VECTOR INDEX {_VECTOR_INDEX_NAME} IF NOT EXISTS "
        f"FOR (i:Iteration) ON i.embedding "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {int(dim)}, "
        f"`vector.similarity_function`: 'cosine'"
        f"}}}}"
    )


def _cushion_node_vector_index_statement(dim: int) -> str:
    """Cosine-similarity index over CushionNode.embedding.

    Constellation Interpreter (2026-06-01). The multi-channel matcher's
    vector channel queries this index with a fingerprint embedding to
    find cushion nodes whose retrieval_face embedding_text lives near
    the fingerprint in structural space.
    """
    return (
        f"CREATE VECTOR INDEX {_CUSHION_NODE_VECTOR_INDEX_NAME} IF NOT EXISTS "
        f"FOR (n:CushionNode) ON n.embedding "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {int(dim)}, "
        f"`vector.similarity_function`: 'cosine'"
        f"}}}}"
    )


def _content_fingerprint_vector_index_statement(dim: int) -> str:
    """Cosine-similarity index over ContentFingerprint.embedding.

    Constellation Interpreter (2026-06-01). Each fetched content piece
    gets a Haiku-extracted structural fingerprint (2-5 phrases) and an
    embedding of those phrases concatenated. This index lets us find
    fingerprints whose structural language matches a given cushion-node
    query without paying for an LLM call per candidate.
    """
    return (
        f"CREATE VECTOR INDEX {_CONTENT_FINGERPRINT_VECTOR_INDEX_NAME} IF NOT EXISTS "
        f"FOR (f:ContentFingerprint) ON f.embedding "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {int(dim)}, "
        f"`vector.similarity_function`: 'cosine'"
        f"}}}}"
    )


async def init_schema(driver: Any, *, database: str = "neo4j", embedding_dim: int = _DEFAULT_EMBEDDING_DIM) -> None:
    """Run all schema DDL. Idempotent — safe to call on every startup.

    Embedding dim must match the model you're using:
      - Gemini text-embedding-005 / OpenAI ada-002: 1536
      - Gemini gemini-embedding-001 (default): 3072
    Set NEO4J_EMBEDDING_DIM to override.
    """
    async with driver.session(database=database) as session:
        for stmt in _SCHEMA_STATEMENTS:
            try:
                await session.run(stmt)
            except Exception as e:
                log.warning("schema init: %s failed: %s", stmt.split()[0:3], e)
        try:
            await session.run(_vector_index_statement(embedding_dim))
        except Exception as e:
            # Vector index requires Neo4j 5.13+; falls back gracefully if older.
            log.warning("vector index creation failed (need Neo4j 5.13+): %s", e)
        # Constellation Interpreter (2026-06-01) — same 5.13+ requirement,
        # same 1536-dim cosine contract. Failure is non-fatal: the iteration
        # path keeps working, only the new matcher's vector channel is offline.
        try:
            await session.run(_cushion_node_vector_index_statement(embedding_dim))
        except Exception as e:
            log.warning("cushion_node vector index creation failed: %s", e)
        try:
            await session.run(_content_fingerprint_vector_index_statement(embedding_dim))
        except Exception as e:
            log.warning("content_fingerprint vector index creation failed: %s", e)
    log.info("Neo4j schema initialized (database=%s, embedding_dim=%d)", database, embedding_dim)


# ─── Neo4jThreadStore — main implementation ───────────────────────────

class Neo4jThreadStore:
    """Neo4j-backed ThreadStore. See module docstring for graph schema."""

    def __init__(self, driver: Any, *, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    async def close(self) -> None:
        """Close the driver. Call at shutdown."""
        try:
            await self._driver.close()
        except Exception as e:
            log.warning("driver close failed: %s", e)

    # ─── Threads ────────────────────────────────────────

    async def save_thread(self, thread: ThreadRecord) -> None:
        thread.updated_at = time.time()
        if not thread.created_at:
            thread.created_at = thread.updated_at
        payload_json = json.dumps(thread.to_payload())
        user_bucket = thread.user_id or GUEST_BUCKET

        async def _tx(tx):
            # 1) Upsert thread node with indexed scalar properties + JSON blob
            await tx.run(
                """
                MERGE (t:Thread {id: $id})
                SET t.title = $title,
                    t.status = $status,
                    t.created_at = $created_at,
                    t.updated_at = $updated_at,
                    t.iteration_count = $iteration_count,
                    t.last_route = $last_route,
                    t.last_confidence = $last_confidence,
                    t.aggregate_time_ms = $aggregate_time_ms,
                    t.aggregate_cost_usd = $aggregate_cost_usd,
                    t.perspectives_run = $perspectives_run,
                    t.user_id = $user_id,
                    t.project_id = $project_id,
                    t.workspace_id = $workspace_id,
                    t.payload_json = $payload_json
                """,
                id=thread.id,
                title=thread.title or "",
                status=thread.status or "active",
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                iteration_count=thread.iteration_count or 0,
                last_route=thread.last_route or "",
                last_confidence=thread.last_confidence or "",
                aggregate_time_ms=int(thread.aggregate_time_ms or 0),
                aggregate_cost_usd=float(thread.aggregate_cost_usd or 0.0),
                perspectives_run=int(thread.perspectives_run or 0),
                user_id=user_bucket,
                project_id=thread.project_id or "",
                workspace_id=thread.workspace_id or "",
                payload_json=payload_json,
            )
            # 2) Owner relationship
            await tx.run(
                """
                MATCH (t:Thread {id: $tid})
                MERGE (u:User {id: $uid})
                MERGE (u)-[:OWNS]->(t)
                """,
                tid=thread.id, uid=user_bucket,
            )
            # 3) Project / workspace links (skip empty)
            if thread.project_id:
                await tx.run(
                    """
                    MATCH (t:Thread {id: $tid})
                    MERGE (p:Project {id: $pid})
                    MERGE (t)-[:IN_PROJECT]->(p)
                    """,
                    tid=thread.id, pid=thread.project_id,
                )
            if thread.workspace_id:
                await tx.run(
                    """
                    MATCH (t:Thread {id: $tid})
                    MERGE (w:Workspace {id: $wid})
                    MERGE (t)-[:IN_WORKSPACE]->(w)
                    """,
                    tid=thread.id, wid=thread.workspace_id,
                )
            # 4) Aggregate entity / tag relationships
            entities = [e.lower() for e in (thread.all_entities or []) if e]
            tags = [t.lower() for t in (thread.all_tags or []) if t]
            if entities:
                await tx.run(
                    """
                    MATCH (t:Thread {id: $tid})
                    UNWIND $names AS n
                        MERGE (e:Entity {name: n})
                        MERGE (t)-[:HAS_ENTITY]->(e)
                    """,
                    tid=thread.id, names=entities,
                )
            if tags:
                await tx.run(
                    """
                    MATCH (t:Thread {id: $tid})
                    UNWIND $names AS n
                        MERGE (g:Tag {name: n})
                        MERGE (t)-[:HAS_TAG]->(g)
                    """,
                    tid=thread.id, names=tags,
                )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_tx)

    async def get_thread(self, thread_id: str) -> ThreadRecord | None:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (t:Thread {id: $id}) RETURN t.payload_json AS payload",
                id=thread_id,
            )
            record = await result.single()
        if not record or not record["payload"]:
            return None
        try:
            return ThreadRecord.from_payload(json.loads(record["payload"]))
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("thread %s malformed: %s", thread_id, e)
            return None

    async def list_threads(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThreadRecord]:
        # Pick the most-specific filter available, mirror FalkorThreadStore priority.
        if workspace_id:
            cypher = (
                "MATCH (w:Workspace {id: $key})<-[:IN_WORKSPACE]-(t:Thread) "
                "RETURN t.payload_json AS payload "
                "ORDER BY t.created_at DESC SKIP $offset LIMIT $limit"
            )
            key = workspace_id
        elif project_id:
            cypher = (
                "MATCH (p:Project {id: $key})<-[:IN_PROJECT]-(t:Thread) "
                "RETURN t.payload_json AS payload "
                "ORDER BY t.created_at DESC SKIP $offset LIMIT $limit"
            )
            key = project_id
        else:
            cypher = (
                "MATCH (u:User {id: $key})-[:OWNS]->(t:Thread) "
                "RETURN t.payload_json AS payload "
                "ORDER BY t.created_at DESC SKIP $offset LIMIT $limit"
            )
            key = user_id or GUEST_BUCKET

        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, key=key, offset=offset, limit=limit)
            records = [r async for r in result]

        out: list[ThreadRecord] = []
        for r in records:
            blob = r.get("payload")
            if not blob:
                continue
            try:
                out.append(ThreadRecord.from_payload(json.loads(blob)))
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("malformed thread record: %s", e)
        return out

    async def delete_thread(self, thread_id: str) -> bool:
        # First check existence (mirror FalkorThreadStore contract: return False
        # if missing, True if deleted).
        existing = await self.get_thread(thread_id)
        if not existing:
            return False

        async def _tx(tx):
            # DETACH DELETE cascades through all relationships from the thread,
            # then explicitly delete owned iterations (they outlive the thread
            # node if not explicitly handled).
            await tx.run(
                """
                MATCH (t:Thread {id: $id})
                OPTIONAL MATCH (t)-[:HAS_ITERATION]->(i:Iteration)
                DETACH DELETE i, t
                """,
                id=thread_id,
            )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_tx)
        return True

    # ─── Iterations ────────────────────────────────────────

    async def save_iteration(self, iteration: IterationRecord) -> None:
        if not iteration.created_at:
            iteration.created_at = time.time()
        payload_json = json.dumps(iteration.to_payload())

        entities = [e.name.lower() for e in (iteration.entities or []) if getattr(e, "name", None)]
        tags = [t.lower() for t in (iteration.tags or []) if t]
        domains = [d.lower() for d in (iteration.domains or []) if d]

        async def _tx(tx):
            # 1) Upsert iteration node with embedding (so vector index picks it up).
            # Phase 1: workspace_id + surface_id + structured_at are denormalized
            # as indexed scalar properties so the sweeper and retrievers can
            # filter without joining through Thread. Order matters less than
            # the property being set EVERY save — partial updates would leave
            # the indexes pointing at stale state.
            await tx.run(
                """
                MERGE (i:Iteration {id: $id})
                SET i.thread_id = $thread_id,
                    i.sequence_num = $sequence_num,
                    i.status = $status,
                    i.created_at = $created_at,
                    i.completed_at = $completed_at,
                    i.embedding = $embedding,
                    i.embedding_model = $embedding_model,
                    i.workspace_id = coalesce(nullif($workspace_id, ''), i.workspace_id),
                    i.surface_id = coalesce(nullif($surface_id, ''), i.surface_id),
                    i.structured_at = coalesce($structured_at, i.structured_at),
                    i.payload_json = $payload_json
                """,
                id=iteration.id,
                thread_id=iteration.thread_id or "",
                sequence_num=int(iteration.sequence_num or 0),
                status=iteration.status or "done",
                created_at=iteration.created_at,
                completed_at=iteration.completed_at,
                embedding=list(iteration.embedding) if iteration.embedding else None,
                embedding_model=iteration.embedding_model or "",
                # Phase 1 fields. workspace_id + surface_id default to empty string
                # rather than None at the Neo4j layer so the property type stays
                # consistent for the index. structured_at stays None until the
                # sweeper stamps it, because indexes on numeric properties handle
                # null correctly and null is the sentinel the sweeper queries on.
                workspace_id=iteration.workspace_id or "",
                surface_id=iteration.surface_id or "",
                structured_at=iteration.structured_at,
                payload_json=payload_json,
            )
            # 2) Thread → Iteration relationship (with sequence number on the edge)
            if iteration.thread_id:
                await tx.run(
                    """
                    MATCH (i:Iteration {id: $iid})
                    MERGE (t:Thread {id: $tid})
                    MERGE (t)-[r:HAS_ITERATION]->(i)
                    SET r.seq = $seq
                    """,
                    iid=iteration.id, tid=iteration.thread_id,
                    seq=int(iteration.sequence_num or 0),
                )
            # 3) Per-iteration entity / tag / domain links
            if entities:
                await tx.run(
                    """
                    MATCH (i:Iteration {id: $iid})
                    UNWIND $names AS n
                        MERGE (e:Entity {name: n})
                        MERGE (i)-[:MENTIONS]->(e)
                    """,
                    iid=iteration.id, names=entities,
                )
            if tags:
                await tx.run(
                    """
                    MATCH (i:Iteration {id: $iid})
                    UNWIND $names AS n
                        MERGE (g:Tag {name: n})
                        MERGE (i)-[:TAGGED]->(g)
                    """,
                    iid=iteration.id, names=tags,
                )
            if domains:
                await tx.run(
                    """
                    MATCH (i:Iteration {id: $iid})
                    UNWIND $names AS n
                        MERGE (d:Domain {name: n})
                        MERGE (i)-[:IN_DOMAIN]->(d)
                    """,
                    iid=iteration.id, names=domains,
                )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_tx)

    async def get_iteration(self, iter_id: str) -> IterationRecord | None:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (i:Iteration {id: $id}) RETURN i.payload_json AS payload",
                id=iter_id,
            )
            record = await result.single()
        if not record or not record["payload"]:
            return None
        try:
            return IterationRecord.from_payload(json.loads(record["payload"]))
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("iteration %s malformed: %s", iter_id, e)
            return None

    async def list_iterations_for_thread(self, thread_id: str) -> list[IterationRecord]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (t:Thread {id: $tid})-[r:HAS_ITERATION]->(i:Iteration)
                RETURN i.payload_json AS payload
                ORDER BY r.seq ASC
                """,
                tid=thread_id,
            )
            records = [r async for r in result]
        out: list[IterationRecord] = []
        for r in records:
            blob = r.get("payload")
            if not blob:
                continue
            try:
                out.append(IterationRecord.from_payload(json.loads(blob)))
            except (json.JSONDecodeError, ValueError) as e:
                log.warning("malformed iteration in thread %s: %s", thread_id, e)
        return out

    # ─── Outcome attachment ────────────────────────────────────────

    async def attach_outcome(self, iter_id: str, outcome: OutcomeRecord) -> bool:
        existing = await self.get_iteration(iter_id)
        if not existing:
            return False
        existing.outcome_followup = outcome
        await self.save_iteration(existing)
        return True

    # ─── Similarity search (native vector index) ────────────────────

    async def find_similar_iterations(
        self,
        embedding: list[float],
        k: int = 5,
        exclude_iter_id: str | None = None,
    ) -> list[tuple[float, IterationRecord]]:
        """Native vector-index query. Replaces the brute-force cosine
        loop in FalkorThreadStore. Requires Neo4j 5.13+ with the vector
        index created at schema init time."""
        if not embedding:
            return []

        # Request k+1 if we're filtering one out, so we still return k after exclusion.
        query_k = k + 1 if exclude_iter_id else k

        async with self._driver.session(database=self._database) as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
                    YIELD node, score
                    RETURN node.id AS id, node.payload_json AS payload, score
                    """,
                    index_name=_VECTOR_INDEX_NAME, k=query_k, embedding=list(embedding),
                )
                records = [r async for r in result]
            except Exception as e:
                # Index may not exist yet (no embeddings stored), or older Neo4j.
                log.warning("vector query failed: %s — returning empty", e)
                return []

        out: list[tuple[float, IterationRecord]] = []
        for r in records:
            iid = r.get("id")
            if exclude_iter_id and iid == exclude_iter_id:
                continue
            blob = r.get("payload")
            score = float(r.get("score") or 0.0)
            if not blob:
                continue
            try:
                rec = IterationRecord.from_payload(json.loads(blob))
                out.append((score, rec))
            except (json.JSONDecodeError, ValueError):
                continue
            if len(out) >= k:
                break
        return out

    # ─── Decision Trace sweeper hooks ────────────────────────────────
    # These two methods drive the background sweeper (Phase 3). They're
    # Neo4j-specific (not part of the ThreadStore Protocol) — the sweeper
    # is bound to Neo4j by design, so portability isn't a goal here.

    async def find_unstructured_iteration_ids(
        self,
        *,
        idle_sec: int = 1800,
        limit: int = 50,
    ) -> list[str]:
        """Return iteration ids that need a Decision Trace pass.

        An iteration qualifies when:
          - structured_at IS NULL (never processed by the sweeper)
          - completed_at IS NOT NULL (the trace actually finished)
          - now - completed_at > idle_sec (the user has gone idle long
            enough that we're confident they're done with this turn)

        `timestamp()` in Cypher is milliseconds since epoch; `completed_at`
        is stored as Unix seconds (float). We convert in one direction in
        the WHERE clause for clarity.

        Returns ids only (not full IterationRecords) — the sweeper loads
        each iteration via get_iteration() when it's ready to process,
        keeping the working set small."""
        cutoff_seconds = float(idle_sec)
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (i:Iteration)
                WHERE i.structured_at IS NULL
                  AND i.completed_at IS NOT NULL
                  AND (timestamp() / 1000.0) - i.completed_at > $cutoff
                RETURN i.id AS id
                ORDER BY i.completed_at ASC
                LIMIT $lim
                """,
                cutoff=cutoff_seconds, lim=int(limit),
            )
            return [rec["id"] async for rec in result]

    async def stamp_structured(self, iter_id: str, ts: float) -> bool:
        """Mark an iteration as processed by the sweeper.

        We stamp structured_at AFTER the Decision Trace bundle has been
        written, so a crash between write and stamp leaves the iteration
        unstructured and the next sweep picks it up. The writer's MERGE
        makes that retry idempotent.

        Returns True if the iteration was found and stamped, False if no
        node matched (e.g., the iteration got deleted between query and
        stamp — possible but rare)."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (i:Iteration {id: $id})
                SET i.structured_at = $ts
                RETURN i.id AS id
                """,
                id=iter_id, ts=float(ts),
            )
            rec = await result.single()
            return rec is not None

    # ─── Entity / tag lookups ────────────────────────────────────────

    async def find_threads_mentioning_entity(self, entity_name: str) -> list[str]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (e:Entity {name: $name})<-[:HAS_ENTITY]-(t:Thread)
                RETURN t.id AS id
                """,
                name=entity_name.lower(),
            )
            return [r["id"] async for r in result]

    async def find_threads_by_tag(self, tag: str) -> list[str]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (g:Tag {name: $name})<-[:HAS_TAG]-(t:Thread)
                RETURN t.id AS id
                """,
                name=tag.lower(),
            )
            return [r["id"] async for r in result]


# ─── Factory: build from env ─────────────────────────────────────────

def build_neo4j_thread_store_from_env() -> Neo4jThreadStore | None:
    """Build a Neo4jThreadStore from NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD.

    Returns None when:
      - Any of the three env vars are unset (caller falls back)
      - The `neo4j` package isn't installed
      - The driver fails to construct (caller falls back)

    Note: this returns the store WITHOUT running schema init — call
    `await init_schema(store._driver)` at startup before first use.
    The thread_persistence singleton handles this in _ensure_initialized().
    """
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USERNAME", "").strip() or "neo4j"
    password = os.environ.get("NEO4J_PASSWORD", "").strip()
    database = os.environ.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j"

    if not uri or not password:
        return None

    try:
        from neo4j import AsyncGraphDatabase  # type: ignore[import-not-found]
    except ImportError:
        log.warning("neo4j driver not installed — falling back")
        return None

    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    except Exception as e:
        log.warning("failed to construct Neo4j driver: %s", e)
        return None

    log.info("Neo4jThreadStore: built driver for %s (database=%s)", uri, database)
    return Neo4jThreadStore(driver, database=database)


# ═══════════════════════════════════════════════════════════════════════
# Bridge backends — DecisionAnchor + ConversationStore entities
# ═══════════════════════════════════════════════════════════════════════
#
# These mirror the Protocols defined in src/bridge/redis_backend.py:
#
#     AnchorBackend       — DecisionAnchor CRUD
#     ConversationBackend — Session / Iteration / TurningPoint / DecisionLink
#
# Same contracts, Neo4j-backed instead of Redis-backed. Drop-in compatible.
# Graph schema: every entity is a node carrying its full record as a
# `payload_json` property; structural relationships (`IN_PROJECT`,
# `IN_SESSION`, `IN_ITERATION`, `FROM_DECISION`, `TO_DECISION`) light up
# graph-native query power for future work without touching the existing
# CRUD protocol.
#
# THE LABEL NAMESPACE GOTCHA
# ==========================
# The bridge's Iteration type is *not* the same as ThreadStore's
# IterationRecord. Different fields, different lifecycle, different
# meaning. We label the bridge variant `:BridgeIteration` so it cannot
# silently mingle with `:Iteration` nodes from ThreadStore. If you ever
# rename or extend either, keep the labels distinct.

# ─── JSON helpers (shared with redis_backend.py — pure data) ──────────

def _decision_to_json(decision: DecisionAnchor) -> str:
    from dataclasses import asdict
    return json.dumps(asdict(decision))


def _decision_from_json(raw: str) -> DecisionAnchor | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    code_refs_raw = data.pop("code_refs", []) or []
    code_refs = [CodeRef(**r) for r in code_refs_raw if isinstance(r, dict)]
    try:
        return DecisionAnchor(code_refs=code_refs, **data)
    except (TypeError, ValueError):
        return None


_CONV_REBUILDERS = {
    "sessions":       Session,
    "iterations":     BridgeIteration,
    "turning_points": TurningPoint,
    "decision_links": DecisionLink,
}


def _conv_to_json(entity: Any) -> str:
    from dataclasses import asdict
    return json.dumps(asdict(entity))


def _conv_from_json(entity_type: str, raw: str) -> Any | None:
    rebuilder = _CONV_REBUILDERS.get(entity_type)
    if rebuilder is None:
        return None
    try:
        data = json.loads(raw)
        return rebuilder(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ─── Neo4jAnchorBackend — DecisionAnchor CRUD ─────────────────────────

class Neo4jAnchorBackend:
    """Neo4j-backed implementation of the AnchorBackend Protocol.

    Schema: (:DecisionAnchor {id, project_id, status, payload_json})
            -[:IN_PROJECT]-> (:Project {id})

    Reads pull the full DecisionAnchor back from `payload_json` so the
    JSON shape stays the authoritative wire format. The indexed scalar
    properties (status, project_id) let us answer status-filter and
    project-scope queries without a JSON parse on every node."""

    def __init__(self, driver: Any, *, database: str = "neo4j", owns_driver: bool = False) -> None:
        self._driver = driver
        self._database = database
        self._owns_driver = owns_driver  # only close if we built the driver ourselves

    async def close(self) -> None:
        if self._owns_driver:
            try:
                await self._driver.close()
            except Exception as e:
                log.warning("anchor backend driver close failed: %s", e)

    async def get(self, project_id: str, decision_id: str) -> DecisionAnchor | None:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (d:DecisionAnchor {id: $id, project_id: $pid})
                RETURN d.payload_json AS payload
                """,
                id=decision_id, pid=project_id,
            )
            record = await result.single()
        if not record or not record["payload"]:
            return None
        return _decision_from_json(record["payload"])

    async def put(self, project_id: str, decision: DecisionAnchor) -> None:
        if not decision.id:
            raise ValueError("DecisionAnchor.id must be non-empty before storing")
        payload = _decision_to_json(decision)

        async def _tx(tx):
            await tx.run(
                """
                MERGE (d:DecisionAnchor {id: $id})
                SET d.project_id = $pid,
                    d.status = $status,
                    d.payload_json = $payload
                """,
                id=decision.id, pid=project_id,
                status=decision.status or "OPEN", payload=payload,
            )
            await tx.run(
                """
                MATCH (d:DecisionAnchor {id: $id})
                MERGE (p:Project {id: $pid})
                MERGE (d)-[:IN_PROJECT]->(p)
                """,
                id=decision.id, pid=project_id,
            )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_tx)

    async def delete(self, project_id: str, decision_id: str) -> bool:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (d:DecisionAnchor {id: $id, project_id: $pid})
                WITH d, count(d) AS n
                DETACH DELETE d
                RETURN n
                """,
                id=decision_id, pid=project_id,
            )
            record = await result.single()
        return bool(record and (record["n"] or 0) > 0)

    async def list_for_project(self, project_id: str) -> list[DecisionAnchor]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (d:DecisionAnchor {project_id: $pid})
                RETURN d.payload_json AS payload
                """,
                pid=project_id,
            )
            records = [r async for r in result]
        out: list[DecisionAnchor] = []
        for r in records:
            blob = r.get("payload")
            if not blob:
                continue
            anchor = _decision_from_json(blob)
            if anchor is not None:
                out.append(anchor)
        return out

    async def update_status(self, project_id: str, decision_id: str, status: str) -> bool:
        # Status lives both as an indexed property (for fast filtering) AND
        # inside the JSON payload (the source of truth for the record). Both
        # must stay in sync — we fetch, mutate, and rewrite the JSON in one tx.
        current = await self.get(project_id, decision_id)
        if current is None:
            return False
        from dataclasses import replace
        updated = replace(current, status=status)
        await self.put(project_id, updated)
        return True

    async def known_projects(self) -> list[str]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (d:DecisionAnchor)
                WHERE d.project_id IS NOT NULL
                RETURN DISTINCT d.project_id AS project_id
                """
            )
            return [r["project_id"] async for r in result]


# ─── Neo4jConversationBackend — 4 entity types ────────────────────────

class Neo4jConversationBackend:
    """Neo4j-backed implementation of the ConversationBackend Protocol.

    Handles four entity types uniformly via (entity_type, entity_id):
        sessions       → (:Session)
        iterations     → (:BridgeIteration)   ← deliberately NOT :Iteration
        turning_points → (:TurningPoint)
        decision_links → (:DecisionLink)

    Structural relationships:
        (:Session)-[:IN_PROJECT]->(:Project)
        (:BridgeIteration)-[:IN_SESSION {seq}]->(:Session)
        (:TurningPoint)-[:IN_SESSION]->(:Session)
        (:TurningPoint)-[:IN_ITERATION]->(:BridgeIteration)
        (:DecisionLink)-[:IN_PROJECT]->(:Project)
        (:DecisionLink)-[:FROM_DECISION]->(:DecisionAnchor)
        (:DecisionLink)-[:TO_DECISION]->(:DecisionAnchor)

    Every node carries `project_id` as a denormalized property so the
    project-scoped CRUD protocol can answer in one MATCH without graph
    traversal. This trades a bit of redundancy for protocol-level
    performance parity with the Redis backend."""

    def __init__(self, driver: Any, *, database: str = "neo4j", owns_driver: bool = False) -> None:
        self._driver = driver
        self._database = database
        self._owns_driver = owns_driver

    async def close(self) -> None:
        if self._owns_driver:
            try:
                await self._driver.close()
            except Exception as e:
                log.warning("conversation backend driver close failed: %s", e)

    def _label(self, entity_type: str) -> str:
        label = _BRIDGE_LABEL_BY_TYPE.get(entity_type)
        if label is None:
            raise ValueError(
                f"unknown entity_type {entity_type!r}; expected one of {_BRIDGE_ENTITY_TYPES}"
            )
        return label

    async def get(self, project_id: str, entity_type: str, entity_id: str) -> Any | None:
        label = self._label(entity_type)
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"MATCH (n:{label} {{id: $id, project_id: $pid}}) RETURN n.payload_json AS payload",
                id=entity_id, pid=project_id,
            )
            record = await result.single()
        if not record or not record["payload"]:
            return None
        return _conv_from_json(entity_type, record["payload"])

    async def put(self, project_id: str, entity_type: str, entity: Any) -> None:
        label = self._label(entity_type)
        eid = getattr(entity, "id", None)
        if not eid:
            raise ValueError(f"{entity_type} entity must have non-empty id")
        payload = _conv_to_json(entity)

        # Pull entity-type-specific fields that get promoted to indexed
        # properties for fast scoping. Everything else lives inside payload_json.
        session_id    = getattr(entity, "session_id", "") or ""
        iteration_id  = getattr(entity, "iteration_id", "") or ""
        sequence_num  = int(getattr(entity, "sequence_num", 0) or 0)
        from_decision = getattr(entity, "from_decision_id", "") or ""
        to_decision   = getattr(entity, "to_decision_id", "") or ""
        link_type     = getattr(entity, "link_type", "") or ""

        async def _tx(tx):
            # 1) Upsert the entity node with indexed scalar properties + JSON blob.
            await tx.run(
                f"""
                MERGE (n:{label} {{id: $id}})
                SET n.project_id = $pid,
                    n.session_id = $session_id,
                    n.iteration_id = $iteration_id,
                    n.sequence_num = $sequence_num,
                    n.from_decision_id = $from_decision,
                    n.to_decision_id = $to_decision,
                    n.link_type = $link_type,
                    n.payload_json = $payload
                """,
                id=eid, pid=project_id,
                session_id=session_id, iteration_id=iteration_id,
                sequence_num=sequence_num,
                from_decision=from_decision, to_decision=to_decision,
                link_type=link_type, payload=payload,
            )
            # 2) Type-specific structural relationships.
            if entity_type == "sessions":
                await tx.run(
                    f"""
                    MATCH (n:{label} {{id: $id}})
                    MERGE (p:Project {{id: $pid}})
                    MERGE (n)-[:IN_PROJECT]->(p)
                    """,
                    id=eid, pid=project_id,
                )
            elif entity_type == "iterations":
                if session_id:
                    await tx.run(
                        f"""
                        MATCH (n:{label} {{id: $id}})
                        MERGE (s:Session {{id: $sid}})
                        MERGE (n)-[r:IN_SESSION]->(s)
                        SET r.seq = $seq
                        """,
                        id=eid, sid=session_id, seq=sequence_num,
                    )
            elif entity_type == "turning_points":
                if session_id:
                    await tx.run(
                        f"""
                        MATCH (n:{label} {{id: $id}})
                        MERGE (s:Session {{id: $sid}})
                        MERGE (n)-[:IN_SESSION]->(s)
                        """,
                        id=eid, sid=session_id,
                    )
                if iteration_id:
                    await tx.run(
                        f"""
                        MATCH (n:{label} {{id: $id}})
                        MERGE (i:BridgeIteration {{id: $iid}})
                        MERGE (n)-[:IN_ITERATION]->(i)
                        """,
                        id=eid, iid=iteration_id,
                    )
            elif entity_type == "decision_links":
                await tx.run(
                    f"""
                    MATCH (n:{label} {{id: $id}})
                    MERGE (p:Project {{id: $pid}})
                    MERGE (n)-[:IN_PROJECT]->(p)
                    """,
                    id=eid, pid=project_id,
                )
                if from_decision:
                    await tx.run(
                        f"""
                        MATCH (n:{label} {{id: $id}})
                        MERGE (d:DecisionAnchor {{id: $did}})
                        MERGE (n)-[:FROM_DECISION]->(d)
                        """,
                        id=eid, did=from_decision,
                    )
                if to_decision:
                    await tx.run(
                        f"""
                        MATCH (n:{label} {{id: $id}})
                        MERGE (d:DecisionAnchor {{id: $did}})
                        MERGE (n)-[:TO_DECISION]->(d)
                        """,
                        id=eid, did=to_decision,
                    )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_tx)

    async def delete(self, project_id: str, entity_type: str, entity_id: str) -> bool:
        label = self._label(entity_type)
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (n:{label} {{id: $id, project_id: $pid}})
                WITH n, count(n) AS c
                DETACH DELETE n
                RETURN c
                """,
                id=entity_id, pid=project_id,
            )
            record = await result.single()
        return bool(record and (record["c"] or 0) > 0)

    async def list_for_project(self, project_id: str, entity_type: str) -> list[Any]:
        label = self._label(entity_type)
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"MATCH (n:{label} {{project_id: $pid}}) RETURN n.payload_json AS payload",
                pid=project_id,
            )
            records = [r async for r in result]
        out: list[Any] = []
        for r in records:
            blob = r.get("payload")
            if not blob:
                continue
            entity = _conv_from_json(entity_type, blob)
            if entity is not None:
                out.append(entity)
        return out

    async def known_projects(self) -> list[str]:
        # Union of distinct project_ids across all four bridge labels.
        # We compute it in one query rather than four to keep the round-trip
        # cost flat. Empty / null project_ids are filtered out — the Redis
        # backend has the same behavior (the projects SET only ever holds
        # non-empty strings because put() refuses to add empties).
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (n)
                WHERE (n:Session OR n:BridgeIteration OR n:TurningPoint OR n:DecisionLink)
                  AND n.project_id IS NOT NULL AND n.project_id <> ''
                RETURN DISTINCT n.project_id AS project_id
                """
            )
            return [r["project_id"] async for r in result]


# ─── Factories: build bridge backends from env ────────────────────────

def build_neo4j_driver_from_env() -> tuple[Any, str] | None:
    """Single source of truth for "build a driver from NEO4J_* env vars."

    Returns (driver, database_name) or None if creds are missing / driver
    can't construct. Callers can share this driver across multiple
    backends to avoid spinning up multiple connection pools against the
    same Aura instance."""
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USERNAME", "").strip() or "neo4j"
    password = os.environ.get("NEO4J_PASSWORD", "").strip()
    database = os.environ.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
    if not uri or not password:
        return None
    try:
        from neo4j import AsyncGraphDatabase  # type: ignore[import-not-found]
    except ImportError:
        log.warning("neo4j driver not installed — falling back")
        return None
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    except Exception as e:
        log.warning("failed to construct Neo4j driver: %s", e)
        return None
    return driver, database


def build_neo4j_anchor_backend_from_env() -> Neo4jAnchorBackend | None:
    """Build a standalone Neo4jAnchorBackend (owns its driver). For server
    contexts that already have a shared driver, construct the class
    directly with `owns_driver=False`."""
    built = build_neo4j_driver_from_env()
    if built is None:
        return None
    driver, database = built
    log.info("Neo4jAnchorBackend: built standalone driver (database=%s)", database)
    return Neo4jAnchorBackend(driver, database=database, owns_driver=True)


def build_neo4j_conversation_backend_from_env() -> Neo4jConversationBackend | None:
    """Build a standalone Neo4jConversationBackend (owns its driver). Same
    notes as build_neo4j_anchor_backend_from_env."""
    built = build_neo4j_driver_from_env()
    if built is None:
        return None
    driver, database = built
    log.info("Neo4jConversationBackend: built standalone driver (database=%s)", database)
    return Neo4jConversationBackend(driver, database=database, owns_driver=True)


# ═══════════════════════════════════════════════════════════════════════
# Decision Trace writer — typed event nodes per Iteration
# ═══════════════════════════════════════════════════════════════════════
#
# Persists the output of InlineClassifier (Phase 2b) or
# MarkdownDecisionLogExtractor (Phase 2c) to Neo4j with the dual-label
# pattern (:DecisionTrace:<Type>). One bundle = one Iteration's worth of
# typed events, all committed atomically.
#
# DUAL LABELS
# ===========
# Every node is created with TWO labels:
#   :DecisionTrace  — the source-namespace marker (vs :CodeGraph for graphify)
#   :<TypeLabel>    — :UserMessage, :Decision, :Question, etc.
#
# Application code MATCHes by source label to isolate Decision Trace queries
# from code-graph queries, by type label to find specific kinds.
#
# RELATIONSHIPS BACK TO ITERATION
# ===============================
# Each event MERGEs an edge back to its parent Iteration:
#   (i:Iteration)-[:HAS_USER_MESSAGE]->(:UserMessage)
#   (i:Iteration)-[:HAS_SYSTEM_RESPONSE]->(:SystemResponse)
#   (i:Iteration)-[:MADE_DECISION]->(:Decision)
#   (i:Iteration)-[:RAISED_QUESTION]->(:Question)
#   (i:Iteration)-[:CITED]->(:Reference)
#   (i:Iteration)-[:RECORDED_INSIGHT]->(:Insight)
#
# If a Decision.supersedes is set, also creates:
#   (:Decision {new})-[:SUPERSEDES]->(:Decision {old})
#
# IDEMPOTENCY
# ===========
# Every write uses MERGE on the typed-label/id key. Re-running write_bundle
# with the same node IDs is a no-op (same scalar properties get re-SET, no
# duplicate nodes appear). This matters when the sweeper retries after a
# transient Neo4j hiccup.


class Neo4jDecisionTraceWriter:
    """Writes typed Decision Trace events to Neo4j with the dual-label
    (:DecisionTrace:<Type>) pattern. One bundle commits atomically."""

    def __init__(self, driver: Any, *, database: str = "neo4j", owns_driver: bool = False) -> None:
        self._driver = driver
        self._database = database
        self._owns_driver = owns_driver

    async def close(self) -> None:
        if self._owns_driver:
            try:
                await self._driver.close()
            except Exception as e:
                log.warning("decision-trace writer driver close failed: %s", e)

    # ─── Public API ────────────────────────────────────────────────

    async def write_bundle(self, bundle: DecisionTraceBundle) -> dict[str, int]:
        """Commit every event in the bundle atomically.

        Returns counts of nodes written by type — useful for logging and
        for the sweeper to confirm progress per iteration."""
        counts = {
            "user_message":    0,
            "system_response": 0,
            "decision":        0,
            "question":        0,
            "reference":       0,
            "insight":         0,
        }

        async def _tx(tx):
            if bundle.user_message is not None:
                await self._write_user_message_tx(tx, bundle.user_message)
                counts["user_message"] = 1
            if bundle.system_response is not None:
                await self._write_system_response_tx(tx, bundle.system_response)
                counts["system_response"] = 1
            for d in bundle.decisions:
                await self._write_decision_tx(tx, d)
                counts["decision"] += 1
            for q in bundle.questions:
                await self._write_question_tx(tx, q)
                counts["question"] += 1
            for r in bundle.references:
                await self._write_reference_tx(tx, r)
                counts["reference"] += 1
            for ins in bundle.insights:
                await self._write_insight_tx(tx, ins)
                counts["insight"] += 1

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(_tx)
        return counts

    # ─── Per-type tx helpers (private — called inside one transaction) ─

    async def _write_user_message_tx(self, tx, m: UserMessage) -> None:
        await tx.run(
            """
            MERGE (n:DecisionTrace:UserMessage {id: $id})
            SET n.iteration_id = $iteration_id,
                n.thread_id    = $thread_id,
                n.workspace_id = $workspace_id,
                n.surface_id   = $surface_id,
                n.user_id      = $user_id,
                n.project_id   = $project_id,
                n.text         = $text,
                n.ts           = $ts
            WITH n
            MATCH (i:Iteration {id: $iteration_id})
            MERGE (i)-[:HAS_USER_MESSAGE]->(n)
            """,
            id=m.id, iteration_id=m.iteration_id, thread_id=m.thread_id,
            workspace_id=m.workspace_id, surface_id=m.surface_id,
            user_id=m.user_id, project_id=m.project_id or "",
            text=m.text, ts=m.ts,
        )

    async def _write_system_response_tx(self, tx, r: SystemResponse) -> None:
        await tx.run(
            """
            MERGE (n:DecisionTrace:SystemResponse {id: $id})
            SET n.iteration_id = $iteration_id,
                n.thread_id    = $thread_id,
                n.workspace_id = $workspace_id,
                n.surface_id   = $surface_id,
                n.user_id      = $user_id,
                n.project_id   = $project_id,
                n.text         = $text,
                n.ts           = $ts
            WITH n
            MATCH (i:Iteration {id: $iteration_id})
            MERGE (i)-[:HAS_SYSTEM_RESPONSE]->(n)
            """,
            id=r.id, iteration_id=r.iteration_id, thread_id=r.thread_id,
            workspace_id=r.workspace_id, surface_id=r.surface_id,
            user_id=r.user_id, project_id=r.project_id or "",
            text=r.text, ts=r.ts,
        )

    async def _write_decision_tx(self, tx, d: Decision) -> None:
        await tx.run(
            """
            MERGE (n:DecisionTrace:Decision {id: $id})
            SET n.iteration_id = $iteration_id,
                n.thread_id    = $thread_id,
                n.workspace_id = $workspace_id,
                n.surface_id   = $surface_id,
                n.user_id      = $user_id,
                n.project_id   = $project_id,
                n.text         = $text,
                n.status       = $status,
                n.confidence   = $confidence,
                n.ts           = $ts
            WITH n
            MATCH (i:Iteration {id: $iteration_id})
            MERGE (i)-[:MADE_DECISION]->(n)
            """,
            id=d.id, iteration_id=d.iteration_id, thread_id=d.thread_id,
            workspace_id=d.workspace_id, surface_id=d.surface_id,
            user_id=d.user_id, project_id=d.project_id or "",
            text=d.text, status=d.status, confidence=float(d.confidence),
            ts=d.ts,
        )
        # SUPERSEDES edge — only if the classifier flagged a predecessor.
        # MERGE'd separately so it doesn't fail when the predecessor doesn't
        # exist yet (the predecessor node is MERGE'd by id, so dangling
        # supersedes references self-heal: when the older node later
        # arrives, the relationship resolves to a real edge).
        if d.supersedes:
            await tx.run(
                """
                MERGE (new:DecisionTrace:Decision {id: $new_id})
                MERGE (old:DecisionTrace:Decision {id: $old_id})
                MERGE (new)-[:SUPERSEDES]->(old)
                """,
                new_id=d.id, old_id=d.supersedes,
            )

    async def _write_question_tx(self, tx, q: Question) -> None:
        await tx.run(
            """
            MERGE (n:DecisionTrace:Question {id: $id})
            SET n.iteration_id = $iteration_id,
                n.thread_id    = $thread_id,
                n.workspace_id = $workspace_id,
                n.surface_id   = $surface_id,
                n.user_id      = $user_id,
                n.project_id   = $project_id,
                n.text         = $text,
                n.resolved     = $resolved,
                n.confidence   = $confidence,
                n.ts           = $ts
            WITH n
            MATCH (i:Iteration {id: $iteration_id})
            MERGE (i)-[:RAISED_QUESTION]->(n)
            """,
            id=q.id, iteration_id=q.iteration_id, thread_id=q.thread_id,
            workspace_id=q.workspace_id, surface_id=q.surface_id,
            user_id=q.user_id, project_id=q.project_id or "",
            text=q.text, resolved=bool(q.resolved),
            confidence=float(q.confidence), ts=q.ts,
        )

    async def _write_reference_tx(self, tx, r: Reference) -> None:
        await tx.run(
            """
            MERGE (n:DecisionTrace:Reference {id: $id})
            SET n.iteration_id = $iteration_id,
                n.thread_id    = $thread_id,
                n.workspace_id = $workspace_id,
                n.surface_id   = $surface_id,
                n.user_id      = $user_id,
                n.project_id   = $project_id,
                n.kind         = $kind,
                n.target       = $target,
                n.label        = $label,
                n.confidence   = $confidence,
                n.ts           = $ts
            WITH n
            MATCH (i:Iteration {id: $iteration_id})
            MERGE (i)-[:CITED]->(n)
            """,
            id=r.id, iteration_id=r.iteration_id, thread_id=r.thread_id,
            workspace_id=r.workspace_id, surface_id=r.surface_id,
            user_id=r.user_id, project_id=r.project_id or "",
            kind=r.kind, target=r.target, label=r.label or "",
            confidence=float(r.confidence), ts=r.ts,
        )

    async def _write_insight_tx(self, tx, ins: Insight) -> None:
        await tx.run(
            """
            MERGE (n:DecisionTrace:Insight {id: $id})
            SET n.iteration_id = $iteration_id,
                n.thread_id    = $thread_id,
                n.workspace_id = $workspace_id,
                n.surface_id   = $surface_id,
                n.user_id      = $user_id,
                n.project_id   = $project_id,
                n.text         = $text,
                n.confidence   = $confidence,
                n.ts           = $ts
            WITH n
            MATCH (i:Iteration {id: $iteration_id})
            MERGE (i)-[:RECORDED_INSIGHT]->(n)
            """,
            id=ins.id, iteration_id=ins.iteration_id, thread_id=ins.thread_id,
            workspace_id=ins.workspace_id, surface_id=ins.surface_id,
            user_id=ins.user_id, project_id=ins.project_id or "",
            text=ins.text, confidence=float(ins.confidence), ts=ins.ts,
        )


def build_neo4j_decision_trace_writer_from_env() -> Neo4jDecisionTraceWriter | None:
    """Build a standalone Neo4jDecisionTraceWriter. Mirrors the other
    backend factories — useful for scripts that don't already have a
    shared driver. In server.py the writer should share the driver with
    Neo4jThreadStore once driver consolidation lands."""
    built = build_neo4j_driver_from_env()
    if built is None:
        return None
    driver, database = built
    log.info("Neo4jDecisionTraceWriter: built standalone driver (database=%s)", database)
    return Neo4jDecisionTraceWriter(driver, database=database, owns_driver=True)
