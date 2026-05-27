"""
MemoryRetriever — the read path for Decision Trace.

ROLE IN THE PIPELINE
====================
Phase 1 added provenance fields to Iterations. Phase 2 built the typed
event nodes + classifier + markdown parser. Phase 3 made structuring
automatic via the background sweeper. This file is what reads it all
back when the model needs context for a new turn.

THE CROSS-EVERYTHING CONTRACT
=============================
Every retrieval is scoped by `user_id` (REQUIRED — no cross-user leakage
ever). Within the user's data, the retriever optionally narrows by:

    project_id    — only entries in this project
    workspace_id  — only from this platform (cursor / web / claude / ...)
    surface_id    — only from this surface (chat / map-room / wandering-room)
    thread_id     — only this thread (with cross_thread=False)

Default behavior (cross_thread=True): pull recent turns from `thread_id`
AND semantically-similar entries from EVERY other thread the user owns.
The model reading the rendered timeline never sees thread boundaries
unless we explicitly tag them — which we DO (every row carries
thread_id + thread_title) so the model can cite by name.

THE TWO RETRIEVAL PATHS
=======================
1. **Local thread context** (when thread_id is provided):
   Last N iterations in the thread, ordered by sequence_num DESC. The
   model gets the "what just happened" view.

2. **Cross-thread vector recall** (when cross_thread=True and query
   produces a vector):
   Top-K iterations across the user's WHOLE history ranked by cosine
   similarity on Iteration.embedding. From each matched iteration, pull
   its typed events. The model gets the "what's relevant from anywhere"
   view.

These compose: the dispatcher's pre-LLM context gets both, rendered as
one markdown timeline (CURRENT THREAD section + CROSS-THREAD section).

OUTPUT SHAPE
============
`MemoryEntry` = one iteration's worth of context:
    - provenance (thread_id, thread_title, workspace_id, surface_id, ts)
    - user_message + system_response (verbatim)
    - decisions / questions / references / insights (lists, may be empty)
    - score (cosine similarity, only for cross-thread matches)
    - source ("local" | "cross_thread") — distinguishes mode in the renderer

`RetrievalResult` = the bundle for one query (list of entries + metadata).

SAFETY
======
- user_id is REQUIRED and used in EVERY filter — defense in depth.
- Empty input or no API key for the embedder → graceful: returns local
  thread context only, skips vector path. The model still gets recent
  turns.
- Cypher uses parameterized queries — no string interpolation, no
  injection surface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.bridge.embedding_service import GeminiEmbeddingService
from src.bridge.neo4j_backend import Neo4jThreadStore

log = logging.getLogger("constellax.memory_retriever")


# ─── Result types ────────────────────────────────────────────────────

@dataclass
class MemoryEvent:
    """A flattened view of any DecisionTrace event for the renderer.

    The retriever doesn't need the full type-specific shape of each
    event (UserMessage vs Decision vs Reference) — the renderer only
    cares about kind, text, and a handful of fields. This wrapper
    collapses six dataclasses into one so the render path is uniform."""
    kind:        str               # "user_message"|"system_response"|"decision"|"question"|"reference"|"insight"
    text:        str
    ts:          float
    confidence:  float = 1.0       # 1.0 for verbatim events (user_message, system_response)
    status:      str = ""          # decision status, when kind=="decision"
    resolved:    bool = False      # question resolved, when kind=="question"
    ref_kind:    str = ""          # url/file/etc, when kind=="reference"
    ref_target:  str = ""          # the actual reference value
    ref_label:   str = ""          # human-readable label for the reference


@dataclass
class MemoryEntry:
    """One iteration's contribution to a retrieval result. Carries full
    provenance plus all typed events from that iteration."""
    thread_id:     str
    thread_title:  str
    workspace_id:  str
    surface_id:    str
    iteration_id:  str
    sequence_num:  int
    ts:            float
    source:        str             # "local" | "cross_thread"
    score:         float | None    # cosine similarity for cross_thread; None for local
    user_message:  MemoryEvent | None = None
    system_response: MemoryEvent | None = None
    decisions:     list[MemoryEvent] = field(default_factory=list)
    questions:     list[MemoryEvent] = field(default_factory=list)
    references:    list[MemoryEvent] = field(default_factory=list)
    insights:      list[MemoryEvent] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """The bundle for one retrieve() call.

    `local` — entries from the current thread (when thread_id was provided).
    `cross_thread` — entries from other threads ranked by similarity.
    Both lists carry `MemoryEntry` shape; the renderer groups them into
    separate sections.

    `meta` captures latency + counts for observability."""
    query:          str
    user_id:        str
    local:          list[MemoryEntry] = field(default_factory=list)
    cross_thread:   list[MemoryEntry] = field(default_factory=list)
    latency_ms:     int = 0
    embedded_ok:    bool = False
    vector_dim:     int | None = None
    error:          str | None = None


# ─── The retriever ───────────────────────────────────────────────────

class MemoryRetriever:
    """Reads Decision Trace from Neo4j and returns LLM-ready entries.

    Composes with a Neo4jThreadStore (driver access for Cypher) and a
    GeminiEmbeddingService (for the cross-thread vector path). The
    embedder is optional — if absent, vector recall returns empty and
    the local-thread path still works."""

    def __init__(
        self,
        store: Neo4jThreadStore,
        embedder: GeminiEmbeddingService | None = None,
        *,
        vector_index: str = "iteration_embedding_idx",
    ):
        self.store = store
        self.embedder = embedder
        self.vector_index = vector_index

    async def retrieve(
        self,
        query: str,
        *,
        user_id: str,                       # REQUIRED — never cross-user
        project_id: str | None = None,
        workspace_id: str | None = None,
        surface_id: str | None = None,
        thread_id: str | None = None,
        cross_thread: bool = True,
        k_local: int = 5,
        k_cross: int = 5,
    ) -> RetrievalResult:
        """Run the read path. Returns local + cross-thread entries.

        - user_id MUST be non-empty (we never default to "everyone").
        - thread_id: if provided, fetches recent local turns. If absent, only
          cross-thread results (when cross_thread=True).
        - cross_thread: if True and the embedder produces a vector, runs
          a vector query scoped to the user. Optional filters (project_id,
          workspace_id, surface_id) narrow the candidate set.
        - k_local / k_cross: per-section caps. Default keeps prompt size
          manageable; bump for analysis use cases."""
        start = time.time()
        result = RetrievalResult(query=query, user_id=user_id)
        if not user_id:
            result.error = "user_id is required"
            return result

        # 1) Local thread context — always runs when a thread_id is given.
        if thread_id:
            try:
                result.local = await self._fetch_local_thread(
                    thread_id=thread_id, user_id=user_id, k=k_local,
                )
            except Exception as e:
                log.warning("retriever: local fetch failed for thread %s: %s", thread_id, e)
                result.error = (result.error or "") + f"local:{type(e).__name__};"

        # 2) Cross-thread vector recall — only runs if cross_thread is on
        #    AND we have an embedder AND it produces a vector. Anything that
        #    fails silently degrades to "local only" with a logged warning.
        if cross_thread and self.embedder is not None and (query or "").strip():
            try:
                emb = await self.embedder.embed(query)
                if emb.success and emb.vector:
                    result.embedded_ok = True
                    result.vector_dim = len(emb.vector)
                    result.cross_thread = await self._fetch_cross_thread(
                        query_vector=emb.vector,
                        user_id=user_id,
                        project_id=project_id,
                        workspace_id=workspace_id,
                        surface_id=surface_id,
                        exclude_thread_id=thread_id,
                        k=k_cross,
                    )
                else:
                    log.info("retriever: embed failed (%s) — skipping vector path", emb.error)
            except Exception as e:
                log.warning("retriever: cross-thread vector path failed: %s", e)
                result.error = (result.error or "") + f"cross:{type(e).__name__};"

        result.latency_ms = int((time.time() - start) * 1000)
        return result

    # ─── Cypher: local thread ────────────────────────────────────────

    async def _fetch_local_thread(
        self, *, thread_id: str, user_id: str, k: int,
    ) -> list[MemoryEntry]:
        """Last `k` iterations in the given thread, with all typed events.

        Filters by user_id at the Thread level (Thread.user_id matches),
        AND verifies the User-OWNS-Thread edge — belt-and-suspenders against
        cross-user contamination."""
        async with self.store._driver.session(database=self.store._database) as session:
            result = await session.run(
                """
                MATCH (u:User {id: $uid})-[:OWNS]->(t:Thread {id: $tid})
                MATCH (t)-[:HAS_ITERATION]->(i:Iteration)
                WITH t, i
                ORDER BY i.sequence_num DESC
                LIMIT $k
                CALL (i) {
                    OPTIONAL MATCH (i)-[:HAS_USER_MESSAGE]->(um:UserMessage)
                    OPTIONAL MATCH (i)-[:HAS_SYSTEM_RESPONSE]->(sr:SystemResponse)
                    OPTIONAL MATCH (i)-[:MADE_DECISION]->(d:Decision)
                    OPTIONAL MATCH (i)-[:RAISED_QUESTION]->(q:Question)
                    OPTIONAL MATCH (i)-[:CITED]->(ref:Reference)
                    OPTIONAL MATCH (i)-[:RECORDED_INSIGHT]->(ins:Insight)
                    RETURN um, sr,
                        collect(DISTINCT d)   AS decisions,
                        collect(DISTINCT q)   AS questions,
                        collect(DISTINCT ref) AS references,
                        collect(DISTINCT ins) AS insights
                }
                RETURN t.id AS thread_id, t.title AS thread_title,
                       i.id AS iter_id, i.sequence_num AS seq,
                       i.workspace_id AS workspace_id, i.surface_id AS surface_id,
                       i.completed_at AS ts,
                       um, sr, decisions, questions, references, insights
                ORDER BY i.sequence_num DESC
                """,
                uid=user_id, tid=thread_id, k=int(k),
            )
            return [self._row_to_entry(rec, source="local", score=None) async for rec in result]

    # ─── Cypher: cross-thread vector recall ─────────────────────────

    async def _fetch_cross_thread(
        self,
        *,
        query_vector: list[float],
        user_id: str,
        project_id: str | None,
        workspace_id: str | None,
        surface_id: str | None,
        exclude_thread_id: str | None,
        k: int,
    ) -> list[MemoryEntry]:
        """Vector-similar iterations from any of this user's threads,
        excluding the current thread (which is handled by local path).

        Strategy: ask the vector index for more candidates than we need
        (`k * 3`), then filter by user / project / workspace / surface in
        Cypher and trim to `k`. Trimming after filter avoids the edge
        case where all top-K candidates belong to other users."""
        async with self.store._driver.session(database=self.store._database) as session:
            result = await session.run(
                """
                CALL db.index.vector.queryNodes($idx, $candidates, $vec)
                YIELD node AS i, score
                MATCH (u:User {id: $uid})-[:OWNS]->(t:Thread)-[:HAS_ITERATION]->(i)
                WHERE ($exclude IS NULL OR t.id <> $exclude)
                  AND ($pid IS NULL OR t.project_id = $pid)
                  AND ($wid IS NULL OR coalesce(i.workspace_id, '') = $wid)
                  AND ($sid IS NULL OR coalesce(i.surface_id, '') = $sid)
                WITH t, i, score
                CALL (i) {
                    OPTIONAL MATCH (i)-[:HAS_USER_MESSAGE]->(um:UserMessage)
                    OPTIONAL MATCH (i)-[:HAS_SYSTEM_RESPONSE]->(sr:SystemResponse)
                    OPTIONAL MATCH (i)-[:MADE_DECISION]->(d:Decision)
                    OPTIONAL MATCH (i)-[:RAISED_QUESTION]->(q:Question)
                    OPTIONAL MATCH (i)-[:CITED]->(ref:Reference)
                    OPTIONAL MATCH (i)-[:RECORDED_INSIGHT]->(ins:Insight)
                    RETURN um, sr,
                        collect(DISTINCT d)   AS decisions,
                        collect(DISTINCT q)   AS questions,
                        collect(DISTINCT ref) AS references,
                        collect(DISTINCT ins) AS insights
                }
                RETURN t.id AS thread_id, t.title AS thread_title,
                       i.id AS iter_id, i.sequence_num AS seq,
                       i.workspace_id AS workspace_id, i.surface_id AS surface_id,
                       i.completed_at AS ts,
                       um, sr, decisions, questions, references, insights, score
                ORDER BY score DESC
                LIMIT $k
                """,
                idx=self.vector_index, candidates=int(k) * 3, vec=list(query_vector),
                uid=user_id, exclude=exclude_thread_id,
                pid=project_id, wid=workspace_id, sid=surface_id,
                k=int(k),
            )
            return [self._row_to_entry(rec, source="cross_thread", score=rec["score"]) async for rec in result]

    # ─── Row → MemoryEntry conversion ───────────────────────────────

    def _row_to_entry(self, rec: Any, *, source: str, score: float | None) -> MemoryEntry:
        """Translate a single Cypher result row into a MemoryEntry. The
        row is expected to carry: thread_id, thread_title, iter_id, seq,
        workspace_id, surface_id, ts, um, sr, decisions[], questions[],
        references[], insights[]."""
        return MemoryEntry(
            thread_id=rec["thread_id"] or "",
            thread_title=rec["thread_title"] or "(untitled)",
            workspace_id=rec["workspace_id"] or "",
            surface_id=rec["surface_id"] or "",
            iteration_id=rec["iter_id"] or "",
            sequence_num=int(rec["seq"] or 0),
            ts=float(rec["ts"] or 0.0),
            source=source,
            score=float(score) if score is not None else None,
            user_message=_node_to_event(rec["um"], "user_message"),
            system_response=_node_to_event(rec["sr"], "system_response"),
            decisions=[_node_to_event(n, "decision") for n in (rec["decisions"] or []) if n],
            questions=[_node_to_event(n, "question") for n in (rec["questions"] or []) if n],
            references=[_node_to_event(n, "reference") for n in (rec["references"] or []) if n],
            insights=[_node_to_event(n, "insight") for n in (rec["insights"] or []) if n],
        )


# ─── Neo4j node → MemoryEvent ────────────────────────────────────────

def _node_to_event(node: Any, kind: str) -> MemoryEvent | None:
    """Defensive node-to-event conversion. Returns None for falsy/missing
    nodes so the caller can use `if event` to test presence."""
    if node is None:
        return None
    # neo4j.Node is dict-like; use .get() pattern
    try:
        get = node.get
    except AttributeError:
        return None
    text = get("text") or ""
    if kind == "reference":
        # References have no .text — they carry kind/target/label instead.
        text = get("label") or get("target") or ""
    return MemoryEvent(
        kind=kind,
        text=text,
        ts=float(get("ts") or 0.0),
        confidence=float(get("confidence") or 1.0),
        status=str(get("status") or ""),
        resolved=bool(get("resolved") or False),
        ref_kind=str(get("kind") or "") if kind == "reference" else "",
        ref_target=str(get("target") or "") if kind == "reference" else "",
        ref_label=str(get("label") or "") if kind == "reference" else "",
    )


# ─── Renderer ────────────────────────────────────────────────────────

def render_timeline(result: RetrievalResult, *, show_provenance: bool = True) -> str:
    """Format a RetrievalResult as LLM-readable markdown.

    Two sections: CURRENT THREAD (chronological) and CROSS-THREAD MEMORY
    (similarity-ranked). Each entry shows the verbatim user/system text
    plus typed event annotations (decisions / questions / references /
    insights). The model reading this picks up the conversation history
    without parsing JSON or graph structure."""
    parts: list[str] = []
    parts.append(f'## Memory recall for: "{_clip(result.query, 200)}"')

    if result.local:
        parts.append("")
        parts.append(f"### CURRENT THREAD — \"{result.local[0].thread_title}\""
                     + (_provenance_suffix(result.local[0]) if show_provenance else ""))
        # Local entries arrive newest-first from Cypher; reverse for readable
        # chronological order in the rendered timeline.
        for entry in reversed(result.local):
            parts.extend(_render_entry(entry, show_score=False))

    if result.cross_thread:
        parts.append("")
        parts.append(f"### CROSS-THREAD MEMORY ({len(result.cross_thread)} relevant past iterations)")
        for entry in result.cross_thread:
            parts.append("")
            parts.append(f"#### From \"{entry.thread_title}\""
                         + (_provenance_suffix(entry) if show_provenance else ""))
            parts.extend(_render_entry(entry, show_score=True))

    if not result.local and not result.cross_thread:
        parts.append("")
        parts.append("(no relevant memory found for this query)")
        if result.error:
            parts.append(f"(retriever note: {result.error.rstrip(';')})")

    return "\n".join(parts)


def _render_entry(entry: MemoryEntry, *, show_score: bool) -> list[str]:
    """Render one MemoryEntry as a Turn block. Lines are flat-indented
    Markdown — no nested code fences. The LLM reads this naturally."""
    lines: list[str] = []
    header = f"- Turn {entry.sequence_num}"
    if entry.ts:
        header += f" ({_format_ts(entry.ts)})"
    if show_score and entry.score is not None:
        header += f" — sim={entry.score:.2f}"
    lines.append("")
    lines.append(header)
    if entry.user_message and entry.user_message.text:
        lines.append(f"  user: {_clip(entry.user_message.text, 600)}")
    if entry.system_response and entry.system_response.text:
        lines.append(f"  system: {_clip(entry.system_response.text, 800)}")
    if entry.decisions:
        for d in entry.decisions:
            lines.append(f"  decision ({d.status}, conf {d.confidence:.2f}): {_clip(d.text, 240)}")
    if entry.questions:
        for q in entry.questions:
            resolved = "resolved" if q.resolved else "open"
            lines.append(f"  question ({resolved}, conf {q.confidence:.2f}): {_clip(q.text, 240)}")
    if entry.references:
        for r in entry.references:
            label_part = f" — {r.ref_label}" if r.ref_label else ""
            lines.append(f"  reference ({r.ref_kind}): {r.ref_target}{label_part}")
    if entry.insights:
        for i in entry.insights:
            lines.append(f"  insight (conf {i.confidence:.2f}): {_clip(i.text, 240)}")
    return lines


def _provenance_suffix(entry: MemoryEntry) -> str:
    bits = []
    if entry.workspace_id: bits.append(entry.workspace_id)
    if entry.surface_id and entry.surface_id != "chat": bits.append(entry.surface_id)
    return f" [{', '.join(bits)}]" if bits else ""


def _clip(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else (s[:n - 1] + "…")


def _format_ts(ts: float) -> str:
    """Format a Unix timestamp as YYYY-MM-DD HH:MM UTC. We deliberately
    use UTC so renderings are stable across operator timezones."""
    import datetime
    try:
        dt = datetime.datetime.utcfromtimestamp(float(ts))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return ""
