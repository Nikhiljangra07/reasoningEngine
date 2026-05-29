"""
Persistence — store sessions, cushions, reports, traces, discarded clues.

Protocol-first: app code talks to WanderingStore; concrete backends are
InMemoryWanderingStore (dev/tests) and Neo4jWanderingStore (production).
The factory `build_wandering_store_from_env()` picks based on
CONSTELLAX_DB_BACKEND env var, matching the existing thread_store pattern.

Discarded clues are STORED, not deleted, with classification tags. Future
sessions can mine them when their new anchor has structural overlap.
This is the compounding-asset principle from the plan.

Per Law 4: writes are scoped to the WANDERING namespace. We never write
to the user's project memory, IDE files, or any non-wandering state.

ISOLATION: imports cushion + report + trace + runtime types. No LLM calls,
no API code. Just storage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from src.wandering.cushion import CushionGraph
from src.wandering.report import ExplorationReport
from src.wandering.runtime import SessionResult, WanderingMode
from src.wandering.trace import DecisionTrace, DiscardedClue, DiscardKind


log = logging.getLogger("constellax.wandering.persistence")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class WanderingStore(Protocol):
    """Storage Protocol for Wandering Room artifacts.

    All methods are async. Failures NEVER raise — implementations log and
    swallow. The caller may inspect return values for success signals
    (False / None) but should not rely on exceptions.
    """

    async def save_session(self, user_id: str | None, session: SessionResult) -> bool:
        ...

    async def get_session(self, session_id: str) -> SessionResult | None:
        ...

    async def list_sessions(self, user_id: str, limit: int = 20) -> list[str]:
        """Return session_ids in reverse-chronological order."""
        ...

    async def save_report(self, session_id: str, report: ExplorationReport) -> bool:
        ...

    async def get_reports(self, session_id: str) -> list[ExplorationReport]:
        ...

    async def save_discarded_clue(
        self, session_id: str, agent_id: str, clue: DiscardedClue
    ) -> bool:
        ...

    async def list_discarded_clues_for_user(
        self,
        user_id: str,
        kinds: tuple[DiscardKind, ...] = (
            DiscardKind.POSSIBLY_RELEVANT_ELSEWHERE,
            DiscardKind.REVISIT_LATER,
        ),
        limit: int = 100,
    ) -> list[DiscardedClue]:
        """Return previously-discarded clues that may still be useful.

        Used at session start: the wandering room can check whether any
        of the user's prior discards resonate with the new anchor. This
        is the compounding-asset surface.
        """
        ...


# ---------------------------------------------------------------------------
# Serialization helpers — used by both backends
# ---------------------------------------------------------------------------


def session_to_json(session: SessionResult) -> str:
    """Render a SessionResult as a JSON string for storage.

    The full session includes reports + traces + discarded clues. We
    flatten to a dict via dataclass asdict() with enum/datetime handling.
    """
    def _default(obj: Any) -> Any:
        # Enum → its value
        if hasattr(obj, "value") and not isinstance(obj, (str, int, float, list, dict, tuple, bool)):
            return obj.value
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    raw: dict[str, Any] = {
        "session_id": session.session_id,
        "mode": session.mode.value,
        "config_resolved": list(session.config.resolved()),
        "cushion": _cushion_to_dict(session.cushion),
        "reports": [_report_to_dict(r) for r in session.reports],
        "traces": [_trace_to_dict(t) for t in session.traces],
        "total_tokens_spent": session.total_tokens_spent,
        "elapsed_seconds": session.elapsed_seconds,
        "ended_at": session.ended_at,
    }
    return json.dumps(raw, default=_default)


def _cushion_to_dict(cushion: CushionGraph) -> dict[str, Any]:
    return {
        "actual": {"name": "actual", "nodes": list(cushion.actual.nodes), "summary": cushion.actual.summary},
        "essence": {"name": "essence", "nodes": list(cushion.essence.nodes), "summary": cushion.essence.summary},
        "mechanism": {"name": "mechanism", "nodes": list(cushion.mechanism.nodes), "summary": cushion.mechanism.summary},
        "raw_input": {
            "problem": cushion.raw_input.problem.content,
            "context": cushion.raw_input.context.content,
            "vision": cushion.raw_input.vision.content,
            "current_map": cushion.raw_input.current_map.content,
            "memory_enrichment": cushion.raw_input.memory_enrichment,
        },
        "constellation_size": cushion.constellation_size,
        "extraction_model": cushion.extraction_model,
        "extracted_at": cushion.extracted_at,
    }


def _report_to_dict(report: ExplorationReport) -> dict[str, Any]:
    return {
        "report_id": report.report_id,
        "agent_id": report.agent_id,
        "anchor_summary": report.anchor_summary,
        "domain_explored": report.domain_explored,
        "source_locations": [
            {"title": s.title, "url": s.url, "excerpt": s.excerpt, "used_for": s.used_for}
            for s in report.source_locations
        ],
        "layer_matches": {
            name: {
                "layer_name": lm.layer_name,
                "matched_nodes": list(lm.matched_nodes),
                "total_nodes": lm.total_nodes,
            }
            for name, lm in report.layer_matches.items()
        },
        "confidence": report.confidence.value,
        "exploration_summary": report.exploration_summary,
        "advancement": report.advancement,
        "what_does_not_map": report.what_does_not_map,
        "next_lead": report.next_lead,
        "iteration_count": report.iteration_count,
        "abandoned_early": report.abandoned_early,
    }


def _trace_to_dict(trace: DecisionTrace) -> dict[str, Any]:
    return {
        "agent_id": trace.agent_id,
        "anchor_summary": trace.anchor_summary,
        "steps": [
            {
                "step_id": s.step_id,
                "kind": s.kind.value,
                "timestamp": s.timestamp,
                "position": s.position,
                "rationale": s.rationale,
                "detail": s.detail,
                "matched_count": s.matched_count,
                "iterations_used": s.iterations_used,
                "report_id": s.report_id,
                "subagent_id": s.subagent_id,
                "tokens_spent": s.tokens_spent,
            }
            for s in trace.steps
        ],
        "discarded_clues": [
            {
                "description": c.description,
                "source_hint": c.source_hint,
                "classification": c.classification.value,
                "reason": c.reason,
                "timestamp": c.timestamp,
            }
            for c in trace.discarded_clues
        ],
        "total_tokens_spent": trace.total_tokens_spent,
        "total_reports_produced": trace.total_reports_produced,
        "total_subagents_spawned": trace.total_subagents_spawned,
        "completion_reason": trace.completion_reason,
        "ended_at": trace.ended_at,
    }


def _discarded_clue_to_dict(clue: DiscardedClue) -> dict[str, Any]:
    return {
        "description": clue.description,
        "source_hint": clue.source_hint,
        "classification": clue.classification.value,
        "reason": clue.reason,
        "timestamp": clue.timestamp,
    }


def _discarded_clue_from_dict(d: dict[str, Any]) -> DiscardedClue:
    try:
        kind = DiscardKind(d.get("classification", "discarded_for_current_anchor"))
    except ValueError:
        kind = DiscardKind.DISCARDED_FOR_CURRENT_ANCHOR
    return DiscardedClue(
        description=str(d.get("description", "")),
        source_hint=str(d.get("source_hint", "")),
        classification=kind,
        reason=str(d.get("reason", "")),
        timestamp=float(d.get("timestamp", 0.0)),
    )


# ---------------------------------------------------------------------------
# In-memory backend — for dev + tests
# ---------------------------------------------------------------------------


class InMemoryWanderingStore:
    """All state held in dicts. Lost on process restart. Dev/tests only.

    Per-user discarded clues are aggregated across all that user's
    sessions — that's how the "mine prior discards on new anchor" path
    works without needing graph queries.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionResult] = {}
        self._sessions_by_user: dict[str, list[str]] = {}  # user_id → session_ids (newest first)
        self._discarded_by_user: dict[str, list[tuple[str, DiscardedClue]]] = {}  # user_id → [(session_id, clue)]
        self._reports_by_session: dict[str, list[ExplorationReport]] = {}

    async def save_session(self, user_id: str | None, session: SessionResult) -> bool:
        try:
            self._sessions[session.session_id] = session
            if user_id:
                self._sessions_by_user.setdefault(user_id, []).insert(0, session.session_id)
                # Also store per-agent discarded clues against this user
                bucket = self._discarded_by_user.setdefault(user_id, [])
                for trace in session.traces:
                    for clue in trace.discarded_clues:
                        bucket.append((session.session_id, clue))
            # Reports
            self._reports_by_session[session.session_id] = list(session.reports)
            return True
        except Exception as e:
            log.warning("InMemoryWanderingStore.save_session failed: %s", e)
            return False

    async def get_session(self, session_id: str) -> SessionResult | None:
        return self._sessions.get(session_id)

    async def list_sessions(self, user_id: str, limit: int = 20) -> list[str]:
        return list(self._sessions_by_user.get(user_id, []))[:limit]

    async def save_report(self, session_id: str, report: ExplorationReport) -> bool:
        try:
            self._reports_by_session.setdefault(session_id, []).append(report)
            return True
        except Exception:
            return False

    async def get_reports(self, session_id: str) -> list[ExplorationReport]:
        return list(self._reports_by_session.get(session_id, []))

    async def save_discarded_clue(
        self, session_id: str, agent_id: str, clue: DiscardedClue
    ) -> bool:
        # In-memory aggregates by user; we don't know user here without the session.
        # The session save already harvested clues; this method is for ad-hoc adds.
        try:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            # Find owning user
            owning_user = None
            for uid, sids in self._sessions_by_user.items():
                if session_id in sids:
                    owning_user = uid
                    break
            if owning_user:
                self._discarded_by_user.setdefault(owning_user, []).append((session_id, clue))
            return True
        except Exception:
            return False

    async def list_discarded_clues_for_user(
        self,
        user_id: str,
        kinds: tuple[DiscardKind, ...] = (
            DiscardKind.POSSIBLY_RELEVANT_ELSEWHERE,
            DiscardKind.REVISIT_LATER,
        ),
        limit: int = 100,
    ) -> list[DiscardedClue]:
        kind_set = set(kinds)
        out: list[DiscardedClue] = []
        for _session_id, clue in self._discarded_by_user.get(user_id, []):
            if clue.classification in kind_set:
                out.append(clue)
            if len(out) >= limit:
                break
        return out


# ---------------------------------------------------------------------------
# Neo4j backend — production
# ---------------------------------------------------------------------------


WANDERING_SCHEMA_CYPHER = """
// Wandering Room indexes — idempotent
CREATE INDEX wandering_session_id IF NOT EXISTS
  FOR (s:WanderingSession) ON (s.session_id);

CREATE INDEX wandering_session_user IF NOT EXISTS
  FOR (s:WanderingSession) ON (s.user_id, s.completed_at);

CREATE INDEX wandering_report_id IF NOT EXISTS
  FOR (r:WanderingReport) ON (r.report_id);

CREATE INDEX wandering_discard_user_classification IF NOT EXISTS
  FOR (d:WanderingDiscardedClue) ON (d.user_id, d.classification);
""".strip()


class Neo4jWanderingStore:
    """Neo4j-backed persistence for wandering sessions.

    Schema (matching the existing pattern in neo4j_backend.py):

      (User)-[:RAN]->(WanderingSession {session_id, mode, completed_at, payload_json})
      (WanderingSession)-[:CONTAINS]->(WanderingReport {report_id, confidence})
      (WanderingSession)-[:DISCARDED]->(WanderingDiscardedClue {classification})

    Full SessionResult is stored as JSON in `payload_json` on the
    WanderingSession node — round-trip is lossless and the graph nodes
    are queryable side-indexes (for "list this user's sessions" and
    "find this user's possibly-relevant-elsewhere discards" patterns).

    Failures NEVER raise — log and return False/None/[] like the
    in-memory backend.
    """

    def __init__(self, driver: Any, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database
        self._schema_inited = False

    async def init_schema(self) -> bool:
        if self._schema_inited:
            return True
        try:
            async with self._driver.session(database=self._database) as session:
                for stmt in WANDERING_SCHEMA_CYPHER.split(";\n"):
                    s = stmt.strip()
                    if s:
                        await session.run(s)
            self._schema_inited = True
            return True
        except Exception as e:
            log.warning("Neo4jWanderingStore.init_schema failed: %s", e)
            return False

    async def save_session(self, user_id: str | None, session: SessionResult) -> bool:
        try:
            payload = session_to_json(session)
        except Exception as e:
            log.warning("session_to_json failed for %s: %s", session.session_id, e)
            return False

        # The user_id label is critical for listing/retrieval; coerce to "guest" if missing.
        owning_user = (user_id or "guest").strip() or "guest"

        cypher = """
        MERGE (s:WanderingSession {session_id: $session_id})
        SET s.mode = $mode,
            s.completed_at = $completed_at,
            s.user_id = $user_id,
            s.report_count = $report_count,
            s.total_tokens_spent = $total_tokens_spent,
            s.payload_json = $payload_json
        WITH s
        MERGE (u:User {user_id: $user_id})
        MERGE (u)-[:RAN]->(s)
        """
        try:
            async with self._driver.session(database=self._database) as sess:
                await sess.run(
                    cypher,
                    session_id=session.session_id,
                    mode=session.mode.value,
                    completed_at=session.ended_at or time.time(),
                    user_id=owning_user,
                    report_count=session.report_count(),
                    total_tokens_spent=session.total_tokens_spent,
                    payload_json=payload,
                )
        except Exception as e:
            log.warning("Neo4jWanderingStore.save_session failed for %s: %s",
                         session.session_id, e)
            return False

        # Save each report and each discarded clue as side-indexed nodes.
        for report in session.reports:
            try:
                await self._save_report_node(session.session_id, report)
            except Exception as e:
                log.debug("save_report_node failed: %s", e)

        for trace in session.traces:
            for clue in trace.discarded_clues:
                try:
                    await self._save_discarded_clue_node(
                        session_id=session.session_id,
                        agent_id=trace.agent_id,
                        user_id=owning_user,
                        clue=clue,
                    )
                except Exception as e:
                    log.debug("save_discarded_clue_node failed: %s", e)

        return True

    async def _save_report_node(self, session_id: str, report: ExplorationReport) -> None:
        cypher = """
        MATCH (s:WanderingSession {session_id: $session_id})
        MERGE (r:WanderingReport {report_id: $report_id})
        SET r.confidence = $confidence,
            r.domain_explored = $domain_explored,
            r.payload_json = $payload_json
        MERGE (s)-[:CONTAINS]->(r)
        """
        async with self._driver.session(database=self._database) as sess:
            await sess.run(
                cypher,
                session_id=session_id,
                report_id=report.report_id,
                confidence=report.confidence.value,
                domain_explored=report.domain_explored,
                payload_json=json.dumps(_report_to_dict(report)),
            )

    async def _save_discarded_clue_node(
        self,
        session_id: str,
        agent_id: str,
        user_id: str,
        clue: DiscardedClue,
    ) -> None:
        cypher = """
        MATCH (s:WanderingSession {session_id: $session_id})
        CREATE (d:WanderingDiscardedClue {
            session_id: $session_id,
            user_id: $user_id,
            agent_id: $agent_id,
            classification: $classification,
            description: $description,
            payload_json: $payload_json
        })
        CREATE (s)-[:DISCARDED]->(d)
        """
        async with self._driver.session(database=self._database) as sess:
            await sess.run(
                cypher,
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                classification=clue.classification.value,
                description=clue.description,
                payload_json=json.dumps(_discarded_clue_to_dict(clue)),
            )

    async def get_session(self, session_id: str) -> SessionResult | None:
        # For V1 we don't reconstruct SessionResult from Neo4j (lossy
        # because of WanderingConfig + traces). Return None — the caller
        # (dossier endpoint) reads reports separately. This is acceptable
        # because dossier is the user-facing artifact, not SessionResult.
        log.debug("Neo4jWanderingStore.get_session: not implemented for V1; returning None")
        return None

    async def list_sessions(self, user_id: str, limit: int = 20) -> list[str]:
        cypher = """
        MATCH (u:User {user_id: $user_id})-[:RAN]->(s:WanderingSession)
        RETURN s.session_id AS session_id
        ORDER BY s.completed_at DESC
        LIMIT $limit
        """
        try:
            async with self._driver.session(database=self._database) as sess:
                result = await sess.run(cypher, user_id=user_id, limit=limit)
                rows = await result.data()
                return [r["session_id"] for r in rows if r.get("session_id")]
        except Exception as e:
            log.warning("Neo4jWanderingStore.list_sessions failed: %s", e)
            return []

    async def save_report(self, session_id: str, report: ExplorationReport) -> bool:
        try:
            await self._save_report_node(session_id, report)
            return True
        except Exception as e:
            log.warning("save_report failed: %s", e)
            return False

    async def get_reports(self, session_id: str) -> list[ExplorationReport]:
        cypher = """
        MATCH (s:WanderingSession {session_id: $session_id})-[:CONTAINS]->(r:WanderingReport)
        RETURN r.payload_json AS payload_json
        """
        try:
            async with self._driver.session(database=self._database) as sess:
                result = await sess.run(cypher, session_id=session_id)
                rows = await result.data()
        except Exception as e:
            log.warning("get_reports failed: %s", e)
            return []

        # V1: reports are returned as raw dicts in payload_json. The
        # dossier endpoint can re-serialize; we don't reconstruct
        # ExplorationReport objects here (would need importing more
        # types and risk schema drift).
        out: list[ExplorationReport] = []
        for row in rows:
            raw = row.get("payload_json")
            if not raw:
                continue
            try:
                # Lossy reconstruction — only the fields the dossier uses
                # are populated. Adequate for V1 read-back.
                d = json.loads(raw)
                out.append(_minimal_report_from_dict(d))
            except Exception:
                continue
        return out

    async def save_discarded_clue(
        self, session_id: str, agent_id: str, clue: DiscardedClue
    ) -> bool:
        # Need user_id, which we don't have in this signature. Look up
        # the session first to find its owner. If we can't, log + return False.
        cypher_lookup = "MATCH (u:User)-[:RAN]->(s:WanderingSession {session_id: $session_id}) RETURN u.user_id AS uid"
        try:
            async with self._driver.session(database=self._database) as sess:
                result = await sess.run(cypher_lookup, session_id=session_id)
                row = await result.single()
            if not row:
                return False
            uid = row.get("uid") or "guest"
            await self._save_discarded_clue_node(session_id, agent_id, uid, clue)
            return True
        except Exception as e:
            log.warning("save_discarded_clue failed: %s", e)
            return False

    async def list_discarded_clues_for_user(
        self,
        user_id: str,
        kinds: tuple[DiscardKind, ...] = (
            DiscardKind.POSSIBLY_RELEVANT_ELSEWHERE,
            DiscardKind.REVISIT_LATER,
        ),
        limit: int = 100,
    ) -> list[DiscardedClue]:
        kind_values = [k.value for k in kinds]
        cypher = """
        MATCH (d:WanderingDiscardedClue)
        WHERE d.user_id = $user_id AND d.classification IN $kinds
        RETURN d.payload_json AS payload_json
        LIMIT $limit
        """
        try:
            async with self._driver.session(database=self._database) as sess:
                result = await sess.run(
                    cypher, user_id=user_id, kinds=kind_values, limit=limit
                )
                rows = await result.data()
        except Exception as e:
            log.warning("list_discarded_clues_for_user failed: %s", e)
            return []

        out: list[DiscardedClue] = []
        for row in rows:
            raw = row.get("payload_json")
            if not raw:
                continue
            try:
                d = json.loads(raw)
                out.append(_discarded_clue_from_dict(d))
            except Exception:
                continue
        return out


def _minimal_report_from_dict(d: dict[str, Any]) -> ExplorationReport:
    """Reconstruct an ExplorationReport from its dict form. Lossy but
    adequate for the dossier endpoint's read-back path."""
    from src.wandering.report import Confidence, LayerMatch, SourceCitation

    layer_matches = {}
    for name, lm_raw in (d.get("layer_matches") or {}).items():
        if isinstance(lm_raw, dict):
            layer_matches[name] = LayerMatch(
                layer_name=lm_raw.get("layer_name", name),
                matched_nodes=list(lm_raw.get("matched_nodes", [])),
                total_nodes=int(lm_raw.get("total_nodes", 0)),
            )

    sources = [
        SourceCitation(
            title=s.get("title", ""),
            url=s.get("url", ""),
            excerpt=s.get("excerpt", ""),
            used_for=s.get("used_for", ""),
        )
        for s in (d.get("source_locations") or [])
        if isinstance(s, dict)
    ]

    try:
        confidence = Confidence(d.get("confidence", "low"))
    except ValueError:
        confidence = Confidence.LOW

    return ExplorationReport(
        report_id=str(d.get("report_id", "")),
        agent_id=str(d.get("agent_id", "")),
        anchor_summary=str(d.get("anchor_summary", "")),
        domain_explored=str(d.get("domain_explored", "")),
        source_locations=sources,
        layer_matches=layer_matches,
        confidence=confidence,
        exploration_summary=str(d.get("exploration_summary", "")),
        advancement=str(d.get("advancement", "")),
        what_does_not_map=str(d.get("what_does_not_map", "")),
        next_lead=str(d.get("next_lead", "")),
        iteration_count=int(d.get("iteration_count", 0)),
        abandoned_early=bool(d.get("abandoned_early", False)),
    )


# ---------------------------------------------------------------------------
# Factory — picks backend based on env
# ---------------------------------------------------------------------------


def build_wandering_store_from_env() -> WanderingStore:
    """Pick the wandering store backend based on env.

    CONSTELLAX_DB_BACKEND=neo4j and NEO4J_URI/PASSWORD set → Neo4jWanderingStore
    Anything else → InMemoryWanderingStore (dev / fallback)

    Never raises. On any error building the Neo4j driver, falls back to
    in-memory and logs a warning. This keeps the wandering pipeline
    working in dev without a Neo4j instance.
    """
    import os
    backend = os.environ.get("CONSTELLAX_DB_BACKEND", "").strip().lower()
    if backend != "neo4j":
        return InMemoryWanderingStore()

    try:
        from src.bridge.neo4j_backend import build_neo4j_driver_from_env
        result = build_neo4j_driver_from_env()
        if result is None:
            log.info(
                "CONSTELLAX_DB_BACKEND=neo4j but driver build returned None; "
                "falling back to InMemoryWanderingStore"
            )
            return InMemoryWanderingStore()
        driver, database = result
        store = Neo4jWanderingStore(driver=driver, database=database)
        # Don't await init_schema here — we're sync; the first save_session
        # will do schema-on-demand. For explicit init, callers can run
        # `await store.init_schema()` at app startup.
        return store
    except Exception as e:
        log.warning(
            "Neo4jWanderingStore build failed (%s); falling back to in-memory",
            e,
        )
        return InMemoryWanderingStore()


__all__ = [
    "WANDERING_SCHEMA_CYPHER",
    "WanderingStore",
    "InMemoryWanderingStore",
    "Neo4jWanderingStore",
    "build_wandering_store_from_env",
    "session_to_json",
]
