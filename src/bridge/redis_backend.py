"""
Bridge storage backends — Protocols + in-memory implementations.

Two interfaces, one implementation each (the in-memory one). Neo4j-backed
implementations live in src/bridge/neo4j_backend.py and satisfy the same
Protocols (drop-in compatible).

    AnchorBackend       — DecisionAnchor CRUD (project-scoped)
        InMemoryAnchorBackend       (dev / tests — process-local dict)

    ConversationBackend — Sessions / Iterations / TurningPoints / DecisionLinks
        InMemoryConversationBackend (dev / tests — process-local dict)

HISTORICAL NAME
    This module is called `redis_backend.py` for historical reasons — it
    once held RedisAnchorBackend / RedisConversationBackend. Those were
    removed in Phase 6 of the Neo4j migration; the file kept its name to
    avoid churning every import site (BridgeClient, MemoryAdapter,
    ConversationStore, parity scripts). Neo4j is now the sole persistent
    backend; in-memory remains for tests and stub mode.

KEY NAMESPACE — In-memory shape
    self._store: dict[project_id -> dict[entity_id -> entity_obj]]
    Project scoping is enforced at the outer key; a query for project A
    cannot see data from project B because they live in disjoint buckets.

CONSISTENCY MODEL
    Single-process MemoryAdapter/ConversationStore with in-memory backend:
    immediate read-your-write. For cross-process / persistent storage,
    Neo4jAnchorBackend / Neo4jConversationBackend (src/bridge/neo4j_backend.py)
    is the production path.

NO HIDDEN MAGIC
    Backends do not enforce business rules (TTL semantics, link-type
    validation, iteration_count bookkeeping). Those stay in the
    Store classes. Backends are pure storage.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from src.bridge.types import DecisionAnchor


# Bridge ConversationBackend entity types — must match the labels Neo4j
# uses in src/bridge/neo4j_backend.py. Single source of truth here.
_ENTITY_TYPES = ("sessions", "iterations", "turning_points", "decision_links")


# ---------------------------------------------------------------------------
# AnchorBackend — DecisionAnchor storage
# ---------------------------------------------------------------------------


class AnchorBackend(Protocol):
    """Project-scoped storage for DecisionAnchors. Implementations must be
    safe to share across MemoryAdapter instances within a process."""

    async def get(
        self, project_id: str, decision_id: str
    ) -> DecisionAnchor | None: ...

    async def put(self, project_id: str, decision: DecisionAnchor) -> None: ...

    async def delete(self, project_id: str, decision_id: str) -> bool: ...

    async def list_for_project(
        self, project_id: str
    ) -> list[DecisionAnchor]: ...

    async def update_status(
        self, project_id: str, decision_id: str, status: str
    ) -> bool: ...

    async def known_projects(self) -> list[str]: ...


class InMemoryAnchorBackend:
    """Default backend. Thin dict wrapper — keeps existing semantics intact.

    Accepts an optional shared store dict so several MemoryAdapter
    instances can share a single in-process namespace (useful for the
    FastAPI server where each request constructs its own adapter).
    """

    def __init__(
        self,
        store: dict[str, dict[str, DecisionAnchor]] | None = None,
    ):
        self._store: dict[str, dict[str, DecisionAnchor]] = (
            store if store is not None else {}
        )

    @property
    def raw(self) -> dict[str, dict[str, DecisionAnchor]]:
        """Direct access to the underlying dict (used by save/load JSON path)."""
        return self._store

    def _bucket(self, project_id: str) -> dict[str, DecisionAnchor]:
        if project_id not in self._store:
            self._store[project_id] = {}
        return self._store[project_id]

    async def get(self, project_id, decision_id):
        return self._store.get(project_id, {}).get(decision_id)

    async def put(self, project_id, decision):
        if not decision.id:
            raise ValueError("DecisionAnchor.id must be non-empty before storing")
        self._bucket(project_id)[decision.id] = decision

    async def delete(self, project_id, decision_id):
        bucket = self._store.get(project_id)
        if not bucket or decision_id not in bucket:
            return False
        del bucket[decision_id]
        return True

    async def list_for_project(self, project_id):
        return list(self._store.get(project_id, {}).values())

    async def update_status(self, project_id, decision_id, status):
        bucket = self._store.get(project_id)
        if not bucket or decision_id not in bucket:
            return False
        bucket[decision_id] = replace(bucket[decision_id], status=status)
        return True

    async def known_projects(self):
        return list(self._store.keys())


# ---------------------------------------------------------------------------
# ConversationBackend — Session/Iteration/TurningPoint/DecisionLink storage
# ---------------------------------------------------------------------------


class ConversationBackend(Protocol):
    """Project-scoped storage for the four conversation entity types."""

    async def get(
        self, project_id: str, entity_type: str, entity_id: str
    ) -> Any | None: ...

    async def put(
        self, project_id: str, entity_type: str, entity: Any
    ) -> None: ...

    async def delete(
        self, project_id: str, entity_type: str, entity_id: str
    ) -> bool: ...

    async def list_for_project(
        self, project_id: str, entity_type: str
    ) -> list[Any]: ...

    async def known_projects(self) -> list[str]: ...


class InMemoryConversationBackend:
    """Default backend. Mirrors the legacy ConversationStore dict layout.

    Shape: { project_id: { entity_type: { entity_id: entity_obj } } }
    """

    def __init__(self, store: dict[str, dict[str, dict[str, Any]]] | None = None):
        self._store: dict[str, dict[str, dict[str, Any]]] = (
            store if store is not None else {}
        )

    @property
    def raw(self) -> dict[str, dict[str, dict[str, Any]]]:
        return self._store

    def _project_buckets(self, project_id: str) -> dict[str, dict[str, Any]]:
        if project_id not in self._store:
            self._store[project_id] = {et: {} for et in _ENTITY_TYPES}
        else:
            for et in _ENTITY_TYPES:
                self._store[project_id].setdefault(et, {})
        return self._store[project_id]

    def _bucket(self, project_id: str, entity_type: str) -> dict[str, Any]:
        if entity_type not in _ENTITY_TYPES:
            raise ValueError(
                f"unknown entity_type {entity_type!r}; expected one of {_ENTITY_TYPES}"
            )
        return self._project_buckets(project_id)[entity_type]

    async def get(self, project_id, entity_type, entity_id):
        return self._bucket(project_id, entity_type).get(entity_id)

    async def put(self, project_id, entity_type, entity):
        eid = getattr(entity, "id", None)
        if not eid:
            raise ValueError(f"{entity_type} entity must have non-empty id")
        self._bucket(project_id, entity_type)[eid] = entity

    async def delete(self, project_id, entity_type, entity_id):
        bucket = self._bucket(project_id, entity_type)
        if entity_id not in bucket:
            return False
        del bucket[entity_id]
        return True

    async def list_for_project(self, project_id, entity_type):
        return list(self._bucket(project_id, entity_type).values())

    async def known_projects(self):
        return list(self._store.keys())
