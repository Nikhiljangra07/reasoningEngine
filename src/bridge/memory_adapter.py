"""
Memory V2 adapter — PLUGGABLE STORAGE BACKEND with optional disk persistence.

Memory V2 was originally a TypeScript system at
/Users/nikhil/Desktop/lora-v1-frontend/src/emotion-core/memory-v2/ for
life-decision context. This file is the Python port adapted for
code-decision context. Storage is now backend-pluggable:

    backend=None (default)        → InMemoryAnchorBackend (process-local dict)
    backend=RedisAnchorBackend(c) → production Redis-backed storage

Existing tests construct adapters with the default backend — semantics
preserved. Production callers (server.py) build a Redis backend from
CONSTELLAX_REDIS_URL and pass it in.

Public API (unchanged):
    - store_decision()              persist a DecisionAnchor
    - get_decision()                fetch by id
    - get_decisions_touching_file() filter by code_ref file_path
    - get_code_refs_for_decision()  the code_refs on a stored decision
    - update_decision_status()      mutate status (OPEN → SETTLED → DRIFTED ...)
    - find_similar_decisions()      delegates to a pluggable SimilarityScorer
                                    (default = KeywordJaccardScorer)
    - save() / load()               JSON disk persistence (opt-in via storage_path,
                                    only valid with InMemoryAnchorBackend)

PROJECT SCOPING — the catastrophic-blending defense:
Every adapter instance is scoped by a `project_id`. Decisions stored under
project A are physically partitioned from decisions queryable under project
B. Two repos can NEVER blend memory — the backend partitions by project_id
at the storage layer (dict keys for in-memory, Redis key prefix for Redis).
When `project_id` is None (back-compat), a single shared "unscoped" bucket
is used — fine for tests, never recommended for production.

PERSISTENCE — opt-in via `storage_path` (in-memory backend only):
When provided AND backend is InMemoryAnchorBackend, every store_decision()
and update_decision_status() call auto-saves to disk via atomic write
(.tmp + os.replace). Construction loads existing state from disk;
missing/corrupted files cause silent fresh start. Schema drift in stored
entries is tolerated — unknown fields cause that one entry to be skipped,
the rest load normally.

For RedisAnchorBackend, `storage_path` is ignored (Redis IS the persistence).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from src.bridge.redis_backend import AnchorBackend, InMemoryAnchorBackend
from src.bridge.similarity import KeywordJaccardScorer, SimilarityScorer
from src.bridge.types import CodeRef, DecisionAnchor


# Sentinel for the "no project_id supplied" bucket. Used only when callers
# construct MemoryAdapter without a project_id (back-compat). Production
# code should always pass a project_id from compute_fingerprint().
_UNSCOPED = "__unscoped__"


class MemoryAdapter:
    """
    Project-scoped decision store backed by a pluggable AnchorBackend.

    The backend handles raw storage. This class layers project scoping,
    similarity search, code-ref filtering, and (in-memory backend only)
    JSON file persistence on top.

    Pass `store=` to share a dict across InMemoryAnchorBackend instances
    in the same process (the legacy back-compat path). Pass `backend=`
    for explicit backend selection (the new path).
    """

    def __init__(
        self,
        repo_root: str,
        project_id: str | None = None,
        store: dict[str, dict[str, DecisionAnchor]] | None = None,
        scorer: SimilarityScorer | None = None,
        storage_path: str | None = None,
        backend: AnchorBackend | None = None,
    ):
        self.repo_root = repo_root
        self.project_id = project_id

        # Backend resolution:
        #   - explicit `backend` wins
        #   - else InMemoryAnchorBackend wrapping `store` (legacy path)
        if backend is not None:
            if store is not None:
                raise ValueError(
                    "pass either `store=` (in-memory shared dict) or "
                    "`backend=` (explicit backend), not both"
                )
            self._backend: AnchorBackend = backend
        else:
            self._backend = InMemoryAnchorBackend(store=store)

        # Pluggable similarity backend. Defaults to keyword Jaccard. Implement
        # the SimilarityScorer Protocol (src/bridge/similarity.py) and pass
        # `scorer=YourScorer()` to swap in a real embedding model later.
        self._scorer: SimilarityScorer = (
            scorer if scorer is not None else KeywordJaccardScorer()
        )

        # Optional disk persistence — only meaningful with the in-memory
        # backend, since Redis already persists. Silently ignored otherwise.
        self._storage_path = storage_path
        if storage_path and isinstance(self._backend, InMemoryAnchorBackend):
            self.load()

    # Back-compat property: legacy callers read `.store_dict()` or `._store`
    # directly to share state across adapters. With the new backend layer,
    # only the InMemoryAnchorBackend exposes a raw dict; Redis returns None.
    @property
    def _store(self) -> dict[str, dict[str, DecisionAnchor]] | None:
        """Legacy access to the underlying dict (in-memory backend only)."""
        if isinstance(self._backend, InMemoryAnchorBackend):
            return self._backend.raw
        return None

    # -----------------------------------------------------------------------
    # Internal: project key
    # -----------------------------------------------------------------------

    def _scope_key(self) -> str:
        return self.project_id if self.project_id else _UNSCOPED

    # -----------------------------------------------------------------------
    # Reads
    # -----------------------------------------------------------------------

    async def get_decision(self, decision_id: str) -> DecisionAnchor | None:
        """Fetch a single decision by id. None if not found in this project."""
        return await self._backend.get(self._scope_key(), decision_id)

    async def get_decisions_touching_file(
        self, file_path: str
    ) -> list[DecisionAnchor]:
        """
        Return every decision in the current project whose code_refs include `file_path`.

        Project-scoped — decisions stored under a different project_id are
        invisible. Backend-agnostic implementation (filters in-memory after
        listing) keeps the backend interface narrow.
        """
        matches: list[DecisionAnchor] = []
        all_decisions = await self._backend.list_for_project(self._scope_key())
        for decision in all_decisions:
            for ref in decision.code_refs:
                if ref.file_path == file_path:
                    matches.append(decision)
                    break
        return matches

    async def find_similar_decisions(
        self, context_text: str, k: int = 5
    ) -> list[DecisionAnchor]:
        """
        Return the top-k decisions most similar to `context_text`.

        Scoring is delegated to `self._scorer`. Returns decisions with
        score > 0 only, sorted descending.
        """
        if not context_text:
            return []

        all_decisions = await self._backend.list_for_project(self._scope_key())
        scored: list[tuple[float, DecisionAnchor]] = []
        for decision in all_decisions:
            document = " ".join([
                decision.title or "",
                decision.rationale or "",
                " ".join(decision.tags or []),
            ])
            score = await self._scorer.score(context_text, document)
            if score > 0:
                scored.append((score, decision))

        scored.sort(reverse=True, key=lambda t: t[0])
        return [d for _, d in scored[: max(0, k)]]

    async def get_code_refs_for_decision(
        self, decision_id: str
    ) -> list[CodeRef]:
        """
        Return the code_refs of a stored decision, or [] if it doesn't exist.
        """
        decision = await self.get_decision(decision_id)
        if decision is None:
            return []
        return list(decision.code_refs)

    # -----------------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------------

    async def store_decision(self, decision: DecisionAnchor) -> str:
        """
        Persist a decision into the current project's bucket. Returns the
        decision id. Existing decisions with the same id are overwritten
        (idempotent upsert).

        Auto-saves to disk if storage_path was set on construction AND
        the backend is InMemoryAnchorBackend.
        """
        if not decision.id:
            raise ValueError("DecisionAnchor.id must be non-empty before storing")
        await self._backend.put(self._scope_key(), decision)
        self._autosave()
        return decision.id

    async def update_decision_status(
        self, decision_id: str, status: str
    ) -> None:
        """
        Mutate the status field of an existing decision. Raises KeyError
        if the id doesn't exist in the current project's bucket.

        Auto-saves to disk if storage_path was set on construction AND
        the backend is InMemoryAnchorBackend.
        """
        ok = await self._backend.update_status(self._scope_key(), decision_id, status)
        if not ok:
            raise KeyError(
                f"decision {decision_id!r} not found in project "
                f"{self.project_id or '(unscoped)'!r}"
            )
        self._autosave()

    # -----------------------------------------------------------------------
    # Persistence (opt-in via storage_path, in-memory backend only)
    # -----------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """
        Atomically write the in-memory store to disk as JSON. No-op if the
        backend isn't InMemoryAnchorBackend (Redis IS the persistence).

        Writes go through a `.tmp` file + `os.replace` so a crash mid-write
        can't corrupt the file.
        """
        target = path or self._storage_path
        if not target:
            return
        if not isinstance(self._backend, InMemoryAnchorBackend):
            return  # Redis backend handles persistence itself

        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = {
            "schema_version": 1,
            "projects": {
                project_id: {
                    decision_id: asdict(decision)
                    for decision_id, decision in bucket.items()
                }
                for project_id, bucket in self._backend.raw.items()
            },
        }
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, target)

    def load(self, path: str | None = None) -> None:
        """
        Replace the in-memory store with state read from disk. No-op if the
        backend isn't InMemoryAnchorBackend.

        Silent no-op on:
            - missing file (first install)
            - corrupted JSON (don't crash; the engine still boots)
            - schema drift on individual decisions (skip the bad entry,
              keep loading the rest)
        """
        target = path or self._storage_path
        if not target or not os.path.exists(target):
            return
        if not isinstance(self._backend, InMemoryAnchorBackend):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return

        loaded: dict[str, dict[str, DecisionAnchor]] = {}
        for project_id, bucket_raw in (data.get("projects") or {}).items():
            if not isinstance(bucket_raw, dict):
                continue
            new_bucket: dict[str, DecisionAnchor] = {}
            for decision_id, raw in bucket_raw.items():
                if not isinstance(raw, dict):
                    continue
                try:
                    code_refs_raw = raw.get("code_refs", []) or []
                    code_refs = [
                        CodeRef(**ref) for ref in code_refs_raw
                        if isinstance(ref, dict)
                    ]
                    kwargs = {k: v for k, v in raw.items() if k != "code_refs"}
                    anchor = DecisionAnchor(code_refs=code_refs, **kwargs)
                    new_bucket[decision_id] = anchor
                except TypeError:
                    continue
            loaded[project_id] = new_bucket
        # Replace the backend's underlying dict in place so any shared
        # reference still works.
        self._backend.raw.clear()
        self._backend.raw.update(loaded)

    def _autosave(self) -> None:
        """Called after every mutation; no-op when persistence is disabled
        or the backend isn't in-memory."""
        if self._storage_path and isinstance(self._backend, InMemoryAnchorBackend):
            self.save()
