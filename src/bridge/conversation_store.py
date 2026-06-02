"""
ConversationStore — the structured-storage spine for sessions, iterations,
turning points, and decision links.

Storage is backend-pluggable (matching MemoryAdapter):

    backend=None (default)              → InMemoryConversationBackend
    backend=RedisConversationBackend(c) → production Redis-backed storage

Design intent (locked):
    - Store EVERYTHING from a conversation in structured form. No
      consolidation step, no privacy-first framing — the codebase is
      already being indexed, so memory pretending to be opt-in would
      be hypocrisy. We store, we structure, we make it visible.
    - Sophisticated presentation: tree/graph views the UI can render
      as actual trees, boxes, and lineage diagrams.
    - 30-day TTL by default. Users pin to keep forever.
    - Project-scoped so two projects NEVER blend (same defense as
      MemoryAdapter).

Storage shape (in-memory backend):
    {
      project_id: {
        "sessions":         { id: Session,        ... },
        "iterations":       { id: Iteration,      ... },
        "turning_points":   { id: TurningPoint,   ... },
        "decision_links":   { id: DecisionLink,   ... },
      },
      ...
    }

Persistence:
    - InMemoryConversationBackend + storage_path → JSON file (atomic write,
      corruption tolerance).
    - RedisConversationBackend → keys live in Redis directly; storage_path
      is ignored.

ISOLATION: imports stdlib + src.bridge.types + src.bridge.redis_backend only.
No engine, no LLM, no MemoryAdapter (decision IDs are referenced by string;
resolution happens at a higher orchestration layer).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, replace

from src.bridge.redis_backend import (
    ConversationBackend,
    InMemoryConversationBackend,
)
from src.bridge.types import (
    DecisionLink,
    Iteration,
    Session,
    TurningPoint,
)


# ---------------------------------------------------------------------------
# ExpiryAlert — surfaces "this is about to be deleted, want to pin it?" to UI
# ---------------------------------------------------------------------------

# Alert tier thresholds in DAYS. The user-spec'd cadence:
# 15 days remaining → first warning; 7 days → second; 3 days → final; 0 = expired.
_ALERT_DAYS_FIRST = 15
_ALERT_DAYS_SECOND = 7
_ALERT_DAYS_FINAL = 3
_SECONDS_PER_DAY = 86400.0


@dataclass
class ExpiryAlert:
    """
    A notification that an entity is approaching auto-deletion.

    Surfaced to the UI so the user can decide whether to pin (keep forever),
    accept (let it auto-delete on schedule), or — eventually — delete now.

    Tiers (the user's locked spec):
        "15_days"  — between 15 and 7 days remaining (FIRST warning, kicks in
                     at the halfway point of the default 30-day TTL)
        "7_days"   — between 7 and 3 days remaining (SECOND warning)
        "3_days"   — between 3 and 0 days remaining (FINAL warning)
        "expired"  — already past expires_at but not yet swept; user can still
                     rescue with a pin if sweep_expired() hasn't run yet
    """
    entity_type: str               # "sessions" | "iterations" | "turning_points" | "decision_links"
    entity_id: str
    title: str                     # human-readable identifier (Session.title, etc.)
    days_remaining: float          # negative when expired
    tier: str                      # one of the four tier strings above
    expires_at: float              # unix timestamp
    user_options: list[dict]       # [{"id": "pin", "label": "Keep forever"}, ...]


def _alert_tier(days_remaining: float) -> str | None:
    """Map remaining-days to an alert tier, or None if no alert yet."""
    if days_remaining <= 0:
        return "expired"
    if days_remaining <= _ALERT_DAYS_FINAL:
        return "3_days"
    if days_remaining <= _ALERT_DAYS_SECOND:
        return "7_days"
    if days_remaining <= _ALERT_DAYS_FIRST:
        return "15_days"
    return None


def _derive_alert_title(entity_type: str, obj) -> str:
    """Per-type human-friendly identifier for the alert card."""
    if entity_type == "sessions":
        return obj.title or f"Session {obj.id}"
    if entity_type == "iterations":
        text = (obj.user_text or "").strip().replace("\n", " ")
        snippet = text[:60] + ("…" if len(text) > 60 else "")
        return f"Turn {obj.sequence_num}: {snippet}" if snippet else f"Turn {obj.sequence_num}"
    if entity_type == "turning_points":
        return f"Turning point: {obj.title}"
    if entity_type == "decision_links":
        return f"{obj.from_decision_id} →[{obj.link_type}]→ {obj.to_decision_id}"
    return getattr(obj, "id", "(unknown)")


# Default time-to-live for any new conversation entity. 30 days = 2_592_000s.
_DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60

# Sentinel for the "no project_id supplied" bucket.
_UNSCOPED = "__unscoped__"

# Entity-type names. Single source of truth.
_BUCKETS = ("sessions", "iterations", "turning_points", "decision_links")

# Acceptable link types — store-enforced.
_VALID_LINK_TYPES = frozenset({
    "leads_to", "supersedes", "depends_on", "contradicts", "informed_by",
})


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


class ConversationStore:
    """
    Project-scoped storage for the conversation hierarchy + lineage graph.

    Every CRUD method is async. Reads filter out expired entries by default;
    pass `include_expired=True` to bypass for admin/debug.

    Pinning: pass `expires_at=None` to any create method to skip TTL.
    Or call `.pin(entity_type, id)` after the fact.
    """

    DEFAULT_TTL_SECONDS = _DEFAULT_TTL_SECONDS

    def __init__(
        self,
        project_id: str | None = None,
        storage_path: str | None = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        store: dict | None = None,
        backend: ConversationBackend | None = None,
    ):
        self.project_id = project_id
        self.ttl_seconds = ttl_seconds

        # Backend resolution:
        #   - explicit `backend` wins
        #   - else InMemoryConversationBackend wrapping `store` (legacy)
        if backend is not None:
            if store is not None:
                raise ValueError(
                    "pass either `store=` (in-memory shared dict) or "
                    "`backend=` (explicit backend), not both"
                )
            self._backend: ConversationBackend = backend
        else:
            self._backend = InMemoryConversationBackend(store=store)

        self._storage_path = storage_path
        if storage_path and isinstance(self._backend, InMemoryConversationBackend):
            self.load()

        # Identity-layer state: position-restatement counter, keyed by
        # (session_id, position_hash). Incremented on every
        # `add_iteration`. The counter observes pushback patterns
        # without yet modifying dispatcher behavior — once the counter
        # has reliable data, a follow-up sprint can wire the
        # cartography directive into the dispatcher's system prompt.
        # In-process only; resets on store rebuild.
        from src.identity import MapNotMarchCounter
        self._map_not_march = MapNotMarchCounter()

    # -----------------------------------------------------------------------
    # Internal: project key + TTL helpers
    # -----------------------------------------------------------------------

    def _scope_key(self) -> str:
        return self.project_id if self.project_id else _UNSCOPED

    def _default_expires_at(self) -> float | None:
        """Default expiration timestamp for newly-created entities."""
        return _now() + self.ttl_seconds if self.ttl_seconds > 0 else None

    @staticmethod
    def _is_expired(obj, now: float) -> bool:
        """An entity is expired iff expires_at is a float AND in the past."""
        exp = getattr(obj, "expires_at", None)
        if exp is None:
            return False  # pinned forever
        return exp < now

    # -----------------------------------------------------------------------
    # Sessions
    # -----------------------------------------------------------------------

    async def start_session(
        self,
        title: str = "",
        expires_at: float | None = ...,  # type: ignore[assignment]
    ) -> Session:
        sid = _new_id("S")
        sess = Session(
            id=sid,
            project_id=self.project_id,
            title=title or f"Session {sid}",
            started_at=_now(),
            expires_at=(self._default_expires_at()
                        if expires_at is ...
                        else expires_at),
        )
        await self._backend.put(self._scope_key(), "sessions", sess)
        self._autosave()
        return sess

    async def end_session(self, session_id: str) -> Session | None:
        sess = await self._backend.get(self._scope_key(), "sessions", session_id)
        if sess is None:
            return None
        sess.ended_at = _now()
        sess.status = "ended"
        await self._backend.put(self._scope_key(), "sessions", sess)
        self._autosave()
        return sess

    async def get_session(
        self, session_id: str, include_expired: bool = False
    ) -> Session | None:
        sess = await self._backend.get(self._scope_key(), "sessions", session_id)
        if sess is None:
            return None
        if not include_expired and self._is_expired(sess, _now()):
            return None
        return sess

    async def list_sessions(
        self, include_expired: bool = False
    ) -> list[Session]:
        now = _now()
        all_sessions = await self._backend.list_for_project(
            self._scope_key(), "sessions"
        )
        out = [
            s for s in all_sessions
            if include_expired or not self._is_expired(s, now)
        ]
        out.sort(key=lambda s: s.started_at, reverse=True)
        return out

    # -----------------------------------------------------------------------
    # Iterations
    # -----------------------------------------------------------------------

    async def add_iteration(
        self,
        session_id: str,
        user_text: str,
        engine_response: str,
        route: str = "",
        effort: str = "",
        parent_iteration_id: str | None = None,
        expires_at: float | None = ...,  # type: ignore[assignment]
    ) -> Iteration:
        sess = await self._backend.get(self._scope_key(), "sessions", session_id)
        if sess is None:
            raise KeyError(f"session {session_id!r} not found")

        seq = sess.iteration_count + 1
        iter_id = _new_id("I")
        it = Iteration(
            id=iter_id,
            session_id=session_id,
            sequence_num=seq,
            user_text=user_text,
            engine_response=engine_response,
            created_at=_now(),
            route=route,
            effort=effort,
            parent_iteration_id=parent_iteration_id,
            expires_at=(self._default_expires_at()
                        if expires_at is ...
                        else expires_at),
        )
        await self._backend.put(self._scope_key(), "iterations", it)
        sess.iteration_count = seq
        await self._backend.put(self._scope_key(), "sessions", sess)
        self._autosave()

        # Identity-layer observation: record the user's position. The
        # counter normalizes the text (strip filler, lowercase, hash)
        # and increments the count for (session_id, position_hash). It
        # does NOT modify the engine_response or alter dispatch. A
        # follow-up sprint will read `map_not_march_strike(session_id,
        # user_text)` from the dispatcher to inject a cartography
        # directive when the user has restated past threshold.
        if user_text and user_text.strip():
            try:
                self._map_not_march.note(session_id, user_text)
            except Exception:  # pragma: no cover — counter is defensive
                pass

        return it

    async def get_iteration(
        self, iteration_id: str, include_expired: bool = False
    ) -> Iteration | None:
        it = await self._backend.get(self._scope_key(), "iterations", iteration_id)
        if it is None:
            return None
        if not include_expired and self._is_expired(it, _now()):
            return None
        return it

    async def iterations_for_session(
        self, session_id: str, include_expired: bool = False
    ) -> list[Iteration]:
        now = _now()
        all_iters = await self._backend.list_for_project(
            self._scope_key(), "iterations"
        )
        out = [
            it for it in all_iters
            if it.session_id == session_id
            and (include_expired or not self._is_expired(it, now))
        ]
        out.sort(key=lambda i: i.sequence_num)
        return out

    # -----------------------------------------------------------------------
    # Identity-layer accessors
    # -----------------------------------------------------------------------

    def map_not_march_strike(self, session_id: str, user_text: str) -> int:
        """Return how many times `user_text` (normalized) has been seen
        in this session. Read-only — does not increment the counter.

        Intended for a follow-up dispatcher integration: when the
        return value crosses the `MAP_NOT_MARCH_THRESHOLD`, the
        dispatcher injects a cartography directive into the system
        prompt so the model switches from arguing to mapping paths.
        Until that wiring lands, this accessor is exercised only by
        tests."""
        if not user_text or not user_text.strip():
            return 0
        return self._map_not_march.current(session_id, user_text)

    # -----------------------------------------------------------------------
    # Attaching DecisionAnchors (by id) to iterations
    # -----------------------------------------------------------------------

    async def attach_decision(
        self, iteration_id: str, decision_id: str
    ) -> None:
        it = await self._backend.get(self._scope_key(), "iterations", iteration_id)
        if it is None:
            raise KeyError(f"iteration {iteration_id!r} not found")
        if decision_id not in it.decision_ids:
            it.decision_ids.append(decision_id)
            await self._backend.put(self._scope_key(), "iterations", it)
            sess = await self._backend.get(
                self._scope_key(), "sessions", it.session_id
            )
            if sess is not None:
                sess.decision_count = sess.decision_count + 1
                await self._backend.put(self._scope_key(), "sessions", sess)
            self._autosave()

    # -----------------------------------------------------------------------
    # Turning points
    # -----------------------------------------------------------------------

    async def record_turning_point(
        self,
        iteration_id: str,
        title: str,
        description: str = "",
        triggered_by: list[str] | None = None,
        led_to: list[str] | None = None,
        expires_at: float | None = ...,  # type: ignore[assignment]
    ) -> TurningPoint:
        it = await self._backend.get(self._scope_key(), "iterations", iteration_id)
        if it is None:
            raise KeyError(f"iteration {iteration_id!r} not found")
        tp_id = _new_id("T")
        tp = TurningPoint(
            id=tp_id,
            session_id=it.session_id,
            iteration_id=iteration_id,
            title=title,
            description=description,
            triggered_by_decisions=list(triggered_by or []),
            led_to_decisions=list(led_to or []),
            created_at=_now(),
            expires_at=(self._default_expires_at()
                        if expires_at is ...
                        else expires_at),
        )
        await self._backend.put(self._scope_key(), "turning_points", tp)
        it.turning_point_ids.append(tp_id)
        await self._backend.put(self._scope_key(), "iterations", it)
        sess = await self._backend.get(
            self._scope_key(), "sessions", it.session_id
        )
        if sess is not None:
            sess.turning_point_count = sess.turning_point_count + 1
            await self._backend.put(self._scope_key(), "sessions", sess)
        self._autosave()
        return tp

    async def get_turning_point(
        self, tp_id: str, include_expired: bool = False
    ) -> TurningPoint | None:
        tp = await self._backend.get(self._scope_key(), "turning_points", tp_id)
        if tp is None:
            return None
        if not include_expired and self._is_expired(tp, _now()):
            return None
        return tp

    async def turning_points_for_session(
        self, session_id: str, include_expired: bool = False
    ) -> list[TurningPoint]:
        now = _now()
        all_tps = await self._backend.list_for_project(
            self._scope_key(), "turning_points"
        )
        out = [
            tp for tp in all_tps
            if tp.session_id == session_id
            and (include_expired or not self._is_expired(tp, now))
        ]
        out.sort(key=lambda t: t.created_at)
        return out

    # -----------------------------------------------------------------------
    # DecisionLinks (the lineage graph)
    # -----------------------------------------------------------------------

    async def link_decisions(
        self,
        from_decision_id: str,
        to_decision_id: str,
        link_type: str,
        rationale: str = "",
        expires_at: float | None = ...,  # type: ignore[assignment]
    ) -> DecisionLink:
        if from_decision_id == to_decision_id:
            raise ValueError("cannot link a decision to itself")
        if link_type not in _VALID_LINK_TYPES:
            raise ValueError(
                f"link_type must be one of {sorted(_VALID_LINK_TYPES)}, "
                f"got {link_type!r}"
            )
        link_id = _new_id("L")
        link = DecisionLink(
            id=link_id,
            project_id=self.project_id,
            from_decision_id=from_decision_id,
            to_decision_id=to_decision_id,
            link_type=link_type,
            rationale=rationale,
            created_at=_now(),
            expires_at=(self._default_expires_at()
                        if expires_at is ...
                        else expires_at),
        )
        await self._backend.put(self._scope_key(), "decision_links", link)
        self._autosave()
        return link

    async def decisions_linked_from(
        self, decision_id: str, include_expired: bool = False
    ) -> list[DecisionLink]:
        now = _now()
        all_links = await self._backend.list_for_project(
            self._scope_key(), "decision_links"
        )
        return [
            link for link in all_links
            if link.from_decision_id == decision_id
            and (include_expired or not self._is_expired(link, now))
        ]

    async def decisions_linked_to(
        self, decision_id: str, include_expired: bool = False
    ) -> list[DecisionLink]:
        now = _now()
        all_links = await self._backend.list_for_project(
            self._scope_key(), "decision_links"
        )
        return [
            link for link in all_links
            if link.to_decision_id == decision_id
            and (include_expired or not self._is_expired(link, now))
        ]

    # -----------------------------------------------------------------------
    # Pinning (opt out of TTL)
    # -----------------------------------------------------------------------

    async def pin(self, entity_type: str, entity_id: str) -> bool:
        obj = await self._backend.get(self._scope_key(), entity_type, entity_id)
        if obj is None:
            return False
        updated = replace(obj, expires_at=None)
        await self._backend.put(self._scope_key(), entity_type, updated)
        self._autosave()
        return True

    async def unpin(self, entity_type: str, entity_id: str) -> bool:
        obj = await self._backend.get(self._scope_key(), entity_type, entity_id)
        if obj is None:
            return False
        updated = replace(obj, expires_at=self._default_expires_at())
        await self._backend.put(self._scope_key(), entity_type, updated)
        self._autosave()
        return True

    # -----------------------------------------------------------------------
    # TTL sweep — actually delete expired entries from storage
    # -----------------------------------------------------------------------

    async def sweep_expired(self, now: float | None = None) -> dict[str, int]:
        """
        Walk every bucket in every project and delete entries whose
        expires_at < now. Returns a count per bucket (across all projects).
        """
        n = now if now is not None else _now()
        counts: dict[str, int] = {b: 0 for b in _BUCKETS}
        project_ids = await self._backend.known_projects()
        for pid in project_ids:
            for bucket_name in _BUCKETS:
                entries = await self._backend.list_for_project(pid, bucket_name)
                for obj in entries:
                    if self._is_expired(obj, n):
                        if await self._backend.delete(pid, bucket_name, obj.id):
                            counts[bucket_name] += 1
        if any(counts.values()):
            self._autosave()
        return counts

    # -----------------------------------------------------------------------
    # Expiry alerts — surfaces "about to be deleted, pin or let it go?" to UI
    # -----------------------------------------------------------------------

    async def get_expiry_alerts(
        self,
        now: float | None = None,
        project_only: bool = True,
    ) -> list[ExpiryAlert]:
        """
        Walk every entity in scope and emit an ExpiryAlert for any that
        falls into the 15-day / 7-day / 3-day / expired window.

        Pinned entities (expires_at is None) NEVER produce alerts.

        project_only=True (default) scopes to this adapter's project_id.
        Set False for an admin/dashboard view across all projects.
        """
        n = now if now is not None else _now()
        alerts: list[ExpiryAlert] = []

        if project_only:
            project_ids = [self._scope_key()]
        else:
            project_ids = await self._backend.known_projects()

        for pid in project_ids:
            for bucket_name in _BUCKETS:
                entries = await self._backend.list_for_project(pid, bucket_name)
                for obj in entries:
                    exp = getattr(obj, "expires_at", None)
                    if exp is None:
                        continue  # pinned — never alerts
                    days_remaining = (exp - n) / _SECONDS_PER_DAY
                    tier = _alert_tier(days_remaining)
                    if tier is None:
                        continue  # > 15 days out — no alert yet
                    alerts.append(ExpiryAlert(
                        entity_type=bucket_name,
                        entity_id=getattr(obj, "id", ""),
                        title=_derive_alert_title(bucket_name, obj),
                        days_remaining=round(days_remaining, 2),
                        tier=tier,
                        expires_at=exp,
                        user_options=[
                            {
                                "id": "pin",
                                "label": "Keep forever",
                                "action": f"pin {bucket_name} {getattr(obj, 'id', '')}",
                            },
                            {
                                "id": "let_expire",
                                "label": "Let it auto-delete",
                                "action": "noop",
                            },
                        ],
                    ))

        alerts.sort(key=lambda a: a.days_remaining)
        return alerts

    # -----------------------------------------------------------------------
    # The "fancy" views — render-ready output for the UI
    # -----------------------------------------------------------------------

    async def get_session_tree(
        self, session_id: str, include_expired: bool = False
    ) -> dict:
        """
        Return a session with its iterations + turning points nested,
        plus all decision links scoped to decisions mentioned anywhere
        in the session. Shaped for direct UI rendering.
        """
        sess = await self.get_session(session_id, include_expired=include_expired)
        if sess is None:
            return {"session": None, "iterations": [], "decision_links": []}

        iterations = await self.iterations_for_session(
            session_id, include_expired=include_expired,
        )
        # Pull turning points once for the whole session, then bucket by
        # iteration_id — avoids O(N) extra backend hits inside the loop.
        all_tps = await self.turning_points_for_session(
            session_id, include_expired=include_expired,
        )
        tps_by_iter: dict[str, list[TurningPoint]] = {}
        for tp in all_tps:
            tps_by_iter.setdefault(tp.iteration_id, []).append(tp)

        iter_blocks = []
        all_decision_ids: set[str] = set()
        for it in iterations:
            all_decision_ids.update(it.decision_ids)
            tps = tps_by_iter.get(it.id, [])
            for tp in tps:
                all_decision_ids.update(tp.triggered_by_decisions)
                all_decision_ids.update(tp.led_to_decisions)
            iter_blocks.append({
                "iteration": it,
                "turning_points": tps,
            })

        # Pull all decision links where either endpoint is in the session.
        now = _now()
        all_links = await self._backend.list_for_project(
            self._scope_key(), "decision_links"
        )
        links = [
            link for link in all_links
            if (include_expired or not self._is_expired(link, now))
            and (link.from_decision_id in all_decision_ids
                 or link.to_decision_id in all_decision_ids)
        ]

        return {
            "session": sess,
            "iterations": iter_blocks,
            "decision_links": links,
        }

    async def get_decision_lineage(
        self,
        decision_id: str,
        max_depth: int = 3,
        include_expired: bool = False,
    ) -> dict:
        """
        Walk the decision-link graph outward from one decision, in both
        directions, up to max_depth hops.
        """
        incoming = await self._walk_lineage(
            decision_id, "to", max_depth, include_expired,
        )
        outgoing = await self._walk_lineage(
            decision_id, "from", max_depth, include_expired,
        )
        return {
            "root": decision_id,
            "max_depth": max_depth,
            "incoming": incoming,
            "outgoing": outgoing,
        }

    async def _walk_lineage(
        self,
        decision_id: str,
        direction: str,
        max_depth: int,
        include_expired: bool,
    ) -> list[dict]:
        """BFS the decision-link graph in one direction."""
        out: list[dict] = []
        seen: set[str] = {decision_id}
        frontier: list[tuple[str, int]] = [(decision_id, 0)]
        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            if direction == "from":
                edges = await self.decisions_linked_from(
                    current_id, include_expired=include_expired,
                )
            else:
                edges = await self.decisions_linked_to(
                    current_id, include_expired=include_expired,
                )
            for link in edges:
                neighbor = (link.to_decision_id if direction == "from"
                            else link.from_decision_id)
                out.append({"link": link, "depth": depth + 1})
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append((neighbor, depth + 1))
        return out

    # -----------------------------------------------------------------------
    # Persistence — opt-in via storage_path, in-memory backend only
    # -----------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        target = path or self._storage_path
        if not target:
            return
        if not isinstance(self._backend, InMemoryConversationBackend):
            return  # Redis backend handles persistence itself

        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)

        data = {
            "schema_version": 1,
            "ttl_seconds": self.ttl_seconds,
            "projects": {
                pid: {
                    bucket: {oid: asdict(obj) for oid, obj in entries.items()}
                    for bucket, entries in proj_buckets.items()
                }
                for pid, proj_buckets in self._backend.raw.items()
            },
        }
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, target)

    def load(self, path: str | None = None) -> None:
        target = path or self._storage_path
        if not target or not os.path.exists(target):
            return
        if not isinstance(self._backend, InMemoryConversationBackend):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return

        rebuilders = {
            "sessions": Session,
            "iterations": Iteration,
            "turning_points": TurningPoint,
            "decision_links": DecisionLink,
        }
        loaded: dict[str, dict[str, dict]] = {}
        for pid, proj_buckets_raw in (data.get("projects") or {}).items():
            if not isinstance(proj_buckets_raw, dict):
                continue
            loaded[pid] = {b: {} for b in _BUCKETS}
            for bucket_name in _BUCKETS:
                entries_raw = proj_buckets_raw.get(bucket_name, {}) or {}
                if not isinstance(entries_raw, dict):
                    continue
                cls = rebuilders[bucket_name]
                for entity_id, raw in entries_raw.items():
                    if not isinstance(raw, dict):
                        continue
                    try:
                        loaded[pid][bucket_name][entity_id] = cls(**raw)
                    except TypeError:
                        # Schema drift on this entry — skip, keep the rest.
                        continue
        self._backend.raw.clear()
        self._backend.raw.update(loaded)

    def _autosave(self) -> None:
        if self._storage_path and isinstance(self._backend, InMemoryConversationBackend):
            self.save()
