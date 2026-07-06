"""
Wandering Room job manager — in-process durable jobs.

Wandering sessions are long-running (3 min for TRIPLE_PENDULUM up to 60 min
for ABSOLUTE_CHAOS). Holding the HTTP request open for that whole duration
is fragile — any network blip drops the client connection, and the client
never even learns the session_id because it disconnected BEFORE the
response landed. The UI promises "you can close the tab and come back."
The synchronous endpoint can't honour that promise.

This module flips the model:

  POST /session spawns the wander as `asyncio.create_task(...)`, registers
  a JobState here, and returns 202 IMMEDIATELY with the session_id +
  job_id. The frontend polls GET /session/<id>/status until status flips
  to "completed", then fetches the dossier via GET /session/<id>.

  The browser tab can close, navigate away, or crash — the task keeps
  running inside the FastAPI process. When the user comes back (same
  tab, new tab, another device with the session_id remembered), polling
  resumes against the same JobState.

  POST /session/<id>/abort cancels the running task by calling
  asyncio.Task.cancel(). The cancellation propagates through the next
  await point (typically the next LLM call) and the worker catches the
  CancelledError to mark the job aborted.

LIFETIME: state lives in-process. Survives tab close (same process keeps
the task alive). Does NOT survive server restart — same constraint as
the existing _SESSION_CACHE in routes.py. Eventually we mirror this to
Neo4j; for now in-process is the right pragmatic step.

THREAD SAFETY: FastAPI runs all asyncio code in one event loop. Reads +
writes on _JOBS dict are not atomic in general, but the asyncio model
means no real concurrency on the dict — only cooperative yields at
await points. The lifecycle helpers below never await, so they're safe.

ISOLATION: this module touches nothing outside the wandering namespace.
No Thread/Iteration writes, no master graph reads.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


log = logging.getLogger("constellax.wandering.jobs")


class JobStatus(str, Enum):
    """The four states a wander job can be in.

    `str, Enum` inheritance makes JobStatus.RUNNING.value == "running"
    serialize cleanly through JSONResponse without extra conversion."""

    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    ABORTED   = "aborted"


@dataclass
class JobState:
    """One wandering job's lifecycle state.

    Held in-process. Mutated by the worker as it transitions through
    statuses. `task` is the asyncio.Task reference — used to support
    abort, never serialized."""

    session_id:   str
    job_id:       str
    user_id:      str | None
    started_at:   float
    # Snapshot fields so /status can drive the LiveWandering UI without
    # the frontend having to keep the original brief inputs around. The
    # pursuit string is also what we want to show in the wait screen if
    # the user reopened the tab and lost their local state.
    mode:         str            # WanderingMode.value, e.g. "multi_pendulum"
    agents:       int            # resolved agent count
    time_budget_seconds: float   # resolved time budget
    pursuit:      str            # snapshot of cushion.raw_input.problem.content
    # Lifecycle fields — set by mark_*() helpers as the job progresses.
    status:       JobStatus       = JobStatus.RUNNING
    completed_at: float | None    = None
    error:        str | None      = None
    # The asyncio.Task running the wander. Kept private (repr=False)
    # so it never leaks into logs or JSON responses.
    task:         asyncio.Task[Any] | None = field(default=None, repr=False)

    def is_terminal(self) -> bool:
        return self.status in (
            JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.ABORTED,
        )

    def to_dict(self) -> dict[str, Any]:
        # `elapsed_seconds` is computed live for running jobs so the
        # frontend can render an honest counter without a second clock.
        now = self.completed_at if self.is_terminal() else time.time()
        elapsed = max(0.0, now - self.started_at)
        return {
            "session_id":         self.session_id,
            "job_id":             self.job_id,
            "status":             self.status.value,
            "started_at":         self.started_at,
            "completed_at":       self.completed_at,
            "elapsed_seconds":    elapsed,
            "error":              self.error,
            "mode":               self.mode,
            "agents":             self.agents,
            "time_budget_seconds": self.time_budget_seconds,
            "pursuit":            self.pursuit,
        }


# Module-level registry. One entry per active OR recently-completed job.
# Completed entries are NOT auto-evicted in V1 — they're tiny and we
# want polling clients to be able to confirm completion even after
# they've drifted offline for a while.
_JOBS: dict[str, JobState] = {}


# ---------------------------------------------------------------------------
# Persistent mirror (F5) — Neo4j-backed durability across server restart
# ---------------------------------------------------------------------------
#
# The in-process registry above is the source of truth WHILE THE PROCESS
# IS ALIVE. To survive Railway redeploys, OOM-kills, or crashes, every
# lifecycle transition mirrors a JSON-able dict of the state to a store
# implementing the WanderingStore.save_job_state() protocol.
#
# Wiring: the routes module sets `_STORE` at startup via set_store().
# All `save_job_state` calls are awaited from the caller's context if
# possible; otherwise spawned as fire-and-forget tasks. Failures NEVER
# crash the in-process flow.

_STORE: Any = None  # WanderingStore protocol; lazily wired by routes.py


def set_store(store: Any) -> None:
    """Inject the durable store. Called once at app startup by routes.py.

    Passing None disables persistence — useful for the engine-test harness
    which doesn't need (or have) a real store. The in-process registry
    keeps working in either case.
    """
    global _STORE
    _STORE = store


def _mirror_state_async(state: JobState) -> None:
    """Fire-and-forget mirror of `state` to the durable store.

    We don't await here because the caller (mark_*) is sync and we don't
    want lifecycle transitions to block on I/O. asyncio.create_task is
    safe because all callers run inside the FastAPI event loop. If for
    any reason we're NOT on a loop (rare), we silently skip — the
    in-process state is still correct.
    """
    store = _STORE
    if store is None:
        return
    payload = state.to_dict()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop available — likely a test/cli context. Skip.
        return

    async def _do_save():
        try:
            await store.save_job_state(payload)
        except Exception as e:
            log.debug("job mirror save failed for %s: %s",
                      payload.get("session_id"), e)

    loop.create_task(_do_save())


def register_job(
    *,
    session_id: str,
    job_id:     str,
    user_id:    str | None,
    mode:       str,
    agents:     int,
    time_budget_seconds: float,
    pursuit:    str,
    task:       asyncio.Task[Any],
) -> JobState:
    """Create a JobState for a newly-spawned wander task.

    Raises RuntimeError if a RUNNING job for this session_id already
    exists — the caller's responsibility to either resume the existing
    job (via /status) or choose a different session_id."""
    existing = _JOBS.get(session_id)
    if existing is not None and existing.status == JobStatus.RUNNING:
        raise RuntimeError(
            f"a running job for session_id={session_id} already exists; "
            f"job_id={existing.job_id}"
        )
    state = JobState(
        session_id=session_id,
        job_id=job_id,
        user_id=user_id,
        started_at=time.time(),
        mode=mode,
        agents=agents,
        time_budget_seconds=time_budget_seconds,
        pursuit=pursuit,
        task=task,
    )
    _JOBS[session_id] = state
    _mirror_state_async(state)  # F5 — durability mirror
    return state


def get_job(session_id: str) -> JobState | None:
    """Lookup a job by session_id. Returns None if no job has been
    registered (or it was cleared)."""
    return _JOBS.get(session_id)


def mark_completed(session_id: str) -> None:
    state = _JOBS.get(session_id)
    if state is None:
        log.warning("mark_completed: no job for %s", session_id)
        return
    state.status       = JobStatus.COMPLETED
    state.completed_at = time.time()
    state.task         = None
    _mirror_state_async(state)


def mark_failed(session_id: str, error: str) -> None:
    state = _JOBS.get(session_id)
    if state is None:
        log.warning("mark_failed: no job for %s", session_id)
        return
    state.status       = JobStatus.FAILED
    state.error        = error
    state.completed_at = time.time()
    state.task         = None
    _mirror_state_async(state)


def mark_aborted(session_id: str) -> None:
    state = _JOBS.get(session_id)
    if state is None:
        log.warning("mark_aborted: no job for %s", session_id)
        return
    state.status       = JobStatus.ABORTED
    state.completed_at = time.time()
    state.task         = None
    _mirror_state_async(state)


# ---------------------------------------------------------------------------
# Restart-boundary helpers (F5)
# ---------------------------------------------------------------------------


async def sweep_interrupted_from_store() -> int:
    """At server startup, transition any persisted RUNNING jobs to FAILED
    with reason `server_restart_during_wander`. Returns count swept.

    These are jobs the PRIOR process left running when it exited (Railway
    redeploy, OOM, crash). Their asyncio tasks are gone — we don't fake
    a resume (per WANDERING_ROOM_FUTURE_WORK.md N3). We mark them
    cleanly failed so the frontend's resume path sees a usable status
    and can offer "Restart this wander?" to the user.

    No-op when no store is wired.
    """
    store = _STORE
    if store is None:
        return 0
    try:
        rows = await store.list_running_jobs()
    except Exception as e:
        log.warning("sweep_interrupted_from_store: list failed: %s", e)
        return 0

    swept = 0
    for row in rows:
        sid = row.get("session_id")
        if not sid:
            continue
        # Build an in-memory ghost so we can transition via the normal
        # path AND emit the mirror update.
        try:
            state = JobState(
                session_id=sid,
                job_id=row.get("job_id", ""),
                user_id=row.get("user_id"),
                started_at=float(row.get("started_at", 0.0) or 0.0),
                mode=str(row.get("mode", "")),
                agents=int(row.get("agents", 0) or 0),
                time_budget_seconds=float(
                    row.get("time_budget_seconds", 0.0) or 0.0
                ),
                pursuit=str(row.get("pursuit", "")),
                status=JobStatus.FAILED,
                completed_at=time.time(),
                error="server_restart_during_wander",
            )
            # Persist the transition. Don't re-insert into _JOBS — these
            # jobs belong to a dead PID; future status polls fall through
            # to the store via get_status_durable() below.
            try:
                await store.save_job_state(state.to_dict())
            except Exception as e:
                log.debug("sweep: save failed for %s: %s", sid, e)
            swept += 1
        except Exception as e:
            log.debug("sweep: bad row %s: %s", sid, e)

    if swept:
        log.info("sweep_interrupted_from_store: marked %d jobs failed", swept)
    return swept


async def get_status_durable(session_id: str) -> dict[str, Any] | None:
    """Lookup that consults the in-process registry first, then the
    durable store. Returns the same shape JobState.to_dict() emits, or
    None when neither layer has the session.

    `/status` route uses this. After a server restart, the in-process
    registry is empty but the store still has the terminal state — so
    the frontend can render the failed/aborted status correctly.
    """
    state = _JOBS.get(session_id)
    if state is not None:
        return state.to_dict()

    store = _STORE
    if store is None:
        return None
    try:
        return await store.get_job_state(session_id)
    except Exception as e:
        log.debug("get_status_durable store lookup failed for %s: %s",
                  session_id, e)
        return None


def drain_running_for_shutdown() -> list[JobState]:
    """Synchronous helper for SIGTERM. Returns RUNNING jobs in the
    registry; caller is responsible for marking them interrupted +
    persisting before exit. No mutation here — we want the SIGTERM
    handler to control the ordering (so logs make sense).
    """
    return [s for s in _JOBS.values() if s.status == JobStatus.RUNNING]


def abort_job(session_id: str) -> bool:
    """Cancel a running job. Returns True iff there was a running job
    to cancel. The caller should expect the worker to transition the
    job state to ABORTED via mark_aborted() shortly after — but we
    don't wait for that here (cancellation is cooperative; the await
    point may not surface until the next LLM call returns)."""
    state = _JOBS.get(session_id)
    if state is None or state.status != JobStatus.RUNNING:
        return False
    task = state.task
    if task is None or task.done():
        return False
    task.cancel()
    return True


def clear_all() -> None:
    """Test helper. Drops every JobState from the registry."""
    _JOBS.clear()


def all_jobs() -> list[JobState]:
    """Debug / observability helper. Returns a snapshot list of all jobs
    currently in the registry (running and terminal)."""
    return list(_JOBS.values())


__all__ = [
    "JobState",
    "JobStatus",
    "register_job",
    "get_job",
    "mark_completed",
    "mark_failed",
    "mark_aborted",
    "abort_job",
    "clear_all",
    "all_jobs",
    # F5 durability surface
    "set_store",
    "sweep_interrupted_from_store",
    "get_status_durable",
    "drain_running_for_shutdown",
]
