"""
Tests for src/wandering/jobs.py — in-process JobState registry.

Pure-logic + async-cancellation tests. No FastAPI, no LLM. The route-
layer tests live in test_wandering_wiring.py (route registration).

Run: PYTHONPATH=. python3 tests/test_wandering_jobs.py
"""

from __future__ import annotations

import asyncio
import time

from src.wandering import jobs
from src.wandering.jobs import JobState, JobStatus


PASSED = 0
FAILED = 0
ERRORS: list[tuple[str, str]] = []


def test(name: str):
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    global PASSED, FAILED
    name = getattr(fn, "_test_name", fn.__name__)
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ===========================================================================
# Helpers
# ===========================================================================


def _fresh_state() -> None:
    jobs.clear_all()


async def _dummy_task() -> None:
    """A task that runs forever — used as a stand-in for the wander
    task in tests that don't actually need the wander to do anything."""
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        raise


def _register(session_id: str = "wsess-test", task: asyncio.Task | None = None) -> JobState:
    """Convenience: register a job with sensible defaults."""
    if task is None:
        # Caller doesn't care about the task — give them a no-op coroutine.
        async def _noop():
            await asyncio.sleep(60)
        task = asyncio.get_event_loop().create_task(_noop())
    return jobs.register_job(
        session_id=session_id,
        job_id=f"wjob-{session_id[-4:]}",
        user_id="user-1",
        mode="multi_pendulum",
        agents=5,
        time_budget_seconds=1800.0,
        pursuit="test pursuit",
        task=task,
    )


# ===========================================================================
# Lifecycle tests
# ===========================================================================


@test("register_job: creates a RUNNING state with all fields populated")
async def t_register():
    _fresh_state()
    state = _register("wsess-a")
    assert state.session_id == "wsess-a"
    assert state.status     == JobStatus.RUNNING
    assert state.completed_at is None
    assert state.error      is None
    assert state.started_at > 0
    assert state.mode       == "multi_pendulum"
    assert state.agents     == 5
    assert state.pursuit    == "test pursuit"
    assert state.task is not None
    # And it's discoverable by lookup.
    found = jobs.get_job("wsess-a")
    assert found is state
    # Cleanup: cancel the task so the test runner exits cleanly.
    state.task.cancel()
    try:
        await state.task
    except (asyncio.CancelledError, BaseException):
        pass


@test("register_job: rejects duplicate while running")
async def t_register_duplicate_running():
    _fresh_state()
    first = _register("wsess-dup")
    raised = False
    try:
        _register("wsess-dup")
    except RuntimeError as e:
        assert "already exists" in str(e)
        raised = True
    assert raised, "expected RuntimeError on duplicate RUNNING session_id"
    first.task.cancel()
    try:
        await first.task
    except BaseException:
        pass


@test("register_job: allows re-register after previous job terminates")
async def t_register_after_terminal():
    _fresh_state()
    first = _register("wsess-reuse")
    first.task.cancel()
    try:
        await first.task
    except BaseException:
        pass
    jobs.mark_aborted("wsess-reuse")
    # Now a new registration with the same session_id should succeed
    # because the prior job is in a terminal state.
    second = _register("wsess-reuse")
    assert second is not first
    assert second.status == JobStatus.RUNNING
    second.task.cancel()
    try:
        await second.task
    except BaseException:
        pass


@test("mark_completed: transitions to COMPLETED, sets completed_at, clears task ref")
async def t_mark_completed():
    _fresh_state()
    state = _register("wsess-c")
    t0 = time.time()
    jobs.mark_completed("wsess-c")
    assert state.status == JobStatus.COMPLETED
    assert state.completed_at is not None
    assert state.completed_at >= t0
    assert state.task is None  # cleared after terminal
    # task was orphaned — cancel it via the local reference we kept
    # (state.task is None now, but we still need to clean up the orphan).


@test("mark_failed: sets FAILED status + error message")
def t_mark_failed():
    async def _go():
        _fresh_state()
        state = _register("wsess-f")
        jobs.mark_failed("wsess-f", "Anthropic rate-limit hit")
        assert state.status == JobStatus.FAILED
        assert state.error  == "Anthropic rate-limit hit"
        assert state.completed_at is not None
        # Orphaned task cleanup
        if state.task is None:
            pass  # already cleared
    asyncio.run(_go())


@test("mark_aborted: sets ABORTED status, no error required")
def t_mark_aborted():
    async def _go():
        _fresh_state()
        state = _register("wsess-ab")
        jobs.mark_aborted("wsess-ab")
        assert state.status == JobStatus.ABORTED
        assert state.error is None
        assert state.completed_at is not None
    asyncio.run(_go())


@test("mark_*: no-op when session_id not registered")
def t_mark_unknown():
    _fresh_state()
    # None of these should raise.
    jobs.mark_completed("nonexistent")
    jobs.mark_failed("nonexistent", "x")
    jobs.mark_aborted("nonexistent")
    assert jobs.get_job("nonexistent") is None


# ===========================================================================
# abort_job tests
# ===========================================================================


@test("abort_job: cancels the underlying task, returns True")
def t_abort_running():
    async def _go():
        _fresh_state()
        async def _noop():
            await asyncio.sleep(60)
        task = asyncio.get_event_loop().create_task(_noop())
        state = jobs.register_job(
            session_id="wsess-abrt",
            job_id="wjob-x",
            user_id=None,
            mode="multi_pendulum",
            agents=5,
            time_budget_seconds=1800.0,
            pursuit="p",
            task=task,
        )
        result = jobs.abort_job("wsess-abrt")
        assert result is True
        # Yield once so the cancellation has a chance to surface.
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled()
        # State is NOT yet marked aborted by abort_job itself — the worker's
        # CancelledError handler does that. For this test we simulate it.
        jobs.mark_aborted("wsess-abrt")
        assert state.status == JobStatus.ABORTED
    asyncio.run(_go())


@test("abort_job: returns False on already-terminal job")
def t_abort_terminal():
    async def _go():
        _fresh_state()
        state = _register("wsess-already-done")
        jobs.mark_completed("wsess-already-done")
        # Task ref was cleared by mark_completed; abort should be a no-op.
        result = jobs.abort_job("wsess-already-done")
        assert result is False
        assert state.status == JobStatus.COMPLETED
    asyncio.run(_go())


@test("abort_job: returns False for unknown session")
def t_abort_unknown():
    _fresh_state()
    assert jobs.abort_job("nonexistent") is False


# ===========================================================================
# to_dict + is_terminal
# ===========================================================================


@test("to_dict: surfaces all relevant fields including live elapsed_seconds")
def t_to_dict_running():
    async def _go():
        _fresh_state()
        state = _register("wsess-d")
        # Wait long enough that elapsed becomes a measurable positive number.
        await asyncio.sleep(0.05)
        d = state.to_dict()
        assert d["session_id"] == "wsess-d"
        assert d["job_id"].startswith("wjob-")
        assert d["status"] == "running"
        assert d["completed_at"] is None
        assert d["elapsed_seconds"] > 0
        assert d["mode"]    == "multi_pendulum"
        assert d["agents"]  == 5
        assert d["pursuit"] == "test pursuit"
        assert d["error"]   is None
        # task ref must not leak into the dict
        assert "task" not in d
        state.task.cancel()
        try:
            await state.task
        except BaseException:
            pass
    asyncio.run(_go())


@test("to_dict: elapsed_seconds is frozen once the job is terminal")
def t_to_dict_terminal():
    async def _go():
        _fresh_state()
        state = _register("wsess-frozen")
        await asyncio.sleep(0.02)
        jobs.mark_completed("wsess-frozen")
        d1 = state.to_dict()
        await asyncio.sleep(0.05)
        d2 = state.to_dict()
        # Elapsed should NOT have advanced between d1 and d2.
        assert d1["elapsed_seconds"] == d2["elapsed_seconds"]
        assert d2["status"] == "completed"
    asyncio.run(_go())


@test("is_terminal: True for completed/failed/aborted, False for running")
def t_is_terminal():
    async def _go():
        _fresh_state()
        state = _register("wsess-t")
        assert state.is_terminal() is False
        jobs.mark_completed("wsess-t")
        assert state.is_terminal() is True
        # Reset
        _fresh_state()
        s2 = _register("wsess-t2")
        jobs.mark_failed("wsess-t2", "x")
        assert s2.is_terminal() is True
        _fresh_state()
        s3 = _register("wsess-t3")
        jobs.mark_aborted("wsess-t3")
        assert s3.is_terminal() is True
    asyncio.run(_go())


# ===========================================================================
# Status string serialization
# ===========================================================================


@test("JobStatus serializes to the expected string values")
def t_status_values():
    assert JobStatus.RUNNING.value   == "running"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value    == "failed"
    assert JobStatus.ABORTED.value   == "aborted"


@test("all_jobs: returns snapshot of registry")
def t_all_jobs():
    async def _go():
        _fresh_state()
        assert jobs.all_jobs() == []
        s1 = _register("wsess-1")
        s2 = _register("wsess-2")
        all_states = jobs.all_jobs()
        assert len(all_states) == 2
        ids = {s.session_id for s in all_states}
        assert ids == {"wsess-1", "wsess-2"}
        for s in (s1, s2):
            s.task.cancel()
            try:
                await s.task
            except BaseException:
                pass
    asyncio.run(_go())


# ===========================================================================
# Runner
# ===========================================================================


if __name__ == "__main__":
    tests = [
        t_register,
        t_register_duplicate_running,
        t_register_after_terminal,
        t_mark_completed,
        t_mark_failed,
        t_mark_aborted,
        t_mark_unknown,
        t_abort_running,
        t_abort_terminal,
        t_abort_unknown,
        t_to_dict_running,
        t_to_dict_terminal,
        t_is_terminal,
        t_status_values,
        t_all_jobs,
    ]
    print(f"\nRunning {len(tests)} jobs tests...\n")
    for fn in tests:
        run_test(fn)
    print(f"\n{'=' * 60}")
    print(f"  {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 60}")
    if FAILED:
        for name, msg in ERRORS:
            print(f"  ✗ {name}: {msg}")
        raise SystemExit(1)
