"""
Phase 3C-strict ownership tests — _assert_thread_ownership semantics.

The router-level ownership gate is the single security primitive that
keeps signed-in users' threads from being read/deleted by anyone with
a thread ID. These tests pin the behavior of the gate itself; the
endpoint-level wiring (does each endpoint call the gate?) is verified
by code review since the endpoints depend on a live FastAPI request +
ThreadStore, which are heavy to fake just for that.

Run: PYTHONPATH=. python tests/test_thread_ownership.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException

from src.auth.supabase_auth import VerifiedUser
from src.bridge.thread_persistence import (
    _assert_thread_ownership,
    _load_thread_for_iteration,
)


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


@dataclass
class _FakeState:
    verified_user: VerifiedUser | None = None


@dataclass
class _FakeRequest:
    state: _FakeState = field(default_factory=_FakeState)
    headers: dict = field(default_factory=dict)


@dataclass
class _FakeThread:
    user_id: str | None = None
    id: str = "thr-xyz"


@dataclass
class _FakeIteration:
    id: str = "iter-abc"
    thread_id: str = "thr-xyz"


# ---------------------------------------------------------------------------
# 1. _assert_thread_ownership
# ---------------------------------------------------------------------------

@test("1.1 anonymous caller (no verified user) passes through unchanged")
def test_anonymous_passes():
    req = _FakeRequest()
    req.state.verified_user = None
    thread = _FakeThread(user_id="some-other-user")
    # Should NOT raise — anonymous calls preserve legacy behavior.
    _assert_thread_ownership(thread, req)  # type: ignore[arg-type]


@test("1.2 signed-in caller, owns the thread → passes")
def test_owner_passes():
    req = _FakeRequest()
    req.state.verified_user = VerifiedUser(user_id="user-A")
    thread = _FakeThread(user_id="user-A")
    _assert_thread_ownership(thread, req)  # type: ignore[arg-type]


@test("1.3 signed-in caller, different owner → 404")
def test_wrong_owner_404():
    req = _FakeRequest()
    req.state.verified_user = VerifiedUser(user_id="user-A")
    thread = _FakeThread(user_id="user-B")
    try:
        _assert_thread_ownership(thread, req)  # type: ignore[arg-type]
    except HTTPException as e:
        assert e.status_code == 404, f"expected 404, got {e.status_code}"
        return
    raise AssertionError("expected HTTPException(404)")


@test("1.4 signed-in caller, orphan thread (user_id=None) → 404")
def test_orphan_rejected_for_authed():
    req = _FakeRequest()
    req.state.verified_user = VerifiedUser(user_id="user-A")
    thread = _FakeThread(user_id=None)
    try:
        _assert_thread_ownership(thread, req)  # type: ignore[arg-type]
    except HTTPException as e:
        assert e.status_code == 404
        return
    raise AssertionError("expected HTTPException(404) for orphan thread")


@test("1.5 anonymous caller, orphan thread → passes (legacy reads allowed)")
def test_anonymous_can_read_orphan():
    req = _FakeRequest()
    req.state.verified_user = None
    thread = _FakeThread(user_id=None)
    _assert_thread_ownership(thread, req)  # type: ignore[arg-type]


@test("1.6 404 message is generic (no ID/owner leak)")
def test_404_message_generic():
    req = _FakeRequest()
    req.state.verified_user = VerifiedUser(user_id="user-A")
    thread = _FakeThread(user_id="user-B", id="thr-secret")
    try:
        _assert_thread_ownership(thread, req)  # type: ignore[arg-type]
    except HTTPException as e:
        # Must not leak the thread ID or owner in the detail.
        detail_str = str(e.detail).lower()
        assert "thr-secret" not in detail_str, "thread id leaked in error detail"
        assert "user-b" not in detail_str, "owner id leaked in error detail"
        assert "user-a" not in detail_str, "caller id leaked in error detail"
        return
    raise AssertionError("expected HTTPException")


# ---------------------------------------------------------------------------
# 2. _load_thread_for_iteration
# ---------------------------------------------------------------------------

class _FakeStore:
    """Minimal async stub of ThreadStore.{get_iteration, get_thread}."""

    def __init__(self, iterations: dict, threads: dict):
        self.iterations = iterations
        self.threads = threads

    async def get_iteration(self, iter_id: str):
        return self.iterations.get(iter_id)

    async def get_thread(self, thread_id: str):
        return self.threads.get(thread_id)


@test("2.1 iteration + thread both exist → returns both")
async def test_load_both():
    it = _FakeIteration(id="iter-1", thread_id="thr-1")
    thread = _FakeThread(id="thr-1", user_id="u-1")
    store = _FakeStore({"iter-1": it}, {"thr-1": thread})
    got_it, got_thread = await _load_thread_for_iteration(store, "iter-1")
    assert got_it is it
    assert got_thread is thread


@test("2.2 iteration missing → returns (None, None)")
async def test_load_iter_missing():
    store = _FakeStore({}, {})
    got_it, got_thread = await _load_thread_for_iteration(store, "iter-missing")
    assert got_it is None
    assert got_thread is None


@test("2.3 iteration has no thread_id → returns (iteration, None)")
async def test_load_no_thread_id():
    it = _FakeIteration(id="iter-orphan", thread_id="")
    store = _FakeStore({"iter-orphan": it}, {})
    got_it, got_thread = await _load_thread_for_iteration(store, "iter-orphan")
    assert got_it is it
    assert got_thread is None


@test("2.4 iteration exists but thread missing → returns (iteration, None)")
async def test_load_thread_missing():
    it = _FakeIteration(id="iter-1", thread_id="thr-deleted")
    store = _FakeStore({"iter-1": it}, {})  # thread not in store
    got_it, got_thread = await _load_thread_for_iteration(store, "iter-1")
    assert got_it is it
    assert got_thread is None


@test("2.5 store.get_thread raises → returns (iteration, None), no propagation")
async def test_load_thread_raises():
    class _RaisingStore:
        async def get_iteration(self, iter_id: str):
            return _FakeIteration(id=iter_id, thread_id="thr-1")
        async def get_thread(self, thread_id: str):
            raise RuntimeError("simulated DB outage")
    got_it, got_thread = await _load_thread_for_iteration(_RaisingStore(), "iter-x")
    assert got_it is not None
    assert got_thread is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_anonymous_passes,
    test_owner_passes,
    test_wrong_owner_404,
    test_orphan_rejected_for_authed,
    test_anonymous_can_read_orphan,
    test_404_message_generic,
    test_load_both,
    test_load_iter_missing,
    test_load_no_thread_id,
    test_load_thread_missing,
    test_load_thread_raises,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} thread-ownership tests...")
    print()
    for fn in ALL_TESTS:
        run_test(fn)
    print()
    print(f"{PASSED} passed, {FAILED} failed")
    if ERRORS:
        print()
        print("Failures:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
