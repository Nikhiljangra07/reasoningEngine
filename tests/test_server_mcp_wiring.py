"""
Server-side MCP wiring tests — validate that the helpers server.py
exposes wire requests through to the right CapabilityRegistry state.

Slow: importing `server` brings up the full module (logging, Neo4j
driver init, MCP handler registration). Worth the cost because these
tests verify the actual production wiring, not a mocked copy.

Run: PYTHONPATH=. python tests/test_server_mcp_wiring.py
"""

from __future__ import annotations

import asyncio

import server
from src.capabilities.registry import (
    CapabilityStatus,
    PermissionState,
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


# ---------------------------------------------------------------------------
# 1. _normalize_selected_mcps
# ---------------------------------------------------------------------------

@test("1.1 _normalize_selected_mcps returns [] for non-list inputs")
def test_normalize_non_list():
    norm = server._normalize_selected_mcps
    assert norm(None) == []
    assert norm("github,filesystem") == []
    assert norm({"github": True}) == []
    assert norm(42) == []


@test("1.2 _normalize_selected_mcps keeps valid string names, strips whitespace")
def test_normalize_valid():
    norm = server._normalize_selected_mcps
    assert norm(["github", "  web_search  ", "filesystem"]) == [
        "github", "web_search", "filesystem",
    ]


@test("1.3 _normalize_selected_mcps drops empty/non-string entries")
def test_normalize_drops_bad():
    norm = server._normalize_selected_mcps
    assert norm(["github", "", None, 42, "  ", "filesystem"]) == [
        "github", "filesystem",
    ]


# ---------------------------------------------------------------------------
# 2. _make_capability_registry — selected_mcps grants
# ---------------------------------------------------------------------------

@test("2.1 selected_mcps grants ASK_ONCE_PENDING → ASK_ONCE_GRANTED")
def test_grant_for_pending():
    # github is MISSING by default; we don't expect grant to change status,
    # only the permission state. Use a cap that IS available + pending —
    # web_search (AVAILABLE post-Phase-A, ASK_ONCE_PENDING by default).
    reg = server._make_capability_registry(selected_mcps=["web_search"])
    cap = reg.get("web_search")
    assert cap.status == CapabilityStatus.AVAILABLE
    assert cap.permission == PermissionState.ASK_ONCE_GRANTED


@test("2.2 NOT selected → permission stays ASK_ONCE_PENDING (default)")
def test_unselected_stays_pending():
    reg = server._make_capability_registry(selected_mcps=["github"])
    cap = reg.get("web_search")
    # web_search wasn't in selected_mcps → stays at default pending
    assert cap.permission == PermissionState.ASK_ONCE_PENDING


@test("2.3 selected_mcps with unknown name is silently dropped")
def test_unknown_name_no_crash():
    # Should not raise, should not affect other caps
    reg = server._make_capability_registry(selected_mcps=["totally_made_up_mcp"])
    assert reg.get("web_search").permission == PermissionState.ASK_ONCE_PENDING


@test("2.4 selected_mcps for ABSENT_BY_DESIGN cap is a no-op (safety)")
def test_grant_absent_by_design_noop():
    reg = server._make_capability_registry(selected_mcps=["file_write"])
    cap = reg.get("file_write")
    # Permission stays absent_by_design — registry refuses to flip it
    assert cap.permission == PermissionState.ABSENT_BY_DESIGN


@test("2.5 selected_mcps for ALWAYS_ALLOWED (local read) is a no-op")
def test_grant_always_allowed_noop():
    reg = server._make_capability_registry(selected_mcps=["memory_v2"])
    cap = reg.get("memory_v2")
    # memory_v2 is LOCAL_READ + ALWAYS_ALLOWED — no consent needed.
    # The helper's `if cap.permission == ASK_ONCE_PENDING:` guard skips
    # it, so permission stays ALWAYS_ALLOWED untouched.
    assert cap.permission == PermissionState.ALWAYS_ALLOWED


@test("2.6 attached_files + selected_mcps work together (filesystem + grants)")
def test_combined_layers():
    reg = server._make_capability_registry(
        attached_files=[{"path": "x.py", "content": "y"}],
        selected_mcps=["web_search", "filesystem"],
    )
    # filesystem flipped AVAILABLE by attached_files
    assert reg.get("filesystem").status == CapabilityStatus.AVAILABLE
    # web_search granted consent by selected_mcps
    assert reg.get("web_search").permission == PermissionState.ASK_ONCE_GRANTED
    # filesystem is ALWAYS_ALLOWED so the grant is a no-op (correct)
    assert reg.get("filesystem").permission == PermissionState.ALWAYS_ALLOWED


@test("2.7 empty/None selected_mcps → no grants, all defaults")
def test_empty_selected():
    reg_none = server._make_capability_registry(selected_mcps=None)
    reg_empty = server._make_capability_registry(selected_mcps=[])
    for reg in (reg_none, reg_empty):
        # web_search stays pending
        assert reg.get("web_search").permission == PermissionState.ASK_ONCE_PENDING


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_normalize_non_list,
    test_normalize_valid,
    test_normalize_drops_bad,
    test_grant_for_pending,
    test_unselected_stays_pending,
    test_unknown_name_no_crash,
    test_grant_absent_by_design_noop,
    test_grant_always_allowed_noop,
    test_combined_layers,
    test_empty_selected,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} server MCP wiring tests...")
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
