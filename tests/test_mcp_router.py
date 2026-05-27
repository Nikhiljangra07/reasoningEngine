"""
MCP Router (stub) tests.

No LLM calls, no real MCP I/O. Verifies the router contract end-to-end
against the capability registry's permission model.

Run: PYTHONPATH=. python3 tests/test_mcp_router.py
"""

from __future__ import annotations

import asyncio

from src.capabilities import (
    Capability,
    CapabilityRegistry,
    CapabilityStatus,
    CapabilityType,
    PermissionState,
)
from src.mcp_router import McpResult, fire_mcp, fire_mcps


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
# 1. AVAILABLE + ALWAYS_ALLOWED (local read) → ok stub fire
# ---------------------------------------------------------------------------

@test("1.1 fire_mcp on AVAILABLE local read returns ok=True with stub flag")
async def test_local_read_fires_stub():
    reg = CapabilityRegistry()
    result = await fire_mcp(reg, "graphify", purpose="get code structure")
    assert result.ok is True
    assert result.stub is True
    assert result.capability == "graphify"
    assert "stub" in result.result
    assert result.notified_user is False  # local read — silent fire


@test("1.2 fire_mcp on AVAILABLE local read logs successful usage")
async def test_local_read_logs_usage():
    reg = CapabilityRegistry()
    await fire_mcp(reg, "graphify", purpose="checking auth module")
    history = reg.usage_history("graphify")
    assert len(history) == 1
    assert history[0].success is True
    assert history[0].purpose == "checking auth module"


# ---------------------------------------------------------------------------
# 2. MISSING → blocked with reason
# ---------------------------------------------------------------------------

@test("2.1 fire_mcp on MISSING capability returns ok=False with not-connected reason")
async def test_missing_blocked():
    # github is still MISSING by default (its MCP client isn't wired yet).
    # memory_v2 was MISSING pre-Phase-A but is now AVAILABLE.
    reg = CapabilityRegistry()
    result = await fire_mcp(reg, "github", purpose="check PR state")
    assert result.ok is False
    assert "not connected" in result.blocked_reason


@test("2.2 blocked fires still log usage (with success=False)")
async def test_blocked_logs_failure():
    reg = CapabilityRegistry()
    await fire_mcp(reg, "github", purpose="x")
    history = reg.usage_history("github")
    assert len(history) == 1
    assert history[0].success is False


# ---------------------------------------------------------------------------
# 3. ABSENT_BY_DESIGN → blocked structurally
# ---------------------------------------------------------------------------

@test("3.1 fire_mcp on ABSENT_BY_DESIGN returns ok=False")
async def test_absent_blocked():
    reg = CapabilityRegistry()
    result = await fire_mcp(reg, "file_write", purpose="edit a file")
    assert result.ok is False
    assert "absent by design" in result.blocked_reason


# ---------------------------------------------------------------------------
# 4. ASK_ONCE_PENDING → needs consent, blocked
# ---------------------------------------------------------------------------

@test("4.1 fire_mcp on ASK_ONCE_PENDING external read returns ok=False")
async def test_pending_blocked():
    reg = CapabilityRegistry()
    reg.mark_available("web_search")  # status=AVAILABLE but permission still pending
    result = await fire_mcp(reg, "web_search", purpose="check Stripe pricing")
    assert result.ok is False
    assert "consent" in result.blocked_reason or "first use" in result.blocked_reason


# ---------------------------------------------------------------------------
# 5. After grant_permission → fires + notifies user
# ---------------------------------------------------------------------------

@test("5.1 fire_mcp on AVAILABLE + ASK_ONCE_GRANTED returns ok=True with user notification")
async def test_external_read_after_grant():
    reg = CapabilityRegistry()
    reg.mark_available("web_search")
    reg.grant_permission("web_search")
    result = await fire_mcp(reg, "web_search", purpose="check current Stripe pricing")
    assert result.ok is True
    assert result.stub is True
    assert result.notified_user is True  # external read — user sees a notification


# ---------------------------------------------------------------------------
# 6. After deny_permission → blocked
# ---------------------------------------------------------------------------

@test("6.1 fire_mcp after deny_permission returns ok=False")
async def test_denied_blocked():
    reg = CapabilityRegistry()
    reg.mark_available("github")
    reg.deny_permission("github")
    result = await fire_mcp(reg, "github", purpose="check PRs")
    assert result.ok is False
    assert "denied" in result.blocked_reason


# ---------------------------------------------------------------------------
# 7. Unknown capability → blocked
# ---------------------------------------------------------------------------

@test("7.1 fire_mcp on unknown capability returns ok=False")
async def test_unknown_blocked():
    reg = CapabilityRegistry()
    result = await fire_mcp(reg, "imaginary_thing", purpose="x")
    assert result.ok is False
    assert "not registered" in result.blocked_reason


# ---------------------------------------------------------------------------
# 8. fire_mcps batch helper
# ---------------------------------------------------------------------------

@test("8.1 fire_mcps returns one result per request, same order")
async def test_fire_mcps_batch():
    # Use github (still MISSING) instead of memory_v2 (now AVAILABLE post-Phase-A).
    reg = CapabilityRegistry()
    requests = [
        {"name": "graphify", "purpose": "code structure"},
        {"name": "github", "purpose": "check PR state"},
        {"name": "file_write", "purpose": "edit"},
    ]
    results = await fire_mcps(reg, requests)
    assert len(results) == 3
    assert results[0].ok is True            # graphify available
    assert results[1].ok is False           # github missing
    assert results[2].ok is False           # file_write absent
    assert "absent by design" in results[2].blocked_reason


@test("8.2 fire_mcps handles missing purpose gracefully")
async def test_fire_mcps_missing_purpose():
    reg = CapabilityRegistry()
    results = await fire_mcps(reg, [{"name": "graphify"}])
    assert results[0].ok is True


# ---------------------------------------------------------------------------
# 9. args passthrough
# ---------------------------------------------------------------------------

@test("9.1 fire_mcp args appear in stub result")
async def test_args_passthrough():
    reg = CapabilityRegistry()
    result = await fire_mcp(
        reg, "graphify",
        purpose="get callers of foo",
        args={"symbol": "run_formation"},
    )
    assert result.ok is True
    assert result.result["args"] == {"symbol": "run_formation"}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_local_read_fires_stub,
    test_local_read_logs_usage,
    test_missing_blocked,
    test_blocked_logs_failure,
    test_absent_blocked,
    test_pending_blocked,
    test_external_read_after_grant,
    test_denied_blocked,
    test_unknown_blocked,
    test_fire_mcps_batch,
    test_fire_mcps_missing_purpose,
    test_args_passthrough,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} MCP router tests...")
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
