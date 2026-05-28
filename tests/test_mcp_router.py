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
from src.mcp_router import (
    McpHandlerRegistry,
    McpResult,
    fire_mcp,
    fire_mcps,
    format_mcp_results_for_prompt,
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
# 10. McpHandlerRegistry — real-dispatch path (Phase B1)
# ---------------------------------------------------------------------------

@test("10.1 handler-less fire_mcp still returns stub (back-compat)")
async def test_fire_mcp_without_handlers():
    reg = CapabilityRegistry()
    result = await fire_mcp(reg, "graphify", purpose="x", handlers=None)
    assert result.ok is True
    assert result.stub is True


@test("10.2 fire_mcp dispatches to a registered handler and returns real data")
async def test_fire_mcp_dispatch_to_handler():
    reg = CapabilityRegistry()
    handlers = McpHandlerRegistry()

    async def fake_handler(args, purpose):
        return {"text": "real data here", "echo_args": args, "purpose": purpose}

    handlers.register("graphify", fake_handler)

    result = await fire_mcp(
        reg, "graphify",
        purpose="get structure",
        args={"file": "x.py"},
        handlers=handlers,
    )
    assert result.ok is True
    assert result.stub is False
    assert result.result["text"] == "real data here"
    assert result.result["echo_args"] == {"file": "x.py"}
    assert result.result["purpose"] == "get structure"


@test("10.3 handler exception → ok=False with the error in blocked_reason")
async def test_handler_exception_becomes_failure():
    reg = CapabilityRegistry()
    handlers = McpHandlerRegistry()

    async def broken_handler(args, purpose):
        raise RuntimeError("network down")

    handlers.register("graphify", broken_handler)

    result = await fire_mcp(reg, "graphify", purpose="x", handlers=handlers)
    assert result.ok is False
    assert "network down" in result.blocked_reason
    # Failure is logged into usage history with success=False
    history = reg.usage_history("graphify")
    assert history[-1].success is False
    assert "network down" in (history[-1].error or "")


@test("10.4 handler returning non-dict is treated as failure (defensive)")
async def test_handler_non_dict_result():
    reg = CapabilityRegistry()
    handlers = McpHandlerRegistry()

    async def bad_shape(args, purpose):
        return "just a string"  # type: ignore[return-value]

    handlers.register("graphify", bad_shape)
    result = await fire_mcp(reg, "graphify", purpose="x", handlers=handlers)
    assert result.ok is False
    assert "non-dict" in result.blocked_reason


@test("10.5 registry without this name → stub fallback (graceful)")
async def test_handlers_present_but_name_unregistered():
    reg = CapabilityRegistry()
    handlers = McpHandlerRegistry()

    async def github_handler(args, purpose):
        return {"text": "issues..."}

    handlers.register("github", github_handler)

    # graphify has no handler registered → stub fallback
    result = await fire_mcp(reg, "graphify", purpose="x", handlers=handlers)
    assert result.ok is True
    assert result.stub is True


@test("10.6 fire_mcps passes handlers through to each call")
async def test_fire_mcps_uses_handlers():
    reg = CapabilityRegistry()
    handlers = McpHandlerRegistry()

    async def graphify_handler(args, purpose):
        return {"text": "code structure"}

    handlers.register("graphify", graphify_handler)

    results = await fire_mcps(
        reg,
        [
            {"name": "graphify", "purpose": "structure"},
            {"name": "bridge", "purpose": "decisions"},
        ],
        handlers=handlers,
    )
    assert results[0].stub is False
    assert results[0].result["text"] == "code structure"
    assert results[1].stub is True  # bridge has no handler


@test("10.7b McpHandlerRegistry.extend() returns an isolated copy")
def test_handler_registry_extend():
    base = McpHandlerRegistry()

    async def base_handler(args, purpose):
        return {"text": "base"}

    async def extra_handler(args, purpose):
        return {"text": "extra"}

    base.register("github", base_handler)

    # Extended registry inherits base's handlers
    extended = base.extend()
    assert extended.has("github")
    assert extended.get("github") is base_handler

    # Mutating the extended registry does NOT touch the base
    extended.register("filesystem", extra_handler)
    assert extended.has("filesystem")
    assert not base.has("filesystem")


@test("10.7 McpHandlerRegistry rejects invalid registrations")
def test_handler_registry_validation():
    h = McpHandlerRegistry()
    try:
        h.register("", lambda a, p: None)  # empty name
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty name")

    try:
        h.register("github", "not a callable")  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError on non-callable handler")


# ---------------------------------------------------------------------------
# 11. format_mcp_results_for_prompt — Phase B1 prompt injection helper
# ---------------------------------------------------------------------------

@test("11.1 format with no real results returns empty string")
def test_format_empty_when_all_stubs():
    out = format_mcp_results_for_prompt([
        McpResult(ok=True, capability="graphify", stub=True, result={"stub": True}),
        McpResult(ok=False, capability="github", blocked_reason="missing"),
    ])
    assert out == ""


@test("11.2 format renders real handler results with PRIOR-MEMORY-style header")
def test_format_renders_real_results():
    real = McpResult(
        ok=True, capability="github", stub=False,
        result={"text": "PR #482 is approved.", "_purpose": "check PR state"},
    )
    out = format_mcp_results_for_prompt([real])
    assert "## MCP CONTEXT" in out
    assert "### github — check PR state" in out
    assert "PR #482 is approved." in out


@test("11.3 format falls back to JSON dump when handler has no 'text' field")
def test_format_falls_back_to_dump():
    real = McpResult(
        ok=True, capability="docs", stub=False,
        result={"library": "stripe", "snippet": "Idempotency-Key header"},
    )
    out = format_mcp_results_for_prompt([real])
    assert "### docs" in out
    assert "stripe" in out


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
    test_fire_mcp_without_handlers,
    test_fire_mcp_dispatch_to_handler,
    test_handler_exception_becomes_failure,
    test_handler_non_dict_result,
    test_handlers_present_but_name_unregistered,
    test_fire_mcps_uses_handlers,
    test_handler_registry_extend,
    test_handler_registry_validation,
    test_format_empty_when_all_stubs,
    test_format_renders_real_results,
    test_format_falls_back_to_dump,
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
