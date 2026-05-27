"""
Capability Registry tests.

No LLM calls, no API hits. Tests verify:
    1. Default seed registers expected capabilities
    2. Queries return correct subsets (available / missing / absent_by_design)
    3. State mutations respect the absent-by-design safety property
    4. Permission grant / deny / reset lifecycle
    5. Usage logging and history queries
    6. can_fire decision logic across all permission states
    7. needs_consent identifies ASK_ONCE_PENDING correctly
    8. Conversational missing-capability response builder shapes (3 variants)
    9. Custom-seed initialization

Run directly: PYTHONPATH=. python3 tests/test_capabilities.py
"""

from __future__ import annotations

from src.capabilities import (
    Capability,
    CapabilityRegistry,
    CapabilityStatus,
    CapabilityType,
    CapabilityUsage,
    PermissionState,
)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

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
        fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ---------------------------------------------------------------------------
# 1. Default seed
# ---------------------------------------------------------------------------

@test("1.1 default registry seeds expected capabilities")
def test_default_seed():
    reg = CapabilityRegistry()
    for name in ("memory_v2", "graphify", "bridge", "web_search",
                 "github", "docs", "browser", "file_write",
                 "shell_exec", "send_message"):
        assert reg.get(name) is not None, f"missing seed: {name}"


@test("1.2 unknown capability returns None / empty / False cleanly")
def test_unknown():
    reg = CapabilityRegistry()
    assert reg.get("foobar") is None
    assert reg.describe("foobar") == ""
    assert reg.permission("foobar") is None
    assert reg.is_available("foobar") is False
    assert reg.is_absent_by_design("foobar") is False
    assert reg.needs_consent("foobar") is False


# ---------------------------------------------------------------------------
# 2. Query subsets
# ---------------------------------------------------------------------------

@test("2.1 available() includes graphify and bridge")
def test_available_query():
    reg = CapabilityRegistry()
    names = {c.name for c in reg.available()}
    assert "graphify" in names
    assert "bridge" in names


@test("2.2 missing() includes web_search, github, docs, browser, memory_v2")
def test_missing_query():
    reg = CapabilityRegistry()
    names = {c.name for c in reg.missing()}
    for expected in ("web_search", "github", "docs", "browser", "memory_v2"):
        assert expected in names, f"expected missing: {expected}"


@test("2.3 absent_by_design() includes all write capabilities")
def test_absent_query():
    reg = CapabilityRegistry()
    names = {c.name for c in reg.absent_by_design()}
    for expected in ("file_write", "shell_exec", "send_message"):
        assert expected in names, f"expected absent_by_design: {expected}"


# ---------------------------------------------------------------------------
# 3. State mutations + absent-by-design safety
# ---------------------------------------------------------------------------

@test("3.1 mark_available flips MISSING → AVAILABLE")
def test_mark_available():
    reg = CapabilityRegistry()
    reg.mark_available("web_search")
    assert reg.is_available("web_search")


@test("3.2 mark_missing flips AVAILABLE → MISSING")
def test_mark_missing():
    reg = CapabilityRegistry()
    reg.mark_missing("graphify")
    assert reg.get("graphify").status == CapabilityStatus.MISSING


@test("3.3 SAFETY: mark_available cannot flip ABSENT_BY_DESIGN")
def test_safety_mark_available():
    reg = CapabilityRegistry()
    reg.mark_available("file_write")
    assert reg.get("file_write").status == CapabilityStatus.ABSENT_BY_DESIGN


@test("3.4 SAFETY: mark_missing cannot flip ABSENT_BY_DESIGN")
def test_safety_mark_missing():
    reg = CapabilityRegistry()
    reg.mark_missing("shell_exec")
    assert reg.get("shell_exec").status == CapabilityStatus.ABSENT_BY_DESIGN


# ---------------------------------------------------------------------------
# 4. Permission lifecycle
# ---------------------------------------------------------------------------

@test("4.1 grant_permission: ASK_ONCE_PENDING → ASK_ONCE_GRANTED")
def test_grant():
    reg = CapabilityRegistry()
    reg.grant_permission("web_search")
    assert reg.permission("web_search") == PermissionState.ASK_ONCE_GRANTED


@test("4.2 deny_permission sets DENIED")
def test_deny():
    reg = CapabilityRegistry()
    reg.deny_permission("github")
    assert reg.permission("github") == PermissionState.DENIED


@test("4.3 SAFETY: grant cannot flip ABSENT_BY_DESIGN permission")
def test_grant_absent_safety():
    reg = CapabilityRegistry()
    reg.grant_permission("file_write")
    assert reg.permission("file_write") == PermissionState.ABSENT_BY_DESIGN


@test("4.4 SAFETY: deny cannot flip ABSENT_BY_DESIGN permission")
def test_deny_absent_safety():
    reg = CapabilityRegistry()
    reg.deny_permission("send_message")
    assert reg.permission("send_message") == PermissionState.ABSENT_BY_DESIGN


@test("4.5 reset_permission: external read → ASK_ONCE_PENDING")
def test_reset_external():
    reg = CapabilityRegistry()
    reg.grant_permission("web_search")
    reg.reset_permission("web_search")
    assert reg.permission("web_search") == PermissionState.ASK_ONCE_PENDING


@test("4.6 reset_permission: local read → ALWAYS_ALLOWED")
def test_reset_local():
    reg = CapabilityRegistry()
    reg.deny_permission("memory_v2")
    reg.reset_permission("memory_v2")
    assert reg.permission("memory_v2") == PermissionState.ALWAYS_ALLOWED


# ---------------------------------------------------------------------------
# 5. Usage logging
# ---------------------------------------------------------------------------

@test("5.1 log_usage records successful firing")
def test_log_usage():
    reg = CapabilityRegistry()
    reg.log_usage("github", purpose="checking PR #482")
    hist = reg.usage_history("github")
    assert len(hist) == 1
    assert hist[0].purpose == "checking PR #482"
    assert hist[0].success is True


@test("5.2 log_usage with error records failure")
def test_log_usage_failure():
    reg = CapabilityRegistry()
    reg.log_usage("web_search", purpose="...", success=False, error="rate limited")
    hist = reg.usage_history()
    assert hist[0].success is False
    assert hist[0].error == "rate limited"


@test("5.3 usage_history filters by capability name")
def test_usage_history_filter():
    reg = CapabilityRegistry()
    reg.log_usage("github", purpose="A")
    reg.log_usage("web_search", purpose="B")
    reg.log_usage("github", purpose="C")
    assert len(reg.usage_history("github")) == 2
    assert len(reg.usage_history("web_search")) == 1
    assert len(reg.usage_history()) == 3


@test("5.4 clear_usage_log empties history")
def test_clear_usage():
    reg = CapabilityRegistry()
    reg.log_usage("github", purpose="x")
    reg.clear_usage_log()
    assert reg.usage_history() == []


# ---------------------------------------------------------------------------
# 6. can_fire decision logic
# ---------------------------------------------------------------------------

@test("6.1 can_fire=True for local-read AVAILABLE + ALWAYS_ALLOWED")
def test_can_fire_local():
    reg = CapabilityRegistry()
    allowed, _ = reg.can_fire("graphify")
    assert allowed is True


@test("6.2 can_fire=False for MISSING capability")
def test_can_fire_missing():
    reg = CapabilityRegistry()
    allowed, reason = reg.can_fire("web_search")
    assert allowed is False
    assert "not connected" in reason


@test("6.3 can_fire=False for ABSENT_BY_DESIGN")
def test_can_fire_absent():
    reg = CapabilityRegistry()
    allowed, reason = reg.can_fire("file_write")
    assert allowed is False
    assert "absent by design" in reason


@test("6.4 can_fire=True after mark_available + grant_permission")
def test_can_fire_after_grant():
    reg = CapabilityRegistry()
    reg.mark_available("github")
    reg.grant_permission("github")
    allowed, _ = reg.can_fire("github")
    assert allowed is True


@test("6.5 can_fire=False after deny even if AVAILABLE")
def test_can_fire_denied():
    reg = CapabilityRegistry()
    reg.mark_available("github")
    reg.deny_permission("github")
    allowed, reason = reg.can_fire("github")
    assert allowed is False
    assert "denied" in reason


@test("6.6 can_fire=False for unknown capability")
def test_can_fire_unknown():
    reg = CapabilityRegistry()
    allowed, reason = reg.can_fire("foobar")
    assert allowed is False
    assert "not registered" in reason


# ---------------------------------------------------------------------------
# 7. needs_consent
# ---------------------------------------------------------------------------

@test("7.1 needs_consent=True for ASK_ONCE_PENDING")
def test_needs_consent_pending():
    reg = CapabilityRegistry()
    assert reg.needs_consent("web_search") is True


@test("7.2 needs_consent=False for ALWAYS_ALLOWED (local read)")
def test_needs_consent_local():
    reg = CapabilityRegistry()
    assert reg.needs_consent("graphify") is False


@test("7.3 needs_consent=False after grant")
def test_needs_consent_after_grant():
    reg = CapabilityRegistry()
    reg.grant_permission("web_search")
    assert reg.needs_consent("web_search") is False


# ---------------------------------------------------------------------------
# 8. Conversational missing-capability response builder
# ---------------------------------------------------------------------------

@test("8.1 build_missing_capability_response: connectable MCP shape is complete")
def test_build_response_connectable():
    reg = CapabilityRegistry()
    resp = reg.build_missing_capability_response(
        "web_search",
        why_needed="checking current Stripe pricing",
        current_confidence=0.65,
        confidence_if_connected=0.9,
        fallback_caveat="working from 2025 training data",
    )
    assert resp["capability"] == "web_search"
    assert resp["why_needed"] == "checking current Stripe pricing"
    assert resp["current_confidence"] == 0.65
    assert resp["confidence_if_connected"] == 0.9
    assert resp["fallback_caveat"] == "working from 2025 training data"
    # Connect option present + carries install hint
    connect = [o for o in resp["user_options"] if o["id"] == "connect"]
    assert len(connect) == 1
    assert connect[0]["instruction"] == "claude mcp add web_search"
    # Other options present
    ids = {o["id"] for o in resp["user_options"]}
    assert "fallback" in ids and "skip" in ids


@test("8.2 build_response for ABSENT_BY_DESIGN: no install option, explanation set")
def test_build_response_absent():
    reg = CapabilityRegistry()
    resp = reg.build_missing_capability_response(
        "file_write",
        why_needed="user asked us to edit a file",
    )
    assert resp.get("absent_by_design") is True
    assert resp["user_options"] == []
    assert "thinking partner" in resp["explanation"]


@test("8.3 build_response for unknown capability returns error shape")
def test_build_response_unknown():
    reg = CapabilityRegistry()
    resp = reg.build_missing_capability_response("foobar", why_needed="x")
    assert resp.get("error") == "unknown_capability"


@test("8.4 build_response for capability with no install_hint omits connect option")
def test_build_response_no_hint():
    # Custom capability without install_hint
    custom = [Capability(
        name="custom_thing",
        type=CapabilityType.EXTERNAL_READ,
        status=CapabilityStatus.MISSING,
        permission=PermissionState.ASK_ONCE_PENDING,
        description="A thing.",
        install_hint="",
    )]
    reg = CapabilityRegistry(capabilities=custom)
    resp = reg.build_missing_capability_response("custom_thing", why_needed="x")
    ids = {o["id"] for o in resp["user_options"]}
    assert "connect" not in ids  # no hint → no connect option
    assert "fallback" in ids and "skip" in ids


# ---------------------------------------------------------------------------
# 9. Custom seed
# ---------------------------------------------------------------------------

@test("9.1 custom capabilities seed replaces defaults")
def test_custom_seed():
    custom = [Capability(
        name="custom_test",
        type=CapabilityType.LOCAL_READ,
        status=CapabilityStatus.AVAILABLE,
        permission=PermissionState.ALWAYS_ALLOWED,
        description="just for testing",
    )]
    reg = CapabilityRegistry(capabilities=custom)
    assert len(reg.all()) == 1
    assert reg.get("custom_test") is not None
    assert reg.get("memory_v2") is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_default_seed,
    test_unknown,
    test_available_query,
    test_missing_query,
    test_absent_query,
    test_mark_available,
    test_mark_missing,
    test_safety_mark_available,
    test_safety_mark_missing,
    test_grant,
    test_deny,
    test_grant_absent_safety,
    test_deny_absent_safety,
    test_reset_external,
    test_reset_local,
    test_log_usage,
    test_log_usage_failure,
    test_usage_history_filter,
    test_clear_usage,
    test_can_fire_local,
    test_can_fire_missing,
    test_can_fire_absent,
    test_can_fire_after_grant,
    test_can_fire_denied,
    test_can_fire_unknown,
    test_needs_consent_pending,
    test_needs_consent_local,
    test_needs_consent_after_grant,
    test_build_response_connectable,
    test_build_response_absent,
    test_build_response_unknown,
    test_build_response_no_hint,
    test_custom_seed,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} capability registry tests...")
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
