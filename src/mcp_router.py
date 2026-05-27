"""
MCP Router (stub) — uniform interface for firing MCP tools.

This is the architecture's last piece. It does NOT yet make real network
calls to external MCP servers. It provides the contract the rest of the
system codes against, so when the real MCP clients land later, the only
file that changes is this one.

What this stub does:
    1. Provides a uniform fire_mcp() / fire_mcps() API the dispatcher calls
    2. Checks CapabilityRegistry.can_fire() before attempting anything
    3. On AVAILABLE + allowed → returns a placeholder stub result
    4. On any block (MISSING, ABSENT_BY_DESIGN, DENIED, needs consent)
       → returns ok=False with the reason
    5. Logs every attempt into registry.usage_log

The two-tier permission model (locked design) is enforced here:
    - LOCAL_READ + ALWAYS_ALLOWED → fires silently
    - EXTERNAL_READ + ASK_ONCE_PENDING → blocked (caller must get consent)
    - EXTERNAL_READ + ASK_ONCE_GRANTED → fires + notifies
    - * + ABSENT_BY_DESIGN → blocked structurally, no permission concept
    - * + DENIED → blocked

ISOLATION: imports only from src.capabilities. No engine, no LLM client,
no triage. Pure capability-routing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.capabilities import CapabilityRegistry


@dataclass
class McpResult:
    """Outcome of a single MCP fire attempt."""
    ok: bool
    capability: str
    result: dict = field(default_factory=dict)
    blocked_reason: str = ""
    stub: bool = False              # True when result is a stub placeholder
    notified_user: bool = False     # True if this fire should produce a UI notification


async def fire_mcp(
    registry: CapabilityRegistry,
    name: str,
    purpose: str,
    args: dict | None = None,
) -> McpResult:
    """
    Attempt to fire one MCP capability.

    Returns McpResult with ok=True when AVAILABLE + permission allows;
    ok=False with blocked_reason otherwise.

    Every attempt — successful or blocked — is logged into the registry's
    usage history so the dispatcher can surface "checking GitHub for #482"
    style notifications. The notified_user flag flips True for fires that
    crossed an external-read boundary, i.e. the user should see a
    transparency notice.
    """
    args = args or {}
    allowed, reason = registry.can_fire(name)
    cap = registry.get(name)

    if not allowed:
        registry.log_usage(name, purpose, success=False, error=reason)
        return McpResult(
            ok=False,
            capability=name,
            blocked_reason=reason,
        )

    # Allowed → stub fire. Real network call lands later.
    registry.log_usage(name, purpose, success=True)

    # External-read fires get user-facing notification.
    from src.capabilities.registry import CapabilityType
    notified = (cap is not None and cap.type == CapabilityType.EXTERNAL_READ)

    return McpResult(
        ok=True,
        capability=name,
        result={
            "stub": True,
            "capability": name,
            "purpose": purpose,
            "args": args,
            "note": (
                "MCP router is stubbed — real network call lands in a future "
                "drop. Capability is registered and permitted; the wiring "
                "contract is in place."
            ),
        },
        stub=True,
        notified_user=notified,
    )


async def fire_mcps(
    registry: CapabilityRegistry,
    requests: list[dict],
) -> list[McpResult]:
    """
    Fire a list of MCP requests in sequence.

    Each request dict: {"name": str, "purpose": str, "args": dict (optional)}.
    Returns one McpResult per request in the same order.

    Real implementation will run these in parallel via asyncio.gather().
    The stub runs them sequentially since each call is essentially a
    registry lookup — no I/O to overlap.
    """
    results: list[McpResult] = []
    for req in requests:
        results.append(await fire_mcp(
            registry=registry,
            name=req["name"],
            purpose=req.get("purpose", ""),
            args=req.get("args"),
        ))
    return results
