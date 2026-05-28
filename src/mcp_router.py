"""
MCP Router — uniform interface for firing MCP tools.

CONTRACT
========
Every MCP capability — local read, external read, future third-party — is
fired through fire_mcp(). The router is the single place that:

    1. Consults CapabilityRegistry.can_fire() before any real work
    2. Dispatches to a real handler when one is registered for the name
    3. Falls back to a stub result when no handler is wired yet
    4. Logs every attempt (successful or blocked) into registry.usage_log
    5. Sets notified_user=True for external-read fires (transparency UI)

TWO-TIER PERMISSION (locked design, enforced here)
==================================================
    LOCAL_READ + ALWAYS_ALLOWED        → fires silently
    EXTERNAL_READ + ASK_ONCE_PENDING   → blocked (caller must get consent)
    EXTERNAL_READ + ASK_ONCE_GRANTED   → fires + sets notified_user
    *  + ABSENT_BY_DESIGN              → blocked structurally
    *  + DENIED                        → blocked

REAL DISPATCH vs STUB
=====================
fire_mcp() accepts an optional `handlers: McpHandlerRegistry`. When the
registry has a registered handler for the capability name:

    - The handler is called with (args, purpose) and its return value is
      placed in McpResult.result. stub=False.
    - Handler exceptions become ok=False with the error in blocked_reason.

When no handler is registered for the name (or handlers=None):

    - We return a stub McpResult with ok=True + stub=True. This preserves
      the pre-Phase-B contract — capabilities marked AVAILABLE that aren't
      wired through a handler yet still "succeed" the permission gate, just
      with placeholder data. Lets us roll out real handlers one capability
      at a time without breaking call sites.

GRADUATION PATH TO MCP-SPEC PROTOCOL
====================================
Handlers are plain async callables of shape (args: dict, purpose: str) ->
dict. To later swap a handler for a real MCP-server-over-stdio client, only
that handler implementation changes. The contract — the dict in, the dict
out — is intentionally compatible with the MCP tool-call shape, so the
migration is local.

ISOLATION: imports only from src.capabilities. No engine, no LLM client,
no triage. Pure capability-routing logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from src.capabilities import CapabilityRegistry
from src.capabilities.registry import CapabilityType


log = logging.getLogger("constellax.mcp_router")


# Handler shape — see module docstring.
#
# Implementations should:
#   - Be async
#   - Take (args: dict, purpose: str) and return a dict
#   - Treat empty args as a valid case (default behavior)
#   - Raise on hard failures; fire_mcp() converts exceptions to ok=False
#
# The returned dict is opaque to the router. Callers (dispatcher) decide
# how to render it into prompt context. A "text" key, if present, is the
# canonical human-readable summary the dispatcher uses for prompt
# injection — handlers should populate it when possible.
McpHandler = Callable[[dict, str], Awaitable[dict]]


class McpHandlerRegistry:
    """Maps capability name → registered async handler.

    Constructed once at server startup; passed through dispatch() into
    fire_mcp(). Per-request usage is read-only after construction, so a
    single instance is safe to share across concurrent requests.

    Servers without any MCP backends wired can pass `None` instead of
    constructing an empty registry — fire_mcp treats both the same.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, McpHandler] = {}

    def register(self, name: str, handler: McpHandler) -> None:
        """Bind a handler for a capability name. Last-write-wins on rebind."""
        if not name:
            raise ValueError("handler name must be non-empty")
        if not callable(handler):
            raise TypeError(f"handler for {name!r} must be callable")
        self._handlers[name] = handler

    def get(self, name: str) -> McpHandler | None:
        return self._handlers.get(name)

    def has(self, name: str) -> bool:
        return name in self._handlers

    def names(self) -> list[str]:
        return list(self._handlers.keys())

    def extend(self) -> "McpHandlerRegistry":
        """Return a shallow copy with the same handlers.

        Used by per-request wiring (e.g. filesystem, which closes over
        the request's attached_files payload) so the request-scoped
        handler doesn't mutate the shared module-level registry.
        """
        new = McpHandlerRegistry()
        new._handlers = dict(self._handlers)
        return new


@dataclass
class McpResult:
    """Outcome of a single MCP fire attempt.

    `result` carries the handler's return dict on success. When a stub was
    returned (no real handler wired yet), `stub=True` and `result` contains
    a placeholder note. The dispatcher distinguishes the two by stub flag —
    stub results are not injected into prompts.
    """
    ok: bool
    capability: str
    result: dict = field(default_factory=dict)
    blocked_reason: str = ""
    stub: bool = False              # True when result is a placeholder
    notified_user: bool = False     # True when this fire warrants a UI notification


async def fire_mcp(
    registry: CapabilityRegistry,
    name: str,
    purpose: str,
    args: dict | None = None,
    handlers: McpHandlerRegistry | None = None,
) -> McpResult:
    """
    Attempt to fire one MCP capability.

    Returns McpResult with:
      - ok=True + stub=False + real `result` data, when a handler is
        registered for this name and runs cleanly
      - ok=True + stub=True, when AVAILABLE + permitted but no handler
        is wired yet (preserves pre-Phase-B contract)
      - ok=False + blocked_reason, when can_fire blocks the call OR the
        handler raises an exception

    Every attempt — successful, stubbed, or blocked — is logged into the
    registry's usage history so the dispatcher can surface "checking
    GitHub for #482" style notifications.

    The notified_user flag flips True for fires that crossed an
    external-read boundary, i.e. the user should see a transparency
    notice. This applies to both real and stubbed fires.
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

    notified = cap is not None and cap.type == CapabilityType.EXTERNAL_READ

    # Real dispatch path — handler is registered for this capability.
    handler = handlers.get(name) if handlers is not None else None
    if handler is not None:
        try:
            result_data = await handler(args, purpose)
        except Exception as e:  # noqa: BLE001 — handler failures must not crash dispatch
            log.warning("MCP handler %r raised: %s", name, e)
            registry.log_usage(name, purpose, success=False, error=str(e))
            return McpResult(
                ok=False,
                capability=name,
                blocked_reason=f"{name} handler failed: {e}",
            )

        if not isinstance(result_data, dict):
            # Defensive: handlers must return dicts (matches MCP shape). A
            # broken handler producing a non-dict is treated as a failure,
            # not a crash.
            log.warning(
                "MCP handler %r returned non-dict (%s); treating as failure",
                name, type(result_data).__name__,
            )
            registry.log_usage(name, purpose, success=False, error="non-dict handler result")
            return McpResult(
                ok=False,
                capability=name,
                blocked_reason=f"{name} handler returned non-dict result",
            )

        registry.log_usage(name, purpose, success=True)
        return McpResult(
            ok=True,
            capability=name,
            result=result_data,
            stub=False,
            notified_user=notified,
        )

    # Stub fallback — capability is AVAILABLE + permitted but no real
    # handler is registered yet. Lets callers exercise the contract
    # incrementally as handlers are wired in over Phase B follow-ups.
    registry.log_usage(name, purpose, success=True)
    return McpResult(
        ok=True,
        capability=name,
        result={
            "stub": True,
            "capability": name,
            "purpose": purpose,
            "args": args,
            "note": (
                "MCP router permitted this fire but no real handler is "
                "registered yet. Register one via McpHandlerRegistry."
            ),
        },
        stub=True,
        notified_user=notified,
    )


async def fire_mcps(
    registry: CapabilityRegistry,
    requests: list[dict],
    handlers: McpHandlerRegistry | None = None,
) -> list[McpResult]:
    """
    Fire a list of MCP requests in sequence.

    Each request dict: {"name": str, "purpose": str, "args": dict (optional)}.
    Returns one McpResult per request in the same order.

    Sequential (not asyncio.gather) on purpose — keeps the usage log
    order deterministic, which the UI uses for "checking X then Y..."
    progressive notifications. If a future MCP genuinely needs parallel
    fan-out (e.g., GitHub + Context7 for the same question), add a
    separate parallel helper rather than changing this one's contract.
    """
    results: list[McpResult] = []
    for req in requests:
        results.append(await fire_mcp(
            registry=registry,
            name=req["name"],
            purpose=req.get("purpose", ""),
            args=req.get("args"),
            handlers=handlers,
        ))
    return results


# ---------------------------------------------------------------------------
# Prompt formatting helper — used by the dispatcher to fold MCP results
# into the system prompt. Stub results are filtered out.
# ---------------------------------------------------------------------------


def format_mcp_results_for_prompt(results: list[McpResult]) -> str:
    """Render the real-data MCP results into a single prompt block.

    Stubs and blocked fires are skipped — only `ok=True` + `stub=False`
    handler outputs land in the prompt. The block is empty string when
    no real results are present, so the caller can simply concatenate
    without conditional checks.

    Output shape (markdown, matches the PRIOR MEMORY block style):

        ## MCP CONTEXT
        (Live capability output fired this turn. Use it the same way
        you'd use a tool result — ground your answer in it, never
        fabricate references beyond what's shown.)

        ### {capability_name} — {purpose}
        {handler's "text" field, or a JSON dump as fallback}

    """
    real = [r for r in results if r.ok and not r.stub and r.result]
    if not real:
        return ""

    parts = [
        "## MCP CONTEXT",
        "(Live capability output fired this turn. Use it the same way "
        "you'd use a tool result — ground your answer in it, never "
        "fabricate references beyond what's shown.)",
        "",
    ]
    for r in real:
        purpose = (r.result.get("_purpose") or "").strip()
        header = f"### {r.capability}"
        if purpose:
            header = f"{header} — {purpose}"
        parts.append(header)

        text = r.result.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
        else:
            # Fallback: dump the dict (minus internal _purpose key) so the
            # model still has structured data even when the handler didn't
            # produce a text summary.
            clean = {k: v for k, v in r.result.items() if k != "_purpose"}
            parts.append(f"```\n{clean}\n```")
        parts.append("")

    return "\n".join(parts).rstrip()
