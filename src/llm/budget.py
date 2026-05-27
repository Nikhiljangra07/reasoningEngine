"""
Budget enforcer — per-request caps on wall time, cost, iterations, MCP calls.

Why this exists:
    AUTO mode + always-allowed local reads + unattended sessions means the
    engine could in principle run indefinitely on a single request. The
    no-edit rule keeps that safe from a destructiveness perspective, but
    not from a *financial* perspective. The budget enforcer is the
    bankruptcy floor.

Caps (locked design, anchored to real benchmarks):
    max_iterations      = 12     (engine's own MAX_ITERATIONS ceiling)
    max_wall_time_sec   = 720    (12 min — fits a full AUTO run with breathing room)
    max_cost_usd        = 1.00   (~2x typical AUTO cost on the cost-conscious model map)
    max_mcp_calls       = 20     (stops external API thrash)

Lifecycle:
    tracker = BudgetTracker(caps=BudgetCaps(...))
    for iteration in range(MAX):
        check = tracker.check()
        if not check.allowed:
            break  # graceful stop; caller returns partial result + breach note
        # ... do work ...
        response = await client.call(...)
        tracker.record_llm_response(response)
        tracker.increment_iteration()
    if tracker.breached:
        # Surface "I hit the cap; here's what I have" in the response

The tracker is per-request. The route dispatcher (Step 4) constructs one
per `/api/trace` call and threads it through to the engine and MCP router.

ISOLATION: imports only from src.llm.provider_map for pricing. No engine,
no bridge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.llm.provider_map import get_pricing


# ---------------------------------------------------------------------------
# Caps and state
# ---------------------------------------------------------------------------

@dataclass
class BudgetCaps:
    """Per-request budget caps. Defaults match the locked design."""
    max_iterations: int = 12
    max_wall_time_sec: float = 720.0   # 12 minutes
    max_cost_usd: float = 1.00
    max_mcp_calls: int = 20


@dataclass
class BudgetState:
    """Live counters maintained by BudgetTracker."""
    started_at: float                  # time.monotonic() snapshot at construction
    iterations: int = 0
    cost_usd: float = 0.0
    mcp_calls: int = 0
    breached: bool = False
    breach_reason: str = ""


@dataclass
class BudgetCheck:
    """Result of BudgetTracker.check()."""
    allowed: bool
    reason: str = ""
    remaining_iterations: int = 0
    remaining_wall_time_sec: float = 0.0
    remaining_cost_usd: float = 0.0
    remaining_mcp_calls: int = 0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class BudgetTracker:
    """
    Per-request budget tracking with pre-check enforcement.

    Callers invoke `check()` BEFORE each iteration / LLM call / MCP call.
    If `check.allowed` is False, callers must stop and return whatever
    partial result they have, surfacing the breach reason to the user.

    The tracker never raises. It returns False and lets the caller decide.
    """

    def __init__(self, caps: BudgetCaps | None = None):
        self.caps = caps or BudgetCaps()
        self.state = BudgetState(started_at=time.monotonic())

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    def elapsed_sec(self) -> float:
        return time.monotonic() - self.state.started_at

    @property
    def breached(self) -> bool:
        return self.state.breached

    # -----------------------------------------------------------------------
    # Pre-check (call this before every iteration / LLM call / MCP call)
    # -----------------------------------------------------------------------

    def check(self) -> BudgetCheck:
        """
        Return a BudgetCheck describing whether work can continue.

        Order of checks: wall time → cost → iterations → MCP calls.
        First cap to breach wins. Once breached, the tracker stays breached
        until reset.
        """
        elapsed = self.elapsed_sec()
        remaining = self._remaining(elapsed)

        if elapsed >= self.caps.max_wall_time_sec:
            return self._mark_breach(
                f"wall time exceeded: {elapsed:.1f}s >= {self.caps.max_wall_time_sec:.0f}s",
                remaining,
            )
        if self.state.cost_usd >= self.caps.max_cost_usd:
            return self._mark_breach(
                f"cost exceeded: ${self.state.cost_usd:.4f} >= ${self.caps.max_cost_usd:.2f}",
                remaining,
            )
        if self.state.iterations >= self.caps.max_iterations:
            return self._mark_breach(
                f"iteration cap reached: {self.state.iterations}/{self.caps.max_iterations}",
                remaining,
            )
        if self.state.mcp_calls >= self.caps.max_mcp_calls:
            return self._mark_breach(
                f"MCP call cap reached: {self.state.mcp_calls}/{self.caps.max_mcp_calls}",
                remaining,
            )

        return BudgetCheck(
            allowed=True,
            remaining_iterations=remaining["iterations"],
            remaining_wall_time_sec=remaining["wall_time_sec"],
            remaining_cost_usd=remaining["cost_usd"],
            remaining_mcp_calls=remaining["mcp_calls"],
        )

    def _remaining(self, elapsed: float) -> dict:
        return {
            "iterations": max(0, self.caps.max_iterations - self.state.iterations),
            "wall_time_sec": max(0.0, self.caps.max_wall_time_sec - elapsed),
            "cost_usd": max(0.0, self.caps.max_cost_usd - self.state.cost_usd),
            "mcp_calls": max(0, self.caps.max_mcp_calls - self.state.mcp_calls),
        }

    def _mark_breach(self, reason: str, remaining: dict) -> BudgetCheck:
        self.state.breached = True
        self.state.breach_reason = reason
        return BudgetCheck(
            allowed=False,
            reason=reason,
            remaining_iterations=remaining["iterations"],
            remaining_wall_time_sec=remaining["wall_time_sec"],
            remaining_cost_usd=remaining["cost_usd"],
            remaining_mcp_calls=remaining["mcp_calls"],
        )

    # -----------------------------------------------------------------------
    # Recording (call after each LLM / MCP call completes)
    # -----------------------------------------------------------------------

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Record an LLM call's cost based on provider_map.PRICING.
        Returns the cost delta added (USD).
        """
        in_price, out_price = get_pricing(model)
        cost = (input_tokens * in_price / 1_000_000) + (
            output_tokens * out_price / 1_000_000
        )
        if cost < 0:
            cost = 0.0
        self.state.cost_usd += cost
        return cost

    def record_llm_response(self, response) -> float:
        """
        Convenience: record cost from an LLMResponse-shaped object.

        Accepts any object with `.model`, `.input_tokens`, `.output_tokens`,
        and `.success` attributes (matches src.llm.client.LLMResponse).
        Failed calls (success=False) record zero cost.
        """
        if not getattr(response, "success", False):
            return 0.0
        return self.record_llm_call(
            getattr(response, "model", "") or "",
            int(getattr(response, "input_tokens", 0) or 0),
            int(getattr(response, "output_tokens", 0) or 0),
        )

    def record_mcp_call(self, name: str = "") -> None:
        """Record one MCP firing. The name is for the caller's logging, not stored here."""
        self.state.mcp_calls += 1

    def increment_iteration(self) -> None:
        self.state.iterations += 1

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Serializable summary for inclusion in the API response.
        Surfaced to the user alongside the engine's output so they always
        see what was spent.
        """
        return {
            "iterations": self.state.iterations,
            "wall_time_sec": round(self.elapsed_sec(), 2),
            "cost_usd": round(self.state.cost_usd, 4),
            "mcp_calls": self.state.mcp_calls,
            "breached": self.state.breached,
            "breach_reason": self.state.breach_reason,
            "caps": {
                "max_iterations": self.caps.max_iterations,
                "max_wall_time_sec": self.caps.max_wall_time_sec,
                "max_cost_usd": self.caps.max_cost_usd,
                "max_mcp_calls": self.caps.max_mcp_calls,
            },
        }
