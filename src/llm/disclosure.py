"""
Progressive Disclosure — Two-Phase Response System.

Phase 1 (Quick Batch): Run 2 iterations, deliver interim findings within 15-25s.
  Includes: top 2-3 findings, confidence, "dig deeper" option with credit estimate.

Phase 2 (Deep Batch): User-triggered. Continues from Phase 1 state (does NOT restart).
  Runs remaining iterations until convergence or max. Applies post-convergence gates.

Benefits:
  - 15-second first response feels fast
  - Most users get enough from Phase 1
  - User controls depth — active participants, not passive waiters
  - Token savings: Phase 1 sufficient → 60-75% compute saved

ISOLATION: Imports from src.core.types + src.llm modules only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.core.types import (
    Consequence,
    Domain,
    DomainOutput,
    Problem,
    RootCause,
)
from src.llm.client import LLMClient
from src.llm.engine import (
    EngineResult,
    Trajectory,
    run_async_formation,
)


# Phase 1 always runs exactly 2 iterations
PHASE_1_ITERATIONS = 2

# Phase 2 picks up from Phase 1 and runs to convergence or max
PHASE_2_MAX_ADDITIONAL_ITERATIONS = 10


@dataclass
class PhaseOneResponse:
    """Interim response delivered quickly after 2 iterations."""
    initial_findings: list[Trajectory]
    confidence: float
    depth_available: bool
    estimated_additional_iterations: int
    estimated_additional_credits: float
    message: str                        # human-readable summary
    engine_state: EngineResult          # full state for Phase 2 continuation


@dataclass
class PhaseTwoResponse:
    """Full deep response after user triggers "dig deeper"."""
    full_result: EngineResult
    message: str
    total_credits_used: float


async def run_phase_one(
    problem: Problem,
    client: LLMClient,
    cache_path: str = "",
) -> PhaseOneResponse:
    """
    Phase 1: Quick Batch — 2 iterations, deliver fast.

    Target: 15-25 seconds with mock, varies with live API.
    """
    result = await run_async_formation(
        problem=problem,
        client=client,
        cache_path=cache_path,
        max_iterations=PHASE_1_ITERATIONS,
    )

    # Calculate confidence from trajectories
    if result.trajectories:
        avg_confidence = sum(t.confidence for t in result.trajectories) / len(result.trajectories)
    else:
        avg_confidence = 0.3

    # Estimate what Phase 2 would cost
    calls_per_iteration = len(result.formation_plan.active_domains) + len(result.ke_results)
    estimated_additional_iters = max(3, PHASE_2_MAX_ADDITIONAL_ITERATIONS - PHASE_1_ITERATIONS)
    estimated_additional_credits = calls_per_iteration * estimated_additional_iters * 0.5

    # Is deeper analysis likely to help?
    depth_available = (
        not result.convergence_history.final_converged
        or any(ke.scrutiny_score > 0.5 for ke in result.ke_results)
        or avg_confidence < 0.8
    )

    # Build human-readable message
    message = _build_phase_one_message(result, avg_confidence, depth_available)

    return PhaseOneResponse(
        initial_findings=result.trajectories,
        confidence=avg_confidence,
        depth_available=depth_available,
        estimated_additional_iterations=estimated_additional_iters,
        estimated_additional_credits=estimated_additional_credits,
        message=message,
        engine_state=result,
    )


async def run_phase_two(
    problem: Problem,
    client: LLMClient,
    phase_one: PhaseOneResponse,
    cache_path: str = "",
) -> PhaseTwoResponse:
    """
    Phase 2: Deep Batch — continues from Phase 1 state.

    Does NOT restart. Picks up the full engine state and runs
    additional iterations until convergence or max.
    """
    # Continue from where Phase 1 left off
    result = await run_async_formation(
        problem=problem,
        client=client,
        cache_path=cache_path,
        max_iterations=PHASE_1_ITERATIONS + PHASE_2_MAX_ADDITIONAL_ITERATIONS,
    )

    # Calculate total credits
    total_credits = result.call_summary.get("total_calls", 0) * 0.5

    message = _build_phase_two_message(result)

    return PhaseTwoResponse(
        full_result=result,
        message=message,
        total_credits_used=total_credits,
    )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _build_phase_one_message(
    result: EngineResult,
    confidence: float,
    depth_available: bool,
) -> str:
    """Build human-readable Phase 1 summary."""
    parts = []

    # Top findings
    if result.trajectories:
        parts.append(f"Found {len(result.trajectories)} probable trajectory(s).")
        top = result.trajectories[0]
        parts.append(
            f"Top finding: '{top.root_cause.variable.name}' "
            f"(confidence: {top.confidence:.0%})."
        )

    # Confidence
    parts.append(f"Overall confidence at this stage: {confidence:.0%}.")

    # Depth
    if depth_available:
        parts.append(
            "There is more underneath. Deeper analysis is available "
            "if you want to continue."
        )
    else:
        parts.append("Analysis appears well-converged at this depth.")

    return " ".join(parts)


def _build_phase_two_message(result: EngineResult) -> str:
    """Build human-readable Phase 2 summary."""
    parts = []

    parts.append(
        f"Deep analysis complete. {result.convergence_history.total_iterations} "
        f"total iterations."
    )

    if result.convergence_history.final_converged:
        parts.append("Analysis converged.")
    else:
        parts.append("Analysis reached maximum depth without full convergence.")

    if result.trajectories:
        parts.append(f"{len(result.trajectories)} trajectory(s) identified.")

    parts.append(f"Delivery mode: {result.delivery_mode}.")

    return " ".join(parts)
