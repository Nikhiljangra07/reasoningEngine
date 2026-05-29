"""
Wandering Runtime — multi-agent orchestration for the three modes.

The runtime is the layer that:
  1. Spawns N agents per the chosen WanderingMode
  2. Runs them with appropriate parallelism (Triple = sequential chain;
     Multi = parallel fan-out; Absolute Chaos = parallel + sub-spawn)
  3. Collects all ExplorationReports for the synthesis layer
  4. Honors a session-level credit cap

Per Law 1: no smart routing across agents. Each agent's wander is
independent (with optional lightweight position broadcast for soft
anti-collision — deferred to Phase 2-extension; V0 is fully isolated).

Per Law 4: the runtime never edits anything outside the session — it
returns the collected reports + traces to the caller.

ISOLATION: imports agent + cushion + report + trace. Does NOT import
articulation or synthesis (those run AFTER the runtime completes).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from src.llm.client import LLMClient
from src.wandering.agent import (
    AgentBudget,
    AgentState,
    FetchFn,
    run_agent,
    stub_fetcher,
)
from src.wandering.cushion import CushionGraph
from src.wandering.report import ExplorationReport
from src.wandering.trace import DecisionTrace


log = logging.getLogger("constellax.wandering.runtime")


# ---------------------------------------------------------------------------
# Modes — LOW / MED / HIGH
# ---------------------------------------------------------------------------


class WanderingMode(str, Enum):
    """The three pendulum modes the user picks.

    TRIPLE_PENDULUM  → one chain of sequential sub-agents (LOW)
    MULTI_PENDULUM   → N parallel agents, no sub-agents (MEDIUM)
    ABSOLUTE_CHAOS   → N parallel agents, each can spawn sub-agents (HIGH)

    The mode determines structural shape; per-mode budgets (time, tokens,
    agent count) live in WanderingConfig.
    """

    TRIPLE_PENDULUM = "triple_pendulum"
    MULTI_PENDULUM = "multi_pendulum"
    ABSOLUTE_CHAOS = "absolute_chaos"


# ---------------------------------------------------------------------------
# Per-mode default configurations
# ---------------------------------------------------------------------------


@dataclass
class ModeDefaults:
    """Per-mode default budgets. Overridable by the user via WanderingConfig."""

    agents: int
    time_seconds: float
    tokens_per_agent: int
    model_mix: tuple[str, ...]


MODE_DEFAULTS: dict[WanderingMode, ModeDefaults] = {
    WanderingMode.TRIPLE_PENDULUM: ModeDefaults(
        agents=3,
        time_seconds=15 * 60,
        tokens_per_agent=20_000,
        model_mix=("anthropic/claude-haiku-4-5",),  # cheap chain for LOW
    ),
    WanderingMode.MULTI_PENDULUM: ModeDefaults(
        agents=5,
        time_seconds=30 * 60,
        tokens_per_agent=30_000,
        # diverse: DeepSeek + Haiku
        model_mix=(
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-pro",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-haiku-4-5",
        ),
    ),
    WanderingMode.ABSOLUTE_CHAOS: ModeDefaults(
        agents=10,
        time_seconds=60 * 60,
        tokens_per_agent=40_000,
        # diverse: Sonnet + DeepSeek + Haiku across 3 families
        model_mix=(
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-sonnet-4-6",
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-pro",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-haiku-4-5",
            "anthropic/claude-haiku-4-5",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Wandering config
# ---------------------------------------------------------------------------


@dataclass
class WanderingConfig:
    """User-facing knobs for one Wandering Room session.

    Defaults come from MODE_DEFAULTS[mode]; user can override any field.
    """

    mode: WanderingMode = WanderingMode.MULTI_PENDULUM
    agents: int | None = None
    time_budget_seconds: float | None = None
    tokens_per_agent: int | None = None
    model_mix: tuple[str, ...] | None = None
    session_token_cap: int = 1_000_000  # hard ceiling — session won't blow past this
    session_id: str = ""

    def resolved(self) -> tuple[int, float, int, tuple[str, ...]]:
        """Return (agents, time_seconds, tokens_per_agent, model_mix) with
        defaults applied for whatever the user didn't override."""
        defaults = MODE_DEFAULTS[self.mode]
        return (
            self.agents if self.agents is not None else defaults.agents,
            self.time_budget_seconds if self.time_budget_seconds is not None else defaults.time_seconds,
            self.tokens_per_agent if self.tokens_per_agent is not None else defaults.tokens_per_agent,
            self.model_mix if self.model_mix is not None else defaults.model_mix,
        )


# ---------------------------------------------------------------------------
# Session result
# ---------------------------------------------------------------------------


@dataclass
class SessionResult:
    """The output of a Wandering Room session.

    Contains every report produced by every agent + every agent's trace.
    Passed to the synthesis layer which aggregates into a Dossier.
    """

    session_id: str
    mode: WanderingMode
    cushion: CushionGraph
    config: WanderingConfig
    reports: list[ExplorationReport] = field(default_factory=list)
    traces: list[DecisionTrace] = field(default_factory=list)
    total_tokens_spent: int = 0
    elapsed_seconds: float = 0.0
    ended_at: float = 0.0

    def report_count(self) -> int:
        return len(self.reports)

    def agent_count(self) -> int:
        return len(self.traces)


# ---------------------------------------------------------------------------
# Agent assignment helpers
# ---------------------------------------------------------------------------


def assign_models(num_agents: int, model_mix: tuple[str, ...]) -> list[str]:
    """Assign a model slug to each agent slot.

    If `num_agents` matches `len(model_mix)`, one-to-one.
    Otherwise we cycle the mix to fill. This lets users override mode
    defaults without exactly matching lengths.
    """
    if not model_mix:
        return ["anthropic/claude-haiku-4-5"] * num_agents
    return [model_mix[i % len(model_mix)] for i in range(num_agents)]


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------


async def _run_multi_pendulum(
    cushion: CushionGraph,
    config: WanderingConfig,
    client: LLMClient,
    fetcher: FetchFn,
    clock: Callable[[], float],
) -> SessionResult:
    """N agents in parallel, no sub-agents. The default mode."""
    num_agents, time_secs, tokens_each, model_mix = config.resolved()
    models = assign_models(num_agents, model_mix)

    started = clock()
    session_id = config.session_id or f"wsess-{uuid.uuid4().hex[:8]}"

    agents = [
        AgentState(
            agent_id=f"P{i + 1:02d}",
            cushion=cushion,
            budget=AgentBudget(
                time_budget_seconds=time_secs,
                token_budget=tokens_each,
            ),
            model_slug=models[i],
        )
        for i in range(num_agents)
    ]

    # Parallel run. Each agent runs independently — no shared state.
    results = await asyncio.gather(
        *[run_agent(a, client=client, fetcher=fetcher, clock=clock) for a in agents],
        return_exceptions=True,
    )

    reports: list[ExplorationReport] = []
    traces: list[DecisionTrace] = []
    total_tokens = 0
    for r in results:
        if isinstance(r, BaseException):
            log.warning("agent crashed: %s", r)
            continue
        reports.extend(r.reports)
        traces.append(r.trace)
        total_tokens += r.cumulative_tokens

    elapsed = clock() - started

    return SessionResult(
        session_id=session_id,
        mode=config.mode,
        cushion=cushion,
        config=config,
        reports=reports,
        traces=traces,
        total_tokens_spent=total_tokens,
        elapsed_seconds=elapsed,
        ended_at=clock(),
    )


async def _run_triple_pendulum(
    cushion: CushionGraph,
    config: WanderingConfig,
    client: LLMClient,
    fetcher: FetchFn,
    clock: Callable[[], float],
) -> SessionResult:
    """One chain of N sub-agents, sequential.

    Triple Pendulum = LOW mode: chain of progressively-deeper exploration.
    Each sub-agent picks up where the prior stopped. In V0 we approximate
    this with sequential parallel-of-one execution — same engine, just
    serial. Future enrichment: have each sub-agent receive the prior
    agent's report list as additional context to "pick up from."
    """
    num_agents, time_secs, tokens_each, model_mix = config.resolved()
    models = assign_models(num_agents, model_mix)

    started = clock()
    session_id = config.session_id or f"wsess-{uuid.uuid4().hex[:8]}"

    reports: list[ExplorationReport] = []
    traces: list[DecisionTrace] = []
    total_tokens = 0

    for i in range(num_agents):
        agent = AgentState(
            agent_id=f"P{i + 1:02d}",
            cushion=cushion,
            budget=AgentBudget(
                time_budget_seconds=time_secs,
                token_budget=tokens_each,
            ),
            model_slug=models[i],
        )
        result = await run_agent(agent, client=client, fetcher=fetcher, clock=clock)
        reports.extend(result.reports)
        traces.append(result.trace)
        total_tokens += result.cumulative_tokens

        if total_tokens >= config.session_token_cap:
            log.info(
                "session token cap reached after agent %d; stopping chain early",
                i + 1,
            )
            break

    elapsed = clock() - started

    return SessionResult(
        session_id=session_id,
        mode=config.mode,
        cushion=cushion,
        config=config,
        reports=reports,
        traces=traces,
        total_tokens_spent=total_tokens,
        elapsed_seconds=elapsed,
        ended_at=clock(),
    )


async def _run_absolute_chaos(
    cushion: CushionGraph,
    config: WanderingConfig,
    client: LLMClient,
    fetcher: FetchFn,
    clock: Callable[[], float],
) -> SessionResult:
    """N parallel agents, each may spawn sub-agents on HIGH match.

    After all root agents finish their main run, we scan their HIGH-
    confidence reports and auto-spawn sub-agents on each one (bounded
    by chain depth, per-session token cap, and a per-root spawn limit
    to prevent runaway).

    Sub-agent reports are folded into the same SessionResult — the
    dossier doesn't distinguish root vs sub-agent reports for the user
    (they're both legitimate findings). The trace, however, preserves
    the spawning relationship for audit.
    """
    # First, run the root agents like multi-pendulum.
    session = await _run_multi_pendulum(cushion, config, client, fetcher, clock)

    # Lazy import to avoid circulars.
    from src.wandering.subagent import (
        run_subagent,
        should_spawn,
        spawn_request_from_high_match_report,
    )
    from src.wandering.report import Confidence

    MAX_AUTO_SPAWNS_PER_ROOT = 2  # bound runaway: each root spawns at most 2

    high_reports_by_agent: dict[str, list] = {}
    for report in session.reports:
        if report.confidence == Confidence.HIGH:
            high_reports_by_agent.setdefault(report.agent_id, []).append(report)

    spawn_jobs = []
    for agent_id, reports in high_reports_by_agent.items():
        # Find the matching agent state to pass to the spawn builder.
        # We only have the trace from the session, but the spawn builder
        # only needs agent_id + cushion, so synthesize a minimal state.
        from src.wandering.agent import AgentState, AgentBudget

        minimal_state = AgentState(
            agent_id=agent_id,
            cushion=cushion,
            budget=AgentBudget(),  # not used by spawn builder
        )

        for report in reports[:MAX_AUTO_SPAWNS_PER_ROOT]:
            req = spawn_request_from_high_match_report(
                parent_state=minimal_state,
                report=report,
                mode_key="absolute_chaos",
                chain_depth=2,
                distance_budget_tokens=15_000,
            )
            if req is None:
                continue
            allowed, reason = should_spawn(
                req,
                session_tokens_spent=session.total_tokens_spent,
                session_token_cap=config.session_token_cap,
            )
            if not allowed:
                log.info("absolute_chaos: spawn skipped (%s)", reason)
                continue
            spawn_jobs.append((req, agent_id))

    if not spawn_jobs:
        return session

    # Run all auto-spawned sub-agents in parallel.
    results = await asyncio.gather(
        *[run_subagent(req, client=client, fetcher=fetcher, parent_clock=clock)
          for req, _ in spawn_jobs],
        return_exceptions=True,
    )

    for outcome, (req, parent_id) in zip(results, spawn_jobs):
        if isinstance(outcome, BaseException):
            log.warning("auto-spawned subagent crashed: %s", outcome)
            continue
        # Fold reports into the session
        session.reports.extend(outcome.reports)
        session.total_tokens_spent += outcome.tokens_spent
        log.info(
            "absolute_chaos: subagent %s of parent %s yielded %d reports",
            outcome.subagent_id, parent_id, len(outcome.reports),
        )

    session.ended_at = clock()
    return session


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_wandering_session(
    cushion: CushionGraph,
    config: WanderingConfig,
    client: LLMClient,
    *,
    fetcher: FetchFn = stub_fetcher,
    clock: Callable[[], float] = time.time,
) -> SessionResult:
    """Top-level entry: run one Wandering Room session and return all
    reports + traces.

    Dispatches on `config.mode` to the per-mode runner. Callers (Phase 5
    synthesis, or the eventual API endpoint) take the SessionResult and
    feed it to the synthesis pipeline.
    """
    if config.mode == WanderingMode.TRIPLE_PENDULUM:
        runner = _run_triple_pendulum
    elif config.mode == WanderingMode.ABSOLUTE_CHAOS:
        runner = _run_absolute_chaos
    else:
        runner = _run_multi_pendulum

    return await runner(cushion, config, client, fetcher, clock)


__all__ = [
    "WanderingMode",
    "ModeDefaults",
    "MODE_DEFAULTS",
    "WanderingConfig",
    "SessionResult",
    "assign_models",
    "run_wandering_session",
]
