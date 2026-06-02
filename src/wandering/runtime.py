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
from typing import Any, Callable

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
from src.wandering.session_state import SessionState
from src.wandering.trace import DecisionTrace


# ---------------------------------------------------------------------------
# Progress handle — live view into the wander, used by the abort path
# ---------------------------------------------------------------------------


@dataclass
class WanderingProgress:
    """Live progress reference for an in-flight wander.

    The caller (routes.py worker) constructs one of these, hands it to
    `run_wandering_session(progress=...)`, and stashes it in a per-session
    dict. The runtime registers each spawned agent here as it's created;
    `tokens_used` is a real-time sum across those agents' cumulative
    token counters.

    Used at abort time to compute an HONEST refund — we know exactly
    how many tokens the agents had spent at the moment of cancel, so
    the credit math doesn't rely on guesses (time-ratio, etc.) which
    would either over-charge or over-refund.

    Failures NEVER raise. If `progress` is None throughout, the wander
    runs exactly as before — this is opt-in observability for callers
    that need it.
    """

    agents: list[AgentState] = field(default_factory=list)

    def register(self, *new_agents: AgentState) -> None:
        """Add the given agents to the live progress set. Idempotent
        against duplicates by identity (an agent is registered once
        even if the callback fires twice in some edge case)."""
        seen = {id(a) for a in self.agents}
        for a in new_agents:
            if id(a) not in seen:
                self.agents.append(a)

    @property
    def tokens_used(self) -> int:
        """Sum of all agents' cumulative_tokens. Reads in-memory
        primitives only — safe to call from any context without
        awaiting."""
        return sum(int(a.cumulative_tokens) for a in self.agents)

    @property
    def reports_count(self) -> int:
        """Total reports finalized across all registered agents."""
        return sum(len(a.reports) for a in self.agents)

    @property
    def urls_visited(self) -> int:
        """Distinct URLs touched across the wander, read from the shared
        SessionState if any agent has one wired. Reads in-memory primitives
        only. Returns 0 when no SessionState was plumbed."""
        for a in self.agents:
            ss = a.session_state
            if ss is not None:
                try:
                    return len(ss.visited_urls)
                except Exception:
                    return 0
        return 0

    @property
    def followon_queue_size(self) -> int:
        """Depth of the shared follow-on queue if a SessionState is wired."""
        for a in self.agents:
            ss = a.session_state
            if ss is not None:
                try:
                    return len(ss.followon_queue)
                except Exception:
                    return 0
        return 0

    def live_state(self) -> list[dict[str, Any]]:
        """Snapshot per-agent live state for /status to surface. Pure
        in-memory reads — never awaits, never raises. The shape is the
        wire contract for the frontend's per-agent panel.

        Each entry:
          agent_id          — "P01", "P02", ... (display label)
          model_slug        — full model identifier in use
          tokens            — current cumulative_tokens
          reports_count     — reports finalized so far
          steps_taken       — agent's internal step counter
          current_phase     — last trace step's kind (e.g. "matched", "fetched")
          current_position  — last trace step's `position` (URL or domain)
          last_step_at      — timestamp of the last trace step
          discarded_count   — # clues this agent classified as discarded
        """
        out: list[dict[str, Any]] = []
        for a in self.agents:
            last = a.trace.steps[-1] if a.trace.steps else None
            out.append({
                "agent_id":         a.agent_id,
                "model_slug":       a.model_slug,
                "tokens":           int(a.cumulative_tokens),
                "reports_count":    len(a.reports),
                "steps_taken":      int(a.steps_taken),
                "current_phase":    last.kind.value if last is not None else "initialized",
                "current_position": (last.position if last is not None else "")[:120],
                "last_step_at":     float(last.timestamp) if last is not None else 0.0,
                "discarded_count":  len(a.trace.discarded_clues),
            })
        return out


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

    `session_state` carries the shared per-wander dedup set + follow-on
    queue. Stored here so absolute_chaos sub-agents can inherit the
    parent wander's visited URLs (avoids re-fetching pages already read).
    Set None when no SessionState was created (legacy callers).
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
    session_state: SessionState | None = None

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
    progress: "WanderingProgress | None" = None,
) -> SessionResult:
    """N agents in parallel, no sub-agents. The default mode."""
    num_agents, time_secs, tokens_each, model_mix = config.resolved()
    models = assign_models(num_agents, model_mix)

    started = clock()
    session_id = config.session_id or f"wsess-{uuid.uuid4().hex[:8]}"

    # One SessionState per wander, shared across all agents — drives
    # cross-agent URL dedup and the follow-on queue. See
    # WANDERING_ROOM_DECISIONS.md D8 (dedup) and D9 (follow-on).
    session_state = SessionState(session_id=session_id)

    agents = [
        AgentState(
            agent_id=f"P{i + 1:02d}",
            cushion=cushion,
            budget=AgentBudget(
                time_budget_seconds=time_secs,
                token_budget=tokens_each,
            ),
            model_slug=models[i],
            session_state=session_state,
        )
        for i in range(num_agents)
    ]

    # Register with the live progress handle if the caller supplied one.
    # This lets the abort route read real cumulative_tokens at cancel
    # time instead of guessing from a time-ratio estimate.
    if progress is not None:
        progress.register(*agents)

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
        session_state=session_state,
    )


async def _run_triple_pendulum(
    cushion: CushionGraph,
    config: WanderingConfig,
    client: LLMClient,
    fetcher: FetchFn,
    clock: Callable[[], float],
    progress: "WanderingProgress | None" = None,
) -> SessionResult:
    """One chain of N sub-agents, sequential.

    Triple Pendulum = LOW mode: chain of progressively-deeper exploration.
    Each sub-agent picks up where the prior stopped. In V0 we approximate
    this with sequential parallel-of-one execution — same engine, just
    serial. Future enrichment: have each sub-agent receive the prior
    agent's report list as additional context to "pick up from."

    SHARED SESSION DEADLINE:
    All sequential agents share a single deadline = `started + time_secs`.
    Each agent's time budget is the REMAINING time at the moment it
    starts. The chain naturally terminates at the deadline:
      - Agent 1 starts with the full budget; uses what it needs.
      - Agent 2 starts with whatever's left; same.
      - Once an agent finishes and the deadline has passed (or there's
        too little time left to be useful), we stop the chain.
    This makes the UI promise honest: "15 minutes" means the chain
    finishes within 15 minutes total, not 15 × num_agents minutes.

    Token cap is still enforced between agents (defense in depth).
    """
    num_agents, time_secs, tokens_each, model_mix = config.resolved()
    models = assign_models(num_agents, model_mix)

    started = clock()
    session_deadline = started + time_secs
    session_id = config.session_id or f"wsess-{uuid.uuid4().hex[:8]}"

    # Sequential mode still shares one SessionState — even though agents
    # run one-after-another, each agent's visited URLs and queued
    # follow-ons benefit the next agent in the chain.
    session_state = SessionState(session_id=session_id)

    reports: list[ExplorationReport] = []
    traces: list[DecisionTrace] = []
    total_tokens = 0

    # Minimum time we'll bother spawning a new agent for. Below this,
    # the agent won't have time to do useful retrieval before its own
    # budget exhausts, so we end the chain instead of starting a
    # near-zero-budget run. 10 seconds is enough for at least one LLM
    # round trip; below that, skip.
    MIN_AGENT_BUDGET_SEC = 10.0

    for i in range(num_agents):
        remaining = session_deadline - clock()
        if remaining < MIN_AGENT_BUDGET_SEC:
            log.info(
                "triple_pendulum: ending chain after agent %d "
                "(remaining=%.1fs below threshold)",
                i, remaining,
            )
            break

        agent = AgentState(
            agent_id=f"P{i + 1:02d}",
            cushion=cushion,
            budget=AgentBudget(
                time_budget_seconds=remaining,
                token_budget=tokens_each,
            ),
            model_slug=models[i],
            session_state=session_state,
        )
        # Register THIS sequentially-spawned agent before it runs, so
        # an abort fired mid-chain reads the right cumulative_tokens
        # value off the currently-active agent.
        if progress is not None:
            progress.register(agent)
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
        session_state=session_state,
    )


async def _run_absolute_chaos(
    cushion: CushionGraph,
    config: WanderingConfig,
    client: LLMClient,
    fetcher: FetchFn,
    clock: Callable[[], float],
    progress: "WanderingProgress | None" = None,
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
    # First, run the root agents like multi-pendulum. The progress
    # handle (if supplied) accumulates root agents AND sub-agents — see
    # the spawn block below for sub-agent registration.
    session = await _run_multi_pendulum(cushion, config, client, fetcher, clock, progress)

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

    # Run all auto-spawned sub-agents in parallel. Pass through the
    # parent wander's session_state so sub-agents skip URLs the root
    # agents already visited and can dequeue follow-on items.
    results = await asyncio.gather(
        *[run_subagent(
            req,
            client=client,
            fetcher=fetcher,
            parent_clock=clock,
            session_state=session.session_state,
        ) for req, _ in spawn_jobs],
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
    progress: "WanderingProgress | None" = None,
) -> SessionResult:
    """Top-level entry: run one Wandering Room session and return all
    reports + traces.

    Dispatches on `config.mode` to the per-mode runner. Callers (Phase 5
    synthesis, or the eventual API endpoint) take the SessionResult and
    feed it to the synthesis pipeline.

    `progress`: optional WanderingProgress handle. When provided, each
    spawned agent registers itself there as it's created, giving the
    caller a live read on cumulative_tokens for the abort/refund path.
    When None, the wander runs exactly as before.
    """
    if config.mode == WanderingMode.TRIPLE_PENDULUM:
        runner = _run_triple_pendulum
    elif config.mode == WanderingMode.ABSOLUTE_CHAOS:
        runner = _run_absolute_chaos
    else:
        runner = _run_multi_pendulum

    return await runner(cushion, config, client, fetcher, clock, progress)


__all__ = [
    "WanderingMode",
    "ModeDefaults",
    "MODE_DEFAULTS",
    "WanderingConfig",
    "SessionResult",
    "WanderingProgress",
    "assign_models",
    "run_wandering_session",
]
