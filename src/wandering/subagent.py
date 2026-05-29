"""
Sub-agent spawning — the recursive depth primitive for Absolute Chaos mode.

Per the plan: sub-agents are spawned via tool call (NOT MCP). Each
wandering agent in Triple Pendulum or Absolute Chaos mode is offered the
`spawn_subagent` tool. When the agent's main dig finds a HIGH-confidence
resonance with remaining budget and not yet at max chain depth, the
runtime calls this module to spawn a child agent inheriting the cushion.

The user-requested "dig deeper" path also lands here — the API endpoint
calls spawn_subagent_on_report() with the original report's matched
subgraph as the child agent's starting focus.

Bounded by:
  - per-mode max chain depth (Triple = 3, Absolute = 5)
  - per-session credit cap
  - parent agent's remaining budget

Per Law 1: sub-agents inherit the same chaos policy as their parent. The
only thing that changes is the starting domain hint — they DON'T get a
"smarter" policy that exploits the parent's discovery.

ISOLATION: imports agent + cushion types + LLM client. Does NOT import
runtime.py (avoids circular imports) — runtime.py imports US.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from src.llm.client import LLMClient
from src.wandering.agent import (
    AgentBudget,
    AgentState,
    FetchFn,
    run_agent,
    stub_fetcher,
)
from src.wandering.cushion import CushionGraph
from src.wandering.report import Confidence, ExplorationReport
from src.wandering.trace import StepKind, TraceStep


log = logging.getLogger("constellax.wandering.subagent")


# Per-mode chain-depth caps. Beyond these, no further spawning.
MAX_CHAIN_DEPTH = {
    "triple_pendulum": 3,    # 1 root + 2 children = 3 deep
    "multi_pendulum": 1,     # multi has NO sub-agents; cap is 1 to forbid
    "absolute_chaos": 5,     # 1 root + 4 deep children
}


@dataclass
class SpawnRequest:
    """A request to spawn a sub-agent.

    Produced by:
      - A parent agent's dig that found HIGH-confidence resonance
      - The API's user-requested "dig deeper" route

    Validated by `should_spawn` before runtime executes.
    """

    parent_agent_id: str
    cushion: CushionGraph
    focus_area: str             # what the sub-agent should pursue
    starting_domain: str = ""   # initial domain hint
    distance_budget_tokens: int = 15_000
    inherits_anchor: bool = True
    chain_depth: int = 1        # 1 = first sub-agent of root
    mode_key: str = "absolute_chaos"  # for chain-depth lookup


@dataclass
class SpawnResult:
    """The output of running a spawned sub-agent.

    Returned to the parent, which folds the reports back into the session.
    """

    subagent_id: str
    reports: list[ExplorationReport] = field(default_factory=list)
    tokens_spent: int = 0
    chain_depth_used: int = 0
    aborted: bool = False
    abort_reason: str = ""


def should_spawn(
    request: SpawnRequest,
    session_tokens_spent: int,
    session_token_cap: int,
) -> tuple[bool, str]:
    """Decide whether a spawn should proceed.

    Returns (allowed, reason). The reason is for trace logging — when
    spawning is blocked we want to know WHY (depth / budget / mode).
    """
    max_depth = MAX_CHAIN_DEPTH.get(request.mode_key, 1)
    if request.chain_depth > max_depth:
        return False, f"chain_depth {request.chain_depth} > max {max_depth} for {request.mode_key}"

    if request.distance_budget_tokens <= 0:
        return False, "zero distance budget"

    # Reserve at least the requested budget within the session cap.
    if session_tokens_spent + request.distance_budget_tokens > session_token_cap:
        return False, "session_token_cap would be exceeded"

    if not request.focus_area.strip():
        return False, "no focus_area provided"

    return True, ""


async def run_subagent(
    request: SpawnRequest,
    client: LLMClient,
    *,
    fetcher: FetchFn = stub_fetcher,
    parent_clock=None,
) -> SpawnResult:
    """Execute one sub-agent under the given SpawnRequest.

    The sub-agent runs the same agent loop as a root agent, with a
    smaller budget and the same cushion. The starting position
    (focus_area / starting_domain) influences ONLY the first step;
    after that, the standard chaos policy applies.
    """
    sub_id = f"S{uuid.uuid4().hex[:6]}"

    state = AgentState(
        agent_id=sub_id,
        cushion=request.cushion,
        budget=AgentBudget(
            time_budget_seconds=10 * 60,  # sub-agents get tighter time too
            token_budget=request.distance_budget_tokens,
            max_steps=30,
        ),
    )

    # Seed the trace with a SPAWNED_SUBAGENT origin marker so the
    # session-level analysis can trace which subagents came from where.
    import time as _time
    state.trace.append(TraceStep(
        step_id=0,
        kind=StepKind.SPAWNED_SUBAGENT,
        timestamp=_time.time(),
        position=request.starting_domain or request.focus_area,
        rationale=(
            f"spawned by {request.parent_agent_id} for focus: {request.focus_area}"
        ),
    ))

    clock = parent_clock or _time.time
    try:
        await run_agent(state, client, fetcher=fetcher, clock=clock)
    except Exception as e:
        log.warning("subagent %s crashed: %s", sub_id, e)
        return SpawnResult(
            subagent_id=sub_id,
            reports=state.reports,
            tokens_spent=state.cumulative_tokens,
            chain_depth_used=request.chain_depth,
            aborted=True,
            abort_reason=str(e),
        )

    return SpawnResult(
        subagent_id=sub_id,
        reports=state.reports,
        tokens_spent=state.cumulative_tokens,
        chain_depth_used=request.chain_depth,
    )


def spawn_request_from_high_match_report(
    parent_state: AgentState,
    report: ExplorationReport,
    mode_key: str,
    chain_depth: int,
    distance_budget_tokens: int = 15_000,
) -> SpawnRequest | None:
    """Build a SpawnRequest from a HIGH-confidence parent report.

    Returns None if the report doesn't qualify for auto-spawn (i.e., it's
    not HIGH confidence). HIGH confidence on STRUCTURAL axes (essence
    OR mechanism >= 0.7) is what triggers automatic sub-agent spawning
    in Absolute Chaos mode.
    """
    if report.confidence != Confidence.HIGH:
        return None

    # Build focus from matched essence + mechanism nodes — those are what
    # made this report HIGH. The sub-agent inherits the resonance and
    # digs deeper into the same structural pattern.
    matched_nodes: list[str] = []
    essence = report.layer_matches.get("essence")
    if essence:
        matched_nodes.extend(essence.matched_nodes)
    mechanism = report.layer_matches.get("mechanism")
    if mechanism:
        matched_nodes.extend(mechanism.matched_nodes)

    if not matched_nodes:
        return None  # HIGH on actual layer alone isn't worth a sub-spawn

    focus = (
        f"Dig deeper on structural resonance: {', '.join(matched_nodes[:3])}. "
        f"Found in {report.domain_explored}; look for adjacent domains that "
        f"share the same essence/mechanism."
    )

    return SpawnRequest(
        parent_agent_id=parent_state.agent_id,
        cushion=parent_state.cushion,
        focus_area=focus,
        starting_domain="",  # let chaos policy pick fresh; only focus seeds
        distance_budget_tokens=distance_budget_tokens,
        chain_depth=chain_depth,
        mode_key=mode_key,
    )


def spawn_request_from_user_dig_deeper(
    cushion: CushionGraph,
    report: ExplorationReport,
    user_request_text: str = "",
    distance_budget_tokens: int = 20_000,
) -> SpawnRequest:
    """Build a SpawnRequest from a user clicking 'dig deeper' on a report.

    Users can manually trigger sub-agents from any confidence band. The
    user's text (if provided) becomes the focus prompt; otherwise we
    use the report's matched-nodes summary.
    """
    if user_request_text.strip():
        focus = user_request_text.strip()
    else:
        # Derive focus from the report's matched layers
        matched_essence = report.layer_matches.get("essence")
        matched_mech = report.layer_matches.get("mechanism")
        bits = []
        if matched_essence and matched_essence.matched_nodes:
            bits.append(f"essence: {', '.join(matched_essence.matched_nodes)}")
        if matched_mech and matched_mech.matched_nodes:
            bits.append(f"mechanism: {', '.join(matched_mech.matched_nodes)}")
        focus = (
            f"Dig deeper on report {report.report_id}. "
            f"Resonance: {'; '.join(bits) if bits else 'continue exploration'}."
        )

    return SpawnRequest(
        parent_agent_id=report.agent_id,
        cushion=cushion,
        focus_area=focus,
        starting_domain="",
        distance_budget_tokens=distance_budget_tokens,
        chain_depth=2,  # user-initiated sub-agents start at depth 2 conventionally
        mode_key="absolute_chaos",
    )


__all__ = [
    "MAX_CHAIN_DEPTH",
    "SpawnRequest",
    "SpawnResult",
    "should_spawn",
    "run_subagent",
    "spawn_request_from_high_match_report",
    "spawn_request_from_user_dig_deeper",
]
