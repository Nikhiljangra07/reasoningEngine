"""
The Wandering Agent — one pendulum.

A single agent receives the anchor (CushionGraph), a budget, and a model
slug. It then loops: pick next domain (policy.next_move), fetch content,
match against cushion (matching.match_content), if matches → dig with
iteration count from matching.iterations_for_match, run self-critique
(critique.run_self_critique) at iteration boundaries, produce
ExplorationReport. Repeat until budget exhausted.

Content fetching is abstracted via a FetchFn callback — Phase 1 ships
with a stub fetcher; Phase wiring will plug in Tavily/Notion/etc. This
isolation means the agent loop is testable WITHOUT any real internet.

Dig content (the iterations of analysis on a found resonance) is also
LLM-mediated — the agent uses Sonnet for the actual write-up of the
report (where prose quality matters) and Haiku for the matching/critique
(where structured judgment matters).

Per Law 4: the agent NEVER edits anything. All side-effects are appends
to the trace. The agent's only outputs are ExplorationReports.

ISOLATION: imports cushion, report, trace, matching, critique, policy,
LLM client. Does NOT import runtime (the runtime orchestrates many
agents; the agent doesn't know about other agents).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from src.llm.client import LLMClient, LLMResponse
from src.wandering.cushion import CushionGraph
from src.wandering.critique import (
    CritiqueResult,
    CritiqueVerdict,
    run_self_critique,
)
from src.wandering.matching import (
    MatchResult,
    iterations_for_match,
    match_content,
)
from src.wandering.policy import NextMove, next_move
from src.wandering.report import (
    Confidence,
    ExplorationReport,
    LayerMatch,
    SourceCitation,
)
from src.wandering.trace import (
    DecisionTrace,
    DiscardedClue,
    DiscardKind,
    StepKind,
    TraceStep,
)


log = logging.getLogger("constellax.wandering.agent")


# ---------------------------------------------------------------------------
# Fetch interface — pluggable for tests / future Tavily wiring
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """One unit of content fetched from a source.

    Phase 1 stubs this with synthetic content. Real fetcher (Tavily,
    Notion, etc.) returns SourceCitation-friendly metadata + body text.
    """

    title: str
    url: str = ""
    body: str = ""
    domain_hint: str = ""  # which domain the agent was searching


#: A FetchFn takes (domain, query_hint) and returns FetchResult.
#: query_hint is typically the anchor problem summary; fetcher decides
#: how to use it.
FetchFn = Callable[[str, str], Awaitable[FetchResult]]


async def stub_fetcher(domain: str, query_hint: str) -> FetchResult:
    """No-op fetcher for tests. Returns a synthetic FetchResult.

    Wired in Phase 0-engine as the default; the runtime injects a real
    fetcher (Tavily/Notion) later.
    """
    return FetchResult(
        title=f"[stub content from {domain}]",
        url=f"https://stub.example/{domain}",
        body=f"Synthetic placeholder content for {domain} domain.",
        domain_hint=domain,
    )


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


@dataclass
class AgentBudget:
    """Resource limits for one agent."""

    time_budget_seconds: float = 30 * 60     # default 30 min
    token_budget: int = 30_000               # ~30k tokens per agent
    max_steps: int = 100                     # hard cap on loop iterations

    def is_exhausted(
        self,
        elapsed_seconds: float,
        tokens_spent: int,
        steps: int,
    ) -> tuple[bool, str]:
        """Return (exhausted, reason)."""
        if elapsed_seconds >= self.time_budget_seconds:
            return True, "time_budget"
        if tokens_spent >= self.token_budget:
            return True, "token_budget"
        if steps >= self.max_steps:
            return True, "step_cap"
        return False, ""


@dataclass
class AgentState:
    """Per-agent runtime state. Lives in memory during the session.

    `cumulative_tokens` is a rough running count fed by LLMResponse
    .input_tokens + .output_tokens. Not exact (some models report
    differently), but consistent enough to bound spending.
    """

    agent_id: str
    cushion: CushionGraph
    budget: AgentBudget
    model_slug: str = "anthropic/claude-haiku-4-5"

    # Mutable as the agent runs
    cumulative_tokens: int = 0
    steps_taken: int = 0
    started_at: float = 0.0
    reports: list[ExplorationReport] = field(default_factory=list)
    trace: DecisionTrace = field(default_factory=lambda: DecisionTrace(agent_id=""))

    def __post_init__(self) -> None:
        if not self.trace.agent_id:
            self.trace.agent_id = self.agent_id
        if not self.trace.anchor_summary and self.cushion:
            self.trace.anchor_summary = self.cushion.raw_input.problem.content[:120]


# ---------------------------------------------------------------------------
# Dig — multi-iteration deep dive on a resonance
# ---------------------------------------------------------------------------


_DIG_SYSTEM_PROMPT = """\
You are one iteration of a deep dig inside Constellax's Wandering Room.

The user's anchor is below. You ALREADY established structural match with
the source content (per layer_matches). Your job RIGHT NOW: write the
honest mini-report of what this source resonates with and where the
analogy breaks.

CRITICAL RULES (Constellax Laws):

  1. The insight happens in the USER's head. You are NOT delivering a
     solution. You are surfacing a structural bridge. Use language like
     "this resonates with..." not "this solves your problem...".

  2. The `what_does_not_map` field is MANDATORY and load-bearing. If you
     leave it empty or write something like "everything maps fine", the
     report is invalid and will be rejected. Find AT LEAST ONE concrete
     mismatch between source and anchor.

  3. Be honest about confidence. If only 1 node in 1 layer matched, label
     LOW. Don't oversell.

  4. Cite the source — title, brief excerpt, what role it played.

# OUTPUT FORMAT

Return ONE JSON object:

{
  "exploration_summary": "<2-3 sentences: what you found, in human terms>",
  "advancement": "<1-2 sentences: how this resonates with the anchor>",
  "what_does_not_map": "<1-2 sentences: where the analogy breaks; MANDATORY>",
  "next_lead": "<optional: where to dig further if user wants more>"
}

No prose preamble. No code fences. Just JSON.
"""


async def _run_dig_iteration(
    cushion: CushionGraph,
    fetched: FetchResult,
    match: MatchResult,
    iteration_index: int,
    client: LLMClient,
) -> tuple[str, int, int]:
    """Run ONE iteration of dig analysis. Returns (raw_text, in_tokens, out_tokens).

    The text is JSON per the dig prompt. The agent loop accumulates these
    across the dig and builds a final ExplorationReport at the end.
    """
    blocks = [
        "# ANCHOR",
        cushion.to_anchor_prompt(),
        "\n# SOURCE",
        f"Title: {fetched.title}",
        f"URL: {fetched.url}",
        f"Domain: {fetched.domain_hint}",
        f"\nBody:\n{fetched.body}",
        "\n# MATCH RESULT",
        f"Total nodes matched: {match.total_matched_nodes}",
    ]
    for layer_name, lm in match.matches.items():
        if lm.match_count:
            blocks.append(
                f"  {layer_name}: matched {lm.match_count}/{lm.total_nodes} "
                f"({', '.join(lm.matched_nodes)})"
            )
    blocks.append(f"\n# YOUR ITERATION: {iteration_index + 1} of {match.dig_iterations}")
    blocks.append("Write the dig report JSON. Be honest about what does not map.")
    user_message = "\n".join(blocks)

    response: LLMResponse = await client.call(
        system_prompt=_DIG_SYSTEM_PROMPT,
        user_message=user_message,
        domain="synthesizer",     # Sonnet 4.6 for the prose
        concept="wandering_dig",
    )
    return (
        response.content if response.success else "",
        response.input_tokens,
        response.output_tokens,
    )


def _parse_dig_response(raw: str) -> dict[str, str]:
    """Parse the JSON from a dig iteration. Returns dict with the four
    string fields (or empty strings on failure).

    Defensive: even malformed responses produce SOMETHING the agent loop
    can use. We don't halt the agent on parse errors — we treat it as
    a thin iteration and let critique catch the pattern.
    """
    import json
    import re

    text = raw.strip()
    fenced = re.match(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    if start < 0:
        return {}
    # bracket-walk to find the matching close
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return {}

    try:
        payload = json.loads(text[start:end])
        if not isinstance(payload, dict):
            return {}
        return {k: str(v).strip() for k, v in payload.items() if isinstance(k, str)}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Build report from accumulated dig output
# ---------------------------------------------------------------------------


def _build_report_from_dig(
    state: AgentState,
    fetched: FetchResult,
    match: MatchResult,
    iteration_payloads: list[dict[str, str]],
    iterations_completed: int,
    abandoned_early: bool,
) -> ExplorationReport:
    """Assemble an ExplorationReport from the dig's per-iteration payloads.

    The final iteration's payload is the canonical source for the four
    string fields (exploration_summary, advancement, what_does_not_map,
    next_lead). Earlier iterations are accessible via the trace.
    """
    last = iteration_payloads[-1] if iteration_payloads else {}

    report_id = f"wander-{state.agent_id}-{len(state.reports) + 1:03d}"
    report = ExplorationReport(
        report_id=report_id,
        agent_id=state.agent_id,
        anchor_summary=state.trace.anchor_summary,
        domain_explored=fetched.domain_hint,
        source_locations=[
            SourceCitation(
                title=fetched.title,
                url=fetched.url,
                excerpt=fetched.body[:200],
                used_for="structural comparison",
            )
        ],
        layer_matches=match.matches,
        exploration_summary=last.get("exploration_summary", ""),
        advancement=last.get("advancement", ""),
        what_does_not_map=last.get("what_does_not_map", ""),
        next_lead=last.get("next_lead", ""),
        iteration_count=iterations_completed,
        abandoned_early=abandoned_early,
    )
    report.confidence = report.compute_confidence()
    return report


# ---------------------------------------------------------------------------
# Top-level agent loop
# ---------------------------------------------------------------------------


async def run_agent(
    state: AgentState,
    client: LLMClient,
    fetcher: FetchFn = stub_fetcher,
    *,
    clock: Callable[[], float] = time.time,
) -> AgentState:
    """Run one wandering agent to completion.

    Returns the updated state (reports populated, trace populated, budget
    consumed). Caller (the runtime) typically just discards the returned
    state after extracting `reports` — the state object is the same one
    passed in.

    Loop invariant: every operation appends a TraceStep before returning.
    The trace is the audit log; nothing happens silently.
    """
    state.started_at = clock()
    state.trace.append(TraceStep(
        step_id=0,
        kind=StepKind.INITIALIZED,
        timestamp=state.started_at,
        rationale=f"agent {state.agent_id} initialized with anchor",
        tokens_spent=0,
    ))

    while True:
        elapsed = clock() - state.started_at
        exhausted, reason = state.budget.is_exhausted(
            elapsed_seconds=elapsed,
            tokens_spent=state.cumulative_tokens,
            steps=state.steps_taken,
        )
        if exhausted:
            state.trace.append(TraceStep(
                step_id=0,
                kind=StepKind.EXHAUSTED,
                timestamp=clock(),
                rationale=f"budget exhausted: {reason}",
                tokens_spent=state.cumulative_tokens,
            ))
            state.trace.completion_reason = f"exhausted_{reason}"
            state.trace.ended_at = clock()
            return state

        # 1. Policy decides next move
        move: NextMove = next_move(state.cushion, state.trace)
        state.trace.append(TraceStep(
            step_id=0,
            kind=move.kind,
            timestamp=clock(),
            position=move.position,
            rationale=move.rationale,
            tokens_spent=state.cumulative_tokens,
        ))
        state.steps_taken += 1

        if move.kind == StepKind.RETURNED_TO_ANCHOR:
            # Re-orientation: skip fetch, next loop iteration will chaos-pick again
            continue

        # 2. Fetch content from the chosen domain
        fetched = await fetcher(move.position, state.cushion.raw_input.problem.content)

        # 3. Match content against cushion
        match = await match_content(
            cushion=state.cushion,
            content=f"{fetched.title}\n\n{fetched.body}",
            client=client,
            domain_hint=fetched.domain_hint,
        )
        state.trace.append(TraceStep(
            step_id=0,
            kind=StepKind.MATCHED,
            timestamp=clock(),
            position=fetched.domain_hint or move.position,
            rationale=f"matched {match.total_matched_nodes} nodes across layers",
            matched_count=match.total_matched_nodes,
            tokens_spent=state.cumulative_tokens,
        ))
        state.steps_taken += 1

        if not match.has_any_match():
            # No resonance — discard with classification, move on
            state.trace.discard(DiscardedClue(
                description=fetched.title,
                source_hint=fetched.url or fetched.domain_hint,
                classification=DiscardKind.DISCARDED_FOR_CURRENT_ANCHOR,
                reason="no node match in any cushion layer",
                timestamp=clock(),
            ))
            state.trace.append(TraceStep(
                step_id=0,
                kind=StepKind.MOVED_ON,
                timestamp=clock(),
                position=move.position,
                rationale="no match — moving on",
                tokens_spent=state.cumulative_tokens,
            ))
            state.steps_taken += 1
            continue

        # 4. Dig: run match.dig_iterations of analysis
        iteration_payloads: list[dict[str, str]] = []
        iterations_completed = 0
        abandoned_early = False

        for iteration_idx in range(match.dig_iterations):
            raw, in_toks, out_toks = await _run_dig_iteration(
                cushion=state.cushion,
                fetched=fetched,
                match=match,
                iteration_index=iteration_idx,
                client=client,
            )
            state.cumulative_tokens += in_toks + out_toks
            iteration_payloads.append(_parse_dig_response(raw))
            iterations_completed += 1

            state.trace.append(TraceStep(
                step_id=0,
                kind=StepKind.DUG,
                timestamp=clock(),
                position=fetched.domain_hint,
                rationale=f"dig iteration {iteration_idx + 1}/{match.dig_iterations}",
                iterations_used=iteration_idx + 1,
                tokens_spent=state.cumulative_tokens,
            ))
            state.steps_taken += 1

            # 5. Self-critique at iteration boundary
            latest_finding = iteration_payloads[-1].get("exploration_summary", "")
            critique = await run_self_critique(
                cushion=state.cushion,
                agent_position=fetched.domain_hint,
                latest_finding=latest_finding,
                cumulative_tokens=state.cumulative_tokens,
                iterations_so_far=iteration_idx + 1,
                client=client,
            )
            state.trace.append(TraceStep(
                step_id=0,
                kind=StepKind.SELF_CRITIQUED,
                timestamp=clock(),
                position=fetched.domain_hint,
                rationale=f"critique verdict: {critique.verdict.value}",
                detail=critique.summary,
                tokens_spent=state.cumulative_tokens,
            ))
            state.steps_taken += 1

            if critique.verdict == CritiqueVerdict.ABANDON_DIG:
                abandoned_early = True
                state.trace.append(TraceStep(
                    step_id=0,
                    kind=StepKind.ABANDONED,
                    timestamp=clock(),
                    position=fetched.domain_hint,
                    rationale="critique recommended abandon",
                    tokens_spent=state.cumulative_tokens,
                ))
                state.steps_taken += 1
                break

            if critique.verdict == CritiqueVerdict.RETURN_TO_ANCHOR:
                # Stop digging this source, return to anchor on next outer loop
                state.trace.append(TraceStep(
                    step_id=0,
                    kind=StepKind.RETURNED_TO_ANCHOR,
                    timestamp=clock(),
                    position="(anchor)",
                    rationale="critique recommended return to anchor",
                    tokens_spent=state.cumulative_tokens,
                ))
                state.steps_taken += 1
                break

            if critique.verdict == CritiqueVerdict.HAND_OFF:
                # Hand-off recorded but the agent itself completes the dig.
                # Sub-agent spawning is Phase 7 (Absolute Chaos mode); for
                # now we just log intent so the runtime can detect it.
                state.trace.append(TraceStep(
                    step_id=0,
                    kind=StepKind.SPAWNED_SUBAGENT,
                    timestamp=clock(),
                    position=fetched.domain_hint,
                    rationale="critique flagged hand-off (not wired in Phase 1)",
                    tokens_spent=state.cumulative_tokens,
                ))
                state.steps_taken += 1
                # Continue digging in current iteration — hand-off is for
                # future routing, not current break.

        # 6. Build report from accumulated iterations
        report = _build_report_from_dig(
            state=state,
            fetched=fetched,
            match=match,
            iteration_payloads=iteration_payloads,
            iterations_completed=iterations_completed,
            abandoned_early=abandoned_early,
        )

        # Validate; if missing what_does_not_map, mark abandoned (no
        # re-prompt loop in Phase 1; we add it later if needed). Per Law 7
        # we don't ship dishonest reports — better to skip than mislead.
        errors = report.validate()
        if errors:
            log.info(
                "agent %s: dig report failed validation (%s); discarding",
                state.agent_id,
                "; ".join(errors),
            )
            state.trace.discard(DiscardedClue(
                description=f"unvalidated dig on {fetched.title}",
                source_hint=fetched.url,
                classification=DiscardKind.DISCARDED_FOR_CURRENT_ANCHOR,
                reason="; ".join(errors),
                timestamp=clock(),
            ))
            continue

        state.reports.append(report)
        state.trace.append(TraceStep(
            step_id=0,
            kind=StepKind.REPORTED,
            timestamp=clock(),
            position=fetched.domain_hint,
            rationale=f"produced report {report.report_id} ({report.confidence.value})",
            report_id=report.report_id,
            tokens_spent=state.cumulative_tokens,
        ))
        state.steps_taken += 1


__all__ = [
    "FetchResult",
    "FetchFn",
    "stub_fetcher",
    "AgentBudget",
    "AgentState",
    "run_agent",
]
