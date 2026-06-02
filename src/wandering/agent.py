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

from src.identity import compose_system_prompt
from src.identity.disciplines import attachment_detection
from src.llm.client import LLMClient, LLMResponse
from src.wandering.cushion import CushionGraph
from src.wandering.critique import (
    CritiqueResult,
    CritiqueVerdict,
    run_self_critique,
)
from src.wandering.extractors import (
    ExtractResult,
    extract_links,
    extract_url,
    should_escalate_to_tier2,
)
from src.wandering.matching import (
    MatchResult,
    iterations_for_match,
    match_content,
)
from src.wandering import exa_provider
# Constellation Interpreter (Phase 5, 2026-06-01) — feature-flagged.
# Default OFF. Set CONSTELLAX_USE_INTERPRETER=1 to route body matches
# through the multi-channel interpreter instead of the legacy single-LLM
# matcher. The interpreter is structurally stricter and refuses to DIG
# without a structural foothold (vector cos >= 0.6 or overlap >= 1).
from src.wandering import interpreter as _interpreter
from src.wandering.fingerprint import get_or_create_fingerprint as _get_or_create_fingerprint
import os as _os
from src.wandering.trust import adjust_confidence
from src.wandering.policy import NextMove, next_move
from src.wandering.report import (
    Confidence,
    ExplorationReport,
    LayerMatch,
    SourceCitation,
)
from src.wandering.session_state import (
    FollowonItem,
    SessionState,
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
# Feature-flagged interpreter wrapper (Phase 5, 2026-06-01)
# ---------------------------------------------------------------------------


def _use_interpreter() -> bool:
    """Resolved at call time so tests / runtime config can flip mid-run."""
    return _os.environ.get("CONSTELLAX_USE_INTERPRETER", "0") == "1"


# Module-level cached Neo4j driver for the fingerprint cache. Constructed
# lazily on first need from env vars. The driver carries its own connection
# pool, so subsequent calls are cheap. None means no NEO4J_URI is set (or
# driver construction failed) — fingerprint cache is then disabled and
# every interpreter call pays the full Haiku + Gemini cost.
_NEO4J_DRIVER: object | None = None
_NEO4J_DRIVER_INITIALIZED: bool = False
_NEO4J_DATABASE: str = "neo4j"


def _get_neo4j_driver() -> tuple[object | None, str]:
    """Return (driver, database) for the fingerprint cache, or
    (None, "neo4j") if Neo4j isn't configured.

    Constellation Interpreter (Phase 6, 2026-06-01). Constructed on
    first call, cached for the process lifetime. Driver failure logs
    once and returns None — the agent keeps running without cache,
    so a Neo4j outage degrades fingerprint latency but doesn't block
    the wander.
    """
    global _NEO4J_DRIVER, _NEO4J_DRIVER_INITIALIZED, _NEO4J_DATABASE
    if _NEO4J_DRIVER_INITIALIZED:
        return _NEO4J_DRIVER, _NEO4J_DATABASE

    _NEO4J_DRIVER_INITIALIZED = True  # set first so a failure isn't retried per-call
    uri = _os.environ.get("NEO4J_URI", "").strip()
    if not uri:
        return None, _NEO4J_DATABASE

    user = _os.environ.get("NEO4J_USERNAME", "").strip() or "neo4j"
    pwd = _os.environ.get("NEO4J_PASSWORD", "").strip()
    _NEO4J_DATABASE = _os.environ.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j"

    try:
        from neo4j import AsyncGraphDatabase
        _NEO4J_DRIVER = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
        log.info(
            "fingerprint cache: Neo4j driver initialized for database=%s",
            _NEO4J_DATABASE,
        )
    except Exception as e:
        log.warning("fingerprint cache disabled: Neo4j driver init failed: %s", e)
        _NEO4J_DRIVER = None

    return _NEO4J_DRIVER, _NEO4J_DATABASE


def _verdict_to_match_result(
    verdict: _interpreter.InterpreterVerdict,
    cushion: "CushionGraph",
) -> MatchResult:
    """Convert an InterpreterVerdict into the legacy MatchResult shape so
    the wandering loop's downstream code stays unchanged.

    Mapping:
      DIG             — matches populated by layer, iterations 3-5
      SAVE_FOR_LATER  — 1 minimal match (so has_any_match returns True),
                        iterations capped at 1 (minimal dig)
      SKIP            — no matches, iterations 0
    """
    matched_set = set(verdict.matched_nodes)
    layer_matches: dict[str, LayerMatch] = {}
    for layer in cushion.layers():
        hits = [t for t in layer.nodes if t in matched_set]
        layer_matches[layer.name] = LayerMatch(
            layer_name=layer.name,
            matched_nodes=hits,
            total_nodes=layer.node_count(),
        )

    if verdict.decision == "dig":
        total = max(1, len(verdict.matched_nodes))
        iterations = min(5, max(3, total))
    elif verdict.decision == "save_for_later":
        # Surface the match without burning dig budget.
        total = 1
        iterations = 1
    else:  # skip
        total = 0
        iterations = 0

    return MatchResult(
        matches=layer_matches,
        total_matched_nodes=total,
        dig_iterations=iterations,
        raw_response=f"interpreter:{verdict.decision} | {verdict.reason}",
    )


async def _run_match(
    *,
    cushion: "CushionGraph",
    content: str,
    client: LLMClient,
    domain_hint: str,
    url: str = "",
    interpreter_state: _interpreter.SessionState | None = None,
) -> MatchResult:
    """Feature-flagged match dispatch.

    When CONSTELLAX_USE_INTERPRETER=1, runs the Constellation Interpreter
    (fingerprint -> 7-channel scorer -> verdict) and adapts the verdict
    to a MatchResult shape. Otherwise calls the legacy single-LLM
    match_content. Any exception in the interpreter path falls back to
    legacy so the wander never blocks on the new code path.

    `url` is the source URL of the content — needed for fingerprint
    persistence + the evidence channel. Passing "" disables those.
    """
    if _use_interpreter():
        try:
            # Phase 6 (2026-06-01): plumb the lazy Neo4j driver to the
            # fingerprint cache. Cache hits return in <100ms; misses pay
            # the full Haiku + Gemini cost and then persist for future
            # reads. Driver=None gracefully disables the cache without
            # failing the call.
            driver, database = _get_neo4j_driver()
            fp = await _get_or_create_fingerprint(
                content, url, client=client,
                neo4j_driver=driver, neo4j_database=database,
            )
            verdict = await _interpreter.interpret(
                fp, cushion, client=client,
                session_state=interpreter_state,
            )
            # Mark the fingerprint as seen so the novelty channel on
            # subsequent agents in the same wander has accumulated
            # history to consult. Without this the novelty score stays
            # at constant 1.0 for the whole run and pollutes the
            # disagreement variance. Per the interpreter's own
            # docstring, mark_seen runs AFTER the verdict — never
            # before, otherwise the calling content would self-suppress.
            if interpreter_state is not None:
                try:
                    interpreter_state.mark_seen(fp)
                except Exception as _e:  # pragma: no cover — defensive
                    log.debug("interpreter_state.mark_seen failed: %s", _e)
            return _verdict_to_match_result(verdict, cushion)
        except Exception as e:
            log.warning("interpreter path failed, falling back to legacy: %s", e)
    return await match_content(
        cushion=cushion, content=content, client=client,
        domain_hint=domain_hint,
    )


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
#:
#: Real fetchers also accept `session_state` as a keyword argument so they
#: can consult the per-wander dedup set and follow-on queue. We type this
#: as `Callable[..., Awaitable[FetchResult]]` rather than a strict
#: 2-argument form so legacy fetchers (stub_fetcher) and state-aware
#: fetchers (web_search_fetcher) both satisfy the protocol.
FetchFn = Callable[..., Awaitable[FetchResult]]


async def stub_fetcher(
    domain: str, query_hint: str, **_kwargs: object,
) -> FetchResult:
    """No-op fetcher for tests. Returns a synthetic FetchResult.

    Accepts arbitrary keyword arguments (e.g. `session_state`) so it
    satisfies the same FetchFn protocol as real fetchers without forcing
    every test to thread through state plumbing it doesn't need.
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

    `session_state` is the shared per-wander state (dedup set + follow-on
    queue). It's optional so existing tests can construct an AgentState
    without plumbing one through — when None, the agent runs in isolated
    mode (no dedup, no follow-on hops, no cross-agent serendipity).
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

    # Per-wander shared state — populated by the runtime when N agents
    # are spawned, so every agent in the same session sees the same
    # visited_urls / followon_queue. None for legacy single-agent tests.
    session_state: "SessionState | None" = None

    # Per-wander interpreter novelty memory (separate from the runtime
    # SessionState above). The interpreter's `score_novelty` channel
    # consults this to detect content the wander already saw, and the
    # agent's _run_match marks each fingerprint as seen AFTER the
    # verdict. Without this wiring (the run-#1 state), the novelty
    # channel returned constant 1.0 and added pure noise to the
    # disagreement signal. None for legacy callers that don't go
    # through the runtime's tracker creation path.
    interpreter_state: "_interpreter.SessionState | None" = None

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
        system_prompt=compose_system_prompt(_DIG_SYSTEM_PROMPT, mode="wandering_dig"),
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

    Confidence is computed from layer matches, then run through the
    domain-trust tiebreaker (trust.adjust_confidence). Trust can only
    PROMOTE a borderline match — it cannot demote a strong one. See
    WANDERING_ROOM_DECISIONS.md D7.
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
    base_confidence = report.compute_confidence()
    report.confidence = adjust_confidence(
        base_confidence,
        url=fetched.url,
        total_matched_nodes=match.total_matched_nodes,
    )

    # Identity-layer metadata: scan the report's prose for distortion
    # patterns. Pure-Python regex scan, no LLM, no network. Result is
    # attached as metadata; the dossier and frontend may render the
    # flags as "patterns to watch for" but no engine logic gates on
    # them. Failure of the scan (none expected — it's deterministic)
    # would leave the field empty, which is harmless.
    combined_text = " ".join(filter(None, (
        report.exploration_summary,
        report.advancement,
        report.what_does_not_map,
    )))
    if combined_text.strip():
        try:
            report.attachment_flags = attachment_detection.scan(combined_text)
        except Exception:  # pragma: no cover — defensive; scan is pure
            log.exception("attachment_detection.scan raised; leaving flags empty")
            report.attachment_flags = []

    return report


# ---------------------------------------------------------------------------
# Tier-2 escalation + follow-on queueing
# ---------------------------------------------------------------------------


def _layer_ratio(match: MatchResult, layer_name: str) -> float:
    """Helper — ratio for one layer, or 0.0 if absent. Used by the
    tier-2 escalation gate (essence/mechanism ratios)."""
    m = match.matches.get(layer_name)
    if m is None or m.total_nodes <= 0:
        return 0.0
    return m.match_count / m.total_nodes


async def _maybe_tier2_escalate(
    fetched: FetchResult,
    match: MatchResult,
    state: AgentState,
    client: LLMClient,
    clock: Callable[[], float],
) -> tuple[FetchResult, MatchResult, bool]:
    """Decide whether to escalate this fetch to tier-2 (full-page read
    via Jina Reader), perform the escalation if so, and return the
    (possibly upgraded) fetched + match pair plus a flag.

    Returns (new_fetched, new_match, escalated_bool).

    Gating (see extractors.should_escalate_to_tier2):
      - tier-1 must have at least 1 matched node (some signal)
      - essence and mechanism ratios both below the strong-signal ceiling

    If escalation fires:
      - fetch the URL via Jina Reader
      - replace body with the extract
      - re-run the matcher on the richer body
      - take whichever match has more total_matched_nodes (never
        regress; if Jina's body for some reason produces a weaker
        match, we keep the tier-1 result)
    """
    if not fetched.url:
        return fetched, match, False
    if not should_escalate_to_tier2(
        total_matched_nodes=match.total_matched_nodes,
        essence_ratio=_layer_ratio(match, "essence"),
        mechanism_ratio=_layer_ratio(match, "mechanism"),
        url=fetched.url,
    ):
        return fetched, match, False

    extract = await extract_url(fetched.url)
    if not extract.ok or not extract.body:
        return fetched, match, False

    upgraded = FetchResult(
        title=fetched.title,
        url=fetched.url,
        body=extract.body,
        domain_hint=fetched.domain_hint,
    )
    new_match = await _run_match(
        cushion=state.cushion,
        content=f"{upgraded.title}\n\n{upgraded.body}",
        client=client,
        domain_hint=upgraded.domain_hint,
        url=upgraded.url,
        interpreter_state=state.interpreter_state,
    )
    # Never regress: if richer body produced fewer matches, keep tier-1.
    if new_match.total_matched_nodes < match.total_matched_nodes:
        return fetched, match, False

    state.trace.append(TraceStep(
        step_id=0,
        kind=StepKind.MATCHED,
        timestamp=clock(),
        position=upgraded.domain_hint,
        rationale=(
            f"tier-2 escalation: matched {new_match.total_matched_nodes} "
            f"nodes (was {match.total_matched_nodes}) on full-page read"
        ),
        matched_count=new_match.total_matched_nodes,
        tokens_spent=state.cumulative_tokens,
    ))
    state.steps_taken += 1
    return upgraded, new_match, True


_LINK_SCORE_THRESHOLD = 0.5
_LINK_QUEUE_MAX_PER_DIG = 2


async def _score_links_for_followon(
    candidate_links: list[tuple[str, str]],
    cushion: "CushionGraph",
    client: LLMClient,
) -> list[tuple[str, float]]:
    """Score each (anchor_text, url) link against the cushion via Haiku
    and return [(url, score)] sorted descending.

    We don't fetch the link's body — that would multiply tier-2 cost. We
    score the ANCHOR TEXT alone against the cushion, same matcher LLM as
    body matching. Anchor text is a tight summary; if it resonates, the
    page very likely does too.

    `score` is total_matched_nodes / cushion_total — a coarse 0..1
    quantity. Above _LINK_SCORE_THRESHOLD → queue eligible.
    """
    if not candidate_links:
        return []

    total_cushion_nodes = sum(
        layer.node_count() for layer in cushion.layers()
    )
    if total_cushion_nodes == 0:
        return []

    out: list[tuple[str, float]] = []
    for anchor_text, url in candidate_links[:10]:  # Cap parallelism at 10
        # The matcher takes a piece of content; anchor text is short, so
        # we wrap it with a one-line header so the matcher knows what
        # this is (otherwise a bare anchor like "see also" matches zero).
        content = f"Link anchor: {anchor_text}\n(considering as a candidate to follow up)"
        # NOTE (Phase 6, 2026-06-01): this callsite intentionally stays
        # on the legacy single-LLM match_content even when the
        # Constellation Interpreter is enabled. Link anchors are 1-5
        # words long — the fingerprint pipeline needs ~10-20 words of
        # structural language per phrase, and short anchors produce
        # either zero phrases or surface-keyword phrases that defeat
        # the vector channel. Haiku's single-call judgment on a short
        # snippet is the right tool for this specific use case.
        match = await match_content(
            cushion=cushion,
            content=content,
            client=client,
            domain_hint="link-candidate",
        )
        score = match.total_matched_nodes / max(total_cushion_nodes, 1)
        out.append((url, score))

    out.sort(key=lambda pair: pair[1], reverse=True)
    return out


async def _queue_followon_from_dig(
    state: AgentState,
    upgraded_body: str,
    parent_url: str,
    client: LLMClient,
) -> int:
    """After a successful tier-2 dig, extract links from the page body,
    score them against the cushion, and queue the top scorers in the
    session follow-on queue. Returns how many were queued.

    Cheap-but-bounded: caps the matcher calls at 10 links per dig and
    queues at most _LINK_QUEUE_MAX_PER_DIG.

    Per Law 1 (chaos is the feature), we don't crawl the page — we let
    the AGENT pick which links to follow. Score-gated insertion means
    junk links (footer, share, "related sponsored") get filtered out
    without a heuristic blocklist.
    """
    if state.session_state is None:
        return 0

    candidates = extract_links(upgraded_body, max_links=10)
    if not candidates:
        return 0

    scored = await _score_links_for_followon(
        candidates, state.cushion, client,
    )
    queued = 0
    for url, score in scored:
        if queued >= _LINK_QUEUE_MAX_PER_DIG:
            break
        if score < _LINK_SCORE_THRESHOLD:
            continue
        added = await state.session_state.enqueue_followon(FollowonItem(
            url=url,
            score=score,
            parent_url=parent_url,
            origin="link",
        ))
        if added:
            queued += 1
    return queued


async def _queue_findsimilar_hops(
    state: AgentState,
    seed_url: str,
) -> int:
    """Call Exa.findSimilar on the seed URL and queue top hits as
    follow-ons. Returns how many were queued.

    The findSimilar primitive is Constellax's chaos-hop differentiator:
    embedding-space neighbors of a URL we already matched against the
    cushion. We trust Exa's ranking and queue the top-N — no rescoring
    via Haiku (which would multiply cost). The cushion-level matcher
    fires later when the follow-on URL is fetched.

    No-op if Exa is not configured or the seed URL is empty.
    """
    if state.session_state is None:
        return 0
    if not seed_url:
        return 0
    if not exa_provider.is_available():
        return 0

    result = await exa_provider.find_similar(
        seed_url, num_results=exa_provider.DEFAULT_NUM_RESULTS,
    )
    if not result.ok:
        return 0

    queued = 0
    # Take the top 2 — beyond that the queue fills with marginal hops.
    for hit in result.hits[:2]:
        # Exa's `score` is roughly 0..1; map it to FollowonItem score.
        score = max(0.0, min(1.0, hit.score))
        added = await state.session_state.enqueue_followon(FollowonItem(
            url=hit.url,
            score=score if score > 0 else 0.5,
            parent_url=seed_url,
            origin="findsimilar",
        ))
        if added:
            queued += 1
    return queued


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

        # 2. Fetch content from the chosen domain.
        # `session_state` is forwarded so the fetcher can dedup against
        # visited URLs and drain the follow-on queue when available.
        # Legacy fetchers (e.g., stub_fetcher) accept **_kwargs and
        # ignore session_state safely.
        fetched = await fetcher(
            move.position,
            state.cushion.raw_input.problem.content,
            session_state=state.session_state,
        )

        # 3. Match content against cushion. Feature-flagged: when
        # CONSTELLAX_USE_INTERPRETER=1, this runs the 7-channel
        # interpreter (fingerprint -> vector/overlap/role/mech/non-map
        # /evidence/novelty -> verdict). Default off keeps legacy
        # single-LLM matcher. `state.interpreter_state` carries the
        # wander's novelty memory so subsequent matches see prior
        # fingerprints (constant-1.0 noise fix).
        match = await _run_match(
            cushion=state.cushion,
            content=f"{fetched.title}\n\n{fetched.body}",
            client=client,
            domain_hint=fetched.domain_hint,
            url=fetched.url,
            interpreter_state=state.interpreter_state,
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

        # 3a. Tier-2 escalation: if the tier-1 hit is borderline (some
        # signal, but neither essence nor mechanism is strong), re-read
        # the URL via Jina and re-match on the fuller body. The helper
        # only returns an upgraded result when the new match is at
        # least as strong as tier-1's; never regresses.
        fetched, match, escalated_to_tier2 = await _maybe_tier2_escalate(
            fetched=fetched,
            match=match,
            state=state,
            client=client,
            clock=clock,
        )

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

        # 7. Post-report serendipity hooks. Fire only when the dig
        # produced a strong-enough match (>=2 matched nodes — the same
        # threshold WANDERING_ROOM_DECISIONS.md D6 specifies for
        # findSimilar). We don't want every weak match polluting the
        # follow-on queue with marginal hops.
        if (
            state.session_state is not None
            and match.total_matched_nodes >= 2
            and fetched.url
        ):
            # Exa /findSimilar chaos-hop — embedding-space neighbors of
            # a URL we just matched against the cushion. No-op when Exa
            # is not configured.
            try:
                await _queue_findsimilar_hops(state, fetched.url)
            except Exception as e:
                log.debug("findSimilar hop failed: %s", e)

            # Link follow-on — only fires when we have a tier-2 body to
            # extract links from. Tier-1 stitched snippets don't have
            # full link markdown.
            if escalated_to_tier2 and fetched.body:
                try:
                    await _queue_followon_from_dig(
                        state=state,
                        upgraded_body=fetched.body,
                        parent_url=fetched.url,
                        client=client,
                    )
                except Exception as e:
                    log.debug("link follow-on failed: %s", e)


__all__ = [
    "FetchResult",
    "FetchFn",
    "stub_fetcher",
    "AgentBudget",
    "AgentState",
    "run_agent",
]
