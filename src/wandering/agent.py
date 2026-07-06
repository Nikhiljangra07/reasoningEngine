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

import json
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
    GateDecision,
    enforce_abandon_gate,
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


def _use_dig_revision_mode() -> bool:
    """Resolved at call time. When WANDER_DIG_REVISION_MODE=1, the dig step
    runs a two-iteration chain (find → critique → revise) instead of the
    legacy 3-5 redundant rerolls that keep only the last.

    Iter 1 (find): structured 7-lens reasoning scaffold producing the
        four-field report.
    Critique: existing six-question self-critique. If verdict is
        abandon_dig, iter 2 is skipped and honest-abandonment ships.
    Iter 2 (revise): reads iter 1 + critique feedback, revises while
        preserving any unusual cross-domain vocabulary from iter 1.

    Default off — keeps the r6 working design as production until A/B
    confirms the new path is denser AND preserves cross-domain vocabulary.
    """
    return _os.environ.get("WANDER_DIG_REVISION_MODE", "0") == "1"


def _use_agent_noticeboard() -> bool:
    """Resolved at call time. When WANDER_AGENT_NOTICEBOARD=1, each agent
    posts a short notice (domain, match strength, one-sentence finding,
    direction pointed) to a shared session-level noticeboard after each
    completed dig, and reads the recent notices before picking its next
    domain. Notices are INFORMATIONAL — they may influence which domain
    an agent picks next but do NOT enter the dig content itself, so each
    report remains an independent sample for the synthesizer.

    Default off — keeps cohort full isolation (current production
    behavior) until A/B confirms the noticeboard improves coverage
    breadth without collapsing cross-domain diversity.
    """
    return _os.environ.get("WANDER_AGENT_NOTICEBOARD", "0") == "1"


def _use_governor() -> bool:
    """Resolved at call time. When CONSTELLAX_GOVERNOR=1, a session-level flow
    governor (src/wandering/governor.py) watches findings arrive on the shared
    noticeboard and may seize FLOW — setting session_state.governor_halt on a
    confirmed CLOSE (the swarm converged AND formed a structured skeleton). Each
    agent checks that flag at the top of its loop and exits gracefully.

    Governs FLOW only (stop/continue), never quality — the human stays the sole
    judge. Default off — zero behavior change until validated on live runs.
    """
    return _os.environ.get("CONSTELLAX_GOVERNOR", "0") == "1"


def _use_contribution_board() -> bool:
    """Resolved at call time. When WANDER_CONTRIBUTION_BOARD=1, each agent
    reads the recent peer notices (what other agents have already surfaced)
    and a CONTRIBUTION-BOARD block is injected into its dig prompt: an
    additive, POSITIVE-SUM directive to add the layer peers have NOT reached
    — go deeper, find the transferable structure beneath what is already on
    the board — explicitly NOT a competition, with an anti-inflation guard.

    Reuses the noticeboard's posted notices as its data source, so this flag
    also enables notice POSTING (so there is something to read) but does NOT
    enable the lateral domain-downweighting (that stays under
    WANDER_AGENT_NOTICEBOARD). Chaos-safe: reads only the PAST (what peers
    already found), never predicts or steers the walk. Default off — the
    legacy dig path is byte-for-byte unchanged when this is unset.

    NOTE: wired into the default dig path (PATH B, _run_dig_iteration). The
    revision-mode path (PATH A, WANDER_DIG_REVISION_MODE) is not yet hooked.
    """
    return _os.environ.get("WANDER_CONTRIBUTION_BOARD", "0") == "1"


def _use_node_query_retrieval() -> bool:
    """Resolved at call time. When WANDER_NODE_QUERY_RETRIEVAL=1, the agent
    seeds each fetch with a cushion-NODE-derived query (the metal-detector
    design: graph nodes are the probe) instead of the pursuit text.

    Default off — keeps the legacy pursuit-text seed (current behavior)
    so r6 and earlier runs are A/B-comparable against the metal-detector
    run."""
    return _os.environ.get("WANDER_NODE_QUERY_RETRIEVAL", "0") == "1"


def _forensics_path() -> str | None:
    """Resolved at call time. When WANDER_FORENSICS_PATH is set, each
    PATH A dig appends a structured record to that file.

    r8 surveillance layer (June 2026). Captures everything the pipeline
    does per dig WITHOUT changing pipeline behavior. Each entry includes
    iter-1/critique/iter-2 payload lengths, timing, token spend, critique
    verdict + red flags + Q1-Q6 answers, lexical-preservation metrics,
    sanding auto-revert events, and which payload was actually shipped.

    Append-only writes; no read-back during the run; try/except wrapped.
    A surveillance failure NEVER reaches the pipeline. No new LLM calls;
    no cost increase. Default off (env unset) keeps production unchanged.
    """
    return _os.environ.get("WANDER_FORENSICS_PATH") or None


def _write_forensics_entry(entry: dict) -> None:
    """Append one structured forensics record to the configured path.

    Two stages of fault tolerance:
      1. JSON encode failure → log + drop entry, pipeline continues.
      2. File append failure → log + drop entry, pipeline continues.

    The pipeline always finishes even if surveillance is down.
    """
    path = _forensics_path()
    if not path:
        return
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str)
    except Exception:
        log.exception("forensics: json encode failed; entry dropped")
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        log.exception("forensics: append failed (path=%s); entry dropped", path)


# Layer weighting for node-query retrieval. The METAL-DETECTOR intent
# requires the wander to be driven by the structural / abstract layers of
# the cushion (essence + mechanism), NOT by the actual layer (literal
# entities from the pursuit). The actual layer is kept at a low weight as
# an occasional noise source — entirely zeroing it would lose its rare
# value for grounding when an agent needs to re-anchor.
_NODE_LAYER_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("mechanism", 0.60),
    ("essence",   0.30),
    ("actual",    0.10),
)


def _draw_node_query(cushion: "CushionGraph") -> tuple[str, str, str]:
    """Draw a single (query, layer_name, node_text) triple from the cushion
    graph for the next fetch.

    Picks a layer weighted by `_NODE_LAYER_WEIGHTS`, then picks a node
    uniformly from that layer, then picks one of the node's
    `search_queries` uniformly (or falls back to `node.text` when the
    node has no search_queries — e.g. legacy cushions composed before
    the dual-artifact upgrade). Falls all the way back to ("", "", "")
    when the cushion is empty across all layers.

    This is the metal-detector retrieval seed Nikhil's original design
    described: the cushion graph IS the probe, queried against the
    Internet, not audited after the fact. The per-node search_queries
    were already being computed by composer.py:155-218 (Sonnet emits
    2-4 distinct queries per node, varying vocabulary so different
    queries surface different domains) — this function plugs that
    pre-computed structure into the actual retrieval path.
    """
    import random as _random_local

    layers_by_name = {
        "mechanism": cushion.mechanism,
        "essence":   cushion.essence,
        "actual":    cushion.actual,
    }
    # Drop layers with no nodes so the weighted draw doesn't pick an empty
    # layer when another has content. Recomputing weights keeps the
    # relative ratio between the remaining layers intact.
    filtered: list[tuple[str, float, "CushionLayer"]] = []
    for name, weight in _NODE_LAYER_WEIGHTS:
        lyr = layers_by_name.get(name)
        if lyr is not None and lyr.node_count() > 0:
            filtered.append((name, weight, lyr))
    if not filtered:
        return ("", "", "")

    names   = [n for n, _, _ in filtered]
    weights = [w for _, w, _ in filtered]
    layers  = [l for _, _, l in filtered]
    chosen_idx = _random_local.choices(range(len(filtered)), weights=weights, k=1)[0]
    chosen_layer_name = names[chosen_idx]
    chosen_layer      = layers[chosen_idx]

    records = chosen_layer.records_or_synth()
    if not records:
        return ("", chosen_layer_name, "")

    chosen_node = _random_local.choice(records)
    if chosen_node.search_queries:
        chosen_query = _random_local.choice(chosen_node.search_queries)
    else:
        # Legacy cushions without per-node search_queries — fall back to
        # the node text itself. Still much narrower than pursuit text.
        chosen_query = chosen_node.text

    return (str(chosen_query), chosen_layer_name, str(chosen_node.text))


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
            # Q5 run-#2-followup: surface bias_mode at INFO so per-call
            # JSONL audit + Railway logs can prove the 5 bias modes
            # diversified across the wander instead of collapsing to
            # one mode. Without this, run #2's bias-mode prediction (P3)
            # only had circumstantial evidence; run #3 makes it provable.
            log.info(
                "interpreter verdict bias_mode=%s decision=%s url=%s",
                verdict.bias_mode, verdict.decision, url or "<no-url>",
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

    # r9 Fix #2 — per-agent abandon history for the structural gate's
    # Layer 2 circuit breaker. Appended at the end of each dig: True
    # if the dig terminated at iter-1 via the gated ABANDON_DIG path,
    # False otherwise. The gate examines the FIRST N entries (not
    # last N) so a model-personality eager-abandon bias is caught
    # early and applied for the rest of the run.
    abandon_history: list[bool] = field(default_factory=list)

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


# ---------------------------------------------------------------------------
# Contribution board — the additive, positive-sum dig directive
# ---------------------------------------------------------------------------
#
# Injected into the dig user-message (after the anchor, before the source)
# only when WANDER_CONTRIBUTION_BOARD=1 AND peers have already posted notices.
# The framing is deliberate (see the long design discussion): ADDITIVE, not
# competitive — "add the layer they missed," never "beat them / don't be
# worthless." Competition optimizes for *looking* like you contributed
# (inflation, padding, flag-planting); contribution optimizes for the deeper,
# transferable structure. The anti-inflation paragraph + the still-binding
# what_does_not_map rule keep it honest.
_CONTRIBUTION_BOARD_PREAMBLE = """\
# CONTRIBUTION BOARD — what peers have already surfaced
Other agents in this room have already contributed the findings below. They
are NOT yours to repeat. Your value is the layer they have NOT reached: go
DEEPER — find the structural level beneath what is listed, the part that would
transfer to a DIFFERENT domain. Contribute what is MISSING, not what is
already on the board.

This is NOT a competition and there is no prize for volume. Do not inflate,
pad, or claim significance you have not established — one honest, deeper
structural bridge is worth more than ten loud surface ones. If your honest
find genuinely overlaps the board, say so plainly and dig past it. The
what_does_not_map rule still binds: stay honest.

Already on the board:
"""


def _build_contribution_block(notices: list) -> str:
    """Render recent peer notices into the dig CONTRIBUTION BOARD block.

    Returns "" when there are no peer notices yet (the first agent into a
    fresh room sees no board and digs normally). Each line is one peer's
    one-sentence finding plus its load-bearing principle, truncated. Pure
    string assembly — no LLM call, reads only already-posted notices.
    """
    lines = []
    for n in notices:
        summary = (getattr(n, "summary", "") or "").strip()
        if not summary:
            continue
        principle = (getattr(n, "principle", "") or "").strip()
        tail = f" — principle: {principle[:160]}" if principle else ""
        lines.append(f"  - {summary[:200]}{tail}")
    if not lines:
        return ""
    return _CONTRIBUTION_BOARD_PREAMBLE + "\n".join(lines)


_DIG_FIND_SYSTEM_PROMPT = """\
You are one wandering agent in Constellax's Wandering Room.

You have just fetched an article from a possibly-unrelated domain. The
metal detector beeped — some patches on the user's problem map lit up
against this article's content.

Your job RIGHT NOW is to do quick but thoughtful research. Not a one-shot
summary. A real read.

Think through these seven lenses BEFORE writing your output. The lenses
are reasoning scaffolding — they organize your thinking — they are NOT
mandatory output fields. Your output is the same four-field report as
always.

THE SEVEN LENSES (think through them; do not write them as headers):

  1. ORIENT. What is this article actually about? Plain terms, one
     sentence. Example: "This is about how stream-insect biodiversity
     collapses when one species crowds out variety under land-use
     intensification."

  2. IDENTIFY THE UNDERLYING MECHANISM. Not the surface topic — the
     structural machinery. Example: "Local sites lose variety when
     generalists outcompete specialists, and that propagates as
     homogenization across the network."

  3. FOR EACH PATCH THE MATCHER SAID LIT UP — SAY WHY in concrete terms.
     If a match is shallow, name it shallow. Don't hide thin matches.

  4. CROSS-WORLD ANALOGY. Now do real cross-world thinking. How does
     this article's mechanism MAP back to the user's problem? Name what
     corresponds to what — be specific, not abstract. This is where
     genuine analogical work happens.

  5. EXTRACT THE LOAD-BEARING PRINCIPLE. What general principle is at
     work here that the user could apply? One sentence.

  6. WHAT DOES NOT MAP. Where does the analogy honestly break? Required.

  7. NEXT LEAD. If the user wanted to dig further in this domain, where
     would you point them? Optional.

CRITICAL RULES (Constellax Laws):

  - The insight happens in the USER's head. You surface a structural
    bridge; you do not deliver a solution. Use "this resonates with…"
    not "this solves your problem…".
  - what_does_not_map is MANDATORY. Leaving it empty or writing
    "everything maps fine" → report rejected.
  - Honest about confidence. If only 1 patch lit up, label LOW.
  - Cite the source — title, brief excerpt, what role it played.

OUTPUT FORMAT — return ONE JSON object with the four standard fields:

{
  "exploration_summary": "<2-3 sentences: what you found, in human terms>",
  "advancement": "<1-2 sentences: how this resonates with the anchor>",
  "what_does_not_map": "<1-2 sentences: where the analogy breaks; MANDATORY>",
  "next_lead": "<optional: where to dig further if user wants more>"
}

This is ITERATION 1 of 2. A self-critique will run after this. Iteration
2 will revise based on that critique. So don't aim for perfect — aim for
specific and honest. The revision will sharpen what's worth sharpening.

No prose preamble. No code fences. Just JSON.
"""


_DIG_REVISE_SYSTEM_PROMPT = """\
You are the same wandering agent. You just produced ITERATION 1 of this
dig. A self-critique just ran on it. Now you revise.

YOU HAVE:
  - Your iteration 1 work (the four-field report you just wrote)
  - The self-critique's verdict + any red-flagged questions + summary
  - The same article and problem map you read in iteration 1

YOUR JOB — revise. Do these four things in order:

  1. READ THE CRITIQUE HONESTLY. What did it flag? What did it confirm?
     Were any of its red flags real?

  2. REVISE iteration 1's content. Keep what's solid, fix what's weak.
     - If critique flagged you for projecting structure that isn't there
       (Q3 red) → tighten or remove the projection.
     - If critique flagged you for deflecting from the user's actual
       problem (Q2 red) → pull harder on the structural connection.
     - If critique flagged you for generalizing instead of being specific
       → get specific.
     - If critique returned diminishing-returns concerns (Q4 red) → cut
       the rephrasing, keep only what's load-bearing.

  3. EXTEND where the critique opened a new thread. If it noticed
     something iteration 1 missed, surface it now.

  4. STRENGTHEN the cross-world analogy if iteration 1's was thin. Do
     genuine second-pass thinking, NOT rephrasing.

CRITICAL — PRESERVE THE VOCABULARY OF ITERATION 1.

  If iteration 1 used unusual, specific, or cross-domain language — terms
  like "attractor hazard", "negative topology", "crystallization from
  collision", "weak-signal accumulation", "phase transition" — KEEP THEM
  in iteration 2 unless they are structurally wrong (factually incorrect
  about the article).

  Do NOT sand iteration 1's prose down into safer, more conventional
  phrasing. The unusual vocabulary IS the signal of cross-domain
  analogical work. Replacing "attractor hazard" with "a tendency to get
  stuck" is a regression even when it sounds more polished. Polished
  prose with generic vocabulary is the failure mode — not the success
  mode.

HONEST DISAGREEMENT IS ALLOWED.

  If iteration 1 was right and the critique was off-base, write iteration
  2 essentially identical to iteration 1, and use the next_lead field
  to note something like "iteration 1 stands; the critique misread on
  point X." The system respects honest disagreement.

NAME THE SHARPEST JOINT — DO NOT LABEL IT, DO NOT REUSE BOILERPLATE.

  Your `advancement` must name the ONE point where THIS source and THIS
  anchor correspond most tightly, in the form:
    "<a specific feature of the source in front of you> IS <a specific
     anchor mechanism> — and this predicts <a concrete consequence>."
  Use the actual vocabulary of THIS source and THIS anchor. Never a stock
  phrase. TEST: if your advancement sentence could be pasted onto a
  DIFFERENT source unchanged, it is boilerplate, not a joint — rewrite it
  with this source's terms.
    WEAK (a label): "a proven design pattern for holding the divergence-
      convergence tension", "a mechanism for dual-signal fusion". The fix
      is NOT to swap in a different stock phrase — it is to name what, in
      THIS source, corresponds to what, in THIS anchor, and what that buys.

CALIBRATE — DO NOT INFLATE, DO NOT FORCE THE JOINT.

  One concrete, defensible correspondence beats three loose ones. If a
  mapping is partial, name which part holds and which part breaks. If THIS
  source does not actually contain a tight joint, say so plainly — a forced
  "X IS Y" on a source that doesn't support it is WORSE than an honest "the
  tightest correspondence here is only partial, because…". Never round a
  partial match up to a clean one.

OUTPUT FORMAT — same four-field JSON as iteration 1:

{
  "exploration_summary": "<2-3 sentences>",
  "advancement": "<1-2 sentences>",
  "what_does_not_map": "<1-2 sentences; MANDATORY>",
  "next_lead": "<optional>"
}

This is what ships as the final report. No prose preamble. No code
fences. Just JSON.
"""


async def _run_dig_find(
    cushion: CushionGraph,
    fetched: FetchResult,
    match: MatchResult,
    client: LLMClient,
) -> tuple[str, int, int]:
    """Run ITERATION 1 of the two-iteration dig (WANDER_DIG_REVISION_MODE).

    Same call shape as `_run_dig_iteration` but uses the seven-lens
    `_DIG_FIND_SYSTEM_PROMPT` (reasoning scaffolding in the system
    message, four-field JSON output preserved). Returns (raw_text,
    in_tokens, out_tokens).
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
    blocks.append("\n# YOUR TASK")
    blocks.append(
        "Work through the seven lenses in your head, then write the "
        "four-field dig report JSON. Be specific. Be honest about what "
        "doesn't map."
    )
    user_message = "\n".join(blocks)

    response: LLMResponse = await client.call(
        system_prompt=compose_system_prompt(
            _DIG_FIND_SYSTEM_PROMPT, mode="wandering_dig_find",
        ),
        user_message=user_message,
        domain="synthesizer",
        concept="wandering_dig_find",
    )
    return (
        response.content if response.success else "",
        response.input_tokens,
        response.output_tokens,
    )


async def _run_dig_revise(
    cushion: CushionGraph,
    fetched: FetchResult,
    match: MatchResult,
    iter1_payload: dict[str, str],
    critique_summary: str,
    critique_red_flags: list[str],
    critique_verdict: str,
    client: LLMClient,
) -> tuple[str, int, int]:
    """Run ITERATION 2 (revision) of the two-iteration dig.

    Receives iter-1's payload + the critique's verdict + red flags + a
    one-sentence summary. The system prompt guides the LLM to revise
    while preserving iter-1's unusual cross-domain vocabulary. Returns
    (raw_text, in_tokens, out_tokens).
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
    blocks.append("\n# YOUR ITERATION 1 OUTPUT (revise this)")
    blocks.append(json.dumps(iter1_payload, indent=2, ensure_ascii=False))
    blocks.append("\n# SELF-CRITIQUE FEEDBACK")
    blocks.append(f"Verdict: {critique_verdict}")
    if critique_red_flags:
        blocks.append(f"Red flags raised: {', '.join(critique_red_flags)}")
    else:
        blocks.append("Red flags raised: none")
    if critique_summary:
        blocks.append(f"Critique summary: {critique_summary}")
    blocks.append("\n# YOUR TASK")
    blocks.append(
        "Revise iteration 1 per the four steps in your instructions. "
        "Preserve unusual cross-domain vocabulary unless structurally "
        "wrong. Output the four-field JSON."
    )
    user_message = "\n".join(blocks)

    response: LLMResponse = await client.call(
        system_prompt=compose_system_prompt(
            _DIG_REVISE_SYSTEM_PROMPT, mode="wandering_dig_revise",
        ),
        user_message=user_message,
        domain="synthesizer",
        concept="wandering_dig_revise",
    )
    return (
        response.content if response.success else "",
        response.input_tokens,
        response.output_tokens,
    )


# Common English stopwords / cushion-vocabulary patterns to EXCLUDE when
# tallying "unusual" vocabulary across iterations. The lexical-preservation
# check is a safety net for the documented "Sonnet revision sands prose
# into conventional phrasing" risk — we want to detect when iter-2 dropped
# distinctive cross-domain terminology that iter-1 had.
_LEXICAL_STOPWORDS_BASIC: frozenset[str] = frozenset({
    "about", "above", "after", "again", "against", "all", "also", "and",
    "another", "any", "are", "around", "because", "been", "before",
    "being", "below", "between", "both", "but", "can", "could", "did",
    "does", "doing", "down", "during", "each", "either", "even", "every",
    "few", "for", "from", "further", "had", "has", "have", "having", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "however",
    "into", "its", "itself", "just", "like", "made", "make", "many",
    "may", "might", "more", "most", "much", "must", "near", "neither",
    "never", "non", "nor", "not", "now", "off", "once", "one", "only",
    "onto", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "since", "some", "such", "take", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "too", "two",
    "under", "until", "upon", "use", "used", "uses", "using", "very",
    "was", "way", "were", "what", "when", "where", "which", "while",
    "who", "whom", "whose", "why", "will", "with", "within", "without",
    "would", "you", "your", "yours", "yourself", "yourselves",
})


def _extract_unusual_vocabulary(
    text: str,
    cushion_text: str,
    min_word_len: int = 5,
) -> set[str]:
    """Extract content words from `text` that are:
      - at least `min_word_len` characters long
      - NOT in the basic stopword set
      - NOT already present in `cushion_text` (the agent's problem map)

    The intent is to surface words the agent BROUGHT IN from the source
    article that aren't in the user's own vocabulary — the cross-domain
    terminology whose preservation across iter-1 → iter-2 we want to
    measure.

    Returns a set of lowercase content words.
    """
    import re as _re_lp
    text_lower = (text or "").lower()
    cushion_words = {
        w for w in _re_lp.findall(r"[a-z]+", (cushion_text or "").lower())
        if len(w) >= min_word_len
    }
    out: set[str] = set()
    for w in _re_lp.findall(r"[a-z]+", text_lower):
        if len(w) < min_word_len:
            continue
        if w in _LEXICAL_STOPWORDS_BASIC:
            continue
        if w in cushion_words:
            continue
        out.add(w)
    return out


def _lexical_preservation_rate(
    iter1_payload: dict[str, str],
    iter2_payload: dict[str, str],
    cushion_text: str,
) -> tuple[float, int, int, list[str]]:
    """Compute the share of unusual cross-domain vocabulary from iter-1
    that survives into iter-2.

    Returns (preservation_rate, iter1_unusual_count, iter2_kept_count,
    dropped_terms_sample).

    Used as a safety-net diagnostic — logged on every revision dig.
    A low rate (≤ 0.5) flags lexical sanding (Sonnet's documented failure
    mode of conventionalizing prose during revision) for human review.
    """
    iter1_text = " ".join(filter(None, (
        iter1_payload.get("exploration_summary", ""),
        iter1_payload.get("advancement", ""),
        iter1_payload.get("what_does_not_map", ""),
        iter1_payload.get("next_lead", ""),
    )))
    iter2_text = " ".join(filter(None, (
        iter2_payload.get("exploration_summary", ""),
        iter2_payload.get("advancement", ""),
        iter2_payload.get("what_does_not_map", ""),
        iter2_payload.get("next_lead", ""),
    )))

    iter1_unusual = _extract_unusual_vocabulary(iter1_text, cushion_text)
    iter2_unusual = _extract_unusual_vocabulary(iter2_text, cushion_text)

    if not iter1_unusual:
        return (1.0, 0, 0, [])

    kept = iter1_unusual & iter2_unusual
    dropped = iter1_unusual - iter2_unusual
    rate = len(kept) / len(iter1_unusual)
    dropped_sample = sorted(dropped)[:8]
    return (rate, len(iter1_unusual), len(kept), dropped_sample)


async def _run_dig_iteration(
    cushion: CushionGraph,
    fetched: FetchResult,
    match: MatchResult,
    iteration_index: int,
    client: LLMClient,
    contribution_block: str = "",
) -> tuple[str, int, int]:
    """Run ONE iteration of dig analysis. Returns (raw_text, in_tokens, out_tokens).

    The text is JSON per the dig prompt. The agent loop accumulates these
    across the dig and builds a final ExplorationReport at the end.

    `contribution_block` (WANDER_CONTRIBUTION_BOARD) is the optional
    positive-sum "what peers already found, add the missing/deeper layer"
    block; "" (default) leaves the prompt byte-for-byte as the legacy path.
    """
    blocks = ["# ANCHOR", cushion.to_anchor_prompt()]
    if contribution_block:
        blocks.append("\n" + contribution_block)
    blocks += [
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
    confidence_cap: str | None = None,
) -> ExplorationReport:
    """Assemble an ExplorationReport from the dig's per-iteration payloads.

    The final iteration's payload is the canonical source for the four
    string fields (exploration_summary, advancement, what_does_not_map,
    next_lead). Earlier iterations are accessible via the trace.

    Confidence is computed from layer matches, then run through the
    domain-trust tiebreaker (trust.adjust_confidence). Trust can only
    PROMOTE a borderline match — it cannot demote a strong one. See
    WANDERING_ROOM_DECISIONS.md D7.

    r9 Fix #2 Layer 3 — `confidence_cap` ("medium" | "low" | None) caps
    the final confidence regardless of layer-match ratio AND regardless
    of trust adjustment. Used by the structural abandon gate to keep
    iter-1 honest abandonments from shipping at HIGH on a lucky match
    ratio (this is exactly the projection-leak failure r7 exhibited
    with the Nanao fan-site and Korean drama reports).
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

    # r9 Fix #2 Layer 3 — confidence cap applied AFTER trust adjustment.
    # report.confidence is a Confidence enum (str-Enum, values "low" /
    # "medium" / "high"). The cap parameter arrives as a plain string;
    # we MUST coerce it back to the enum before assignment, otherwise
    # downstream code that calls `.value` on it (runtime trace builder,
    # report serializer) blows up with `'str' object has no attribute
    # 'value'`. That class of crash silently dropped P01/P02 from the
    # r9 cohort — codex caught it; the fix is to round-trip through the
    # enum constructor on every assignment.
    if confidence_cap is not None:
        cap_order = {"low": 0, "medium": 1, "high": 2}
        if cap_order.get(report.confidence, 2) > cap_order.get(confidence_cap, 2):
            log.info(
                "report %s: confidence capped from %s to %s "
                "(iter-1 honest abandonment via gate)",
                report_id, report.confidence, confidence_cap,
            )
            report.confidence = Confidence(confidence_cap)

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

        # CONSTELLAX_GOVERNOR: the session-level flow governor may seize flow —
        # halting the wander early when the swarm has converged AND formed a
        # structured skeleton (a confirmed CLOSE). Checked here, alongside budget
        # exhaustion, so the agent finishes its current step and exits cleanly.
        # Flow only — the governor never touches a finding's content or quality.
        if (_use_governor() and state.session_state is not None
                and getattr(state.session_state, "governor_halt", False)):
            reason_g = getattr(state.session_state, "governor_halt_reason", "") or "governor_close"
            state.trace.append(TraceStep(
                step_id=0,
                kind=StepKind.EXHAUSTED,
                timestamp=clock(),
                rationale=f"governor halt: {reason_g}",
                tokens_spent=state.cumulative_tokens,
            ))
            state.trace.completion_reason = "governor_halt"
            state.trace.ended_at = clock()
            return state

        # 1. Policy decides next move
        # WANDER_AGENT_NOTICEBOARD: optionally pass peer-covered domains as
        # a soft hint to the domain picker. Covered domains get half-weight
        # in pick_next_domain — not excluded, just less likely. Notices
        # by the SAME agent are excluded from the read so an agent's
        # own past coverage doesn't double-discount.
        covered_domains: set[str] | None = None
        if _use_agent_noticeboard() and state.session_state is not None:
            recent = state.session_state.recent_notices(
                n=10, exclude_agent_id=state.agent_id,
            )
            if recent:
                covered_domains = {n.domain for n in recent if n.domain}
                log.info(
                    "noticeboard: agent=%s reading %d peer notices covering "
                    "domains=%s",
                    state.agent_id, len(recent), sorted(covered_domains),
                )

        move: NextMove = next_move(
            state.cushion, state.trace,
            noticeboard_covered_domains=covered_domains,
        )
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
        #
        # METAL-DETECTOR RETRIEVAL (WANDER_NODE_QUERY_RETRIEVAL=1):
        # Seed the search with a cushion-NODE-derived query instead of
        # the pursuit text. This is the original design Nikhil
        # specified: the graph constellation IS the metal detector, and
        # node-derived queries are the signal that drags content back
        # from unrelated domains by structural resonance rather than
        # topic similarity to the pursuit. Composer's Sonnet already
        # populates per-node search_queries (composer.py:155-218); this
        # path simply consumes them. Default off so r5-era runs stay
        # comparable as a baseline.
        if _use_node_query_retrieval():
            node_query, node_layer, node_text = _draw_node_query(state.cushion)
            if node_query:
                query_seed = node_query
                log.info(
                    "metal_detector: agent=%s domain=%s layer=%s "
                    "node=%r query=%r",
                    state.agent_id, move.position, node_layer,
                    node_text[:60], node_query[:80],
                )
            else:
                # Empty cushion across all layers — graceful degrade to
                # the pursuit text so the wander doesn't stall.
                query_seed = state.cushion.raw_input.problem.content
                log.warning(
                    "metal_detector: empty cushion graph, falling back to pursuit text",
                )
        else:
            query_seed = state.cushion.raw_input.problem.content

        fetched = await fetcher(
            move.position,
            query_seed,
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

        # 4. Dig: two paths based on WANDER_DIG_REVISION_MODE flag.
        #
        # PATH A (flag ON, June 2026 test scaffold) — Two-iteration design
        # Nikhil specified: structured "find" iteration followed by
        # self-critique-informed "revise" iteration. Each iteration has a
        # distinct goal and the chain is real (iter 2 reads iter 1 +
        # critique). Abandon-verdict is preserved as the kill switch so
        # Law 7 (honest abandonment > polished corpse) holds.
        #
        # PATH B (flag OFF, default) — Legacy 3-5 redundant iteration loop.
        # Same prompt run multiple times, only the last shipped. Kept as
        # production baseline until A/B confirms PATH A is denser AND
        # preserves cross-domain vocabulary.
        iteration_payloads: list[dict[str, str]] = []
        iterations_completed = 0
        abandoned_early = False
        # r9 Fix #2 — default no cap; PATH A sets this from the gate
        # when a true honest-abandonment passes all layers. PATH B
        # (legacy) leaves it None.
        confidence_cap_for_report: str | None = None

        if _use_dig_revision_mode():
            # ─── PATH A: find → critique → revise ────────────────────────
            #
            # r8 surveillance: timing captures around each LLM call + a
            # forensics entry written at the end of this block (when
            # WANDER_FORENSICS_PATH is set). These are pure
            # measurement — no behavior change. All writes are
            # try/except wrapped inside _write_forensics_entry.
            _t_dig_start = clock()
            sanding_reverted = False

            # Iter 1: structured 7-lens "find" pass.
            _t_find_start = clock()
            raw1, in1, out1 = await _run_dig_find(
                cushion=state.cushion,
                fetched=fetched,
                match=match,
                client=client,
            )
            _t_find_ms = (clock() - _t_find_start) * 1000.0
            state.cumulative_tokens += in1 + out1
            iter1_payload = _parse_dig_response(raw1)
            iteration_payloads.append(iter1_payload)
            iterations_completed += 1
            state.trace.append(TraceStep(
                step_id=0,
                kind=StepKind.DUG,
                timestamp=clock(),
                position=fetched.domain_hint,
                rationale="dig iter 1 (find — 7-lens scaffold)",
                iterations_used=1,
                tokens_spent=state.cumulative_tokens,
            ))
            state.steps_taken += 1

            # Critique runs as the ADVISORY default (r8 design — feeds
            # red flags into iter-2's revise step). r9 adds a STRUCTURAL
            # GATE on top: when the LLM verdict is ABANDON_DIG at iter-1
            # AND the gate's three layers all pass, the dig terminates
            # at iter-1 with a confidence cap (Law 7 honest abandonment
            # restored — see critique.py enforce_abandon_gate).
            #
            # Most digs (CONTINUE / RTA / HAND_OFF / gate-demoted
            # abandonments) still run iter-2 as r8 design.
            _t_crit_start = clock()
            critique = await run_self_critique(
                cushion=state.cushion,
                agent_position=fetched.domain_hint,
                latest_finding=iter1_payload.get("exploration_summary", ""),
                cumulative_tokens=state.cumulative_tokens,
                iterations_so_far=1,
                client=client,
            )
            _t_crit_ms = (clock() - _t_crit_start) * 1000.0

            # r9 Fix #2 — apply structural gate to the LLM verdict. The
            # gate may demote ABANDON_DIG → RETURN_TO_ANCHOR or CONTINUE
            # based on (a) Q3+Q4 co-firing requirement, (b) per-agent
            # circuit breaker on eager-abandon model bias. The gate
            # also returns confidence_cap when a true honest-abandonment
            # passes all layers.
            gate = enforce_abandon_gate(
                critique,
                iteration_so_far=1,
                abandon_history=list(state.abandon_history),
            )
            confidence_cap_for_report: str | None = gate.confidence_cap

            # r8 surveillance: enrich the trace step's detail field with
            # red flags + Q1-Q6 answers so the result JSON carries the
            # full critique reasoning. r9 also surfaces the gate action.
            _critique_detail_bits = [critique.summary]
            if critique.red_flags:
                _critique_detail_bits.append(
                    f"red_flags={','.join(critique.red_flags)}"
                )
            if gate.gate_action != "passed":
                _critique_detail_bits.append(
                    f"gate={gate.gate_action} "
                    f"(orig={gate.original_verdict.value}, "
                    f"final={gate.verdict.value})"
                )
            for _qkey in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
                _ans = critique.answers.get(_qkey, "").strip()
                if _ans:
                    _critique_detail_bits.append(f"{_qkey}: {_ans}")
            state.trace.append(TraceStep(
                step_id=0,
                kind=StepKind.SELF_CRITIQUED,
                timestamp=clock(),
                position=fetched.domain_hint,
                rationale=(
                    f"critique verdict: {critique.verdict.value} "
                    f"→ gated: {gate.verdict.value}"
                ),
                detail=" | ".join(_critique_detail_bits),
                tokens_spent=state.cumulative_tokens,
            ))
            state.steps_taken += 1

            # r9 branch: if the gate produces a true honest-abandonment,
            # ship iter-1 with a MEDIUM confidence cap and skip iter-2.
            # Else run iter-2 as r8 design.
            iter2_was_skipped = (gate.verdict == CritiqueVerdict.ABANDON_DIG)
            if iter2_was_skipped:
                abandoned_early = True
                # Placeholder values so the forensics write below stays
                # well-formed even when iter-2 didn't run.
                iter2_payload: dict[str, str] = {}
                in2, out2 = 0, 0
                _t_revise_ms = 0.0
                rate, total_unusual, kept_count, dropped_sample = (
                    1.0, 0, 0, [],
                )
                state.trace.append(TraceStep(
                    step_id=0,
                    kind=StepKind.ABANDONED,
                    timestamp=clock(),
                    position=fetched.domain_hint,
                    rationale=(
                        f"honest abandonment via gate "
                        f"({gate.gate_action}); shipping iter-1 "
                        f"with confidence_cap={confidence_cap_for_report}"
                    ),
                    iterations_used=1,
                    tokens_spent=state.cumulative_tokens,
                ))
                state.steps_taken += 1
                log.info(
                    "gated_abandon: agent=%s domain=%s action=%s "
                    "red_flags=%s",
                    state.agent_id, fetched.domain_hint,
                    gate.gate_action, list(critique.red_flags),
                )

            else:
                # iter-2 runs as r8 design.
                _t_revise_start = clock()
                raw2, in2, out2 = await _run_dig_revise(
                    cushion=state.cushion,
                    fetched=fetched,
                    match=match,
                    iter1_payload=iter1_payload,
                    critique_summary=critique.summary,
                    critique_red_flags=list(critique.red_flags),
                    critique_verdict=critique.verdict.value,
                    client=client,
                )
                _t_revise_ms = (clock() - _t_revise_start) * 1000.0
                state.cumulative_tokens += in2 + out2
                iter2_payload = _parse_dig_response(raw2)
                iteration_payloads.append(iter2_payload)
                iterations_completed += 1
                state.trace.append(TraceStep(
                    step_id=0,
                    kind=StepKind.DUG,
                    timestamp=clock(),
                    position=fetched.domain_hint,
                    rationale=(
                        f"dig iter 2 (revise — critique-informed; "
                        f"verdict was {critique.verdict.value} "
                        f"→ gated {gate.verdict.value})"
                    ),
                    iterations_used=2,
                    tokens_spent=state.cumulative_tokens,
                ))
                state.steps_taken += 1

                # Lexical-preservation diagnostic + AUTO-REVERT on sanding.
                # r9 Fix #3 — thresholds tightened from r8 (rate < 0.5
                # AND total_unusual >= 4) to rate < 0.35 AND total_unusual
                # >= 6, PLUS a what_does_not_map preservation override:
                # do NOT revert when iter-2's what_does_not_map field is
                # genuinely longer than iter-1's (signals iter-2 is
                # doing real honest-limit work, not sanding). r8 fired
                # 11 reverts of which ~9 were false positives — these
                # tighter triggers should bring false-positive rate
                # below 30%.
                cushion_text = state.cushion.to_anchor_prompt()
                rate, total_unusual, kept_count, dropped_sample = (
                    _lexical_preservation_rate(
                        iter1_payload, iter2_payload, cushion_text,
                    )
                )
                log.info(
                    "dig_revision: agent=%s domain=%s "
                    "preservation_rate=%.2f (iter1_unusual=%d kept=%d) "
                    "dropped_sample=%r",
                    state.agent_id, fetched.domain_hint, rate,
                    total_unusual, kept_count, dropped_sample,
                )
                _iter1_wnm_len = len(iter1_payload.get("what_does_not_map", "") or "")
                _iter2_wnm_len = len(iter2_payload.get("what_does_not_map", "") or "")
                _wnm_preserved = _iter2_wnm_len > _iter1_wnm_len
                if rate < 0.35 and total_unusual >= 6 and not _wnm_preserved:
                    log.warning(
                        "dig_revision: LEXICAL_SANDING_DETECTED "
                        "agent=%s domain=%s rate=%.2f "
                        "total_unusual=%d kept=%d "
                        "iter1_wnm=%d iter2_wnm=%d "
                        "→ AUTO-REVERTING to iter-1",
                        state.agent_id, fetched.domain_hint, rate,
                        total_unusual, kept_count,
                        _iter1_wnm_len, _iter2_wnm_len,
                    )
                    # Swap iter-1 back as the canonical payload. The trace
                    # above still shows iter-2 fired; this trace step
                    # records the auto-revert so analysis can see it.
                    iteration_payloads[-1] = iter1_payload
                    sanding_reverted = True
                    state.trace.append(TraceStep(
                        step_id=0,
                        kind=StepKind.DUG,
                        timestamp=clock(),
                        position=fetched.domain_hint,
                        rationale=(
                            f"sanding auto-revert (rate={rate:.2f}, "
                            f"kept={kept_count}/{total_unusual}, "
                            f"wnm i1={_iter1_wnm_len} i2={_iter2_wnm_len})"
                        ),
                        iterations_used=2,
                        tokens_spent=state.cumulative_tokens,
                    ))
                    state.steps_taken += 1
                elif rate < 0.35 and total_unusual >= 6 and _wnm_preserved:
                    log.info(
                        "dig_revision: sanding-thresholds-tripped but "
                        "what_does_not_map preserved (i1=%d i2=%d) — "
                        "iter-2 ships",
                        _iter1_wnm_len, _iter2_wnm_len,
                    )

            # Both branches converge here. Record this dig in the agent's
            # abandon-history for Layer 2 of the gate (per-agent circuit
            # breaker). True iff the gate fired a true honest-abandonment.
            state.abandon_history.append(iter2_was_skipped)

            # r8 SURVEILLANCE: structured forensics entry for this dig.
            # No behavior change — pure observability. Try/except wrapped
            # inside _write_forensics_entry; a write failure never reaches
            # the pipeline. Skipped entirely when WANDER_FORENSICS_PATH
            # is unset (production default).
            # r9 shipped-source resolution: gated-abandon ships iter-1
            # (iter-2 never ran); sanding-revert also ships iter-1
            # (iter-2 ran but was reverted); otherwise iter-2 ships.
            if iter2_was_skipped:
                _shipped_source = "iter1_gated_abandon"
            elif sanding_reverted:
                _shipped_source = "iter1_sanding_revert"
            else:
                _shipped_source = "iter2"
            _write_forensics_entry({
                "session_id": state.session_state.session_id
                    if state.session_state else None,
                "agent_id": state.agent_id,
                "dig_index": len(state.reports) + 1,
                "timestamp_start": _t_dig_start,
                "timestamp_end": clock(),
                "duration_ms": (clock() - _t_dig_start) * 1000.0,
                "domain": fetched.domain_hint,
                "fetched": {
                    "url": fetched.url,
                    "title": fetched.title,
                    "body_len": len(fetched.body or ""),
                },
                "match": {
                    "total_matched_nodes": match.total_matched_nodes,
                    "total_cushion_nodes": (
                        getattr(match, "total_cushion_nodes", None)
                    ),
                },
                # r9 Fix #2 — full gate decision for forensic
                # reconstruction of why iter-1 abandonment fired (or
                # was demoted).
                "gate": {
                    "original_verdict": gate.original_verdict.value,
                    "final_verdict":    gate.verdict.value,
                    "gate_action":      gate.gate_action,
                    "confidence_cap":   gate.confidence_cap,
                    "iter2_was_skipped": iter2_was_skipped,
                    "abandon_history_so_far": list(state.abandon_history),
                },
                "iter1": {
                    "exploration_summary": iter1_payload.get("exploration_summary", ""),
                    "advancement": iter1_payload.get("advancement", ""),
                    "what_does_not_map": iter1_payload.get("what_does_not_map", ""),
                    "next_lead": iter1_payload.get("next_lead", ""),
                    "exploration_summary_len": len(iter1_payload.get("exploration_summary", "")),
                    "advancement_len": len(iter1_payload.get("advancement", "")),
                    "what_does_not_map_len": len(iter1_payload.get("what_does_not_map", "")),
                    "next_lead_len": len(iter1_payload.get("next_lead", "")),
                    "timing_ms": _t_find_ms,
                    "in_tokens": in1,
                    "out_tokens": out1,
                },
                "critique": {
                    "verdict": critique.verdict.value,
                    "red_flags": list(critique.red_flags),
                    "summary": critique.summary,
                    "answers": dict(critique.answers),
                    "timing_ms": _t_crit_ms,
                },
                "iter2": {
                    "exploration_summary": iter2_payload.get("exploration_summary", ""),
                    "advancement": iter2_payload.get("advancement", ""),
                    "what_does_not_map": iter2_payload.get("what_does_not_map", ""),
                    "next_lead": iter2_payload.get("next_lead", ""),
                    "exploration_summary_len": len(iter2_payload.get("exploration_summary", "")),
                    "advancement_len": len(iter2_payload.get("advancement", "")),
                    "what_does_not_map_len": len(iter2_payload.get("what_does_not_map", "")),
                    "next_lead_len": len(iter2_payload.get("next_lead", "")),
                    "timing_ms": _t_revise_ms,
                    "in_tokens": in2,
                    "out_tokens": out2,
                },
                "preservation": {
                    "rate": rate,
                    "total_unusual": total_unusual,
                    "kept_count": kept_count,
                    "dropped_sample": list(dropped_sample) if dropped_sample else [],
                },
                "sanding_reverted": sanding_reverted,
                "shipped_payload_source": _shipped_source,
                "cumulative_tokens_after_dig": state.cumulative_tokens,
            })

            # PATH A handles its own critique loop above; skip the legacy
            # iteration loop below by jumping to the report build step.
            # (The legacy code path is the else: branch.)
            # → fall through to step 6 (_build_report_from_dig)

        else:
          # ─── PATH B (legacy default): redundant iteration loop ─────────
          # Contribution board (WANDER_CONTRIBUTION_BOARD): read peers' posted
          # findings once and inject the additive "go deeper / add what's
          # missing" block into every iteration of this dig. "" when off or
          # when no peer has posted yet — legacy prompt then.
          contribution_block = ""
          if _use_contribution_board() and state.session_state is not None:
              _peer_notes = state.session_state.recent_notices(
                  n=8, exclude_agent_id=state.agent_id,
              )
              contribution_block = _build_contribution_block(_peer_notes)
          for iteration_idx in range(match.dig_iterations):
            raw, in_toks, out_toks = await _run_dig_iteration(
                cushion=state.cushion,
                fetched=fetched,
                match=match,
                iteration_index=iteration_idx,
                client=client,
                contribution_block=contribution_block,
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
            confidence_cap=confidence_cap_for_report,
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

        # 6a. Noticeboard post (WANDER_AGENT_NOTICEBOARD test scaffold).
        # Append a short heads-up notice to the shared session noticeboard
        # so other agents see what's been covered before picking their
        # next domain. Pure informational — does NOT enter dig content,
        # does NOT trigger cross-agent critique. Each report remains an
        # independent sample for the synthesizer. Notice content is
        # derived from existing report fields — no extra LLM call.
        if (_use_agent_noticeboard() or _use_contribution_board()) and state.session_state is not None:
            from src.wandering.session_state import AgentNotice as _AgentNotice
            mn = match.total_matched_nodes
            strength = "weak" if mn <= 1 else ("moderate" if mn == 2 else "strong")
            notice = _AgentNotice(
                agent_id=state.agent_id,
                domain=fetched.domain_hint or "(unspecified)",
                match_strength=strength,
                summary=(report.exploration_summary or "")[:200],
                principle=(report.advancement or "")[:200],
                direction=(report.next_lead or "")[:200],
                timestamp=clock(),
            )
            await state.session_state.post_notice(notice)
            log.info(
                "noticeboard: agent=%s posted notice domain=%s strength=%s "
                "summary=%r",
                state.agent_id, notice.domain, notice.match_strength,
                notice.summary[:120],
            )

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
