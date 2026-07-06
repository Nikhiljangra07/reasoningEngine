"""
Route dispatcher — single entry point that wires triage + capabilities +
budget + engine into one flow.

Flow per request:
    1. Run triage gate (cheap classifier — Gemini Flash-Lite in live mode)
    2. Construct per-request budget tracker + capability registry
    3. Branch on triage.route:
         - TRIVIAL      → canned response, no LLM call
         - DIRECT       → single LLM call (Sonnet)
         - DIRECT_PLUS  → single LLM call + bridge retrieval
         - DEEP         → full wuxing engine at resolved effort tier
    4. Effort resolution (DEEP only):
         - gate.effort ≤ user.effort  → use user.effort
         - gate.effort > user.effort  → policy decides:
             strict → cap at user.effort
             auto   → escalate to gate.effort
             ask    → return escalation_offer; engine NOT run yet
    5. Return DispatchResult with response text + budget summary + capability state

What this step does NOT include (deferred to later steps):
    - Resume-after-escalation-accepted mechanism (Step 5)
    - AUTO effort mode + budget-bounded iteration extension (Step 6)
    - 27 universal questions injected into prompts (Step 7)
    - Real MCP fan-out (Step 8)

ISOLATION: imports from src.llm.*, src.capabilities.*, src.bridge.*,
src.core.types. No reverse imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.bridge.conversation_store import ConversationStore
from src.capabilities import CapabilityRegistry
from src.core.types import Direction, FrameworkID, Problem, Variable
from src.identity import MAP_NOT_MARCH_THRESHOLD, compose_system_prompt, gate_output_async
from src.identity.sovereignty import AntiLawKind, get_anti_law
from src.identity.voice.lint import LintContext
from src.llm.budget import BudgetCaps, BudgetTracker
from src.llm.checklist import build_checklist_block, get_checklist_codes
from src.llm.client import LLMClient
from src.llm.effort import Effort, iterations_for, normalize_effort
from src.llm.engine import EngineResult, run_async_formation
from src.llm.speech import extract_speech_input, generate_speech
from src.llm.triage import Route, TriageResult, triage
from src.mcp_router import (
    McpHandlerRegistry,
    McpResult,
    fire_mcp,
    format_mcp_results_for_prompt,
)


# ---------------------------------------------------------------------------
# TRIVIAL responses — keyed by pattern
# ---------------------------------------------------------------------------

_TRIVIAL_RESPONSES: dict[str, str] = {
    "greeting": "Hi. What's on your mind?",
    "thanks": "You got it.",
    "ack": "Got it.",
    "yesno": "Noted.",
    "farewell": "Talk soon.",
    "presence": "I'm here. What do you need?",
    "default": "Got it. What would you like to think through?",
}


def _classify_trivial(text: str) -> str:
    """Map a trivial message to a canned-response key."""
    lower = text.strip().lower()
    if not lower:
        return "default"
    if lower in ("yes", "no", "yeah", "nah", "yep", "nope"):
        return "yesno"
    if any(lower.startswith(w) for w in ("hi", "hello", "hey", "sup", "yo")):
        return "greeting"
    if any(w in lower for w in ("thanks", "thank you", "ty", "thx")):
        return "thanks"
    if any(w in lower for w in ("got it", "okay", "cool", "fine", "sure", "ok")):
        return "ack"
    if any(w in lower for w in ("bye", "goodbye", "see ya", "later", "cya")):
        return "farewell"
    if "you there" in lower or "are you there" in lower or lower == "hello?":
        return "presence"
    return "default"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Outcome of dispatching one request."""
    route: Route
    response_text: str
    triage_result: TriageResult
    budget_summary: dict
    engine_result: EngineResult | None = None
    capability_state: dict = field(default_factory=dict)
    escalation_offer: dict | None = None
    missing_capability_offers: list[dict] = field(default_factory=list)
    debug: dict = field(default_factory=dict)
    # Phase 2: structured memo emitted by the synthesizer (verdict line,
    # reasoning items, alternatives, falsifiers, open questions,
    # confidence, visuals). Only populated for DEEP route. The frontend's
    # Thinking Room + Map Room render from this directly when present and
    # fall back to client-side parsing of response_text when None.
    memo: dict | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def dispatch(
    text: str,
    client: LLMClient,
    user_effort: Effort | str | None = Effort.MEDIUM,
    policy: str = "ask",
    caps: BudgetCaps | None = None,
    registry: CapabilityRegistry | None = None,
    force_triage_result: TriageResult | None = None,
    conversation_store: ConversationStore | None = None,
    session_id: str | None = None,
    memory_directive: str = "",
    mcp_handlers: McpHandlerRegistry | None = None,
) -> DispatchResult:
    """
    Route a request through the appropriate path.

    Parameters:
        text: user message
        client: LLMClient (live or mock)
        user_effort: user's preferred effort tier; defaults to MEDIUM
        policy: "strict" | "ask" | "auto" — governs escalation behavior
        caps: budget caps; defaults to BudgetCaps()
        registry: capability registry; defaults to fresh CapabilityRegistry()
        force_triage_result: skip live classification (testing convenience)
        conversation_store: if provided AND session_id is also provided,
            every dispatched request is auto-recorded as an Iteration in
            the store. Both must be set to enable recording — either alone
            is a no-op (back-compat with callers that don't pass them).
        session_id: the conversation session this request belongs to.
            See conversation_store above for the recording contract.
        mcp_handlers: optional McpHandlerRegistry mapping capability name
            → real async handler. When provided, fire_mcp() dispatches to
            the matching handler instead of returning a stub; the
            dispatcher then folds the handler results into the prompt
            via format_mcp_results_for_prompt. When None, the router
            returns stub results (the pre-Phase-B contract) and nothing
            gets injected into prompts. Callers that don't wire handlers
            see no behavior change.
    """
    triage_result = (
        force_triage_result
        if force_triage_result is not None
        else await triage(text, client=client)
    )

    budget = BudgetTracker(caps=caps)
    reg = registry or CapabilityRegistry()
    user_eff = normalize_effort(user_effort)

    route = triage_result.route

    if route == Route.TRIVIAL:
        # Trivial responses are canned — no LLM, no memory needed.
        result = _dispatch_trivial(text, triage_result, budget, reg)
    elif route == Route.DIRECT:
        result = await _dispatch_direct(
            text, client, triage_result, budget, reg,
            memory_directive=memory_directive,
            conversation_store=conversation_store,
            session_id=session_id,
        )
    elif route == Route.DIRECT_PLUS:
        result = await _dispatch_direct_plus(
            text, client, triage_result, budget, reg,
            memory_directive=memory_directive,
            mcp_handlers=mcp_handlers,
            conversation_store=conversation_store,
            session_id=session_id,
        )
    else:
        # DEEP route — memory injection appends to the speech extra_directives
        # alongside the deep-checklist prompt.
        result = await _dispatch_deep(
            text, client, triage_result, budget, reg,
            user_effort=user_eff, policy=policy,
            memory_directive=memory_directive,
            mcp_handlers=mcp_handlers,
        )

    # Conversation recording — opt-in, back-compat. If either arg is missing
    # the dispatch returns exactly the same shape as before. Recording never
    # fails the request — a bad session_id is logged into debug and the
    # response still ships.
    await _record_iteration(
        result, text, triage_result, conversation_store, session_id,
    )

    return result


async def _record_iteration(
    result: DispatchResult,
    text: str,
    triage_result: TriageResult,
    conversation_store: ConversationStore | None,
    session_id: str | None,
) -> None:
    """
    Append the just-completed request to the conversation store as an
    Iteration. No-op if either arg is missing. KeyError on unknown
    session is captured into result.debug rather than raised — recording
    must never break the user-facing response.
    """
    if conversation_store is None or session_id is None:
        return
    try:
        iteration = await conversation_store.add_iteration(
            session_id=session_id,
            user_text=text,
            engine_response=result.response_text,
            route=result.route.value,
            effort=triage_result.recommended_effort.value,
        )
        if result.debug is None:
            result.debug = {}
        result.debug["iteration_id"] = iteration.id
        result.debug["session_id"] = session_id
    except KeyError as e:
        if result.debug is None:
            result.debug = {}
        result.debug["iteration_record_error"] = str(e)


# ---------------------------------------------------------------------------
# Resume — called after the user responds to an escalation_offer
# ---------------------------------------------------------------------------

async def resume_with_choice(
    text: str,
    client: LLMClient,
    accepted_effort: Effort | str,
    caps: BudgetCaps | None = None,
    registry: CapabilityRegistry | None = None,
    conversation_store: ConversationStore | None = None,
    session_id: str | None = None,
    mcp_handlers: McpHandlerRegistry | None = None,
) -> DispatchResult:
    """
    Run dispatch at the effort the user explicitly authorized.

    Called after the frontend shows an escalation_offer and the user
    picks "accept" (effort goes up) or "decline" (effort stays).
    Either way, the caller passes the chosen effort here and we run
    with policy="strict" — no further escalation, no further offers.

    Optional conversation_store + session_id propagate through to dispatch()
    so the resumed call gets recorded as another Iteration in the same
    session (continuing the thread, not branching). mcp_handlers
    propagate the same way so handler-driven MCP results land in the
    resumed call's prompt too.
    """
    return await dispatch(
        text=text,
        client=client,
        user_effort=accepted_effort,
        policy="strict",
        caps=caps,
        registry=registry,
        conversation_store=conversation_store,
        session_id=session_id,
        mcp_handlers=mcp_handlers,
    )


# ---------------------------------------------------------------------------
# Identity-layer enforcement helpers (0.3.4)
# ---------------------------------------------------------------------------

def _map_not_march_enabled() -> bool:
    """Feature flag for the MapNotMarch cartography directive.

    Read on every dispatch (not cached at module import) so operators
    can flip the env var without redeploying. Default is enabled —
    Codex's enforcement push lands as on-by-default. Operators can
    disable fast with `CONSTELLAX_MAP_NOT_MARCH=0`."""
    return os.getenv("CONSTELLAX_MAP_NOT_MARCH", "1") != "0"


def _maybe_cartography_directive(
    text: str,
    conversation_store: ConversationStore | None,
    session_id: str | None,
) -> str:
    """Return the cartography directive when the user has restated this
    position past the Map-Not-March threshold; empty string otherwise.

    The directive matches the AntiLaw NO_ARGUMENT remediation text so
    the wording stays in sync with the doctrine. The check is
    defensive: missing store, missing session_id, missing counter
    accessor, or any unexpected error returns empty string (no
    directive). That keeps the worst case "no behavior change" rather
    than "directive fires when it shouldn't."

    Why the check happens BEFORE add_iteration records this turn:
    `map_not_march_strike(session_id, text)` returns the count of
    PRIOR recordings. If the user is now sending their (N+1)th
    statement of the same position, the count is N. With
    MAP_NOT_MARCH_THRESHOLD=2, the directive fires on the 3rd+
    statement — exactly the "stated once, restated once, restated
    again" sequence the doctrine warns about."""

    if not _map_not_march_enabled():
        return ""
    if conversation_store is None or not session_id:
        return ""
    if not text or not text.strip():
        return ""
    try:
        strike = conversation_store.map_not_march_strike(session_id, text)
    except AttributeError:
        # Older ConversationStore / mock without the accessor.
        return ""
    except Exception:  # pragma: no cover — defensive
        return ""
    if strike < MAP_NOT_MARCH_THRESHOLD:
        return ""

    remediation = get_anti_law(AntiLawKind.NO_ARGUMENT).remediation
    return (
        "CARTOGRAPHY MODE (identity enforcement — Map-Not-March "
        f"threshold reached: {strike} prior restatements of this "
        f"position).\n{remediation}"
    )


# ---------------------------------------------------------------------------
# Per-route handlers
# ---------------------------------------------------------------------------

def _dispatch_trivial(
    text: str,
    triage_result: TriageResult,
    budget: BudgetTracker,
    registry: CapabilityRegistry,
) -> DispatchResult:
    """Canned response. No LLM call. ~1ms."""
    key = _classify_trivial(text)
    return DispatchResult(
        route=Route.TRIVIAL,
        response_text=_TRIVIAL_RESPONSES.get(key, _TRIVIAL_RESPONSES["default"]),
        triage_result=triage_result,
        budget_summary=budget.summary(),
        capability_state=_summarize_capabilities(registry),
        debug={"trivial_key": key},
    )


async def _dispatch_direct(
    text: str,
    client: LLMClient,
    triage_result: TriageResult,
    budget: BudgetTracker,
    registry: CapabilityRegistry,
    *,
    memory_directive: str = "",
    conversation_store: ConversationStore | None = None,
    session_id: str | None = None,
) -> DispatchResult:
    """Single LLM call. No retrieval. ~3s on live."""
    check = budget.check()
    if not check.allowed:
        return _budget_breach_result(
            triage_result, budget, registry, Route.DIRECT, check.reason
        )

    # Local mode-specific prompt — direct route asks for a short, decisive
    # answer with a guardrail around irreversible intents. Identity (no
    # therapy, no execution-promise, failure-mode attached, etc.) is
    # supplied by `compose_system_prompt` below; this block only carries
    # the direct-route-specific guidance.
    local_prompt = (
        "Direct-route guidance: give a short, decisive answer. No "
        "hedging, no 'on the other hand'. Pick a side. If the user has "
        "stated intent to do something irreversible, flag what they "
        "should think through BEFORE commenting on the action itself."
    )
    # Identity-layer ENFORCEMENT (0.3.4): when the Map-Not-March
    # counter sees a position restated past threshold, append the
    # cartography directive so the response switches from arguing to
    # mapping paths. Helper returns "" when the counter is below
    # threshold or unavailable — direct dispatch stays unchanged in
    # the steady state.
    cartography = _maybe_cartography_directive(text, conversation_store, session_id)
    if cartography:
        local_prompt = local_prompt + "\n\n" + cartography
    checklist = build_checklist_block(Route.DIRECT, Effort.LOW)
    if checklist:
        local_prompt = local_prompt + "\n\n" + checklist
    if memory_directive:
        # Phase 4 wiring: prior memory block (Decision Trace recall).
        # Appended last so it's the most recent context the model sees
        # before the user message.
        local_prompt = local_prompt + "\n\n" + memory_directive

    system_prompt = compose_system_prompt(local_prompt, mode="direct")

    response = await client.call(
        system_prompt=system_prompt,
        user_message=text,
        domain="synthesizer",       # routes to Sonnet 4.6 via provider_map
        concept="direct_answer",
    )
    budget.record_llm_response(response)

    # Output gate — run user-facing prose through strip + lint, and
    # regenerate once with a stronger directive if any blocking rule
    # fires. The regenerate is a budget-paid second model call; the
    # gate is async-only so we can await it without blocking the loop.
    if response.success and response.content:
        async def _regen_direct(directive: str) -> str:
            resp = await client.call(
                system_prompt=system_prompt + "\n\n" + directive,
                user_message=text,
                domain="synthesizer",
                concept="direct_answer",
            )
            budget.record_llm_response(resp)
            return resp.content if resp.success else ""

        gated = await gate_output_async(
            response.content,
            regenerate_fn=_regen_direct,
            context=LintContext(),
        )
        response_text = gated.text
    else:
        response_text = f"[direct call failed: {response.error}]"

    return DispatchResult(
        route=Route.DIRECT,
        response_text=response_text,
        triage_result=triage_result,
        budget_summary=budget.summary(),
        capability_state=_summarize_capabilities(registry),
    )


async def _dispatch_direct_plus(
    text: str,
    client: LLMClient,
    triage_result: TriageResult,
    budget: BudgetTracker,
    registry: CapabilityRegistry,
    *,
    memory_directive: str = "",
    mcp_handlers: McpHandlerRegistry | None = None,
    conversation_store: ConversationStore | None = None,
    session_id: str | None = None,
) -> DispatchResult:
    """
    Single LLM call + bridge retrieval.

    MCPs that triage requested are fired through the router. Blocked
    capabilities (MISSING / DENIED / consent-pending) surface as
    conversational missing-capability offers. Successful real-handler
    fires get folded into the prompt as an MCP CONTEXT block (Phase B1).
    Stub fires count toward the budget but don't inject prompt context
    — they're placeholders until a handler is registered for that name.
    """
    check = budget.check()
    if not check.allowed:
        return _budget_breach_result(
            triage_result, budget, registry, Route.DIRECT_PLUS, check.reason
        )

    # Fire each MCP the triage gate asked for through the router. Blocked
    # capabilities surface as conversational missing-capability offers;
    # successful fires count toward the budget's mcp_calls cap and are
    # noted in the debug log.
    missing_offers: list[dict] = []
    mcp_fired: list[dict] = []
    mcp_results: list[McpResult] = []
    for mcp in triage_result.mcps_needed:
        purpose = mcp.why or "context for this question"
        result = await fire_mcp(
            registry=registry,
            name=mcp.name,
            purpose=purpose,
            handlers=mcp_handlers,
        )
        # Stamp the purpose into the handler result so the prompt
        # renderer can echo it back when listing the MCP context block.
        if result.ok and not result.stub:
            result.result.setdefault("_purpose", purpose)
        mcp_results.append(result)
        mcp_fired.append({
            "name": mcp.name,
            "ok": result.ok,
            "stub": result.stub,
            "notified_user": result.notified_user,
            "blocked_reason": result.blocked_reason,
        })
        if not result.ok:
            missing_offers.append(
                registry.build_missing_capability_response(
                    mcp.name,
                    why_needed=mcp.why or "context for this question",
                    current_confidence=0.65,
                    confidence_if_connected=0.90,
                    fallback_caveat=(
                        "Working from your message alone — no prior project context."
                    ),
                )
            )
        else:
            # Successful fire counts toward the budget's MCP cap.
            budget.record_mcp_call(mcp.name)

    # Local mode-specific prompt — direct_plus answers from memory /
    # MCP context the user is asking about. Identity rules (no therapy,
    # no execution promise, failure mode, etc.) come from
    # `compose_system_prompt`; this block only carries direct_plus
    # guidance.
    local_prompt = (
        "Direct-plus guidance: the user is asking about prior decisions "
        "or project context. Answer from what's available. If context "
        "is missing or you don't have access to it, say so plainly "
        "rather than making up details. Be direct."
    )
    # Identity-layer ENFORCEMENT (0.3.4): cartography directive when
    # the Map-Not-March counter is past threshold. See helper docstring.
    cartography = _maybe_cartography_directive(text, conversation_store, session_id)
    if cartography:
        local_prompt = local_prompt + "\n\n" + cartography
    checklist = build_checklist_block(Route.DIRECT_PLUS, Effort.LOW)
    if checklist:
        local_prompt = local_prompt + "\n\n" + checklist
    if memory_directive:
        # Phase 4 — Decision Trace prior memory.
        local_prompt = local_prompt + "\n\n" + memory_directive
    # Phase B1 — fold real MCP handler results into the prompt. Stubs
    # and blocked fires are filtered out by format_mcp_results_for_prompt;
    # empty string when no real results, so the concat is no-op-safe.
    mcp_block = format_mcp_results_for_prompt(mcp_results)
    if mcp_block:
        local_prompt = local_prompt + "\n\n" + mcp_block

    system_prompt = compose_system_prompt(local_prompt, mode="direct_plus")

    response = await client.call(
        system_prompt=system_prompt,
        user_message=text,
        domain="synthesizer",
        concept="direct_plus_answer",
    )
    budget.record_llm_response(response)

    # Output gate — direct_plus prose is user-facing. Strip + lint +
    # one regenerate budget-paid retry if any blocking rule fires.
    if response.success and response.content:
        async def _regen_direct_plus(directive: str) -> str:
            resp = await client.call(
                system_prompt=system_prompt + "\n\n" + directive,
                user_message=text,
                domain="synthesizer",
                concept="direct_plus_answer",
            )
            budget.record_llm_response(resp)
            return resp.content if resp.success else ""

        gated = await gate_output_async(
            response.content,
            regenerate_fn=_regen_direct_plus,
            context=LintContext(),
        )
        response_text = gated.text
    else:
        response_text = f"[direct_plus call failed: {response.error}]"

    return DispatchResult(
        route=Route.DIRECT_PLUS,
        response_text=response_text,
        triage_result=triage_result,
        budget_summary=budget.summary(),
        capability_state=_summarize_capabilities(registry),
        missing_capability_offers=missing_offers,
        debug={"mcps_fired": mcp_fired},
    )


async def _dispatch_deep(
    text: str,
    client: LLMClient,
    triage_result: TriageResult,
    budget: BudgetTracker,
    registry: CapabilityRegistry,
    user_effort: Effort,
    policy: str,
    *,
    memory_directive: str = "",
    mcp_handlers: McpHandlerRegistry | None = None,
) -> DispatchResult:
    """Full wuxing engine at resolved effort tier.

    mcp_handlers is accepted for signature parity with the other route
    dispatchers; the deep route's MCP integration lives inside the engine
    itself (it has its own per-iteration MCP gating). When deep-route
    handler wiring lands as a follow-up, this parameter is the seam.
    """
    resolved_effort, escalation_offer = _resolve_effort(
        gate_effort=triage_result.recommended_effort,
        user_effort=user_effort,
        policy=policy,
    )

    # Policy=ask + gate wants more → return offer WITHOUT running the engine.
    # User accepts via Step 5 mechanism (resume endpoint) — not built yet.
    if escalation_offer is not None:
        return DispatchResult(
            route=Route.DEEP,
            response_text=(
                f"This looks deeper than {user_effort.value} effort. "
                f"Want me to extend to {triage_result.recommended_effort.value}?"
            ),
            triage_result=triage_result,
            budget_summary=budget.summary(),
            capability_state=_summarize_capabilities(registry),
            escalation_offer=escalation_offer,
        )

    check = budget.check()
    if not check.allowed:
        return _budget_breach_result(
            triage_result, budget, registry, Route.DEEP, check.reason
        )

    max_iterations = iterations_for(resolved_effort)

    # Track engine call count delta for debug reporting.
    pre_log_len = len(client.call_log)

    problem = _text_to_problem(text)
    # Engine now tracks its own per-iteration costs into the shared budget.
    engine_result = await run_async_formation(
        problem=problem,
        client=client,
        max_iterations=max_iterations,
        budget=budget,
    )

    # Snapshot before the speech call so we can attribute its cost separately
    # (engine doesn't see the speech-layer call).
    speech_pre_log_len = len(client.call_log)
    speech_input = extract_speech_input(
        engine_result=engine_result,
        user_original_text=text,
        is_phase_one=(max_iterations <= 2),
        estimated_additional_credits=10.0,
    )
    # Inject the angular checklist for this depth into the synthesizer prompt.
    deep_checklist = build_checklist_block(Route.DEEP, resolved_effort)
    # Phase 4 wiring: append Decision Trace prior memory after the checklist.
    # `memory_directive` is empty when no memory was found OR Neo4j was
    # unreachable — keeping the existing behavior for those cases.
    deep_directives = deep_checklist
    if memory_directive:
        deep_directives = deep_directives + "\n\n" + memory_directive
    speech_result = await generate_speech(
        client, speech_input, extra_directives=deep_directives,
    )

    # Roll up the speech call's cost (engine handled its own iteration costs).
    for log_entry in client.call_log[speech_pre_log_len:]:
        budget.record_llm_call(
            log_entry.model or "",
            log_entry.input_tokens,
            log_entry.output_tokens,
        )

    return DispatchResult(
        route=Route.DEEP,
        response_text=speech_result.response_text,
        triage_result=triage_result,
        budget_summary=budget.summary(),
        engine_result=engine_result,
        capability_state=_summarize_capabilities(registry),
        debug={
            "effort_resolved": resolved_effort.value,
            "max_iterations": max_iterations,
            "converged": engine_result.convergence_history.final_converged,
            "engine_calls": len(client.call_log) - pre_log_len,
        },
        memo=speech_result.memo,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EFFORT_ORDER: dict[Effort, int] = {
    Effort.LOW: 1,
    Effort.MEDIUM: 2,
    Effort.HIGH: 3,
    Effort.AUTO: 4,    # AUTO is the highest tier — user opted into full discretion
}


def _resolve_effort(
    gate_effort: Effort,
    user_effort: Effort,
    policy: str,
) -> tuple[Effort, dict | None]:
    """
    Decide the effort tier to actually use.

    Returns (resolved_effort, escalation_offer_or_None).
    escalation_offer is only set when policy=="ask" AND gate wants more
    than user authorized.
    """
    gate_rank = _EFFORT_ORDER.get(gate_effort, 2)
    user_rank = _EFFORT_ORDER.get(user_effort, 2)

    if gate_rank <= user_rank:
        return user_effort, None

    # Gate wants MORE than user authorized.
    if policy == "strict":
        return user_effort, None
    if policy == "auto":
        return gate_effort, None

    # policy == "ask" (default)
    return user_effort, {
        "user_effort": user_effort.value,
        "recommended_effort": gate_effort.value,
        "user_options": [
            {"id": "accept", "label": f"Extend to {gate_effort.value} effort"},
            {"id": "decline", "label": f"Stay at {user_effort.value} effort"},
        ],
    }


def _text_to_problem(text: str) -> Problem:
    """Minimal Problem from raw user text. Chemistry router does the real parsing."""
    return Problem(
        statement=text,
        variables=[
            Variable(
                name="user_statement_0",
                description=text[:200],
                magnitude=0.6,
                direction=Direction.NEUTRAL,
                confidence=0.8,
                source_framework=FrameworkID.FIRST_PRINCIPLES,
                is_user_stated=True,
            ),
        ],
    )


def _budget_breach_result(
    triage_result: TriageResult,
    budget: BudgetTracker,
    registry: CapabilityRegistry,
    route: Route,
    reason: str,
) -> DispatchResult:
    return DispatchResult(
        route=route,
        response_text=f"[budget exceeded before starting: {reason}]",
        triage_result=triage_result,
        budget_summary=budget.summary(),
        capability_state=_summarize_capabilities(registry),
    )


def _summarize_capabilities(reg: CapabilityRegistry) -> dict:
    return {
        "available": [c.name for c in reg.available()],
        "missing": [c.name for c in reg.missing()],
        "absent_by_design": [c.name for c in reg.absent_by_design()],
    }
