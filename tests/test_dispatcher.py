"""
Dispatcher tests.

All tests use the MOCK LLMClient so they run fast (~2s total) with no API
calls. The mock engine path exercises full run_async_formation against
mock JSON responses, so DEEP route gets real coverage.

Verifies:
    1. TRIVIAL route returns canned response, no LLM calls
    2. DIRECT route makes exactly one LLM call
    3. DIRECT_PLUS route surfaces missing-capability offer for memory_v2
    4. DEEP route runs engine and produces non-empty response_text
    5. Effort resolution:
         gate <= user                  → use user, no offer
         gate > user, policy=strict    → cap at user
         gate > user, policy=auto      → escalate to gate
         gate > user, policy=ask       → escalation_offer, engine NOT run
    6. Budget summary populated on every route
    7. Capability state populated on every route

Run: PYTHONPATH=. python3 tests/test_dispatcher.py
"""

from __future__ import annotations

import asyncio

from src.capabilities import CapabilityRegistry
from src.dispatcher import (
    DispatchResult,
    _classify_trivial,
    _resolve_effort,
    dispatch,
    resume_with_choice,
)
from src.llm.budget import BudgetCaps
from src.llm.client import ClientMode, LLMClient
from src.llm.effort import Effort
from src.llm.triage import MCPNeed, Route, TriageResult


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

PASSED = 0
FAILED = 0
ERRORS: list[tuple[str, str]] = []


def test(name: str):
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    global PASSED, FAILED
    name = getattr(fn, "_test_name", fn.__name__)
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


def mk_client() -> LLMClient:
    """Mock client — deterministic, no API."""
    return LLMClient(mode=ClientMode.MOCK)


# ---------------------------------------------------------------------------
# 1. TRIVIAL route
# ---------------------------------------------------------------------------

@test("1.1 'hi' → TRIVIAL, canned response, zero LLM calls")
async def test_trivial_hi():
    client = mk_client()
    r = await dispatch("hi", client=client)
    assert r.route == Route.TRIVIAL
    assert r.response_text
    assert len(client.call_log) == 0


@test("1.2 'thanks' → TRIVIAL with thanks canned response")
async def test_trivial_thanks():
    r = await dispatch("thanks!", client=mk_client())
    assert r.route == Route.TRIVIAL
    assert "got it" in r.response_text.lower()


@test("1.3 trivial classify keys")
def test_classify_trivial_keys():
    assert _classify_trivial("hi") == "greeting"
    assert _classify_trivial("thanks") == "thanks"
    assert _classify_trivial("ok") == "ack"
    assert _classify_trivial("yes") == "yesno"
    assert _classify_trivial("bye") == "farewell"
    assert _classify_trivial("are you there") == "presence"


# ---------------------------------------------------------------------------
# 2. DIRECT route
# ---------------------------------------------------------------------------

@test("2.1 factual question → DIRECT, exactly 1 LLM call")
async def test_direct_factual():
    client = mk_client()
    r = await dispatch("what's the difference between async and threading?",
                       client=client)
    assert r.route == Route.DIRECT
    assert len(client.call_log) == 1
    assert r.response_text  # mock returns non-empty content


@test("2.2 DIRECT response uses synthesizer domain")
async def test_direct_uses_synthesizer():
    client = mk_client()
    await dispatch("what is X", client=client)
    assert client.call_log[0].domain == "synthesizer"


# ---------------------------------------------------------------------------
# 3. DIRECT_PLUS route
# ---------------------------------------------------------------------------

@test("3.1 'what did we decide' → DIRECT_PLUS")
async def test_direct_plus_route():
    client = mk_client()
    r = await dispatch("what did we decide about auth last week?", client=client)
    assert r.route == Route.DIRECT_PLUS


@test("3.2 DIRECT_PLUS surfaces memory_v2 missing-capability offer")
async def test_direct_plus_missing_offer():
    # memory_v2 is AVAILABLE by default post-Phase-A. To exercise the
    # "missing → surface an offer" flow, simulate the pre-wired state by
    # explicitly marking it missing on a custom registry.
    reg = CapabilityRegistry()
    reg.mark_missing("memory_v2")
    r = await dispatch(
        "what did we decide about routing?",
        client=mk_client(),
        registry=reg,
    )
    assert r.route == Route.DIRECT_PLUS
    assert len(r.missing_capability_offers) >= 1
    offer = r.missing_capability_offers[0]
    assert offer["capability"] == "memory_v2"
    assert "why_needed" in offer
    assert offer["current_confidence"] < offer["confidence_if_connected"]


# ---------------------------------------------------------------------------
# 4. DEEP route
# ---------------------------------------------------------------------------

@test("4.1 'should I X or Y' → DEEP, engine runs, non-empty response")
async def test_deep_engine_runs():
    client = mk_client()
    # Force LOW effort to keep test fast (2 iterations only).
    r = await dispatch(
        "should I refactor this module or just extend it?",
        client=client,
        user_effort=Effort.LOW,
        policy="strict",  # don't escalate even if gate wants more
    )
    assert r.route == Route.DEEP
    assert r.engine_result is not None
    assert r.response_text  # speech module produces narrative
    assert r.debug["max_iterations"] == 2


@test("4.2 DEEP populates budget summary with engine cost + iterations")
async def test_deep_budget_populated():
    client = mk_client()
    r = await dispatch(
        "I'm torn between staying or leaving",
        client=client,
        user_effort=Effort.LOW,
        policy="strict",
    )
    # Engine now increments budget.iterations as it runs.
    assert r.budget_summary["iterations"] >= 1
    assert r.budget_summary["cost_usd"] >= 0  # mock has fake but consistent token counts


# ---------------------------------------------------------------------------
# 5. Effort resolution
# ---------------------------------------------------------------------------

@test("5.1 gate=low, user=high → resolve to high, no offer")
def test_resolve_gate_below_user():
    resolved, offer = _resolve_effort(Effort.LOW, Effort.HIGH, policy="ask")
    assert resolved == Effort.HIGH
    assert offer is None


@test("5.2 gate=high, user=medium, policy=strict → cap at medium")
def test_resolve_strict():
    resolved, offer = _resolve_effort(Effort.HIGH, Effort.MEDIUM, policy="strict")
    assert resolved == Effort.MEDIUM
    assert offer is None


@test("5.3 gate=high, user=medium, policy=auto → escalate to high")
def test_resolve_auto():
    resolved, offer = _resolve_effort(Effort.HIGH, Effort.MEDIUM, policy="auto")
    assert resolved == Effort.HIGH
    assert offer is None


@test("5.4 gate=high, user=medium, policy=ask → escalation offer, no escalation")
def test_resolve_ask():
    resolved, offer = _resolve_effort(Effort.HIGH, Effort.MEDIUM, policy="ask")
    assert resolved == Effort.MEDIUM  # capped pending user response
    assert offer is not None
    assert offer["user_effort"] == "medium"
    assert offer["recommended_effort"] == "high"
    assert len(offer["user_options"]) == 2


@test("5.5 gate=medium, user=medium → no offer regardless of policy")
def test_resolve_equal():
    for policy in ("strict", "ask", "auto"):
        resolved, offer = _resolve_effort(Effort.MEDIUM, Effort.MEDIUM, policy=policy)
        assert resolved == Effort.MEDIUM
        assert offer is None


# ---------------------------------------------------------------------------
# 6. Escalation offer end-to-end
# ---------------------------------------------------------------------------

@test("6.1 DEEP with policy=ask + gate>user → escalation_offer returned, engine NOT run")
async def test_escalation_offer_returns_without_running_engine():
    client = mk_client()
    # Force triage to recommend HIGH while user_effort=MEDIUM and policy=ask
    forced_triage = TriageResult(
        route=Route.DEEP,
        recommended_effort=Effort.HIGH,
        why="forced for test",
        classifier_mode="forced",
    )
    r = await dispatch(
        "test question",
        client=client,
        user_effort=Effort.MEDIUM,
        policy="ask",
        force_triage_result=forced_triage,
    )
    assert r.route == Route.DEEP
    assert r.escalation_offer is not None
    assert r.engine_result is None  # engine NOT run
    # No engine calls should have been made
    assert len(client.call_log) == 0


@test("6.2 DEEP with policy=strict + gate>user → engine runs at user effort")
async def test_strict_caps_at_user_effort():
    client = mk_client()
    forced_triage = TriageResult(
        route=Route.DEEP,
        recommended_effort=Effort.HIGH,
        why="forced",
        classifier_mode="forced",
    )
    r = await dispatch(
        "test", client=client,
        user_effort=Effort.LOW, policy="strict",
        force_triage_result=forced_triage,
    )
    assert r.escalation_offer is None
    assert r.engine_result is not None
    assert r.debug["max_iterations"] == 2  # LOW = 2 iterations


# ---------------------------------------------------------------------------
# 7. Common invariants across all routes
# ---------------------------------------------------------------------------

@test("7.1 every route populates budget_summary and capability_state")
async def test_all_routes_populate_summary():
    client = mk_client()
    for msg, expected_route in [
        ("hi", Route.TRIVIAL),
        ("what is async", Route.DIRECT),
        ("what did we decide about auth", Route.DIRECT_PLUS),
    ]:
        r = await dispatch(msg, client=client, policy="strict")
        assert r.route == expected_route
        assert isinstance(r.budget_summary, dict)
        assert "caps" in r.budget_summary
        assert "available" in r.capability_state
        assert "missing" in r.capability_state
        assert "absent_by_design" in r.capability_state


@test("7.2 every result includes triage_result")
async def test_triage_passed_through():
    r = await dispatch("hi", client=mk_client())
    assert r.triage_result is not None
    assert r.triage_result.route == Route.TRIVIAL


# ---------------------------------------------------------------------------
# 8. Custom registry / caps
# ---------------------------------------------------------------------------

@test("8.1 custom CapabilityRegistry is honored")
async def test_custom_registry():
    reg = CapabilityRegistry()
    reg.mark_available("memory_v2")  # pretend memory is wired up
    r = await dispatch(
        "what did we decide about auth?",
        client=mk_client(),
        registry=reg,
    )
    # Now memory_v2 should NOT appear in missing_capability_offers
    names = [o["capability"] for o in r.missing_capability_offers]
    assert "memory_v2" not in names


@test("8.2 custom BudgetCaps is honored")
async def test_custom_caps():
    r = await dispatch(
        "hi", client=mk_client(),
        caps=BudgetCaps(max_iterations=99, max_cost_usd=0.5),
    )
    assert r.budget_summary["caps"]["max_iterations"] == 99
    assert r.budget_summary["caps"]["max_cost_usd"] == 0.5


# ---------------------------------------------------------------------------
# 9. resume_with_choice — user response to an escalation_offer
# ---------------------------------------------------------------------------

@test("9.1 resume with HIGH after accepting → engine runs at high (5 iter)")
async def test_resume_accept_high():
    client = mk_client()
    r = await resume_with_choice(
        text="should I quit my job to start a company?",
        client=client,
        accepted_effort=Effort.HIGH,
    )
    assert r.route == Route.DEEP
    assert r.engine_result is not None
    assert r.escalation_offer is None  # policy=strict means no offer
    assert r.debug["max_iterations"] == 5


@test("9.2 resume with MEDIUM after declining → engine runs at medium (3 iter)")
async def test_resume_decline_stay_medium():
    client = mk_client()
    r = await resume_with_choice(
        text="should I refactor the auth layer?",
        client=client,
        accepted_effort=Effort.MEDIUM,
    )
    assert r.route == Route.DEEP
    assert r.engine_result is not None
    assert r.escalation_offer is None
    assert r.debug["max_iterations"] == 3


@test("9.3 resume NEVER produces a second escalation_offer (policy=strict)")
async def test_resume_no_second_offer():
    client = mk_client()
    # Even if triage internally recommends HIGH, resume at LOW must cap.
    r = await resume_with_choice(
        text="should I do X or Y when restructuring the whole architecture?",
        client=client,
        accepted_effort=Effort.LOW,
    )
    assert r.escalation_offer is None
    assert r.debug["max_iterations"] == 2  # LOW = 2 iter


@test("9.4 resume accepts string effort ('high') equivalent to enum")
async def test_resume_string_effort():
    client = mk_client()
    r = await resume_with_choice(
        text="should I rewrite or extend?", client=client, accepted_effort="high",
    )
    assert r.debug["max_iterations"] == 5


@test("9.5 resume populates budget + capability state like dispatch()")
async def test_resume_populates_state():
    r = await resume_with_choice(
        text="should I X or Y?", client=mk_client(),
        accepted_effort=Effort.LOW,
    )
    assert "caps" in r.budget_summary
    assert "available" in r.capability_state
    assert "missing" in r.capability_state


@test("9.6 resume on a TRIVIAL message degrades cleanly (TRIVIAL route preserved)")
async def test_resume_trivial_message():
    # If the user somehow resumes on a trivial message (frontend bug, but defensive),
    # triage still routes to TRIVIAL — accepted_effort is ignored.
    r = await resume_with_choice(
        text="hi", client=mk_client(),
        accepted_effort=Effort.HIGH,
    )
    assert r.route == Route.TRIVIAL
    assert r.engine_result is None  # TRIVIAL doesn't run the engine


# ---------------------------------------------------------------------------
# 10. AUTO mode — user opts into full discretion up to engine cap
# ---------------------------------------------------------------------------

@test("10.1 Effort.AUTO maps to 8 iterations (benchmark-validated ceiling)")
def test_auto_iterations_cap():
    from src.llm.effort import iterations_for
    assert iterations_for(Effort.AUTO) == 8


@test("10.2 _resolve_effort: user=AUTO, gate=HIGH → resolve to AUTO, no offer")
def test_resolve_user_auto_no_offer():
    resolved, offer = _resolve_effort(Effort.HIGH, Effort.AUTO, policy="ask")
    assert resolved == Effort.AUTO
    assert offer is None  # AUTO can never trigger an escalation offer


@test("10.3 _resolve_effort: user=AUTO + every gate level → no offer ever")
def test_resolve_user_auto_all_gates():
    for gate in (Effort.LOW, Effort.MEDIUM, Effort.HIGH):
        for policy in ("strict", "ask", "auto"):
            resolved, offer = _resolve_effort(gate, Effort.AUTO, policy=policy)
            assert resolved == Effort.AUTO, f"gate={gate}, policy={policy}"
            assert offer is None, f"gate={gate}, policy={policy}"


@test("10.4 dispatch with effort=AUTO + DEEP question uses 8 iter cap")
async def test_dispatch_auto_deep():
    client = mk_client()
    r = await dispatch(
        "should I X or Y in this architecture refactor?",
        client=client,
        user_effort=Effort.AUTO,
        policy="ask",
    )
    assert r.route == Route.DEEP
    assert r.escalation_offer is None  # AUTO never produces offers
    assert r.debug["max_iterations"] == 8


@test("10.5 dispatch with effort=AUTO + tight cost cap stops engine early")
async def test_dispatch_auto_with_tight_budget():
    client = mk_client()
    # AUTO would normally run to 8 iter, but cost cap is impossibly tight —
    # the first mock LLM call costs more than $0.0001, so engine's first
    # pre-iteration check breaches.
    r = await dispatch(
        "should I refactor or extend this module?",
        client=client,
        user_effort=Effort.AUTO,
        policy="auto",
        caps=BudgetCaps(max_cost_usd=0.0001),
    )
    assert r.route == Route.DEEP
    # Engine must be capped well under AUTO's normal 8 iterations.
    assert r.budget_summary["iterations"] < 8
    assert r.budget_summary["breached"] is True


@test("10.6 resume_with_choice('auto') runs at 8-iter cap")
async def test_resume_with_auto():
    r = await resume_with_choice(
        text="should I do this thing?",
        client=mk_client(),
        accepted_effort=Effort.AUTO,
    )
    assert r.route == Route.DEEP
    assert r.escalation_offer is None
    assert r.debug["max_iterations"] == 8


# ---------------------------------------------------------------------------
# 11. Checklist injection — questions actually reach the LLM system prompt
# ---------------------------------------------------------------------------

def _capture_prompts(client):
    """Monkey-patch client.call to capture every system_prompt passed in."""
    captured: list[str] = []
    original = client.call

    async def wrapped(system_prompt, user_message, **kwargs):
        captured.append(system_prompt)
        return await original(system_prompt, user_message, **kwargs)

    client.call = wrapped
    return captured


@test("11.1 DIRECT injects F25 into the system prompt")
async def test_direct_includes_f25():
    client = mk_client()
    prompts = _capture_prompts(client)
    await dispatch("what is async?", client=client, policy="strict")
    # At least one captured prompt contains the F25 code
    assert any("F25" in p for p in prompts), "no prompt contained F25"


@test("11.2 DIRECT_PLUS injects A1 + F25-F27 into the system prompt")
async def test_direct_plus_includes_checklist():
    client = mk_client()
    prompts = _capture_prompts(client)
    await dispatch("what did we decide about auth?", client=client, policy="strict")
    full = "\n".join(prompts)
    for code in ("A1", "F25", "F26", "F27"):
        assert code in full, f"missing {code} in DIRECT_PLUS prompts"


@test("11.3 DEEP synthesizer receives the depth-appropriate checklist")
async def test_deep_synth_receives_checklist():
    client = mk_client()
    prompts = _capture_prompts(client)
    await dispatch(
        "should I refactor or extend this?",
        client=client,
        user_effort=Effort.LOW,
        policy="strict",
    )
    # LOW deep checklist = A1-A4 + D14 + E20
    full = "\n".join(prompts)
    # The synthesizer call is the LAST one (after engine + speech extraction)
    # It should contain these codes
    for code in ("A1", "A2", "A3", "A4", "D14", "E20"):
        assert code in full, f"missing {code} in DEEP/LOW prompts"


@test("11.4 TRIVIAL makes no LLM calls (so no prompt to verify)")
async def test_trivial_no_prompts():
    client = mk_client()
    prompts = _capture_prompts(client)
    r = await dispatch("hi", client=client)
    assert r.route == Route.TRIVIAL
    assert prompts == []


# ---------------------------------------------------------------------------
# 12. MCP router integration into DIRECT_PLUS
# ---------------------------------------------------------------------------

@test("12.1 DIRECT_PLUS fires MCPs through the router; missing → offer surfaced")
async def test_direct_plus_uses_router():
    # memory_v2 is AVAILABLE by default post-Phase-A. Mark it missing on a
    # custom registry so the router's missing-path + offer-surfacing flow
    # is the thing under test (mk_client's mock triage hardcodes memory_v2
    # for memory queries).
    reg = CapabilityRegistry()
    reg.mark_missing("memory_v2")
    r = await dispatch(
        "what did we decide about routing last week?",
        client=mk_client(),
        registry=reg,
    )
    assert r.route == Route.DIRECT_PLUS
    # mcp_fired present in debug, each entry tells us what happened
    fired = r.debug.get("mcps_fired", [])
    assert len(fired) >= 1
    memory_attempts = [f for f in fired if f["name"] == "memory_v2"]
    assert len(memory_attempts) == 1
    assert memory_attempts[0]["ok"] is False  # forced MISSING by this test
    # And the missing-capability offer should be surfaced
    offer_names = [o["capability"] for o in r.missing_capability_offers]
    assert "memory_v2" in offer_names


@test("12.2 DIRECT_PLUS with memory_v2 marked available → router fires successfully")
async def test_direct_plus_router_when_available():
    reg = CapabilityRegistry()
    reg.mark_available("memory_v2")
    r = await dispatch(
        "what did we decide about auth?",
        client=mk_client(),
        registry=reg,
    )
    fired = r.debug.get("mcps_fired", [])
    memory_attempts = [f for f in fired if f["name"] == "memory_v2"]
    assert len(memory_attempts) == 1
    assert memory_attempts[0]["ok"] is True
    assert memory_attempts[0]["stub"] is True
    # No missing-capability offer when fire succeeded
    offer_names = [o["capability"] for o in r.missing_capability_offers]
    assert "memory_v2" not in offer_names


@test("12.3 successful MCP fire increments budget.mcp_calls")
async def test_router_fire_counts_to_budget():
    reg = CapabilityRegistry()
    reg.mark_available("memory_v2")
    r = await dispatch(
        "what did we decide?",
        client=mk_client(),
        registry=reg,
    )
    assert r.budget_summary["mcp_calls"] >= 1


@test("12.4 mcp_handlers: registered handler runs and result lands in dispatcher")
async def test_dispatcher_uses_mcp_handlers():
    from src.mcp_router import McpHandlerRegistry
    handlers = McpHandlerRegistry()

    captured = {}
    async def memory_handler(args, purpose):
        captured["purpose"] = purpose
        return {"text": "User decided to use OAuth2 last week."}

    handlers.register("memory_v2", memory_handler)

    reg = CapabilityRegistry()  # memory_v2 already AVAILABLE post-Phase-A
    r = await dispatch(
        "what did we decide about auth?",
        client=mk_client(),
        registry=reg,
        mcp_handlers=handlers,
    )

    # Handler fired with the triage-provided purpose
    assert captured.get("purpose")
    # Dispatcher records the real (non-stub) fire in debug
    fired = r.debug.get("mcps_fired", [])
    memory_attempts = [f for f in fired if f["name"] == "memory_v2"]
    assert len(memory_attempts) == 1
    assert memory_attempts[0]["ok"] is True
    assert memory_attempts[0]["stub"] is False  # real handler, not stub
    # No missing-capability offer when the handler ran cleanly
    offer_names = [o["capability"] for o in r.missing_capability_offers]
    assert "memory_v2" not in offer_names


@test("12.5 mcp_handlers None → back-compat stub behavior preserved")
async def test_dispatcher_without_handlers_is_back_compat():
    # No mcp_handlers provided → fire_mcp returns stub for memory_v2.
    # Dispatcher still works exactly as it did pre-Phase-B1.
    r = await dispatch(
        "what did we decide about auth?",
        client=mk_client(),
    )
    fired = r.debug.get("mcps_fired", [])
    memory_attempts = [f for f in fired if f["name"] == "memory_v2"]
    assert len(memory_attempts) == 1
    assert memory_attempts[0]["ok"] is True
    assert memory_attempts[0]["stub"] is True


@test("12.6 handler exception → missing offer surfaces (no crash)")
async def test_dispatcher_handler_exception_graceful():
    from src.mcp_router import McpHandlerRegistry
    handlers = McpHandlerRegistry()

    async def broken_handler(args, purpose):
        raise RuntimeError("retriever offline")

    handlers.register("memory_v2", broken_handler)

    reg = CapabilityRegistry()
    r = await dispatch(
        "what did we decide?",
        client=mk_client(),
        registry=reg,
        mcp_handlers=handlers,
    )
    fired = r.debug.get("mcps_fired", [])
    memory_attempts = [f for f in fired if f["name"] == "memory_v2"]
    assert len(memory_attempts) == 1
    assert memory_attempts[0]["ok"] is False
    # The dispatcher surfaces a missing-capability offer when fire failed
    offer_names = [o["capability"] for o in r.missing_capability_offers]
    assert "memory_v2" in offer_names


# ---------------------------------------------------------------------------
# 13. ConversationStore integration — opt-in recording on dispatch
# ---------------------------------------------------------------------------

async def _mk_session_and_store(project_id="proj-1"):
    from src.bridge.conversation_store import ConversationStore
    cs = ConversationStore(project_id=project_id)
    sess = await cs.start_session(title="test session")
    return cs, sess


@test("13.1 dispatch without conversation_store: no recording, no error")
async def test_no_conv_store_silent():
    r = await dispatch("hi", client=mk_client())
    # No iteration_id should appear in debug
    assert "iteration_id" not in (r.debug or {})


@test("13.2 dispatch with store + session_id records a TRIVIAL iteration")
async def test_record_trivial():
    cs, sess = await _mk_session_and_store()
    r = await dispatch(
        "hi", client=mk_client(),
        conversation_store=cs, session_id=sess.id,
    )
    assert r.route == Route.TRIVIAL
    assert "iteration_id" in r.debug
    # Verify the iteration is in the store
    iters = await cs.iterations_for_session(sess.id)
    assert len(iters) == 1
    assert iters[0].user_text == "hi"
    assert iters[0].route == "trivial"


@test("13.3 dispatch records a DIRECT iteration with route + effort")
async def test_record_direct():
    cs, sess = await _mk_session_and_store()
    r = await dispatch(
        "what is async?", client=mk_client(),
        conversation_store=cs, session_id=sess.id,
        policy="strict",
    )
    assert r.route == Route.DIRECT
    iters = await cs.iterations_for_session(sess.id)
    assert len(iters) == 1
    assert iters[0].route == "direct"


@test("13.4 dispatch records a DEEP iteration with effort tier")
async def test_record_deep():
    cs, sess = await _mk_session_and_store()
    r = await dispatch(
        "should I refactor or extend?", client=mk_client(),
        user_effort=Effort.LOW, policy="strict",
        conversation_store=cs, session_id=sess.id,
    )
    assert r.route == Route.DEEP
    iters = await cs.iterations_for_session(sess.id)
    assert len(iters) == 1
    assert iters[0].route == "deep"
    # The triage recommends effort; recorded iteration captures it
    assert iters[0].effort in ("low", "medium", "high")


@test("13.5 sequence_num auto-increments across multiple dispatches in same session")
async def test_record_multiple_seq():
    cs, sess = await _mk_session_and_store()
    client = mk_client()
    for msg in ["hi", "ok", "thanks"]:
        await dispatch(
            msg, client=client,
            conversation_store=cs, session_id=sess.id,
        )
    iters = await cs.iterations_for_session(sess.id)
    assert [it.sequence_num for it in iters] == [1, 2, 3]


@test("13.6 unknown session_id captured in debug, does NOT break the response")
async def test_record_bad_session_does_not_fail():
    from src.bridge.conversation_store import ConversationStore
    cs = ConversationStore(project_id="proj-1")
    # No session started — session_id "S-nope" is invalid
    r = await dispatch(
        "hi", client=mk_client(),
        conversation_store=cs, session_id="S-nope",
    )
    # Response still returns
    assert r.response_text
    # Error captured in debug
    assert "iteration_record_error" in r.debug
    # But iteration_id is NOT set (recording failed)
    assert "iteration_id" not in r.debug


@test("13.7 resume_with_choice also records when conv_store+session_id passed")
async def test_resume_records():
    cs, sess = await _mk_session_and_store()
    r = await resume_with_choice(
        "should I refactor?", client=mk_client(),
        accepted_effort=Effort.HIGH,
        conversation_store=cs, session_id=sess.id,
    )
    iters = await cs.iterations_for_session(sess.id)
    assert len(iters) == 1
    assert iters[0].route == "deep"


@test("13.8 escalation_offer dispatch (engine NOT run) still records the iteration")
async def test_record_escalation_offer():
    cs, sess = await _mk_session_and_store()
    forced = TriageResult(
        route=Route.DEEP,
        recommended_effort=Effort.HIGH,
        why="forced for test",
        classifier_mode="forced",
    )
    r = await dispatch(
        "test", client=mk_client(),
        user_effort=Effort.MEDIUM, policy="ask",
        conversation_store=cs, session_id=sess.id,
        force_triage_result=forced,
    )
    # Even though the engine didn't run (escalation offer surfaced),
    # the iteration IS recorded — user did interact, we capture it.
    assert r.escalation_offer is not None
    iters = await cs.iterations_for_session(sess.id)
    assert len(iters) == 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_trivial_hi,
    test_trivial_thanks,
    test_classify_trivial_keys,
    test_direct_factual,
    test_direct_uses_synthesizer,
    test_direct_plus_route,
    test_direct_plus_missing_offer,
    test_deep_engine_runs,
    test_deep_budget_populated,
    test_resolve_gate_below_user,
    test_resolve_strict,
    test_resolve_auto,
    test_resolve_ask,
    test_resolve_equal,
    test_escalation_offer_returns_without_running_engine,
    test_strict_caps_at_user_effort,
    test_all_routes_populate_summary,
    test_triage_passed_through,
    test_custom_registry,
    test_custom_caps,
    test_resume_accept_high,
    test_resume_decline_stay_medium,
    test_resume_no_second_offer,
    test_resume_string_effort,
    test_resume_populates_state,
    test_resume_trivial_message,
    test_auto_iterations_cap,
    test_resolve_user_auto_no_offer,
    test_resolve_user_auto_all_gates,
    test_dispatch_auto_deep,
    test_dispatch_auto_with_tight_budget,
    test_resume_with_auto,
    test_direct_includes_f25,
    test_direct_plus_includes_checklist,
    test_deep_synth_receives_checklist,
    test_trivial_no_prompts,
    test_direct_plus_uses_router,
    test_direct_plus_router_when_available,
    test_router_fire_counts_to_budget,
    test_dispatcher_uses_mcp_handlers,
    test_dispatcher_without_handlers_is_back_compat,
    test_dispatcher_handler_exception_graceful,
    test_no_conv_store_silent,
    test_record_trivial,
    test_record_direct,
    test_record_deep,
    test_record_multiple_seq,
    test_record_bad_session_does_not_fail,
    test_resume_records,
    test_record_escalation_offer,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} dispatcher tests...")
    print()
    for fn in ALL_TESTS:
        run_test(fn)
    print()
    print(f"{PASSED} passed, {FAILED} failed")
    if ERRORS:
        print()
        print("Failures:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
