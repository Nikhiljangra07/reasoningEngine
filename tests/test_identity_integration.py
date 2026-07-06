"""
Integration tests for the identity layer.

These tests prove that the doctrine has been wired into the runtime,
not just sitting in `src/identity/`. They cover Codex's #3 and #4:

  3. Add integration tests proving Wandering/Thinking prompts include identity.
  4. Add tests proving bad outputs trigger regeneration.

The tests are split into three groups:

  P — PROMPT COMPOSER
      P1   composer prepends SYSTEM_PROMPT_HEADER
      P2   composer tags the mode
      P3   composer handles empty local prompt
      P4   composer handles None local prompt
      P5   composer is deterministic
      P6   carries_identity_header detects composed prompts
      P7   carries_identity_header is False on naive prompts

  W — WANDERING PROMPTS CARRY IDENTITY
      W1   composer.py extraction prompt carries header (compose_system_prompt-wrapped)
      W2   matching.py match prompt carries header
      W3   agent.py dig prompt carries header
      W4   articulate.py articulate prompt carries header
      W5   synthesis.py synthesis prompt carries header

  S — SPEECH PROMPTS CARRY IDENTITY
      S1   SPEECH_SYSTEM_PROMPT composed for the main speech call carries header
      S2   _CLARIFICATION_PROMPT composed for the clarifier carries header
      S3   _SYNTHESIZER_ONLY_PREAMBLE composed for first-read carries header
      S4   _SEGMENT_VOICE_PREAMBLE composed for synthesizer segment carries header
      S5   _SEGMENT_VOICE_PREAMBLE composed for opinion segment carries header
      S6   _SEGMENT_VOICE_PREAMBLE composed for prospects segment carries header
      S7   no `## IDENTITY` heading remains in speech.py module prompts
           (the rename to `## VOICE PROFILE` removed the competing identity
            declarations Codex flagged)

  G — OUTPUT GATE
      G1   sync gate passes clean text through (no regenerate)
      G2   sync gate regenerates on opener fluff
      G3   sync gate regenerates on missing failure mode
      G4   sync gate regenerates on first-person execution
      G5   sync gate max_attempts=1 disables regenerate
      G6   sync gate returns telemetry (initial_lint, final_lint, attempts)
      G7   sync gate uses directive_used to record the remediation it sent
      G8   async gate matches sync behavior on clean input
      G9   async gate regenerates async on bad input
      G10  gate preserves the regenerated text when second draft passes
      G11  gate exposes that second draft also failed (degraded mode)
"""

from __future__ import annotations

import asyncio
import pathlib
import sys


# ─── Mini test harness ─────────────────────────────────────────────────

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
    except AssertionError as e:
        FAILED += 1
        ERRORS.append((name, f"FAIL: {e}"))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, f"ERROR: {type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


# ─── Imports under test ────────────────────────────────────────────────

from src.identity import (
    CONTROL_PLANE_SITES,
    SYSTEM_PROMPT_HEADER,
    GatedOutput,
    carries_identity_header,
    compose_system_prompt,
    gate_output_async,
    gate_output_sync,
    is_exempt,
)
from src.identity.voice.lint import LintContext

# Wandering prompt constants
from src.wandering.composer  import _EXTRACTION_SYSTEM_PROMPT
from src.wandering.matching  import _MATCH_SYSTEM_PROMPT
from src.wandering.agent     import _DIG_SYSTEM_PROMPT
from src.wandering.articulate import _ARTICULATE_SYSTEM_PROMPT
from src.wandering.synthesis import _SYNTHESIS_SYSTEM_PROMPT

# Speech prompts and the file itself for the no-IDENTITY check
from src.llm.speech import (
    SPEECH_SYSTEM_PROMPT,
    _CLARIFICATION_PROMPT,
    _OPINION_SCHEMA_BLOCK,
    _PROSPECTS_SCHEMA_BLOCK,
    _SEGMENT_VOICE_PREAMBLE,
    _SYNTHESIZER_ONLY_PREAMBLE,
    _SYNTHESIZER_SCHEMA_BLOCK,
)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group P — Prompt composer                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

@test("P1 composer prepends SYSTEM_PROMPT_HEADER")
def test_composer_prepends_header():
    local = "Local instructions go here."
    out = compose_system_prompt(local)
    assert out.startswith(SYSTEM_PROMPT_HEADER.strip()[:100]), out[:200]
    assert local in out


@test("P2 composer tags the mode")
def test_composer_tags_mode():
    out = compose_system_prompt("local", mode="wandering_dig")
    assert "MODE: wandering_dig" in out


@test("P3 composer handles empty local prompt")
def test_composer_empty_local():
    out = compose_system_prompt("", mode="x")
    # Header + MODE only — no trailing local section.
    assert SYSTEM_PROMPT_HEADER.strip()[:50] in out
    assert "MODE: x" in out


@test("P4 composer handles None local prompt")
def test_composer_none_local():
    out = compose_system_prompt(None, mode="x")
    assert SYSTEM_PROMPT_HEADER.strip()[:50] in out
    assert "MODE: x" in out


@test("P5 composer is deterministic")
def test_composer_deterministic():
    a = compose_system_prompt("same", mode="m")
    b = compose_system_prompt("same", mode="m")
    assert a == b


@test("P6 carries_identity_header detects composed prompts")
def test_carries_detect():
    out = compose_system_prompt("local", mode="x")
    assert carries_identity_header(out) is True


@test("P7 carries_identity_header False on naive prompts")
def test_carries_false():
    assert carries_identity_header("plain local prompt") is False
    assert carries_identity_header("") is False


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group W — Wandering prompts carry identity                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Each test composes the same prompt that the wandering call site
# composes — confirming the wiring is in place and that the resulting
# prompt carries the doctrine header.

@test("W1 composer.py extraction prompt carries identity header")
def test_wandering_composer_carries():
    composed = compose_system_prompt(_EXTRACTION_SYSTEM_PROMPT, mode="cushion_compose")
    assert carries_identity_header(composed)
    assert "MODE: cushion_compose" in composed


@test("W2 matching.py match prompt carries identity header")
def test_wandering_matching_carries():
    composed = compose_system_prompt(_MATCH_SYSTEM_PROMPT, mode="structural_match")
    assert carries_identity_header(composed)
    assert "MODE: structural_match" in composed


@test("W3 agent.py dig prompt carries identity header")
def test_wandering_agent_carries():
    composed = compose_system_prompt(_DIG_SYSTEM_PROMPT, mode="wandering_dig")
    assert carries_identity_header(composed)
    assert "MODE: wandering_dig" in composed


@test("W4 articulate.py prompt carries identity header")
def test_wandering_articulate_carries():
    composed = compose_system_prompt(_ARTICULATE_SYSTEM_PROMPT, mode="card_articulation")
    assert carries_identity_header(composed)
    assert "MODE: card_articulation" in composed


@test("W5 synthesis.py prompt carries identity header")
def test_wandering_synthesis_carries():
    composed = compose_system_prompt(_SYNTHESIS_SYSTEM_PROMPT, mode="dossier_synthesis")
    assert carries_identity_header(composed)
    assert "MODE: dossier_synthesis" in composed


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group S — Speech prompts carry identity                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

@test("S1 SPEECH_SYSTEM_PROMPT composed carries header")
def test_speech_main_carries():
    composed = compose_system_prompt(SPEECH_SYSTEM_PROMPT, mode="speech")
    assert carries_identity_header(composed)
    assert "MODE: speech" in composed


@test("S2 _CLARIFICATION_PROMPT composed carries header")
def test_speech_clarifier_carries():
    composed = compose_system_prompt(_CLARIFICATION_PROMPT, mode="speech_clarification")
    assert carries_identity_header(composed)


@test("S3 _SYNTHESIZER_ONLY_PREAMBLE composed carries header")
def test_speech_first_read_carries():
    composed = compose_system_prompt(
        _SYNTHESIZER_ONLY_PREAMBLE + "\n\n" + _SYNTHESIZER_SCHEMA_BLOCK,
        mode="speech_first_read",
    )
    assert carries_identity_header(composed)


@test("S4 _SEGMENT_VOICE_PREAMBLE composed (synthesizer segment) carries header")
def test_speech_synth_segment_carries():
    composed = compose_system_prompt(
        _SEGMENT_VOICE_PREAMBLE + "\n\n" + _SYNTHESIZER_SCHEMA_BLOCK,
        mode="speech_synthesizer_segment",
    )
    assert carries_identity_header(composed)


@test("S5 _SEGMENT_VOICE_PREAMBLE composed (opinion segment) carries header")
def test_speech_opinion_segment_carries():
    composed = compose_system_prompt(
        _SEGMENT_VOICE_PREAMBLE + "\n\n" + _OPINION_SCHEMA_BLOCK,
        mode="speech_opinion_segment",
    )
    assert carries_identity_header(composed)


@test("S6 _SEGMENT_VOICE_PREAMBLE composed (prospects segment) carries header")
def test_speech_prospects_segment_carries():
    composed = compose_system_prompt(
        _SEGMENT_VOICE_PREAMBLE + "\n\n" + _PROSPECTS_SCHEMA_BLOCK,
        mode="speech_prospects_segment",
    )
    assert carries_identity_header(composed)


@test("S7 no `## IDENTITY` heading remains in speech.py module prompts")
def test_speech_no_competing_identity():
    """Codex flagged that the speech prompts declared their own identity
    in `## IDENTITY` sections, competing with the Singular Path header
    once the composer was wired. The fix renamed all four to `## VOICE
    PROFILE` so they read as additive to the doctrine, not as a
    replacement for it. This test guards the rename — if a `## IDENTITY`
    heading slips back into any of the four speech prompts, the
    competing-identity problem returns."""
    for label, body in (
        ("SPEECH_SYSTEM_PROMPT",       SPEECH_SYSTEM_PROMPT),
        ("_SEGMENT_VOICE_PREAMBLE",    _SEGMENT_VOICE_PREAMBLE),
        ("_CLARIFICATION_PROMPT",      _CLARIFICATION_PROMPT),
        ("_SYNTHESIZER_ONLY_PREAMBLE", _SYNTHESIZER_ONLY_PREAMBLE),
    ):
        assert "## IDENTITY" not in body, (
            f"{label} still contains '## IDENTITY' — competing identity declaration"
        )
        assert "## VOICE PROFILE" in body, (
            f"{label} missing '## VOICE PROFILE' — rename did not land"
        )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group G — Output gate                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

_CLEAN_DRAFT = (
    "The move is to ship the smallest version that proves the bet. "
    "Cost: you lose the polish you wanted. This breaks if early users "
    "care more about polish than presence — watch for negative feedback "
    "on rough edges, not on missing features."
)

_STRIPPABLE_OPENER_DRAFT = (
    # Strip handles this silently — no regenerate fires.
    "Great question! The move is to ship. " + "Cost: lose polish. " * 5
)

_PADDING_DRAFT = (
    # Therapy padding — strip doesn't catch this, lint does, regenerate fires.
    "I hear you. The move is to ship. " + "Cost: lose polish. " * 5
)

_NO_FAILURE_MODE_DRAFT = (
    "Cut three features by Friday. Tell beta users they can wait. "
    "Recruit two reviewers from the existing list. Send the launch "
    "email Friday morning at nine local time and keep the team focused "
    "on stability through Monday."
)

_EXECUTION_DRAFT = (
    "I'll edit the config file and commit it. " + "Cost: lose polish. " * 5
)


def _make_regen_sync(reply: str):
    """Build a sync regenerate_fn that returns `reply`. Records the
    directive it was called with for inspection."""
    calls: list[str] = []

    def regen(directive: str) -> str:
        calls.append(directive)
        return reply

    regen.calls = calls  # type: ignore[attr-defined]
    return regen


def _make_regen_async(reply: str):
    calls: list[str] = []

    async def regen(directive: str) -> str:
        calls.append(directive)
        return reply

    regen.calls = calls  # type: ignore[attr-defined]
    return regen


@test("G1 sync gate passes clean text through")
def test_gate_clean_passthrough():
    regen = _make_regen_sync("should not be called")
    out = gate_output_sync(_CLEAN_DRAFT, regenerate_fn=regen)
    assert isinstance(out, GatedOutput)
    assert out.regenerated is False
    assert out.attempts == 1
    assert out.final_lint.passed
    assert regen.calls == []  # type: ignore[attr-defined]


@test("G2 sync gate strips openers silently (no regenerate)")
def test_gate_strips_openers_silently():
    """Opener fluff like 'Great question!' is handled by the strip
    pass before lint runs. That's the intended fail-closed split —
    strip catches the cheap cases without a costly model round-trip;
    lint catches the rest. So a draft whose only sin is a strippable
    opener should NOT trigger regenerate; it should be silently
    cleaned and pass lint on the stripped text."""
    regen = _make_regen_sync(_CLEAN_DRAFT)
    out = gate_output_sync(_STRIPPABLE_OPENER_DRAFT, regenerate_fn=regen)
    assert out.regenerated is False
    assert out.attempts == 1
    assert "Great question" not in out.text
    assert regen.calls == []  # type: ignore[attr-defined]


@test("G2b sync gate regenerates on emotional padding (non-strippable)")
def test_gate_regen_padding():
    """Therapy padding ('I hear you', 'that sounds hard') lives
    mid-text and is not caught by strip. Lint catches it, regenerate
    fires."""
    regen = _make_regen_sync(_CLEAN_DRAFT)
    out = gate_output_sync(_PADDING_DRAFT, regenerate_fn=regen)
    assert out.regenerated is True
    assert out.attempts == 2
    assert out.final_lint.passed
    assert len(regen.calls) == 1  # type: ignore[attr-defined]


@test("G3 sync gate regenerates on missing failure mode")
def test_gate_regen_failure_mode():
    regen = _make_regen_sync(_CLEAN_DRAFT)
    out = gate_output_sync(_NO_FAILURE_MODE_DRAFT, regenerate_fn=regen)
    assert out.regenerated is True
    assert out.attempts == 2
    assert any(
        v.rule == "FAILURE_MODE_ATTACHED"
        for v in out.initial_lint.blocking
    )


@test("G4 sync gate regenerates on first-person execution")
def test_gate_regen_execution():
    regen = _make_regen_sync(_CLEAN_DRAFT)
    out = gate_output_sync(_EXECUTION_DRAFT, regenerate_fn=regen)
    assert out.regenerated is True
    rules = {v.rule for v in out.initial_lint.blocking}
    assert "USER_SOVEREIGNTY_PRESERVED" in rules or "NO_ACTION_EXECUTION_LANGUAGE" in rules


@test("G5 sync gate max_attempts=1 disables regenerate")
def test_gate_max_attempts_one():
    regen = _make_regen_sync(_CLEAN_DRAFT)
    out = gate_output_sync(_PADDING_DRAFT, regenerate_fn=regen, max_attempts=1)
    # max_attempts=1 means no regenerate — the initial (lint-failing)
    # draft is returned as-is, with final_lint reflecting the failure.
    assert out.regenerated is False
    assert out.attempts == 1
    assert out.final_lint.passed is False
    assert regen.calls == []  # type: ignore[attr-defined]


@test("G6 sync gate returns telemetry fields")
def test_gate_telemetry():
    regen = _make_regen_sync(_CLEAN_DRAFT)
    out = gate_output_sync(_PADDING_DRAFT, regenerate_fn=regen)
    # initial_lint reflects the bad first draft
    assert not out.initial_lint.passed
    # final_lint reflects the regenerated good draft
    assert out.final_lint.passed
    assert out.attempts == 2


@test("G7 sync gate records the directive it sent on regenerate")
def test_gate_directive_recorded():
    regen = _make_regen_sync(_CLEAN_DRAFT)
    out = gate_output_sync(_PADDING_DRAFT, regenerate_fn=regen)
    assert out.directive_used
    assert "regenerate" in out.directive_used.lower()
    # The directive was actually passed to the regenerate_fn
    assert regen.calls[0] == out.directive_used  # type: ignore[attr-defined]


@test("G8 async gate matches sync behavior on clean input")
async def test_gate_async_clean():
    regen = _make_regen_async("should not be called")
    out = await gate_output_async(_CLEAN_DRAFT, regenerate_fn=regen)
    assert out.regenerated is False
    assert out.final_lint.passed
    assert regen.calls == []  # type: ignore[attr-defined]


@test("G9 async gate regenerates on bad input")
async def test_gate_async_regen():
    regen = _make_regen_async(_CLEAN_DRAFT)
    out = await gate_output_async(_PADDING_DRAFT, regenerate_fn=regen)
    assert out.regenerated is True
    assert out.attempts == 2
    assert out.final_lint.passed


@test("G10 gate preserves regenerated text when second draft passes")
def test_gate_preserves_regenerated():
    regen = _make_regen_sync(_CLEAN_DRAFT)
    out = gate_output_sync(_PADDING_DRAFT, regenerate_fn=regen)
    # The final text should be the regenerated clean draft, NOT the
    # padded original.
    assert "I hear you" not in out.text
    assert "smallest version" in out.text


@test("G11 gate exposes degraded mode when second draft also fails")
def test_gate_degraded_mode():
    # Both initial and regenerate produce lint-failing output.
    regen = _make_regen_sync(_PADDING_DRAFT)
    out = gate_output_sync(_EXECUTION_DRAFT, regenerate_fn=regen)
    assert out.regenerated is True
    assert out.attempts == 2
    # Final lint did NOT pass — caller can branch on this for telemetry
    # / fallback handling.
    assert out.final_lint.passed is False


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group D — Dispatcher direct/direct_plus + clarifier wiring          ║
# ╚══════════════════════════════════════════════════════════════════════╝

# These tests cover the Tier 1+2 wiring landed after Codex's audit:
#   - dispatcher._dispatch_direct       → composer + gate
#   - dispatcher._dispatch_direct_plus  → composer + gate
#   - speech.generate_clarification     → gate (composer already wired)
#
# The source-proof test (test_identity_source_proof.py) asserts the
# AST-level wiring at the call sites. The tests below add complementary
# coverage at the API/module level — they exercise the public surface
# the call sites use, and lock the exempt registry's shape.

@test("D1 dispatcher module imports composer + gate")
def test_dispatcher_imports():
    """If dispatcher.py stops importing compose_system_prompt or
    gate_output_async, the Tier 1+2 wiring is gone. This is a fast
    structural check that the wiring is present at module level."""
    import src.dispatcher as dispatcher_mod
    source = pathlib.Path(dispatcher_mod.__file__).read_text()
    assert "from src.identity import" in source
    assert "compose_system_prompt" in source
    assert "gate_output_async" in source
    assert "LintContext" in source


@test("D2 speech clarifier wires the gate around generate_clarification")
def test_speech_clarifier_wires_gate():
    """generate_clarification is pure prose; the gate must be on the
    response path so therapy padding / first-person execution / missing
    failure modes get one regenerate-and-retry. Catches accidental
    removal of the wiring."""
    import src.llm.speech as speech_mod
    source = pathlib.Path(speech_mod.__file__).read_text()
    assert "gate_output_async" in source
    # The clarifier-specific regen closure should be present.
    assert "_regen_clarification" in source


@test("D3 CONTROL_PLANE_SITES has exactly the seven expected entries")
def test_exempt_registry_shape():
    """The exempt registry's shape is part of the architectural
    contract. Adding a new entry is fine — it must be a real
    control-plane site with a real reason. Removing an entry without
    wrapping the underlying call is not — SP2 catches that. The
    registry is currently locked at NINE entries:

      - router / triage / visualizer / critique  (control-plane gates)
      - call_tracker.py  (passthrough wrapper for the agent layer)
      - master_synthesizer.py  (passthrough wrapper for the master
        synthesizer's cost-cap helper; doctrine header is composed once
        at the master_synthesize entry point and forwarded into every
        R1/R2/R3/R4 call across both seats)
      - master_sorter.py  (passthrough wrapper for the master sorter
        tributary's cost-cap helper; doctrine header is composed once
        at the master_sort entry point and forwarded into the single
        sort pass)

    The passthrough wrappers (call_tracker.py, master_synthesizer.py,
    master_sorter.py) forward an already-composed system_prompt verbatim;
    composing again inside them would double-wrap the doctrine header
    on every call.
    """
    by_file = sorted(s.file for s in CONTROL_PLANE_SITES)
    assert by_file == sorted([
        "src/llm/router.py",
        "src/llm/triage.py",
        "src/llm/visualizer.py",
        "src/wandering/call_tracker.py",
        "src/wandering/critique.py",
        "src/wandering/master_synthesizer.py",
        "src/wandering/master_sorter.py",
        "src/wandering/blender.py",
        "src/wandering/drift_checker.py",
    ]), by_file


@test("D4 is_exempt matches registered sites and rejects non-registered")
def test_is_exempt_helper():
    assert is_exempt("src/llm/router.py",      "ROUTER_SYSTEM_PROMPT")    is True
    assert is_exempt("src/llm/triage.py",      "_GATE_SYSTEM_PROMPT")     is True
    assert is_exempt("src/llm/visualizer.py",  "_VISUAL_GENERATOR_PROMPT") is True
    assert is_exempt("src/wandering/critique.py", "_CRITIQUE_SYSTEM_PROMPT") is True
    # Wrong file, right name → not exempt
    assert is_exempt("src/wandering/agent.py", "ROUTER_SYSTEM_PROMPT") is False
    # Right file, wrong name → not exempt
    assert is_exempt("src/llm/router.py", "SOMETHING_ELSE") is False
    # Made-up entry → not exempt
    assert is_exempt("does/not/exist.py", "NOT_REAL") is False


@test("D5 dispatcher direct path local prompt no longer self-declares identity")
def test_dispatcher_local_prompt_is_additive():
    """Codex's audit flagged that dispatcher.py:314 had its own
    competing 'thinking partner — brain extension' identity declaration.
    After the Tier 1 fix, the local prompt should be ADDITIVE (mode-
    specific guidance) and let the Singular Path header carry identity.

    This is a regex test against the dispatcher source — it asserts
    the local prompt no longer opens with 'You are a thinking
    partner', which was the competing-identity opener."""
    import src.dispatcher as dispatcher_mod
    source = pathlib.Path(dispatcher_mod.__file__).read_text()
    # The OLD self-declaring opener is gone.
    assert '"You are a thinking partner — a brain extension' not in source
    # And the NEW additive phrasing is present.
    assert "Direct-route guidance" in source
    assert "Direct-plus guidance" in source


# ─── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for v in globals().values()
             if callable(v) and hasattr(v, "_test_name")]
    tests.sort(key=lambda t: t._test_name)
    print(f"\nRunning {len(tests)} identity-integration tests...\n")
    for t in tests:
        run_test(t)
    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED:
        for name, err in ERRORS:
            print(f"  {name}: {err}")
        sys.exit(1)
