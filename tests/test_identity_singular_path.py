"""
Tests for the identity layer — The Singular Path.

Covers, in order:

  Group A — Discipline unit tests
    A1  goal_supremacy.discriminate scores by goal overlap
    A2  goal_supremacy.discriminate flags attachment-only claims
    A3  goal_supremacy.surface_real_goal detects contradiction
    A4  goal_supremacy.surface_real_goal stays quiet on consistent
    A5  long_horizon.compounding_signal direction
    A6  long_horizon.project returns horizons in order
    A7  opportunity_capture.test scoring and verdict buckets
    A8  opportunity_capture.test penalizes fashion + no cost
    A9  attachment_detection.scan flags sunk_cost
    A10 attachment_detection.scan flags identity_protection
    A11 attachment_detection.scan flags urgency_as_fear
    A12 attachment_detection.scan flags patience_as_avoidance
    A13 attachment_detection.scan flags consensus_drift
    A14 attachment_detection.scan empty text returns no flags
    A15 resource_conversion.latent_uses matches category
    A16 resource_conversion.evaluate scores convertibility

  Group B — Sovereignty (anti-laws + Map-Not-March)
    B1  ANTI_LAWS contains all three kinds
    B2  position_hash stable across whitespace/filler variants
    B3  MapNotMarchCounter increments on note()
    B4  MapNotMarchCounter should_force_map fires at threshold
    B5  MapNotMarchCounter reset clears one session, keeps others

  Group C — Voice (strip + lint)
    C1  strip_openers removes "great question"
    C2  strip_openers removes "happy to help"
    C3  strip_openers chains multiple openers
    C4  strip_openers leaves clean text alone
    C5  lint passes clean output
    C6  lint catches opener residue
    C7  lint catches emotional padding
    C8  lint allows emotional naming when state is relevant
    C9  lint catches missing failure mode
    C10 lint catches first-person execution
    C11 lint requires cartography after Map-Not-March threshold
    C12 lint requires real-goal probe when flagged
    C13 should_regenerate fires on any BLOCK violation
    C14 build_regenerate_directive composes remediation lines

  Group D — Seven identity criteria (end-to-end on synthetic outputs)
    D1  no_opener_fluff
    D2  no_emotional_padding_when_irrelevant
    D3  failure_mode_attached
    D4  no_repeated_argument (post-threshold)
    D5  user_sovereignty_preserved
    D6  real_goal_surfaced_when_flagged
    D7  no_action_execution_language

  Group E — Smoke
    E1  package imports cleanly
    E2  SYSTEM_PROMPT_HEADER length in target window
    E3  THINKING_CHECKLIST has eight entries
"""

from __future__ import annotations

import asyncio
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
    DOCTRINE_NAME,
    DOCTRINE_VERSION,
    SYSTEM_PROMPT_HEADER,
    THINKING_CHECKLIST,
    Context,
    Goal,
    Position,
    ANTI_LAWS,
    MAP_NOT_MARCH_THRESHOLD,
    MapNotMarchCounter,
    position_hash,
)
from src.identity.sovereignty import AntiLawKind
from src.identity.disciplines import (
    AttachmentFlag,
    AttachmentKind,
    CaptureVerdict,
    ConvertibilityScore,
    HorizonRead,
    Opening,
    Resource,
    ServeScore,
    compounding_signal,
    discriminate,
    evaluate,
    latent_uses,
    project,
    scan,
    surface_real_goal,
    test as opp_test,
)
from src.identity.voice import (
    LintContext,
    Severity,
    build_regenerate_directive,
    lint,
    should_regenerate,
    strip_openers,
)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group A — Discipline unit tests                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

@test("A1 goal_supremacy.discriminate scores by goal overlap")
def test_discriminate_overlap():
    goal = Goal(stated="ship the product", real="ship the product")
    serves = discriminate("ship the product to first ten users", goal)
    diverts = discriminate("research more frameworks for six months", goal)
    assert serves.verdict in ("serves", "neutral"), serves
    assert diverts.verdict == "diverts", diverts
    assert serves.score > diverts.score


@test("A2 goal_supremacy.discriminate flags attachment-only claims")
def test_discriminate_attachment_only():
    goal = Goal(
        stated="ship the product",
        real="cut scope and launch",
        surfaced=True,
        signals=("I keep adding features",),
    )
    claim = "preserve the full product vision and ship the product later"
    score = discriminate(claim, goal)
    assert score.serves_attachment_only is True, score
    assert score.verdict == "diverts"


@test("A3 goal_supremacy.surface_real_goal detects contradiction")
def test_surface_real_goal_detects():
    goal = surface_real_goal(
        "I want to ship",
        ("I keep finding things to make perfect first",),
    )
    assert goal.surfaced is True
    assert goal.real != goal.stated
    assert goal.signals  # non-empty


@test("A4 goal_supremacy.surface_real_goal stays quiet on consistent")
def test_surface_real_goal_consistent():
    goal = surface_real_goal(
        "I want to ship",
        ("I have a small scope locked",),
    )
    assert goal.surfaced is False
    assert goal.real == goal.stated
    assert goal.is_consistent()


@test("A5 long_horizon.compounding_signal direction")
def test_compounding_signal():
    assert compounding_signal("ship and document the system") == 1
    assert compounding_signal("wait and polish for another quarter") == -1
    assert compounding_signal("unrelated neutral text") == 0


@test("A6 long_horizon.project returns horizons in order")
def test_project_horizons():
    read = project("build the audience", (3, 6, 24))
    assert isinstance(read, HorizonRead)
    months = [s.months_out for s in read.signals]
    assert months == [3, 6, 24]
    assert read.dominant == read.signals[-1].direction


@test("A7 opportunity_capture.test scoring and verdict buckets")
def test_opportunity_buckets():
    goal = Goal(stated="ship the product", real="ship the product")
    # An opening that ticks most boxes
    strong = Opening(
        description="cut scope and ship the product to ten users now",
        claimed_cost="lose two planned features",
        requires=(),
    )
    v = opp_test(strong, goal)
    assert v.verdict in ("capture", "surface"), v
    assert v.score >= 4

    # An opening that fails most boxes
    weak = Opening(
        description="permanent irreversible burn the bridge",
        claimed_cost=None,
        requires=("series A funding",),
    )
    v2 = opp_test(weak, goal, position=Position(text="bootstrapping", hash="x"))
    assert v2.verdict == "skip", v2


@test("A8 opportunity_capture.test penalizes fashion + no cost")
def test_opportunity_fashion():
    goal = Goal(stated="ship the product", real="ship the product")
    op = Opening(
        description="everyone is doing AI agents and it's a trend you need",
        claimed_cost=None,
    )
    v = opp_test(op, goal)
    assert v.answers["real_opening"] is False, v.answers


@test("A9 attachment_detection.scan flags sunk_cost")
def test_attachment_sunk_cost():
    flags = scan("I've already spent two years on this and can't waste that work")
    kinds = {f.kind for f in flags}
    assert AttachmentKind.SUNK_COST in kinds


@test("A10 attachment_detection.scan flags identity_protection")
def test_attachment_identity():
    flags = scan("I'm not the kind of person who quits — people will think I gave up")
    kinds = {f.kind for f in flags}
    assert AttachmentKind.IDENTITY_PROTECTION in kinds


@test("A11 attachment_detection.scan flags urgency_as_fear")
def test_attachment_urgency():
    flags = scan("I have to act right now before someone else does — I'm freaking out")
    kinds = {f.kind for f in flags}
    assert AttachmentKind.URGENCY_AS_FEAR in kinds


@test("A12 attachment_detection.scan flags patience_as_avoidance")
def test_attachment_patience():
    flags = scan("I just need more research and more data before I'm ready, maybe next quarter")
    kinds = {f.kind for f in flags}
    assert AttachmentKind.PATIENCE_AS_AVOIDANCE in kinds


@test("A13 attachment_detection.scan flags consensus_drift")
def test_attachment_consensus():
    flags = scan("Everyone says you have to do the standard advice — the playbook is clear")
    kinds = {f.kind for f in flags}
    assert AttachmentKind.CONSENSUS_DRIFT in kinds


@test("A14 attachment_detection.scan empty text returns no flags")
def test_attachment_empty():
    assert scan("") == []
    assert scan("   ") == []


@test("A15 resource_conversion.latent_uses matches category")
def test_resource_latent_uses():
    sunk = latent_uses("I wasted two years on this — it didn't work out")
    assert any("apprenticeship" in u or "credible" in u or "calibrated" in u or "proof" in u
               for u in sunk), sunk


@test("A16 resource_conversion.evaluate scores convertibility")
def test_resource_evaluate():
    goal = Goal(stated="ship the product", real="ship the product")
    r = Resource(text="dead end approach that doesn't work", goal_relevance=0.5)
    score = evaluate(r, goal)
    assert isinstance(score, ConvertibilityScore)
    assert score.score > 0.40
    assert score.category in ("dead_end", "general")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group B — Sovereignty                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

@test("B1 ANTI_LAWS contains all three kinds")
def test_anti_laws_complete():
    kinds = {law.kind for law in ANTI_LAWS}
    assert AntiLawKind.NO_EXECUTION in kinds
    assert AntiLawKind.NO_ARGUMENT  in kinds
    assert AntiLawKind.NO_PADDING   in kinds


@test("B2 position_hash stable across whitespace/filler variants")
def test_position_hash_stable():
    h1 = position_hash("I really think we should ship now")
    h2 = position_hash("I think we should ship now.")
    h3 = position_hash("  i  think we    should ship now!  ")
    assert h1 == h2 == h3, (h1, h2, h3)
    h_diff = position_hash("we should wait")
    assert h_diff != h1


@test("B3 MapNotMarchCounter increments on note()")
def test_counter_increments():
    c = MapNotMarchCounter()
    n1 = c.note("S1", "we should ship now")
    n2 = c.note("S1", "we should ship now")
    n3 = c.note("S1", "we should ship now")
    assert (n1, n2, n3) == (1, 2, 3)


@test("B4 MapNotMarchCounter should_force_map fires at threshold")
def test_counter_force_map():
    c = MapNotMarchCounter()
    text = "we should ship now"
    c.note("S1", text)
    assert c.should_force_map("S1", text) is False
    c.note("S1", text)
    assert c.should_force_map("S1", text) is True
    assert c.current("S1", text) >= MAP_NOT_MARCH_THRESHOLD


@test("B5 MapNotMarchCounter reset clears one session, keeps others")
def test_counter_reset_scoped():
    c = MapNotMarchCounter()
    c.note("S1", "ship now")
    c.note("S2", "ship now")
    c.reset("S1")
    assert c.current("S1", "ship now") == 0
    assert c.current("S2", "ship now") == 1


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group C — Voice (strip + lint)                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

@test("C1 strip_openers removes 'great question'")
def test_strip_great_question():
    out = strip_openers("Great question! The answer is to cut scope.")
    assert out.startswith("The answer"), out


@test("C2 strip_openers removes 'happy to help'")
def test_strip_happy_to_help():
    out = strip_openers("I'm happy to help! Here's the move.")
    assert out.startswith("Here's the move"), out


@test("C3 strip_openers chains multiple openers")
def test_strip_chained():
    out = strip_openers("Great question! Happy to help. The real call is to ship.")
    assert out.startswith("The real call"), out


@test("C4 strip_openers leaves clean text alone")
def test_strip_clean():
    src = "The move is to cut scope and ship by Friday."
    assert strip_openers(src) == src


@test("C5 lint passes clean output")
def test_lint_clean():
    text = (
        "The move is to ship the smallest version that proves the bet. "
        "Cost: you lose the polish you wanted. This breaks if early "
        "users care more about polish than presence — watch for "
        "negative feedback on rough edges, not on missing features."
    )
    result = lint(text)
    assert result.passed, result.violations


@test("C6 lint catches opener residue")
def test_lint_opener():
    text = "Great question! The move is to ship. " + ("Cost: lose polish. " * 5)
    result = lint(text)
    assert any(v.rule == "NO_OPENER_FLUFF" for v in result.violations), result


@test("C7 lint catches emotional padding")
def test_lint_padding():
    text = (
        "I hear you. That sounds really hard. The decision space is "
        "between A and B. Cost: A burns runway, B burns morale. "
        "Wrong if the team is more resilient than you think."
    )
    result = lint(text)
    assert any(v.rule == "NO_EMOTIONAL_PADDING" for v in result.violations), result


@test("C8 lint allows emotional naming when state is relevant")
def test_lint_padding_relevant():
    text = (
        "That sounds really hard, and your exhaustion is directly "
        "shaping the timeline here. The move is to take a structured "
        "two-week rest before deciding. Wrong if rest itself raises "
        "your anxiety past the work."
    )
    result = lint(text, LintContext(emotional_state_relevant=True))
    # One mention is allowed; should not block.
    blocking_padding = [v for v in result.blocking if v.rule == "NO_EMOTIONAL_PADDING"]
    assert blocking_padding == [], blocking_padding


@test("C9 lint catches missing failure mode")
def test_lint_missing_failure():
    text = (
        "The move is to cut three features and ship by Friday. "
        "Tell beta users they can wait for the polished version. "
        "Recruit two reviewers from the existing list. Send the "
        "launch email Friday morning at nine local time."
    )
    result = lint(text)
    assert any(v.rule == "FAILURE_MODE_ATTACHED" for v in result.violations), result


@test("C10 lint catches first-person execution")
def test_lint_sovereignty():
    text = "I'll edit the config file for you and commit it. " + ("Cost: lose polish. " * 5)
    result = lint(text)
    assert any(v.rule == "USER_SOVEREIGNTY_PRESERVED" for v in result.violations), result


@test("C11 lint requires cartography after Map-Not-March threshold")
def test_lint_cartography_required():
    text = (
        "As I mentioned earlier, the right move is to ship now. "
        "You really need to commit to that direction. I have to push "
        "back on the idea of waiting. " + ("Cost: lose polish. " * 3)
    )
    result = lint(text, LintContext(map_not_march_strike=2))
    assert any(v.rule == "NO_REPEATED_ARGUMENT" for v in result.violations), result


@test("C12 lint requires real-goal probe when flagged")
def test_lint_real_goal_probe():
    text = "The move is to ship. Cost: polish goes. Wrong if users care."
    result = lint(text, LintContext(real_goal_surfaced=True))
    assert any(v.rule == "REAL_GOAL_SURFACED" for v in result.violations), result


@test("C13 should_regenerate fires on any BLOCK violation")
def test_should_regenerate():
    text = "Great question! The move is to ship. Done."
    result = lint(text)
    assert should_regenerate(result) is True
    clean = (
        "The move is to ship the smallest version that proves the bet. "
        "Cost: you lose the polish you wanted. This breaks if early "
        "users care more about polish than presence."
    )
    clean_result = lint(clean)
    assert should_regenerate(clean_result) is False


@test("C14 build_regenerate_directive composes remediation lines")
def test_regenerate_directive():
    text = "Great question! I'll commit the changes. Done."
    result = lint(text)
    directive = build_regenerate_directive(result)
    assert "regenerate" in directive.lower()
    assert len(directive.splitlines()) >= 2


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group D — Seven identity criteria (end-to-end on synthetic outputs) ║
# ╚══════════════════════════════════════════════════════════════════════╝

# Each test composes a synthetic violating output, runs lint, and
# asserts the named criterion fires. These are the seven criteria the
# user named as the must-cover surface.

@test("D1 no_opener_fluff (criterion 1)")
def test_criterion_opener():
    bad = "Great question! Cut scope and ship. " + ("Cost: lose polish. " * 5)
    r = lint(bad)
    assert any(v.rule == "NO_OPENER_FLUFF" for v in r.violations)


@test("D2 no_emotional_padding_when_irrelevant (criterion 2)")
def test_criterion_padding():
    bad = (
        "I hear you. The decision space is between A and B. "
        + ("Cost: A burns runway. " * 5)
    )
    r = lint(bad, LintContext(emotional_state_relevant=False))
    assert any(v.rule == "NO_EMOTIONAL_PADDING" for v in r.violations)


@test("D3 failure_mode_attached (criterion 3)")
def test_criterion_failure_mode():
    bad = (
        "Cut three features by Friday. Tell beta users they can wait. "
        "Recruit two reviewers from the existing list. Send the "
        "launch email Friday morning at nine local time, then keep "
        "the team focused on stability through Monday."
    )
    r = lint(bad)
    assert any(v.rule == "FAILURE_MODE_ATTACHED" for v in r.violations)


@test("D4 no_repeated_argument post-threshold (criterion 4)")
def test_criterion_no_repeat_arg():
    bad = "As I said earlier, ship now. I have to insist on that. " + ("Cost: lose polish. " * 3)
    r = lint(bad, LintContext(map_not_march_strike=2))
    assert any(v.rule == "NO_REPEATED_ARGUMENT" for v in r.violations)


@test("D5 user_sovereignty_preserved (criterion 5)")
def test_criterion_sovereignty():
    bad = "I'll edit the config and commit it. " + ("Cost: lose polish. " * 5)
    r = lint(bad)
    assert any(v.rule == "USER_SOVEREIGNTY_PRESERVED" for v in r.violations)


@test("D6 real_goal_surfaced_when_flagged (criterion 6)")
def test_criterion_real_goal():
    bad = "Ship by Friday. Cut three features. Wrong if users want polish."
    r = lint(bad, LintContext(real_goal_surfaced=True))
    assert any(v.rule == "REAL_GOAL_SURFACED" for v in r.violations)


@test("D7 no_action_execution_language (criterion 7)")
def test_criterion_no_execution():
    bad = "I'll deploy the fix to production now. " + ("Cost: lose polish. " * 5)
    r = lint(bad)
    assert any(v.rule == "NO_ACTION_EXECUTION_LANGUAGE" for v in r.violations)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Group E — Smoke                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

@test("E1 package imports cleanly")
def test_smoke_imports():
    assert DOCTRINE_NAME == "The Singular Path"
    assert DOCTRINE_VERSION.startswith("0.")


@test("E2 SYSTEM_PROMPT_HEADER length in target window")
def test_smoke_header_length():
    # Target window 250-500 words. The lower bound enforces that the
    # header carries the full doctrine; the upper bound guards against
    # unbounded growth that would compete with mode-local prompts for
    # the model's attention budget. 0.3.1 refinements
    # ("current confirmed real goal" + "substantive recommendation"
    # + small-answers carve-out) landed at ~453 words.
    words = SYSTEM_PROMPT_HEADER.split()
    n = len(words)
    assert 250 <= n <= 500, f"SYSTEM_PROMPT_HEADER has {n} words; target 250-500"


@test("E3 THINKING_CHECKLIST has eight entries")
def test_smoke_checklist_eight():
    assert len(THINKING_CHECKLIST) == 8
    assert all(isinstance(q, str) and q.strip().endswith("?") for q in THINKING_CHECKLIST)


# ─── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for v in globals().values()
             if callable(v) and hasattr(v, "_test_name")]
    tests.sort(key=lambda t: t._test_name)
    print(f"\nRunning {len(tests)} identity-layer tests...\n")
    for t in tests:
        run_test(t)
    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED:
        for name, err in ERRORS:
            print(f"  {name}: {err}")
        sys.exit(1)
