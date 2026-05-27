"""
Angular discipline LAWS tests.

Verifies that every domain agent's system prompt contains the three
non-negotiable laws:
    1. STAY IN YOUR ANGLE — lane discipline
    2. HIDDEN VARIABLE QUOTA — at least one is_hidden:true
    3. LAYERED DEPTH WITHIN YOUR ANGLE — specific mechanism, not just label

These laws are the architectural backbone — without them, the wuxing
engine collapses into 5 LLMs producing similar answers. Tests are
prompt-string contract tests; no LLM calls.

Run: PYTHONPATH=. python3 tests/test_angular_discipline.py
"""

from __future__ import annotations

from src.llm.prompts import get_domain_prompt


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
        fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


DOMAINS = ("physics", "mathematics", "psychology", "philosophy", "chemistry")


# ---------------------------------------------------------------------------
# 1. ANGULAR DISCIPLINE block exists in all 5 domains
# ---------------------------------------------------------------------------

@test("1.1 every domain prompt contains 'ANGULAR DISCIPLINE — LAWS'")
def test_block_present_everywhere():
    for d in DOMAINS:
        prompt = get_domain_prompt(d)
        assert "ANGULAR DISCIPLINE" in prompt, f"missing in {d}"


# ---------------------------------------------------------------------------
# 2. STAY IN YOUR ANGLE law present
# ---------------------------------------------------------------------------

@test("2.1 every domain has 'STAY IN YOUR ANGLE' law")
def test_stay_in_angle_law():
    for d in DOMAINS:
        prompt = get_domain_prompt(d)
        assert "STAY IN YOUR ANGLE" in prompt, f"missing STAY IN YOUR ANGLE in {d}"


@test("2.2 each domain names its OWN identity in the lane law")
def test_lane_law_names_own_domain():
    expected = {
        "physics": "PHYSICS",
        "mathematics": "MATHEMATICS",
        "psychology": "PSYCHOLOGY",
        "philosophy": "PHILOSOPHY",
        "chemistry": "CHEMISTRY",
    }
    for d, ident in expected.items():
        prompt = get_domain_prompt(d)
        # The identity callout "You are X" should appear in the lane law section
        assert f"You are {ident}" in prompt, f"missing 'You are {ident}' in {d}"


@test("2.3 each domain names at least 3 sibling domains to NOT do")
def test_lane_law_names_siblings():
    # Each domain's lane law should disambiguate against 3+ siblings by name
    sibling_names = {
        "physics":     ["Psychology", "Philosophy", "Mathematics", "Chemistry"],
        "mathematics": ["Physics", "Psychology", "Philosophy", "Chemistry"],
        "psychology":  ["Physics", "Mathematics", "Philosophy", "Chemistry"],
        "philosophy":  ["Physics", "Mathematics", "Psychology", "Chemistry"],
        "chemistry":   ["Physics", "Mathematics", "Psychology", "Philosophy"],
    }
    for d, siblings in sibling_names.items():
        prompt = get_domain_prompt(d)
        named = [s for s in siblings if s in prompt]
        assert len(named) >= 3, (
            f"{d}: only {len(named)} sibling(s) named ({named}); need at least 3"
        )


# ---------------------------------------------------------------------------
# 3. HIDDEN VARIABLE QUOTA law present
# ---------------------------------------------------------------------------

@test("3.1 every domain has 'HIDDEN VARIABLE QUOTA' law")
def test_hidden_var_quota_law():
    for d in DOMAINS:
        prompt = get_domain_prompt(d)
        assert "HIDDEN VARIABLE QUOTA" in prompt, f"missing in {d}"


@test("3.2 hidden-var law references is_hidden:true requirement")
def test_hidden_var_law_references_field():
    for d in DOMAINS:
        prompt = get_domain_prompt(d)
        # The actual JSON-schema field is `is_hidden`; the law must reference it
        assert "is_hidden: true" in prompt or "`is_hidden: true`" in prompt, (
            f"hidden-var law in {d} doesn't reference is_hidden:true"
        )


@test("3.3 hidden-var law sets explicit FAILED consequence for skipping")
def test_hidden_var_failure_clause():
    # If you skip the hidden-variable surfacing, you have FAILED.
    for d in DOMAINS:
        prompt = get_domain_prompt(d)
        assert "FAILED" in prompt, f"{d}: missing FAILED clause"


# ---------------------------------------------------------------------------
# 4. LAYERED DEPTH law present
# ---------------------------------------------------------------------------

@test("4.1 every domain has 'LAYERED DEPTH WITHIN YOUR ANGLE' law")
def test_layered_depth_law():
    for d in DOMAINS:
        prompt = get_domain_prompt(d)
        assert "LAYERED DEPTH" in prompt, f"missing LAYERED DEPTH in {d}"


@test("4.2 layered-depth law contrasts failure vs success example")
def test_layered_depth_examples():
    for d in DOMAINS:
        prompt = get_domain_prompt(d)
        assert "is failure" in prompt and "is success" in prompt, (
            f"{d}: missing failure-vs-success exemplar pair"
        )


# ---------------------------------------------------------------------------
# 5. Block placement — before OUTPUT FORMAT
# ---------------------------------------------------------------------------

@test("5.1 ANGULAR DISCIPLINE block appears BEFORE OUTPUT FORMAT in every domain")
def test_block_ordering():
    for d in DOMAINS:
        prompt = get_domain_prompt(d)
        angular_idx = prompt.find("ANGULAR DISCIPLINE")
        output_idx = prompt.find("OUTPUT FORMAT")
        assert angular_idx > 0, f"{d}: ANGULAR DISCIPLINE missing"
        assert output_idx > 0, f"{d}: OUTPUT FORMAT missing"
        assert angular_idx < output_idx, (
            f"{d}: ANGULAR DISCIPLINE at {angular_idx} comes AFTER OUTPUT FORMAT at {output_idx}"
        )


# ---------------------------------------------------------------------------
# 6. No domain accidentally references the wrong identity
# ---------------------------------------------------------------------------

@test("6.1 chemistry uniquely references 'synthesize ACROSS' language")
def test_chemistry_is_synthesizer():
    prompt = get_domain_prompt("chemistry")
    assert "synthesize ACROSS" in prompt or "bond what" in prompt, (
        "chemistry should explicitly position itself as cross-domain synthesizer"
    )


@test("6.2 chemistry requires findings to reference OTHER domains")
def test_chemistry_cross_reference_required():
    prompt = get_domain_prompt("chemistry")
    assert "other domain" in prompt.lower() or "OTHER domain" in prompt, (
        "chemistry should require cross-domain references"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_block_present_everywhere,
    test_stay_in_angle_law,
    test_lane_law_names_own_domain,
    test_lane_law_names_siblings,
    test_hidden_var_quota_law,
    test_hidden_var_law_references_field,
    test_hidden_var_failure_clause,
    test_layered_depth_law,
    test_layered_depth_examples,
    test_block_ordering,
    test_chemistry_is_synthesizer,
    test_chemistry_cross_reference_required,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} angular discipline LAWS tests...")
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
