"""
Checklist tests — verifies the 27 questions are well-formed and that the
route+effort distribution matches the locked design.

No LLM calls, no API. Pure data + helpers.

Run: PYTHONPATH=. python3 tests/test_checklist.py
"""

from __future__ import annotations

from src.llm.checklist import (
    QUESTIONS,
    build_checklist_block,
    get_checklist_codes,
)
from src.llm.effort import Effort
from src.llm.triage import Route


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


# ---------------------------------------------------------------------------
# 1. Question bank integrity
# ---------------------------------------------------------------------------

@test("1.1 exactly 27 questions in the bank")
def test_27_questions():
    assert len(QUESTIONS) == 27


@test("1.2 every question has non-empty text")
def test_non_empty_text():
    for code, text in QUESTIONS.items():
        assert text and len(text) > 20, f"empty/short text for {code}"


@test("1.3 codes cover A1-A4, B5-B8, C9-C13, D14-D19, E20-E24, F25-F27")
def test_codes_coverage():
    expected = (
        ["A1", "A2", "A3", "A4"]
        + ["B5", "B6", "B7", "B8"]
        + ["C9", "C10", "C11", "C12", "C13"]
        + ["D14", "D15", "D16", "D17", "D18", "D19"]
        + ["E20", "E21", "E22", "E23", "E24"]
        + ["F25", "F26", "F27"]
    )
    assert sorted(QUESTIONS.keys()) == sorted(expected)


# ---------------------------------------------------------------------------
# 2. Distribution per route + effort
# ---------------------------------------------------------------------------

@test("2.1 TRIVIAL → empty")
def test_trivial_empty():
    assert get_checklist_codes(Route.TRIVIAL, Effort.LOW) == []
    assert get_checklist_codes(Route.TRIVIAL, Effort.HIGH) == []


@test("2.2 DIRECT → only F25")
def test_direct_only_f25():
    assert get_checklist_codes(Route.DIRECT, Effort.LOW) == ["F25"]


@test("2.3 DIRECT_PLUS → A1 + F25-F27 (4 codes)")
def test_direct_plus_codes():
    codes = get_checklist_codes(Route.DIRECT_PLUS, Effort.LOW)
    assert len(codes) == 4
    assert "A1" in codes
    assert "F25" in codes
    assert "F26" in codes
    assert "F27" in codes


@test("2.4 DEEP / LOW → 6 codes (A1-A4 + D14 + E20)")
def test_deep_low_six():
    codes = get_checklist_codes(Route.DEEP, Effort.LOW)
    assert len(codes) == 6
    assert "A1" in codes and "A2" in codes and "A3" in codes and "A4" in codes
    assert "D14" in codes
    assert "E20" in codes


@test("2.5 DEEP / MEDIUM → 18 codes (A + B + D + E22-E24 + F25)")
def test_deep_medium_eighteen():
    codes = get_checklist_codes(Route.DEEP, Effort.MEDIUM)
    assert len(codes) == 18
    # A complete
    for c in ("A1", "A2", "A3", "A4"):
        assert c in codes
    # B complete
    for c in ("B5", "B6", "B7", "B8"):
        assert c in codes
    # D complete
    for c in ("D14", "D15", "D16", "D17", "D18", "D19"):
        assert c in codes
    # E22-E24 (later three)
    for c in ("E22", "E23", "E24"):
        assert c in codes
    # F25
    assert "F25" in codes


@test("2.6 DEEP / HIGH → all 27")
def test_deep_high_all():
    codes = get_checklist_codes(Route.DEEP, Effort.HIGH)
    assert len(codes) == 27


@test("2.7 DEEP / AUTO → all 27 (same as HIGH)")
def test_deep_auto_all():
    codes = get_checklist_codes(Route.DEEP, Effort.AUTO)
    assert len(codes) == 27


# ---------------------------------------------------------------------------
# 3. build_checklist_block
# ---------------------------------------------------------------------------

@test("3.1 TRIVIAL → empty string")
def test_block_trivial_empty():
    assert build_checklist_block(Route.TRIVIAL, Effort.LOW) == ""


@test("3.2 DIRECT block contains F25 question text + header")
def test_block_direct_contains_f25():
    block = build_checklist_block(Route.DIRECT, Effort.LOW)
    assert "ANGULAR CHECKLIST" in block
    assert "F25" in block
    # F25 references "engine's own gravity" per the question bank
    assert "engine has its own gravity" in block.lower() or "gravity" in block.lower()


@test("3.3 DEEP/HIGH block contains all 27 codes")
def test_block_deep_high_all_codes():
    block = build_checklist_block(Route.DEEP, Effort.HIGH)
    for code in QUESTIONS.keys():
        assert code in block, f"missing {code} in HIGH block"


@test("3.4 DEEP/LOW block contains exactly 6 codes, not more")
def test_block_deep_low_count():
    block = build_checklist_block(Route.DEEP, Effort.LOW)
    # Count code occurrences as line prefixes "Cn. "
    import re
    code_lines = re.findall(r"^[A-F]\d+\.\s", block, re.MULTILINE)
    assert len(code_lines) == 6


@test("3.5 block always tells the LLM NOT to enumerate codes in output")
def test_block_implicit_directive():
    block = build_checklist_block(Route.DIRECT, Effort.LOW)
    assert "IMPLICITLY" in block or "implicitly" in block.lower()
    assert "do not enumerate" in block.lower() or "do not list" in block.lower() \
        or "do not enumerate them" in block.lower()


# ---------------------------------------------------------------------------
# 4. Determinism — same inputs always yield same output
# ---------------------------------------------------------------------------

@test("4.1 get_checklist_codes is deterministic across calls")
def test_codes_deterministic():
    for route in (Route.TRIVIAL, Route.DIRECT, Route.DIRECT_PLUS, Route.DEEP):
        for effort in (Effort.LOW, Effort.MEDIUM, Effort.HIGH, Effort.AUTO):
            c1 = get_checklist_codes(route, effort)
            c2 = get_checklist_codes(route, effort)
            assert c1 == c2


@test("4.2 returned lists are independent (mutation does not affect future calls)")
def test_codes_independent():
    c1 = get_checklist_codes(Route.DEEP, Effort.HIGH)
    c1.append("BOGUS")
    c2 = get_checklist_codes(Route.DEEP, Effort.HIGH)
    assert "BOGUS" not in c2


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_27_questions,
    test_non_empty_text,
    test_codes_coverage,
    test_trivial_empty,
    test_direct_only_f25,
    test_direct_plus_codes,
    test_deep_low_six,
    test_deep_medium_eighteen,
    test_deep_high_all,
    test_deep_auto_all,
    test_block_trivial_empty,
    test_block_direct_contains_f25,
    test_block_deep_high_all_codes,
    test_block_deep_low_count,
    test_block_implicit_directive,
    test_codes_deterministic,
    test_codes_independent,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} checklist tests...")
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
