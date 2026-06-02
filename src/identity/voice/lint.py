"""
Identity lint — structured checks over a generated output.

The lint pass produces a LintResult with zero or more LintViolations.
The caller decides whether to regenerate based on severity. When
regeneration fires, the caller composes a stronger directive
(build_regenerate_directive) and re-runs the model — it does NOT try
to rewrite the output in place.

Why lint-then-regenerate rather than regex-rewrite the output:

  - Hedging and padding live mid-sentence and depend on context.
    Regex rewriting at that depth is brittle and produces
    grammatically awkward text the model would never write.
  - The model already knows the doctrine; if the lint says a rule
    failed, a stronger directive almost always produces a clean
    second draft.
  - Each regenerate doubles the LLM cost on the failing path, so
    the lint thresholds are tuned to keep regenerate rate <20% on
    nominal traffic. The caller increments an
    `identity_regenerate` counter to watch this in production.

The seven identity criteria checked here:

  1. NO_OPENER_FLUFF              — strip should have handled it; if
                                    not, lint catches the remainder
  2. NO_EMOTIONAL_PADDING         — therapy openers and cushioning
                                    prefixes that don't carry signal
  3. FAILURE_MODE_ATTACHED        — the output must name where the
                                    advice could break
  4. NO_REPEATED_ARGUMENT         — after Map-Not-March threshold,
                                    the response must shift to
                                    cartography format
  5. USER_SOVEREIGNTY_PRESERVED   — no "I'll edit / I'll commit /
                                    I'll run" first-person actions
  6. REAL_GOAL_SURFACED           — when the engine flagged a real-
                                    goal contradiction, the output
                                    must include the probe
  7. NO_ACTION_EXECUTION_LANGUAGE — covered by criterion 5 above,
                                    kept separate for telemetry
                                    granularity

Each criterion maps to a small set of regex patterns and a structural
check. False positives cost a regenerate; false negatives cost a
violation in the user's output. The thresholds favor false positives
within a small budget — better to regenerate occasionally than to
ship a violation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from src.identity.sovereignty import (
    ANTI_LAWS,
    AntiLawKind,
    get_anti_law,
)


class Severity(str, Enum):
    """Lint severities.

    BLOCK   — must regenerate; will not ship as-is.
    WARN    — surface to the operator, do not block.
    INFO    — telemetry only, no action.
    """

    BLOCK = "block"
    WARN  = "warn"
    INFO  = "info"


@dataclass(frozen=True)
class LintViolation:
    """One violation found by the lint pass."""

    rule:     str
    severity: Severity
    detail:   str
    excerpt:  str | None = None


@dataclass(frozen=True)
class LintContext:
    """Inputs the lint pass needs beyond the output text itself.

    `real_goal_surfaced` is True when goal_supremacy fired and the
    engine expects the probe to appear in the output.

    `map_not_march_strike` is the current strike count for the
    user's position — when >= sovereignty.MAP_NOT_MARCH_THRESHOLD the
    output must already be in cartography format.

    `emotional_state_relevant` is True when the user's emotional
    state is directly tied to the decision (e.g. burnout question on
    a 12-month roadmap). When True, emotional-naming is allowed; when
    False, padding patterns are violations."""

    real_goal_surfaced:        bool = False
    map_not_march_strike:      int  = 0
    emotional_state_relevant:  bool = False


@dataclass(frozen=True)
class LintResult:
    """Output of a lint pass."""

    passed:     bool
    violations: tuple[LintViolation, ...]

    @property
    def blocking(self) -> tuple[LintViolation, ...]:
        return tuple(v for v in self.violations if v.severity == Severity.BLOCK)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

_OPENER_RESIDUE = (
    re.compile(r"^\s*(?:great|excellent|wonderful) (?:question|point)", re.I),
    re.compile(r"^\s*(?:certainly|absolutely|of course)[,!.]", re.I),
    re.compile(r"^\s*(?:thanks|thank you) for (?:sharing|asking)", re.I),
)

_PADDING_PATTERNS = (
    re.compile(r"\bi hear you\b", re.I),
    re.compile(r"\bthat sounds (?:really )?(?:hard|tough|difficult)\b", re.I),
    re.compile(r"\bi['’]m sorry you['’]re (?:going through|dealing with|feeling)\b", re.I),
    re.compile(r"\bjust take a deep breath\b", re.I),
    re.compile(r"\bbe kind to yourself\b", re.I),
    re.compile(r"\bgive yourself grace\b", re.I),
    re.compile(r"\bhold space (?:for|with)\b", re.I),
    re.compile(r"\bsit with (?:that|this) (?:feeling|emotion)\b", re.I),
)

_FAILURE_MODE_CUES = (
    "fail",
    "wrong",
    "break",
    "breaks",
    "broken",
    "trap",
    "risk",
    "if i'm wrong",
    "if this is wrong",
    "this could be wrong",
    "where this breaks",
    "what would prove",
    "watch for",
    "warning sign",
    "red flag",
    "danger",
    "downside",
    "cost",
    "could backfire",
)

_REPEAT_ARGUMENT_PATTERNS = (
    re.compile(r"\bas i (?:said|mentioned|noted) (?:earlier|before|already)\b", re.I),
    re.compile(r"\bto reiterate\b", re.I),
    re.compile(r"\bi (?:still|must|have to) (?:disagree|insist|push back)\b", re.I),
    re.compile(r"\byou['’]re missing the point\b", re.I),
    re.compile(r"\blet me try this again\b", re.I),
)

_EXECUTION_PATTERNS = (
    re.compile(r"\bi['’]ll (?:edit|update|modify|patch|commit|push|deploy|run|fix)\b", re.I),
    re.compile(r"\bi (?:will|am going to) (?:edit|update|modify|patch|commit|push|deploy|run|fix)\b", re.I),
    re.compile(r"\blet me (?:edit|update|modify|patch|commit|push|deploy|run|fix)\b", re.I),
    re.compile(r"\bi['’]ve (?:edited|updated|modified|patched|committed|pushed|deployed|fixed)\b", re.I),
)

_CARTOGRAPHY_MARKERS = (
    "path a",
    "path b",
    "option a",
    "option b",
    "first path",
    "second path",
    "one option",
    "another option",
    "if you go that way",
    "if you go the other way",
)

# The probe sentinel is the immutable lead-in of RECOVER_GOAL_PROBE.
# We check substring match rather than re-importing the template — the
# template uses .format placeholders that the engine fills before the
# lint runs.
_REAL_GOAL_PROBE_SENTINEL = "before i commit to an answer"


def _check_opener_fluff(text: str) -> LintViolation | None:
    for pat in _OPENER_RESIDUE:
        m = pat.search(text)
        if m:
            return LintViolation(
                rule="NO_OPENER_FLUFF",
                severity=Severity.BLOCK,
                detail="Opener fluff at the start of the response.",
                excerpt=m.group(0),
            )
    return None


def _check_padding(text: str, emotional_relevant: bool) -> LintViolation | None:
    for pat in _PADDING_PATTERNS:
        m = pat.search(text)
        if m:
            if emotional_relevant:
                # The user's emotional state is directly relevant to
                # the decision — naming the emotion once is fine.
                # Multiple naming instances still trigger; check for
                # density rather than first hit.
                hits = sum(1 for p in _PADDING_PATTERNS if p.search(text))
                if hits >= 3:
                    return LintViolation(
                        rule="NO_EMOTIONAL_PADDING",
                        severity=Severity.BLOCK,
                        detail=(
                            "Emotional-naming density too high even "
                            "though the state is relevant. Name it "
                            "once, then move."
                        ),
                        excerpt=m.group(0),
                    )
                return None
            return LintViolation(
                rule="NO_EMOTIONAL_PADDING",
                severity=Severity.BLOCK,
                detail=(
                    "Therapy/emotional padding present and the user's "
                    "state is not directly relevant to the decision."
                ),
                excerpt=m.group(0),
            )
    return None


def _check_failure_mode(text: str) -> LintViolation | None:
    """The response must name where its advice could break.

    Short responses (under ~120 chars) are exempt — there isn't room
    for both the answer and the failure mode in a one-liner, and
    forcing it leads to padded outputs. The threshold is generous on
    purpose: any substantive response carries a failure-mode note."""

    if len(text.strip()) < 120:
        return None
    low = text.lower()
    if any(cue in low for cue in _FAILURE_MODE_CUES):
        return None
    return LintViolation(
        rule="FAILURE_MODE_ATTACHED",
        severity=Severity.BLOCK,
        detail=(
            "No failure mode named. Every recommendation must say "
            "where it breaks or what signal would prove it wrong."
        ),
    )


def _check_repeat_argument(text: str, strike: int) -> LintViolation | None:
    from src.identity.sovereignty import MAP_NOT_MARCH_THRESHOLD

    if strike < MAP_NOT_MARCH_THRESHOLD:
        # Pre-threshold: arguing once is allowed; lint stays quiet.
        return None

    # Post-threshold: the response must be cartography.
    low = text.lower()
    cartography_present = any(m in low for m in _CARTOGRAPHY_MARKERS)
    if not cartography_present:
        return LintViolation(
            rule="NO_REPEATED_ARGUMENT",
            severity=Severity.BLOCK,
            detail=(
                "User has restated this position past threshold; the "
                "response must lay out Path A / Path B / consequences "
                "rather than argue again."
            ),
        )

    # Cartography is present; check we're not also still arguing.
    for pat in _REPEAT_ARGUMENT_PATTERNS:
        m = pat.search(text)
        if m:
            return LintViolation(
                rule="NO_REPEATED_ARGUMENT",
                severity=Severity.BLOCK,
                detail=(
                    "Cartography mode is correctly active but the "
                    "response is also still arguing. Drop the "
                    "argument language."
                ),
                excerpt=m.group(0),
            )
    return None


def _check_sovereignty(text: str) -> LintViolation | None:
    for pat in _EXECUTION_PATTERNS:
        m = pat.search(text)
        if m:
            return LintViolation(
                rule="USER_SOVEREIGNTY_PRESERVED",
                severity=Severity.BLOCK,
                detail=(
                    "First-person action language present. The engine "
                    "surfaces; the user moves."
                ),
                excerpt=m.group(0),
            )
    return None


def _check_real_goal_surfaced(text: str, expected: bool) -> LintViolation | None:
    if not expected:
        return None
    if _REAL_GOAL_PROBE_SENTINEL in text.lower():
        return None
    return LintViolation(
        rule="REAL_GOAL_SURFACED",
        severity=Severity.BLOCK,
        detail=(
            "Goal supremacy flagged a real-goal contradiction, but "
            "the recovery probe is missing from the output."
        ),
    )


def _check_action_execution(text: str) -> LintViolation | None:
    """Telemetry-granular split of the sovereignty check.

    Sovereignty fires on any 'I will / let me' action verb;
    NO_ACTION_EXECUTION_LANGUAGE narrows to verbs that strongly imply
    side-effects on the user's systems (deploy, push, commit, etc.).
    The split lets the operational counters distinguish 'model
    drifted into helpful-assistant mode' (sovereignty WARN) from
    'model promised to deploy something' (BLOCK)."""

    strong = re.compile(
        r"\b(?:i['’]ll|i will|let me|i['’]ve)\s+"
        r"(?:deploy|commit|push|merge|delete|drop|kill|terminate)\b",
        re.I,
    )
    m = strong.search(text)
    if m:
        return LintViolation(
            rule="NO_ACTION_EXECUTION_LANGUAGE",
            severity=Severity.BLOCK,
            detail="Output promises a side-effectful action on the user's systems.",
            excerpt=m.group(0),
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lint(text: str, context: LintContext | None = None) -> LintResult:
    """Run the full lint pass over `text`.

    Returns a LintResult with `passed=True` when no BLOCK-severity
    violations fired. WARN and INFO violations do not flip `passed`."""

    ctx = context or LintContext()
    checks: list[LintViolation | None] = [
        _check_opener_fluff(text),
        _check_padding(text, ctx.emotional_state_relevant),
        _check_failure_mode(text),
        _check_repeat_argument(text, ctx.map_not_march_strike),
        _check_sovereignty(text),
        _check_real_goal_surfaced(text, ctx.real_goal_surfaced),
        _check_action_execution(text),
    ]
    violations = tuple(v for v in checks if v is not None)
    passed = not any(v.severity == Severity.BLOCK for v in violations)
    return LintResult(passed=passed, violations=violations)


def should_regenerate(result: LintResult) -> bool:
    """The caller passes the lint result here. True when at least one
    BLOCK violation fired and the caller should re-run the model with
    a strengthened directive."""
    return bool(result.blocking)


def build_regenerate_directive(result: LintResult) -> str:
    """Compose a short directive to append to the system prompt on
    regeneration. Names the failing rules and their remediation.

    The directive is short on purpose — long re-prompts confuse the
    model. Each blocking violation contributes one or two lines.
    Duplicates across rules are merged."""

    if not result.blocking:
        return ""

    # Map rule names to remediation lines from the anti-laws when the
    # rule corresponds to one. Pure-criterion rules get their own
    # remediation text inlined.
    remediation_map: dict[str, str] = {
        AntiLawKind.NO_EXECUTION.value: get_anti_law(AntiLawKind.NO_EXECUTION).remediation,
        AntiLawKind.NO_ARGUMENT.value:  get_anti_law(AntiLawKind.NO_ARGUMENT).remediation,
        AntiLawKind.NO_PADDING.value:   get_anti_law(AntiLawKind.NO_PADDING).remediation,
    }
    rule_to_remediation: dict[str, str] = {
        "NO_OPENER_FLUFF":              remediation_map[AntiLawKind.NO_PADDING.value],
        "NO_EMOTIONAL_PADDING":         remediation_map[AntiLawKind.NO_PADDING.value],
        "USER_SOVEREIGNTY_PRESERVED":   remediation_map[AntiLawKind.NO_EXECUTION.value],
        "NO_ACTION_EXECUTION_LANGUAGE": remediation_map[AntiLawKind.NO_EXECUTION.value],
        "NO_REPEATED_ARGUMENT":         remediation_map[AntiLawKind.NO_ARGUMENT.value],
        "FAILURE_MODE_ATTACHED": (
            "The recommendation must name where it breaks or what "
            "signal would prove it wrong. Add a single short clause "
            "on the failure mode."
        ),
        "REAL_GOAL_SURFACED": (
            "The engine has flagged that the stated goal contradicts "
            "other signals. Open the response with the recovery "
            "probe before any analysis."
        ),
    }

    seen_remediations: list[str] = []
    for v in result.blocking:
        rem = rule_to_remediation.get(v.rule, "")
        if rem and rem not in seen_remediations:
            seen_remediations.append(rem)

    header = "Your previous draft violated identity rules. Regenerate with:"
    body = "\n".join(f"- {r}" for r in seen_remediations)
    return f"{header}\n{body}"
