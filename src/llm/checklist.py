"""
The 27 universal questions — angular thinking discipline as prompt scaffold.

These are the questions the engine must answer (implicitly, in its reasoning)
before delivering any response. They are NOT enumerated in the output — they
shape the answer, not its format. The user sees a direct, brain-extension
response; the engine has worked through these internally.

Six categories:
    A. UNDERSTANDING (A1-A4)  — what is the user actually asking
    B. LANDSCAPE    (B5-B8)   — the full path-space before committing
    C. EVALUATING   (C9-C13)  — why this answer, what assumptions
    D. CONSEQUENCE  (D14-D19) — first/second/third-order, opportunity cost
    E. ADVERSARIAL  (E20-E24) — steel-man, anti-validation, anti-hedge
    F. RELEVANCE    (F25-F27) — staying on the user's question

Distribution per route+effort (locked design):
    TRIVIAL              none
    DIRECT               F25
    DIRECT_PLUS          A1 + F25-F27
    DEEP / LOW           A1-A4 + D14 + E20            (6)
    DEEP / MEDIUM        A + B + D + E22-E24 + F25    (18)
    DEEP / HIGH or AUTO  all 27

Cost-wise this adds 100-1100 prompt tokens depending on tier — meaningful
but proportionate to the depth the user authorized.

ISOLATION: imports only from src.llm.effort and src.llm.triage (for the
Route enum). No engine, no bridge.
"""

from __future__ import annotations

from src.llm.effort import Effort
from src.llm.triage import Route


# ---------------------------------------------------------------------------
# The 27 questions, keyed by code (A1, B5, etc.)
# ---------------------------------------------------------------------------

QUESTIONS: dict[str, str] = {
    # A. UNDERSTANDING — surface vs real question
    "A1": (
        "What is the user actually asking? The surface text is rarely the "
        "real question. Name the real question explicitly in your own head."
    ),
    "A2": (
        "What triggered this question NOW? A specific incident, a deadline, "
        "a stuck loop? The trigger is signal — it tells you what the question "
        "is really about."
    ),
    "A3": (
        "What is the user implicitly asking that they're NOT saying? "
        "Permission to leave. Validation to stay. A way out of a decision "
        "they've already made. The unsaid request is usually the real one."
    ),
    "A4": (
        "Is this grounded or speculative? Tied to a concrete situation or "
        "abstract/hypothetical? Treat them differently — say so plainly if "
        "speculative."
    ),

    # B. LANDSCAPE — full path space before commitment
    "B5": (
        "What is the full set of paths forward — including ones the user "
        "hasn't considered? List obvious AND unobvious. Don't pick yet — "
        "premature commitment is failure mode #1."
    ),
    "B6": (
        "What constraints actually bind here? Money, time, identity, sunk "
        "cost, social proof. Which are real, which are imagined? Imagined "
        "constraints are often the most decisive and the most dissolvable."
    ),
    "B7": (
        "What angle has the user already worked through? Don't restate their "
        "own thinking back at them — build on it or break it."
    ),
    "B8": (
        "What angle have they not even seen? The thing they can't see "
        "because of their position, bias, or information set. This is "
        "where the engine earns its keep."
    ),

    # C. EVALUATING — why this answer, not another
    "C9": (
        "Why this answer rather than the alternative? Force a comparison. "
        "If you can't say why-not-X for at least one X, the answer is "
        "under-tested."
    ),
    "C10": (
        "What concrete advantages does this path have? Specifically what "
        "improves and by when. Not 'personal growth' — testable outcomes."
    ),
    "C11": (
        "What concrete disadvantages does this path have? Specifically what "
        "gets worse and by when. The honest answer names the cost of its "
        "own recommendation."
    ),
    "C12": (
        "What assumptions does this answer rely on? Surface them. If any "
        "are shaky, the answer is shaky — and the user deserves to know."
    ),
    "C13": (
        "What would have to be true for this answer to be WRONG? The "
        "failure condition. If you can't name it, you don't understand "
        "the answer."
    ),

    # D. CONSEQUENCE — first/second/third-order
    "D14": (
        "First-order consequence — in 30 days, what happens if the user "
        "follows this? Concrete, time-bound, testable."
    ),
    "D15": (
        "Second-order consequence — in 6 months, what does that lead to? "
        "Compounding effects, relationship shifts, position changes."
    ),
    "D16": (
        "Third-order consequence — in 2 years, what's the downstream shape? "
        "Most advice falls apart here — optimize for 30 days, pay in 2 years."
    ),
    "D17": (
        "What is the cost of NOT acting on this answer? Inaction is a "
        "choice. Price it. The asymmetry usually decides the call."
    ),
    "D18": (
        "What positive prospects does this open up? What new doors? "
        "What becomes possible that wasn't before?"
    ),
    "D19": (
        "What does this foreclose? Every yes is a no to something else. "
        "Name what dies. Show the user the closing doors, not just the "
        "opening ones."
    ),

    # E. ADVERSARIAL — steel-man, anti-validation
    "E20": (
        "What is the strongest argument AGAINST this answer? Steel-man the "
        "opposition. If it survives, the answer is real. If not, the user "
        "never sees it."
    ),
    "E21": (
        "Is this generic or specific to THIS user? If you could give the "
        "same answer to anyone, you've failed. Find the user-specific "
        "anchor."
    ),
    "E22": (
        "Am I validating? Did the answer just confirm what the user already "
        "wanted? That's a red flag, not a success signal. Fight the "
        "chatbot's gravity well."
    ),
    "E23": (
        "Am I hedging? 'Things might', 'It depends', 'On the other hand' — "
        "escape hatches. Pick a side. If you genuinely can't, say so and "
        "name the missing information."
    ),
    "E24": (
        "What will the user push back on first? Pre-empt it. The strongest "
        "delivery anticipates the strongest objection."
    ),

    # F. RELEVANCE — staying on the user's question
    "F25": (
        "Is this answer relevant to the user's actual goal — or to a goal "
        "I find more interesting? The engine has its own gravity. Resist it."
    ),
    "F26": (
        "Is this advice actionable in the user's current context? Generic "
        "advice that needs resources they don't have is decorative, not "
        "useful."
    ),
    "F27": (
        "Am I solving the question they asked, or a different one I prefer? "
        "If yes to the second, course-correct."
    ),
}

# Sanity check at import time.
assert len(QUESTIONS) == 27, f"expected 27 questions, got {len(QUESTIONS)}"


# ---------------------------------------------------------------------------
# Distribution per route + effort
# ---------------------------------------------------------------------------

_TRIVIAL_CODES: list[str] = []
_DIRECT_CODES: list[str] = ["F25"]
_DIRECT_PLUS_CODES: list[str] = ["A1", "F25", "F26", "F27"]

_DEEP_LOW_CODES: list[str] = ["A1", "A2", "A3", "A4", "D14", "E20"]

_DEEP_MEDIUM_CODES: list[str] = [
    # A category
    "A1", "A2", "A3", "A4",
    # B category
    "B5", "B6", "B7", "B8",
    # D category (full)
    "D14", "D15", "D16", "D17", "D18", "D19",
    # E category (later three — adversarial voice)
    "E22", "E23", "E24",
    # F category (relevance check)
    "F25",
]

# HIGH and AUTO fire all 27, in stable order.
_DEEP_HIGH_CODES: list[str] = list(QUESTIONS.keys())


def get_checklist_codes(route: Route, effort: Effort) -> list[str]:
    """
    Return the question codes that should fire for a given (route, effort).

    TRIVIAL → []
    DIRECT → ["F25"]
    DIRECT_PLUS → ["A1", "F25", "F26", "F27"]
    DEEP / LOW → 6 codes
    DEEP / MEDIUM → 18 codes
    DEEP / HIGH or AUTO → all 27
    """
    if route == Route.TRIVIAL:
        return list(_TRIVIAL_CODES)
    if route == Route.DIRECT:
        return list(_DIRECT_CODES)
    if route == Route.DIRECT_PLUS:
        return list(_DIRECT_PLUS_CODES)

    # DEEP
    if effort == Effort.LOW:
        return list(_DEEP_LOW_CODES)
    if effort == Effort.MEDIUM:
        return list(_DEEP_MEDIUM_CODES)
    # HIGH and AUTO
    return list(_DEEP_HIGH_CODES)


# ---------------------------------------------------------------------------
# Prompt fragment builder
# ---------------------------------------------------------------------------

_CHECKLIST_HEADER = (
    "## ANGULAR CHECKLIST\n"
    "\n"
    "Answer these questions IMPLICITLY in your reasoning. Do NOT enumerate "
    "them in your output. Do NOT label sections by question code. Let your "
    "response show the work — directly, in your own voice. Skip any question "
    "that genuinely doesn't apply to this specific user message. If multiple "
    "questions converge on the same answer, surface it once with conviction."
    "\n"
)


def build_checklist_block(route: Route, effort: Effort) -> str:
    """
    Build the prompt fragment to inject into a system prompt for this route+effort.

    Returns an empty string for TRIVIAL (no checklist) so callers can
    unconditionally concatenate without worrying about empty headers.
    """
    codes = get_checklist_codes(route, effort)
    if not codes:
        return ""

    lines = [_CHECKLIST_HEADER]
    for code in codes:
        question = QUESTIONS.get(code, "")
        if question:
            lines.append(f"{code}. {question}")
    lines.append("")  # trailing newline
    return "\n".join(lines)
