"""
Triage gate — the front-door classifier for the reasoningEngine.

Every request hits this first. It decides:
    - Which route to take (TRIVIAL / DIRECT / DIRECT_PLUS / DEEP)
    - How deep to go if DEEP (LOW / MEDIUM / HIGH)
    - Which external capabilities (MCPs) would help this request
    - Whether the user is stating intent to do something irreversible
      (interrupt flag — surfaced as urgency in the response)
    - Risk flags for downstream prompts
    - A one-sentence "why" the user sees

Two modes:
    LIVE — Single Gemini Flash-Lite call via OpenRouter (~$0.0001, ~500ms)
    MOCK — Deterministic keyword/phrase classifier (no API, no cost)

The gate does NOT execute the request. It only classifies. The route
dispatcher (server.py) consumes its output and decides the actual flow.

ISOLATION: Imports from src.llm.client and src.llm.effort only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from src.llm.client import ClientMode, LLMClient
from src.llm.effort import Effort, normalize_effort


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """The four classification routes the gate can emit."""
    TRIVIAL = "trivial"            # greeting / ack / clarification — canned response
    DIRECT = "direct"              # single LLM call, no retrieval
    DIRECT_PLUS = "direct_plus"    # single LLM call + bridge retrieval (memory/graphify)
    DEEP = "deep"                  # full wuxing fan-out at the requested effort tier


@dataclass
class MCPNeed:
    """A single external capability the gate identified as useful for this request."""
    name: str                      # "web_search" | "github" | "memory_v2" | "graphify" | "browser" | "docs"
    why: str                       # one-sentence explanation for the user
    required: bool = False         # True if the engine genuinely cannot give a meaningful answer without it


@dataclass
class TriageResult:
    """Structured output from the triage gate."""
    route: Route
    recommended_effort: Effort
    mcps_needed: list[MCPNeed] = field(default_factory=list)
    interrupt: bool = False
    risk_flags: list[str] = field(default_factory=list)
    why: str = ""
    raw_response: str = ""         # only populated in live mode (for debug)
    classifier_mode: str = ""      # "mock" | "live" | "live_failed_fallback_mock" | "live_unparseable_fallback_mock"


# ---------------------------------------------------------------------------
# Mock classifier — deterministic, no API call
# ---------------------------------------------------------------------------

# Explicit triggers — user is forcing depth.
_EXPLICIT_DEEP_TRIGGERS: tuple[str, ...] = (
    "go deeper", "think harder", "full picture", "extend on this",
    "spend time on this", "open this up", "war room this",
    "pressure-test this", "stress-test this", "debate this",
    "council this", "run the council",
)

# Semantic triggers — phrases that reliably mean "this is a real decision".
_SEMANTIC_DEEP_TRIGGERS: tuple[str, ...] = (
    "should i ", "i'm torn", "i am torn", "torn between",
    "i can't decide", "i cannot decide", "can't tell which",
    "which option", "which way", "what would you do",
    "is this the right move", "validate this",
    "get multiple perspectives", "help me think through",
    "help me think this through", "talk me through",
    "i keep going back and forth", "whether to ",
    "thinking about whether",
)

# Memory / project-context queries — DIRECT_PLUS.
_MEMORY_TRIGGERS: tuple[str, ...] = (
    "what did we decide", "what was decided", "where is the",
    "where's the", "summarize this repo", "summarize the repo",
    "summarize this area", "continue from yesterday",
    "continue from where", "earlier decisions", "last session",
    "prior decisions", "in this repo", "what does this codebase",
)

# Greetings / acks / clarifications — TRIVIAL.
_TRIVIAL_PATTERNS: tuple[str, ...] = (
    r"^(hi|hello|hey|sup|yo)[!.\s]*$",
    r"^(thanks|thank you|ty|thx)[!.\s]*$",
    r"^(ok|okay|got it|cool|fine|sure)[!.\s]*$",
    r"^(yes|no|yeah|nah|yep|nope)[!.\s]*$",
    r"^(bye|goodbye|see ya|later|cya)[!.\s]*$",
    r"^(are you there\??|you there\??|hello\?)\s*$",
)

# Risk-flag patterns — surface red flags for downstream prompts.
_IRREVERSIBLE_PATTERNS: tuple[str, ...] = (
    "delete ", "drop table", "rm -rf", "force push", "force-push",
    "push to prod", "deploy ", "send email", "send the email",
    "merge to main", "remove the",
)
_ARCHITECTURE_PATTERNS: tuple[str, ...] = (
    "refactor", "rewrite", "redesign", "migrate", "architecture",
    "schema change", "breaking change",
)
_LONG_TERM_PATTERNS: tuple[str, ...] = (
    "career", "marry", "marriage", "founder", "quit my job",
    "leave my job", "move to", "investor", "raise money",
    "start a company", "start a business",
)


def _is_trivial(text: str) -> bool:
    """Short greetings, acks, single-word replies."""
    normalized = text.strip().lower()
    if not normalized:
        return True
    if len(normalized) < 4 and not normalized.endswith("?"):
        return True
    for pat in _TRIVIAL_PATTERNS:
        if re.match(pat, normalized):
            return True
    return False


def _has_any(text_lower: str, phrases: tuple[str, ...]) -> bool:
    return any(p in text_lower for p in phrases)


def _mock_classify(text: str) -> TriageResult:
    """Deterministic keyword/phrase classifier — used in tests and as a fallback."""
    if _is_trivial(text):
        return TriageResult(
            route=Route.TRIVIAL,
            recommended_effort=Effort.LOW,
            why="Short greeting or acknowledgment.",
            classifier_mode="mock",
        )

    text_lower = text.lower()

    explicit_deep = _has_any(text_lower, _EXPLICIT_DEEP_TRIGGERS)
    semantic_deep = _has_any(text_lower, _SEMANTIC_DEEP_TRIGGERS)
    memory_query = _has_any(text_lower, _MEMORY_TRIGGERS)
    irreversible = _has_any(text_lower, _IRREVERSIBLE_PATTERNS)
    architecture = _has_any(text_lower, _ARCHITECTURE_PATTERNS)
    long_term = _has_any(text_lower, _LONG_TERM_PATTERNS)

    risk_flags: list[str] = []
    if irreversible:
        risk_flags.append("irreversible_action")
    if architecture:
        risk_flags.append("architecture_decision")
    if long_term:
        risk_flags.append("long_term_consequence")

    # Interrupt: short statement of intent to do something irreversible,
    # not a long deliberation about it.
    word_count = len(text.split())
    interrupt = irreversible and word_count < 15

    # DEEP route — explicit or semantic depth triggers, architecture-impacting,
    # or long-term-consequence decisions.
    if explicit_deep or semantic_deep or architecture or long_term:
        if long_term or (irreversible and word_count > 15):
            effort = Effort.HIGH
            why = "High-stakes or long-term decision; engaging deep analysis."
        elif architecture or explicit_deep:
            effort = Effort.MEDIUM
            why = "Multi-path decision with meaningful tradeoffs."
        else:
            effort = Effort.MEDIUM
            why = "Decision with competing forces; medium depth fits."
        return TriageResult(
            route=Route.DEEP,
            recommended_effort=effort,
            mcps_needed=[],
            interrupt=interrupt,
            risk_flags=risk_flags,
            why=why,
            classifier_mode="mock",
        )

    # DIRECT_PLUS — memory or project-context query.
    if memory_query:
        return TriageResult(
            route=Route.DIRECT_PLUS,
            recommended_effort=Effort.LOW,
            mcps_needed=[
                MCPNeed(
                    name="memory_v2",
                    why="Question references prior decisions or project context.",
                    required=False,
                ),
            ],
            risk_flags=risk_flags,
            why="Memory or project-context query.",
            classifier_mode="mock",
        )

    # DIRECT — factual / single-answer / generic.
    return TriageResult(
        route=Route.DIRECT,
        recommended_effort=Effort.LOW,
        interrupt=interrupt,
        risk_flags=risk_flags,
        why="Single-answer or factual question." if not interrupt
             else "Action statement; surfacing considerations before acting.",
        classifier_mode="mock",
    )


# ---------------------------------------------------------------------------
# Live classifier — single Gemini Flash-Lite call via OpenRouter
# ---------------------------------------------------------------------------

_GATE_SYSTEM_PROMPT = """You are the triage gate for a thinking-partner reasoning engine.

Your only job: classify the user's message into one of four routes, estimate
the depth needed, identify any external capabilities (MCPs) that would help,
and surface risk flags.

ROUTES:
- "trivial": greetings, acks, clarifications, tests ("hi", "thanks", "ok", "yes").
- "direct": factual lookup, definitions, single-right-answer tasks.
- "direct_plus": single-model answer that benefits from memory/project-graph
                 context ("what did we decide", "where is X", "summarize this area").
- "deep": multi-path decision, real tradeoffs, anchoring risk, architecture
          impact, irreversible or long-term consequence, OR explicit user request
          for depth ("go deeper", "should I X or Y", "I'm torn", "what would you do").

EFFORT (only meaningful for "deep"; use "low" for trivial/direct/direct_plus):
- "low":    3 iterations  — one clear tradeoff
- "medium": 6 iterations  — 2-3 competing forces (default for deep)
- "high":   10 iterations — irreversible, life-direction, architecture-impacting

RISK FLAGS (zero or more, only include those that apply):
- "irreversible_action"
- "architecture_decision"
- "long_term_consequence"
- "user_overconfident"        (user is leaning hard one way; anchoring risk)
- "missing_context"           (engine needs info it can't infer)
- "high_emotional_stakes"

INTERRUPT FLAG: set true ONLY if the user is stating intent to do something
irreversible (delete, deploy, push to prod, send email) AND is asking advice
BEFORE acting. If they're already past the point of advice, set false.

MCPS_NEEDED: list external capabilities that would sharpen the answer:
- "web_search"  (current events / library versions / docs after 2025)
- "github"      (PRs, issues, recent commits)
- "browser"     (UI behavior verification)
- "memory_v2"   (prior decisions in this project)
- "graphify"    (code structure of this repo)
- "docs"        (specific library or API documentation)
For each MCP, give a one-sentence "why" and set "required": true ONLY if the
engine cannot give a meaningful answer without it.

OUTPUT — strict JSON only, no prose, no markdown fences:
{
  "route": "trivial|direct|direct_plus|deep",
  "recommended_effort": "low|medium|high",
  "mcps_needed": [{"name": "...", "why": "...", "required": false}],
  "interrupt": false,
  "risk_flags": ["..."],
  "why": "one short sentence the user will see"
}

ROUTING PRINCIPLE: bias toward depth when ambiguous, irreversible,
architecture-impacting, or showing user overconfidence. Bias toward shallow
ONLY when the question has one obvious correct answer.
"""


async def _live_classify(client: LLMClient, text: str) -> TriageResult:
    """Single Gemini Flash-Lite call via provider_map (domain='gating')."""
    response = await client.call(
        system_prompt=_GATE_SYSTEM_PROMPT,
        user_message=text,
        domain="gating",
        concept="triage",
        max_tokens=512,
        temperature=0.1,
    )

    if not response.success:
        # Fail-soft: degrade to mock classifier with explicit mode tag.
        result = _mock_classify(text)
        result.classifier_mode = "live_failed_fallback_mock"
        result.why = (
            f"Live classifier failed ({response.error or 'unknown'}); "
            "used keyword fallback."
        )
        return result

    return _parse_triage_json(response.content, text)


def _parse_triage_json(raw: str, original_text: str) -> TriageResult:
    """Parse the gate's JSON output. Fail-soft to mock on any error."""
    cleaned = raw.strip()
    # Strip code fences if the model adds them despite instructions.
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE
        ).strip()

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        result = _mock_classify(original_text)
        result.classifier_mode = "live_unparseable_fallback_mock"
        result.raw_response = raw
        return result

    if not isinstance(data, dict):
        result = _mock_classify(original_text)
        result.classifier_mode = "live_unparseable_fallback_mock"
        result.raw_response = raw
        return result

    # Route — fall back to DEEP (bias-toward-depth) on unknown value.
    try:
        route = Route(str(data.get("route", "deep")).strip().lower())
    except ValueError:
        route = Route.DEEP

    effort = normalize_effort(data.get("recommended_effort", "medium"))

    mcps: list[MCPNeed] = []
    for m in data.get("mcps_needed", []) or []:
        if isinstance(m, dict) and "name" in m:
            mcps.append(MCPNeed(
                name=str(m["name"]).strip(),
                why=str(m.get("why", "")).strip(),
                required=bool(m.get("required", False)),
            ))

    risk_flags = [
        str(f).strip() for f in (data.get("risk_flags") or [])
        if isinstance(f, str) and f.strip()
    ]

    return TriageResult(
        route=route,
        recommended_effort=effort,
        mcps_needed=mcps,
        interrupt=bool(data.get("interrupt", False)),
        risk_flags=risk_flags,
        why=str(data.get("why", "")).strip(),
        raw_response=raw,
        classifier_mode="live",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def triage(
    text: str,
    client: LLMClient | None = None,
    force_mock: bool = False,
) -> TriageResult:
    """
    Classify a user message.

    If `client` is None, `force_mock` is True, or the client is in MOCK mode,
    runs the deterministic keyword classifier. Otherwise dispatches a single
    Gemini Flash-Lite call via OpenRouter.

    Always returns a valid TriageResult — never raises on classification
    failure, always falls back to the mock classifier with an explicit
    `classifier_mode` tag.
    """
    if not text or not text.strip():
        return TriageResult(
            route=Route.TRIVIAL,
            recommended_effort=Effort.LOW,
            why="Empty message.",
            classifier_mode="mock",
        )

    if force_mock or client is None or client.mode == ClientMode.MOCK:
        return _mock_classify(text)

    return await _live_classify(client, text)
