"""
Articulation layer — turn raw ExplorationReports into user-readable cards.

Per the plan, articulation is its own layer DISTINCT from synthesis. Its
job: render each individual report into the six-field format the user
actually reads.

  Spark        — what was noticed (the surprising connection)
  Source Shape — what structure it came from (the source's underlying form)
  Bridge       — how it maps to the user's problem
  Use          — what the user can do with it (a concrete suggestion)
  Limit        — where the analogy breaks
  Confidence   — how strongly to treat it

Per Law 6: honest doubt over performative confidence. The "Limit" field
is rendered directly from `what_does_not_map` and gets equal weight as
"Bridge" — the dossier visually balances them.

Articulation uses Sonnet because PROSE QUALITY matters here — this is
user-facing text, the bridge between raw signal and human comprehension.
We accept the higher per-card cost.

ISOLATION: imports report types + LLM client. No persistence, no
synthesis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from src.llm.client import LLMClient, LLMResponse
from src.wandering.report import Confidence, ExplorationReport


log = logging.getLogger("constellax.wandering.articulate")


#: Articulation uses Sonnet 4.6 (via synthesizer key) — user-facing prose.
ARTICULATE_DOMAIN = "synthesizer"
ARTICULATE_CONCEPT = "wandering_articulation"


# ---------------------------------------------------------------------------
# Articulated card — the user-facing rendering of one report
# ---------------------------------------------------------------------------


@dataclass
class ArticulatedCard:
    """One report, rendered for the user.

    All six fields are strings. None is optional — if articulation can't
    produce a field, we render "(unable to articulate)" rather than
    silently dropping it. This is a transparency choice: the user sees
    that something was attempted, not that nothing exists.
    """

    report_id: str
    spark: str
    source_shape: str
    bridge: str
    use: str
    limit: str
    confidence: Confidence
    confidence_detail: str = ""  # ratio summary like "act:0/4 ess:4/5 mec:5/5"

    def to_dict(self) -> dict[str, str]:
        return {
            "report_id": self.report_id,
            "spark": self.spark,
            "source_shape": self.source_shape,
            "bridge": self.bridge,
            "use": self.use,
            "limit": self.limit,
            "confidence": self.confidence.value,
            "confidence_detail": self.confidence_detail,
        }


# ---------------------------------------------------------------------------
# Articulation prompt
# ---------------------------------------------------------------------------


_ARTICULATE_SYSTEM_PROMPT = """\
You are Constellax's articulation layer for Wandering Room.

You receive ONE raw ExplorationReport (from a wandering agent that found
structural resonance between some external source and the user's anchor).

Your job: render it into the SIX-FIELD card the user actually reads.

# THE SIX FIELDS

  Spark        — ONE sentence describing the surprising connection the
                 agent noticed. Be specific. ("Jazz improvisation operates
                 inside harmonic constraint.")

  Source Shape — ONE-TO-TWO sentences describing the structure the source
                 has, in the source's own terms. ("Soloists choose notes
                 freely within the chord changes; the changes constrain
                 the freedom and make musical sense possible.")

  Bridge       — ONE-TO-TWO sentences mapping the source structure to the
                 user's problem. ("Your wandering agents face the same
                 shape — they need bounded freedom that constrains
                 without dictating.")

  Use          — ONE specific, concrete thing the user might do with this
                 insight. ("Consider designing the agent's drift radius
                 as a 'harmonic frame' — not a cage.")

  Limit        — Render the report's what_does_not_map field directly.
                 This is the honest brake on the analogy. DO NOT soften
                 it. ("Jazz operates in real-time emergent ensemble; your
                 agents run async in parallel — coordination
                 patterns differ.")

  Confidence   — Use the confidence value from the report (LOW/MEDIUM/HIGH).
                 Don't override it.

# CRITICAL

1. The insight happens in the USER's head. You are RENDERING a bridge,
   not delivering a conclusion. Phrase the "Use" field as a possibility
   the user might pursue, not a solution they should adopt.

2. The "Limit" field is LOAD-BEARING. The user needs to know where the
   analogy breaks to trust the bridge. Honest break > perfect-looking bridge.

3. Plain language. No academic hedging. No filler ("Interestingly,..."
   "It's worth noting that...").

# OUTPUT FORMAT

Return ONE JSON object:

{
  "spark": "<one sentence>",
  "source_shape": "<one-to-two sentences>",
  "bridge": "<one-to-two sentences>",
  "use": "<one sentence — concrete possibility>",
  "limit": "<one-to-two sentences from what_does_not_map>"
}

No prose preamble. No code fences. Just JSON.
"""


def build_articulation_user_message(report: ExplorationReport) -> str:
    """Render the raw report into the user-message payload for Sonnet."""
    blocks = [
        f"# REPORT {report.report_id}",
        f"Agent: {report.agent_id}",
        f"Anchor: {report.anchor_summary}",
        f"Domain explored: {report.domain_explored}",
        f"Confidence: {report.confidence.value}",
        f"Match summary: {report.match_ratio_summary()}",
    ]

    if report.source_locations:
        blocks.append("\nSources:")
        for src in report.source_locations:
            line = f"  - {src.title}"
            if src.url:
                line += f" ({src.url})"
            blocks.append(line)
            if src.excerpt:
                blocks.append(f"    Excerpt: {src.excerpt[:200]}")

    blocks.append("\nMatched layer nodes:")
    for layer_name, lm in report.layer_matches.items():
        if lm.matched_nodes:
            blocks.append(f"  {layer_name}: {', '.join(lm.matched_nodes)}")

    blocks.append("\n# AGENT'S RAW WRITE-UP")
    blocks.append(f"exploration_summary:\n{report.exploration_summary}")
    blocks.append(f"\nadvancement:\n{report.advancement}")
    blocks.append(f"\nwhat_does_not_map (LOAD-BEARING):\n{report.what_does_not_map}")
    if report.next_lead:
        blocks.append(f"\nnext_lead:\n{report.next_lead}")

    blocks.append("\n# YOUR TASK")
    blocks.append(
        "Render this raw report into the six-field card. Return JSON per the spec. "
        "The 'limit' field MUST faithfully convey what_does_not_map."
    )

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def _extract_json_object(text: str) -> str:
    text = _strip_code_fences(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in articulation response")
    depth = 0
    in_str = False
    escape = False
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
                return text[start : i + 1]
    raise ValueError("unterminated JSON in articulation response")


def parse_articulation_response(response_text: str) -> dict[str, str]:
    """Parse the JSON. Returns dict with the five string fields. Missing
    fields default to '(unable to articulate)' — transparent fallback."""
    try:
        json_text = _extract_json_object(response_text)
        payload = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("articulation response unparseable: %s", e)
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    def _get(key: str) -> str:
        val = payload.get(key, "")
        s = str(val).strip()
        return s if s else "(unable to articulate)"

    return {
        "spark": _get("spark"),
        "source_shape": _get("source_shape"),
        "bridge": _get("bridge"),
        "use": _get("use"),
        "limit": _get("limit"),
    }


# ---------------------------------------------------------------------------
# Top-level articulation function
# ---------------------------------------------------------------------------


async def articulate_report(
    report: ExplorationReport,
    client: LLMClient,
) -> ArticulatedCard:
    """Render one ExplorationReport as a user-readable ArticulatedCard.

    Defensive: if the LLM call fails entirely, we still produce a card
    using the report's raw fields as fallback content. The user sees
    something — even if it's slightly raw — rather than a hole.
    """
    user_message = build_articulation_user_message(report)
    response: LLMResponse = await client.call(
        system_prompt=_ARTICULATE_SYSTEM_PROMPT,
        user_message=user_message,
        domain=ARTICULATE_DOMAIN,
        concept=ARTICULATE_CONCEPT,
    )

    if not response.success:
        log.warning("articulation LLM call failed: %s", response.error)
        # Fallback: use the report's raw fields directly.
        return ArticulatedCard(
            report_id=report.report_id,
            spark=report.exploration_summary or "(unable to articulate)",
            source_shape=(
                report.source_locations[0].title
                if report.source_locations
                else "(unable to articulate)"
            ),
            bridge=report.advancement or "(unable to articulate)",
            use=report.next_lead or "(unable to articulate)",
            limit=report.what_does_not_map or "(unable to articulate)",
            confidence=report.confidence,
            confidence_detail=report.match_ratio_summary(),
        )

    fields = parse_articulation_response(response.content)

    return ArticulatedCard(
        report_id=report.report_id,
        spark=fields["spark"],
        source_shape=fields["source_shape"],
        bridge=fields["bridge"],
        use=fields["use"],
        limit=fields["limit"],
        confidence=report.confidence,
        confidence_detail=report.match_ratio_summary(),
    )


__all__ = [
    "ARTICULATE_DOMAIN",
    "ARTICULATE_CONCEPT",
    "ArticulatedCard",
    "build_articulation_user_message",
    "parse_articulation_response",
    "articulate_report",
]
