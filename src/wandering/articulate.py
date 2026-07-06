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
from dataclasses import dataclass, field

from src.identity import compose_system_prompt
from src.identity.disciplines.goal_supremacy import ServeScore
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

    The six articulated text fields are strings. None is optional — if
    articulation can't produce a field, we render "(unable to articulate)"
    rather than silently dropping it. This is a transparency choice: the
    user sees that something was attempted, not that nothing exists.

    The provenance fields (agent_id, domain, citations, match_strength)
    pass through from the raw ExplorationReport unchanged — they're not
    re-derived or "improved" by articulation. The frontend uses them
    for the card header, the cited-sources block, and the match bar.
    """

    report_id: str
    spark: str
    source_shape: str
    bridge: str
    use: str
    limit: str
    confidence: Confidence
    confidence_detail: str = ""  # ratio summary like "act:0/4 ess:4/5 mec:5/5"

    # Provenance — copied verbatim from the source ExplorationReport.
    agent_id: str = ""
    domain: str = ""
    citations: list[dict] = field(default_factory=list)
    match_strength: float = 0.0  # 0.0–1.0, total_matched / total_cushion

    # Identity-layer metadata (additive — does not gate band placement
    # or filter the card). Populated by `goal_supremacy.discriminate()`
    # in `build_dossier` against the cushion's real goal. The frontend
    # can render this as a "this card serves your real goal" badge or
    # a "serves stated goal but not real goal" warning. Engine logic
    # does not branch on it. See doctrine §10 "discipline metadata".
    serve_score: ServeScore | None = None

    def to_dict(self) -> dict:
        sscore: dict | None = None
        if self.serve_score is not None:
            sscore = {
                "score":                  float(self.serve_score.score),
                "verdict":                str(self.serve_score.verdict),
                "reasons":                list(self.serve_score.reasons),
                "serves_attachment_only": bool(self.serve_score.serves_attachment_only),
            }
        return {
            "report_id": self.report_id,
            "agent_id": self.agent_id,
            "domain": self.domain,
            "spark": self.spark,
            "source_shape": self.source_shape,
            "bridge": self.bridge,
            "use": self.use,
            "limit": self.limit,
            "confidence": self.confidence.value,
            "confidence_detail": self.confidence_detail,
            "citations": list(self.citations),
            "match_strength": float(self.match_strength),
            "serve_score": sscore,
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

  Be EXPRESSIVE. The reader needs to fully visualize the bridge — a card
  that ends before the mapping is clear is useless. Explain the actual
  mechanism, not just assert that a resemblance exists. Length targets
  below are minimums for clarity, not caps; never pad, but never cut the
  explanation short either.

  Spark        — ONE-TO-TWO sentences naming the surprising connection the
                 agent noticed. Be specific and vivid. ("Jazz improvisation
                 produces unpredictable melodies, yet every note is chosen
                 inside the song's harmonic constraints — freedom and
                 structure are not opposites here, they are the same
                 mechanism.")

  Source Shape — TWO-TO-THREE sentences describing the structure the source
                 has, in the source's own terms. Explain HOW the source
                 mechanism actually works, enough that someone unfamiliar
                 with the domain understands it. ("In jazz, soloists choose
                 notes freely, but only notes that fit the underlying chord
                 changes sound musical. The chord progression is a moving
                 constraint — it never tells the soloist which note to play,
                 it only tells them which notes are available right now. The
                 result is improvisation that is genuinely free yet never
                 random.")

  Bridge       — THREE-TO-FOUR sentences mapping the source structure to the
                 user's problem, STEP BY STEP. This is the most important
                 field. Walk the reader through the mapping: what in the
                 source corresponds to what in their problem, and why the
                 correspondence holds. Do not merely assert "your problem is
                 the same shape" — show the reader EACH point of contact so
                 they can see the bridge, not just be told it exists. ("Your
                 wandering agents face this exact shape. The user's anchor is
                 the chord progression — a constraint that bounds where the
                 agents can go without dictating the path. The agents'
                 traversal is the solo: free to be surprising, but only
                 'musical' when it stays in harmonic relationship with the
                 anchor. And just as a soloist who ignores the changes
                 produces noise, an agent that drifts past the anchor's pull
                 produces irrelevance — the constraint is what makes the
                 freedom productive rather than random.")

  Use          — ONE-TO-TWO sentences naming a specific, concrete thing the
                 user might do with this insight. Make it actionable enough
                 that they could start tomorrow. ("Consider designing the
                 agent's drift radius as a 'harmonic frame' rather than a
                 hard cage — a soft, always-present pull toward the anchor
                 that bounds the wander without ever forbidding a direction.")

  Limit        — TWO-TO-THREE sentences rendering the report's
                 what_does_not_map field. This is the honest brake on the
                 analogy. DO NOT soften it — explain clearly where the
                 mapping breaks and why, so the reader knows exactly how far
                 to trust the bridge. ("Jazz operates as real-time emergent
                 ensemble: musicians hear each other and adjust in the
                 moment. Your agents run async in parallel with no live
                 mutual listening, so the coordination that makes a jazz
                 group cohere has no direct analog here. The bridge holds for
                 the freedom-within-constraint shape, but breaks at the
                 question of how independent voices stay coherent.")

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
  "spark": "<one-to-two sentences>",
  "source_shape": "<two-to-three sentences explaining the source mechanism>",
  "bridge": "<three-to-four sentences walking the mapping step by step>",
  "use": "<one-to-two sentences — concrete, actionable possibility>",
  "limit": "<two-to-three sentences from what_does_not_map>"
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


def _citations_from_report(report: ExplorationReport) -> list[dict]:
    """Project the report's source_locations into the dict shape the
    frontend's card citation block reads. Pass-through only — we do not
    filter or re-rank sources here."""
    out: list[dict] = []
    for src in report.source_locations:
        out.append({
            "title": src.title,
            "url": src.url,
            "snippet": src.excerpt,
        })
    return out


def _match_strength_from_report(report: ExplorationReport) -> float:
    """Single 0.0–1.0 number for the card's match bar.

    Uses the report's existing aggregate (total matched / total cushion).
    No new weighting scheme — this is the same ratio compute_confidence
    already reads. The bar is the visual surface of that ratio."""
    total_cushion = report.total_cushion_nodes()
    if total_cushion <= 0:
        return 0.0
    return min(1.0, max(0.0, report.total_matched_nodes() / total_cushion))


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
        system_prompt=compose_system_prompt(_ARTICULATE_SYSTEM_PROMPT, mode="card_articulation"),
        user_message=user_message,
        domain=ARTICULATE_DOMAIN,
        concept=ARTICULATE_CONCEPT,
    )

    citations = _citations_from_report(report)
    match_strength = _match_strength_from_report(report)

    if not response.success:
        log.warning("articulation LLM call failed: %s", response.error)
        # Fallback: use the report's raw fields directly.
        return ArticulatedCard(
            report_id=report.report_id,
            agent_id=report.agent_id,
            domain=report.domain_explored,
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
            citations=citations,
            match_strength=match_strength,
        )

    fields = parse_articulation_response(response.content)

    return ArticulatedCard(
        report_id=report.report_id,
        agent_id=report.agent_id,
        domain=report.domain_explored,
        spark=fields["spark"],
        source_shape=fields["source_shape"],
        bridge=fields["bridge"],
        use=fields["use"],
        limit=fields["limit"],
        confidence=report.confidence,
        confidence_detail=report.match_ratio_summary(),
        citations=citations,
        match_strength=match_strength,
    )


__all__ = [
    "ARTICULATE_DOMAIN",
    "ARTICULATE_CONCEPT",
    "ArticulatedCard",
    "build_articulation_user_message",
    "parse_articulation_response",
    "articulate_report",
]
