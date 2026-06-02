"""
Synthesis layer — aggregate all reports into a coherent dossier.

Distinct from articulation: articulate.py renders each report individually
for human reading. synthesis.py reads ALL reports together and produces
the dossier-level structure:

  - Top insights (HIGH confidence cards surfaced first)
  - Clusters of related partial-matches
  - Contradictions (reports pointing different directions)
  - Opportunity paths (where multiple HIGH-confidence sparks converge)
  - Open questions (what didn't resolve)
  - Recommended next direction

Per Law 3: LOW confidence reports are NOT filtered out at this layer.
They go to the LOW shelf in the dossier — the Heisenberg zone is
surfaced, not hidden. The synthesis layer's job is to ORGANIZE, not
to GATEKEEP.

Per Law 2: synthesis does NOT produce conclusions. It produces a STRUCTURED
MAP that the user reads to do their own synthesis. We're cartographers,
not navigators.

ISOLATION: imports report + articulate types + LLM client. No persistence.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from src.identity import compose_system_prompt
from src.llm.client import LLMClient, LLMResponse
from src.wandering.articulate import ArticulatedCard
from src.wandering.report import Confidence, ExplorationReport


log = logging.getLogger("constellax.wandering.synthesis")


#: Synthesis uses Sonnet 4.6 — it reads 20-30 cards and produces the
#: dossier-level map. The depth needs prose quality + cross-doc reasoning.
SYNTHESIS_DOMAIN = "synthesizer"
SYNTHESIS_CONCEPT = "wandering_synthesis"


# ---------------------------------------------------------------------------
# Synthesis structures
# ---------------------------------------------------------------------------


@dataclass
class InsightCluster:
    """A group of articulated cards that share structural resonance.

    Identified by the synthesis layer (Sonnet groups them based on the
    actual content of cards, not by domain). The label is a one-line
    human-friendly description of what unites the cluster.
    """

    label: str
    card_ids: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class Contradiction:
    """Two cards (or two clusters) that point opposite directions.

    Surfaced explicitly because contradictions are signal — they show
    the user where their problem might be genuinely ambiguous.
    """

    description: str
    card_ids: list[str] = field(default_factory=list)


@dataclass
class OpportunityPath:
    """A direction multiple cards converge on; a candidate way forward.

    Different from "clusters" — opportunity paths combine cards across
    clusters that point at the same actionable possibility.
    """

    description: str
    supporting_card_ids: list[str] = field(default_factory=list)
    confidence_estimate: Confidence = Confidence.MEDIUM

    # Identity-layer metadata (additive — does not filter or reorder).
    # Populated by `opportunity_capture.test()` in `build_dossier`
    # against the cushion's goal. One of "capture" / "surface" / "skip"
    # / "" (empty when scoring wasn't run). The frontend may show a
    # badge ("strong opening" for capture, "novel but check first" for
    # surface) but engine logic does not drop "skip" paths in this
    # sprint — every surfaced path still reaches the user. See
    # doctrine §10 "discipline metadata".
    verdict: str = ""
    verdict_score: int = 0   # 0..6 — number of the six questions passed


@dataclass
class SynthesisMap:
    """The dossier-level structured map.

    Used by the dossier layer to render the final user-facing artifact.
    """

    top_insights: list[str] = field(default_factory=list)        # card_ids, HIGH first
    clusters: list[InsightCluster] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    opportunity_paths: list[OpportunityPath] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recommended_next_direction: str = ""
    what_would_change_the_verdict: str = ""

    # Identity-layer enforcement (0.3.4). Paths with
    # `opportunity_capture.test` verdict='skip' are moved here in
    # `build_dossier` rather than dropped — the user can still see
    # them under a "weak signals" section in the frontend, but the
    # primary opportunity_paths list is curated to the paths that
    # passed at least 4/6 questions in the six-question test. This
    # is the SOFT enforcement of opportunity discipline: filter
    # without erasing.
    deprioritized_paths: list[OpportunityPath] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------


_SYNTHESIS_SYSTEM_PROMPT = """\
You are Constellax's synthesis layer for Wandering Room.

You receive ALL articulated cards from one session — typically 10-30 cards,
each one a structural bridge an agent found between some external domain
and the user's anchor.

Your job: produce the DOSSIER MAP. Not conclusions. Not "the answer." A
STRUCTURED MAP the user reads to do their own synthesis.

CRITICAL CONSTRAINTS (Constellax Laws):

  1. The insight happens in the USER's head. You ORGANIZE; you do NOT
     deliver a conclusion. Frame the "recommended_next_direction" as a
     direction the user might explore, not the answer they should take.

  2. LOW confidence cards are the Heisenberg zone. Do NOT filter them
     out. They go into clusters and open questions just like HIGH
     confidence cards. The user filters at READ time, not at
     synthesis time.

  3. Honest doubt > performative confidence. If the cards genuinely
     contradict each other, name the contradiction. If many cards
     converge weakly, say "weakly converge" — don't oversell.

# YOUR OUTPUT

Return ONE JSON object:

{
  "top_insights": [<list of card_ids, HIGH confidence first, then MEDIUM>],
  "clusters": [
    {
      "label": "<one-line group description>",
      "card_ids": [<card_ids>],
      "summary": "<one-paragraph: what unites these cards>"
    }
  ],
  "contradictions": [
    {
      "description": "<one-line: what cards or clusters disagree on>",
      "card_ids": [<card_ids involved>]
    }
  ],
  "opportunity_paths": [
    {
      "description": "<one-line: a direction the user might explore>",
      "supporting_card_ids": [<card_ids that support this path>],
      "confidence_estimate": "low|medium|high"
    }
  ],
  "open_questions": [
    "<question the cards raise but don't answer — string>"
  ],
  "recommended_next_direction": "<one paragraph: a direction the user might explore, NOT a conclusion>",
  "what_would_change_the_verdict": "<one sentence: what new info would flip the picture>"
}

No prose preamble. No code fences. Just JSON.
"""


def build_synthesis_user_message(
    anchor_summary: str,
    cards: list[ArticulatedCard],
) -> str:
    """Render all articulated cards into the user-message payload."""
    blocks = [
        "# USER'S ANCHOR (one-line summary)",
        anchor_summary,
        f"\n# ALL ARTICULATED CARDS ({len(cards)} total)",
    ]
    for c in cards:
        blocks.append(f"\n## Card {c.report_id} [{c.confidence.value}]")
        blocks.append(f"  Spark: {c.spark}")
        blocks.append(f"  Source Shape: {c.source_shape}")
        blocks.append(f"  Bridge: {c.bridge}")
        blocks.append(f"  Use: {c.use}")
        blocks.append(f"  Limit: {c.limit}")
        if c.confidence_detail:
            blocks.append(f"  Match: {c.confidence_detail}")

    blocks.append("\n# YOUR TASK")
    blocks.append(
        "Synthesize the dossier map per the spec. Do NOT filter LOW cards. "
        "Do NOT deliver conclusions. Return JSON."
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
        raise ValueError("no JSON object in synthesis response")
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
    raise ValueError("unterminated JSON in synthesis response")


def parse_synthesis_response(response_text: str) -> SynthesisMap:
    """Parse the JSON. Returns a SynthesisMap. Defensive against missing
    fields — empty list / empty string defaults."""
    try:
        json_text = _extract_json_object(response_text)
        payload = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("synthesis response unparseable: %s", e)
        return SynthesisMap()

    if not isinstance(payload, dict):
        return SynthesisMap()

    def _string_list(raw) -> list[str]:
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        return []

    clusters_raw = payload.get("clusters", [])
    clusters = []
    if isinstance(clusters_raw, list):
        for c in clusters_raw:
            if not isinstance(c, dict):
                continue
            clusters.append(InsightCluster(
                label=str(c.get("label", "")).strip(),
                card_ids=_string_list(c.get("card_ids", [])),
                summary=str(c.get("summary", "")).strip(),
            ))

    contradictions_raw = payload.get("contradictions", [])
    contradictions = []
    if isinstance(contradictions_raw, list):
        for c in contradictions_raw:
            if not isinstance(c, dict):
                continue
            contradictions.append(Contradiction(
                description=str(c.get("description", "")).strip(),
                card_ids=_string_list(c.get("card_ids", [])),
            ))

    opportunities_raw = payload.get("opportunity_paths", [])
    opportunities = []
    if isinstance(opportunities_raw, list):
        for o in opportunities_raw:
            if not isinstance(o, dict):
                continue
            conf_raw = str(o.get("confidence_estimate", "medium")).strip().lower()
            try:
                conf = Confidence(conf_raw)
            except ValueError:
                conf = Confidence.MEDIUM
            opportunities.append(OpportunityPath(
                description=str(o.get("description", "")).strip(),
                supporting_card_ids=_string_list(o.get("supporting_card_ids", [])),
                confidence_estimate=conf,
            ))

    return SynthesisMap(
        top_insights=_string_list(payload.get("top_insights", [])),
        clusters=clusters,
        contradictions=contradictions,
        opportunity_paths=opportunities,
        open_questions=_string_list(payload.get("open_questions", [])),
        recommended_next_direction=str(
            payload.get("recommended_next_direction", "")
        ).strip(),
        what_would_change_the_verdict=str(
            payload.get("what_would_change_the_verdict", "")
        ).strip(),
    )


# ---------------------------------------------------------------------------
# Top-level synthesis function
# ---------------------------------------------------------------------------


async def synthesize_dossier(
    anchor_summary: str,
    cards: list[ArticulatedCard],
    client: LLMClient,
) -> SynthesisMap:
    """Produce the synthesis map from all articulated cards.

    Defensive: empty input returns empty map (no LLM call wasted on 0 cards).
    """
    if not cards:
        return SynthesisMap()

    user_message = build_synthesis_user_message(anchor_summary, cards)
    response: LLMResponse = await client.call(
        system_prompt=compose_system_prompt(_SYNTHESIS_SYSTEM_PROMPT, mode="dossier_synthesis"),
        user_message=user_message,
        domain=SYNTHESIS_DOMAIN,
        concept=SYNTHESIS_CONCEPT,
    )

    if not response.success:
        log.warning("synthesis LLM call failed: %s", response.error)
        # Fallback: return an empty map but still preserve top_insights
        # by listing HIGH-confidence cards in order. The user gets SOMETHING.
        fallback = SynthesisMap()
        fallback.top_insights = [
            c.report_id for c in cards if c.confidence == Confidence.HIGH
        ] + [
            c.report_id for c in cards if c.confidence == Confidence.MEDIUM
        ]
        return fallback

    return parse_synthesis_response(response.content)


__all__ = [
    "SYNTHESIS_DOMAIN",
    "SYNTHESIS_CONCEPT",
    "InsightCluster",
    "Contradiction",
    "OpportunityPath",
    "SynthesisMap",
    "build_synthesis_user_message",
    "parse_synthesis_response",
    "synthesize_dossier",
]
