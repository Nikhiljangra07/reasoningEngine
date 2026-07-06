"""
Matching — the metal detector mechanic.

When a wandering agent fetches content from any domain, it compares that
content against the three-layer cushion. Any node match (1+) in any layer
triggers a dig. Match strength determines iteration depth (locked at start,
no rescaling).

Per Law 7: match on STRUCTURAL ESSENCE, not topical surface. The matcher's
job is to score essence and mechanism resonance HIGHER than surface entity
overlap. A movie scene and an AI architecture problem may share zero
nouns and still match strongly on essence (bounded freedom) and mechanism
(soft constraint enables emergence) — that's the Heisenberg-zone signal.

The matching call itself is LLM-mediated: an LLM judges whether each
cushion node "appears in" the content's structure. We don't try to do this
with embeddings alone — embeddings can't reliably catch cross-domain
structural resonance (they're trained on surface co-occurrence). The LLM
acts as a structural-pattern recognizer.

ISOLATION: imports cushion types + LLM client only. No persistence, no
runtime orchestration.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from src.identity import compose_system_prompt
from src.llm.client import LLMClient, LLMResponse
from src.wandering.cushion import CushionGraph, CushionLayer
from src.wandering.report import LayerMatch


log = logging.getLogger("constellax.wandering.matching")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Match scoring uses a cheap-but-capable model — Haiku 4.5 — because the
#: question per node is small and structured ("does this content exhibit
#: <node> in any form?"). We don't need Sonnet's depth for this judgment;
#: the cost would compound badly across many agents × many fetches.
MATCH_DOMAIN = "psychology"  # routes to Haiku 4.5 per provider_map
MATCH_CONCEPT = "structural_match_check"

#: Iteration scaling per total match count (across all layers).
#: Locked at start of dig — no mid-dig rescaling. From plan.
#:   1 node matched → 3 iterations
#:   2 nodes        → 4
#:   3 nodes        → 5
#:   4+ nodes       → 5 (capped)
MIN_DIG_ITERATIONS = 3
MAX_DIG_ITERATIONS = 5


def iterations_for_match(total_matched_nodes: int) -> int:
    """Compute iteration budget for a dig based on initial match count.

    Locked at start; no mid-dig rescaling. Formula: min(2 + N, 5),
    floored at MIN_DIG_ITERATIONS for any match >= 1.

    Zero matches → 0 iterations (no dig, agent moves on).
    """
    if total_matched_nodes <= 0:
        return 0
    raw = 2 + total_matched_nodes
    return max(MIN_DIG_ITERATIONS, min(raw, MAX_DIG_ITERATIONS))


# ---------------------------------------------------------------------------
# Match call
# ---------------------------------------------------------------------------


_MATCH_SYSTEM_PROMPT = """\
You are Constellax's structural-match judge for Wandering Room.

You will be given:
  1. A cushion graph — three layers (actual / essence / mechanism) of
     structural primitives describing a user's problem.
  2. A piece of CONTENT from some domain (could be anything — a movie
     scene, a Wikipedia article, a tweet, ancient philosophy).

Your job: judge which cushion nodes the content exhibits.

CRITICAL: match on STRUCTURAL ESSENCE, not topical surface.

  - "actual" layer matches when the content's surface entities or scope
    overlap with the cushion's literal description.
  - "essence" layer matches when the content exhibits the same structural
    dynamics (forces, tensions, constraints, cycles) — REGARDLESS of
    whether the nouns overlap.
  - "mechanism" layer matches when the content operates under the same
    causal primitive — REGARDLESS of domain.

EXAMPLE: cushion essence node is "bounded freedom". Content is about a
jazz soloist improvising within a chord progression. The nouns share
nothing, but the structural dynamic is identical — MATCH.

Be generous on structural axes. Be strict on actual layer (don't claim
surface match unless the entities really overlap).

# OUTPUT FORMAT

Return ONE JSON object listing which nodes from each layer match:

{
  "actual": ["<exact node string from cushion>", ...],
  "essence": ["<exact node string from cushion>", ...],
  "mechanism": ["<exact node string from cushion>", ...]
}

Use the EXACT node strings from the cushion. Empty list if no nodes match
in a layer. No prose preamble. No code fences.
"""


def build_match_user_message(
    cushion: CushionGraph,
    content: str,
    domain_hint: str = "",
) -> str:
    """Render the cushion + content into a user-message payload."""
    blocks = ["# CUSHION (the user's problem)"]
    for layer in cushion.layers():
        blocks.append(f"\n## {layer.name.upper()} layer")
        if layer.summary:
            blocks.append(layer.summary)
        blocks.append("Nodes:")
        for n in layer.nodes:
            blocks.append(f"  - {n}")

    blocks.append("\n# CONTENT to evaluate")
    if domain_hint:
        blocks.append(f"(Source domain hint: {domain_hint})")
    blocks.append(content.strip())

    blocks.append(
        "\n# TASK\n"
        "Return the JSON listing which cushion nodes match this content. "
        "Match on essence and mechanism (structural) even if actual surface "
        "is zero. Use exact node strings."
    )

    return "\n\n".join(blocks)


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
        raise ValueError("no JSON object in match response")
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
    raise ValueError("unterminated JSON in match response")


def parse_match_response(
    response_text: str,
    cushion: CushionGraph,
) -> dict[str, LayerMatch]:
    """Parse the LLM's match response into per-layer LayerMatch records.

    Filters matched nodes to ONLY those that actually appear in the cushion
    (defense against the LLM inventing nodes that aren't in the graph).
    """
    json_text = _extract_json_object(response_text)
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError("match payload is not a JSON object")

    matches: dict[str, LayerMatch] = {}
    for layer in cushion.layers():
        cushion_nodes = set(layer.nodes)
        claimed = payload.get(layer.name, [])
        if not isinstance(claimed, list):
            claimed = []

        # Filter to nodes that actually exist in the cushion. Strict.
        valid = [str(n).strip() for n in claimed if str(n).strip() in cushion_nodes]
        # Deduplicate while preserving order.
        seen: set[str] = set()
        dedup: list[str] = []
        for n in valid:
            if n not in seen:
                seen.add(n)
                dedup.append(n)

        matches[layer.name] = LayerMatch(
            layer_name=layer.name,
            matched_nodes=dedup,
            total_nodes=layer.node_count(),
        )

    return matches


# ---------------------------------------------------------------------------
# Top-level match function
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """The outcome of a single match call.

    `matches` is keyed by layer name ("actual", "essence", "mechanism").
    `total_matched_nodes` is the sum across all layers — used directly
    by iterations_for_match() to compute dig depth.
    `dig_iterations` is what the agent's loop should use if it dives.
    """

    matches: dict[str, LayerMatch]
    total_matched_nodes: int
    dig_iterations: int
    raw_response: str = ""

    def has_any_match(self) -> bool:
        return self.total_matched_nodes > 0


async def match_content(
    cushion: CushionGraph,
    content: str,
    client: LLMClient,
    *,
    domain_hint: str = "",
) -> MatchResult:
    """Run the cushion against a piece of content and score the match.

    This is the metal-detector beep. Even a 1/N match in any layer counts
    as "beep loud enough to dig" — the agent's loop decides what to do
    with the result (digs if has_any_match, moves on if not).
    """
    user_message = build_match_user_message(cushion, content, domain_hint)
    response: LLMResponse = await client.call(
        system_prompt=compose_system_prompt(_MATCH_SYSTEM_PROMPT, mode="structural_match"),
        user_message=user_message,
        domain=MATCH_DOMAIN,
        concept=MATCH_CONCEPT,
    )
    if not response.success:
        log.warning("match LLM call failed: %s", response.error)
        return MatchResult(
            matches={
                layer.name: LayerMatch(
                    layer_name=layer.name,
                    matched_nodes=[],
                    total_nodes=layer.node_count(),
                )
                for layer in cushion.layers()
            },
            total_matched_nodes=0,
            dig_iterations=0,
            raw_response="",
        )

    try:
        matches = parse_match_response(response.content, cushion)
    except (ValueError, json.JSONDecodeError) as e:
        # The model occasionally returns prose / no JSON object. Silently
        # treating that as "no match" drops a REAL dig trigger (observed twice
        # in the 2026-06-17 autonomous run). Re-ask ONCE with a hard format
        # reminder before giving up — bounded to one extra call, only on the
        # rare parse failure, so the hot path cost is unchanged.
        log.warning("match response unparseable (%s) — one strict retry", e)
        retry = await client.call(
            system_prompt=compose_system_prompt(_MATCH_SYSTEM_PROMPT, mode="structural_match"),
            user_message=(
                user_message
                + "\n\nREMINDER: Output ONLY the JSON object "
                '{"actual":[...],"essence":[...],"mechanism":[...]} — '
                "no prose, no preamble, no code fences."
            ),
            domain=MATCH_DOMAIN,
            concept=MATCH_CONCEPT,
        )
        try:
            matches = parse_match_response(
                retry.content if getattr(retry, "success", False) else "", cushion
            )
            response = retry  # raw_response should reflect the parsed payload
        except (ValueError, json.JSONDecodeError) as e2:
            log.warning("match retry still unparseable (%s) — treating as no-match", e2)
            # Treat unparseable as "no match" — safe, the agent moves on.
            matches = {
                layer.name: LayerMatch(
                    layer_name=layer.name,
                    matched_nodes=[],
                    total_nodes=layer.node_count(),
                )
                for layer in cushion.layers()
            }

    total = sum(m.match_count for m in matches.values())
    return MatchResult(
        matches=matches,
        total_matched_nodes=total,
        dig_iterations=iterations_for_match(total),
        raw_response=response.content,
    )


__all__ = [
    "MATCH_DOMAIN",
    "MATCH_CONCEPT",
    "MIN_DIG_ITERATIONS",
    "MAX_DIG_ITERATIONS",
    "iterations_for_match",
    "build_match_user_message",
    "parse_match_response",
    "MatchResult",
    "match_content",
]
