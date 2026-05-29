"""
Brief composer — turn the user's four-field intake into a three-layer cushion.

Flow:

  CushionInput  (user's four fields + auto-enriched memory context)
       │
       ▼
  build_extraction_prompt()  ──── Sonnet 4.6  ─────►  structured JSON
       │
       ▼
  parse_extraction_response()  ────►  CushionLayer × 3
       │
       ▼
  CushionGraph (the immutable anchor for the session)

The user provides four fields (Problem / Context / Vision / Current Map).
Sonnet reads all four plus auto-enriched project memory and derives the
three structural layers (Actual / Essence / Mechanism). This separation
matters: the user does NOT need to articulate "essence" or "mechanism" in
abstract terms — Sonnet extracts those from their concrete answers.

NON-NEGOTIABLE: the cushion is immutable once built. Wandering agents may
swing freely; the anchor never moves. (Law 2.)

ISOLATION: this module depends on src.llm.client (the LLM seam) and
src.wandering.cushion (the type module). It does NOT import from
src.formation, src.dispatcher, or any wandering-runtime modules. The
composer's only job is to turn input into anchor; it does not run agents,
score reports, or persist anything. Persistence lives in the wandering
runtime; this module hands back an in-memory CushionGraph.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from src.llm.client import LLMClient, LLMResponse
from src.wandering.cushion import (
    CushionField,
    CushionGraph,
    CushionInput,
    CushionLayer,
    SkipReason,
)

log = logging.getLogger("constellax.wandering.composer")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sonnet 4.6 — routed via provider_map's "synthesizer" domain key. We use
#: the same model that handles anchor building elsewhere in the system
#: (speech.py synthesis) for voice + reasoning consistency. Three-layer
#: extraction is a deep, structured reasoning task — DeepSeek and Haiku
#: handle the wandering itself, but the anchor build needs Sonnet's depth.
EXTRACTION_DOMAIN = "synthesizer"
EXTRACTION_CONCEPT = "cushion_extraction"

#: Minimum/maximum nodes per layer. 3 is the floor for a useful metal-
#: detector graph; below that, matching becomes trivial overlap. 8 is
#: the ceiling — beyond it, the graph becomes noise and matches become
#: too easy. Inside [3, 8] the model picks based on problem complexity.
MIN_NODES_PER_LAYER = 3
MAX_NODES_PER_LAYER = 8


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------


_EXTRACTION_SYSTEM_PROMPT = """\
You are Constellax's anchor builder for Wandering Room.

The user is about to launch a Wandering Room session — a research mode where
10 AI agents will wander across all of human knowledge (movies, religion,
ancient philosophy, sports, anything — domain is UNCONSTRAINED) and look for
partial structural matches to the user's problem.

Your job RIGHT NOW: turn the user's four-field brief into a three-layer
structural anchor that the agents will match against. The user does NOT need
to know about "essence" or "mechanism" in the abstract — you extract those
from their concrete answers.

# THE THREE LAYERS

## ACTUAL layer
The literal, concrete description. What the problem IS in surface terms.
Entities involved, scope, history. 3-8 nodes capturing concrete elements.

Example node list for an "AI agent control" problem:
  ["wandering AI agents", "internet exploration", "research anchor",
   "credit/token budget", "user's research project"]

## ESSENCE layer
The structural-dynamic pattern. The forces, tensions, constraints, cycles,
and asymmetries underlying the problem — regardless of domain. 3-8 nodes
capturing dynamics, NOT topics.

Example node list for the same problem:
  ["bounded freedom", "productive constraint",
   "anchored chaos", "soft-vs-hard constraint paradox",
   "trust through observation"]

These should match against ANY domain (a poem about a kite, jazz
improvisation, parenting teenagers) that exhibits the same dynamics.

## MECHANISM layer
The causal primitive — the abstract operating logic that, applied to any
domain, would produce this kind of problem. 3-8 nodes capturing the
problem's underlying causal shape, NOT a solution.

Example node list:
  ["systems aiming for unpredictable output under resource limits require
    soft structural constraint",
   "hard constraint optimizes outputs into predictability",
   "the goal IS unpredictability, so constraint must be structural",
   "observation as control without interruption"]

These should match against ANY system (educational, biological,
organizational, mechanical) that operates by the same causal primitive.

# CRITICAL RULES

1. Match on STRUCTURAL ESSENCE, not topical surface. A movie about a kite
   with a string should match an "AI agent control" anchor at the essence
   layer (bounded freedom) and mechanism layer (soft constraint enables
   unpredictable value), even though "AI" and "kite" share no nouns.

2. Each layer must have 3-8 nodes. Below 3, the metal detector is too
   trivial. Above 8, it's noise.

3. The mechanism layer describes the PROBLEM SPACE primitive, not the
   SOLUTION SPACE primitive. "Systems aiming for X under Y require Z" —
   not "the way to solve this is W". We are not delivering solutions; we
   are delivering anchors against which inspiration can resonate.

4. Per-layer summaries should be ONE PARAGRAPH each. Compact, dense,
   structural. Agents will read these on every turn.

# OUTPUT FORMAT

Return ONE valid JSON object with this exact shape:

{
  "actual": {
    "summary": "one paragraph",
    "nodes": ["node 1", "node 2", "node 3", ...]
  },
  "essence": {
    "summary": "one paragraph",
    "nodes": ["dynamic 1", "tension 2", "force 3", ...]
  },
  "mechanism": {
    "summary": "one paragraph",
    "nodes": ["causal primitive 1", "causal primitive 2", ...]
  }
}

No prose preamble. No code fences. JUST the JSON object.
"""


def build_extraction_user_message(input_data: CushionInput) -> str:
    """Build the user-message payload that Sonnet sees.

    Includes every filled field with its name, and the auto-enriched memory
    context if any. Skipped fields are marked explicitly so Sonnet knows
    not to invent content for them.
    """
    blocks: list[str] = []

    for field_obj in input_data.fields():
        label = field_obj.name.replace("_", " ").upper()
        if field_obj.is_filled():
            blocks.append(f"# {label}\n{field_obj.content.strip()}")
        else:
            blocks.append(
                f"# {label}\n"
                f"(skipped by user — extract what you can from other fields)"
            )

    if input_data.memory_enrichment.strip():
        blocks.append(
            "# AUTO-ENRICHED PROJECT CONTEXT (from user's memory graph)\n"
            + input_data.memory_enrichment.strip()
        )

    blocks.append(
        "\n# YOUR TASK\n"
        "Extract the three structural layers (actual / essence / mechanism) "
        "from the user's brief above. Return JSON per the system prompt's spec."
    )

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Strip ```json ... ``` style fences if the model added them anyway.

    Sonnet usually obeys the no-code-fences instruction, but other models
    or rare Sonnet responses may include them. Cheap to handle.
    """
    fenced = re.match(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from a string, even if surrounded
    by prose. Bracket-depth walk; handles nested objects and string literals.
    """
    text = _strip_code_fences(text)

    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in extraction response")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("unterminated JSON object in extraction response")


def _parse_layer(name: str, raw: Any) -> CushionLayer:
    """Convert one layer's JSON payload into a CushionLayer. Tolerant of
    minor schema drift (extra keys, slight type wobbles) but strict about
    node counts."""
    if not isinstance(raw, dict):
        raise ValueError(f"layer {name!r} is not a JSON object")

    nodes_raw = raw.get("nodes", [])
    if not isinstance(nodes_raw, list):
        raise ValueError(f"layer {name!r} has non-list 'nodes' field")
    nodes = [str(n).strip() for n in nodes_raw if str(n).strip()]

    if len(nodes) < MIN_NODES_PER_LAYER:
        raise ValueError(
            f"layer {name!r} has {len(nodes)} nodes; "
            f"minimum is {MIN_NODES_PER_LAYER}"
        )
    if len(nodes) > MAX_NODES_PER_LAYER:
        # Truncate rather than reject — the model overshot; keep first N.
        nodes = nodes[:MAX_NODES_PER_LAYER]
        log.debug(
            "layer %r had >%d nodes; truncating",
            name,
            MAX_NODES_PER_LAYER,
        )

    summary = str(raw.get("summary", "")).strip()

    return CushionLayer(name=name, nodes=nodes, summary=summary)


def parse_extraction_response(response_text: str) -> dict[str, CushionLayer]:
    """Parse Sonnet's JSON output into three CushionLayers.

    Raises ValueError if the response can't be parsed into a well-formed
    three-layer structure. Callers should catch and degrade (e.g., re-prompt
    or fall back to a minimal cushion) — see compose_cushion() for the
    integrated retry path.
    """
    json_text = _extract_json_object(response_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"extraction response is not valid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError("extraction payload is not a JSON object")

    required = ("actual", "essence", "mechanism")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"extraction missing required layer(s): {missing}")

    return {
        "actual": _parse_layer("actual", payload["actual"]),
        "essence": _parse_layer("essence", payload["essence"]),
        "mechanism": _parse_layer("mechanism", payload["mechanism"]),
    }


# ---------------------------------------------------------------------------
# Memory enrichment (stub for Phase 0 — wires to project memory later)
# ---------------------------------------------------------------------------


async def fetch_memory_enrichment(user_id: str | None) -> str:
    """Auto-enrich the cushion with relevant project context from memory.

    Delegates to memory_enrichment.fetch_memory_enrichment_real which
    queries the thread store for the user's recent activity. Defensive:
    returns empty string on any failure (no thread store available,
    Neo4j down, etc.) — the cushion is built without enrichment in
    that case, and the session still works.

    Wired up post Phase 0: the real query is in
    src/wandering/memory_enrichment.py. This thin re-export keeps the
    composer's public surface stable while letting the implementation
    grow independently.
    """
    if not user_id:
        return ""
    try:
        from src.wandering.memory_enrichment import fetch_memory_enrichment_real
        return await fetch_memory_enrichment_real(user_id)
    except Exception as e:
        log.debug("memory enrichment failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


async def compose_cushion(
    input_data: CushionInput,
    client: LLMClient,
    *,
    user_id: str | None = None,
    auto_enrich: bool = True,
) -> CushionGraph:
    """Build a CushionGraph from the user's four-field intake.

    This is the public entry point. Steps:
      1. Auto-enrich the input with project memory context (if user_id known)
      2. Validate minimal viability (problem field must be filled)
      3. Call Sonnet via the synthesizer route to extract three layers
      4. Parse the JSON response into CushionLayers
      5. Construct and return the CushionGraph

    Raises:
      ValueError — if the input is not minimally viable (problem skipped)
      RuntimeError — if extraction fails after retries (caller decides
        whether to fall back to a degraded cushion or surface the error
        to the user)
    """
    if not input_data.is_minimally_viable():
        raise ValueError(
            "cushion input is not minimally viable: 'problem' field is required"
        )

    # Step 1: auto-enrich (Phase 0 stub returns "")
    if auto_enrich and not input_data.memory_enrichment:
        input_data.memory_enrichment = await fetch_memory_enrichment(user_id)

    # Step 2: build the extraction request
    user_message = build_extraction_user_message(input_data)

    # Step 3: call Sonnet
    response: LLMResponse = await client.call(
        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
        user_message=user_message,
        domain=EXTRACTION_DOMAIN,
        concept=EXTRACTION_CONCEPT,
    )

    if not response.success:
        raise RuntimeError(
            f"cushion extraction LLM call failed: {response.error}"
        )

    # Step 4: parse
    layers = parse_extraction_response(response.content)

    # Step 5: assemble the graph
    graph = CushionGraph(
        actual=layers["actual"],
        essence=layers["essence"],
        mechanism=layers["mechanism"],
        raw_input=input_data,
        extraction_model=response.model or "unknown",
        extracted_at=time.time(),
    )

    if not graph.is_well_formed():
        raise RuntimeError(
            "extracted cushion is not well-formed (a layer is empty); "
            "Sonnet did not honor the schema"
        )

    return graph


__all__ = [
    "EXTRACTION_DOMAIN",
    "EXTRACTION_CONCEPT",
    "MIN_NODES_PER_LAYER",
    "MAX_NODES_PER_LAYER",
    "build_extraction_user_message",
    "parse_extraction_response",
    "fetch_memory_enrichment",
    "compose_cushion",
]
