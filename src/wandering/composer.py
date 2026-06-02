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

import asyncio
import json
import logging
import re
import time
from typing import Any

from src.bridge.embedding_service import EmbeddingResult, GeminiEmbeddingService
from src.identity import RECOVER_GOAL_PROBE, compose_system_prompt
from src.identity.disciplines.goal_supremacy import surface_real_goal
from src.llm.client import LLMClient, LLMResponse
from src.wandering.cushion import (
    CushionField,
    CushionGraph,
    CushionInput,
    CushionLayer,
    CushionNode,
    SkipReason,
    make_cushion_node_id,
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

Return ONE valid JSON object. Each node is an OBJECT with three fields,
not a bare string:

{
  "actual": {
    "summary": "one paragraph",
    "nodes": [
      {
        "text": "wandering AI agents",
        "embedding_text": "autonomous agents exploring under structural constraint",
        "search_queries": ["agentic AI exploration", "autonomous research agents", "bounded autonomous behavior"]
      },
      ...
    ]
  },
  "essence": {
    "summary": "one paragraph",
    "nodes": [
      {
        "text": "bounded freedom",
        "embedding_text": "freedom emerging within fixed structural limits — chaos that stays useful because constraint anchors it",
        "search_queries": ["constraint as enabler", "bounded chaos productive", "structure enabling creativity"]
      },
      ...
    ]
  },
  "mechanism": {
    "summary": "one paragraph",
    "nodes": [
      {
        "text": "soft constraint enables emergence",
        "embedding_text": "systems aiming for unpredictable output under resource limits require structural rather than imperative constraint",
        "search_queries": ["soft constraint systems", "structural constraint vs hard rules", "emergence under bounded resources"]
      },
      ...
    ]
  }
}

FIELD CONVENTIONS:

  text — the canonical short label for the node (1-5 words for actual/essence,
         1 short sentence acceptable for mechanism). This is what the user
         and operator see; keep it tight.

  embedding_text — a one-sentence structural rephrasing of the node in
         language that would naturally appear in content from ANY domain
         exhibiting this pattern. This is what gets embedded for the
         vector channel; it must read like prose, not like a label.
         Default to the text itself if no better rephrasing comes to mind.

  search_queries — 2-4 distinct queries an internet search engine could
         consume to find content exhibiting this pattern. Vary the
         vocabulary so different queries surface different domains;
         do NOT just rephrase the text three ways.

No prose preamble. No code fences. JUST the JSON object.

LEGACY FORMAT (still accepted, but please use the OBJECT form above):
You MAY return nodes as plain strings instead of objects — the parser
tolerates it for backward compatibility. But the new format unlocks
cross-domain retrieval; prefer it.
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


def _parse_node_entry(layer_name: str, entry: Any, *, session_id: str) -> CushionNode | None:
    """Parse one node entry into a CushionNode.

    Accepts BOTH the new dual-artifact object form
        {"text": "...", "embedding_text": "...", "search_queries": [...]}
    and the legacy bare-string form
        "node text"

    Returns None if the entry is malformed or has no text — the caller
    filters Nones out before count validation.

    Constellation Interpreter (2026-06-01).
    """
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            return None
        return CushionNode(
            id=make_cushion_node_id(session_id, layer_name, text),
            text=text,
            layer=layer_name,
            embedding_text=text,  # default — same as text when LLM didn't rephrase
        )

    if isinstance(entry, dict):
        text = str(entry.get("text", "")).strip()
        if not text:
            return None
        emb_text_raw = entry.get("embedding_text")
        emb_text = str(emb_text_raw).strip() if emb_text_raw else text

        sq_raw = entry.get("search_queries") or []
        if not isinstance(sq_raw, list):
            sq_raw = []
        search_queries = tuple(
            str(q).strip() for q in sq_raw
            if isinstance(q, (str, int, float)) and str(q).strip()
        )

        return CushionNode(
            id=make_cushion_node_id(session_id, layer_name, text),
            text=text,
            layer=layer_name,
            search_queries=search_queries,
            embedding_text=emb_text,
        )

    return None


def _parse_layer(name: str, raw: Any, *, session_id: str = "") -> CushionLayer:
    """Convert one layer's JSON payload into a CushionLayer. Tolerant of
    minor schema drift (extra keys, slight type wobbles) but strict about
    node counts.

    Constellation Interpreter (2026-06-01): each layer now carries both
    the canonical `nodes: list[str]` (text only, backward-compat) AND a
    parallel `node_records: list[CushionNode]` with the dual-artifact
    metadata. The records' `embedding` field is left as None here; the
    caller populates it after parsing via the embedding service.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"layer {name!r} is not a JSON object")

    nodes_raw = raw.get("nodes", [])
    if not isinstance(nodes_raw, list):
        raise ValueError(f"layer {name!r} has non-list 'nodes' field")

    records: list[CushionNode] = []
    for entry in nodes_raw:
        rec = _parse_node_entry(name, entry, session_id=session_id)
        if rec is not None:
            records.append(rec)

    if len(records) < MIN_NODES_PER_LAYER:
        raise ValueError(
            f"layer {name!r} has {len(records)} nodes; "
            f"minimum is {MIN_NODES_PER_LAYER}"
        )
    if len(records) > MAX_NODES_PER_LAYER:
        # Truncate rather than reject — the model overshot; keep first N.
        records = records[:MAX_NODES_PER_LAYER]
        log.debug(
            "layer %r had >%d nodes; truncating",
            name,
            MAX_NODES_PER_LAYER,
        )

    nodes_text = [r.text for r in records]
    summary = str(raw.get("summary", "")).strip()

    return CushionLayer(
        name=name,
        nodes=nodes_text,
        summary=summary,
        node_records=records,
    )


def parse_extraction_response(
    response_text: str,
    *,
    session_id: str = "",
) -> dict[str, CushionLayer]:
    """Parse Sonnet's JSON output into three CushionLayers.

    `session_id` scopes the deterministic node ids generated for each
    CushionNode. Default "" produces session-less ids — fine for tests
    and for graphs that won't be persisted. The Neo4j upsert path
    regenerates ids with the real session_id when persisting.

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
        "actual":    _parse_layer("actual",    payload["actual"],    session_id=session_id),
        "essence":   _parse_layer("essence",   payload["essence"],   session_id=session_id),
        "mechanism": _parse_layer("mechanism", payload["mechanism"], session_id=session_id),
    }


# ---------------------------------------------------------------------------
# Per-node embedding — Constellation Interpreter (2026-06-01)
# ---------------------------------------------------------------------------


async def _embed_layer_nodes(
    layers: dict[str, CushionLayer],
    embedder: GeminiEmbeddingService,
) -> None:
    """In-place: populate each CushionNode.embedding via the embedding
    service. Runs all embeddings concurrently for latency.

    Nodes without a non-empty embedding_text are skipped (their
    embedding stays None). Embedding-call failures are logged at WARNING
    and the node's embedding stays None — the matcher's vector channel
    will skip those nodes but the rest of the cushion still works.

    The matcher's vector channel is degradation-tolerant by design: a
    cushion where 2 of 12 nodes failed to embed still scores fine on
    the 10 that succeeded.
    """
    # Collect every node across all layers, paired with its record so we
    # can write the embedding back into the right slot.
    targets: list[CushionNode] = []
    for layer in layers.values():
        if layer.node_records:
            for rec in layer.node_records:
                if rec.embedding_text and rec.embedding_text.strip():
                    targets.append(rec)

    if not targets:
        return

    # Parallel embed. `gather` with return_exceptions=True keeps one
    # failure from sinking the whole batch.
    results = await asyncio.gather(
        *(embedder.embed(rec.embedding_text) for rec in targets),
        return_exceptions=True,
    )

    fail_count = 0
    for rec, result in zip(targets, results, strict=False):
        if isinstance(result, BaseException):
            log.warning("embed() raised for node %r: %s", rec.text[:40], result)
            fail_count += 1
            continue
        if not isinstance(result, EmbeddingResult):
            fail_count += 1
            continue
        if result.success and result.vector:
            rec.embedding = result.vector
        else:
            log.warning(
                "embed() returned failure for node %r: %s",
                rec.text[:40], result.error or "no_vector",
            )
            fail_count += 1

    if fail_count:
        log.warning(
            "embedding partial: %d/%d nodes failed to embed",
            fail_count, len(targets),
        )


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
    session_id: str = "",
    auto_enrich: bool = True,
    embedder: GeminiEmbeddingService | None = None,
) -> CushionGraph:
    """Build a CushionGraph from the user's four-field intake.

    This is the public entry point. Steps:
      1. Auto-enrich the input with project memory context (if user_id known)
      2. Validate minimal viability (problem field must be filled)
      3. Call Sonnet via the synthesizer route to extract three layers
      4. Parse the JSON response into CushionLayers (now dual-artifact)
      5. Embed each node's embedding_text via Gemini (parallel)
      6. Construct and return the CushionGraph

    `session_id` scopes deterministic node ids. Pass it when the cushion
    will be persisted to Neo4j — without it, node ids collide across
    sessions sharing the same node text (still correct for in-memory
    use, just degraded uniqueness for persistence).

    `embedder` defaults to a freshly-constructed GeminiEmbeddingService.
    If embedding fails entirely (no API key, network down), the cushion
    is still returned with all CushionNode.embedding = None — the new
    matcher's vector channel will skip those nodes; the rest of the
    pipeline keeps working.

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
        system_prompt=compose_system_prompt(_EXTRACTION_SYSTEM_PROMPT, mode="cushion_compose"),
        user_message=user_message,
        domain=EXTRACTION_DOMAIN,
        concept=EXTRACTION_CONCEPT,
    )

    if not response.success:
        raise RuntimeError(
            f"cushion extraction LLM call failed: {response.error}"
        )

    # Step 4: parse (with session_id for deterministic node ids)
    layers = parse_extraction_response(response.content, session_id=session_id)

    # Step 5: embed each node — Constellation Interpreter (2026-06-01).
    # Failures here are non-fatal; nodes without embeddings just won't
    # participate in the vector channel of the new matcher.
    try:
        emb = embedder or GeminiEmbeddingService()
        await _embed_layer_nodes(layers, emb)
    except Exception as e:  # pragma: no cover — embedder construction failure
        log.warning("cushion node embedding step skipped: %s", e)

    # Step 6: assemble the graph
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

    # Step 6: identity-layer metadata — check whether the user's stated
    # problem contradicts other signals in their brief. Pure-Python
    # heuristic, no LLM. When it fires, the rendered probe is attached
    # to the graph and the frontend may show it to the user before
    # committing to the wander. The wander itself is not gated by this
    # — the graph is built and shipped either way; the probe is a hint.
    try:
        stated = input_data.problem.content
        signals = tuple(filter(None, (
            input_data.context.content,
            input_data.vision.content,
            input_data.current_map.content,
        )))
        recovered = surface_real_goal(stated, signals)
        if recovered.surfaced:
            graph.real_goal_probe = RECOVER_GOAL_PROBE.format(
                stated=recovered.stated,
                alternative=recovered.real,
            )
    except Exception:  # pragma: no cover — surface_real_goal is pure
        log.exception("surface_real_goal raised; leaving real_goal_probe empty")
        graph.real_goal_probe = None

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
