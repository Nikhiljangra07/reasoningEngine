"""Procedural visual-spec pipeline for the Map Room.

The synthesizer (speech.py) is overloaded — it's juggling voice, verdict
structure, reasoning, alternatives, falsifiers, AND visuals. When the prompt
contention gets thick, visuals are the first thing it drops.

This module owns visuals as its own pipeline phase:

    classify_visual_intent(memo, question)  → list[str]    # which visual types
    generate_visual_spec(...)               → dict | None  # one VisualSpec
    validate_visual_spec(spec)              → dict | None  # repaired or None
    build_visuals(client, memo, question)   → list[dict]   # full pipeline

The pipeline emits at most 2 visuals (Map Room layout caps gracefully above
that). Vega-Lite is deliberately excluded — it requires real CSV data and
speech.py still owns the CSV path. Procedural pipeline covers the prose-only
path: decision trees + comparison tables built from the memo's own content.

Wire point: server.py opinion phase, right after dispatch completes and the
full memo lands in cache. Runs once per memo, fire-and-await (≤8s budget).
Failures degrade silently to visuals=[] — Map Room already renders that
gracefully.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.llm.client import LLMClient

log = logging.getLogger("visualizer")


# Hard budget on the visualizer LLM call. The synthesizer is "synthesizer"
# domain (Sonnet 4.6). We reuse it — the visualizer is one structured JSON
# emission, no different in cost shape.
_DOMAIN  = "synthesizer"
_CONCEPT = "map_room_visuals"

# Cap on how many visuals we ever emit per memo. Map Room layout works
# best at 1–2; more becomes wallpaper.
_MAX_VISUALS = 2


# ---------------------------------------------------------------------------
# Intent classification — heuristic FORM selector
# ---------------------------------------------------------------------------
#
# Design (corrected from the first iteration): the classifier does NOT decide
# *whether* a visual is warranted — it decides *what form* the visual takes.
# The "should we visualize at all?" decision is delegated downstream:
#   1. The LLM generator can emit `{"type": "none"}` when it can't build a
#      useful spec from the memo content.
#   2. The validator drops malformed/thin specs.
#
# Why: heuristic gating was too conservative. Reflective and philosophical
# memos still have rich visual structure (contrast, reframing, process
# flow) — but never trigger keyword heuristics. Defaulting to "try a visual,
# trust the generator + validator to back off when it can't" gives the Map
# Room a fair chance at every substantive memo. The cost is one extra LLM
# call (~3-5s) which is acceptable inside the opinion phase (5-13 min).

_COMPARE_HINTS = (
    " or ", " vs ", " versus ", "which one", "which of",
    "should i pick", "should i choose", "trade-off", "tradeoff",
    "compare", "comparing", "pros and cons", "either", "between",
)

# Signal that a memo carries TENSION structure — two opposing forces
# pulling against each other. Distinguished from "comparison" because
# tension is about *frame conflict*, not option selection. Examples:
# "you're not waiting, you're permitting", "two versions of you",
# "the passive frame vs the active frame".
_TENSION_HINTS = (
    "tension", "contradiction", "two versions", "two sides",
    "not x but y", "not waiting", "you're not", "you are not",
    " but you ", " yet you ", " however ", "underneath",
    "passive frame", "active frame", "frame underneath",
)

# Signal that a memo discusses many ENTITIES with named relationships
# — the right shape for a KNOWLEDGE-GRAPH (Cytoscape). This is the
# heaviest visual we offer and fires only when the entity count is
# high enough to justify the load.
_KNOWLEDGE_GRAPH_HINTS = (
    "stakeholder", "stakeholders", "ecosystem", "actors",
    "concept map", "knowledge graph", "entity", "entities",
    "who's involved", "who is involved", "all the players",
    "constellation", "web of", "network of",
    "relationship between", "connected to", "linked to",
    "the cast", "the people involved",
)


# Signal that a memo's structure is rich enough to benefit from a
# FLOW-GRAPH (React Flow) — interactive, drag-zoom-able node-link
# diagram. Wins over mermaid when:
#   - the memo describes a multi-step system or dependency chain
#   - 4+ reasoning items suggest the graph will have 6+ nodes
#   - feedback/loop/system language is present
_FLOW_GRAPH_HINTS = (
    "depends on", "dependency", "dependencies", "requires",
    "feedback loop", "feedback", " loop ", "loops",
    "downstream", "upstream", "cascade", "cascading",
    "chain reaction", "ripple", "ripple effect",
    "system map", "system diagram", "ecosystem",
    "interconnected", "interlocking", "interplay",
    "first-order", "second-order", "knock-on",
)


# Signal that a memo would benefit from a QUADRANT (2×2) view —
# a strategic frame where each option occupies a position on two
# orthogonal axes (effort × impact, risk × reward, urgency × importance).
_QUADRANT_HINTS = (
    "quadrant", "2x2", "2×2", "two-by-two", "matrix",
    "effort vs impact", "effort vs reward", "risk vs reward",
    "risk and reward", "high effort", "low effort",
    "high impact", "low impact", "high risk", "low risk",
    "urgency", "important", "eisenhower",
    "easy wins", "quick wins", "must-do", "should-do",
    "high-leverage", "low-leverage", "leverage",
)


# Signal that a memo carries TIMELINE / temporal-sequence structure.
# A timeline diagram fits when events are dated, ordered in time, or
# tied to specific milestones (Q1, week 1, month 6, "by July", etc.).
_TIMELINE_HINTS = (
    "timeline", "milestone", "schedule", "roadmap", "phase ",
    "phases", "stage 1", "stage 2", "first month", "first week",
    "by july", "by august", "by september", "by october",
    "by november", "by december", " q1", " q2", " q3", " q4",
    "month 1", "month 2", "month 3", "month 6", "month 12",
    "week 1", "week 2", "week 3", "week 4",
    "day one", "day 1", "in 30 days", "in 60 days", "in 90 days",
    "next quarter", "this quarter",
)


def _memo_is_substantive(memo: dict) -> bool:
    """Cheap floor: does the memo carry enough material to visualize?"""
    verdict_body = (memo.get("verdict_body") or "").strip()
    reasoning    = memo.get("reasoning") or []
    alternatives = memo.get("alternatives") or []
    # Empty memo (route drift, engine no-op) or single-line acknowledgement
    # → no visual. Otherwise yes.
    if not verdict_body or len(verdict_body) < 80:
        return False
    if not reasoning and not alternatives:
        # Reasoning is the load-bearing field; without it there's no
        # structure to translate into a diagram.
        return False
    return True


def _detect_mermaid_pattern(memo: dict, question: str) -> str | None:
    """Pick the most apt mermaid pattern from memo + question.

    Returns one of the pattern tags from MermaidSpec.pattern, or None
    when there's no clear signal. The pattern hint is passed to the
    generator prompt so the LLM builds a structurally-appropriate
    diagram, AND attached to the emitted spec so the frontend can label
    it (e.g. "TENSION" badge instead of generic "MERMAID").

    Scans verdict_line, verdict_body, reasoning titles + bodies, AND
    the user's question. Tension/process/etc. language can land
    anywhere in the memo, not just the verdict.
    """
    chunks: list[str] = [
        memo.get("verdict_line") or "",
        memo.get("verdict_body") or "",
        question or "",
    ]
    for r in (memo.get("reasoning") or []):
        if isinstance(r, dict):
            chunks.append(r.get("title", ""))
            chunks.append(r.get("body", ""))
    haystack = " ".join(chunks).lower()

    # Timeline language wins over tension when both are present —
    # temporal structure usually dominates conceptual contrast in the
    # memo's visual shape.
    if any(w in haystack for w in _TIMELINE_HINTS):
        return "timeline"
    if any(w in haystack for w in _TENSION_HINTS):
        return "tension"
    # More patterns added in later steps (process, cause-effect, etc.).
    return None


def _knowledge_graph_signal_present(memo: dict, question: str) -> bool:
    """True when the memo discusses many entities with relationships.

    The bar is intentionally high — knowledge-graph is the heaviest
    visual we offer (force-directed layout, 30-100 nodes possible).
    Only fires when explicit entity/network language is present.
    """
    chunks: list[str] = [
        memo.get("verdict_line") or "",
        memo.get("verdict_body") or "",
        question or "",
    ]
    for r in (memo.get("reasoning") or []):
        if isinstance(r, dict):
            chunks.append(r.get("title", ""))
            chunks.append(r.get("body", ""))
    haystack = " ".join(chunks).lower()
    return any(w in haystack for w in _KNOWLEDGE_GRAPH_HINTS)


def _flow_graph_signal_present(memo: dict, question: str) -> bool:
    """True when memo's structure suggests a multi-node interactive graph.

    Two paths trigger this:
      (a) Explicit system/dependency/loop language in memo or question
      (b) Many reasoning items (4+) — the graph will have enough nodes
          that the interactive renderer (drag/zoom/click) pays for itself
    """
    reasoning = memo.get("reasoning") or []
    if len(reasoning) >= 4:
        return True
    chunks: list[str] = [
        memo.get("verdict_line") or "",
        memo.get("verdict_body") or "",
        question or "",
    ]
    for r in reasoning:
        if isinstance(r, dict):
            chunks.append(r.get("title", ""))
            chunks.append(r.get("body", ""))
    haystack = " ".join(chunks).lower()
    return any(w in haystack for w in _FLOW_GRAPH_HINTS)


def _quadrant_signal_present(memo: dict, question: str) -> bool:
    """True when memo or question carries quadrant-shaped reasoning.

    Examples:
      - "easy wins vs high-effort projects"
      - "urgency × importance"
      - "rank these options by effort and impact"
    """
    chunks: list[str] = [
        memo.get("verdict_line") or "",
        memo.get("verdict_body") or "",
        question or "",
    ]
    for r in (memo.get("reasoning") or []):
        if isinstance(r, dict):
            chunks.append(r.get("title", ""))
            chunks.append(r.get("body", ""))
    haystack = " ".join(chunks).lower()
    return any(w in haystack for w in _QUADRANT_HINTS)


def classify_visual_intent(memo: dict | None, question: str) -> list[str]:
    """Pick which visual FORM(S) to attempt from memo content.

    Returns intent tags drawn from:
      - "comparison-table" : the memo presents multiple named alternatives,
                              OR the question asks the user to weigh options
      - "mermaid"          : default form for everything else substantive
                              (reasoning chains, reframing, process flow,
                              cause-effect, conditional outcomes)

    Returns [] only when the memo is genuinely thin (no verdict body or no
    reasoning) — at that point there's nothing structural to visualize.

    The downstream LLM generator can still emit `{"type": "none"}` if the
    selected form doesn't fit the actual content; the validator strips
    malformed specs. So this classifier is permissive by design — its job
    is form selection, not gatekeeping.
    """
    if not isinstance(memo, dict):
        return []
    if not _memo_is_substantive(memo):
        return []

    alternatives = memo.get("alternatives") or []
    qlow         = (question or "").lower()

    intents: list[str] = []

    # Comparison-table fires when there's something to put in columns:
    #   - 2+ named alternatives (engine produced explicit options), OR
    #   - question contains comparison cues AND any alternative exists
    has_named_alts = (
        sum(1 for a in alternatives if isinstance(a, dict) and a.get("tag")) >= 2
    )
    has_compare_cue = any(w in qlow for w in _COMPARE_HINTS)

    # Decision tree for primary "compare options" intent:
    #   3+ alternatives + quadrant language  → quadrant (positional)
    #   3+ alternatives + no quadrant cue    → score-chart (bar chart of
    #                                            scored dimensions)
    #   2+ alternatives                       → comparison-table (textual)
    # Each form is "the right weapon" for a different shape of compare.
    if len(alternatives) >= 3 and _quadrant_signal_present(memo, question):
        intents.append("quadrant")
    elif len(alternatives) >= 3:
        intents.append("score-chart")
    elif has_named_alts or (has_compare_cue and alternatives):
        intents.append("comparison-table")

    # Pick the "structural diagram" form, escalating by memo complexity:
    #   knowledge-graph → many entities + relationships (heaviest)
    #   flow-graph      → system / dependency / 4+ reasoning items
    #   mermaid         → default (lightest, 3-7 nodes)
    if _knowledge_graph_signal_present(memo, question):
        intents.append("knowledge-graph")
    elif _flow_graph_signal_present(memo, question):
        intents.append("flow-graph")
    else:
        intents.append("mermaid")

    # De-duplicate while preserving order; cap at MAX_VISUALS.
    seen: set[str] = set()
    deduped: list[str] = []
    for x in intents:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped[:_MAX_VISUALS]


# ---------------------------------------------------------------------------
# Spec generator — one LLM call, dedicated, focused
# ---------------------------------------------------------------------------

_VISUAL_GENERATOR_PROMPT = """You build ONE structured visual spec for a Map Room view. Your only job is to translate an existing analytical memo into a clean diagram or table. You do NOT invent claims, you do NOT analyze, you do NOT add reasoning. You ONLY restructure what the memo already says into a visual form.

Hard rules:
- Output a SINGLE valid JSON object. No prose preamble, no markdown fences.
- Use only the content the memo already provides. Do not introduce new facts.
- Titles are short, mono-caps friendly (3-6 words), no period.
- If you cannot build a useful visual from the memo's content, emit:
    { "type": "none", "reason": "<one short sentence>" }

The caller will tell you the visual TYPE to emit. Stick to it.

═══════════════════════════════════════════════════════════════════
TYPE 1: comparison-table

Used when the memo presents 2-4 named alternatives. Each row is a
comparison dimension (cost, risk, time, who runs it, what compounds).
The recommended column is marked via 0-indexed `recommended_column`.

Schema:
  {
    "type": "comparison-table",
    "title": "<3-6 word title>",
    "columns": ["<option A>", "<option B>", "<option C>"],
    "recommended_column": <0-indexed int, or -1>,
    "rows": [
      { "label": "<dimension>", "cells": ["<text>", "<text>", "<text>"] },
      ...
    ]
  }

Rules:
- 2-4 columns. 3-5 rows.
- Cells are PHRASES, not sentences (max ~12 words).
- Pick dimensions from what the memo actually evaluated. Common rows:
  Compounds?, Risk if X fails, Who runs it?, Time to first signal,
  Cost shape, Reversibility.
- recommended_column must match the memo's verdict if the memo picked one.
- Do NOT pad with weak rows just to fill space.

═══════════════════════════════════════════════════════════════════
TYPE 2: mermaid

The most flexible form. Use it for any of these structural patterns in
the memo:

  - Decision branches              ("if X happens then Y; if not then Z")
  - Sequenced steps / process flow ("first A, then B, then C")
  - Dependency chains              ("X requires Y which depends on Z")
  - Cause-effect maps              ("X drives Y which produces Z")
  - Reframing diagrams             (memo contrasts a passive frame with
                                    an active one, or "before" vs "after",
                                    or "what they think" vs "what's real")
  - Tension diagrams               (memo identifies two forces pulling
                                    against each other and a resolution)

The spec is a Mermaid v10 string (`graph TD` for top-down, `graph LR`
for left-right). Keep nodes terse (≤5 words).

Schema:
  {
    "type": "mermaid",
    "title": "<3-6 word title>",
    "spec": "graph TD\\n  A[<node>] --> B[<node>]\\n  ...",
    "pattern": "<optional: tension | timeline | process | decision-tree | cause-effect | reframing>"
  }

When the user-side instructions ask for a SPECIFIC pattern (look at
the "pattern_hint" field in the input), set `pattern` to that exact
string in your output. The Map Room renders the pattern as a badge so
the user can see which structural form the diagram takes.

Rules:
- 3-7 nodes. More than 7 becomes unreadable.
- Node labels: ≤5 words, no punctuation inside brackets.
- Use `-->` for direction. Use `-.->`for "maybe / weak". Use labels on
  edges only when the condition matters: `A -->|if July flops| B`.
- Top-down for decisions / hierarchies / reframings → `graph TD`.
  Left-to-right for sequences / processes / cause-chains → `graph LR`.
- Do NOT use mermaid subgraphs, styling, or click handlers.

═══════════════════════════════════════════════════════════════════
TENSION PATTERN (a specific mermaid shape)

When pattern_hint = "tension", the memo identifies TWO opposing forces.
Your diagram must:
  - Place the two forces SIDE-BY-SIDE at the top (`graph TD`).
  - Show each force flowing DOWN into its own consequence chain.
  - Connect the consequence streams with a synthesis / resolution node
    at the bottom, OR with a labeled diagonal edge showing how one
    transforms into the other.
  - Use SHORT, contrasting node labels (5 words max each).

Tension example (the passive → active reframe from a reflective memo):

  graph TD
    A[Waiting for inspiration] --> B[Inspiration arrives]
    B --> C[You act]
    D[Treat uncertainty as method] --> E[Design conditions]
    E --> F[Inspiration emerges]
    A -.->|outsources control| D

(Skip styling — the renderer handles theming. The `-.->` edge with a
label is the canonical "transformation" indicator for tension.)

Reframing example (general — when you want both frames visible but
not as a head-to-head tension):

  graph TD
    A[Old frame] --> B[Old consequence]
    A --> C[Reframe]
    C --> D[New consequence]

═══════════════════════════════════════════════════════════════════
TIMELINE PATTERN (a specific mermaid shape — distinct syntax)

When pattern_hint = "timeline", emit a mermaid `timeline` diagram —
NOT a `graph TD`. Use this when the memo lays out events in temporal
order: phases of a plan, milestones in a roadmap, week-by-week or
month-by-month progression.

The first non-blank line MUST be the word `timeline` on its own.
Optional `title` line follows. Then each event is `<when> : <what>`.

Schema (note `spec` is still a string, just different mermaid syntax):
  {
    "type": "mermaid",
    "title": "<3-6 word title>",
    "spec": "timeline\\n    title <heading>\\n    <when> : <event>\\n    ...",
    "pattern": "timeline"
  }

Rules:
- 3-8 entries. More than 8 becomes a wall.
- `<when>` is short: "Week 1", "Month 2", "Q3", "Day 1", "By August".
- `<event>` is ≤8 words.
- Group related events under the same `<when>` by repeating the
  `<when>` label (mermaid groups them visually).

Timeline example:

  timeline
    title 90-day launch plan
    Day 1 : Ship the landing page
    Week 2 : First 10 beta users
    Week 4 : Pricing test
    Month 2 : Open paid signups
    Month 3 : First retention cohort lands

═══════════════════════════════════════════════════════════════════
TYPE 3: quadrant

A 2×2 strategic frame. Use it when the memo asks you to plot 3+
options on TWO orthogonal axes — effort × impact, risk × reward,
urgency × importance, cost × value.

Schema:
  {
    "type": "quadrant",
    "title": "<3-6 word title>",
    "x_axis": { "label": "<axis name>", "low": "<left>", "high": "<right>" },
    "y_axis": { "label": "<axis name>", "low": "<bottom>", "high": "<top>" },
    "items": [
      { "label": "<short>", "x": <0-100>, "y": <0-100>, "tag": "recommended" | "warning" | null },
      ...
    ]
  }

Rules:
- 3-6 items. Fewer than 3 → use comparison-table instead.
- `x` and `y` are percentages (0-100). 0 = far left / bottom edge,
  100 = far right / top edge. Place items thoughtfully — quadrant
  position is the WHOLE point of this visual.
- `label` ≤ 5 words.
- Mark AT MOST ONE item with tag="recommended". Mark items in the
  bad quadrant with tag="warning" (max 1).
- Axis labels are SHORT (1-3 words): "Effort", "Impact", "Risk",
  "Reward", "Urgency", "Importance".
- `low` / `high` are the axis ENDPOINT words: e.g. for an "Effort"
  axis, low="Easy", high="Hard". For "Impact", low="Low", high="High".

Quadrant example:

  {
    "type": "quadrant",
    "title": "Q3 channel bets",
    "x_axis": { "label": "Effort", "low": "Easy", "high": "Hard" },
    "y_axis": { "label": "Impact", "low": "Low", "high": "High" },
    "items": [
      { "label": "Community-led", "x": 70, "y": 65, "tag": null },
      { "label": "Paid ads", "x": 35, "y": 30, "tag": "warning" },
      { "label": "Partnerships", "x": 55, "y": 85, "tag": "recommended" },
      { "label": "Cold outbound", "x": 80, "y": 20, "tag": null }
    ]
  }

═══════════════════════════════════════════════════════════════════
TYPE 4: score-chart (emits a vega-lite spec)

When the requested visual_type is "score-chart", produce a Vega-Lite
bar chart that scores each alternative across 2-4 evaluation
dimensions. Use this when the memo has 3+ named alternatives and the
question is "rank these" / "score them" — but no real CSV data is
attached.

The output's `type` field is "vega-lite" (NOT "score-chart") — the
frontend renders vega-lite. You're producing a vega-lite spec with
scores YOU assign (1-10) based on the memo's reasoning.

Schema:
  {
    "type": "vega-lite",
    "title": "<3-6 word title>",
    "spec": {
      "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
      "data": { "values": [
        { "option": "<name>", "dimension": "<axis>", "score": <1-10> },
        ...
      ] },
      "mark": "bar",
      "encoding": {
        "x":       { "field": "option",    "type": "nominal",      "axis": { "labelAngle": 0, "title": null } },
        "y":       { "field": "score",     "type": "quantitative", "scale": { "domain": [0, 10] }, "title": "Score" },
        "color":   { "field": "dimension", "type": "nominal" },
        "xOffset": { "field": "dimension" }
      }
    }
  }

Rules:
- 3-5 options, 2-4 dimensions. data.values has options × dimensions
  rows (e.g. 4 options × 3 dimensions = 12 rows).
- Scores are 1-10 integers. Be HONEST — if the memo strongly favors
  one option on a dimension, give it 8-10; weak options on that
  dimension should be 2-4.
- Dimensions should match the memo's reasoning: pick the 2-4 axes the
  memo actually evaluated against (e.g. "Compounding", "Risk",
  "Speed", "Cost").
- The xOffset encoding produces grouped bars — one cluster per option,
  one colored bar per dimension. Cleaner than stacked.
- Do NOT include `data.url`. Inline data.values only.

═══════════════════════════════════════════════════════════════════
TYPE 5: flow-graph

Interactive node-link diagram. Use it when the memo describes a
system, dependency chain, feedback loop, or has 4+ reasoning items
that interconnect — anything where the user benefits from being
able to drag nodes, zoom, and trace paths.

Mermaid is lighter weight for 3-5 nodes; flow-graph wins at 6+ nodes
or when nodes have semantic kinds (decision vs outcome vs claim).

Schema:
  {
    "type": "flow-graph",
    "title": "<3-6 word title>",
    "direction": "TB" | "LR",
    "nodes": [
      { "id": "<unique>", "label": "<≤6 words>", "kind": "decision" | "outcome" | "claim" | "default" },
      ...
    ],
    "edges": [
      { "source": "<id>", "target": "<id>", "label": "<optional ≤4 words>", "weight": "strong" | "weak" },
      ...
    ]
  }

Rules:
- 4-12 nodes. Fewer than 4 → use mermaid. More than 12 → trim or
  collapse related nodes.
- 4-20 edges. Make sure every node has at least one edge.
- `id` is a short slug (alphanumeric, no spaces) — used to reference
  nodes from edges. Keep ids stable.
- `kind` colors / shapes the node:
    decision → bordered (a question/branch point)
    outcome  → solid background (a result)
    claim    → dashed border (a load-bearing claim that could be false)
    default  → plain (intermediate state, neutral)
- `direction`: "TB" top-down for decision trees, "LR" left-right for
  process flows / dependency chains.
- `weight`: "weak" → dashed edge for "maybe / contingent on", "strong"
  → solid for "definitely".

Flow-graph example (system dependencies):

  {
    "type": "flow-graph",
    "title": "Q3 distribution dependencies",
    "direction": "TB",
    "nodes": [
      { "id": "july",     "label": "July launch lands",    "kind": "decision" },
      { "id": "signal",   "label": "First-week metrics",   "kind": "default" },
      { "id": "paid_yes", "label": "Paid is viable",       "kind": "outcome" },
      { "id": "paid_no",  "label": "Wrong offer",          "kind": "claim" },
      { "id": "community","label": "Community kicks in",   "kind": "outcome" },
      { "id": "partner",  "label": "Partnership wedge",    "kind": "outcome" }
    ],
    "edges": [
      { "source": "july",     "target": "signal",    "weight": "strong" },
      { "source": "signal",   "target": "paid_yes",  "label": "good",    "weight": "strong" },
      { "source": "signal",   "target": "paid_no",   "label": "flat",    "weight": "weak" },
      { "source": "paid_no",  "target": "community", "weight": "strong" },
      { "source": "paid_yes", "target": "partner",   "weight": "weak" }
    ]
  }

═══════════════════════════════════════════════════════════════════
TYPE 6: knowledge-graph

Force-directed network of named ENTITIES with typed relationships.
Use when the memo describes a constellation of people, concepts,
systems, or actors and their connections. Heavier than flow-graph —
only fire when there are 5+ distinct entities AND named relationships
between them.

Schema:
  {
    "type": "knowledge-graph",
    "title": "<3-6 word title>",
    "layout": "cose" | "concentric" | "breadthfirst" | "grid",
    "nodes": [
      { "id": "<unique slug>", "label": "<entity name>", "kind": "<type>" },
      ...
    ],
    "edges": [
      { "source": "<id>", "target": "<id>", "relation": "<verb phrase>", "label": "<optional ≤4 words>" },
      ...
    ]
  }

Rules:
- 5-25 nodes. Fewer → use flow-graph. More than 25 → trim.
- 4-40 edges. Every node should have ≥1 edge (no orphans).
- `id` is a slug, `label` is the displayed name.
- `kind` is the entity TYPE — short slug, free-form but the renderer
  styles these consistently: "person", "concept", "decision", "claim",
  "entity", "system", "tool", "outcome". Unknown kinds default to
  neutral.
- `relation` is a SHORT verb phrase: "depends on", "blocks",
  "informs", "owns", "supersedes", "competes with". This is the
  edge's identity.
- `layout`: "cose" (default, force-directed) works for most graphs;
  "concentric" if one node is clearly central; "breadthfirst" for
  hierarchies; "grid" only for regular structures.

Knowledge-graph example (Q3 strategy ecosystem):

  {
    "type": "knowledge-graph",
    "title": "Q3 strategic ecosystem",
    "layout": "cose",
    "nodes": [
      { "id": "you",       "label": "You",                "kind": "person" },
      { "id": "cofounder", "label": "Co-founder",         "kind": "person" },
      { "id": "july",      "label": "July launch",        "kind": "decision" },
      { "id": "runway",    "label": "9-month runway",     "kind": "claim" },
      { "id": "paid",      "label": "Paid acquisition",   "kind": "outcome" },
      { "id": "comm",      "label": "Community",          "kind": "outcome" },
      { "id": "warm",      "label": "Warm partnership",   "kind": "entity" }
    ],
    "edges": [
      { "source": "you",       "target": "july",   "relation": "owns" },
      { "source": "cofounder", "target": "july",   "relation": "owns" },
      { "source": "runway",    "target": "paid",   "relation": "constrains" },
      { "source": "runway",    "target": "comm",   "relation": "favors" },
      { "source": "warm",      "target": "paid",   "relation": "supersedes" },
      { "source": "july",      "target": "warm",   "relation": "enables" }
    ]
  }

═══════════════════════════════════════════════════════════════════

If the memo doesn't have enough material to build a useful visual of the
requested TYPE, emit `{ "type": "none", "reason": "..." }` — better to
skip the visual than ship a thin one.
"""


def _build_user_message(
    memo: dict,
    question: str,
    visual_type: str,
    pattern_hint: str | None = None,
) -> str:
    """Compact, JSON-shaped context for the visualizer call.

    `pattern_hint` (optional) names the structural form to use, e.g.
    "tension", "timeline". The generator prompt has dedicated sections
    for each named pattern; when the hint is set, the model is told to
    use that pattern AND echo it back in the spec's `pattern` field
    so the renderer can label it with the correct badge.
    """
    payload: dict[str, Any] = {
        "user_question":   question,
        "visual_type_to_build": visual_type,
        "memo": {
            "verdict_line":   memo.get("verdict_line", ""),
            "verdict_body":   memo.get("verdict_body", ""),
            "reasoning":      memo.get("reasoning", []),
            "alternatives":   memo.get("alternatives", []),
            "falsifiers":     memo.get("falsifiers", []),
        },
    }
    if pattern_hint:
        payload["pattern_hint"] = pattern_hint
    return (
        "Build ONE visual spec of the requested type from the memo below.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def _parse_visual_json(content: str) -> dict | None:
    """Extract a visual-spec JSON object from raw LLM output.

    Mirrors the brace-slice strategy used by speech._extract_memo_json
    but accepts any dict (we validate type/shape separately).
    """
    if not content:
        return None

    candidates: list[str] = [content.strip()]

    fence = _FENCE_RE.match(content.strip())
    if fence:
        candidates.append(fence.group(1).strip())

    first = content.find("{")
    last  = content.rfind("}")
    if first != -1 and last > first:
        candidates.append(content[first : last + 1].strip())

    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


async def generate_visual_spec(
    *,
    client: LLMClient,
    memo: dict,
    question: str,
    visual_type: str,
    pattern_hint: str | None = None,
) -> dict | None:
    """Run the dedicated visualizer LLM call for ONE visual type.

    `pattern_hint` (optional) — when the classifier identified a
    specific mermaid PATTERN (tension / timeline / process / etc.),
    pass it here so the prompt steers the generator toward that
    structural shape AND so the spec carries the pattern field back
    to the renderer for badge labeling.

    Returns the parsed (un-validated) spec dict, or None when the model
    declined (emitted `{"type": "none"}`), the call failed, or no JSON
    could be extracted. The validator runs separately on the caller side.
    """
    if visual_type not in (
        "comparison-table", "mermaid", "quadrant",
        "score-chart", "flow-graph", "knowledge-graph",
    ):
        return None

    user_msg = _build_user_message(memo, question, visual_type, pattern_hint=pattern_hint)

    try:
        response = await client.call(
            system_prompt=_VISUAL_GENERATOR_PROMPT,
            user_message=user_msg,
            domain=_DOMAIN,
            concept=_CONCEPT,
            temperature=0.3,   # low — we want structural compliance, not creativity
            max_tokens=1024,
        )
    except Exception:
        log.exception("visualizer: LLM call raised for type=%s", visual_type)
        return None

    if not response.success:
        log.warning("visualizer: call failed for type=%s", visual_type)
        return None

    parsed = _parse_visual_json(response.content)
    if parsed is None:
        log.warning(
            "visualizer: no JSON in response for type=%s (len=%d)",
            visual_type, len(response.content),
        )
        return None

    # Model declined — emit nothing.
    if parsed.get("type") == "none":
        log.info(
            "visualizer: model declined type=%s (reason=%s)",
            visual_type, parsed.get("reason", "")[:80],
        )
        return None

    return parsed


# ---------------------------------------------------------------------------
# Validator — strict structural checks, no LLM
# ---------------------------------------------------------------------------


def validate_visual_spec(spec: Any) -> dict | None:
    """Validate a visual spec; return a clean copy or None if unfixable.

    Codex's third pipeline stage: "Validator — rejects broken specs."
    Catches all the ways an LLM can drift off the schema without our
    noticing: missing required fields, wrong primitive types, oversized
    arrays, recommended_column out of range, etc.

    Returns:
      - dict — the validated spec (may have minor coercions, e.g. -1 → omitted)
      - None — spec is unfixable (caller drops it)
    """
    if not isinstance(spec, dict):
        return None
    visual_type = spec.get("type")

    if visual_type == "comparison-table":
        return _validate_comparison_table(spec)
    if visual_type == "mermaid":
        return _validate_mermaid(spec)
    if visual_type == "quadrant":
        return _validate_quadrant(spec)
    if visual_type == "vega-lite":
        return _validate_vega_lite(spec)
    if visual_type == "flow-graph":
        return _validate_flow_graph(spec)
    if visual_type == "knowledge-graph":
        return _validate_knowledge_graph(spec)
    return None


def _validate_comparison_table(spec: dict) -> dict | None:
    title = spec.get("title")
    columns = spec.get("columns")
    rows    = spec.get("rows")
    rec     = spec.get("recommended_column")

    if not isinstance(columns, list) or not (2 <= len(columns) <= 4):
        return None
    if not all(isinstance(c, str) and c.strip() for c in columns):
        return None

    if not isinstance(rows, list) or not (1 <= len(rows) <= 8):
        return None

    cleaned_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("label")
        cells = row.get("cells")
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(cells, list) or len(cells) != len(columns):
            continue
        if not all(isinstance(c, (str, int, float)) for c in cells):
            continue
        cleaned_rows.append({
            "label": label.strip(),
            "cells": [str(c).strip() for c in cells],
        })

    if not cleaned_rows:
        return None

    out: dict[str, Any] = {
        "type":    "comparison-table",
        "columns": [c.strip() for c in columns],
        "rows":    cleaned_rows,
    }
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()

    if isinstance(rec, int) and 0 <= rec < len(columns):
        out["recommended_column"] = rec
    return out


_VALID_MERMAID_PATTERNS = {
    "tension", "timeline", "process", "decision-tree",
    "cause-effect", "reframing",
}


def _validate_vega_lite(spec: dict) -> dict | None:
    """Validate a Vega-Lite spec (used by score-chart and any future
    CSV-backed chart).

    We do NOT try to fully validate Vega-Lite — vega-embed handles
    that on the frontend with inline error messages. We DO enforce:
      - spec.spec is an object (not a string)
      - data.values is inline (no data.url, no external fetch)
      - data.values has at least one row
      - the spec doesn't try to load remote anything
    """
    title = spec.get("title")
    inner = spec.get("spec")
    if not isinstance(inner, dict):
        return None

    data = inner.get("data")
    if not isinstance(data, dict):
        return None

    values = data.get("values")
    if not isinstance(values, list) or not values:
        return None
    # Cap inline data — 200 rows is what speech.py already limits.
    if len(values) > 200:
        return None

    # Reject remote-fetch attempts.
    if "url" in data:
        return None

    # mark must be present (string or object).
    mark = inner.get("mark")
    if not (isinstance(mark, str) or isinstance(mark, dict)):
        return None

    out: dict[str, Any] = {
        "type": "vega-lite",
        "spec": inner,
    }
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()
    return out


def _validate_knowledge_graph(spec: dict) -> dict | None:
    """Shape check for a knowledge-graph spec.

    Required: 3-40 nodes, 2-80 edges, layout in
    {cose, concentric, breadthfirst, grid} or omitted (defaults cose).
    Both nodes and edges have optional kind/relation fields.
    """
    title  = spec.get("title")
    nodes_raw = spec.get("nodes")
    edges_raw = spec.get("edges")
    layout = spec.get("layout")

    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        return None
    if not (3 <= len(nodes_raw) <= 40):
        return None
    if not (2 <= len(edges_raw) <= 80):
        return None

    clean_nodes: list[dict] = []
    seen_ids: set[str] = set()
    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        lbl = n.get("label")
        if not isinstance(nid, str) or not nid.strip():
            continue
        if not isinstance(lbl, str) or not lbl.strip():
            continue
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        node: dict[str, Any] = {
            "id":    nid.strip(),
            "label": lbl.strip(),
        }
        if isinstance(n.get("kind"), str) and n["kind"].strip():
            node["kind"] = n["kind"].strip()[:30]  # cap free-form slug length
        clean_nodes.append(node)

    if len(clean_nodes) < 3:
        return None

    known_ids = {n["id"] for n in clean_nodes}
    clean_edges: list[dict] = []
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        tgt = e.get("target")
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        if src not in known_ids or tgt not in known_ids:
            continue
        edge: dict[str, Any] = {
            "source": src.strip(),
            "target": tgt.strip(),
        }
        if isinstance(e.get("relation"), str) and e["relation"].strip():
            edge["relation"] = e["relation"].strip()[:30]
        if isinstance(e.get("label"), str) and e["label"].strip():
            edge["label"] = e["label"].strip()
        clean_edges.append(edge)

    if len(clean_edges) < 2:
        return None

    out: dict[str, Any] = {
        "type":  "knowledge-graph",
        "nodes": clean_nodes,
        "edges": clean_edges,
    }
    if layout in ("cose", "concentric", "breadthfirst", "grid"):
        out["layout"] = layout
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()
    return out


def _validate_flow_graph(spec: dict) -> dict | None:
    """Shape check for a flow-graph (React Flow) spec.

    Required: 2-15 nodes (each with `id` + `label`), 1-25 edges
    referencing existing node ids, optional direction in {TB, LR}.
    Drops malformed nodes/edges silently, but rejects the whole spec
    when fewer than 2 valid nodes or 0 valid edges remain.
    """
    title = spec.get("title")
    nodes_raw = spec.get("nodes")
    edges_raw = spec.get("edges")
    direction = spec.get("direction")

    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        return None
    if not (2 <= len(nodes_raw) <= 15):
        return None
    if not (1 <= len(edges_raw) <= 25):
        return None

    valid_kinds = {"decision", "outcome", "claim", "default"}
    clean_nodes: list[dict] = []
    seen_ids: set[str] = set()
    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        lbl = n.get("label")
        if not isinstance(nid, str) or not nid.strip():
            continue
        if not isinstance(lbl, str) or not lbl.strip():
            continue
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        kind = n.get("kind")
        if not isinstance(kind, str) or kind not in valid_kinds:
            kind = "default"
        clean_nodes.append({
            "id":    nid.strip(),
            "label": lbl.strip(),
            "kind":  kind,
        })

    if len(clean_nodes) < 2:
        return None

    known_ids = {n["id"] for n in clean_nodes}
    clean_edges: list[dict] = []
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        src = e.get("source")
        tgt = e.get("target")
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        if src not in known_ids or tgt not in known_ids:
            continue
        edge: dict[str, Any] = {
            "source": src.strip(),
            "target": tgt.strip(),
        }
        if isinstance(e.get("label"), str) and e["label"].strip():
            edge["label"] = e["label"].strip()
        w = e.get("weight")
        if w in ("strong", "weak"):
            edge["weight"] = w
        clean_edges.append(edge)

    if not clean_edges:
        return None

    out: dict[str, Any] = {
        "type":  "flow-graph",
        "nodes": clean_nodes,
        "edges": clean_edges,
    }
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()
    if direction in ("TB", "LR"):
        out["direction"] = direction
    return out


def _validate_quadrant(spec: dict) -> dict | None:
    """Strict shape check for 2×2 quadrant specs.

    Required: x_axis + y_axis (each with label/low/high), 3-6 items
    with label + numeric x/y in [0, 100]. Tag is optional, restricted
    to "recommended" / "warning" / null.
    """
    title  = spec.get("title")
    xa     = spec.get("x_axis")
    ya     = spec.get("y_axis")
    items  = spec.get("items")

    if not isinstance(xa, dict) or not isinstance(ya, dict):
        return None

    def _axis_ok(a: dict) -> bool:
        for k in ("label", "low", "high"):
            v = a.get(k)
            if not isinstance(v, str) or not v.strip():
                return False
        return True

    if not _axis_ok(xa) or not _axis_ok(ya):
        return None

    if not isinstance(items, list) or not (3 <= len(items) <= 6):
        return None

    clean_items: list[dict] = []
    rec_count = 0
    warn_count = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        label = it.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        x = it.get("x")
        y = it.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        # Coerce ints/floats; clamp to [0, 100].
        x = max(0.0, min(100.0, float(x)))
        y = max(0.0, min(100.0, float(y)))
        tag = it.get("tag")
        if tag not in ("recommended", "warning", None):
            tag = None
        # Enforce at-most-one of each tag type.
        if tag == "recommended":
            if rec_count >= 1:
                tag = None
            else:
                rec_count += 1
        elif tag == "warning":
            if warn_count >= 1:
                tag = None
            else:
                warn_count += 1
        clean_items.append({
            "label": label.strip(),
            "x":     x,
            "y":     y,
            "tag":   tag,
        })

    if len(clean_items) < 3:
        return None

    out: dict[str, Any] = {
        "type":   "quadrant",
        "x_axis": {
            "label": xa["label"].strip(),
            "low":   xa["low"].strip(),
            "high":  xa["high"].strip(),
        },
        "y_axis": {
            "label": ya["label"].strip(),
            "low":   ya["low"].strip(),
            "high":  ya["high"].strip(),
        },
        "items":  clean_items,
    }
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()
    return out


def _validate_mermaid(spec: dict) -> dict | None:
    title    = spec.get("title")
    mermaid  = spec.get("spec")
    pattern  = spec.get("pattern")

    if not isinstance(mermaid, str):
        return None
    mermaid = mermaid.strip()
    if not mermaid:
        return None

    # Strip a leading ```mermaid / ``` fence the model may have re-added
    # despite the prompt.
    if mermaid.startswith("```"):
        m = re.match(r"^```(?:mermaid)?\s*\n?(.*?)\n?```$", mermaid, re.DOTALL)
        if m:
            mermaid = m.group(1).strip()

    # Must declare a diagram type the renderer can handle. mermaid.js
    # handles a wide vocabulary out of the box; we whitelist only the
    # ones we've checked render cleanly in the Map Room shell.
    first_token = mermaid.split(None, 1)[0].lower() if mermaid else ""
    if first_token not in {"graph", "flowchart", "timeline", "gantt"}:
        return None

    # Diagram-type-specific shape checks.
    if first_token in {"graph", "flowchart"}:
        # Edge count guards against spaghetti charts.
        edges = mermaid.count("-->") + mermaid.count("-.->")
        if edges == 0 or edges > 14:
            return None
    elif first_token == "timeline":
        # Timeline entries use `:` separators between event labels. Need
        # at least 2 entries to be useful; cap at 12 to keep it readable.
        entry_count = sum(
            1 for line in mermaid.splitlines()
            if ":" in line and not line.strip().startswith("title")
        )
        if entry_count < 2 or entry_count > 12:
            return None
    elif first_token == "gantt":
        # Gantt needs at least one `section` and one task. Tasks contain
        # `:` followed by a date or duration.
        if "section" not in mermaid.lower():
            return None
        task_count = sum(
            1 for line in mermaid.splitlines()
            if ":" in line and "section" not in line.lower()
            and not line.strip().startswith("title")
            and not line.strip().startswith("dateFormat")
        )
        if task_count < 1 or task_count > 12:
            return None

    out: dict[str, Any] = {
        "type": "mermaid",
        "spec": mermaid,
    }
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()
    if isinstance(pattern, str) and pattern.strip().lower() in _VALID_MERMAID_PATTERNS:
        out["pattern"] = pattern.strip().lower()
    return out


# ---------------------------------------------------------------------------
# Pipeline driver — full classifier → generator → validator chain
# ---------------------------------------------------------------------------


async def build_visuals(
    *,
    client: LLMClient,
    memo: dict | None,
    question: str,
) -> list[dict]:
    """Run the full visual pipeline for one memo. Returns 0–2 valid specs.

    Safe to await even when the memo is empty or the question is missing —
    returns [] in those cases. Never raises (all failures degrade to []).
    """
    if not isinstance(memo, dict) or not memo:
        return []

    intents = classify_visual_intent(memo, question)
    if not intents:
        log.info("visualizer: classifier emitted no intents — skipping")
        return []

    # Detect the most apt mermaid pattern (only used when intent ==
    # "mermaid"). Other intents ignore this hint.
    mermaid_pattern = _detect_mermaid_pattern(memo, question)

    log.info(
        "visualizer: intents=%s pattern=%s for question[:60]=%r",
        intents, mermaid_pattern, (question or "")[:60],
    )

    valid: list[dict] = []
    for visual_type in intents:
        if len(valid) >= _MAX_VISUALS:
            break
        try:
            raw = await generate_visual_spec(
                client=client, memo=memo, question=question,
                visual_type=visual_type,
                pattern_hint=mermaid_pattern if visual_type == "mermaid" else None,
            )
        except Exception:
            log.exception("visualizer: generate_visual_spec raised for %s", visual_type)
            continue
        if raw is None:
            continue
        clean = validate_visual_spec(raw)
        if clean is None:
            log.warning("visualizer: validator rejected spec for %s", visual_type)
            continue
        valid.append(clean)

    log.info("visualizer: built %d/%d valid spec(s)", len(valid), len(intents))
    return valid
