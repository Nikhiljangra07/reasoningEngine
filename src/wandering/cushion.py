"""
Cushion graph — the immutable anchor that wandering agents tether against.

The user fills four fields (Problem / Context / Vision / Current Map). The
system derives a three-layer structural representation (Actual / Essence /
Mechanism) used internally for matching. Agents wandering across any domain
match discovered content against any of the three layers — partial matches
trigger exploration, not termination.

User-facing surface: 4 fields.
Internal representation: 3 layers, each with 3-8 sub-nodes.
The merge is conceptual — one workflow, two views.

Auto-enrichment: the "current map" field is enriched from project memory
(Neo4j graph) transparently. The user provides what they consciously have;
the system supplements with relevant project state.

ISOLATION: this module defines types only. It does NOT call LLMs, hit
storage, or import from other domain modules. The composer module
(src/wandering/composer.py) handles extraction; storage is downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Field-level types — the four user-facing inputs
# ---------------------------------------------------------------------------


class SkipReason(str, Enum):
    """How a field came to be empty/skipped.

    - NOT_SKIPPED: user provided content
    - SKIPPED_AFTER_PROMPT: user clicked skip, was shown follow-up + warning,
      acknowledged the cost, and skipped anyway
    - SKIPPED_NO_PROMPT: shouldn't happen in production but kept for tests
      and for the edge case where the form is bypassed (e.g., API caller
      that doesn't render the warning UI)
    """

    NOT_SKIPPED = "not_skipped"
    SKIPPED_AFTER_PROMPT = "skipped_after_prompt"
    SKIPPED_NO_PROMPT = "skipped_no_prompt"


@dataclass
class CushionField:
    """One of the four user-input fields on the brief composer form.

    `name` is the canonical field key. `content` is the user's text (may be
    empty if skipped). `skip_reason` tracks whether the user saw the warning.

    Field names map to the dimensions they capture:
      problem      → actual problem (concrete description)
      context      → system context + origin (where it sits, what brings it here)
      vision       → future trajectory (where the user is heading)
      current_map  → initial inspirations + related domains (+ auto-enriched memory)
    """

    name: str
    content: str = ""
    skip_reason: SkipReason = SkipReason.NOT_SKIPPED

    def is_filled(self) -> bool:
        """True if the user provided non-empty content."""
        return self.skip_reason == SkipReason.NOT_SKIPPED and bool(self.content.strip())

    def is_skipped(self) -> bool:
        """True if the field is empty (skipped or just blank)."""
        return not self.is_filled()


@dataclass
class CushionInput:
    """The raw four-field input from the user, plus auto-enriched memory context.

    Validation invariant: at minimum, `problem` should be filled OR all four
    skip reasons must be SKIPPED_AFTER_PROMPT. An entirely empty cushion is
    a launch-time error (the agents would have nothing to anchor against).

    `memory_enrichment` is filled in by the composer before extraction —
    the user does not type it. It's pulled from the project's memory graph
    (Neo4j: recent threads, current architecture, ongoing decisions). This
    is transparent and requires no explicit user permission (it's their
    own memory).
    """

    problem: CushionField
    context: CushionField
    vision: CushionField
    current_map: CushionField
    memory_enrichment: str = ""  # auto-filled from project memory graph

    def fields(self) -> list[CushionField]:
        """All four user-facing fields in canonical order."""
        return [self.problem, self.context, self.vision, self.current_map]

    def filled_field_count(self) -> int:
        """How many of the four fields the user actually filled."""
        return sum(1 for f in self.fields() if f.is_filled())

    def is_minimally_viable(self) -> bool:
        """True if the cushion has enough content to anchor wandering against.

        Minimal viability: at least the `problem` field must be filled.
        An anchor with no problem statement is structurally empty; agents
        would wander aimlessly. The user can skip every other field with
        warnings, but `problem` is the structural floor.
        """
        return self.problem.is_filled()


# ---------------------------------------------------------------------------
# Layer-level types — the three-layer structural representation
# ---------------------------------------------------------------------------


@dataclass
class CushionNode:
    """Dual-artifact representation of a single cushion node.

    Constellation Interpreter (2026-06-01). Each node carries TWO faces:

      GRAPH face   — what role this node plays inside the cushion (text,
                     layer; role/parent_ids/tension_ids are Phase 4 work,
                     left empty for v1).
      RETRIEVAL face — how to find related material on the internet
                     (search_queries) and how to match content fingerprints
                     in structural embedding space (embedding_text +
                     embedding vector).

    `id` is stable and deterministic — same (session, layer, text) produces
    the same id across runs, so Neo4j upserts don't duplicate. Generated
    at compose time via `make_cushion_node_id`.

    `embedding` is the 1536-dim Gemini vector of `embedding_text` (which
    defaults to `text` when the composer didn't surface a richer
    structural rephrasing). It's the primary key into
    `cushion_node_embedding_idx` for the multi-channel matcher.

    Note: CushionNode is the RICH representation. CushionLayer keeps the
    plain `nodes: list[str]` for backward compatibility with all existing
    consumers (matching, persistence, routes, tests). The full records
    live in `CushionLayer.node_records` when the composer produced them.
    """

    id:              str
    text:            str
    layer:           str                    # "actual" | "essence" | "mechanism"
    search_queries:  tuple[str, ...] = field(default_factory=tuple)
    embedding_text:  str = ""
    embedding:       list[float] | None = None

    # Graph-face fields — Phase 4 territory; defaulted for v1.
    role:            str = ""               # anchor | constraint | catalyst | tension | failure_mode
    parent_ids:      tuple[str, ...] = field(default_factory=tuple)
    tension_ids:     tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        """A CushionNode stringifies to its text. This means any old
        code path that did `for n in layer.nodes: print(n)` keeps working
        even if (in a future phase) `layer.nodes` becomes typed as
        list[CushionNode] rather than list[str]."""
        return self.text

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for persistence. Embedding is included so the
        cushion's vector representation survives a round-trip through the
        session log."""
        return {
            "id":              self.id,
            "text":            self.text,
            "layer":           self.layer,
            "search_queries":  list(self.search_queries),
            "embedding_text":  self.embedding_text,
            "embedding":       list(self.embedding) if self.embedding is not None else None,
            "role":            self.role,
            "parent_ids":      list(self.parent_ids),
            "tension_ids":     list(self.tension_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CushionNode":
        """Rehydrate from persisted dict. Tolerant of partial payloads
        (older sessions persisted before the dual-artifact upgrade have
        only `text` + `layer`)."""
        emb = payload.get("embedding")
        return cls(
            id=str(payload.get("id", "")),
            text=str(payload.get("text", "")),
            layer=str(payload.get("layer", "")),
            search_queries=tuple(payload.get("search_queries") or ()),
            embedding_text=str(payload.get("embedding_text") or ""),
            embedding=list(emb) if emb else None,
            role=str(payload.get("role") or ""),
            parent_ids=tuple(payload.get("parent_ids") or ()),
            tension_ids=tuple(payload.get("tension_ids") or ()),
        )


def make_cushion_node_id(session_id: str, layer: str, text: str) -> str:
    """Deterministic id for a CushionNode.

    Same (session_id, layer, text) tuple always produces the same id, so
    Neo4j `MERGE (n:CushionNode {id: $id})` upserts cleanly on re-compose
    (e.g., during shadow runs or replay). SHA-256 short hash keeps it
    bounded length while remaining collision-resistant within a session.
    """
    import hashlib
    raw = f"{session_id}|{layer}|{text.strip().lower()}"
    return "cn_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class CushionLayer:
    """One of the three structural layers extracted from the user's input.

    Each layer captures a different abstraction of the same problem:

      ACTUAL    → literal, concrete description (what the problem IS in
                  surface terms — entities, scope, history)
      ESSENCE   → structural-dynamic pattern (forces, tensions, constraints,
                  cycles, asymmetries — the SHAPE of the problem)
      MECHANISM → causal primitive (the abstract operating logic that,
                  applied to any domain, would produce this kind of problem)

    Each layer has 3-8 sub-nodes that form that layer's "metal detector"
    graph. Agents wandering across any domain match discovered content
    against these nodes — partial overlap on any layer triggers exploration.

    The match scoring rule:
      - actual overlap: small weight (surface entities)
      - essence overlap: large weight (structural dynamics)
      - mechanism overlap: large weight (causal primitives)
    An agent finding 0 actual + 4/5 essence + 5/5 mechanism = HIGH confidence
    cross-domain insight. That's the Heisenberg pattern.

    `nodes` is the canonical list[str] used by matcher, persistence, routes,
    and tests. `node_records` is the additive Constellation Interpreter
    upgrade (2026-06-01): when the composer produced dual-artifact records
    (with embeddings, search queries, etc.), they live here in parallel to
    `nodes`. The two lists are kept in 1:1 order so `nodes[i]` always
    corresponds to `node_records[i].text` when both are populated.
    `node_records` is None for old persisted cushions and any cushion
    composed before the upgrade.
    """

    name: str  # "actual" | "essence" | "mechanism"
    nodes: list[str] = field(default_factory=list)
    summary: str = ""  # one-paragraph human-readable description
    node_records: list[CushionNode] | None = None

    def node_count(self) -> int:
        return len(self.nodes)

    def records_or_synth(self, *, session_id: str = "") -> list[CushionNode]:
        """Return the rich records if present, else synthesize minimal
        records from `nodes` strings. Synthesized records have empty
        embeddings — useful for code paths that need a uniform iteration
        target but can't pay for live LLM/embedding calls. The new
        matcher's vector channel will skip records with embedding=None.
        """
        if self.node_records is not None:
            return self.node_records
        return [
            CushionNode(
                id=make_cushion_node_id(session_id, self.name, t),
                text=t,
                layer=self.name,
            )
            for t in self.nodes
        ]


@dataclass
class CushionGraph:
    """The full anchor — three layers derived from the four-field input.

    This is the immutable target every wandering agent matches against.
    Once constructed, it does not change for the duration of the session.
    Agents may wander; the cushion never moves.

    The graph is persisted (Neo4j) at session start and referenced by every
    agent, every report, every trace entry. It's the canonical anchor.

    `raw_input` is preserved so we can:
      1. Re-extract the cushion if the extraction prompt changes
      2. Show the user what they typed when they review the cushion
      3. Audit how the four fields became the three layers
    """

    actual: CushionLayer
    essence: CushionLayer
    mechanism: CushionLayer
    raw_input: CushionInput
    constellation_size: int = 0  # total nodes across all layers
    extraction_model: str = ""  # e.g. "claude-sonnet-4-6"
    extracted_at: float = 0.0  # unix timestamp

    # Identity-layer metadata (additive — does not gate cushion
    # construction or wandering). Populated by
    # `goal_supremacy.surface_real_goal()` in `compose_cushion` from
    # the four-field input. When the stated problem contradicts
    # signals in context/vision/current_map, this field holds a
    # rendered probe ("is X the goal, or is Y the goal underneath?")
    # the frontend may show the user before committing to the
    # wander. None means no contradiction detected. See doctrine §10
    # "discipline metadata".
    real_goal_probe: str | None = None

    def __post_init__(self) -> None:
        # Lazy total — easier than asking callers to track it.
        if not self.constellation_size:
            self.constellation_size = (
                self.actual.node_count()
                + self.essence.node_count()
                + self.mechanism.node_count()
            )

    def layers(self) -> list[CushionLayer]:
        """The three layers in canonical order."""
        return [self.actual, self.essence, self.mechanism]

    def is_well_formed(self) -> bool:
        """Sanity check: every layer has at least one node and the input
        is minimally viable. Used by the composer to validate before
        handing the cushion to wandering agents."""
        return (
            self.raw_input.is_minimally_viable()
            and self.actual.node_count() >= 1
            and self.essence.node_count() >= 1
            and self.mechanism.node_count() >= 1
        )

    def to_anchor_prompt(self) -> str:
        """Render the cushion as a system-prompt-friendly anchor block for
        wandering agents. This is what every agent sees in its system
        prompt as the fixed nail their pendulum hangs from.

        Format is intentionally compact — agents read this on every turn,
        so the token cost compounds. Per-layer summaries + node lists,
        no preamble.
        """
        lines = ["# ANCHOR (do not detach)"]
        for layer in self.layers():
            lines.append(f"\n## {layer.name.upper()} layer")
            if layer.summary:
                lines.append(layer.summary)
            if layer.nodes:
                lines.append("Nodes: " + ", ".join(layer.nodes))
        return "\n".join(lines)
