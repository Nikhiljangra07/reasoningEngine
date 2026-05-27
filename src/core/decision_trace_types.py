"""
decision_trace_types — typed event nodes that hang off each IterationRecord.

These are the structured outputs of the InlineClassifier (Phase 2b) and the
MarkdownDecisionLogExtractor (Phase 2c). They populate the Decision Trace
layer: a chronological, typed, queryable record of what actually happened
in a conversation — without throwing away the raw text.

NAMING DISTINCTION — IMPORTANT
==============================
This file's `Decision` is NOT the same as `src/bridge/types.py:DecisionAnchor`.

  Decision (here)
    A conversation-turn event tag. "In this turn the user/system made
    this decision." Cheap, classifier-extracted, lifetime tied to its
    parent Iteration. Many per session.

  DecisionAnchor (bridge/types.py)
    A project-level architectural decision record (ADR-like). Has
    evidence, code_refs, supersedes lineage, OPEN/SETTLED/DRIFTED
    lifecycle. Few per project. Durable artifact.

In Neo4j they're separate labels: `(:DecisionTrace:Decision)` vs
`(:DecisionAnchor)`. A Decision can graduate into a DecisionAnchor when
it accumulates enough evidence/weight, but they're distinct nodes.

DUAL-LABEL PATTERN
==================
Every node here gets two Neo4j labels:
  - `:DecisionTrace`  — source namespace (vs `:CodeGraph` for graphify nodes)
  - `:<Type>`         — e.g. `:Decision`, `:UserMessage`

Application code MATCH'es by source label when isolating Decision Trace from
code graph queries, and by type label when looking for specific kinds.

PROVENANCE FIELDS — DENORMALIZED ON PURPOSE
============================================
Every event carries the full address (workspace_id / surface_id / user_id /
project_id / thread_id / iteration_id) as scalar properties, not just via
graph traversal. The cost is small storage redundancy; the benefit is that
the cross-thread retriever (Phase 4) filters in one MATCH without a join.

CONFIDENCE
==========
The classifier emits a `confidence` (0.0–1.0) on every event it produces.
Per the locked architecture: we persist everything and surface low-confidence
items with a flag rather than dropping them silently. The retriever can
optionally filter `WHERE confidence >= 0.7`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal


# ─── ID helpers ──────────────────────────────────────────────────────

# Type-prefix tokens used inside generated IDs. Kept here so the namespace
# is documented in one place. If a new event type ships, add its token here.
_DT_ID_TOKENS: dict[str, str] = {
    "user_message":    "msg-user",
    "system_response": "msg-sys",
    "decision":        "decision",
    "question":        "question",
    "reference":       "reference",
    "insight":         "insight",
}


def new_dt_id(kind: str) -> str:
    """Generate a `dt-<token>-<12hex>` identifier for a Decision Trace node.

    The `dt-` prefix is the namespace marker — it separates Decision Trace
    nodes from Code Graph nodes (which use `cg-`) for retrieval clarity.
    12 hex chars = 48 bits of entropy, plenty for collision safety on a
    single-user / single-project Neo4j instance."""
    token = _DT_ID_TOKENS.get(kind)
    if token is None:
        raise ValueError(f"unknown decision-trace kind {kind!r}; expected one of {sorted(_DT_ID_TOKENS)}")
    return f"dt-{token}-{uuid.uuid4().hex[:12]}"


# ─── Type literals ───────────────────────────────────────────────────

DecisionStatus = Literal["noted", "committed", "superseded", "rejected"]
ReferenceKind  = Literal["url", "file", "mcp_resource", "memory_id", "code_symbol", "other"]


# ─── Provenance fields (every event carries these) ───────────────────
#
# A free function builds the provenance kwargs; each dataclass uses it via
# field(default_factory=...) is overkill — the production path passes the
# values explicitly. We just document the address shape here.
#
# Required on every event:
#   id, iteration_id, thread_id, workspace_id, surface_id, user_id, ts
# Optional:
#   project_id (None when iteration isn't project-scoped),
#   confidence (defaults to 1.0 for human-authored events that don't go
#               through the classifier; classifier-produced events override).


# ─── Event types ─────────────────────────────────────────────────────

@dataclass
class UserMessage:
    """The user's text for a single turn. Persisted verbatim — the source
    of truth for what was said. The InlineClassifier reads this (plus
    SystemResponse) to derive Decision/Question/etc. events."""
    id:            str
    iteration_id:  str
    thread_id:     str
    workspace_id:  str
    surface_id:    str
    user_id:       str
    text:          str
    ts:            float
    project_id:    str | None = None


@dataclass
class SystemResponse:
    """The system's text for a single turn. Same role as UserMessage —
    verbatim, source of truth for what the assistant said."""
    id:            str
    iteration_id:  str
    thread_id:     str
    workspace_id:  str
    surface_id:    str
    user_id:       str
    text:          str
    ts:            float
    project_id:    str | None = None


@dataclass
class Decision:
    """A turn-level decision the user or system committed to.

    NOT the same as bridge/types.py:DecisionAnchor — this is the cheap,
    classifier-extracted, conversation-event variant. A Decision here
    may later graduate into a full DecisionAnchor; we don't auto-promote
    in Phase 2."""
    id:            str
    iteration_id:  str
    thread_id:     str
    workspace_id:  str
    surface_id:    str
    user_id:       str
    text:          str
    ts:            float
    status:        DecisionStatus = "noted"
    confidence:    float = 1.0
    project_id:    str | None = None
    # When the classifier flags this as superseding an earlier decision,
    # `supersedes` carries the older Decision.id. Used to build the
    # `[:SUPERSEDES]` edge in the writer.
    supersedes:    str | None = None


@dataclass
class Question:
    """An open question raised in the turn (by user or system).

    `resolved` flips to True when a later turn's Decision answers it.
    Phase 2 just records open questions; resolution-linking is wired
    later via Decision-[:RESOLVES]->Question edges."""
    id:            str
    iteration_id:  str
    thread_id:     str
    workspace_id:  str
    surface_id:    str
    user_id:       str
    text:          str
    ts:            float
    resolved:      bool = False
    confidence:    float = 1.0
    project_id:    str | None = None


@dataclass
class Reference:
    """A citation — URL, file, MCP resource, prior memory entry, etc.

    `target` is the actual reference value (URL string, file path, etc.).
    `kind` selects which kind of resource — used for filtering and for
    rendering icons in the UI later."""
    id:            str
    iteration_id:  str
    thread_id:     str
    workspace_id:  str
    surface_id:    str
    user_id:       str
    kind:          ReferenceKind
    target:        str
    ts:            float
    confidence:    float = 1.0
    project_id:    str | None = None
    # Optional human-readable label (e.g. page title for a URL).
    label:         str | None = None


@dataclass
class Insight:
    """A noteworthy observation — neither a decision nor a question. The
    classifier marks something as an Insight when it's a pattern the
    system noticed worth retrieving later."""
    id:            str
    iteration_id:  str
    thread_id:     str
    workspace_id:  str
    surface_id:    str
    user_id:       str
    text:          str
    ts:            float
    confidence:    float = 1.0
    project_id:    str | None = None


# ─── Bundle — what the classifier emits per iteration ────────────────

@dataclass
class DecisionTraceBundle:
    """Everything one classifier pass produces for a single iteration.

    The InlineClassifier (Phase 2b) returns one of these per iteration;
    the Neo4jDecisionTraceWriter commits all events in a single Neo4j
    transaction (atomic — all-or-nothing). The UserMessage and
    SystemResponse are always present (they're verbatim from the
    iteration); the other lists may be empty."""
    iteration_id:     str
    thread_id:        str
    user_message:     UserMessage | None = None
    system_response:  SystemResponse | None = None
    decisions:        list[Decision]   = field(default_factory=list)
    questions:        list[Question]   = field(default_factory=list)
    references:       list[Reference]  = field(default_factory=list)
    insights:         list[Insight]    = field(default_factory=list)
