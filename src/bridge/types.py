"""
Bridge data types.

Composition over modification: these types do NOT extend or duplicate the
core Wu Xing types in src/core/types.py. They are the bridge's own
contract for talking about decisions, code references, fingerprints, and
drift.

Crossing the bridge into wuxing: a DecisionAnchor or DriftReport.
Crossing out: any of these types as query results.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Code references — the back-pointer from a decision into the codebase.
# Sourced from graphify on the live side; user-supplied on the stub side.
# ---------------------------------------------------------------------------

@dataclass
class CodeRef:
    """
    A pointer to a specific location in the codebase.

    Line numbers are 1-indexed and inclusive on both ends. A whole-file
    reference uses line_start=1, line_end=1 with symbol_name=None.
    """
    file_path: str                         # relative to repo root, e.g. "src/handlers/refund.ts"
    line_start: int                        # 1-indexed
    line_end: int                          # 1-indexed, inclusive
    symbol_name: str | None = None         # function/class name if known
    symbol_type: str | None = None         # "function" | "class" | "method" | "schema" | "config" | None


# ---------------------------------------------------------------------------
# Context fingerprint — the vector embedding of conditions when a decision
# was made. Lives in Chroma on the live Memory V2 side. None vector in
# stub mode.
# ---------------------------------------------------------------------------

@dataclass
class ContextFingerprint:
    """
    Vector embedding of the conditions surrounding a decision.

    Used for similarity search ("find decisions made under conditions
    like the ones we are in now"). The vector itself is None in stub
    mode; metadata is always populated.
    """
    id: str
    vector: list[float] | None             # embedding (None in stub mode)
    metadata: dict                         # project state, active constraints, time pressure, related decisions
    created_at: float


# ---------------------------------------------------------------------------
# Decision anchor — the central memory object. Mirrors the Falkor side of
# Memory V2 (FactAnchor) but adapted for code-decision context.
# ---------------------------------------------------------------------------

@dataclass
class DecisionAnchor:
    """
    A single decision: what was chosen, why, what code it touches, and
    whether it has been superseded.

    The lifecycle is OPEN → SETTLED → (DRIFTED | SUPERSEDED | REJECTED).
    DRIFTED is set by detect_drift() when the code stops honoring the
    decision; SUPERSEDED is set when a newer decision replaces this one.
    """
    id: str                                # e.g. "D-014"
    title: str                             # e.g. "Idempotency keys = request-shape hash + orgId"
    rationale: str                         # why this decision was made
    evidence: list[str]                    # facts/incidents that drove the decision
    status: str                            # "OPEN" | "SETTLED" | "DRIFTED" | "REJECTED" | "SUPERSEDED"
    created_at: float                      # unix timestamp
    superseded_by: str | None = None       # ID of the decision that replaces this one, if any
    supersedes: str | None = None          # ID of the decision this one replaces, if any
    code_refs: list[CodeRef] = field(default_factory=list)
    context_fingerprint_id: str | None = None
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Drift detection output. The killer feature of the bridge.
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """
    Result of comparing a decision's intent against the current code at
    each of its code_refs.

    The top-level is_drifted is True if ANY per-ref check came back
    drifted. per_ref_reports holds one DriftReport per code_ref so
    callers can inspect which specific location drifted; the top-level
    fields summarize the first drifted ref (or the first ref overall if
    nothing drifted).

    suggested_action:
        "reconcile"  — change the code to honor the decision
        "supersede"  — write a new decision; this one is obsolete
        "no_action"  — code and decision agree, or comparator is unsure
    """
    decision: DecisionAnchor
    code_ref: CodeRef                      # the location this report is about
    is_drifted: bool
    drift_description: str
    confidence: float                      # 0.0–1.0
    suggested_action: str                  # "reconcile" | "supersede" | "no_action"
    # Per-ref breakdown — populated by detect_drift() when a decision has
    # multiple code_refs. Empty for per-ref reports themselves and for
    # floating-decision reports (decisions with zero code_refs).
    per_ref_reports: list["DriftReport"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Composite query/result placeholders.
#
# Reserved for future bridge queries that need to combine graphify and
# Memory V2 results in a single request (e.g. "find every decision whose
# code_refs touch any file in the auth subsystem"). Kept minimal until
# the first real consumer arrives, so the shape can be designed against
# a real call site rather than guessed.
# ---------------------------------------------------------------------------

@dataclass
class BridgeQuery:
    """Placeholder for composite bridge queries. Not yet consumed."""
    kind: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class BridgeResult:
    """Placeholder for composite bridge results. Not yet consumed."""
    kind: str = ""
    decisions: list[DecisionAnchor] = field(default_factory=list)
    code_refs: list[CodeRef] = field(default_factory=list)
    drift_reports: list[DriftReport] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversation history types — the structured-storage spine.
#
# Hierarchy:
#   Session                  (conversation thread)
#     └── Iteration[]         (user query + engine response, one turn each)
#           ├── decision_ids[]        — DecisionAnchors made in this turn
#           └── turning_point_ids[]   — pivot moments rooted in this turn
#   DecisionLink              (directed edges between DecisionAnchors)
#
# All entities carry expires_at. None = pinned forever; float = unix epoch
# after which sweep_expired() removes the row. Default TTL is set by
# ConversationStore (30 days). Filter-on-read hides expired entries from
# normal queries; sweep_expired() does the actual delete.
# ---------------------------------------------------------------------------

@dataclass
class Iteration:
    """One user-engine exchange. The atomic unit of a conversation."""
    id: str                                              # globally unique
    session_id: str                                      # parent session
    sequence_num: int                                    # ordering within session (1, 2, 3...)
    user_text: str
    engine_response: str
    created_at: float
    route: str = ""                                      # trivial | direct | direct_plus | deep
    effort: str = ""                                     # low | medium | high | auto
    decision_ids: list[str] = field(default_factory=list)
    turning_point_ids: list[str] = field(default_factory=list)
    parent_iteration_id: str | None = None               # branches off another iteration
    expires_at: float | None = None                      # None = pinned forever


@dataclass
class Session:
    """A conversation thread. Lives in exactly one project."""
    id: str
    project_id: str | None
    title: str
    started_at: float
    ended_at: float | None = None
    iteration_count: int = 0
    decision_count: int = 0
    turning_point_count: int = 0
    status: str = "active"                               # active | ended | archived
    expires_at: float | None = None


@dataclass
class TurningPoint:
    """
    A pivot moment in a conversation. Higher-level than a single Decision —
    captures "this is where direction shifted." Triggered by one or more
    decisions; leads to one or more downstream decisions.
    """
    id: str
    session_id: str
    iteration_id: str
    title: str
    description: str
    triggered_by_decisions: list[str] = field(default_factory=list)
    led_to_decisions: list[str] = field(default_factory=list)
    created_at: float = 0.0
    expires_at: float | None = None


@dataclass
class DecisionLink:
    """
    Directed edge between two decisions. The graph layer that turns isolated
    DecisionAnchors into a navigable lineage.

    link_type values (the only ones the store enforces):
        "leads_to"     — completing A made B necessary or natural
        "supersedes"   — B replaced A (A is now historical)
        "depends_on"   — A holds only while B holds
        "contradicts"  — A and B can't both be true; surface for resolution
        "informed_by"  — B was made considering A as input (lighter than depends_on)
    """
    id: str
    project_id: str | None
    from_decision_id: str
    to_decision_id: str
    link_type: str
    rationale: str = ""
    created_at: float = 0.0
    expires_at: float | None = None
