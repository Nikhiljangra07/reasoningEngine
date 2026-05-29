"""
Exploration report — the unit of value a wandering agent returns.

Every time an agent finds resonance (1+ node match in any layer) and digs,
it produces ONE ExplorationReport. The report is what the user eventually
reads in the dossier; it's the "residue" of wandering (Law 2: insight
happens in the user's head, reports are residue).

Schema is strict on the load-bearing fields:
  - match_per_layer: required, structured
  - confidence: derived from match strength (not freeform)
  - what_does_not_map: MANDATORY non-empty (Law 7 enforcement — without
    this, the report is honest and Heisenberg-friendly)

Validation lives here, not at storage time, so a malformed report never
reaches the dossier. An agent that returns a report without
what_does_not_map filled has to be re-prompted (handled by agent.py).

ISOLATION: imports only from src.wandering.cushion (for layer names).
No LLM calls, no storage, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Confidence — derived, not freeform
# ---------------------------------------------------------------------------


class Confidence(str, Enum):
    """Confidence label on a report. Derived from match strength; never
    chosen freeform by the agent.

    Mapping (over the THREE LAYERS, aggregated):
      LOW    — total layer matches < 30% of total cushion nodes
      MEDIUM — 30-70% of total cushion nodes matched
      HIGH   — 70%+ of total cushion nodes matched

    Plus the structural rule: a report with HIGH match on essence OR
    mechanism layer (even if actual layer is 0) qualifies as HIGH on
    structural axes — this is the Heisenberg-zone gold case.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Per-layer match
# ---------------------------------------------------------------------------


@dataclass
class LayerMatch:
    """Per-layer match record. The agent reports which cushion-layer nodes
    it found resonance with, and how many were checked.

    `matched_nodes` are the actual cushion node STRINGS that resonated
    (e.g., "bounded freedom" from the essence layer). The agent should
    cite the cushion's own language, not paraphrase — this lets the user
    instantly see which part of their problem the report touches.
    """

    layer_name: str  # "actual" | "essence" | "mechanism"
    matched_nodes: list[str] = field(default_factory=list)
    total_nodes: int = 0  # how many nodes the cushion had in this layer

    @property
    def ratio(self) -> float:
        """Match ratio in [0.0, 1.0]. 0 if no nodes in this layer."""
        if self.total_nodes <= 0:
            return 0.0
        return len(self.matched_nodes) / self.total_nodes

    @property
    def match_count(self) -> int:
        return len(self.matched_nodes)

    def ratio_string(self) -> str:
        """Compact 'N/M' string for display."""
        return f"{self.match_count}/{self.total_nodes}"


# ---------------------------------------------------------------------------
# Source citation
# ---------------------------------------------------------------------------


@dataclass
class SourceCitation:
    """One source location the agent fetched and analyzed for this report.

    Wandering Room can read content from many places (web pages, Notion,
    arxiv, IDE files later). All of them become SourceCitations on the
    report so the user can verify the agent's bridge.
    """

    title: str
    url: str = ""
    excerpt: str = ""  # short quote / snippet anchoring the citation
    used_for: str = ""  # what role this source played ("structural comparison", etc.)


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------


@dataclass
class ExplorationReport:
    """The unit of value a wandering agent returns when it finds resonance.

    Required fields are enforced in `validate()` — an agent that produces
    a report missing what_does_not_map must be re-prompted; we do not
    silently ship dishonest reports.

    Optional fields enrich the dossier but their absence does not block
    publication.
    """

    # Identity
    report_id: str
    agent_id: str
    anchor_summary: str  # one-line user problem (for display)

    # Where the agent looked
    domain_explored: str  # "jazz improvisation", "Mongol tactics", etc. — free-form
    source_locations: list[SourceCitation] = field(default_factory=list)

    # What matched
    layer_matches: dict[str, LayerMatch] = field(default_factory=dict)

    # Confidence — derived from layer_matches
    confidence: Confidence = Confidence.LOW

    # Human content
    exploration_summary: str = ""  # what the agent found, in human terms
    advancement: str = ""  # how this advances the anchor

    # LAW 7 LOAD-BEARING: where the analogy breaks. Non-empty validated.
    what_does_not_map: str = ""

    # Optional next-step hint
    next_lead: str = ""

    # Internal
    iteration_count: int = 0  # how many iterations of dig this represents
    abandoned_early: bool = False  # set if self-critique closed the dig early

    # ---- Aggregates ----

    def total_matched_nodes(self) -> int:
        """Sum of matched-node counts across all layers."""
        return sum(m.match_count for m in self.layer_matches.values())

    def total_cushion_nodes(self) -> int:
        """Sum of total-node counts (cushion size) across all layers."""
        return sum(m.total_nodes for m in self.layer_matches.values())

    def match_ratio_summary(self) -> str:
        """Compact 'actual/essence/mechanism' string for display."""
        parts = []
        for name in ("actual", "essence", "mechanism"):
            m = self.layer_matches.get(name)
            if m:
                parts.append(f"{name[:3]}:{m.ratio_string()}")
        return " ".join(parts)

    # ---- Validation ----

    def validate(self) -> list[str]:
        """Return a list of validation errors. Empty list = valid report.

        Required:
          - report_id, agent_id non-empty
          - at least one layer match registered
          - at least one matched node across all layers (else why is this
            a report at all?)
          - exploration_summary non-empty
          - what_does_not_map non-empty (LAW 7 enforcement)
          - confidence value is a valid Confidence

        Soft (not enforced, just logged warnings):
          - source_locations empty (agent did not cite)
          - advancement empty (agent did not link to anchor)
        """
        errors: list[str] = []

        if not self.report_id.strip():
            errors.append("report_id is empty")
        if not self.agent_id.strip():
            errors.append("agent_id is empty")
        if not self.layer_matches:
            errors.append("no layer_matches registered")
        if self.total_matched_nodes() == 0:
            errors.append("zero nodes matched — report should not exist")
        if not self.exploration_summary.strip():
            errors.append("exploration_summary is empty")
        if not self.what_does_not_map.strip():
            errors.append(
                "what_does_not_map is empty (Law 7: agents must articulate "
                "where the analogy breaks; re-prompt the agent)"
            )

        return errors

    def is_valid(self) -> bool:
        return not self.validate()

    # ---- Confidence derivation ----

    def compute_confidence(self) -> Confidence:
        """Derive confidence from layer matches. Does NOT mutate; returns
        the computed value. Caller decides whether to assign to .confidence.

        Rule:
          - If essence ratio >= 0.7 OR mechanism ratio >= 0.7 → HIGH
            (Heisenberg-zone gold: structural resonance is strong even
            if surface is zero)
          - Else if total ratio >= 0.5 → MEDIUM
          - Else if total ratio >= 0.2 → MEDIUM (single-axis partial match
            is still worth the user's attention)
          - Else → LOW

        Note: LOW is NOT a filter. Per Law 3, LOW reports go to the
        dossier; the user filters at output, not exploration time.
        """
        essence = self.layer_matches.get("essence")
        mechanism = self.layer_matches.get("mechanism")

        # Structural HIGH — Heisenberg gold case
        if essence and essence.ratio >= 0.7:
            return Confidence.HIGH
        if mechanism and mechanism.ratio >= 0.7:
            return Confidence.HIGH

        total_matched = self.total_matched_nodes()
        total_cushion = self.total_cushion_nodes()
        if total_cushion <= 0:
            return Confidence.LOW

        overall = total_matched / total_cushion
        if overall >= 0.5:
            return Confidence.MEDIUM
        if overall >= 0.2:
            return Confidence.MEDIUM
        return Confidence.LOW


__all__ = [
    "Confidence",
    "LayerMatch",
    "SourceCitation",
    "ExplorationReport",
]
