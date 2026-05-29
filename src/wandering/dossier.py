"""
Dossier — the final user-facing output of a Wandering Room session.

Combines:
  - The articulated cards (one per report)
  - The synthesis map (clusters, contradictions, opportunity paths)
  - The session metadata (mode, duration, agent count, tokens spent)

Organized into three confidence bands per Law 3 — LOW is surfaced
prominently as the Heisenberg zone, NOT buried.

The Dossier is what the API endpoint (later wiring) will serialize and
return to the frontend. The frontend renders cards, lets the user filter,
export to Notion, or click "dig deeper" on any card to spawn a sub-agent.

ISOLATION: imports articulate + synthesis + report + cushion + runtime.
The dossier builder takes ALL THESE pieces and produces the final
artifact. No LLM calls in this module — it's pure assembly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.wandering.articulate import ArticulatedCard, articulate_report
from src.wandering.cushion import CushionGraph
from src.wandering.report import Confidence, ExplorationReport
from src.wandering.runtime import SessionResult, WanderingMode
from src.wandering.synthesis import SynthesisMap, synthesize_dossier


# ---------------------------------------------------------------------------
# Dossier structures
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceBand:
    """One confidence tier of cards in the dossier.

    Cards within a band are typically rendered in a collapsible section.
    LOW band is the Heisenberg zone — surfaced, not hidden.
    """

    confidence: Confidence
    cards: list[ArticulatedCard] = field(default_factory=list)

    def count(self) -> int:
        return len(self.cards)


@dataclass
class DossierMetadata:
    """Session-level metadata for the dossier header."""

    session_id: str
    mode: WanderingMode
    anchor_summary: str
    cushion_constellation_size: int  # total nodes across three layers
    agent_count: int
    report_count: int
    total_tokens_spent: int
    elapsed_seconds: float
    completed_at: float


@dataclass
class Dossier:
    """The final user-facing artifact.

    Three confidence bands + the synthesis map + metadata.

    Frontend rendering order:
      1. metadata header (anchor + session summary)
      2. recommended_next_direction (one paragraph, prominent)
      3. top_insights cards (HIGH first, then MEDIUM)
      4. clusters (each as expandable group)
      5. contradictions (called out)
      6. opportunity_paths (cards' paths forward)
      7. ALL CARDS by band (HIGH / MEDIUM / LOW)
      8. open_questions (raised but not answered)
      9. what_would_change_the_verdict (sensitivity statement)
    """

    metadata: DossierMetadata
    high: ConfidenceBand = field(default_factory=lambda: ConfidenceBand(Confidence.HIGH))
    medium: ConfidenceBand = field(default_factory=lambda: ConfidenceBand(Confidence.MEDIUM))
    low: ConfidenceBand = field(default_factory=lambda: ConfidenceBand(Confidence.LOW))
    synthesis: SynthesisMap = field(default_factory=SynthesisMap)

    def all_cards(self) -> list[ArticulatedCard]:
        """All cards across all bands in canonical order."""
        return self.high.cards + self.medium.cards + self.low.cards

    def card_by_id(self, report_id: str) -> ArticulatedCard | None:
        for c in self.all_cards():
            if c.report_id == report_id:
                return c
        return None

    def to_dict(self) -> dict:
        """Render as a JSON-safe dict (for the future API endpoint).

        We provide this here so the API layer is thin — it just calls
        Dossier.to_dict() and returns the result. No serialization logic
        in the route handler.
        """
        return {
            "metadata": {
                "session_id": self.metadata.session_id,
                "mode": self.metadata.mode.value,
                "anchor_summary": self.metadata.anchor_summary,
                "cushion_constellation_size": self.metadata.cushion_constellation_size,
                "agent_count": self.metadata.agent_count,
                "report_count": self.metadata.report_count,
                "total_tokens_spent": self.metadata.total_tokens_spent,
                "elapsed_seconds": self.metadata.elapsed_seconds,
                "completed_at": self.metadata.completed_at,
            },
            "high": [c.to_dict() for c in self.high.cards],
            "medium": [c.to_dict() for c in self.medium.cards],
            "low": [c.to_dict() for c in self.low.cards],
            "synthesis": {
                "top_insights": self.synthesis.top_insights,
                "clusters": [
                    {
                        "label": c.label,
                        "card_ids": c.card_ids,
                        "summary": c.summary,
                    }
                    for c in self.synthesis.clusters
                ],
                "contradictions": [
                    {
                        "description": c.description,
                        "card_ids": c.card_ids,
                    }
                    for c in self.synthesis.contradictions
                ],
                "opportunity_paths": [
                    {
                        "description": o.description,
                        "supporting_card_ids": o.supporting_card_ids,
                        "confidence_estimate": o.confidence_estimate.value,
                    }
                    for o in self.synthesis.opportunity_paths
                ],
                "open_questions": self.synthesis.open_questions,
                "recommended_next_direction": self.synthesis.recommended_next_direction,
                "what_would_change_the_verdict": self.synthesis.what_would_change_the_verdict,
            },
        }


# ---------------------------------------------------------------------------
# Dossier builder — orchestrates articulation + synthesis
# ---------------------------------------------------------------------------


async def build_dossier(
    session: SessionResult,
    client,  # LLMClient — not typed to keep imports minimal
) -> Dossier:
    """Build the final Dossier from a completed SessionResult.

    Pipeline:
      1. Articulate every report → ArticulatedCard
      2. Synthesize all cards → SynthesisMap
      3. Sort cards into confidence bands
      4. Assemble Dossier with metadata

    This is the public entry point of the wandering pipeline. The future
    API endpoint will call run_wandering_session() then build_dossier().
    """
    # Step 1: articulate each report
    cards: list[ArticulatedCard] = []
    for report in session.reports:
        card = await articulate_report(report, client)
        cards.append(card)

    # Step 2: synthesize across all cards
    anchor_summary = (
        session.cushion.raw_input.problem.content[:200]
        if session.cushion and session.cushion.raw_input
        else ""
    )
    synthesis_map = await synthesize_dossier(anchor_summary, cards, client)

    # Step 3: bucket by confidence
    high_band = ConfidenceBand(Confidence.HIGH)
    medium_band = ConfidenceBand(Confidence.MEDIUM)
    low_band = ConfidenceBand(Confidence.LOW)
    for c in cards:
        if c.confidence == Confidence.HIGH:
            high_band.cards.append(c)
        elif c.confidence == Confidence.MEDIUM:
            medium_band.cards.append(c)
        else:
            low_band.cards.append(c)

    # Step 4: metadata
    metadata = DossierMetadata(
        session_id=session.session_id,
        mode=session.mode,
        anchor_summary=anchor_summary,
        cushion_constellation_size=session.cushion.constellation_size,
        agent_count=session.agent_count(),
        report_count=session.report_count(),
        total_tokens_spent=session.total_tokens_spent,
        elapsed_seconds=session.elapsed_seconds,
        completed_at=session.ended_at or time.time(),
    )

    return Dossier(
        metadata=metadata,
        high=high_band,
        medium=medium_band,
        low=low_band,
        synthesis=synthesis_map,
    )


__all__ = [
    "ConfidenceBand",
    "DossierMetadata",
    "Dossier",
    "build_dossier",
]
