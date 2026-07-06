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

from src.identity.disciplines.goal_supremacy import discriminate
from src.identity.disciplines.opportunity_capture import Opening, test as opportunity_test
from src.identity.singular_path import Goal
from src.wandering.articulate import ArticulatedCard, articulate_report
from src.wandering.cushion import CushionGraph
from src.wandering.master_sorter import (
    MasterSortProgress,
    SortedReport,
    master_sort,
)
from src.wandering.sorter_verify import (
    DEFAULT_QUERY_MODEL,
    EvidenceLedger,
    SearchFn,
    gather_evidence,
)
from src.wandering.master_synthesizer import (
    DEFAULT_COST_CEILING_USD,
    MasterSynthesis,
    MasterSynthesisProgress,
    master_synthesize,
)
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
    # Master fusion layer — optional, populated only when build_dossier
    # was invoked with run_master_synthesizer=True (the slow, expensive
    # path; ~5-8 min wall-clock + ~$5-7 cost). None when skipped so
    # frontends can branch on presence rather than emptiness.
    master_synthesis: MasterSynthesis | None = None
    # Master sorter layer — tributary that runs in place of the
    # synthesizer when build_dossier(pipeline_mode="sorter"). Single
    # Fable 5 pass; classifies each card into known / invalid / unplaced.
    # The dam: when this is set, master_synthesis is None and vice
    # versa. None when run_master_synthesizer=False.
    master_sorted: SortedReport | None = None
    # Web-verification trace — the EvidenceLedger gathered by sorter_verify
    # before a verified sort (per-card queries + live web hits). Populated
    # only when build_dossier was invoked with verify_web=True (or a
    # pre-built ledger was passed). None otherwise. Serialized so the human
    # reads the evidence behind every bin, not just the verdict.
    master_sorted_evidence: EvidenceLedger | None = None

    def all_cards(self) -> list[ArticulatedCard]:
        """All cards across all bands in canonical order."""
        return self.high.cards + self.medium.cards + self.low.cards

    def card_by_id(self, report_id: str) -> ArticulatedCard | None:
        for c in self.all_cards():
            if c.report_id == report_id:
                return c
        return None

    def suppressed_framings(self) -> list[dict]:
        """Pre-merge framings whose voice was suppressed by the keeper-pick.

        For every master fusion that ran through a cohort-pair merge
        (BOTH_AGREE / MOSTLY_AGREE_REFINED / DISPUTED with a merged
        title), the keeper_seat shipped its prose visibly. The OTHER
        seat's pre-merge framing is preserved on the fusion object as
        pre_merge_opus / pre_merge_gpt but it's buried in the audit
        fields — most callers won't surface it. This convenience
        extracts JUST the suppressed side per fusion so a UI can
        render an "alternate framing" expander next to the visible
        claim.

        Returns one entry per fusion where (a) keeper_seat is set
        and (b) the suppressed seat's snapshot is non-None. Empty list
        when no suppressions occurred (e.g. all SOLO_* + unpaired
        with no cross-seat merge). Entry shape:

          {
            "fusion_title":    "<the merged fusion's title>",
            "suppressed_seat": "opus" | "gpt",
            "title":           "<the suppressed pre-merge title>",
            "claim":           "<the suppressed pre-merge claim>",
            "reasoning":       "<the suppressed pre-merge reasoning>",
            "limit":           "<the suppressed pre-merge limit>",
            "note":            "<plain-text rationale for the UI>",
          }

        Pure derivation from existing master_synthesis data — no
        schema change to MasterFusionReport. Empty list when
        master_synthesis was not run.
        """
        out: list[dict] = []
        if self.master_synthesis is None:
            return out
        for f in self.master_synthesis.master_fusions:
            seat = f.keeper_seat
            if seat not in ("opus", "gpt"):
                continue
            suppressed_seat = "gpt" if seat == "opus" else "opus"
            snap = f.pre_merge_gpt if suppressed_seat == "gpt" else f.pre_merge_opus
            if not snap:
                continue
            out.append({
                "fusion_title":    f.title,
                "suppressed_seat": suppressed_seat,
                "title":           snap.get("title", ""),
                "claim":           snap.get("claim", ""),
                "reasoning":       snap.get("reasoning", ""),
                "limit":           snap.get("limit", ""),
                "note": (
                    "Suppressed by keeper-score; the kept side is in the "
                    "main fusion above. Same evidence, different framing."
                ),
            })
        return out

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
            "master_synthesis": (
                self.master_synthesis.to_dict() if self.master_synthesis is not None else None
            ),
            "master_sorted": (
                self.master_sorted.to_dict() if self.master_sorted is not None else None
            ),
            "master_sorted_evidence": (
                self.master_sorted_evidence.to_dict()
                if self.master_sorted_evidence is not None else None
            ),
            # Fix 7 (audit r4→r5): expose the SUPPRESSED side of every
            # cohort-pair merge as a top-level convenience field for the
            # UI's "alternate framing" expander. Pure derivation from
            # master_synthesis.master_fusions — no schema change to the
            # underlying MasterFusionReport.
            "suppressed_framings": self.suppressed_framings(),
        }


# ---------------------------------------------------------------------------
# Dossier builder — orchestrates articulation + synthesis
# ---------------------------------------------------------------------------


async def build_dossier(
    session: SessionResult,
    client,  # LLMClient — not typed to keep imports minimal
    *,
    run_master_synthesizer: bool = False,
    pipeline_mode: str = "sorter",
    master_synth_cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
    master_synth_progress: MasterSynthesisProgress | None = None,
    master_synth_opus_model: str | None = None,
    master_synth_gpt_model:  str | None = None,
    master_synth_agent_provider_map: dict[str, str] | None = None,
    master_sort_progress: MasterSortProgress | None = None,
    master_sort_fable_model: str | None = None,
    verify_web: bool = False,
    web_evidence: EvidenceLedger | None = None,
    sort_query_model: str | None = None,
    sort_search_fn: SearchFn | None = None,
) -> Dossier:
    """Build the final Dossier from a completed SessionResult.

    Pipeline:
      1. Articulate every report → ArticulatedCard
      2. Synthesize all cards → SynthesisMap (existing single-pass layer)
      3. Sort cards into confidence bands
      4. Assemble Dossier with metadata
      5. [optional, off by default] Run the master synthesizer on top of
         the assembled dossier — Opus 4.6 + GPT-5.4 collaborative critique
         producing 3-5 cross-card master fusion reports with disputed
         angles preserved when the seats genuinely disagree.

    Master synthesis is OFF by default so existing callers (API
    endpoint, /tmp/live_wander.py pre-run-#3) get the existing dossier
    shape unchanged. Pass `run_master_synthesizer=True` to opt in. The
    additional spend is bounded by `master_synth_cost_ceiling_usd`
    (default $8 — Nikhil's test-phase ceiling, raise for production).

    Progress emission: pass a MasterSynthesisProgress instance via
    `master_synth_progress` to receive per-round events (drafting,
    critiquing, finalizing, drilling-into-disputes, complete). When
    omitted, the synthesizer creates an internal one that logs to
    `constellax.wandering.master_synthesizer` at INFO level.
    """
    # Step 1: articulate each report
    cards: list[ArticulatedCard] = []
    for report in session.reports:
        card = await articulate_report(report, client)
        cards.append(card)

    # Anchor summary doubles as the goal text for identity-layer scoring
    # below. The cushion's problem field is the stated goal; without a
    # real-goal surface from the cushion (Hook 1, future), `real`
    # defaults to the same string and `discriminate` runs in
    # consistent-goal mode.
    anchor_summary = (
        session.cushion.raw_input.problem.content[:200]
        if session.cushion and session.cushion.raw_input
        else ""
    )

    # Identity-layer metadata: score each card's bridge against the
    # goal. Result is attached as `card.serve_score`. Band sorting
    # below does NOT branch on this — order within a band stays
    # confidence-determined. The frontend may render a "serves real
    # goal" badge or a "serves stated but not real goal" warning.
    if anchor_summary:
        goal = Goal(stated=anchor_summary, real=anchor_summary)
        for c in cards:
            scoring_text = " ".join(filter(None, (c.bridge, c.use, c.spark)))
            if scoring_text.strip():
                try:
                    c.serve_score = discriminate(scoring_text, goal)
                except Exception:  # pragma: no cover — discriminate is pure
                    c.serve_score = None

    # Step 2: synthesize across all cards
    synthesis_map = await synthesize_dossier(anchor_summary, cards, client)

    # Identity-layer ENFORCEMENT (0.3.4) on opportunity paths.
    # Each path gets a six-question test verdict ("capture" /
    # "surface" / "skip"). The verdict is attached to the path AND
    # used to split the list: capture/surface paths stay in
    # `opportunity_paths` (primary, prominently rendered);
    # "skip" paths move to `deprioritized_paths` (secondary,
    # rendered as a "weak signals" collapsible section in the
    # frontend). Nothing is silently deleted — every path the
    # synthesizer surfaced still reaches the user, but the curated
    # primary list keeps only the paths that passed >= 4/6
    # questions.
    if anchor_summary:
        path_goal = Goal(stated=anchor_summary, real=anchor_summary)
        kept: list = []
        deprioritized: list = []
        for path in synthesis_map.opportunity_paths:
            try:
                opening = Opening(description=path.description)
                vdict = opportunity_test(opening, path_goal)
                path.verdict = vdict.verdict
                path.verdict_score = vdict.score
            except Exception:  # pragma: no cover — test is pure
                path.verdict = ""
                path.verdict_score = 0
            if path.verdict == "skip":
                deprioritized.append(path)
            else:
                kept.append(path)
        synthesis_map.opportunity_paths = kept
        synthesis_map.deprioritized_paths = deprioritized

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

    # Identity-layer ENFORCEMENT (0.3.4): within each band, sort cards
    # by `serve_score.score` descending so goal-aligned bridges land
    # at the top of their band. Confidence bands themselves (HIGH >
    # MEDIUM > LOW) are untouched — this is a tiebreaker within band
    # only, not a re-banding. Cards without a `serve_score` (e.g.
    # short bridges that scored empty) sort to the end of their
    # band, preserving the prior arrival order among themselves via
    # Python's stable sort. The reorder is reversible by removing
    # this loop.
    def _serve_key(card: ArticulatedCard) -> float:
        sc = card.serve_score
        if sc is None:
            return -1.0  # cards without a score go to the end of band
        return float(sc.score)

    for band in (high_band, medium_band, low_band):
        band.cards.sort(key=_serve_key, reverse=True)

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

    dossier = Dossier(
        metadata=metadata,
        high=high_band,
        medium=medium_band,
        low=low_band,
        synthesis=synthesis_map,
    )

    # Step 5 (optional): master tier — runs ONLY when opted in via
    # run_master_synthesizer=True. The flag name is retained for
    # backwards-compat with existing callers but the tier it triggers
    # is selected by `pipeline_mode`:
    #
    #   "sorter"      — Fable 5 classifies each card into known /
    #                   invalid / unplaced. Single-pass, single-seat,
    #                   no fusion. THIS IS THE DEFAULT (the dam).
    #   "synthesizer" — Opus 4.6 + GPT-5.4 collaborative R1-R4 producing
    #                   3-5 cross-card master fusions. The pre-dam path,
    #                   still callable but no longer the default.
    #
    # The two tributaries are mutually exclusive within a single
    # build_dossier call — only one populates the corresponding
    # dossier field, the other stays None.
    if run_master_synthesizer:
        if pipeline_mode == "sorter":
            from src.wandering.master_sorter import FABLE_SEAT_MODEL
            # Web verification (sorter brick 1): gather real web evidence
            # for every card BEFORE the sort, so the sorter bins against
            # the live internet instead of training memory. Opt-in via
            # verify_web=True, or pass a pre-built ledger via web_evidence
            # (replay scripts reuse a saved ledger to skip re-searching).
            evidence = web_evidence
            if evidence is None and verify_web:
                evidence = await gather_evidence(
                    cushion=session.cushion,
                    cards=dossier.all_cards(),
                    client=client,
                    query_model=sort_query_model or DEFAULT_QUERY_MODEL,
                    search_fn=sort_search_fn,
                )
            dossier.master_sorted_evidence = evidence
            dossier.master_sorted = await master_sort(
                cushion=session.cushion,
                cards=dossier.all_cards(),
                synthesis_map=synthesis_map,
                client=client,
                progress=master_sort_progress,
                cost_ceiling_usd=master_synth_cost_ceiling_usd,
                fable_model=master_sort_fable_model or FABLE_SEAT_MODEL,
                web_evidence=evidence,
            )
        elif pipeline_mode == "synthesizer":
            from src.wandering.master_synthesizer import (
                OPUS_SEAT_MODEL, GPT_SEAT_MODEL,
            )
            dossier.master_synthesis = await master_synthesize(
                cushion=session.cushion,
                cards=dossier.all_cards(),
                synthesis_map=synthesis_map,
                client=client,
                progress=master_synth_progress,
                cost_ceiling_usd=master_synth_cost_ceiling_usd,
                opus_model=master_synth_opus_model or OPUS_SEAT_MODEL,
                gpt_model=master_synth_gpt_model  or GPT_SEAT_MODEL,
                agent_provider_map=master_synth_agent_provider_map,
            )
        else:
            raise ValueError(
                f"pipeline_mode must be 'sorter' or 'synthesizer', got: {pipeline_mode!r}"
            )

    return dossier


__all__ = [
    "ConfidenceBand",
    "DossierMetadata",
    "Dossier",
    "build_dossier",
]
