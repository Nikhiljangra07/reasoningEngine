"""
Collision pipeline — the five seats wired into one line.

This is the linear v1 of the collision architecture. It takes the
verified-sorted cards and drives them through the back half of the pipeline,
everything anchored to the cushion:

    (stages 1-2 upstream: wander → verified-sort, via build_dossier(verify_web=True))
    stage 3  BLEND        Opus 4.8 collides cards → candidate concepts (+ discovery_path)
    stage 4  DRIFT-CHECK  DeepSeek V4 Pro supervises; quarantines blends that
                          drifted off the cushion (recorded, not passed downstream)
    stage 5  BLEND-VERIFY Sonnet + web bins survivors: known / adjacent / novel / flawed

The deferred v2 "shuffling system" (drift-checker COMMANDS the blender to
re-blend against a twisted phantom anchor, looping) is NOT here. This is the
straight line: prove it makes gold, then add the loop.

DOCUMENTATION ENVIRONMENT
-------------------------
`build_run_record` stitches the whole chain — cushion → cards → sort+evidence
→ blends+genealogy → drift verdicts → novelty bins — into one auditable
record, plus a `trace` index so any surviving discovery can be reverse-
engineered in a glance: blend_id → discovery_path → source cards → their bins
→ the evidence that verified them → drift verdict → novelty bin.

ISOLATION
---------
Imports the four seat modules + dossier card/sort types. No module imports
this one back (no cycle). Makes no LLM/web calls of its own — it only
sequences the seats and rolls up their cost.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from src.wandering.articulate import ArticulatedCard
from src.wandering.blend_verify import (
    DEFAULT_QUERY_MODEL,
    DEFAULT_VERIFY_MODEL,
    BlendVerificationReport,
    verify_blends,
)
from src.wandering.blender import (
    BLENDER_MODEL,
    BlendBatch,
    blend_cards,
)
from src.wandering.cushion import CushionGraph
from src.wandering.drift_checker import (
    DRIFT_CHECKER_MODEL,
    DriftReport,
    check_drift,
)
from src.wandering.master_sorter import SortedReport
from src.wandering.sorter_verify import SearchFn

log = logging.getLogger("constellax.wandering.collision_pipeline")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class CollisionReport:
    """Everything the collision half produced, for one run."""
    blends:        BlendBatch | None              = None
    drift:         DriftReport | None             = None
    verification:  BlendVerificationReport | None = None
    quarantined_blend_ids: list[str]              = field(default_factory=list)
    models:        dict                           = field(default_factory=dict)
    stage_costs:   dict                           = field(default_factory=dict)
    total_cost_usd: float                         = 0.0
    elapsed_ms:    float                          = 0.0

    def to_dict(self) -> dict:
        return {
            "blends":        self.blends.to_dict() if self.blends else None,
            "drift":         self.drift.to_dict() if self.drift else None,
            "verification":  self.verification.to_dict() if self.verification else None,
            "quarantined_blend_ids": list(self.quarantined_blend_ids),
            "models":        dict(self.models),
            "stage_costs":   dict(self.stage_costs),
            "total_cost_usd": round(self.total_cost_usd, 4),
            "elapsed_ms":    round(self.elapsed_ms, 1),
        }


@dataclass
class CollisionProgress:
    on_event: Callable[[str, dict], None] | None = None
    events:   list[dict] = field(default_factory=list)

    def emit(self, name: str, payload: dict | None = None) -> None:
        payload = payload or {}
        entry = {"name": name, "ts": time.time(), **payload}
        self.events.append(entry)
        log.info("[collision] %s %s", name, payload)
        if self.on_event is not None:
            try:
                self.on_event(name, payload)
            except Exception as e:  # pragma: no cover
                log.warning("collision progress on_event raised (ignored): %s", e)


# ---------------------------------------------------------------------------
# Bin derivation
# ---------------------------------------------------------------------------


def bins_from_sorted(sorted_report: SortedReport) -> tuple[dict[str, str], set[str]]:
    """Map report_id → 'known'|'unplaced' (the blendable material) and return
    the set of INVALID report_ids (dirt — excluded from blending)."""
    bins: dict[str, str] = {}
    for it in sorted_report.known:
        bins[it.card.report_id] = "known"
    for it in sorted_report.unplaced:
        bins[it.card.report_id] = "unplaced"
    invalid_ids = {it.card.report_id for it in sorted_report.invalid}
    return bins, invalid_ids


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


async def run_collision_pipeline(
    *,
    cushion:       CushionGraph | None,
    cards:         list[ArticulatedCard],
    sorted_report: SortedReport,
    client,
    blender_model: str = BLENDER_MODEL,
    drift_model:   str = DRIFT_CHECKER_MODEL,
    query_model:   str = DEFAULT_QUERY_MODEL,
    verify_model:  str = DEFAULT_VERIFY_MODEL,
    search_fn:     SearchFn | None = None,
    progress:      CollisionProgress | None = None,
) -> CollisionReport:
    """Drive verified-sorted cards through blend → drift-check → blend-verify.

    INVALID cards are excluded (dirt). Drifting blends are quarantined
    (recorded, not verified as primary). Each stage is independently budget-
    capped inside its own seat; this orchestrator only sequences + rolls up.
    """
    progress = progress or CollisionProgress()
    report = CollisionReport(models={
        "blender": blender_model, "drift": drift_model,
        "query": query_model, "verify": verify_model,
    })
    t0 = time.time()

    bins, invalid_ids = bins_from_sorted(sorted_report)
    blendable = [c for c in cards if c.report_id not in invalid_ids]
    progress.emit("starting", {
        "cards": len(cards), "blendable": len(blendable), "invalid_excluded": len(invalid_ids),
    })

    # Stage 3 — BLEND
    batch = await blend_cards(
        cushion=cushion, cards=blendable, bins_by_id=bins,
        client=client, model=blender_model,
    )
    report.blends = batch
    report.stage_costs["blend"] = round(batch.total_cost_usd, 4)
    progress.emit("blended", {"blends": len(batch.blends), "cost": batch.total_cost_usd})

    # Stage 4 — DRIFT-CHECK (supervise; quarantine drifters)
    drift = await check_drift(
        cushion=cushion, blends=batch.blends, client=client, model=drift_model,
    )
    report.drift = drift
    report.stage_costs["drift"] = round(drift.total_cost_usd, 4)
    drifting = set(drift.drifting_ids)
    report.quarantined_blend_ids = [b.blend_id for b in batch.blends if b.blend_id in drifting]
    on_course = [b for b in batch.blends if b.blend_id not in drifting]
    progress.emit("drift_checked", {
        "on_course": len(on_course), "quarantined": len(report.quarantined_blend_ids),
    })

    # Stage 5 — BLEND-VERIFY (only the survivors)
    verification = await verify_blends(
        cushion=cushion, blends=on_course, client=client,
        query_model=query_model, verify_model=verify_model, search_fn=search_fn,
    )
    report.verification = verification
    report.stage_costs["verify"] = round(verification.total_cost_usd, 4)
    progress.emit("verified", {
        "known":    len(verification.known),
        "adjacent": len(verification.adjacent),
        "novel":    len(verification.novel),
        "flawed":   len(verification.flawed),
    })

    report.total_cost_usd = (
        batch.total_cost_usd + drift.total_cost_usd + verification.total_cost_usd
    )
    report.elapsed_ms = (time.time() - t0) * 1000
    progress.emit("complete", {
        "total_cost_usd": round(report.total_cost_usd, 4),
        "elapsed_ms": round(report.elapsed_ms, 1),
    })
    return report


# ---------------------------------------------------------------------------
# Documentation environment — the unified, reverse-engineerable run record
# ---------------------------------------------------------------------------


def _novelty_bin_of(blend_id: str, verification: BlendVerificationReport | None) -> str:
    if verification is None:
        return "unverified"
    for bin_name, items in (
        ("known", verification.known), ("adjacent", verification.adjacent),
        ("novel", verification.novel), ("flawed", verification.flawed),
    ):
        if any(v.blend_id == blend_id for v in items):
            return bin_name
    return "unverified"


def build_run_record(
    *,
    cushion_problem: str,
    dossier_dict:    dict,
    collision:       CollisionReport,
) -> dict:
    """Stitch the whole pipeline into ONE auditable record + a `trace` index.

    The trace lets any surviving discovery be reverse-engineered at a glance:
    blend_id → discovery_path (genealogy) → source cards → their sort bin →
    drift verdict → novelty bin.
    """
    batch = collision.blends
    drift = collision.drift
    ver   = collision.verification

    # report_id → sort bin (from the dossier's master_sorted)
    sort_bins: dict[str, str] = {}
    ms = (dossier_dict or {}).get("master_sorted") or {}
    for bin_name in ("known", "invalid", "unplaced"):
        for it in ms.get(bin_name, []) or []:
            rid = (it.get("card") or {}).get("report_id")
            if rid:
                sort_bins[rid] = bin_name

    drift_by_id = {}
    if drift is not None:
        for v in drift.verdicts:
            drift_by_id[v.blend_id] = {"on_course": v.on_course, "resonance": v.resonance}

    trace = []
    if batch is not None:
        for b in batch.blends:
            trace.append({
                "blend_id":        b.blend_id,
                "novelty_bin":     _novelty_bin_of(b.blend_id, ver),
                "drift":           drift_by_id.get(b.blend_id, {"on_course": True, "resonance": 1.0}),
                "quarantined":     b.blend_id in set(collision.quarantined_blend_ids),
                "discovery_path":  b.selection.discovery_path,
                "source_card_ids": list(b.source_card_ids),
                "source_card_bins": {rid: sort_bins.get(rid, "?") for rid in b.source_card_ids},
                "thesis":          b.thesis,
            })

    return {
        "cushion": {"problem": cushion_problem},
        "stage_1_2_cards_and_sort": dossier_dict,
        "stage_3_blends":      batch.to_dict() if batch else None,
        "stage_4_drift":       drift.to_dict() if drift else None,
        "stage_5_verification": ver.to_dict() if ver else None,
        "quarantined_blend_ids": list(collision.quarantined_blend_ids),
        "trace":               trace,
        "cost_rollup": {
            "stages":    collision.stage_costs,
            "collision_total_usd": round(collision.total_cost_usd, 4),
        },
        "models": collision.models,
    }


__all__ = (
    "CollisionReport",
    "CollisionProgress",
    "bins_from_sorted",
    "run_collision_pipeline",
    "build_run_record",
)
