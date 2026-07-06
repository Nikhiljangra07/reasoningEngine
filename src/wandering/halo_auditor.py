"""
Halo auditor — the oversight layer (blend-03, Phase 1: OBSERVER).

WHAT IT IS
----------
A halo that wraps the whole pipeline and audits each layer for BLIND SPOTS —
what's missing, and where the work went SLACK (shallow, hand-wavy, leaving the
heavy lifting unstated). It runs STEP BY STEP, at each interval as artifacts
appear:

    after the cushion  -> audit the QUESTION ITSELF   (what is it blind to?)
    after the wander   -> audit the CARDS             (what territory got missed?)
    after the blends   -> audit the BLENDS            (what holes do the lanes leave?)

PHASE 1 IS OBSERVE-ONLY. It writes prioritized blind-spot notes and acts on
NOTHING. The point is to first SEE whether the blind spots it surfaces are
genuinely good before any feedback is wired. If they prove their worth, a
later Phase 2 "commander" promotes these notes into instructions that re-aim
the wanderers (gap-gravity) and re-blend the blender — that commander is the
deferred reframe/shuffle manager. NOT built here.

WHY A HALO, NOT AN IN-LINE STAGE
--------------------------------
The three things it audits exist at three different times, so no single
position in the linear flow can see all of them. The halo touches the
pipeline at three checkpoints instead — each audit runs when its artifact is
ready, with fresh focused context.

DISCIPLINE
----------
- Everything is anchored to the cushion (the principal question). A blind spot
  only counts if it threatens ADVANCING the cushion.
- Blind spots are angle-dependent and unbounded, so each audit returns a
  PRIORITIZED, bounded set (top-N by severity to the cushion), never a spill.
- The human is the judge. The auditor reports; it does not decide.

ISOLATION
---------
Imports the dossier card type + blend type + LLMClient + pricing + json
helper. Composes its system prompt at each call site (so the identity source-
proof scan is satisfied with no exempt entry). Makes NO web calls and takes
NO actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field

from src.identity import compose_system_prompt
from src.llm.client import LLMClient
from src.llm.provider_map import get_pricing
from src.wandering.articulate import ArticulatedCard
from src.wandering.blender import Blend
from src.wandering.master_synthesizer import _parse_json_safely

log = logging.getLogger("constellax.wandering.halo_auditor")


# ---------------------------------------------------------------------------
# Model + tuning
# ---------------------------------------------------------------------------

#: The auditor seat. Sonnet 4-6 by Nikhil's directive (2026-06-17): the halo
#: auditor is a Sonnet "department" — blind-spot detection is the quality
#: bottleneck, so it gets the stronger judge model. Passed explicitly to
#: client.call(model=...), so it routes to Anthropic regardless of provider_map
#: defaults. (Was deepseek-v4-pro; flipped after the Sonnet-judge decision.)
AUDITOR_MODEL = "anthropic/claude-sonnet-4-6"

AUDITOR_DOMAIN = "halo_auditor"

#: Bounded blind-spot count per layer. Holes are unbounded; force a ranked top-N.
MAX_BLIND_SPOTS = 5

MAX_TOKENS_AUDIT = 3000
AUDIT_TEMPERATURE = 0.3


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    in_p, out_p = get_pricing(model)
    return (in_tok or 0) / 1_000_000 * in_p + (out_tok or 0) / 1_000_000 * out_p


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BlindSpot:
    """One blind spot the auditor found in one layer."""
    layer:           str          # "cushion" | "cards" | "blends"
    blind_spot:      str          # what is missing / where it went slack
    why_it_matters:  str          # why this threatens advancing the cushion
    severity:        str = "medium"   # low | medium | high
    suggested_angle: str = ""     # a direction to address it — for the FUTURE commander, not acted on now

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayerAudit:
    """The auditor's verdict on one layer."""
    layer:        str
    blind_spots:  list[BlindSpot] = field(default_factory=list)
    cost_usd:     float           = 0.0
    ok:           bool            = True
    note:         str             = ""

    def to_dict(self) -> dict:
        return {
            "layer":       self.layer,
            "blind_spots": [b.to_dict() for b in self.blind_spots],
            "cost_usd":    round(self.cost_usd, 4),
            "ok":          self.ok,
            "note":        self.note,
        }


@dataclass
class AuditReport:
    """The full halo audit across all three layers."""
    cushion_audit: LayerAudit | None = None
    cards_audit:   LayerAudit | None = None
    blends_audit:  LayerAudit | None = None
    model:         str               = ""
    total_cost_usd: float            = 0.0

    def all_blind_spots(self) -> list[BlindSpot]:
        out: list[BlindSpot] = []
        for a in (self.cushion_audit, self.cards_audit, self.blends_audit):
            if a is not None:
                out.extend(a.blind_spots)
        return out

    def to_dict(self) -> dict:
        return {
            "cushion_audit": self.cushion_audit.to_dict() if self.cushion_audit else None,
            "cards_audit":   self.cards_audit.to_dict() if self.cards_audit else None,
            "blends_audit":  self.blends_audit.to_dict() if self.blends_audit else None,
            "model":         self.model,
            "total_cost_usd": round(self.total_cost_usd, 4),
        }


# ---------------------------------------------------------------------------
# Doctrine — shared frame + per-layer focus
# ---------------------------------------------------------------------------


_HALO_FRAME = """\
You are the HALO AUDITOR — an oversight layer that watches the pipeline and
finds its BLIND SPOTS. You do not answer the question, fix anything, or take
any action. You AUDIT, and you write down what is missing. A later layer will
decide what to do with your notes; that is not your job.

THE CUSHION — the user's problem, vision, hunches, and the QUESTION: the
explicit checkpoint this run must answer. Anchor your audit to the QUESTION.
{cushion}

Two things you hunt for:
  - BLIND SPOTS: what is genuinely MISSING — an angle not considered, a slot
    the structure implies but never fills, a question left unasked.
  - SLACK: where the work fell short — shallow, hand-wavy, restating instead
    of illuminating, quietly assuming something unproven.

DISCIPLINE:
  - Anchor everything to the QUESTION. A blind spot counts ONLY if it threatens
    ANSWERING the cushion's QUESTION. Ignore tangents.
  - Blind spots are unbounded — do NOT spill. Return the TOP {n} by severity,
    ranked. Be a ruthless skeptic, not a completist.
  - Each blind spot must be SPECIFIC. "Needs more detail" is rejected; name
    the exact absence.
  - suggested_angle is a direction for a FUTURE commander to consider — you do
    not act on it. Leave it crisp but do nothing with it.

{focus}

OUTPUT FORMAT — a single JSON object, nothing around it:
{{"blind_spots": [
  {{"blind_spot": "<the specific absence or slack>",
    "why_it_matters": "<how it threatens answering the QUESTION>",
    "severity": "low|medium|high",
    "suggested_angle": "<crisp direction for a later commander; no action now>"}}
]}}
"""

_CUSHION_FOCUS = """\
YOU ARE AUDITING THE CUSHION ITSELF — the question, before any work is done.
What does this question presuppose without examining? What angle is it blind
to? Where is it under-specified, or quietly assuming something? What would a
sharp adversarial reviewer say the question is failing to ask? This is the
question auditing its own framing for what it cannot see."""

_CARDS_FOCUS = """\
YOU ARE AUDITING THE WANDER OUTPUT — the cards the wanderers brought back.
What TERRITORY did they miss? Where do they OVERLAP or cluster on the same few
sources? Which angle of the cushion got NO card at all? Are they homogeneous —
all the same flavor of on-topic? What rich vein did the wander walk straight
past? Audit coverage and diversity, not individual card correctness.

THE CARDS:
{cards}"""

_BLENDS_FOCUS = """\
YOU ARE AUDITING THE BLENDS — the candidate new lanes the blender produced.
What HOLES do these lanes leave? What collision did the blender NOT try that it
should have? Where is a blend's reasoning slack, hand-wavy, or resting on an
unproven assumption? What blind spot would sink one of these lanes in practice?
Audit what the lanes miss, not whether they are novel.

THE BLENDS:
{blends}"""


def _bounded(severity_rank: list[BlindSpot]) -> list[BlindSpot]:
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(severity_rank, key=lambda b: order.get(b.severity, 1))[:MAX_BLIND_SPOTS]


def _parse(raw: str, layer: str) -> list[BlindSpot]:
    parsed = _parse_json_safely(raw, default={})
    if not isinstance(parsed, dict):
        return []
    out: list[BlindSpot] = []
    for e in parsed.get("blind_spots", []) or []:
        if not isinstance(e, dict):
            continue
        bs = str(e.get("blind_spot", "")).strip()
        if not bs:
            continue
        sev = str(e.get("severity", "medium")).strip().lower()
        if sev not in ("low", "medium", "high"):
            sev = "medium"
        out.append(BlindSpot(
            layer=layer, blind_spot=bs,
            why_it_matters=str(e.get("why_it_matters", "")),
            severity=sev,
            suggested_angle=str(e.get("suggested_angle", "")),
        ))
    return _bounded(out)


async def _audit(*, client: LLMClient, model: str, layer: str, cushion: str,
                 focus: str) -> LayerAudit:
    system = compose_system_prompt(
        _HALO_FRAME.format(cushion=cushion[:1400], n=MAX_BLIND_SPOTS, focus=focus),
        mode="halo_auditor",
    )
    audit = LayerAudit(layer=layer)
    resp = await client.call(
        system_prompt=system,
        user_message=f"Audit the {layer} now. Output the JSON object only.",
        domain=AUDITOR_DOMAIN, concept=f"audit_{layer}", model=model,
        max_tokens=MAX_TOKENS_AUDIT, temperature=AUDIT_TEMPERATURE,
    )
    audit.cost_usd = _cost(model, resp.input_tokens, resp.output_tokens)
    if not resp.success:
        audit.ok = False
        audit.note = f"audit call failed: {(resp.error or '')[:160]}"
        return audit
    audit.blind_spots = _parse(resp.content, layer)
    return audit


# ---------------------------------------------------------------------------
# Per-layer entrypoints (call at each checkpoint, step by step)
# ---------------------------------------------------------------------------


async def audit_cushion(*, cushion: str, client: LLMClient, model: str = AUDITOR_MODEL) -> LayerAudit:
    """Checkpoint 1 — audit the question itself, before any work."""
    return await _audit(client=client, model=model, layer="cushion",
                        cushion=cushion, focus=_CUSHION_FOCUS)


async def audit_cards(*, cushion: str, cards: list[ArticulatedCard],
                      client: LLMClient, model: str = AUDITOR_MODEL) -> LayerAudit:
    """Checkpoint 2 — audit the wander's coverage + diversity."""
    blob = "\n".join(
        f"- [{c.report_id}] from {c.source_shape}: {c.bridge[:200]}"
        for c in cards
    ) or "(no cards)"
    return await _audit(client=client, model=model, layer="cards",
                        cushion=cushion, focus=_CARDS_FOCUS.format(cards=blob[:6000]))


async def audit_blends(*, cushion: str, blends: list[Blend],
                       client: LLMClient, model: str = AUDITOR_MODEL) -> LayerAudit:
    """Checkpoint 3 — audit the blends for the holes their lanes leave."""
    blob = "\n".join(
        f"- [{b.blend_id}] {b.thesis[:240]}"
        for b in blends
    ) or "(no blends)"
    return await _audit(client=client, model=model, layer="blends",
                        cushion=cushion, focus=_BLENDS_FOCUS.format(blends=blob[:6000]))


# ---------------------------------------------------------------------------
# Convenience — audit a whole completed run at once (post-run halo pass)
# ---------------------------------------------------------------------------


async def run_halo_audit(
    *,
    cushion: str,
    cards:   list[ArticulatedCard],
    blends:  list[Blend],
    client:  LLMClient,
    model:   str = AUDITOR_MODEL,
) -> AuditReport:
    """Run all three audits over a completed run. The three are independent
    (each watches a different artifact) so they run concurrently here; in a
    live run they'd fire step-by-step at each checkpoint instead.
    """
    report = AuditReport(model=model)
    tasks = [audit_cushion(cushion=cushion, client=client, model=model)]
    if cards:
        tasks.append(audit_cards(cushion=cushion, cards=cards, client=client, model=model))
    if blends:
        tasks.append(audit_blends(cushion=cushion, blends=blends, client=client, model=model))

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    by_layer = {a.layer: a for a in results}
    report.cushion_audit = by_layer.get("cushion")
    report.cards_audit   = by_layer.get("cards")
    report.blends_audit  = by_layer.get("blends")
    report.total_cost_usd = sum(a.cost_usd for a in results)
    log.info("[halo] audit complete: %d blind spots across %d layers, $%.4f, %.0fms",
             len(report.all_blind_spots()), len(results), report.total_cost_usd,
             (time.time() - t0) * 1000)
    return report


__all__ = (
    "AUDITOR_MODEL",
    "MAX_BLIND_SPOTS",
    "BlindSpot",
    "LayerAudit",
    "AuditReport",
    "audit_cushion",
    "audit_cards",
    "audit_blends",
    "run_halo_audit",
)
