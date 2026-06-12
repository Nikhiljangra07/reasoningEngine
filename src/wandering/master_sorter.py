"""
Master Sorter — the "dirt sorter" tributary above the agent reports.

PURPOSE
-------
This module is the SORTER tributary that runs in place of the
master_synthesizer when `build_dossier(pipeline_mode="sorter")` is set.
It does NOT fuse, NOT synthesize, NOT merge. It CLASSIFIES.

The premise: convergent frontier models are good at one thing —
recognition against a vast training memory. They are bad at the
thing the synthesizer asked them to do (novel cross-card fusion).
The sorter inverts the assignment: give the big model a task that
matches its nature. Classify each card into one of three bins
without transforming it.

THE THREE BUCKETS
-----------------
- known     — the card matches prior published work the sorter can NAME.
              Requires a non-empty `prior_work_name` AND `reference`,
              otherwise the item is demoted to `unplaced`. This forces
              the model to put a checkable label on its recognition
              rather than hand-waving "yes I know this."
- invalid   — the card contradicts established fact or contradicts
              itself. This is "the dirt" — the visible failure mode
              the sorter is for.
- unplaced  — the card matches nothing the sorter can name AND cannot
              be refuted. The candidate gold lives here, but so does
              well-dressed nonsense. Separating gold from nonsense
              inside this bin is the human's job downstream.

The sorter is FORBIDDEN to:
  - merge cards
  - rewrite card content
  - synthesize across cards
  - infer relationships between cards
The original card content passes through unchanged on every item.

ROLE IN THE PIPELINE
--------------------
The synthesizer ran AFTER the dossier was assembled, producing
3-5 cross-card fusions. The sorter runs at the same point but
emits a `SortedReport` instead — every card classified, nothing
fused. Downstream consumers branch on which artifact is present
(`Dossier.master_sorted` vs `Dossier.master_synthesis`).

This is the "dam" architecture: the synthesizer module is
untouched; the dossier.py call site routes between the two by
parameter. Both code paths remain importable and testable.

HARD COST CAP
-------------
Same dollar ceiling discipline as the synthesizer. Every call is
cost-checked AFTER it returns; cumulative spend > ceiling raises
MasterSortBudgetExceeded and the orchestrator returns whatever
partial sort was assembled.

MODEL
-----
Fable 5 only (anthropic/claude-fable-5). One seat, single-pass.
The model slug is registered in src/llm/provider_map.py PRICING.
No second seat — sorting does not need disagreement, only
recognition. The downstream angular machinery (the agent layer
itself) produced the disagreement upstream.

ISOLATION
---------
Imports: dossier types + LLMClient + provider_map for pricing +
stdlib + json-parsing helpers from master_synthesizer (kept DRY).
Does NOT import call_tracker. Runs AFTER run_wandering_session has
fully returned, same as the synthesizer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable

from src.identity import compose_system_prompt
from src.llm.client import LLMClient, LLMResponse
from src.llm.provider_map import get_pricing
from src.wandering.articulate import ArticulatedCard
from src.wandering.cushion import CushionGraph
from src.wandering.master_synthesizer import (
    _extract_json,
    _parse_json_safely,
    _strip_code_fences,
)
from src.wandering.synthesis import SynthesisMap

log = logging.getLogger("constellax.wandering.master_sorter")


# ---------------------------------------------------------------------------
# Model slug pinning — explicit, not via resolve_model()
# ---------------------------------------------------------------------------
FABLE_SEAT_MODEL = "anthropic/claude-fable-5"

SORTER_DOMAIN  = "master_sorter"
SORTER_CONCEPT = "master_sort"

#: Single-pass output cap. The sorter emits one JSON object holding
#: three arrays (known / invalid / unplaced); for typical 15-30 card
#: dossiers this lands well under 8192 tokens. Matched to synthesizer
#: R3 cap for headroom on the larger sessions.
MAX_TOKENS_SORT = 8192


# ---------------------------------------------------------------------------
# Hard cost cap
# ---------------------------------------------------------------------------

DEFAULT_COST_CEILING_USD = 8.00


class MasterSortBudgetExceeded(Exception):
    """Raised when cumulative spend would exceed `cost_ceiling_usd`.

    The orchestrator catches this and returns the partial result so far
    instead of crashing — partial output > none.
    """


def _call_cost_usd(model_slug: str, response: LLMResponse) -> float:
    """Compute USD spend for one LLMResponse using provider_map pricing."""
    in_price, out_price = get_pricing(model_slug)
    in_cost  = (response.input_tokens  or 0) / 1_000_000 * in_price
    out_cost = (response.output_tokens or 0) / 1_000_000 * out_price
    return in_cost + out_cost


# ---------------------------------------------------------------------------
# Bucket enum
# ---------------------------------------------------------------------------


class Bucket(str, Enum):
    """Which sorter bin the card landed in."""
    KNOWN    = "known"
    INVALID  = "invalid"
    UNPLACED = "unplaced"


# ---------------------------------------------------------------------------
# Per-item dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CardSnapshot:
    """Verbatim card content the sorter classified.

    Preserved on every sorted item so downstream consumers don't have
    to re-join against the dossier. Content is COPIED, not referenced
    — the sorter must never transform the source card, and freezing
    a snapshot here makes that invariant inspectable post-hoc.
    """
    report_id:    str
    agent_id:     str
    spark:        str
    bridge:       str
    source_shape: str
    use:          str
    limit:        str

    @classmethod
    def from_card(cls, card: ArticulatedCard) -> "CardSnapshot":
        return cls(
            report_id    = card.report_id,
            agent_id     = card.agent_id or "",
            spark        = card.spark,
            bridge       = card.bridge,
            source_shape = card.source_shape,
            use          = card.use,
            limit        = card.limit,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KnownItem:
    """A card the sorter matched to NAMED prior work.

    `prior_work_name` and `reference` are REQUIRED non-empty. If the
    model returns an item in the known bucket without either, the
    parser demotes it to `unplaced` — the citation requirement is
    enforced by the parser, not by trust.
    """
    card:            CardSnapshot
    prior_work_name: str             # e.g. "Constitutional AI"
    reference:       str             # e.g. "Bai et al. 2022, arxiv 2212.08073"
    confidence:      float = 0.0     # sorter's self-reported 0..1
    reasoning:       str   = ""      # 1-2 sentences explaining the match

    def to_dict(self) -> dict:
        return {
            "card":            self.card.to_dict(),
            "prior_work_name": self.prior_work_name,
            "reference":       self.reference,
            "confidence":      self.confidence,
            "reasoning":       self.reasoning,
        }


@dataclass
class InvalidItem:
    """A card the sorter judged to contradict established fact OR
    contradict itself.

    This is "the dirt" the sorter exists to surface. The contradiction
    must be specific — what does the card conflict with, and how. A
    bare "this is wrong" is rejected by the parser.
    """
    card:        CardSnapshot
    contradicts: str             # what it conflicts with (named fact or self)
    reasoning:   str             # 1-3 sentences explaining the contradiction
    confidence:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "card":        self.card.to_dict(),
            "contradicts": self.contradicts,
            "reasoning":   self.reasoning,
            "confidence":  self.confidence,
        }


@dataclass
class UnplacedItem:
    """A card the sorter cannot match to known work AND cannot refute.

    The candidate gold sits in this bucket. So does well-dressed
    nonsense. The sorter's job ends at the bin; the human's job is
    to read the unplaced items downstream and decide which are
    genuine novelty.

    `why_unplaced` records the sorter's reasoning ("can't match to
    X; can't refute because Y") so the human reading the dossier
    has the sorter's working visible.
    """
    card:         CardSnapshot
    why_unplaced: str             # rationale for can't-match + can't-refute
    confidence:   float = 0.0     # sorter's confidence the item is genuinely unplaced

    def to_dict(self) -> dict:
        return {
            "card":         self.card.to_dict(),
            "why_unplaced": self.why_unplaced,
            "confidence":   self.confidence,
        }


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass
class SortedReport:
    """Container the sorter returns. Three buckets + cost/round telemetry.

    Empty `known` / `invalid` / `unplaced` is a valid state — the sorter
    is allowed to produce any distribution including all-in-one-bucket
    when the dossier genuinely lands that way.
    """
    known:               list[KnownItem]    = field(default_factory=list)
    invalid:             list[InvalidItem]  = field(default_factory=list)
    unplaced:            list[UnplacedItem] = field(default_factory=list)
    total_cost_usd:      float              = 0.0
    cost_ceiling_usd:    float              = DEFAULT_COST_CEILING_USD
    truncated_by_budget: bool               = False
    truncation_reason:   str                = ""
    #: Cards the model emitted that failed parser validation — typically
    #: "known" items missing prior_work_name or reference (demoted to
    #: unplaced) or items with no card match. Kept as a transparency
    #: surface so an auditor can answer "how trustworthy was this sort?"
    parser_demotions:    list[dict]         = field(default_factory=list)
    #: Per-call audit — {round, model, in_tok, out_tok, cost_usd, ms, ok}
    call_log:            list[dict]         = field(default_factory=list)
    #: Input card count vs total classified (sanity invariant: should match)
    input_card_count:    int                = 0
    classified_count:    int                = 0
    #: Cards the model dropped entirely (never appeared in any bucket).
    #: Surveillance metric — a non-empty list means the sort lost cards.
    dropped_report_ids:  list[str]          = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "known":               [k.to_dict() for k in self.known],
            "invalid":             [i.to_dict() for i in self.invalid],
            "unplaced":            [u.to_dict() for u in self.unplaced],
            "total_cost_usd":      round(self.total_cost_usd, 4),
            "cost_ceiling_usd":    self.cost_ceiling_usd,
            "truncated_by_budget": self.truncated_by_budget,
            "truncation_reason":   self.truncation_reason,
            "parser_demotions":    list(self.parser_demotions),
            "call_log":            list(self.call_log),
            "input_card_count":    self.input_card_count,
            "classified_count":    self.classified_count,
            "dropped_report_ids":  list(self.dropped_report_ids),
        }


# ---------------------------------------------------------------------------
# Progress emission — keeps the UX communicative during the sort
# ---------------------------------------------------------------------------


@dataclass
class MasterSortProgress:
    """Live progress reference for an in-flight master sort.

    The sorter is single-pass so there are only three events:
    `starting`, `sort_complete`, and `complete`. Callers can subclass
    or override `on_event` for SSE / logger / print rendering.
    """
    on_event: Callable[[str, dict], None] | None = None
    events:   list[dict] = field(default_factory=list)

    def emit(self, name: str, payload: dict | None = None) -> None:
        payload = payload or {}
        entry = {"name": name, "ts": time.time(), **payload}
        self.events.append(entry)
        log.info("[master_sort] %s %s", name, payload)
        if self.on_event is not None:
            try:
                self.on_event(name, payload)
            except Exception as e:  # pragma: no cover — UX hook must not crash
                log.warning("progress on_event raised (ignored): %s", e)


# ---------------------------------------------------------------------------
# LLM-call helper with cost-cap enforcement
# ---------------------------------------------------------------------------


async def _call_with_budget(
    *,
    client:        LLMClient,
    system_prompt: str,
    user_message:  str,
    model_slug:    str,
    result:        SortedReport,
    max_tokens:    int | None = None,
) -> LLMResponse:
    """One LLM call, recorded against the cost ceiling.

    Same discipline as master_synthesizer._call_with_budget — cost is
    checked AFTER the call returns; cumulative spend > ceiling raises
    MasterSortBudgetExceeded and the orchestrator returns the partial
    sort.
    """
    response: LLMResponse = await client.call(
        system_prompt=system_prompt,
        user_message=user_message,
        domain=SORTER_DOMAIN,
        concept=SORTER_CONCEPT,
        model=model_slug,
        max_tokens=max_tokens,
    )
    cost_usd = _call_cost_usd(model_slug, response)
    result.total_cost_usd += cost_usd
    result.call_log.append({
        "round":    "sort",
        "model":    model_slug,
        "in_tok":   response.input_tokens,
        "out_tok":  response.output_tokens,
        "cost_usd": round(cost_usd, 4),
        "ms":       round(response.latency_ms or 0.0, 1),
        "ok":       response.success,
        "err":      (response.error or "")[:200] if not response.success else "",
    })

    if result.total_cost_usd > result.cost_ceiling_usd:
        result.truncated_by_budget = True
        result.truncation_reason = (
            f"cumulative spend ${result.total_cost_usd:.2f} exceeds "
            f"ceiling ${result.cost_ceiling_usd:.2f} after sort call"
        )
        log.warning("master_sort budget exceeded: %s", result.truncation_reason)
        raise MasterSortBudgetExceeded(result.truncation_reason)

    return response


# ---------------------------------------------------------------------------
# Doctrine + prompt
# ---------------------------------------------------------------------------


_DOCTRINE_PREAMBLE = """\
You are the SORTER seat of Constellax's Wandering Room.

You are NOT a synthesizer. You do NOT fuse cards. You do NOT merge
cards. You do NOT improve, condense, or rewrite cards. You do NOT
infer relationships across cards. Your only job is to CLASSIFY each
card into one of three bins.

Three bins, with strict definitions:

KNOWN — the card's central claim matches PRIOR PUBLISHED WORK you can
        NAME. To place a card here you MUST provide:
          - prior_work_name: the specific paper, theory, framework,
            or concept it matches (e.g. "Constitutional AI",
            "Eigenvalue decomposition", "Conway's Game of Life")
          - reference: a checkable pointer (e.g. "Bai et al. 2022,
            arxiv 2212.08073", "Strang Ch. 6", "Conway 1970")
        If you cannot name the prior work or provide a reference,
        you MUST place the card in UNPLACED instead. A bare "yes I
        know this" is REJECTED. Calling something known without a
        name is hallucinated recognition.

INVALID — the card contradicts established fact OR contradicts
          itself. Be specific: name what it contradicts and how.
          "This card claims X but established physics says Y" — that
          is invalid. "This feels wrong" — that is NOT invalid.

UNPLACED — the card matches nothing you can name AND you cannot
           refute it. This is the residual. It contains both
           genuine novelty and well-dressed nonsense. You are
           NOT responsible for separating those two — the human
           reads unplaced items downstream. Your job ends at the
           bin. Record `why_unplaced` so the human sees your
           reasoning ("can't match to X family; can't refute
           because Y dimension is untested").

INVARIANTS:
  - Every input card MUST appear in exactly one bin.
  - You MUST NOT modify the card content. Original content passes
    through verbatim.
  - You MUST NOT add cards that were not in the input.
  - You MUST NOT merge two cards into one bin entry.
  - Confidence is your self-reported number 0..1 of how sure you
    are about the bin. Honest low confidence is allowed.

OUTPUT FORMAT: a single JSON object with three arrays — `known`,
`invalid`, `unplaced` — each containing per-card entries in the
schema specified in the user message. Output ONLY the JSON. No
prose around it.
"""


def _build_sort_payload(
    cushion: CushionGraph | None,
    cards:   list[ArticulatedCard],
    synthesis_map: SynthesisMap | None,
) -> str:
    """Build the user-message payload for the single sort pass.

    Includes the cushion problem statement (so the sorter knows the
    domain context) and every card's id + content. The schema
    specification is included verbatim so the model knows exactly
    what fields each bucket expects.
    """
    problem = ""
    if cushion is not None and cushion.raw_input is not None:
        problem = cushion.raw_input.problem.content[:500]

    card_blocks = []
    for c in cards:
        card_blocks.append({
            "report_id":    c.report_id,
            "agent_id":     c.agent_id or "",
            "spark":        c.spark,
            "source_shape": c.source_shape,
            "bridge":       c.bridge,
            "use":          c.use,
            "limit":        c.limit,
        })

    schema_spec = {
        "known": [{
            "report_id":       "<copy from input>",
            "prior_work_name": "<REQUIRED, non-empty>",
            "reference":       "<REQUIRED, non-empty>",
            "confidence":      "<float 0..1>",
            "reasoning":       "<1-2 sentences>",
        }],
        "invalid": [{
            "report_id":   "<copy from input>",
            "contradicts": "<REQUIRED, non-empty: what it conflicts with>",
            "reasoning":   "<1-3 sentences explaining the contradiction>",
            "confidence":  "<float 0..1>",
        }],
        "unplaced": [{
            "report_id":    "<copy from input>",
            "why_unplaced": "<REQUIRED, non-empty: can't-match + can't-refute reasoning>",
            "confidence":   "<float 0..1>",
        }],
    }

    payload = {
        "problem_context": problem,
        "card_count":      len(cards),
        "cards":           card_blocks,
        "output_schema":   schema_spec,
        "instruction": (
            "Classify EVERY card into exactly ONE bin. Output the JSON "
            "object only. No prose around it. Remember: known requires "
            "a named prior_work_name AND reference, or the card belongs "
            "in unplaced."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Parser — enforces the citation invariant + tracks dropped cards
# ---------------------------------------------------------------------------


def _parse_sort_response(
    raw:    str,
    cards:  list[ArticulatedCard],
    result: SortedReport,
) -> None:
    """Parse the LLM JSON into typed bucket items and populate `result`.

    Enforces invariants:
      - known items missing prior_work_name OR reference are demoted
        to unplaced and the demotion is logged on result.parser_demotions
      - invalid items missing `contradicts` are demoted to unplaced
      - unplaced items missing `why_unplaced` are kept but flagged
        with a "(missing rationale)" placeholder
      - items referencing report_ids not in the input dossier are
        dropped (logged on parser_demotions)
      - input report_ids never seen in the output are recorded in
        result.dropped_report_ids
    """
    card_by_id: dict[str, ArticulatedCard] = {c.report_id: c for c in cards}
    parsed = _parse_json_safely(raw, default={})
    if not isinstance(parsed, dict):
        result.parser_demotions.append({
            "reason": "top_level_not_dict",
            "raw_type": type(parsed).__name__,
        })
        return

    seen_ids: set[str] = set()

    raw_known    = parsed.get("known", [])    or []
    raw_invalid  = parsed.get("invalid", [])  or []
    raw_unplaced = parsed.get("unplaced", []) or []

    # --- KNOWN bucket — strict citation requirement
    if isinstance(raw_known, list):
        for entry in raw_known:
            if not isinstance(entry, dict):
                result.parser_demotions.append({"reason": "known_entry_not_dict", "entry": str(entry)[:200]})
                continue
            rid = str(entry.get("report_id", ""))
            card = card_by_id.get(rid)
            if card is None:
                result.parser_demotions.append({"reason": "known_unknown_report_id", "report_id": rid})
                continue
            prior = str(entry.get("prior_work_name", "")).strip()
            ref   = str(entry.get("reference", "")).strip()
            if not prior or not ref:
                # Demote to unplaced — citation requirement failed
                result.parser_demotions.append({
                    "reason":    "known_missing_citation",
                    "report_id": rid,
                    "had_prior": bool(prior),
                    "had_ref":   bool(ref),
                })
                result.unplaced.append(UnplacedItem(
                    card=CardSnapshot.from_card(card),
                    why_unplaced=(
                        f"Demoted from known: model claimed match but did not "
                        f"name prior work or reference. Original reasoning: "
                        f"{str(entry.get('reasoning', ''))[:200]}"
                    ),
                    confidence=float(entry.get("confidence", 0.0) or 0.0),
                ))
                seen_ids.add(rid)
                continue
            result.known.append(KnownItem(
                card=CardSnapshot.from_card(card),
                prior_work_name=prior,
                reference=ref,
                confidence=float(entry.get("confidence", 0.0) or 0.0),
                reasoning=str(entry.get("reasoning", "")),
            ))
            seen_ids.add(rid)

    # --- INVALID bucket — must name what it contradicts
    if isinstance(raw_invalid, list):
        for entry in raw_invalid:
            if not isinstance(entry, dict):
                result.parser_demotions.append({"reason": "invalid_entry_not_dict", "entry": str(entry)[:200]})
                continue
            rid = str(entry.get("report_id", ""))
            card = card_by_id.get(rid)
            if card is None:
                result.parser_demotions.append({"reason": "invalid_unknown_report_id", "report_id": rid})
                continue
            contradicts = str(entry.get("contradicts", "")).strip()
            if not contradicts:
                # Demote to unplaced — bare "this is wrong" is rejected
                result.parser_demotions.append({
                    "reason":    "invalid_missing_contradicts",
                    "report_id": rid,
                })
                result.unplaced.append(UnplacedItem(
                    card=CardSnapshot.from_card(card),
                    why_unplaced=(
                        f"Demoted from invalid: model flagged it but did not "
                        f"name what it contradicts. Original reasoning: "
                        f"{str(entry.get('reasoning', ''))[:200]}"
                    ),
                    confidence=float(entry.get("confidence", 0.0) or 0.0),
                ))
                seen_ids.add(rid)
                continue
            result.invalid.append(InvalidItem(
                card=CardSnapshot.from_card(card),
                contradicts=contradicts,
                reasoning=str(entry.get("reasoning", "")),
                confidence=float(entry.get("confidence", 0.0) or 0.0),
            ))
            seen_ids.add(rid)

    # --- UNPLACED bucket — flag missing rationale but keep
    if isinstance(raw_unplaced, list):
        for entry in raw_unplaced:
            if not isinstance(entry, dict):
                result.parser_demotions.append({"reason": "unplaced_entry_not_dict", "entry": str(entry)[:200]})
                continue
            rid = str(entry.get("report_id", ""))
            card = card_by_id.get(rid)
            if card is None:
                result.parser_demotions.append({"reason": "unplaced_unknown_report_id", "report_id": rid})
                continue
            why = str(entry.get("why_unplaced", "")).strip() or "(missing rationale)"
            result.unplaced.append(UnplacedItem(
                card=CardSnapshot.from_card(card),
                why_unplaced=why,
                confidence=float(entry.get("confidence", 0.0) or 0.0),
            ))
            seen_ids.add(rid)

    # Surveillance: input cards the sorter dropped entirely
    for c in cards:
        if c.report_id not in seen_ids:
            result.dropped_report_ids.append(c.report_id)

    result.classified_count = len(seen_ids)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def master_sort(
    *,
    cushion:          CushionGraph | None,
    cards:            list[ArticulatedCard],
    synthesis_map:    SynthesisMap | None,
    client:           LLMClient,
    progress:         MasterSortProgress | None = None,
    cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
    fable_model:      str = FABLE_SEAT_MODEL,
) -> SortedReport:
    """Sort a dossier's articulated cards into three bins.

    Single-pass, single-seat (Fable 5). No fusion, no synthesis. Each
    card lands in exactly one of known / invalid / unplaced.

    Empty input → empty result (no LLM calls fired).

    Budget enforcement matches the synthesizer: cumulative spend
    over `cost_ceiling_usd` raises MasterSortBudgetExceeded; the
    caller surfaces whatever partial sort was assembled.
    """
    result = SortedReport(cost_ceiling_usd=cost_ceiling_usd)
    progress = progress or MasterSortProgress()
    result.input_card_count = len(cards)

    if not cards:
        progress.emit("empty_input", {"reason": "no cards provided"})
        return result

    progress.emit("starting", {
        "card_count":       len(cards),
        "cost_ceiling_usd": cost_ceiling_usd,
        "fable_model":      fable_model,
    })

    system = compose_system_prompt(_DOCTRINE_PREAMBLE, mode="master_sorter")
    payload = _build_sort_payload(cushion, cards, synthesis_map)

    try:
        progress.emit("sort_started", {"seat": "fable"})
        response = await _call_with_budget(
            client=client,
            system_prompt=system,
            user_message=payload,
            model_slug=fable_model,
            result=result,
            max_tokens=MAX_TOKENS_SORT,
        )
        _parse_sort_response(response.content, cards, result)
        progress.emit("sort_complete", {
            "known_count":         len(result.known),
            "invalid_count":       len(result.invalid),
            "unplaced_count":      len(result.unplaced),
            "demotions":           len(result.parser_demotions),
            "dropped":             len(result.dropped_report_ids),
            "cumulative_cost_usd": round(result.total_cost_usd, 4),
        })
    except MasterSortBudgetExceeded:
        # Partial result already populated; truncation flags set.
        pass

    progress.emit("complete", {
        "total_cost_usd":    round(result.total_cost_usd, 4),
        "input_cards":       result.input_card_count,
        "classified_cards":  result.classified_count,
        "truncated":         result.truncated_by_budget,
    })
    return result


__all__ = (
    "FABLE_SEAT_MODEL",
    "DEFAULT_COST_CEILING_USD",
    "MAX_TOKENS_SORT",
    "MasterSortBudgetExceeded",
    "Bucket",
    "CardSnapshot",
    "KnownItem",
    "InvalidItem",
    "UnplacedItem",
    "SortedReport",
    "MasterSortProgress",
    "master_sort",
)
