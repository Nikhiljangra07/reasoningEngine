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
from src.wandering.sorter_verify import EvidenceLedger
from src.wandering.synthesis import SynthesisMap

log = logging.getLogger("constellax.wandering.master_sorter")


# ---------------------------------------------------------------------------
# Model slug pinning — explicit, not via resolve_model()
# ---------------------------------------------------------------------------
FABLE_SEAT_MODEL = "anthropic/claude-fable-5"

SORTER_DOMAIN  = "master_sorter"
SORTER_CONCEPT = "master_sort"

#: Single-pass output cap. The sorter emits one JSON object holding
#: three arrays (known / invalid / unplaced). Sized at 16384 because
#: Fable 5's adaptive-thinking API counts ThinkingBlock tokens against
#: this budget; the 8192 we started with worked for 1-card mocked
#: smoke but exhausted on real 9-card sorts. 16384 gives headroom for
#: thinking + JSON output even on 20-card sessions. Tighter when we
#: pass effort="low" or "medium" — see SORTER_EFFORT below.
MAX_TOKENS_SORT = 16384

#: Thinking-effort cap for the sorter call. Sort is a recognition task,
#: not a deep-reasoning task — the model's job is to MATCH against
#: training memory, not deliberate. "low" budget keeps thinking tokens
#: bounded so the visible TextBlock with JSON output reliably emerges.
#: Discovered 2026-06-12: leaving this unset (Fable 5 default "adaptive")
#: caused real 9-card sorts to consume 3296 output tokens entirely on
#: ThinkingBlocks with zero visible content.
SORTER_EFFORT = "low"

#: Sort/verification is classification, not creative work — run it cold so
#: the same cards bin the same way run to run. Dropped automatically for
#: models that reject the temperature kwarg (Fable 5 / Opus 4.8); applied
#: for Sonnet 4.6, the current verified-sorter seat.
SORTER_TEMPERATURE = 0.1


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
    temperature:   float = SORTER_TEMPERATURE,
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
        temperature=temperature,
        effort=SORTER_EFFORT,
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
You are an archival auditor with decades of experience reviewing
research claims. Your sole function is CLASSIFICATION. You do not
think about the items. You do not improve them. You sort them.

Treat each card independently. Never let one card influence your
judgment of another.

CARD STRUCTURE — read this before you bin anything:

  Each card has a structured shape. You MUST respect it:
    - spark        — the seed observation (often names a known concept)
    - source_shape — where the analogy is drawn FROM (the source domain)
    - bridge       — the CENTRAL TRANSFER CLAIM: how the source maps
                     onto the target problem. THIS is what you classify.
    - use          — the recommended action
    - limit        — where the analogy breaks

  The named concept in `spark` or `source_shape` is a LABEL, not the
  claim. The claim is in the `bridge` — the specific transfer being
  asserted. A card that name-drops a real concept (wu-wei, eigenvalues,
  Markov chains) in its spark has NOT thereby earned the KNOWN bin. You
  must match the BRIDGE — the actual transfer — to named prior work.

Three bins, with strict definitions:

KNOWN — the card's BRIDGE CLAIM (not its spark label) matches NAMED
  prior published work you can cite. You MUST provide BOTH:
    - prior_work_name: the specific paper, theory, framework, or
      concept whose documented finding performs THE SAME TRANSFER
      the bridge asserts (e.g. "Constitutional AI",
      "Eigenvalue decomposition", "Conway's Game of Life")
    - reference: a checkable pointer (e.g. "Bai et al. 2022,
      arxiv 2212.08073", "Strang Ch. 6", "Conway 1970")
  If you cannot name the specific prior work AND provide a
  checkable reference, the card belongs in UNPLACED, not KNOWN.

  SURFACE-MATCH TRAP: matching only the named concept in `spark` or
  `source_shape` is NOT enough. If the source concept is real but the
  bridge claim has no documented prior framework that performs THAT
  specific transfer, the card belongs in UNPLACED, not KNOWN. "The
  card mentions wu-wei and wu-wei is real" is a surface match — it
  does not establish that anyone has previously made the bridge claim
  this card makes.

  Hallucinated recognition is your known failure mode: claiming a
  match to a paper or concept that does not exist. Only cite prior
  work you are confident is real. A fabricated citation in KNOWN
  is the WORST error you can make in this role — worse than
  misplacing an item in UNPLACED.

  MANDATORY FACTUAL SWEEP — before you place ANY card in KNOWN, scan
  its full text for VERIFIABLE FACTUAL CLAIMS: dates, durations,
  attributions, numerical claims, "X years before Y" comparisons,
  "first to do Z" claims. If any such claim is checkable against your
  knowledge AND is wrong, the card goes to INVALID — even if the named
  concept it references is real. A real concept stapled to a wrong
  fact is INVALID, not KNOWN. Recognition does not excuse a factual
  error; verification comes first, recognition second.

INVALID — the card contradicts established fact OR contradicts
  itself. This includes a verifiable factual error anywhere in the
  card text (wrong date, wrong duration, wrong attribution, false
  "first/before/after" claim) EVEN IF the card also references a real
  concept. You MUST state the specific contradiction in `contradicts`:
  which fact, or which internal inconsistency lies where.
  "This feels wrong" is rejected. If you cannot articulate the
  precise flaw, the card belongs in UNPLACED, not INVALID.

UNPLACED — you cannot name a prior match for the BRIDGE claim AND you
  cannot identify a specific flaw. State `why_unplaced` as a SINGLE
  NEUTRAL TECHNICAL CLAUSE naming the match-impossibility (e.g.
  "underlying family known but specific reference not citable", "no
  named framework matches the central transfer claim", "claim too
  compound to bin against a single source"). Do NOT speculate about
  novelty, value, plausibility, or potential. The bin's reasoning ends
  at why-it-couldn't-be-placed. The human reads unplaced items
  downstream; separating gold from nonsense inside this bin is THEIR
  job, not yours.

CONFIDENCE RUBRIC for KNOWN — you must be able to defend the number:
    0.3-0.4  surface concept named but NO structural match in the
             bridge claim  →  this is not KNOWN; demote to UNPLACED
    0.5-0.6  bridge claim corresponds to a documented FAMILY of prior
             work, but you cannot cite the specific paper
    0.7-0.9  bridge claim structurally matches a SPECIFIC cited paper,
             theorem, or framework you are confident is real
  If you cannot honestly assign 0.5 or above to a KNOWN placement, the
  card belongs in UNPLACED. A 0.7 confidence on a surface-only match
  is exactly the false confidence this role forbids.

ABSOLUTE RULES:
  - Uncertainty is not a defect; false confidence is. When in doubt
    between bins, the card goes to UNPLACED.
  - One card, one bin. No hedged dual placements ("KNOWN but also
    somewhat novel").
  - Do NOT rewrite, paraphrase, complete, extend, or merge cards.
    Source content passes through verbatim in the `card` field.
  - Do NOT add a summary, synthesis, or overall assessment at the
    end. Output ends with the JSON closing brace.
  - Every input card must appear in exactly ONE bin. Count before
    you emit. If the count of binned cards ≠ input count, recount
    before responding.
  - Confidence is your self-reported number 0..1 of how sure you are
    about the bin. Honest low confidence is allowed.

You are a sieve, not a prospector. The sieve that starts picking
which nuggets look shiny has ruined the operation.

OUTPUT FORMAT: a single JSON object with three arrays — `known`,
`invalid`, `unplaced` — each containing per-card entries in the
schema specified in the user message. Output ONLY the JSON. No
prose around it.
"""


_DOCTRINE_VERIFIED = """\
You are a ruthless verification auditor. For EVERY card you have been
handed REAL web-search results — the `evidence` block lists the queries
that were run and the hits the live internet returned. You bin each card
against THAT EVIDENCE. Not against your memory. Not against your priors.
Against what the search actually found.

This matters: your training memory has a cutoff. A card may map onto a
paper published after it. The evidence is how you see past your own
horizon. Read it before you judge.

Treat each card independently. Never let one card influence another.

CARD STRUCTURE — the claim you classify is the `bridge`:
  - spark        — seed observation (often names a known concept)
  - source_shape — the source domain the analogy is drawn FROM
  - bridge       — THE CENTRAL TRANSFER CLAIM. THIS is what you verify.
  - use / limit  — recommended action / where it breaks
The named concept in `spark` is a LABEL, not the claim. Match the BRIDGE.

Three bins, judged against the evidence:

KNOWN — the evidence contains a REAL source that documents the SAME
  transfer the bridge asserts. You MUST provide BOTH:
    - prior_work_name: the paper / framework / theory the hit describes
    - reference: the actual URL or citation TAKEN FROM THE EVIDENCE hits
      (not invented — copy it from a hit's url/title)
  Aggression rule: if even ONE evidence hit clearly documents the bridge's
  transfer, the card is KNOWN — do not leave it in unplaced out of caution.
  A single real prior source is enough to place it.
  Surface-match trap (still applies): a hit that merely shares the
  buzzword is NOT a match. The hit must perform the SAME structural
  transfer the bridge claims. "The search returned a page about Markov
  chains" does not make a Markov-chain-trust-decay bridge KNOWN unless the
  page actually models that decay. Co-occurrence of a word is not prior art.

INVALID — the evidence CONTRADICTS a checkable claim in the card (a wrong
  date, a false attribution, a "first to do X" the record refutes, a named
  result that does not exist or works differently). State the specific
  contradiction in `contradicts` and point at the contradicting evidence.
  "Feels wrong" is rejected. A real concept stapled to a contradicted fact
  is INVALID, not KNOWN.

UNPLACED — you ran the queries (they are in the evidence) and the hits
  contain NO source performing the bridge's transfer AND nothing that
  contradicts the card. This is "searched the corners, found nothing." It
  is a genuine non-placement, NOT a verdict of novelty or value.
  Honesty about thin evidence: if the hits for a card are empty or
  off-topic, that is a WEAK signal — it may mean the bridge is unprecedented,
  or it may mean the search missed. Either way the card is UNPLACED with a
  note that the evidence was thin. NEVER fabricate a KNOWN match to fill a
  gap, and never invent a reference the evidence does not contain.
  State `why_unplaced` as a single neutral technical clause (e.g.
  "queries returned no source performing the central transfer";
  "evidence thin — hits off-topic"). No speculation about novelty or worth.

ABSOLUTE RULES:
  - Every `reference` in KNOWN must be copyable from the evidence. A
    citation not present in any hit is a fabrication — the worst error in
    this role.
  - When the evidence genuinely doesn't decide it, the card goes to UNPLACED.
  - One card, one bin. No hedged dual placements.
  - Do NOT rewrite, paraphrase, complete, or merge cards. Content passes
    through verbatim.
  - Every input card appears in exactly ONE bin. Recount before emitting.
  - No summary, no closing assessment. Output ends with the JSON brace.

You are a sieve with a search engine. Use the search. Trust the search
over your gut. The gold lives in unplaced; the dirt lives in invalid; the
already-done lives in known — and the EVIDENCE is what tells them apart.

OUTPUT FORMAT: a single JSON object with three arrays — `known`,
`invalid`, `unplaced` — per the schema in the user message. Output ONLY
the JSON.
"""


def _format_evidence_block(ev) -> dict | None:
    """Compact the per-card evidence into a JSON-serializable block for the
    sort payload. Returns None when there's nothing to attach."""
    if ev is None:
        return None
    return {
        "queries_run": list(ev.queries),
        "searched":    ev.searched,
        "hits": [
            {"title": h.title, "url": h.url, "snippet": h.snippet}
            for h in ev.hits
        ],
        "note": ev.note,
    }


def _build_sort_payload(
    cushion: CushionGraph | None,
    cards:   list[ArticulatedCard],
    synthesis_map: SynthesisMap | None,
    web_evidence: EvidenceLedger | None = None,
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
        block = {
            "report_id":    c.report_id,
            "agent_id":     c.agent_id or "",
            "spark":        c.spark,
            "source_shape": c.source_shape,
            "bridge":       c.bridge,
            "use":          c.use,
            "limit":        c.limit,
        }
        if web_evidence is not None:
            ev_block = _format_evidence_block(web_evidence.evidence_for(c.report_id))
            if ev_block is not None:
                block["evidence"] = ev_block
        card_blocks.append(block)

    # When real web evidence is attached, the `reference` must be copied
    # from a hit; otherwise it's the model's best checkable pointer.
    ref_hint = (
        "<REQUIRED, non-empty: the URL or citation COPIED FROM the card's "
        "evidence hits>"
        if web_evidence is not None
        else "<REQUIRED, non-empty: checkable pointer>"
    )
    schema_spec = {
        "known": [{
            "report_id":       "<copy from input>",
            "prior_work_name": "<REQUIRED, non-empty: named prior work>",
            "reference":       ref_hint,
            "confidence":      "<float 0..1>",
            "reasoning":       "<one sentence: how the card maps to the named prior work>",
        }],
        "invalid": [{
            "report_id":   "<copy from input>",
            "contradicts": "<REQUIRED, non-empty: the specific fact or self-inconsistency violated>",
            "reasoning":   "<one sentence: how the card violates it>",
            "confidence":  "<float 0..1>",
        }],
        "unplaced": [{
            "report_id":    "<copy from input>",
            "why_unplaced": (
                "<REQUIRED, single neutral technical clause naming the "
                "match-impossibility; no speculation about novelty, value, "
                "plausibility, or potential>"
            ),
            "confidence":   "<float 0..1>",
        }],
    }

    if web_evidence is not None:
        instruction = (
            "Each card carries an `evidence` block: the search queries that "
            "were run and the live web hits they returned. Bin EVERY card "
            "against that evidence. KNOWN requires a prior_work_name AND a "
            "reference COPIED from a hit; if no hit performs the bridge's "
            "transfer, the card is UNPLACED — never invent a citation. "
            "Output the JSON object only. Recount before emitting: total "
            "binned items MUST equal the card_count above."
        )
    else:
        instruction = (
            "Classify EVERY card into exactly ONE bin. Output the JSON "
            "object only. No prose around it. Remember: known requires a "
            "named prior_work_name AND a checkable reference, or the card "
            "belongs in unplaced. Recount before emitting: total binned "
            "items MUST equal the card_count above."
        )

    payload = {
        "problem_context": problem,
        "card_count":      len(cards),
        "cards":           card_blocks,
        "output_schema":   schema_spec,
        "instruction":     instruction,
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
    web_evidence:     EvidenceLedger | None = None,
) -> SortedReport:
    """Sort a dossier's articulated cards into three bins.

    Single-pass, single-seat. No fusion, no synthesis. Each card lands in
    exactly one of known / invalid / unplaced.

    When `web_evidence` is provided, the sorter runs its VERIFIED doctrine:
    it bins each card against the real web hits in the ledger instead of
    its training memory, and every KNOWN citation must be copied from a
    hit. This is the fix for the blind-not-lazy failure mode (a card whose
    prior work was published after the model's cutoff). Without evidence,
    the original memory-only doctrine runs (backward-compatible).

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

    verified = web_evidence is not None
    progress.emit("starting", {
        "card_count":       len(cards),
        "cost_ceiling_usd": cost_ceiling_usd,
        "fable_model":      fable_model,
        "web_verified":     verified,
        "evidence_hits":    web_evidence.total_hits if verified else 0,
    })

    doctrine = _DOCTRINE_VERIFIED if verified else _DOCTRINE_PREAMBLE
    system = compose_system_prompt(doctrine, mode="master_sorter")
    payload = _build_sort_payload(cushion, cards, synthesis_map, web_evidence)

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
