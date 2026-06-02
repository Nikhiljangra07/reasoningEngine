"""
Master Synthesizer — the "senior scientist" layer above the agent reports.

PURPOSE
-------
The Wandering Room's agent layer surfaces N raw ExplorationReports
(typically 15-30 per session, each one a structural bridge an agent
found while wandering). The existing `articulate.py` + `synthesis.py`
turn those reports into ArticulatedCards plus a SynthesisMap
(clusters, contradictions, opportunity paths). That's the dossier.

The MASTER SYNTHESIZER sits one layer above the dossier. Its job is
to read everything together and produce 3-5 MASTER FUSION REPORTS —
cross-card fusions that bridge what no single card could bridge alone.

Two frontier models work this layer as COLLEAGUES, not contestants:

  - Anthropic Opus 4.6   ("anthropic_seat")
  - OpenAI    GPT-5.4    ("openai_seat")

They draft independently (R1), critique each other's drafts (R2),
finalize their joint conclusions (R3), and — only when they genuinely
disagree on a specific fusion — produce two angled versions of THAT
fusion side-by-side so the user picks the angle that fits (R4). When
they agree, the fusion ships as ONE merged report with citations drawn
from both. They do NOT average each other's outputs — averaging
recreates the very smartness-layer collapse the Wandering Room exists
to escape.

DOCTRINE
--------
Every claim in a master fusion MUST trace to >= 2 ArticulatedCards
(`citation discipline B`). A "fusion" of one card is not a fusion;
it's an amplification. If a draft can't ground its claim in two cards,
the synthesizer either drops it or downgrades to a "single-card
amplification" tag — never invents a citation.

Hallucination concentrates here. Constraints that suppress it:

  - Strict citation grounding (every fusion carries its CardReference list)
  - Honest LIMIT field (Law 7 — where the fusion breaks; non-empty required)
  - Confidence labels are NOT inflated; honest "low" is allowed
  - LOW-band cards are NOT filtered out at input (Law 3 — Heisenberg zone
    breakthroughs come from low-confidence material the synthesizer fuses)
  - Disputed fusions preserve BOTH angles — no smoothed middle

UX COMMUNICATIVENESS
--------------------
This layer is slow by design (5-8 minutes wall-clock, 4 rounds of
asyncio.gather'd parallel LLM calls). The user is told what the
synthesizer is doing in real-time via `MasterSynthesisProgress`
callbacks — "drafting candidate fusions", "models critiquing each
other's drafts", "drilling into disputed angles" — so the wait feels
like watching a brain work, not a black box hang. Meanwhile the user
can browse the existing dossier cards (already returned by build_dossier);
the master fusion is an ADDITIVE final artifact, not a blocking gate.

HARD COST CAP
-------------
A session-level dollar ceiling is enforced after every LLM call. When
the running total would exceed `cost_ceiling_usd`, subsequent calls
raise `MasterSynthesisBudgetExceeded`. The orchestration catches this
and returns whatever partial result was assembled so far — partial >
nothing. The default ceiling is $8.00 (the test-phase budget Nikhil
set; raise for production once the prompts settle).

ISOLATION
---------
Imports: dossier types + LLMClient + provider_map for pricing + stdlib.
Does NOT import call_tracker (the AgentScopedLLMClient pattern is for
the agent layer; this module talks to LLMClient directly with explicit
model= per call). Does NOT import runtime (this layer runs AFTER
run_wandering_session has fully returned).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Awaitable, Callable

from src.identity import compose_system_prompt
from src.llm.client import LLMClient, LLMResponse
from src.llm.provider_map import get_pricing
from src.wandering.articulate import ArticulatedCard
from src.wandering.cushion import CushionGraph
from src.wandering.report import Confidence
from src.wandering.synthesis import SynthesisMap


log = logging.getLogger("constellax.wandering.master_synthesizer")


# ---------------------------------------------------------------------------
# Model slug pinning — explicit, not via resolve_model()
# ---------------------------------------------------------------------------
OPUS_SEAT_MODEL = "anthropic/claude-opus-4-6"
GPT_SEAT_MODEL  = "openai/gpt-5.4"

#: Domain/concept used when calling client.call. They're internal —
#: provider_map.resolve_model is bypassed via the explicit `model=` kwarg.
#: The (domain, concept) tuple still flows into the observability log so
#: per-round attribution is visible.
SEAT_DOMAIN          = "master_synthesizer"
SEAT_CONCEPT_DRAFT   = "master_draft"
SEAT_CONCEPT_CRITIQUE = "master_critique"
SEAT_CONCEPT_FINAL   = "master_final"
SEAT_CONCEPT_ANGLE   = "master_disputed_angle"

#: Per-round output-token caps. Drafted from the first dry-run on
#: run #2's 23 reports (2026-06-02) where R3 calls truncated at the
#: LLMClient default 4096 cap — both Opus and GPT-5.4 wanted to emit
#: more than 4096 tokens of structured JSON for 5+ final fusions.
#:
#: R1 (draft)    — moderate; each seat produces 3-5 fusion objects
#: R2 (critique) — light; each seat emits one annotation per other-draft
#: R3 (final)    — HEAVIEST; merges drafts + critique into final fusions,
#:                  needs headroom for full reasoning + limit + citations
#: R4 (angles)   — heavy; per-disputed-fusion full angled writeup
MAX_TOKENS_DRAFT     = 4096
MAX_TOKENS_CRITIQUE  = 2048
MAX_TOKENS_FINAL     = 8192
# Bumped 6144 → 8192 (audit Fix 5, r4→r5). R4 GPT angles in run #4
# returned 0 citations on both disputed fusions; the prompt was tightened
# to require citations (see _ANGLE_INSTRUCTIONS) and the extra headroom
# absorbs the re-emitted citation blocks without truncation.
MAX_TOKENS_ANGLE     = 8192


# ---------------------------------------------------------------------------
# Hard cost cap
# ---------------------------------------------------------------------------

DEFAULT_COST_CEILING_USD = 8.00


class MasterSynthesisBudgetExceeded(Exception):
    """Raised when the cumulative spend would exceed `cost_ceiling_usd`.

    The orchestrator catches this and returns the partial result so far
    instead of crashing — partial output > none, especially in the
    testing phase where the cap is intentionally tight.
    """


def _call_cost_usd(model_slug: str, response: LLMResponse) -> float:
    """Compute USD spend for one LLMResponse using provider_map pricing.

    Pricing is (input $/M, output $/M). Falls back to FALLBACK_PRICING
    (Sonnet-tier) when the slug isn't registered — conservative
    (overestimates) so the cap fires safely on unregistered models.
    """
    in_price, out_price = get_pricing(model_slug)
    in_cost  = (response.input_tokens  or 0) / 1_000_000 * in_price
    out_cost = (response.output_tokens or 0) / 1_000_000 * out_price
    return in_cost + out_cost


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class Seat(str, Enum):
    """Which model produced a draft / critique / angle."""
    OPUS  = "opus"
    GPT   = "gpt"


class AgreementStatus(str, Enum):
    """How the two models converged on a given fusion.

      BOTH_AGREE          — both seats independently produced the same fusion
      MOSTLY_AGREE_REFINED — one seat proposed, the other refined; final merged
      DISPUTED            — both seats produced incompatible takes; angles preserved
      SOLO_OPUS           — only Opus surfaced this fusion; GPT didn't push back
      SOLO_GPT            — only GPT surfaced this fusion; Opus didn't push back
    """
    BOTH_AGREE          = "both_agree"
    MOSTLY_AGREE_REFINED = "mostly_agree_refined"
    DISPUTED            = "disputed"
    SOLO_OPUS           = "solo_opus"
    SOLO_GPT            = "solo_gpt"


class CritiqueAnnotation(str, Enum):
    """R2 critique values one seat applies to the other's drafts."""
    AGREE     = "agree"
    REFINE    = "refine"
    DISAGREE  = "disagree"


@dataclass
class CardReference:
    """A pointer from a master fusion back to one source ArticulatedCard.

    `which_field` is the card section the fusion is drawing from —
    "spark", "bridge", "limit", "source_shape", or "use" — so the user
    can read the source card and verify the fusion's grounding.
    """
    report_id:    str
    agent_id:     str
    which_field:  str   # spark | source_shape | bridge | use | limit
    excerpt:      str   # short quote from the card; max ~120 chars

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DisputedAngle:
    """One seat's take on a DISPUTED fusion. Two of these compose a
    disputed master fusion — the user picks the angle that fits."""
    seat:        Seat
    claim:       str   # the synthesized insight (1-2 sentences)
    reasoning:   str   # how the fusion holds (2-4 sentences)
    limit:       str   # where this angle's framing breaks (Law 7; non-empty)
    citations:   list[CardReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "seat":       self.seat.value,
            "claim":      self.claim,
            "reasoning":  self.reasoning,
            "limit":      self.limit,
            "citations":  [c.to_dict() for c in self.citations],
        }


@dataclass
class MasterFusionReport:
    """One cross-card fusion produced by the master synthesizer layer.

    The fusion is the "senior scientist" output — a bridge between
    findings the user could not see by reading any single agent report.
    """
    title:            str                       # one-line description
    claim:            str                       # the insight; empty when DISPUTED
    reasoning:        str                       # how the fusion holds; empty when DISPUTED
    limit:            str                       # where the fusion breaks (Law 7); empty when DISPUTED
    citations:        list[CardReference]       # cards this fusion cites; >= 2 for valid fusion
    confidence:       Confidence
    agreement_status: AgreementStatus
    #: Populated ONLY when agreement_status == DISPUTED. Empty otherwise.
    disputed_angles:  list[DisputedAngle] = field(default_factory=list)
    #: When the fusion is the result of a cohort-pair merge, these
    #: snapshots preserve EACH seat's pre-merge claim/reasoning/limit/
    #: confidence so the user can audit which framing actually shipped
    #: vs which was suppressed by `_merge_cohort_pair`. None when the
    #: fusion is unpaired (no merge happened) or a SOLO_* outcome.
    pre_merge_opus:   dict | None             = None
    pre_merge_gpt:    dict | None             = None
    #: Which seat's prose actually shipped as the visible claim/
    #: reasoning/limit on this fusion ("opus" or "gpt"). Set for every
    #: fusion EXCEPT SOLO_OPUS/SOLO_GPT (where the seat is already
    #: encoded in agreement_status, so populating this would be
    #: redundant). For paired fusions, picked by `_keeper_score` —
    #: multi-factor on cards-cited / providers / confidence /
    #: prose-length. For unpaired BOTH_AGREE / MOSTLY_AGREE_REFINED /
    #: DISPUTED fusions, populated by `_dedupe_across_seats` as the
    #: producing seat (audit Fix 4: r4 unpaired MAR shipped with
    #: keeper_seat='' and pre_merge_* = None).
    keeper_seat:      str                     = ""
    #: Unique agent_ids cited (P01, P02, ...). Set at parse time and
    #: preserved through merges. Distinct from `citation_provider_count`
    #: because the same provider can occupy multiple slots (e.g. run #3
    #: had two DeepSeek slots P01 + P02).
    citation_agent_count:    int = 0
    #: Unique providers cited (deepseek, anthropic, openai, google, xai).
    #: Resolved via the agent→provider map passed into master_synthesize.
    #: Stays 0 when no map was supplied — `master_synthesize` always
    #: passes one when run via `build_dossier`, so this should be > 0
    #: in real runs.
    citation_provider_count: int = 0

    def to_dict(self) -> dict:
        return {
            "title":            self.title,
            "claim":            self.claim,
            "reasoning":        self.reasoning,
            "limit":            self.limit,
            "citations":        [c.to_dict() for c in self.citations],
            "confidence":       self.confidence.value,
            "agreement_status": self.agreement_status.value,
            "disputed_angles":  [a.to_dict() for a in self.disputed_angles],
            "pre_merge_opus":   self.pre_merge_opus,
            "pre_merge_gpt":    self.pre_merge_gpt,
            "keeper_seat":      self.keeper_seat,
            "citation_agent_count":    self.citation_agent_count,
            "citation_provider_count": self.citation_provider_count,
        }


@dataclass
class MasterSynthesis:
    """Container the master synthesizer returns. Holds the fusion list
    plus cost/round telemetry the caller can surface in the UX."""
    master_fusions:        list[MasterFusionReport]   = field(default_factory=list)
    total_cost_usd:        float                      = 0.0
    cost_ceiling_usd:      float                      = DEFAULT_COST_CEILING_USD
    rounds_completed:      list[str]                  = field(default_factory=list)
    truncated_by_budget:   bool                       = False
    truncation_reason:     str                        = ""
    #: Per-status counts (post-merge, post-R4). The pre-existing
    #: `dispute_count` / `agreement_count` / `solo_count` aggregates
    #: are kept for backwards-compatibility but are now derived from
    #: these five fields — never set them directly.
    both_agree_count:           int                   = 0
    mostly_agree_refined_count: int                   = 0
    solo_opus_count:            int                   = 0
    solo_gpt_count:             int                   = 0
    disputed_count:             int                   = 0
    #: Legacy aggregate fields. Kept so older readers (the dry-run
    #: scripts under /tmp) don't break. Derived from the per-status
    #: counts above; do not write to them directly.
    dispute_count:         int                        = 0
    agreement_count:       int                        = 0
    solo_count:            int                        = 0
    #: Per-call audit — each entry: {seat, round, model, in_tok, out_tok, cost_usd, ms, ok}
    call_log:              list[dict]                 = field(default_factory=list)
    #: R2 critique TEXT (each seat's full annotation list). Each entry is
    #: a {draft_index, annotation, reason} object as returned by the seat.
    #: Persisted so a downstream auditor can answer "was the critique
    #: rubber-stamped or substantive?" from artifacts alone — addresses
    #: Blocker #1 of the run-#3 audit.
    r2_critique_opus:      list                       = field(default_factory=list)
    r2_critique_gpt:       list                       = field(default_factory=list)
    #: R3 pre-merge per-seat fusion lists (Opus and GPT each emitted
    #: this many fusions BEFORE _dedupe_across_seats collapsed cohort
    #: pairs). Same Blocker #1: the merged result alone hides the
    #: per-seat counts and pre-merge framing.
    r3_pre_merge_opus:     list[dict]                 = field(default_factory=list)
    r3_pre_merge_gpt:      list[dict]                 = field(default_factory=list)
    #: Within-seat dedupe metric — count of same-seat near-duplicates
    #: collapsed (Opus↔Opus + GPT↔GPT) before bipartite. Addresses
    #: Blocker #4 of the run-#3 audit.
    same_seat_pairs_collapsed: int                    = 0
    #: Hunch-coverage surveillance (Fix 6, audit r4→r5). For each hunch
    #: extracted from cushion.raw_input.current_map.content, names the
    #: master-fusion titles whose text (title+claim+reasoning+limit)
    #: lexically references it. Empty list under a hunch label = the
    #: hunch was NOT exercised by any fusion this run.
    #:
    #: Pure observability — does NOT gate output, trigger regeneration,
    #: or influence the keeper-pick. Surveillance only. Verified on disk
    #: in run #4: Butterfly hunch was absent from all 8 master fusions
    #: and nothing flagged it. This field surfaces that gap.
    hunch_coverage:            dict[str, list[str]]   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "master_fusions":      [f.to_dict() for f in self.master_fusions],
            "total_cost_usd":      round(self.total_cost_usd, 4),
            "cost_ceiling_usd":    self.cost_ceiling_usd,
            "rounds_completed":    list(self.rounds_completed),
            "truncated_by_budget": self.truncated_by_budget,
            "truncation_reason":   self.truncation_reason,
            # per-status (canonical)
            "both_agree_count":           self.both_agree_count,
            "mostly_agree_refined_count": self.mostly_agree_refined_count,
            "solo_opus_count":            self.solo_opus_count,
            "solo_gpt_count":             self.solo_gpt_count,
            "disputed_count":             self.disputed_count,
            # legacy aggregates (deprecated; derived from per-status above)
            "dispute_count":       self.dispute_count,
            "agreement_count":     self.agreement_count,
            "solo_count":          self.solo_count,
            "call_log":            list(self.call_log),
            # blocker #1: preserve R2 + R3 raw artifacts
            "r2_critique_opus":    list(self.r2_critique_opus),
            "r2_critique_gpt":     list(self.r2_critique_gpt),
            "r3_pre_merge_opus":   list(self.r3_pre_merge_opus),
            "r3_pre_merge_gpt":    list(self.r3_pre_merge_gpt),
            # blocker #4 metric
            "same_seat_pairs_collapsed": self.same_seat_pairs_collapsed,
            # Fix 6: hunch-coverage surveillance
            "hunch_coverage": {k: list(v) for k, v in self.hunch_coverage.items()},
        }


# ---------------------------------------------------------------------------
# Progress emission — keeps the UX communicative during the slow rounds
# ---------------------------------------------------------------------------


@dataclass
class MasterSynthesisProgress:
    """Live progress reference for an in-flight master synthesis.

    Pass an instance to `master_synthesize(progress=...)`. The orchestrator
    calls these methods at round boundaries. Default implementation is a
    no-op — callers that want UX feedback subclass and override, or just
    set the `on_event` callback to a `print` / logger / SSE stream.

    Per Nikhil's framing: the synthesizer must SHOW it's actively working,
    not return a black-box result 5 minutes later. The user sees:
      "drafting candidate fusions (Opus + GPT)..."
      "models critiquing each other's drafts (3 from Opus, 4 from GPT)..."
      "finalizing — 2 agreed, 1 disputed..."
      "drilling into 1 disputed angle..."
      "complete — 4 master fusions ready."
    These map to events; the caller decides how to render them.
    """
    on_event: Callable[[str, dict], None] | None = None
    events:   list[dict] = field(default_factory=list)

    def emit(self, name: str, payload: dict | None = None) -> None:
        payload = payload or {}
        entry = {"name": name, "ts": time.time(), **payload}
        self.events.append(entry)
        log.info("[master_synth] %s %s", name, payload)
        if self.on_event is not None:
            try:
                self.on_event(name, payload)
            except Exception as e:  # pragma: no cover — UX hook must not crash the loop
                log.warning("progress on_event raised (ignored): %s", e)


# ---------------------------------------------------------------------------
# LLM-call helper with cost-cap enforcement
# ---------------------------------------------------------------------------


async def _call_with_budget(
    *,
    client:        LLMClient,
    system_prompt: str,
    user_message:  str,
    domain:        str,
    concept:       str,
    seat:          Seat,
    model_slug:    str,
    round_name:    str,
    result:        MasterSynthesis,
    max_tokens:    int | None = None,
) -> LLMResponse:
    """Make one LLM call, record cost, raise if cap would be exceeded.

    Cost is checked AFTER the call (we can't predict spend exactly
    pre-call), so it's possible to slightly overshoot on the call that
    breaches the cap. The truncation logic uses partial results — the
    over-cap call's output is still preserved.

    `max_tokens` overrides LLMClient.MAX_OUTPUT_TOKENS so R3/R4 calls
    can emit longer JSON. Without this override, R3 outputs truncate at
    the 4096-default cap and the parser sees malformed JSON.
    """
    response: LLMResponse = await client.call(
        system_prompt=system_prompt,
        user_message=user_message,
        domain=domain,
        concept=concept,
        model=model_slug,
        max_tokens=max_tokens,
    )
    cost_usd = _call_cost_usd(model_slug, response)
    result.total_cost_usd += cost_usd
    result.call_log.append({
        "seat":     seat.value,
        "round":    round_name,
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
            f"ceiling ${result.cost_ceiling_usd:.2f} after seat={seat.value} "
            f"round={round_name}"
        )
        log.warning("master_synth budget exceeded: %s", result.truncation_reason)
        raise MasterSynthesisBudgetExceeded(result.truncation_reason)

    return response


# ---------------------------------------------------------------------------
# JSON extraction (defensive — frontier models sometimes wrap in prose)
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def _extract_json(text: str) -> str:
    """Strict-extract one balanced JSON object/array.

    Commits to whichever opener appears FIRST (after fence-strip). If
    the text starts with '[' (or '[' appears before '{'), we walk an
    array; if '{' comes first, we walk an object. Mixing modes is a
    bug — when the model emits an array but the array is truncated,
    we must let _parse_json_safely fall through to the truncation-
    recovery path rather than silently extracting the first inner
    object as if it were the whole response.
    """
    text = _strip_code_fences(text)
    bracket_pos = text.find("[")
    brace_pos   = text.find("{")
    # Choose the opener that appears first (and exists)
    if bracket_pos == -1 and brace_pos == -1:
        raise ValueError("no JSON object/array found in response")
    if bracket_pos != -1 and (brace_pos == -1 or bracket_pos < brace_pos):
        opener, closer, start = "[", "]", bracket_pos
    else:
        opener, closer, start = "{", "}", brace_pos
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    # fell off end — unterminated
    raise ValueError(f"unterminated JSON {opener}...{closer} in response")


def _recover_objects_from_truncated_array(text: str) -> list:
    """Recover complete top-level JSON objects from an array body that
    may be truncated (e.g. R3 hit max_tokens mid-fusion-object).

    Strategy: find the first '[' (or skip preamble), then walk forward
    extracting balanced `{...}` blocks one at a time. Stop on the first
    block that fails to parse (the truncated one). Returns the list of
    parseable objects.

    This is the fallback used when `_extract_json` fails because the
    array's closing `]` is missing. Without it, a single truncated
    fusion at the tail discards every fusion that came before it.
    """
    text = _strip_code_fences(text)
    start = text.find("[")
    if start < 0:
        # No array — maybe a bare object (truncated). Try to recover one.
        obj_start = text.find("{")
        if obj_start < 0:
            return []
        recovered = _extract_first_balanced_block(text, obj_start)
        if recovered:
            try:
                return [json.loads(recovered)]
            except (ValueError, json.JSONDecodeError):
                return []
        return []

    out: list = []
    cursor = start + 1
    while cursor < len(text):
        # Skip whitespace, commas
        while cursor < len(text) and text[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(text):
            break
        if text[cursor] == "]":  # clean close
            break
        if text[cursor] != "{":
            # Stray character before next object — bail; cannot recover further
            break
        block = _extract_first_balanced_block(text, cursor)
        if block is None:
            break  # truncated — stop here
        try:
            parsed = json.loads(block)
        except (ValueError, json.JSONDecodeError):
            break  # malformed inside complete braces — stop
        out.append(parsed)
        cursor += len(block)
    return out


def _extract_first_balanced_block(text: str, start: int) -> str | None:
    """Return the substring `text[start:end+1]` that contains one balanced
    `{...}` block, or None if the block is unterminated (truncation).
    Assumes `text[start] == '{'`."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None  # unterminated


def _parse_json_safely(raw: str, default):
    """Defensive JSON parser used for every master-synth round.

    Order of attempts:
      1. Strict extraction via `_extract_json` (full balanced object/array).
      2. Fallback to `_recover_objects_from_truncated_array` — pulls out
         every complete top-level `{...}` block from the array body even
         when the closing `]` is missing (R3/R4 output truncation case).

    Returns `default` only when BOTH attempts fail. The fallback path is
    load-bearing: without it, a single truncated fusion at the array tail
    discards every complete fusion that preceded it.
    """
    try:
        return json.loads(_extract_json(raw))
    except (ValueError, json.JSONDecodeError) as e:
        # Strict parse failed; try to salvage complete objects from the
        # array body if the call was probably truncated.
        recovered = _recover_objects_from_truncated_array(raw)
        if recovered:
            log.warning(
                "master_synth JSON strict-parse failed (%s); recovered %d objects from truncated array",
                e, len(recovered),
            )
            return recovered
        log.warning("master_synth JSON parse failed: %s (raw[:200]=%r)", e, raw[:200])
        return default


# ---------------------------------------------------------------------------
# Prompt fragments — the doctrine is in the prompts; do not soften
# ---------------------------------------------------------------------------


_DOCTRINE_PREAMBLE = """\
You are one seat of a two-seat master synthesizer for Constellax's
Wandering Room. The other seat is being run in parallel by a different
frontier model from a different provider. You are colleagues working
on the same problem, not contestants competing for the best output.

Your job: produce MASTER FUSION REPORTS that bridge what no single
agent's ArticulatedCard could bridge alone. A fusion fuses >= 2 cards.
A single-card "fusion" is not a fusion — it is an amplification of
one card and does not earn the master tier.

CRITICAL CONSTRAINTS — break any of these and the output is rejected:

  1. THE INSIGHT HAPPENS IN THE USER'S HEAD. You do NOT deliver a
     conclusion or a recommendation. You surface a bridge — a fusion
     across cards that the user could not see by reading them
     individually. Frame claims as "these cards converge on…", not
     "you should…".

  2. EVERY CLAIM TRACES TO >= 2 CARDS. Cite specific report_ids and
     which field of the card you are drawing from (spark / source_shape
     / bridge / use / limit). If you cannot ground a claim in two
     cards, drop the fusion. Do not invent citations.

  3. THE LIMIT FIELD IS MANDATORY. Every fusion must articulate where
     it breaks — what context invalidates it, where the analogy stops
     mapping. A fusion without a limit is dishonest. Empty limit → reject.

  4. HONEST DOUBT > PERFORMATIVE CONFIDENCE. If a fusion is fragile,
     label it "low" confidence. Inflated confidence is a worse failure
     than weak signal.

  5. LOW-CONFIDENCE CARDS ARE NOT FILTERED. They are the Heisenberg
     zone — breakthroughs often emerge from low-confidence material
     when fused with higher-confidence cards. Read them. Use them.

  6. NO AVERAGE. When you and the other seat disagree on a fusion,
     do NOT smooth toward a middle. Disagreement is signal. The
     orchestrator will preserve both angles for the user.
"""


_DRAFT_INSTRUCTIONS = """\
ROUND 1 — DRAFT.

Read the cushion (user's pursuit / vision / unfinished threads) and
ALL the articulated cards + the existing synthesis-map context.
Identify 3-5 candidate master fusions: cross-card connections that
bridge multiple cards into one structural insight.

For each candidate fusion, return:
  - title          : one-line description (<= 80 chars)
  - claim          : the synthesized insight (1-2 sentences)
  - reasoning      : how the fusion holds together (2-4 sentences)
  - limit          : where the fusion breaks (1-2 sentences; MANDATORY)
  - citations      : >= 2 entries, each with report_id, agent_id,
                     which_field (spark|source_shape|bridge|use|limit),
                     and a short excerpt (<= 120 chars)
  - confidence     : "low" | "medium" | "high"

OUTPUT FORMAT — return a JSON ARRAY of fusion objects. No prose
preamble. No code fences. Just the JSON array:

[
  {
    "title": "...",
    "claim": "...",
    "reasoning": "...",
    "limit": "...",
    "citations": [
      {"report_id": "...", "agent_id": "...", "which_field": "bridge", "excerpt": "..."},
      {"report_id": "...", "agent_id": "...", "which_field": "limit",  "excerpt": "..."}
    ],
    "confidence": "medium"
  },
  ...
]
"""


_CRITIQUE_INSTRUCTIONS = """\
ROUND 2 — CRITIQUE THE OTHER SEAT'S DRAFTS.

Below are draft fusions from the other seat. Your job: find real
disagreements. This is collegial review AND substantive challenge —
you and the other seat trained on different corpora, so genuine
divergences exist. Surface them. Do NOT default to agreement.

For each draft, decide AGREE / REFINE / DISAGREE. Calibration target:

  - "agree"    : you would have drafted this fusion yourself with the
                 same citations and the same limit. Use SPARINGLY —
                 a 5-of-5 "agree" pass is a smell that you are not
                 reading critically. If everything looks fine, find
                 the weakest link and at minimum "refine" it.

  - "refine"   : the insight is broadly right but the framing,
                 citation set, or limit needs work. Say WHAT to
                 refine specifically (which citation is weak, what
                 the limit misses, where the claim overreaches).

  - "disagree" : you read the same cards but reach a different
                 conclusion. Sketch the FUSION YOU WOULD WRITE
                 instead — title + claim seed + which citations
                 you'd swap or add. Use this when the other seat's
                 reading is plausible but not the strongest reading
                 of the evidence.

Honest disagreements are what make the cross-seat architecture
work. If you genuinely have no critique on a draft, say so
explicitly in the `reason` field (a short note like "I would
have written this; nothing to refine" is acceptable). But do
not blanket-agree — that erases the very signal the architecture
is built to capture.

For each draft you read, return one critique object:
  - draft_index    : the index of the draft in the input list
  - annotation     : "agree" | "refine" | "disagree"
  - reason         : 2-3 sentences explaining your judgment
                     (if "refine", say WHAT to refine specifically)
                     (if "disagree", say what the OTHER fusion would
                      look like from your read — title + claim sketch)

OUTPUT FORMAT — JSON ARRAY of critique objects. Just the array:

[
  {"draft_index": 0, "annotation": "agree", "reason": "..."},
  {"draft_index": 1, "annotation": "refine", "reason": "the framing is..."},
  {"draft_index": 2, "annotation": "disagree", "reason": "I read these cards as..."},
  ...
]
"""


_FINAL_INSTRUCTIONS = """\
ROUND 3 — FINAL FUSIONS.

You now have:
  - Your own R1 drafts (below as YOUR DRAFTS)
  - The other seat's R2 critique of YOUR drafts (below as CRITIQUE)
  - The other seat's R1 drafts (below as OTHER SEAT DRAFTS) and your
    OWN critique of them you produced moments ago (CITED BELOW)

Produce the FINAL set of master fusions — your honest synthesis
incorporating the other seat's input where appropriate. For each
final fusion, ALSO set the `agreement_status` field. Calibrate
strictly — over-claiming "both_agree" erases the cross-seat signal
that the architecture exists to capture.

  - "both_agree"          : STRICT. Use only when (a) the other seat
                            drafted the SAME structural insight,
                            (b) your citation sets substantially
                            overlap (3+ shared report_ids OR the
                            titles share a clear majority of content
                            words), AND (c) you both name the same
                            failure mode in `limit`. If ANY of those
                            three is borderline, downgrade to
                            "mostly_agree_refined".

  - "mostly_agree_refined": DEFAULT for ANY refinement. Use this
                            when the other seat refined your draft
                            and you accepted some of the refinement,
                            OR you refined theirs and they accepted,
                            OR you both reached similar claims but
                            via different citation paths. This label
                            is honest — over-using "both_agree" when
                            the truth is "mostly agree, with edits"
                            is the failure mode to avoid.

  - "disputed"            : you and the other seat read the same cards
                            but reach incompatible conclusions. Mark
                            DISPUTED and SET claim/reasoning/limit to
                            EMPTY STRINGS — round 4 will produce the
                            two angled versions separately. Use this
                            when the architecture's signature feature
                            (preserve genuine disagreement) is the
                            right call. If your R2 critique landed
                            on "disagree" for a draft and you still
                            disagree, "disputed" is the correct label.

  - "solo_opus" or
    "solo_gpt"            : only ONE of you surfaced this fusion and
                            the other did not push back. (Use the seat
                            value that matches the seat producing the
                            output — your job is to label which seat
                            you are.)

Same citation discipline as R1: >= 2 cards per fusion, every claim
grounded, limit mandatory unless agreement_status == "disputed".

OUTPUT FORMAT — JSON ARRAY of final fusion objects:

[
  {
    "title": "...",
    "claim": "...",                    // empty string if disputed
    "reasoning": "...",                // empty string if disputed
    "limit": "...",                    // empty string if disputed
    "citations": [...],
    "confidence": "low|medium|high",
    "agreement_status": "both_agree|mostly_agree_refined|disputed|solo_opus|solo_gpt"
  },
  ...
]
"""


_ANGLE_INSTRUCTIONS = """\
ROUND 4 — DISPUTED ANGLES.

One or more fusions came out DISPUTED in round 3 — you and the other
seat read the same cards but reached incompatible conclusions. For
EACH disputed fusion below, produce YOUR OWN angled version: how
YOU see this fusion. The other seat is doing the same in parallel.
Both angles will be shown to the user side-by-side; the user picks
the angle that fits their problem.

Do NOT try to anticipate or merge with the other seat's angle. Your
job is to give YOUR honest read.

CITATIONS ARE MANDATORY ON EVERY ANGLE. An angle without citations
is just an unfalsifiable assertion — the architecture exists to
keep every claim grounded in cards the user can audit. If you genuinely
cannot justify your angle with >= 2 specific report_ids drawn from the
disputed fusion's citation set (or new ones from the same card pool),
do not emit the angle — it isn't ready. Empty `citations` will be
treated as a defect by the orchestrator and the parent fusion's
citations will be inherited as a best-effort fallback (which leaves
the angle's reasoning ungrounded; do not let this happen).

For each disputed fusion, return one angle object:
  - title          : copy the fusion's title verbatim
  - claim          : YOUR claim (1-2 sentences)
  - reasoning      : YOUR reasoning (2-4 sentences)
  - limit          : YOUR limit (1-2 sentences; MANDATORY)
  - citations      : YOUR citations (>= 2, MANDATORY)

OUTPUT FORMAT — JSON ARRAY of angle objects:

[
  {
    "title": "...",
    "claim": "...",
    "reasoning": "...",
    "limit": "...",
    "citations": [...]
  },
  ...
]
"""


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _format_cushion(cushion: CushionGraph | None) -> str:
    if cushion is None or cushion.raw_input is None:
        return "(no cushion provided)"
    inp = cushion.raw_input
    blocks = [
        "# USER'S PURSUIT",   inp.problem.content,
        "\n# USER'S VISION",  getattr(inp.vision, "content", "") or "(none)",
        "\n# UNFINISHED THREADS / HUNCHES", getattr(inp.current_map, "content", "") or "(none)",
        f"\n# CUSHION CONSTELLATION ({cushion.constellation_size} nodes across 3 layers)",
        "## actual:    " + ", ".join(cushion.actual.nodes),
        "## essence:   " + ", ".join(cushion.essence.nodes),
        "## mechanism: " + ", ".join(cushion.mechanism.nodes),
    ]
    return "\n".join(blocks)


def _format_card(c: ArticulatedCard) -> str:
    return (
        f"## Card {c.report_id} [{c.confidence.value}] agent={c.agent_id} domain={c.domain}\n"
        f"  Spark:        {c.spark}\n"
        f"  Source Shape: {c.source_shape}\n"
        f"  Bridge:       {c.bridge}\n"
        f"  Use:          {c.use}\n"
        f"  Limit:        {c.limit}\n"
        f"  Match:        {c.confidence_detail or '(n/a)'}"
    )


def _format_synthesis_map(smap: SynthesisMap | None) -> str:
    if smap is None:
        return "(no synthesis map)"
    parts = [
        "# EXISTING SYNTHESIS MAP (single-pass Sonnet — your input, not your output)",
        f"  top_insights: {smap.top_insights}",
        f"  clusters: {[{'label': c.label, 'cards': c.card_ids} for c in smap.clusters]}",
        f"  contradictions: {[{'desc': c.description, 'cards': c.card_ids} for c in smap.contradictions]}",
        f"  opportunity_paths: {[{'desc': o.description, 'cards': o.supporting_card_ids} for o in smap.opportunity_paths]}",
        f"  open_questions: {smap.open_questions}",
        f"  recommended_next_direction: {smap.recommended_next_direction}",
    ]
    return "\n".join(parts)


def _build_draft_payload(
    cushion: CushionGraph | None,
    cards:   list[ArticulatedCard],
    smap:    SynthesisMap | None,
) -> str:
    blocks = [
        _format_cushion(cushion),
        _format_synthesis_map(smap),
        f"\n# ALL ARTICULATED CARDS ({len(cards)} total)",
    ]
    for c in cards:
        blocks.append("\n" + _format_card(c))
    blocks.append("\n# YOUR TASK")
    blocks.append(_DRAFT_INSTRUCTIONS)
    return "\n".join(blocks)


def _build_critique_payload(
    cushion:       CushionGraph | None,
    cards:         list[ArticulatedCard],
    other_drafts:  list[dict],
) -> str:
    other_drafts_json = json.dumps(other_drafts, indent=2, ensure_ascii=False)
    blocks = [
        _format_cushion(cushion),
        f"\n# ALL ARTICULATED CARDS ({len(cards)} total — for citation cross-check)",
    ]
    for c in cards:
        blocks.append("\n" + _format_card(c))
    blocks.append("\n# OTHER SEAT'S DRAFTS")
    blocks.append(other_drafts_json)
    blocks.append("\n# YOUR TASK")
    blocks.append(_CRITIQUE_INSTRUCTIONS)
    return "\n".join(blocks)


def _build_final_payload(
    cushion:        CushionGraph | None,
    cards:          list[ArticulatedCard],
    my_drafts:      list[dict],
    critique_of_me: list[dict],
    other_drafts:   list[dict],
    my_critique:    list[dict],
    seat:           Seat,
) -> str:
    blocks = [
        _format_cushion(cushion),
        f"\n# ALL ARTICULATED CARDS ({len(cards)} total)",
    ]
    for c in cards:
        blocks.append("\n" + _format_card(c))
    blocks.append("\n# YOUR DRAFTS (round 1)")
    blocks.append(json.dumps(my_drafts, indent=2, ensure_ascii=False))
    blocks.append("\n# OTHER SEAT'S CRITIQUE OF YOUR DRAFTS")
    blocks.append(json.dumps(critique_of_me, indent=2, ensure_ascii=False))
    blocks.append("\n# OTHER SEAT'S DRAFTS")
    blocks.append(json.dumps(other_drafts, indent=2, ensure_ascii=False))
    blocks.append("\n# YOUR CRITIQUE OF THE OTHER SEAT'S DRAFTS")
    blocks.append(json.dumps(my_critique, indent=2, ensure_ascii=False))
    blocks.append(f"\n# YOUR SEAT: {seat.value}")
    blocks.append("\n# YOUR TASK")
    blocks.append(_FINAL_INSTRUCTIONS)
    return "\n".join(blocks)


def _build_angle_payload(
    cushion:           CushionGraph | None,
    cards:             list[ArticulatedCard],
    disputed_fusions:  list[dict],
    seat:              Seat,
) -> str:
    blocks = [
        _format_cushion(cushion),
        f"\n# ALL ARTICULATED CARDS ({len(cards)} total)",
    ]
    for c in cards:
        blocks.append("\n" + _format_card(c))
    blocks.append("\n# DISPUTED FUSIONS")
    blocks.append(json.dumps(disputed_fusions, indent=2, ensure_ascii=False))
    blocks.append(f"\n# YOUR SEAT: {seat.value}")
    blocks.append("\n# YOUR TASK")
    blocks.append(_ANGLE_INSTRUCTIONS)
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Parsers — convert raw JSON into typed objects with citation validation
# ---------------------------------------------------------------------------


def _parse_citations(raw_list) -> list[CardReference]:
    out: list[CardReference] = []
    if not isinstance(raw_list, list):
        return out
    for c in raw_list:
        if not isinstance(c, dict):
            continue
        rid = str(c.get("report_id", "")).strip()
        aid = str(c.get("agent_id", "")).strip()
        fld = str(c.get("which_field", "")).strip().lower()
        exc = str(c.get("excerpt", "")).strip()
        if not (rid and exc):
            continue
        out.append(CardReference(
            report_id=rid, agent_id=aid, which_field=fld, excerpt=exc[:200],
        ))
    return out


def _coerce_confidence(raw) -> Confidence:
    try:
        return Confidence(str(raw).strip().lower())
    except (ValueError, TypeError):
        return Confidence.LOW


def _coerce_agreement(raw) -> AgreementStatus:
    try:
        return AgreementStatus(str(raw).strip().lower())
    except (ValueError, TypeError):
        return AgreementStatus.SOLO_OPUS   # safe default; orchestrator may rewrite


def _parse_final_fusion(raw) -> MasterFusionReport | None:
    if not isinstance(raw, dict):
        return None
    citations = _parse_citations(raw.get("citations", []))
    title = str(raw.get("title", "")).strip()
    agreement = _coerce_agreement(raw.get("agreement_status", "solo_opus"))
    if not title:
        return None
    return MasterFusionReport(
        title=title,
        claim=str(raw.get("claim", "")).strip(),
        reasoning=str(raw.get("reasoning", "")).strip(),
        limit=str(raw.get("limit", "")).strip(),
        citations=citations,
        confidence=_coerce_confidence(raw.get("confidence", "low")),
        agreement_status=agreement,
    )


def _is_valid_fusion(f: MasterFusionReport) -> bool:
    """Citation discipline B: >= 2 citations per fusion. Limit mandatory
    except when DISPUTED (Round 4 supplies the per-angle limit instead)."""
    if len(f.citations) < 2:
        return False
    if f.agreement_status != AgreementStatus.DISPUTED:
        if not f.limit.strip():
            return False
        if not f.claim.strip():
            return False
    return True


def _compute_citation_counts(
    f: MasterFusionReport,
    agent_provider_map: dict[str, str] | None,
) -> None:
    """Populate `citation_agent_count` and `citation_provider_count`
    on the fusion in place. Run-#3 audit polish: distinguishes fusions
    that genuinely span multiple providers from those resting on a
    single provider's reports even when several agent_ids appear.

    `agent_provider_map` is the caller-supplied {agent_id -> provider}
    map; when None, provider_count stays 0 (agent_count still works).
    """
    seen_agents:    set[str] = set()
    seen_providers: set[str] = set()
    for c in f.citations:
        aid = (c.agent_id or "").strip()
        if not aid: continue
        seen_agents.add(aid)
        if agent_provider_map:
            prov = agent_provider_map.get(aid)
            if prov: seen_providers.add(prov)
    f.citation_agent_count    = len(seen_agents)
    f.citation_provider_count = len(seen_providers)


# ---------------------------------------------------------------------------
# Parallel-seat dedupe — cohort-convergence detection
# ---------------------------------------------------------------------------
#
# Background (audit, 2026-06-02): dry-run #2 produced 13 master fusions
# that contained 5 near-duplicate pairs across seats — both Opus and GPT
# independently surfaced the SAME insight under slightly different titles.
# The exact-title merger only caught byte-identical matches, leaving
# F1/F8, F2/F9, F3/F13, F5/F12, F6/F7 as parallel fusions.
#
# These aren't noise — they're a SIGNAL (cohort convergence is the
# architecture's strongest hallucination-independent evidence). But
# user-facing reports should collapse them into one fusion + the
# convergence label rather than burying the signal in redundancy.
#
# Heuristic: pair fusions across seats by max(title_jaccard,
# citation_jaccard) >= COHORT_CONVERGENCE_THRESHOLD. Greedy bipartite
# matching, highest-similarity pairs first. Catches both lexical near-
# duplicates (different title phrasing, same insight) and citation-set
# near-duplicates (similar phrasing isn't required if both seats are
# building on the same cards).

import re as _re

# Tightened from 0.5 after the run-#3 audit (2026-06-02). F1/F3 hit
# citation Jaccard = exactly 0.50 — a 2-of-4 citation overlap is too
# loose for a cross-seat agreement label meant to carry convergence
# weight. 0.6 requires a clear majority overlap.
COHORT_CONVERGENCE_THRESHOLD = 0.6

_TITLE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "of", "in", "on",
    "at", "to", "for", "with", "by", "as", "that", "this", "these",
    "those", "be", "been", "not", "only", "if", "when", "where", "while",
    "into", "from", "than", "then", "so", "do", "does", "did", "has",
    "have", "had",
}


def _title_tokens(title: str) -> set[str]:
    """Lowercase, alphabetic tokens minus stop words. The token set is what
    the cohort-convergence detector uses to measure title overlap."""
    raw = _re.findall(r"[a-z0-9]+", (title or "").lower())
    return {t for t in raw if t and t not in _TITLE_STOPWORDS and len(t) > 1}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _citation_keys(f: MasterFusionReport) -> set[str]:
    """The report_id set this fusion cites — used for citation-overlap
    similarity, independent of which_field tag."""
    return {c.report_id for c in f.citations if c.report_id}


def _cohort_similarity(o: MasterFusionReport, g: MasterFusionReport) -> float:
    """Combined similarity score: max of title-token Jaccard and citation-
    set Jaccard. The two channels catch different failure modes —
    rephrased-title-same-cards (citation channel wins) and
    different-cards-same-insight (title channel wins)."""
    title_sim = _jaccard(_title_tokens(o.title), _title_tokens(g.title))
    cite_sim  = _jaccard(_citation_keys(o), _citation_keys(g))
    return max(title_sim, cite_sim)


def _combine_citations(a: list, b: list) -> list:
    """Concatenate two citation lists with (report_id, which_field) dedupe."""
    out = list(a)
    seen = {(c.report_id, c.which_field) for c in out}
    for c in b:
        key = (c.report_id, c.which_field)
        if key not in seen:
            out.append(c)
            seen.add(key)
    return out


def _seat_snapshot(f: MasterFusionReport) -> dict:
    """Frozen, JSON-safe snapshot of a fusion's pre-merge state. Used by
    `_merge_cohort_pair` to preserve EACH seat's framing on the merged
    output so the user-facing artifact can show what was suppressed by
    the keeper-pick. Addresses Blocker #3 of the run-#3 audit (the
    keeper-bias problem: Opus 5620 R3 tok vs GPT 3270 tok meant every
    BOTH_AGREE was structurally Opus-voiced)."""
    return {
        "title":            f.title,
        "claim":            f.claim,
        "reasoning":        f.reasoning,
        "limit":            f.limit,
        "confidence":       f.confidence.value,
        "agreement_status": f.agreement_status.value,
        "citations":        [c.to_dict() for c in f.citations],
    }


_CONFIDENCE_ORDER = {
    Confidence.LOW:    0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH:   2,
}


def _max_confidence(a: Confidence, b: Confidence) -> Confidence:
    """Return the higher-confidence of two grades. Tie → first arg."""
    return a if _CONFIDENCE_ORDER.get(a, 0) >= _CONFIDENCE_ORDER.get(b, 0) else b


def _keeper_score(f: MasterFusionReport) -> tuple:
    """Multi-factor keeper score for cohort-pair merging across seats.

    Replaces the prior prose-length-only rule. Runs #3 and #4 both shipped
    6/6 keeper_seat='opus' under the old rule because Opus's discursive
    elaboration unpacked every noun into clauses — it won length even when
    GPT cited more cards or more providers. The length signal was sampling
    voice density, not insight density.

    The tuple orders signals by information density first, voice density last:

      1. cards_cited     — count of distinct report_ids in citations
      2. providers       — count of distinct agent_ids in citations (proxy
                           for cross-provider grounding; the agent->provider
                           map isn't available at this layer, so distinct
                           agent_ids is the closest local signal)
      3. confidence_rank — HIGH=2, MEDIUM=1, LOW=0
      4. prose_len       — claim+reasoning+limit length (final tie-breaker)

    Python tuple comparison is lexicographic, so a fusion with 4 cited
    cards beats one with 3 cited cards regardless of which has longer
    prose. The voice bias is recovered as the bottom tie-break — used
    only when the upstream signals all tie.
    """
    cards_cited = {c.report_id for c in f.citations if c.report_id}
    agents      = {c.agent_id  for c in f.citations if c.agent_id}
    conf_rank   = _CONFIDENCE_ORDER.get(f.confidence, 0)
    prose_len   = len(f.claim) + len(f.reasoning) + len(f.limit)
    return (len(cards_cited), len(agents), conf_rank, prose_len)


def _merge_cohort_pair(
    o: MasterFusionReport,
    g: MasterFusionReport,
) -> MasterFusionReport:
    """Merge two cross-seat near-duplicate fusions into ONE cohort-
    convergence fusion. Picks the more substantive claim/reasoning/limit
    (longer total prose) AS THE PRIMARY SHIPPED text, but ALSO records
    BOTH seats' pre-merge snapshots on the result so the suppressed
    framing survives for audit. Combines citations with dedupe, sets
    agreement_status to BOTH_AGREE (the cohort-convergence signal), and
    takes `max(confidence)` rather than keeper's confidence.

    If EITHER input was DISPUTED, the merged fusion stays DISPUTED with
    empty claim/reasoning/limit (R4 supplied the per-angle prose).
    """
    o_snap = _seat_snapshot(o)
    g_snap = _seat_snapshot(g)

    if (o.agreement_status == AgreementStatus.DISPUTED
            or g.agreement_status == AgreementStatus.DISPUTED):
        # disputed wins — preserve dispute structure
        # keeper picks the title to display; the per-angle claim/reasoning
        # for each seat is preserved separately on disputed_angles.
        # Multi-factor score (cards, providers, confidence, then prose
        # length) replaces the prior title-length-only rule that was
        # symptomatic of the run-#3/#4 voice bias.
        keeper_is_opus = _keeper_score(o) >= _keeper_score(g)
        keeper = o if keeper_is_opus else g
        merged = MasterFusionReport(
            title=keeper.title,
            claim="", reasoning="", limit="",
            citations=_combine_citations(o.citations, g.citations),
            confidence=_max_confidence(o.confidence, g.confidence),
            agreement_status=AgreementStatus.DISPUTED,
            pre_merge_opus=o_snap,
            pre_merge_gpt=g_snap,
            keeper_seat="opus" if keeper_is_opus else "gpt",
        )
        merged.disputed_angles = list(o.disputed_angles or []) + list(g.disputed_angles or [])
        return merged

    # Multi-factor keeper-score (audit fix r4→r5): cards-cited, distinct-
    # agent-ids, confidence rank, prose length — in that order. The prior
    # rule was prose-length-only, which sampled voice density (Opus's
    # discursive style) instead of information density.
    keeper_is_opus = _keeper_score(o) >= _keeper_score(g)
    keeper, _other = (o, g) if keeper_is_opus else (g, o)
    return MasterFusionReport(
        title=keeper.title,
        claim=keeper.claim,
        reasoning=keeper.reasoning,
        limit=keeper.limit,
        citations=_combine_citations(o.citations, g.citations),
        confidence=_max_confidence(o.confidence, g.confidence),
        agreement_status=AgreementStatus.BOTH_AGREE,  # cohort convergence
        pre_merge_opus=o_snap,
        pre_merge_gpt=g_snap,
        keeper_seat="opus" if keeper_is_opus else "gpt",
    )


def _dedupe_within_seat(
    fusions: list[MasterFusionReport],
    threshold: float = COHORT_CONVERGENCE_THRESHOLD,
) -> tuple[list[MasterFusionReport], int]:
    """Pre-pass: collapse same-seat near-duplicates BEFORE bipartite
    cross-seat matching. Addresses Blocker #4 of the run-#3 audit
    (F1/F3 within Opus seat hit Jaccard=0.50 — the bipartite-only
    dedupe missed them because they were both Opus outputs, not
    cross-seat). Greedy by descending similarity. Keeper wins by
    (longer total prose, ties broken by max confidence). The dropped
    fusion's citations are merged into the keeper so no citation
    is lost; agreement_status of the keeper is preserved (the dropped
    fusion's status is discarded since both came from the same seat).
    Returns (kept_fusions, collapsed_count).
    """
    n = len(fusions)
    if n < 2:
        return list(fusions), 0

    # Compute pairwise similarities; only keep pairs above threshold
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cohort_similarity(fusions[i], fusions[j])
            if sim >= threshold:
                candidates.append((sim, i, j))
    candidates.sort(reverse=True)

    # Walk candidates greedily; mark the LOSER (shorter prose, lower
    # confidence on tie) as dropped, fold its citations into the keeper.
    dropped: set[int] = set()
    fusions_mut = list(fusions)
    collapsed = 0
    for sim, i, j in candidates:
        if i in dropped or j in dropped:
            continue
        def _score(f: MasterFusionReport) -> int:
            return len(f.claim) + len(f.reasoning) + len(f.limit)
        s_i, s_j = _score(fusions_mut[i]), _score(fusions_mut[j])
        if s_i == s_j:
            # tie-break: max confidence wins
            keeper_idx, loser_idx = (i, j) if _CONFIDENCE_ORDER.get(
                fusions_mut[i].confidence, 0,
            ) >= _CONFIDENCE_ORDER.get(fusions_mut[j].confidence, 0) else (j, i)
        else:
            keeper_idx, loser_idx = (i, j) if s_i >= s_j else (j, i)
        # Merge loser's citations into keeper; keeper title/claim/etc
        # are preserved as-is (same seat — we're not bridging frames).
        keeper = fusions_mut[keeper_idx]
        loser  = fusions_mut[loser_idx]
        fusions_mut[keeper_idx] = MasterFusionReport(
            title=keeper.title,
            claim=keeper.claim,
            reasoning=keeper.reasoning,
            limit=keeper.limit,
            citations=_combine_citations(keeper.citations, loser.citations),
            confidence=_max_confidence(keeper.confidence, loser.confidence),
            agreement_status=keeper.agreement_status,
            disputed_angles=list(keeper.disputed_angles or []),
        )
        dropped.add(loser_idx)
        collapsed += 1

    kept = [f for i, f in enumerate(fusions_mut) if i not in dropped]
    return kept, collapsed


def _dedupe_across_seats(
    opus_finals: list[MasterFusionReport],
    gpt_finals:  list[MasterFusionReport],
    threshold:   float = COHORT_CONVERGENCE_THRESHOLD,
) -> tuple[list[MasterFusionReport], int, int]:
    """Greedy bipartite match across the two seat outputs, AFTER a
    same-seat dedupe pre-pass on each side.

    Returns (merged_list, cross_seat_paired_count, same_seat_collapsed_count).
    Each output fusion is either:
      - a merged cross-seat pair (BOTH_AGREE / DISPUTED) — when one opus
        and one gpt fusion had max(title_jaccard, citation_jaccard) >= threshold
      - an unpaired opus fusion (status preserved — SOLO_OPUS / SOLO_GPT
        labels supplied by the seat's R3, MOSTLY_AGREE_REFINED preserved)
      - an unpaired gpt fusion (same)

    Pairing is greedy by descending similarity — best matches first, then
    consume both sides. Each fusion can participate in at most one pair.
    """
    # Pre-pass: collapse within-seat near-duplicates first. Blocker #4
    # of the run-#3 audit — F1/F3 (both Opus-side) were near-duplicates
    # the bipartite-only dedupe missed.
    opus_deduped, opus_collapsed = _dedupe_within_seat(opus_finals, threshold)
    gpt_deduped,  gpt_collapsed  = _dedupe_within_seat(gpt_finals,  threshold)
    same_seat_collapsed = opus_collapsed + gpt_collapsed

    # Compute all cross-pair similarities; sort desc
    candidates: list[tuple[float, int, int]] = []
    for i, o in enumerate(opus_deduped):
        for j, g in enumerate(gpt_deduped):
            sim = _cohort_similarity(o, g)
            if sim >= threshold:
                candidates.append((sim, i, j))
    candidates.sort(reverse=True)

    used_o: set[int] = set()
    used_g: set[int] = set()
    merged: list[MasterFusionReport] = []

    for sim, i, j in candidates:
        if i in used_o or j in used_g:
            continue
        merged.append(_merge_cohort_pair(opus_deduped[i], gpt_deduped[j]))
        used_o.add(i)
        used_g.add(j)

    paired_count = len(merged)
    # Append unpaired fusions from both sides; attach pre_merge snapshot
    # and set keeper_seat so downstream audit can identify each fusion's
    # provenance. Fix 4 (audit r4): MAR fusions F6+F7 in run #4 shipped
    # with keeper_seat='' and both pre_merge_* = None — the unpaired
    # passthrough silently dropped provenance. SOLO_OPUS/SOLO_GPT keep
    # keeper_seat empty because the seat is already encoded in
    # agreement_status; setting it again would be redundant.
    for i, o in enumerate(opus_deduped):
        if i not in used_o:
            o.pre_merge_opus = _seat_snapshot(o)
            if o.agreement_status not in (AgreementStatus.SOLO_OPUS, AgreementStatus.SOLO_GPT):
                o.keeper_seat = "opus"
            merged.append(o)
    for j, g in enumerate(gpt_deduped):
        if j not in used_g:
            g.pre_merge_gpt = _seat_snapshot(g)
            if g.agreement_status not in (AgreementStatus.SOLO_OPUS, AgreementStatus.SOLO_GPT):
                g.keeper_seat = "gpt"
            merged.append(g)

    return merged, paired_count, same_seat_collapsed


# ---------------------------------------------------------------------------
# Field-attribution validator — fixes citation `which_field` slot bugs
# ---------------------------------------------------------------------------
#
# Background (audit, 2026-06-02): Fusion 13 in dry-run #2 had 2 excerpts
# tagged to the WRONG card field — real verbatim text from the card, but
# the citation's `which_field` named a different slot (e.g., excerpt is
# substring of `bridge`, tagged `use`). Cheap to fix: when an excerpt is
# a verbatim substring of any field on the cited card, the tag should
# point to THAT field.
#
# Importantly, paraphrases (excerpts that aren't a verbatim substring of
# any field on the card) are LEFT ALONE — they're likely faithful
# rewordings that preserve load-bearing content, not slot errors.

_CARD_FIELDS = ("spark", "source_shape", "bridge", "use", "limit")


def _excerpt_is_in_field(excerpt: str, field_text: str, min_match_len: int = 16) -> bool:
    """Substring check with a minimum length guard.

    Short snippets (e.g., a single common word) shouldn't trigger
    re-attribution — too noisy. The guard requires the excerpt itself
    to be at least `min_match_len` chars before any match counts.
    """
    if not excerpt or not field_text:
        return False
    if len(excerpt) < min_match_len:
        return False
    return excerpt.lower() in field_text.lower()


def _fix_citation_field_for(cite: "CardReference", card: ArticulatedCard) -> bool:
    """If `cite.excerpt` is a verbatim substring of a field on `card`
    other than the one named in `cite.which_field`, re-attribute the
    tag to the correct field. Returns True when a fix was applied.
    """
    excerpt = cite.excerpt or ""
    named = (cite.which_field or "").lower()
    # First check the named field — if it already matches, no fix needed.
    if named in _CARD_FIELDS:
        named_text = getattr(card, named, "") or ""
        if _excerpt_is_in_field(excerpt, named_text):
            return False
    # Try the other fields
    for fld in _CARD_FIELDS:
        if fld == named:
            continue
        other_text = getattr(card, fld, "") or ""
        if _excerpt_is_in_field(excerpt, other_text):
            cite.which_field = fld
            return True
    # Not a verbatim match anywhere — leave alone (paraphrase)
    return False


def _validate_citation_field_attribution(
    fusions: list[MasterFusionReport],
    cards:   list[ArticulatedCard],
) -> int:
    """Walk every fusion's citations (and disputed-angle citations) and
    re-attribute `which_field` when the excerpt is a verbatim substring
    of a different field on the cited card.

    Returns the number of fixes applied. Pure data-only correction —
    citation excerpts and other fields are never modified. Excerpts
    that aren't a verbatim substring of any field are left unchanged
    (likely paraphrases, not slot errors).
    """
    cards_by_id = {c.report_id: c for c in cards}
    fixed = 0
    for f in fusions:
        for cite in f.citations:
            card = cards_by_id.get(cite.report_id)
            if card is None:
                continue
            if _fix_citation_field_for(cite, card):
                fixed += 1
        for angle in f.disputed_angles:
            for cite in angle.citations:
                card = cards_by_id.get(cite.report_id)
                if card is None:
                    continue
                if _fix_citation_field_for(cite, card):
                    fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# Hunch-coverage surveillance (Fix 6, audit r4→r5)
# ---------------------------------------------------------------------------
#
# Pure observability — extracts hunch labels from the cushion's CURRENT_MAP
# block, then scans every master fusion's text for lexical references to
# each label. Does NOT call the LLM, does NOT alter prompts, does NOT gate
# output. Surfaces a coverage map the user / dashboard can read post-run.
#
# Background (run #4): Nikhil's cushion supplied 4 hunches (Markov chains,
# butterfly effect, Heisenberg uncertainty, junior+senior scientist
# hierarchy). One of them — Butterfly — was absent from all 8 master
# fusions and nothing flagged the omission. The architecture has no
# round-prompt mechanism that asks "did you exercise the hunches?", so
# this surveillance is the cheap detection path until a future audit
# decides whether to wire it into a prompt-level law.


_HUNCH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "of", "in", "on",
    "at", "to", "for", "with", "by", "as", "that", "this", "these",
    "those", "be", "been", "not", "only", "if", "when", "where", "while",
    "into", "from", "than", "then", "so", "do", "does", "did", "has",
    "have", "had", "your", "you", "i", "we", "they", "it", "its",
    "about", "like", "just", "very", "more", "most",
}


def _extract_hunches(cushion: CushionGraph | None) -> list[str]:
    """Pull hunch labels from cushion.raw_input.current_map.content.

    The Wandering Room UI binds the 'Hunches' field to backend CURRENT_MAP.
    Expected formats (lenient — covers numbered, bulleted, and
    blank-line-separated paragraphs):

      1. Markov chains: each step depends on the previous...
      2. Butterfly effect — tiny inputs cascade...
      * Heisenberg uncertainty: you can't measure...

    Strategy: split on blank lines OR newline-with-leading-marker, then
    for each block, take the first line, strip leading bullets/numbers,
    and take up to the first colon / em-dash / en-dash as the label.

    Returns a list of short, lowercase labels. Empty list when cushion or
    current_map is missing.
    """
    if cushion is None or cushion.raw_input is None:
        return []
    text = getattr(cushion.raw_input.current_map, "content", "") or ""
    if not text.strip():
        return []

    # First try blank-line block split; if that yields only one block,
    # also try splitting on bullet/numbered prefixes so single-block
    # text with inline numbering still extracts multiple hunches.
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) < 2:
        blocks = [b for b in re.split(r"\n(?=\s*(?:[\-\*]|\d+[\.\)]))", text) if b.strip()]

    labels: list[str] = []
    for block in blocks:
        first_line = block.strip().split("\n", 1)[0].strip()
        # strip leading bullet/number markers like "1.", "2)", "-", "*"
        first_line = re.sub(r"^[\-\*]+\s*|^\d+[\.\)]\s*", "", first_line).strip()
        # cut at first colon, em-dash, en-dash, or hyphen-with-space
        m = re.match(r"^([^:—–]{3,80}?)(?:\s*[:—–]|\s+-\s+|$)", first_line)
        label = (m.group(1) if m else first_line[:60]).strip().lower()
        if label and label not in labels:
            labels.append(label)
    return labels


def _hunch_anchors(label: str) -> list[str]:
    """Return the set of content-word anchors for a hunch label.

    A fusion is considered to reference the hunch when its text contains
    ANY of these anchors. Multi-anchor OR matching avoids the
    "longest word wins" failure mode where the more distinctive term
    loses to a generic one (e.g. label="heisenberg uncertainty" — the
    specific name "heisenberg" matters more than the generic noun
    "uncertainty", but the latter is one character longer).

    Strategy: every non-stopword content word ≥ 5 chars becomes an
    anchor. Falls back to the whole stripped label when no qualifying
    word is found.
    """
    words = [
        w for w in re.findall(r"[a-z]{5,}", label.lower())
        if w not in _HUNCH_STOPWORDS
    ]
    if not words:
        return [label.strip().lower()] if label.strip() else []
    # Deduplicate while preserving discovery order
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _compute_hunch_coverage(
    cushion: CushionGraph | None,
    fusions: list[MasterFusionReport],
) -> dict[str, list[str]]:
    """For each hunch, list the fusion titles whose text references it.

    Lexical (cheap) matching on title+claim+reasoning+limit, OR-matched
    against every content-word anchor for the label (so the rarer of
    "heisenberg" / "uncertainty" still triggers a match when only one
    appears in the fusion). Returns {hunch_label: [fusion_title, ...]}
    with the same keys even when a list is empty — empty list under a
    hunch is the "uncovered" signal.
    """
    labels = _extract_hunches(cushion)
    coverage: dict[str, list[str]] = {l: [] for l in labels}
    if not labels:
        return coverage
    anchors_by_label = {l: _hunch_anchors(l) for l in labels}
    for f in fusions:
        haystack = " ".join((f.title, f.claim, f.reasoning, f.limit)).lower()
        # Include disputed-angle text too — angles carry the real prose on
        # disputed fusions (parent claim/reasoning/limit are empty strings).
        for angle in f.disputed_angles:
            haystack += " " + " ".join((angle.claim, angle.reasoning, angle.limit)).lower()
        for label, anchors in anchors_by_label.items():
            if any(a and a in haystack for a in anchors):
                coverage[label].append(f.title)
    return coverage


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def master_synthesize(
    *,
    cushion:          CushionGraph | None,
    cards:            list[ArticulatedCard],
    synthesis_map:    SynthesisMap | None,
    client:           LLMClient,
    progress:         MasterSynthesisProgress | None = None,
    cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
    opus_model:       str = OPUS_SEAT_MODEL,
    gpt_model:        str = GPT_SEAT_MODEL,
    agent_provider_map: dict[str, str] | None = None,
) -> MasterSynthesis:
    """Produce master fusion reports from a dossier's articulated cards.

    Pipeline:
      R1 (parallel) — both seats draft 3-5 candidate fusions
      R2 (parallel) — each seat critiques the OTHER's drafts
      R3 (parallel) — each seat produces FINAL fusions w/ agreement_status
      R4 (parallel, conditional) — disputed fusions: each seat writes
         its own angle; orchestrator zips angles into MasterFusionReport.disputed_angles

    Budget enforcement: every LLM call is cost-checked AFTER it returns.
    If cumulative spend exceeds `cost_ceiling_usd`, the orchestrator
    catches MasterSynthesisBudgetExceeded and returns whatever was
    assembled so far. Partial > nothing.

    Empty input → empty result (no LLM calls fired).
    """
    result = MasterSynthesis(cost_ceiling_usd=cost_ceiling_usd)
    progress = progress or MasterSynthesisProgress()

    if not cards:
        progress.emit("empty_input", {"reason": "no cards provided"})
        return result

    progress.emit("starting", {
        "card_count":       len(cards),
        "cost_ceiling_usd": cost_ceiling_usd,
        "opus_model":       opus_model,
        "gpt_model":        gpt_model,
    })

    system = compose_system_prompt(_DOCTRINE_PREAMBLE, mode="master_synthesizer")
    draft_payload = _build_draft_payload(cushion, cards, synthesis_map)

    try:
        # ─── R1 — DRAFT (parallel) ──────────────────────────────────────
        progress.emit("round_started", {"round": "R1_draft", "seats": ["opus", "gpt"]})
        opus_draft_task = asyncio.create_task(_call_with_budget(
            client=client, system_prompt=system, user_message=draft_payload,
            domain=SEAT_DOMAIN, concept=SEAT_CONCEPT_DRAFT,
            seat=Seat.OPUS, model_slug=opus_model, round_name="R1_draft",
            result=result, max_tokens=MAX_TOKENS_DRAFT,
        ))
        gpt_draft_task = asyncio.create_task(_call_with_budget(
            client=client, system_prompt=system, user_message=draft_payload,
            domain=SEAT_DOMAIN, concept=SEAT_CONCEPT_DRAFT,
            seat=Seat.GPT, model_slug=gpt_model, round_name="R1_draft",
            result=result, max_tokens=MAX_TOKENS_DRAFT,
        ))
        opus_draft_resp, gpt_draft_resp = await asyncio.gather(opus_draft_task, gpt_draft_task)
        opus_drafts = _parse_json_safely(opus_draft_resp.content, default=[])
        gpt_drafts  = _parse_json_safely(gpt_draft_resp.content,  default=[])
        if not isinstance(opus_drafts, list): opus_drafts = []
        if not isinstance(gpt_drafts,  list): gpt_drafts  = []
        result.rounds_completed.append("R1_draft")
        progress.emit("round_complete", {
            "round": "R1_draft", "opus_count": len(opus_drafts), "gpt_count": len(gpt_drafts),
            "cumulative_cost_usd": round(result.total_cost_usd, 4),
        })

        # ─── R2 — CRITIQUE (parallel) ───────────────────────────────────
        progress.emit("round_started", {"round": "R2_critique"})
        opus_critique_payload = _build_critique_payload(cushion, cards, gpt_drafts)
        gpt_critique_payload  = _build_critique_payload(cushion, cards, opus_drafts)

        opus_critique_task = asyncio.create_task(_call_with_budget(
            client=client, system_prompt=system, user_message=opus_critique_payload,
            domain=SEAT_DOMAIN, concept=SEAT_CONCEPT_CRITIQUE,
            seat=Seat.OPUS, model_slug=opus_model, round_name="R2_critique",
            result=result, max_tokens=MAX_TOKENS_CRITIQUE,
        ))
        gpt_critique_task = asyncio.create_task(_call_with_budget(
            client=client, system_prompt=system, user_message=gpt_critique_payload,
            domain=SEAT_DOMAIN, concept=SEAT_CONCEPT_CRITIQUE,
            seat=Seat.GPT, model_slug=gpt_model, round_name="R2_critique",
            result=result, max_tokens=MAX_TOKENS_CRITIQUE,
        ))
        opus_critique_resp, gpt_critique_resp = await asyncio.gather(
            opus_critique_task, gpt_critique_task,
        )
        opus_critique = _parse_json_safely(opus_critique_resp.content, default=[])  # Opus's notes on GPT's drafts
        gpt_critique  = _parse_json_safely(gpt_critique_resp.content,  default=[])  # GPT's notes on Opus's drafts
        if not isinstance(opus_critique, list): opus_critique = []
        if not isinstance(gpt_critique,  list): gpt_critique  = []
        result.rounds_completed.append("R2_critique")
        # Blocker #1: persist R2 critique BODIES (not just counts) so a
        # downstream auditor can answer "was the critique rubber-stamped
        # or substantive?" from the artifact alone.
        result.r2_critique_opus = list(opus_critique)
        result.r2_critique_gpt  = list(gpt_critique)
        # Quick distribution counts emit so progress consumers see the
        # agree/refine/disagree shape inline without parsing the body.
        def _ann_dist(crit_list: list) -> dict[str, int]:
            d: dict[str, int] = {"agree": 0, "refine": 0, "disagree": 0, "other": 0}
            for c in crit_list:
                if not isinstance(c, dict): continue
                ann = str(c.get("annotation", "")).lower().strip()
                d[ann if ann in d else "other"] += 1
            return d
        progress.emit("round_complete", {
            "round": "R2_critique",
            "opus_annotations": len(opus_critique),
            "gpt_annotations":  len(gpt_critique),
            "opus_annotation_dist": _ann_dist(opus_critique),
            "gpt_annotation_dist":  _ann_dist(gpt_critique),
            "cumulative_cost_usd": round(result.total_cost_usd, 4),
        })

        # ─── R3 — FINAL (parallel) ──────────────────────────────────────
        progress.emit("round_started", {"round": "R3_final"})
        opus_final_payload = _build_final_payload(
            cushion, cards,
            my_drafts=opus_drafts, critique_of_me=gpt_critique,
            other_drafts=gpt_drafts, my_critique=opus_critique,
            seat=Seat.OPUS,
        )
        gpt_final_payload = _build_final_payload(
            cushion, cards,
            my_drafts=gpt_drafts, critique_of_me=opus_critique,
            other_drafts=opus_drafts, my_critique=gpt_critique,
            seat=Seat.GPT,
        )
        opus_final_task = asyncio.create_task(_call_with_budget(
            client=client, system_prompt=system, user_message=opus_final_payload,
            domain=SEAT_DOMAIN, concept=SEAT_CONCEPT_FINAL,
            seat=Seat.OPUS, model_slug=opus_model, round_name="R3_final",
            result=result, max_tokens=MAX_TOKENS_FINAL,
        ))
        gpt_final_task = asyncio.create_task(_call_with_budget(
            client=client, system_prompt=system, user_message=gpt_final_payload,
            domain=SEAT_DOMAIN, concept=SEAT_CONCEPT_FINAL,
            seat=Seat.GPT, model_slug=gpt_model, round_name="R3_final",
            result=result, max_tokens=MAX_TOKENS_FINAL,
        ))
        opus_final_resp, gpt_final_resp = await asyncio.gather(opus_final_task, gpt_final_task)

        opus_finals_raw = _parse_json_safely(opus_final_resp.content, default=[])
        gpt_finals_raw  = _parse_json_safely(gpt_final_resp.content,  default=[])
        if not isinstance(opus_finals_raw, list): opus_finals_raw = []
        if not isinstance(gpt_finals_raw,  list): gpt_finals_raw  = []
        result.rounds_completed.append("R3_final")

        # Parse + label seats correctly. If R3 said "solo_opus" but it
        # came from the GPT seat's call, rewrite to solo_gpt (the model
        # mislabeled itself). Same for the reverse.
        opus_finals: list[MasterFusionReport] = []
        gpt_finals:  list[MasterFusionReport] = []
        for raw in opus_finals_raw:
            f = _parse_final_fusion(raw)
            if f is None: continue
            if f.agreement_status == AgreementStatus.SOLO_GPT:
                f.agreement_status = AgreementStatus.SOLO_OPUS  # came from Opus seat
            opus_finals.append(f)
        for raw in gpt_finals_raw:
            f = _parse_final_fusion(raw)
            if f is None: continue
            if f.agreement_status == AgreementStatus.SOLO_OPUS:
                f.agreement_status = AgreementStatus.SOLO_GPT
            gpt_finals.append(f)

        # Blocker #1: persist R3 pre-merge per-seat fusion lists BEFORE
        # _dedupe_across_seats collapses cohort pairs. Without this, the
        # merged result alone hides which seat said what.
        result.r3_pre_merge_opus = [f.to_dict() for f in opus_finals]
        result.r3_pre_merge_gpt  = [f.to_dict() for f in gpt_finals]

        # Cross-seat dedupe: pair Opus fusions to GPT fusions that are
        # cohort-convergent (max of title-token Jaccard or citation-set
        # Jaccard >= COHORT_CONVERGENCE_THRESHOLD). Run-#3 audit tightened
        # threshold 0.5 → 0.6 and added a within-seat pre-pass to catch
        # same-seat near-duplicates the bipartite-only logic missed
        # (Blocker #4: F1/F3 in run #3 were both Opus-side, Jaccard=0.5).
        # Greedy by descending similarity; each fusion participates in
        # at most one pair. Unpaired fusions retain their seat-supplied
        # status (SOLO_OPUS / SOLO_GPT / MOSTLY_AGREE_REFINED preserved).
        merged_fusions, cohort_pair_count, same_seat_collapsed = _dedupe_across_seats(
            opus_finals, gpt_finals,
            threshold=COHORT_CONVERGENCE_THRESHOLD,
        )
        result.same_seat_pairs_collapsed = same_seat_collapsed
        merged_fusions = [f for f in merged_fusions if _is_valid_fusion(f)]

        # Field-attribution validator — when a citation's excerpt is a
        # verbatim substring of a DIFFERENT field on the cited card,
        # re-attribute the `which_field` tag. Paraphrases (excerpts
        # that aren't a verbatim substring of any field) are left as-is.
        # Fixes the run-#2 dry-run case where Fusion 13 had 2 excerpts
        # tagged to wrong fields (real text, wrong slot).
        field_fixes_applied = _validate_citation_field_attribution(
            merged_fusions, cards,
        )
        if field_fixes_applied:
            log.info(
                "master_synth: field-attribution validator re-tagged %d citation(s)",
                field_fixes_applied,
            )

        pre_r4_disputed = sum(1 for f in merged_fusions if f.agreement_status == AgreementStatus.DISPUTED)
        pre_r4_agreed   = sum(1 for f in merged_fusions if f.agreement_status in (AgreementStatus.BOTH_AGREE, AgreementStatus.MOSTLY_AGREE_REFINED))
        progress.emit("round_complete", {
            "round": "R3_final",
            "merged_fusion_count":          len(merged_fusions),
            "agreed":                       pre_r4_agreed,
            "disputed":                     pre_r4_disputed,
            # Per-seat pre-merge counts so progress consumers don't have
            # to re-parse the result JSON to know what each seat emitted.
            "opus_r3_count":                len(opus_finals_raw),
            "gpt_r3_count":                 len(gpt_finals_raw),
            "opus_r3_parsed":               len(opus_finals),
            "gpt_r3_parsed":                len(gpt_finals),
            "cohort_pairs_collapsed":       cohort_pair_count,
            "same_seat_pairs_collapsed":    same_seat_collapsed,
            "citation_field_fixes":         field_fixes_applied,
            "cumulative_cost_usd":          round(result.total_cost_usd, 4),
        })

        # ─── R4 — DISPUTED ANGLES (parallel, conditional) ──────────────
        disputed = [f for f in merged_fusions if f.agreement_status == AgreementStatus.DISPUTED]
        if disputed:
            progress.emit("round_started", {"round": "R4_angles", "disputed_count": len(disputed)})
            disputed_payload_objs = [
                {"title": f.title, "citations": [c.to_dict() for c in f.citations]}
                for f in disputed
            ]
            opus_angle_payload = _build_angle_payload(cushion, cards, disputed_payload_objs, Seat.OPUS)
            gpt_angle_payload  = _build_angle_payload(cushion, cards, disputed_payload_objs, Seat.GPT)

            opus_angle_task = asyncio.create_task(_call_with_budget(
                client=client, system_prompt=system, user_message=opus_angle_payload,
                domain=SEAT_DOMAIN, concept=SEAT_CONCEPT_ANGLE,
                seat=Seat.OPUS, model_slug=opus_model, round_name="R4_angles",
                result=result, max_tokens=MAX_TOKENS_ANGLE,
            ))
            gpt_angle_task = asyncio.create_task(_call_with_budget(
                client=client, system_prompt=system, user_message=gpt_angle_payload,
                domain=SEAT_DOMAIN, concept=SEAT_CONCEPT_ANGLE,
                seat=Seat.GPT, model_slug=gpt_model, round_name="R4_angles",
                result=result, max_tokens=MAX_TOKENS_ANGLE,
            ))
            opus_angle_resp, gpt_angle_resp = await asyncio.gather(
                opus_angle_task, gpt_angle_task,
            )
            opus_angles_raw = _parse_json_safely(opus_angle_resp.content, default=[])
            gpt_angles_raw  = _parse_json_safely(gpt_angle_resp.content,  default=[])
            if not isinstance(opus_angles_raw, list): opus_angles_raw = []
            if not isinstance(gpt_angles_raw,  list): gpt_angles_raw  = []
            result.rounds_completed.append("R4_angles")

            # Attach each angle to its disputed fusion by title match
            def _angle_from_raw(raw, seat: Seat) -> DisputedAngle | None:
                if not isinstance(raw, dict): return None
                return DisputedAngle(
                    seat=seat,
                    claim=str(raw.get("claim", "")).strip(),
                    reasoning=str(raw.get("reasoning", "")).strip(),
                    limit=str(raw.get("limit", "")).strip(),
                    citations=_parse_citations(raw.get("citations", [])),
                )

            by_title: dict[str, list[DisputedAngle]] = {}
            for raw in opus_angles_raw:
                a = _angle_from_raw(raw, Seat.OPUS)
                if a is None: continue
                title = str(raw.get("title", "")).strip().lower()
                by_title.setdefault(title, []).append(a)
            for raw in gpt_angles_raw:
                a = _angle_from_raw(raw, Seat.GPT)
                if a is None: continue
                title = str(raw.get("title", "")).strip().lower()
                by_title.setdefault(title, []).append(a)

            inherited_citations_count = 0
            for f in disputed:
                key = f.title.strip().lower()
                angles = by_title.get(key, [])
                # Inherit parent fusion citations on any angle that came back
                # citation-empty. Fix 5 (audit r4): GPT R4 angles in run #4
                # returned 0 citations on both disputed fusions despite the
                # prompt — the inherit fallback keeps the angle from being
                # ungrounded. Counted in surveillance so a regression on the
                # prompt-side instruction shows up in /tmp logs.
                for angle in angles:
                    if not angle.citations and f.citations:
                        angle.citations = list(f.citations)
                        inherited_citations_count += 1
                f.disputed_angles = angles
            progress.emit("round_complete", {
                "round": "R4_angles",
                "angles_attached": sum(len(f.disputed_angles) for f in disputed),
                "angles_with_inherited_citations": inherited_citations_count,
                "cumulative_cost_usd": round(result.total_cost_usd, 4),
            })

        # Populate citation counts on every fusion (run-#3 polish).
        # citation_agent_count = unique agent_ids cited.
        # citation_provider_count = unique providers cited (needs the
        #   caller-supplied agent_provider_map; 0 when unavailable).
        for f in merged_fusions:
            _compute_citation_counts(f, agent_provider_map)

        # Hunch-coverage surveillance (Fix 6, audit r4→r5). Pure
        # observability — does not gate output or regenerate. Logs which
        # hunches the user supplied got exercised by at least one master
        # fusion and which were ignored. Run #4 inspiration: Butterfly
        # hunch absent from all 8 fusions, nothing flagged it.
        result.hunch_coverage = _compute_hunch_coverage(cushion, merged_fusions)

        # Final counts — Blocker #2: split agreement_count into
        # per-status fields. The legacy aggregates (dispute_count,
        # agreement_count, solo_count) are still emitted but are now
        # DERIVED from the per-status fields so callers can choose.
        result.master_fusions = merged_fusions
        result.both_agree_count            = sum(1 for f in merged_fusions if f.agreement_status == AgreementStatus.BOTH_AGREE)
        result.mostly_agree_refined_count  = sum(1 for f in merged_fusions if f.agreement_status == AgreementStatus.MOSTLY_AGREE_REFINED)
        result.solo_opus_count             = sum(1 for f in merged_fusions if f.agreement_status == AgreementStatus.SOLO_OPUS)
        result.solo_gpt_count              = sum(1 for f in merged_fusions if f.agreement_status == AgreementStatus.SOLO_GPT)
        result.disputed_count              = sum(1 for f in merged_fusions if f.agreement_status == AgreementStatus.DISPUTED)
        # Legacy aggregates (still set for backwards compat with old
        # readers like the /tmp/master_synth_dry_run.py harness).
        result.dispute_count   = result.disputed_count
        result.agreement_count = result.both_agree_count + result.mostly_agree_refined_count
        result.solo_count      = result.solo_opus_count + result.solo_gpt_count

        progress.emit("complete", {
            "fusion_count":               len(merged_fusions),
            # per-status (canonical)
            "both_agree_count":           result.both_agree_count,
            "mostly_agree_refined_count": result.mostly_agree_refined_count,
            "solo_opus_count":            result.solo_opus_count,
            "solo_gpt_count":             result.solo_gpt_count,
            "disputed_count":             result.disputed_count,
            # legacy aggregates (deprecated)
            "dispute_count":              result.dispute_count,
            "agreement_count":            result.agreement_count,
            "solo_count":                 result.solo_count,
            "same_seat_pairs_collapsed":  result.same_seat_pairs_collapsed,
            # Fix 6: surveillance summary — count of hunches uncovered
            # (i.e. exercised by zero fusions) so the progress log surfaces
            # the gap without dumping the full coverage map.
            "hunches_total":              len(result.hunch_coverage),
            "hunches_uncovered":          sum(1 for v in result.hunch_coverage.values() if not v),
            "total_cost_usd":             round(result.total_cost_usd, 4),
            "rounds_completed":           list(result.rounds_completed),
        })

    except MasterSynthesisBudgetExceeded:
        # Whatever we assembled before the cap is preserved on `result`.
        # The orchestrator's truncated_by_budget flag is already set by
        # _call_with_budget. Surface the truncation in progress emission.
        progress.emit("truncated_by_budget", {
            "rounds_completed":  list(result.rounds_completed),
            "fusions_so_far":    len(result.master_fusions),
            "total_cost_usd":    round(result.total_cost_usd, 4),
            "ceiling_usd":       result.cost_ceiling_usd,
            "reason":            result.truncation_reason,
        })

    except Exception as e:
        log.exception("master_synthesize: unhandled error")
        progress.emit("error", {"message": str(e)[:200]})
        # Don't re-raise — return what we have so callers can render partial.

    return result


__all__ = [
    "OPUS_SEAT_MODEL",
    "GPT_SEAT_MODEL",
    "DEFAULT_COST_CEILING_USD",
    "MasterSynthesisBudgetExceeded",
    "Seat",
    "AgreementStatus",
    "CritiqueAnnotation",
    "CardReference",
    "DisputedAngle",
    "MasterFusionReport",
    "MasterSynthesis",
    "MasterSynthesisProgress",
    "master_synthesize",
]
