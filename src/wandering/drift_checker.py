"""
Drift-checker — the supervisor.

WHY THIS EXISTS
---------------
The blender is powerful and, left alone, can wander off. Give it a vague
anchor and four cards and it will happily produce a brilliant blend that
solves a DIFFERENT problem than the cushion. The drift-checker is the
supervisor on the floor whose ONLY job is to keep the blender in the main
lane — the cushion.

WHAT IT IS (AND IS NOT)
-----------------------
- It is NOT a second blender. It does not improve, rewrite, or re-blend.
- It is NOT a quality / novelty / correctness judge. It does not care
  whether a blend is clever or true — the web-verification sorter (next
  stage) handles "is this real." The drift-checker cares about ONE axis:
  is this blend still serving the cushion, or has it drifted toward a
  different problem?
- It stays OUT of the way. The default verdict is on_course. It speaks up
  only on clear drift, and when it does its one move is a redirect: "this
  has drifted toward X; the main lane is Y — get back to it."

In the LINEAR v1 pipeline the drift-checker FLAGS. Drifting blends are
quarantined (recorded, shown to the human, not passed downstream as
primary); on-course blends proceed to blend verification. The reframe-loop
where a high-confidence drift-checker COMMANDS the blender to re-blend
against a twisted "phantom" anchor is the deferred v2 shuffling system —
not built here.

MODEL
-----
DeepSeek V4 Pro by default, routed via OpenRouter (the existing
OPENROUTER_API_KEY — no new key). A DIFFERENT RLHF lineage from the
Anthropic wanderers/blender on purpose: an independent eye is less likely
to rubber-stamp an Anthropic model's own drift. DeepSeek uses the standard
temperature path (no output_config.effort).

ISOLATION
---------
Imports cushion + Blend type + LLMClient + pricing + json helpers. Composes
its own system prompt at the call site. Makes NO web calls, NO bin
decisions, NO blends. One LLM call, budget-capped.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from src.identity import compose_system_prompt
from src.llm.client import LLMClient, LLMResponse
from src.llm.provider_map import get_pricing
from src.wandering.blender import Blend
from src.wandering.cushion import CushionGraph
from src.wandering.master_synthesizer import _parse_json_safely

log = logging.getLogger("constellax.wandering.drift_checker")


# ---------------------------------------------------------------------------
# Model + tuning
# ---------------------------------------------------------------------------

#: The supervisor seat. Sonnet 4-6 by Nikhil's directive (2026-06-17): the drift
#: checker is a Sonnet "department". NOTE/TRADEOFF: this seat was originally a
#: DIFFERENT lineage (DeepSeek) from the Anthropic blender ON PURPOSE — a
#: cross-lineage check is harder to fool than a same-lineage one. On Sonnet it
#: shares lineage with an Anthropic blender; flagged to Nikhil. (Was deepseek-v4-pro.)
DRIFT_CHECKER_MODEL = "anthropic/claude-sonnet-4-6"

DRIFT_DOMAIN  = "drift_checker"
DRIFT_CONCEPT = "drift_check"

MAX_TOKENS_DRIFT = 4096

#: Direction-judging is near-deterministic — run it cold.
DRIFT_TEMPERATURE = 0.1

#: Below this cushion-resonance, a blend is flagged as drifting even if the
#: model didn't explicitly say so. Belt-and-braces on the supervisor.
RESONANCE_DRIFT_FLOOR = 0.4

DEFAULT_COST_CEILING_USD = 8.00


class DriftBudgetExceeded(Exception):
    """Raised when cumulative spend would exceed the ceiling."""


def _call_cost_usd(model_slug: str, response: LLMResponse) -> float:
    in_price, out_price = get_pricing(model_slug)
    return (
        (response.input_tokens  or 0) / 1_000_000 * in_price
        + (response.output_tokens or 0) / 1_000_000 * out_price
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DriftVerdict:
    """The supervisor's call on one blend — direction only, not quality."""
    blend_id:    str
    on_course:   bool  = True
    resonance:   float = 1.0   # 0..1 fidelity to the cushion
    drift_reason: str  = ""    # if drifting: what problem it drifted toward
    redirect:    str   = ""    # if drifting: the one-line "get back to" note

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    """Container the drift-checker returns."""
    verdicts:            list[DriftVerdict] = field(default_factory=list)
    on_course_ids:       list[str]          = field(default_factory=list)
    drifting_ids:        list[str]          = field(default_factory=list)
    total_cost_usd:      float              = 0.0
    cost_ceiling_usd:    float              = DEFAULT_COST_CEILING_USD
    truncated_by_budget: bool               = False
    truncation_reason:   str                = ""
    parser_notes:        list[dict]         = field(default_factory=list)
    call_log:            list[dict]         = field(default_factory=list)
    model:               str                = ""

    def verdict_for(self, blend_id: str) -> DriftVerdict | None:
        for v in self.verdicts:
            if v.blend_id == blend_id:
                return v
        return None

    def to_dict(self) -> dict:
        return {
            "verdicts":            [v.to_dict() for v in self.verdicts],
            "on_course_ids":       list(self.on_course_ids),
            "drifting_ids":        list(self.drifting_ids),
            "total_cost_usd":      round(self.total_cost_usd, 4),
            "cost_ceiling_usd":    self.cost_ceiling_usd,
            "truncated_by_budget": self.truncated_by_budget,
            "truncation_reason":   self.truncation_reason,
            "parser_notes":        list(self.parser_notes),
            "call_log":            list(self.call_log),
            "model":               self.model,
        }


@dataclass
class DriftProgress:
    on_event: Callable[[str, dict], None] | None = None
    events:   list[dict] = field(default_factory=list)

    def emit(self, name: str, payload: dict | None = None) -> None:
        payload = payload or {}
        entry = {"name": name, "ts": time.time(), **payload}
        self.events.append(entry)
        log.info("[drift] %s %s", name, payload)
        if self.on_event is not None:
            try:
                self.on_event(name, payload)
            except Exception as e:  # pragma: no cover
                log.warning("drift progress on_event raised (ignored): %s", e)


# ---------------------------------------------------------------------------
# Doctrine
# ---------------------------------------------------------------------------


_DRIFT_DOCTRINE = """\
You are THE SUPERVISOR. A blender has produced candidate blended concepts
from a set of cards. Your ONLY job is to keep the work in the main lane —
the cushion's QUESTION (the checkpoint in the payload: `cushion_problem`
frames the situation, `cushion_question` is the precise target it must serve).

You judge ONE axis and one axis only: DIRECTIONAL FIDELITY. Is each blend
still in service of the cushion, or has it drifted toward a DIFFERENT
problem than the one stated?

What you are NOT:
  - You are NOT judging whether a blend is clever, novel, correct, or true.
    A later web-verification stage handles "is this real." Ignore quality.
  - You are NOT a blender. Do not rewrite, improve, or propose new blends.
  - You are NOT picky. Stay OUT of the way. A blend that genuinely advances
    the cushion is on_course even if it's imperfect. Default to on_course.

When you DO speak up — only on clear drift:
  - drift means the blend has quietly substituted a different problem for
    the cushion: it solves something adjacent, or generalizes away from the
    user's actual question, or optimizes a metric the cushion never asked
    for.
  - State, in `drift_reason`, the DIFFERENT problem it drifted toward.
  - State, in `redirect`, the ONE-LINE course correction: "get back to:
    <the cushion aspect it abandoned>." This is the instruction the blender
    would follow.

For EACH blend return:
  - blend_id: copy from input
  - on_course: true if it still serves the cushion, false if it drifted
  - resonance: 0..1, how tightly it serves the cushion (1.0 = dead center,
    0.0 = entirely different problem)
  - drift_reason: "" when on_course; the different-problem description when not
  - redirect: "" when on_course; the one-line get-back-to note when not

Be a calm supervisor, not a gate. Most good blends are on_course. Flag only
real lane departures.

OUTPUT FORMAT: a single JSON object with one array `verdicts`, each entry in
the schema specified in the user message. Output ONLY the JSON.
"""


def _build_drift_payload(
    cushion: CushionGraph | None,
    blends:  list[Blend],
) -> str:
    problem = ""
    question = ""
    if cushion is not None and getattr(cushion, "raw_input", None) is not None:
        problem = cushion.raw_input.problem.content[:800]
        question = cushion.raw_input.question.content[:400]

    blend_blocks = [
        {
            "blend_id":         b.blend_id,
            "thesis":           b.thesis,
            "advances_cushion": b.advances_cushion,
            "emergent_structure": b.emergent_structure,
        }
        for b in blends
    ]
    schema_spec = {
        "verdicts": [{
            "blend_id":     "<copy from input>",
            "on_course":    "<true|false>",
            "resonance":    "<float 0..1>",
            "drift_reason": "<'' if on_course; the different problem it drifted toward if not>",
            "redirect":     "<'' if on_course; the one-line 'get back to: ...' note if not>",
        }],
    }
    payload = {
        "cushion_problem":  problem,
        "cushion_question": question,
        "blend_count":     len(blends),
        "blends":          blend_blocks,
        "output_schema":   schema_spec,
        "instruction": (
            "Judge ONLY directional fidelity to the cushion's QUESTION. Default "
            "to on_course. Flag a blend only if it drifted to a different "
            "question/problem. Output the JSON object only."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_drift_response(raw: str, blends: list[Blend], report: DriftReport) -> None:
    by_id = {b.blend_id for b in blends}
    parsed = _parse_json_safely(raw, default={})
    if not isinstance(parsed, dict):
        report.parser_notes.append({"reason": "top_level_not_dict"})
        return
    raw_verdicts = parsed.get("verdicts", []) or []
    if not isinstance(raw_verdicts, list):
        report.parser_notes.append({"reason": "verdicts_not_list"})
        return

    seen: set[str] = set()
    for entry in raw_verdicts:
        if not isinstance(entry, dict):
            continue
        bid = str(entry.get("blend_id", ""))
        if bid not in by_id:
            report.parser_notes.append({"reason": "unknown_blend_id", "blend_id": bid})
            continue
        try:
            resonance = float(entry.get("resonance", 1.0) or 0.0)
        except (TypeError, ValueError):
            resonance = 1.0
        on_course = bool(entry.get("on_course", True))
        # Belt-and-braces: low resonance forces a drift flag even if the
        # model said on_course (catches a rubber-stamp supervisor).
        if resonance < RESONANCE_DRIFT_FLOOR:
            on_course = False
        verdict = DriftVerdict(
            blend_id=bid,
            on_course=on_course,
            resonance=resonance,
            drift_reason=str(entry.get("drift_reason", "")),
            redirect=str(entry.get("redirect", "")),
        )
        report.verdicts.append(verdict)
        (report.on_course_ids if on_course else report.drifting_ids).append(bid)
        seen.add(bid)

    # Any blend the supervisor skipped → default on_course (stay out of the way).
    for b in blends:
        if b.blend_id not in seen:
            report.parser_notes.append({"reason": "blend_unjudged_defaulted_on_course", "blend_id": b.blend_id})
            report.verdicts.append(DriftVerdict(blend_id=b.blend_id, on_course=True, resonance=1.0))
            report.on_course_ids.append(b.blend_id)


# ---------------------------------------------------------------------------
# LLM-call helper with cost cap
# ---------------------------------------------------------------------------


async def _call_with_budget(
    *,
    client:        LLMClient,
    system_prompt: str,
    user_message:  str,
    model_slug:    str,
    report:        DriftReport,
) -> LLMResponse:
    response: LLMResponse = await client.call(
        system_prompt=system_prompt,
        user_message=user_message,
        domain=DRIFT_DOMAIN,
        concept=DRIFT_CONCEPT,
        model=model_slug,
        max_tokens=MAX_TOKENS_DRIFT,
        temperature=DRIFT_TEMPERATURE,
    )
    cost = _call_cost_usd(model_slug, response)
    report.total_cost_usd += cost
    report.call_log.append({
        "phase":    "drift_check",
        "model":    model_slug,
        "in_tok":   response.input_tokens,
        "out_tok":  response.output_tokens,
        "cost_usd": round(cost, 4),
        "ms":       round(response.latency_ms or 0.0, 1),
        "ok":       response.success,
        "err":      (response.error or "")[:200] if not response.success else "",
    })
    if report.total_cost_usd > report.cost_ceiling_usd:
        report.truncated_by_budget = True
        report.truncation_reason = (
            f"cumulative spend ${report.total_cost_usd:.2f} exceeds ceiling "
            f"${report.cost_ceiling_usd:.2f}"
        )
        raise DriftBudgetExceeded(report.truncation_reason)
    return response


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def check_drift(
    *,
    cushion:          CushionGraph | None,
    blends:           list[Blend],
    client:           LLMClient,
    progress:         DriftProgress | None = None,
    cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
    model:            str = DRIFT_CHECKER_MODEL,
) -> DriftReport:
    """Supervise the blends for drift off the cushion.

    Returns a DriftReport — per-blend verdicts + on_course / drifting id
    lists. Direction only, not quality. Empty input → empty report (no LLM
    call). If the supervisor LLM call fails, all blends default to on_course
    (fail-open: the supervisor must never silently kill the blender's work).
    """
    report = DriftReport(cost_ceiling_usd=cost_ceiling_usd, model=model)
    progress = progress or DriftProgress()

    if not blends:
        progress.emit("empty_input", {})
        return report

    progress.emit("starting", {"blend_count": len(blends), "model": model})
    system = compose_system_prompt(_DRIFT_DOCTRINE, mode="drift_checker")
    payload = _build_drift_payload(cushion, blends)

    try:
        response = await _call_with_budget(
            client=client, system_prompt=system, user_message=payload,
            model_slug=model, report=report,
        )
        if not response.success:
            # Fail-open — never silently kill the blender's output.
            log.warning("drift check call failed (%s); defaulting all on_course", response.error)
            for b in blends:
                report.verdicts.append(DriftVerdict(blend_id=b.blend_id, on_course=True, resonance=1.0))
                report.on_course_ids.append(b.blend_id)
            report.parser_notes.append({"reason": "llm_call_failed_defaulted_on_course", "err": (response.error or "")[:200]})
        else:
            _parse_drift_response(response.content, blends, report)
        progress.emit("drift_complete", {
            "on_course": len(report.on_course_ids),
            "drifting":  len(report.drifting_ids),
            "cost_usd":  round(report.total_cost_usd, 4),
        })
    except DriftBudgetExceeded:
        pass

    progress.emit("complete", {
        "on_course": len(report.on_course_ids),
        "drifting":  len(report.drifting_ids),
        "total_cost_usd": round(report.total_cost_usd, 4),
    })
    return report


# ===========================================================================
# CYCLE-TRAJECTORY DRIFT — the shepherd for the AUTONOMOUS LOOP
# ===========================================================================
# The `check_drift` above supervises one batch of BLENDS for directional
# fidelity (collision pipeline). This second supervisor watches the autonomous
# LOOP itself: holding the cushion as the north star, it reads what each CYCLE
# added against everything prior and calls the trajectory — converging, sliding
# toward a different problem, or circling the same ground.
#
# IT IS A SENSOR, NOT A BOSS (Nikhil's shepherd/sheepdog, 2026-06-18):
#   * ADVISORY ONLY. It informs the dispatcher's aim and the governor's halt; it
#     holds NO halt authority. The governor stays the single halt authority.
#   * It GUIDES, never strangles. Default verdict is on_track; it speaks up only
#     on clear circling/drift, and its one move is a refocus nudge — never a stop.
#   * FAIL-OPEN. If the LLM call fails it returns a safe on_track verdict so the
#     loop is never silently killed by a dead supervisor.
# The refocus string is GOAL-AWARE (it names cushion angles) — it is consumed by
# goal-aware organs (dispatcher/governor) and MUST be laundered through the
# chaos gate before it can ever reach a wander agent. Phase 5 only PRODUCES the
# signal; dispatcher consumption (with laundering) lands in the dispatcher phase.

_CYCLE_DRIFT_DOCTRINE = """\
You are THE SHEPHERD of an autonomous research loop. Each CYCLE, parallel agents
wander and bring back findings ("cards"). Your ONLY job is to watch the LOOP's
TRAJECTORY against its north star — the cushion (`cushion_problem` frames it,
`cushion_question` is the precise target; `open_angles` are checkpoint facets not
yet covered).

You judge DIRECTION and MOMENTUM, never quality (a later stage handles "is this
real / clever"). Decide ONE status for the whole cycle:
  - on_track : this cycle added genuinely new ground that moves toward covering
               the cushion's open angles.
  - circling : this cycle largely re-tread ground already covered in prior
               cycles — little new; the loop is spinning.
  - drifting : the new ground is sliding toward a DIFFERENT problem than the
               cushion's question.

Also report:
  - momentum : 0..1, the fraction of this cycle that is genuinely NEW ground
               (low momentum = circling).
  - repetition : short list of themes being re-tread (only if circling).
  - drift_reason : the different problem it is sliding toward (only if drifting).
  - refocus : ONE line — the shepherd's nudge back toward an OPEN angle or the
              cushion. Phrase it as a territory/topic to steer toward. Empty when
              on_track.
  - rationale : one or two sentences.

Be a calm shepherd, not a gate. Default to on_track. You do NOT stop the loop —
you only guide it. Output ONLY a single JSON object in the schema given.
"""


@dataclass
class CycleDriftVerdict:
    """The shepherd's read on the LOOP after one cycle — direction + momentum,
    never quality. Advisory: informs the dispatcher's aim and the governor's
    halt, holds NO halt authority itself."""
    cycle:        int   = 0
    status:       str   = "on_track"   # on_track | circling | drifting
    momentum:     float = 1.0          # 0..1 — fraction of genuinely-new ground
    repetition:   list  = field(default_factory=list)
    drift_reason: str   = ""
    refocus:      str   = ""           # one-line nudge (goal-aware; launder before wander)
    rationale:    str   = ""
    model:        str   = ""
    cost_usd:     float = 0.0
    ok:           bool  = True         # False ⇒ LLM call failed, verdict is a safe default

    def to_dict(self) -> dict:
        return asdict(self)


def _build_cycle_drift_payload(
    *, cushion_problem: str, cushion_question: str, cycle: int,
    new_card_bridges: list[str], prior_card_bridges: list[str], open_angles: list[str],
) -> str:
    payload = {
        "cushion_problem":  (cushion_problem or "")[:800],
        "cushion_question": (cushion_question or "")[:600],
        "cycle":            int(cycle),
        "open_angles":      [str(a)[:200] for a in (open_angles or [])],
        "this_cycle_new_findings":  [str(b)[:300] for b in (new_card_bridges or [])][:40],
        "prior_cycles_findings":    [str(b)[:200] for b in (prior_card_bridges or [])][:60],
        "output_schema": {
            "status":       "<on_track|circling|drifting>",
            "momentum":     "<float 0..1, fraction of genuinely-new ground this cycle>",
            "repetition":   ["<theme re-tread>", "..."],
            "drift_reason": "<'' unless drifting: the different problem it slides toward>",
            "refocus":      "<'' if on_track; else one-line nudge toward an open angle/cushion>",
            "rationale":    "<1-2 sentences>",
        },
        "instruction": (
            "Judge the LOOP's trajectory only. Default on_track. Flag circling "
            "when this cycle re-tread prior ground, drifting when it slides off "
            "the cushion's question. You guide, you do not stop. Output JSON only."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def check_cycle_drift(
    *,
    cushion_problem:  str,
    cushion_question: str,
    cycle:            int,
    new_card_bridges: list[str],
    prior_card_bridges: list[str],
    open_angles:      list[str],
    client:           LLMClient,
    model:            str = DRIFT_CHECKER_MODEL,
) -> CycleDriftVerdict:
    """Shepherd the autonomous LOOP after one cycle. One Sonnet call. ADVISORY +
    FAIL-OPEN: any failure (empty input, LLM error, parse error) returns a safe
    on_track verdict (ok=False on real failure) so the loop is never killed by a
    dead supervisor. Holds no halt authority."""
    verdict = CycleDriftVerdict(cycle=int(cycle), model=model)
    if not new_card_bridges:
        verdict.rationale = "no new findings this cycle — nothing to judge"
        return verdict
    try:
        system = compose_system_prompt(_CYCLE_DRIFT_DOCTRINE, mode="drift_checker")
        payload = _build_cycle_drift_payload(
            cushion_problem=cushion_problem, cushion_question=cushion_question, cycle=cycle,
            new_card_bridges=new_card_bridges, prior_card_bridges=prior_card_bridges,
            open_angles=open_angles)
        response = await client.call(
            system_prompt=system, user_message=payload,
            domain=DRIFT_DOMAIN, concept=DRIFT_CONCEPT, model=model,
            max_tokens=MAX_TOKENS_DRIFT, temperature=DRIFT_TEMPERATURE)
        verdict.cost_usd = _call_cost_usd(model, response)
        if not response.success:
            verdict.ok = False
            verdict.rationale = f"shepherd call failed ({(response.error or '')[:120]}); defaulted on_track"
            return verdict
        parsed = _parse_json_safely(response.content, default={})
        if not isinstance(parsed, dict):
            verdict.ok = False
            verdict.rationale = "shepherd returned non-JSON; defaulted on_track"
            return verdict
        status = str(parsed.get("status", "on_track")).strip().lower()
        verdict.status = status if status in ("on_track", "circling", "drifting") else "on_track"
        try:
            verdict.momentum = max(0.0, min(1.0, float(parsed.get("momentum", 1.0))))
        except (TypeError, ValueError):
            verdict.momentum = 1.0
        rep = parsed.get("repetition", []) or []
        verdict.repetition = [str(x)[:160] for x in rep][:8] if isinstance(rep, list) else []
        verdict.drift_reason = str(parsed.get("drift_reason", ""))[:400]
        verdict.refocus = str(parsed.get("refocus", ""))[:400]
        verdict.rationale = str(parsed.get("rationale", ""))[:400]
        return verdict
    except Exception as e:  # never raise into the loop
        verdict.ok = False
        verdict.rationale = f"shepherd raised ({type(e).__name__}: {e}); defaulted on_track"
        return verdict


__all__ = (
    "DRIFT_CHECKER_MODEL",
    "MAX_TOKENS_DRIFT",
    "RESONANCE_DRIFT_FLOOR",
    "DEFAULT_COST_CEILING_USD",
    "DriftBudgetExceeded",
    "DriftVerdict",
    "DriftReport",
    "DriftProgress",
    "check_drift",
    "CycleDriftVerdict",
    "check_cycle_drift",
)
