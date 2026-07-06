"""
Quality ranker — the FINAL alignment pass (the stage-1 closer).

Runs LAST, once the blends are novelty-verified AND all halo blind spots are
collected. It does NOT score generic "goodness" (that rewards fluency, and
launders impressive-sounding nonsense). It ranks each blend by ALIGNMENT to
what this run is actually reaching toward — its ALIGNMENT to the cushion.

  RANK SIGNAL — the CUSHION, and ONLY the cushion (the user's real problem;
              human-given ground truth): how far does this blend advance /
              sharpen / help solve it? This single number sets the order.
  SURFACED    — the BLIND SPOTS each blend resolves, and any NEW gap it opens:
              recorded beside the rank as a MAP for the human, never folded
              into the score.

DISCIPLINE (load-bearing — read before changing anything)
- RANK BY ADVANCEMENT ALONE. The cushion is the one external anchor the machine
  is entitled to measure against. Gap-coverage is NOT a score input and severity
  is NEVER a multiplier — flaws are load-bearing (some are the cost of a part of
  the system the user wants to keep), so which flaws are worth solving is the
  HUMAN's call. The machine surfaces the flaw list; the human decides.
- RANK, never DELETE. Every blend keeps its record; a mis-rank loses nothing.
  The human re-judges the full ordered list on the completed output.
- DON'T FABRICATE SEPARATION. When advancement is within TIE_EPSILON, the blends
  SHARE a rank (tied) — the system refuses to invent confidence it doesn't have;
  the human breaks the tie with the surfaced flaw-map.
- RELEVANCE ranking, NOT a correctness oracle. "On-target" is not "true." The
  human judges correctness/worth — on the COMPLETED output, not inside the run.
- The LLM only ASSESSES (advancement 0..1, gaps-addressed, opens-new-gap). The
  order + the ties are computed in CODE — transparent and auditable, not an
  opaque LLM "rank these for me".
- Advisory output: a rank-by-advancement list with the flaw-map laid beside it.
  No action is taken on it.

ISOLATION
Imports the Blend type + LLMClient + pricing + json helper. Composes its own
system prompt at the call site (identity source-proof satisfied, no exempt
entry). Makes NO web calls and takes NO action. Fail-soft: on any LLM/parse
failure every blend is kept, unranked, in input order (nothing lost).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field

from src.identity import compose_system_prompt
from src.llm.client import LLMClient
from src.llm.provider_map import get_pricing
from src.wandering.blender import Blend
from src.wandering.master_synthesizer import _parse_json_safely

log = logging.getLogger("constellax.wandering.quality_ranker")


# ---------------------------------------------------------------------------
# Model + tuning
# ---------------------------------------------------------------------------

#: The alignment judge. Sonnet by default — a judgment task, wants the strong
#: eye. Overridable per call (control_room.RANKER_MODEL).
RANKER_MODEL = "anthropic/claude-sonnet-4-6"

RANKER_DOMAIN = "quality_ranker"
MAX_TOKENS_RANK = 4096
RANK_TEMPERATURE = 0.2

#: The rank is advancement-toward-the-cushion ALONE (see DISCIPLINE). Gap-
#: coverage and new-gap are surfaced beside the rank, never scored. Blends whose
#: advancement is within this band share a rank — the human breaks the tie with
#: the flaw-map; the system does not fabricate separation it can't justify.
TIE_EPSILON = 0.05


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    in_p, out_p = get_pricing(model)
    return (in_tok or 0) / 1_000_000 * in_p + (out_tok or 0) / 1_000_000 * out_p


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BlendRank:
    """One blend's alignment assessment + its computed rank."""
    blend_id:             str
    rank:                 int   = 0     # 1 = best; SHARED on a tie (computed in code)
    tied:                 bool  = False # shares its rank with >=1 other blend
    score:                float = 0.0   # == advancement (the sole rank key; kept for transparency)
    advancement:          float = 0.0   # 0..1 toward the cushion (LLM) — THE rank signal
    advancement_note:     str   = ""    # why it advances the cushion (LLM)
    blind_spots_addressed: list[str] = field(default_factory=list)  # gap refs resolved
    opens_new_gap:        str   = ""    # a gap nobody named, if any (LLM)
    novelty_bin:          str   = ""    # copied from the novelty verification
    rationale:            str   = ""    # the ranker's one-line reasoning

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QualityRanking:
    """The final ranked, gap-mapped list. RANKED, nothing deleted."""
    ranked:        list[BlendRank] = field(default_factory=list)
    model:         str             = ""
    total_cost_usd: float          = 0.0
    ok:            bool            = True
    note:          str             = ""
    parser_notes:  list[dict]      = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ranked":         [r.to_dict() for r in self.ranked],
            "model":          self.model,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "ok":             self.ok,
            "note":           self.note,
            "parser_notes":   list(self.parser_notes),
        }


# ---------------------------------------------------------------------------
# Doctrine
# ---------------------------------------------------------------------------


_RANK_DOCTRINE = """\
You are the ALIGNMENT JUDGE — the final pass over a run's candidate blends.

You do NOT rate generic "quality" or "how impressive it sounds" — polish and
jargon are not value, and rewarding them is the cardinal sin of this role. You
assess one thing that sets the order: how far each blend moves toward what this
run is actually trying to reach.

THE CUSHION — the user's problem, vision, hunches, and (the part that matters
most here) the QUESTION: the explicit checkpoint this run must answer. The
QUESTION is your FIXED target — measure each blend's distance to IT, not to a
fuzzy sense of the whole. If the cushion states no QUESTION, fall back to the
PROBLEM.
{cushion}

THE BLIND SPOTS the run surfaced (a MAP for the human — NOT a scoring input):
{blind_spots}

These are the system's own hypotheses about what's missing. You will note which
gap(s) each blend resolves and whether it opens a NEW one, so the HUMAN can
filter by the flaws THEY care about — but you do NOT rank by how many gaps a
blend closes, and you NEVER weight a gap by its severity. Flaws are load-bearing:
some are the cost of a part of the system the user wants to keep, so which flaws
are worth solving is the user's call, not yours. Rank ONLY by advancement toward
the QUESTION (the cushion's checkpoint).

For EACH blend, assess honestly:
  - advancement (0..1): how far does it ADVANCE, SHARPEN, or help ANSWER the
    QUESTION (the cushion's checkpoint)? 1.0 = a real move toward answering it;
    0.0 = on a different question. Judge the MOVE, not the prose. A plain idea
    that truly advances the QUESTION beats a dazzling one that wanders off it.
    THIS is the only number that orders the list.
  - blind_spots_addressed: list the gap id(s) (e.g. "G2") this blend genuinely
    RESOLVES — not name-drops, resolves. Surfaced for the human. Empty if none.
  - opens_new_gap: if the blend reveals a gap NONE of the listed blind spots
    named, state it in one phrase. Surfaced for the human as new territory. Else "".
  - advancement_note: one clause on what it moves toward in answering the QUESTION.
  - rationale: one honest line on your assessment.

Be a calm, honest judge. Do NOT inflate advancement for a fluent blend. Do NOT
deflate a blunt-but-on-target one. If two blends advance the QUESTION about
equally, score them about equally — do NOT manufacture a gap between them; the
system will tie them and let the human decide. You rank nothing and delete
nothing — you only assess; the system computes order and keeps every record.

OUTPUT FORMAT — a single JSON object, nothing around it:
{{"rankings": [
  {{"blend_id": "<copy>", "advancement": <0..1>,
    "advancement_note": "<one clause>",
    "blind_spots_addressed": ["G1", "G3"],
    "opens_new_gap": "<phrase or ''>",
    "rationale": "<one line>"}}
]}}
"""


def _format_blind_spots(blind_spots: list) -> str:
    if not blind_spots:
        return "(none collected)"
    lines = []
    for i, b in enumerate(blind_spots, 1):
        text = getattr(b, "blind_spot", None) or (b.get("blind_spot", "") if isinstance(b, dict) else str(b))
        layer = getattr(b, "layer", None) or (b.get("layer", "") if isinstance(b, dict) else "")
        lines.append(f"[G{i}] ({layer}) {str(text)[:200]}")
    return "\n".join(lines)


def _build_payload(blends: list[Blend], novelty_by_id: dict[str, str]) -> str:
    blocks = [
        {
            "blend_id":           b.blend_id,
            "novelty_bin":        novelty_by_id.get(b.blend_id, "unverified"),
            "thesis":             b.thesis,
            "mechanism":          b.mechanism,
            "emergent_structure": b.emergent_structure,
            "advances_cushion":   b.advances_cushion,
        }
        for b in blends
    ]
    return json.dumps({
        "blend_count": len(blends),
        "blends": blocks,
        "instruction": (
            "Assess each blend's advancement toward the cushion (primary) and "
            "which blind spots it resolves (secondary). Judge the MOVE, not the "
            "prose. Output the JSON object only."
        ),
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Parser + scoring (scoring is in CODE — transparent, not LLM-decided)
# ---------------------------------------------------------------------------


def _rank_with_ties(items: list[BlendRank]) -> list[BlendRank]:
    """Order by advancement desc and assign competition ranks, sharing a rank
    across any run of blends within TIE_EPSILON of that run's LEADER. Comparing
    to the group leader (not the immediate neighbour) avoids transitive chaining
    — a slow drift of tiny steps can't quietly fuse the top and bottom into one
    tie. Stable sort keeps assessed blends ahead of unassessed at equal score."""
    ordered = sorted(items, key=lambda r: r.advancement, reverse=True)
    leader = None
    cur_rank = 0
    for i, r in enumerate(ordered):
        if leader is None or (leader - r.advancement) > TIE_EPSILON:
            cur_rank = i + 1          # competition rank: jumps past tied members
            leader = r.advancement
        r.rank = cur_rank
    counts: dict[int, int] = {}
    for r in ordered:
        counts[r.rank] = counts.get(r.rank, 0) + 1
    for r in ordered:
        r.tied = counts[r.rank] > 1
    return ordered


def _parse(raw: str, blends: list[Blend], novelty_by_id: dict[str, str],
           report: QualityRanking) -> None:
    by_id = {b.blend_id: b for b in blends}
    parsed = _parse_json_safely(raw, default={})
    assessed: dict[str, BlendRank] = {}
    if isinstance(parsed, dict):
        for e in parsed.get("rankings", []) or []:
            if not isinstance(e, dict):
                continue
            bid = str(e.get("blend_id", ""))
            if bid not in by_id:
                report.parser_notes.append({"reason": "unknown_blend_id", "blend_id": bid})
                continue
            try:
                adv = float(e.get("advancement", 0.0) or 0.0)
            except (TypeError, ValueError):
                adv = 0.0
            adv = max(0.0, min(1.0, adv))
            gaps = [str(g) for g in (e.get("blind_spots_addressed", []) or []) if str(g).strip()]
            new_gap = str(e.get("opens_new_gap", "") or "").strip()
            assessed[bid] = BlendRank(
                blend_id=bid,
                advancement=adv,
                score=adv,                      # the sole rank key — surfaced, not composite
                advancement_note=str(e.get("advancement_note", "")),
                blind_spots_addressed=gaps,     # surfaced map for the human, NOT scored
                opens_new_gap=new_gap,          # surfaced map for the human, NOT scored
                novelty_bin=novelty_by_id.get(bid, "unverified"),
                rationale=str(e.get("rationale", "")),
            )

    # Keep EVERY blend — unassessed ones default to advancement 0, never dropped.
    for b in blends:
        if b.blend_id not in assessed:
            report.parser_notes.append({"reason": "blend_unassessed_kept_last", "blend_id": b.blend_id})
            assessed[b.blend_id] = BlendRank(
                blend_id=b.blend_id, novelty_bin=novelty_by_id.get(b.blend_id, "unverified"),
                rationale="(not assessed — kept, ranked last; never deleted)",
            )

    report.ranked = _rank_with_ties(list(assessed.values()))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def rank_blends(
    *,
    cushion: str,
    blends: list[Blend],
    blind_spots: list,
    novelty_by_id: dict[str, str] | None = None,
    client: LLMClient,
    model: str = RANKER_MODEL,
) -> QualityRanking:
    """Final alignment pass: rank blends by advancement toward the cushion —
    the ONLY rank signal. Gaps resolved + new gaps opened are surfaced beside
    each rank as a map for the human, never scored; severity is never weighted.
    Near-equal advancement shares a rank (tied) rather than inventing
    separation. RANKS, never deletes. Empty input -> empty ranking (no LLM
    call). Fail-soft."""
    report = QualityRanking(model=model)
    novelty_by_id = novelty_by_id or {}
    if not blends:
        return report

    system = compose_system_prompt(
        _RANK_DOCTRINE.format(
            cushion=cushion[:1400],
            blind_spots=_format_blind_spots(blind_spots),
        ),
        mode="quality_ranker",
    )
    t0 = time.time()
    resp = await client.call(
        system_prompt=system,
        user_message=_build_payload(blends, novelty_by_id),
        domain=RANKER_DOMAIN, concept="rank",
        model=model, max_tokens=MAX_TOKENS_RANK, temperature=RANK_TEMPERATURE,
    )
    report.total_cost_usd = _cost(model, resp.input_tokens, resp.output_tokens)
    if not resp.success:
        # Fail-soft: keep every blend, unranked in input order. Nothing lost.
        report.ok = False
        report.note = f"rank call failed: {(resp.error or '')[:160]}"
        for i, b in enumerate(blends, 1):
            report.ranked.append(BlendRank(
                blend_id=b.blend_id, rank=i,
                novelty_bin=(novelty_by_id or {}).get(b.blend_id, "unverified"),
                rationale="(ranker call failed — kept in input order; never deleted)"))
        return report

    _parse(resp.content, blends, novelty_by_id, report)
    log.info("[quality_ranker] ranked %d blends, $%.4f, %.0fms",
             len(report.ranked), report.total_cost_usd, (time.time() - t0) * 1000)
    return report


__all__ = (
    "RANKER_MODEL",
    "BlendRank",
    "QualityRanking",
    "rank_blends",
)
