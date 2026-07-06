"""
Blender — the collision seat.

WHY THIS EXISTS
---------------
The sorter only IDENTIFIES. It reads cards, checks them against the live
web, and bins them: known / invalid / unplaced. That is retrieval. It tells
you what already exists and what doesn't. It does NOT advance the concept.

Real conceptual evolution is BLENDING — two partially-formed pieces collide
and produce a third thing that lives in neither. Only collision writes down
a new thesis. This module is that collision seat.

The blender sits AFTER the first sort. It receives the sorted cards (the
KNOWN as solid building blocks, the UNPLACED as candidate-novel material;
INVALID is discarded — it's dirt) and it does the single most cognitively
demanding job in the pipeline: mindfully pick 2-4 cards whose collision
ADVANCES the cushion, and blend them into a new candidate concept.

BLEND, NOT MERGE
----------------
A MERGE lists or averages two cards ("combine A and B" → a bullet list of
both). A BLEND produces a concept C with EMERGENT structure — something
true in C that is in neither source card. If the blender cannot name what
is emergent, it merged, and the result is discarded. This distinction is
the whole point and is hammered in the doctrine + flagged by the parser.

THE CUSHION IS GRAVITY
----------------------
Every blend is anchored to the cushion — the user's actual problem. The
blender does not blend for its own sake; it blends to advance/solve the
cushion, and it only picks cards that move toward that goal. Like a chemist
building a new molecule with the target compound in front of them.

PROVENANCE
----------
Every blend documents its own reasoning so the human downstream sees the
WHY, not just the WHAT: which cards were chosen, what sparked the collision,
the motive toward the cushion, the productive tension between the cards, the
new thesis, its mechanism, and the emergent structure. Nothing is a black
box — the person reading it is the judge.

MODEL
-----
Opus 4.8 by default (the heaviest seat; this is where the strongest model
earns its cost). Opus rejects `temperature` and uses output_config.effort —
both handled by the LLM client. Budget-capped exactly like the sorter.

ISOLATION
---------
Imports dossier card type + cushion + CardSnapshot (reused from the sorter
for provenance) + LLMClient + pricing + json helpers. Composes its own
system prompt at the call site (identity source-proof satisfied without an
exempt entry). Makes NO web calls and NO bin decisions — it only blends.
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
from src.wandering.articulate import ArticulatedCard
from src.wandering.cushion import CushionGraph
from src.wandering.master_sorter import CardSnapshot
from src.wandering.master_synthesizer import _parse_json_safely

log = logging.getLogger("constellax.wandering.blender")


# ---------------------------------------------------------------------------
# Model + tuning
# ---------------------------------------------------------------------------

#: The collision seat. Opus 4.8 — the most demanding job gets the strongest
#: model. Overridable per call.
BLENDER_MODEL = "anthropic/claude-opus-4-8"

BLENDER_DOMAIN  = "master_blender"
BLENDER_CONCEPT = "blend"

#: Generous output cap — blends carry multi-field reasoning (thesis,
#: mechanism, emergent structure, rationale) for several blends in one
#: JSON object, and Opus at non-zero effort spends some budget on thinking.
MAX_TOKENS_BLEND = 8192

#: Opus 4.8 uses output_config.effort. Blending is deep creative work, not
#: a recognition task — "medium" buys real deliberation while leaving budget
#: for the visible JSON. (Dropped for models that don't use output_config.)
BLENDER_EFFORT = "medium"

#: Creative temperature — applied for models that accept it (e.g. Sonnet if
#: the seat is ever swapped); dropped automatically for Opus 4.8 / Fable 5.
BLEND_TEMPERATURE = 0.8

#: How many blends to aim for. The blender is told to be mindful — a handful
#: of real collisions beats one-merge-per-card. This is guidance in the
#: prompt, not a hard parser cap.
TARGET_BLEND_COUNT = 4

DEFAULT_COST_CEILING_USD = 8.00


class BlendBudgetExceeded(Exception):
    """Raised when cumulative spend would exceed the ceiling. The caller
    surfaces whatever partial batch was assembled."""


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
class SelectionRationale:
    """The blender's documented reasoning for ONE blend.

    This is the provenance the human reads to judge whether the collision
    was principled or arbitrary.
    """
    why_these_cards: str = ""   # why these specific cards, not others
    spark:           str = ""   # what gave it the idea — the seed of the collision
    motive:          str = ""   # what it advances toward the cushion
    tension:         str = ""   # the productive tension between the cards (the anti-merge guard)
    discovery_path:  str = ""   # the short reasoning CHAIN from the cards to THIS thesis —
                                # the genealogy of the discovery, written so a reader can
                                # reverse-engineer what led to it ("A's X clashed with B's Y;
                                # resolving that needed Z; Z is the thesis")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Blend:
    """One candidate concept produced by colliding 2-4 cards.

    `emergent_structure` is the blend-not-merge proof: what is true in this
    concept that is in NEITHER source card. An empty value is the parser's
    signal that the blender may have merged rather than blended — it is kept
    (the human judges) but flagged in `BlendBatch.parser_notes`.
    """
    blend_id:           str
    source_card_ids:    list[str]
    source_cards:       list[CardSnapshot] = field(default_factory=list)
    selection:          SelectionRationale = field(default_factory=SelectionRationale)
    thesis:             str   = ""   # the NEW concept, stated as a claim
    mechanism:          str   = ""   # how it works / the structure that makes it run
    emergent_structure: str   = ""   # what's true in the blend that's in neither input
    advances_cushion:   str   = ""   # how it advances / solves the cushion
    confidence:         float = 0.0

    def to_dict(self) -> dict:
        return {
            "blend_id":           self.blend_id,
            "source_card_ids":    list(self.source_card_ids),
            "source_cards":       [c.to_dict() for c in self.source_cards],
            "selection":          self.selection.to_dict(),
            "thesis":             self.thesis,
            "mechanism":          self.mechanism,
            "emergent_structure": self.emergent_structure,
            "advances_cushion":   self.advances_cushion,
            "confidence":         self.confidence,
        }


@dataclass
class BlendBatch:
    """Container the blender returns: the blends + cost/audit telemetry."""
    blends:              list[Blend] = field(default_factory=list)
    total_cost_usd:      float       = 0.0
    cost_ceiling_usd:    float       = DEFAULT_COST_CEILING_USD
    truncated_by_budget: bool        = False
    truncation_reason:   str         = ""
    #: Transparency surface — blends the parser flagged (empty emergent
    #: structure = possible merge; unknown card ids dropped; out-of-range
    #: card counts). The human reads this to gauge blend quality.
    parser_notes:        list[dict]  = field(default_factory=list)
    call_log:            list[dict]  = field(default_factory=list)
    input_card_count:    int         = 0
    model:               str         = ""

    def to_dict(self) -> dict:
        return {
            "blends":              [b.to_dict() for b in self.blends],
            "total_cost_usd":      round(self.total_cost_usd, 4),
            "cost_ceiling_usd":    self.cost_ceiling_usd,
            "truncated_by_budget": self.truncated_by_budget,
            "truncation_reason":   self.truncation_reason,
            "parser_notes":        list(self.parser_notes),
            "call_log":            list(self.call_log),
            "input_card_count":    self.input_card_count,
            "model":               self.model,
        }


@dataclass
class BlendProgress:
    """Live progress reference for an in-flight blend pass."""
    on_event: Callable[[str, dict], None] | None = None
    events:   list[dict] = field(default_factory=list)

    def emit(self, name: str, payload: dict | None = None) -> None:
        payload = payload or {}
        entry = {"name": name, "ts": time.time(), **payload}
        self.events.append(entry)
        log.info("[blend] %s %s", name, payload)
        if self.on_event is not None:
            try:
                self.on_event(name, payload)
            except Exception as e:  # pragma: no cover — UX hook must not crash
                log.warning("blend progress on_event raised (ignored): %s", e)


# ---------------------------------------------------------------------------
# Doctrine
# ---------------------------------------------------------------------------


_BLEND_DOCTRINE = """\
You are THE BLENDER — the collision seat, the most demanding job in this
pipeline. You do not summarize. You do not list. You do not retrieve. You
COLLIDE partially-formed ideas into new ones.

THE CUSHION IS GRAVITY — AND THE QUESTION IS THE BULLSEYE. The payload carries
TWO things: the PROBLEM (broad context — what the user is building) and the
QUESTION (the SHARP target — the specific thing THIS run must answer). They are
not the same. The problem orients you; the QUESTION is what every blend must
actually answer. A blend that illuminates the problem but does not answer the
question has DRIFTED — it is solving an adjacent puzzle, not this one. Before
you commit a blend, test it against the QUESTION, not just the problem: does
this collision move toward answering IT? If not, pick a different collision.

Staying on the question is NOT a license to play safe. On-target does not mean
near one card or one domain — the sharpest answers still come from colliding
distant cards across domains. Aim the collision AT the question; never dampen
the collision to stay close to it. On-target AND emergent — both, always.

BLEND, NOT MERGE — this is the whole job, read it twice:
  - A MERGE takes card A and card B and lists or averages them: "combine the
    Markov-chain idea with the jazz idea." That is a bullet list wearing a
    trench coat. It is FORBIDDEN.
  - A BLEND produces a THIRD concept, C, that lives in NEITHER A nor B. C has
    EMERGENT STRUCTURE — something true in C that is in neither input. The
    inputs are scaffolding; the blend is the building.
  - Test every blend: name the one thing that is true in your blend that is
    in NEITHER source card. If you cannot name it, you MERGED. Throw it out
    and try a different collision. Report `emergent_structure` for every
    blend — it is the proof you blended.

MINDFUL SELECTION — collisions need friction:
  - Pick cards that are in PRODUCTIVE TENSION — they pull in different
    directions, sit in different domains, or make claims that strain against
    each other. Tension is what produces emergent structure.
  - Blending two SIMILAR cards yields mush — it collapses into a merge. Avoid
    it. Two cards that already agree have nothing to teach each other.
  - You may collide 2, 3, or 4 cards per blend. Be deliberate about how many
    and which — name why in the rationale.
  - The KNOWN cards are SOLID building material (established, web-verified
    prior work). The UNPLACED cards are CANDIDATE-NOVEL material. The most
    powerful blends often collide a solid known structure WITH a novel
    unplaced one, or two cards whose tension is sharpest. Use the bins.

QUANTITY: aim for about %(target)d strong blends. Quality over coverage —
three real collisions beat ten merges. Do not blend every card. Leave a card
unblended if nothing it touches produces emergent structure.

HONESTY: build ONLY from the provided cards. Do not invent external facts,
papers, or data. If a blend's emergent claim needs a fact you don't have,
say so in the rationale rather than fabricating it. The next stages
(drift-check, web verification) will test your blends — do not hand them
something hollow.

For EACH blend, document your reasoning fully — this provenance exists so a
reader can REVERSE-ENGINEER how the discovery was made, so be explicit:
  - why_these_cards: why these specific cards and not others
  - spark: what gave you the idea — the seed of the collision
  - motive: what this blend advances toward the cushion
  - tension: the productive tension between the cards (your anti-merge proof)
  - discovery_path: the SHORT reasoning CHAIN that led from these cards to this
    thesis — the genealogy of the discovery. Write the actual steps so the
    path is reconstructable, e.g. "card A asserts X; card B asserts Y; X and
    Y cannot both hold unless Z mediates them; pursuing Z forced the question
    of W; the thesis is the answer to W." Not a restatement of the thesis —
    the PATH to it. This is what gets mapped to trace how the discovery formed.
  - thesis: the new concept stated as a sharp claim
  - mechanism: how it works — the structure that makes the concept RUN
  - emergent_structure: the one thing true in the blend that is in NEITHER
    source card
  - advances_cushion: how it advances or helps solve the cushion problem
  - confidence: your honest 0..1 in the blend's coherence (not its novelty —
    novelty is verified later)

OUTPUT FORMAT: a single JSON object with one array `blends`, each entry in
the schema specified in the user message. Output ONLY the JSON. No prose
around it.
"""


def _build_blend_payload(
    cushion: CushionGraph | None,
    cards:   list[ArticulatedCard],
    bins_by_id: dict[str, str] | None,
) -> str:
    """User-message payload: the cushion problem + every card with its bin."""
    problem = ""
    question = ""
    if cushion is not None and getattr(cushion, "raw_input", None) is not None:
        problem = cushion.raw_input.problem.content[:800]
        _q = getattr(cushion.raw_input, "question", None)
        if _q is not None and getattr(_q, "content", None):
            question = _q.content[:800]

    bins_by_id = bins_by_id or {}
    card_blocks = []
    for c in cards:
        card_blocks.append({
            "report_id":    c.report_id,
            "bin":          bins_by_id.get(c.report_id, "unsorted"),
            "spark":        c.spark,
            "source_shape": c.source_shape,
            "bridge":       c.bridge,
            "use":          c.use,
            "limit":        c.limit,
        })

    schema_spec = {
        "blends": [{
            "source_card_ids":    ["<report_id>", "<report_id>", "<2-4 ids>"],
            "why_these_cards":    "<why these specific cards>",
            "spark":              "<what seeded the collision>",
            "motive":             "<what it advances toward the cushion>",
            "tension":            "<the productive tension between the cards>",
            "discovery_path":     "<REQUIRED: the short reasoning chain from the cards to this thesis — the steps, not a restatement>",
            "thesis":             "<the NEW concept as a sharp claim>",
            "mechanism":          "<how it works — the structure that makes it run>",
            "emergent_structure": "<REQUIRED: the one thing true in the blend that is in NEITHER source card>",
            "advances_cushion":   "<how it advances/solves the cushion>",
            "confidence":         "<float 0..1 in coherence>",
        }],
    }

    payload = {
        "cushion_problem":  problem,
        "cushion_question": question,
        "card_count":       len(cards),
        "cards":           card_blocks,
        "output_schema":   schema_spec,
        "instruction": (
            "Collide cards into new concepts that ANSWER THE CUSHION QUESTION "
            "(not merely the surrounding problem). BLEND, do not merge — every "
            "blend MUST name its emergent_structure (what is true in the blend "
            "that is in neither source card). Pick cards in productive tension "
            "across domains. Output the JSON object only."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_blend_response(
    raw:    str,
    cards:  list[ArticulatedCard],
    batch:  BlendBatch,
) -> None:
    """Parse the LLM JSON into Blend objects.

    Invariants:
      - source_card_ids referencing unknown cards are dropped from the blend
        (logged); a blend left with < 2 valid cards is dropped entirely.
      - empty emergent_structure → kept but flagged (possible merge).
      - out-of-range card counts (not 2-4) → kept but flagged.
    """
    card_by_id = {c.report_id: c for c in cards}
    parsed = _parse_json_safely(raw, default={})
    if not isinstance(parsed, dict):
        batch.parser_notes.append({"reason": "top_level_not_dict", "raw_type": type(parsed).__name__})
        return

    raw_blends = parsed.get("blends", []) or []
    if not isinstance(raw_blends, list):
        batch.parser_notes.append({"reason": "blends_not_list"})
        return

    for i, entry in enumerate(raw_blends):
        if not isinstance(entry, dict):
            batch.parser_notes.append({"reason": "blend_entry_not_dict", "index": i})
            continue

        raw_ids = entry.get("source_card_ids", []) or []
        if not isinstance(raw_ids, list):
            raw_ids = []
        valid_ids: list[str] = []
        for rid in raw_ids:
            rid = str(rid)
            if rid in card_by_id:
                valid_ids.append(rid)
            else:
                batch.parser_notes.append({"reason": "unknown_card_id", "blend_index": i, "report_id": rid})

        if len(valid_ids) < 2:
            batch.parser_notes.append({"reason": "blend_dropped_too_few_valid_cards", "blend_index": i, "valid": len(valid_ids)})
            continue
        if len(valid_ids) > 4:
            batch.parser_notes.append({"reason": "blend_over_four_cards", "blend_index": i, "count": len(valid_ids)})

        emergent = str(entry.get("emergent_structure", "")).strip()
        if not emergent:
            batch.parser_notes.append({"reason": "empty_emergent_structure_possible_merge", "blend_index": i})
        if not str(entry.get("discovery_path", "")).strip():
            batch.parser_notes.append({"reason": "missing_discovery_path", "blend_index": i})

        blend = Blend(
            blend_id=f"blend-{i+1:02d}",
            source_card_ids=valid_ids,
            source_cards=[CardSnapshot.from_card(card_by_id[r]) for r in valid_ids],
            selection=SelectionRationale(
                why_these_cards=str(entry.get("why_these_cards", "")),
                spark=str(entry.get("spark", "")),
                motive=str(entry.get("motive", "")),
                tension=str(entry.get("tension", "")),
                discovery_path=str(entry.get("discovery_path", "")),
            ),
            thesis=str(entry.get("thesis", "")),
            mechanism=str(entry.get("mechanism", "")),
            emergent_structure=emergent,
            advances_cushion=str(entry.get("advances_cushion", "")),
            confidence=float(entry.get("confidence", 0.0) or 0.0),
        )
        batch.blends.append(blend)


# ---------------------------------------------------------------------------
# LLM-call helper with cost cap
# ---------------------------------------------------------------------------


async def _call_with_budget(
    *,
    client:        LLMClient,
    system_prompt: str,
    user_message:  str,
    model_slug:    str,
    batch:         BlendBatch,
) -> LLMResponse:
    response: LLMResponse = await client.call(
        system_prompt=system_prompt,
        user_message=user_message,
        domain=BLENDER_DOMAIN,
        concept=BLENDER_CONCEPT,
        model=model_slug,
        max_tokens=MAX_TOKENS_BLEND,
        temperature=BLEND_TEMPERATURE,
        effort=BLENDER_EFFORT,
    )
    cost = _call_cost_usd(model_slug, response)
    batch.total_cost_usd += cost
    batch.call_log.append({
        "phase":    "blend",
        "model":    model_slug,
        "in_tok":   response.input_tokens,
        "out_tok":  response.output_tokens,
        "cost_usd": round(cost, 4),
        "ms":       round(response.latency_ms or 0.0, 1),
        "ok":       response.success,
        "err":      (response.error or "")[:200] if not response.success else "",
    })
    if batch.total_cost_usd > batch.cost_ceiling_usd:
        batch.truncated_by_budget = True
        batch.truncation_reason = (
            f"cumulative spend ${batch.total_cost_usd:.2f} exceeds ceiling "
            f"${batch.cost_ceiling_usd:.2f} after blend call"
        )
        log.warning("blend budget exceeded: %s", batch.truncation_reason)
        raise BlendBudgetExceeded(batch.truncation_reason)
    return response


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def blend_cards(
    *,
    cushion:          CushionGraph | None,
    cards:            list[ArticulatedCard],
    bins_by_id:       dict[str, str] | None = None,
    client:           LLMClient,
    progress:         BlendProgress | None = None,
    cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
    model:            str = BLENDER_MODEL,
) -> BlendBatch:
    """Collide the cards into candidate blended concepts.

    `bins_by_id` maps report_id → "known" | "unplaced" so the blender knows
    which cards are solid prior work vs candidate-novel material. INVALID
    cards should be excluded by the caller before this call — they're dirt.

    Single-pass, single-seat (Opus 4.8). Empty input → empty batch (no LLM
    call). Budget-capped like the sorter.
    """
    batch = BlendBatch(cost_ceiling_usd=cost_ceiling_usd, model=model)
    progress = progress or BlendProgress()
    batch.input_card_count = len(cards)

    if len(cards) < 2:
        progress.emit("insufficient_cards", {"count": len(cards)})
        return batch

    progress.emit("starting", {"card_count": len(cards), "model": model, "ceiling": cost_ceiling_usd})

    system = compose_system_prompt(
        _BLEND_DOCTRINE % {"target": TARGET_BLEND_COUNT},
        mode="master_blender",
    )
    payload = _build_blend_payload(cushion, cards, bins_by_id)

    try:
        response = await _call_with_budget(
            client=client, system_prompt=system, user_message=payload,
            model_slug=model, batch=batch,
        )
        _parse_blend_response(response.content, cards, batch)
        progress.emit("blend_complete", {
            "blend_count":    len(batch.blends),
            "parser_notes":   len(batch.parser_notes),
            "cost_usd":       round(batch.total_cost_usd, 4),
        })
    except BlendBudgetExceeded:
        pass  # partial batch already populated

    progress.emit("complete", {
        "blends":         len(batch.blends),
        "total_cost_usd": round(batch.total_cost_usd, 4),
        "truncated":      batch.truncated_by_budget,
    })
    return batch


__all__ = (
    "BLENDER_MODEL",
    "MAX_TOKENS_BLEND",
    "BLENDER_EFFORT",
    "TARGET_BLEND_COUNT",
    "DEFAULT_COST_CEILING_USD",
    "BlendBudgetExceeded",
    "SelectionRationale",
    "Blend",
    "BlendBatch",
    "BlendProgress",
    "blend_cards",
)
