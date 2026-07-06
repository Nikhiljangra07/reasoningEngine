"""
Self-critique layer — the per-agent metacognitive check.

At iteration boundaries (after each dig step), each agent runs the six
metacognitive questions against itself. If critique fires red, the agent
either returns to anchor + re-orients, or closes the current dig early
with an "honest abandonment" report (still valuable — Law 6: honest doubt
beats performative confidence).

Sits inside Wuxing's umbrella as the NEW layer specifically for Wandering
Room agents. Wuxing's external supervision (soft, scoring-based, no hard
interrupt) runs at session level; self-critique runs per-step per-agent.

The six questions are LOCKED. They came directly from the user's
description of how she thinks under uncertainty.

ISOLATION: imports cushion + LLM client. No persistence, no runtime.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from src.llm.client import LLMClient, LLMResponse
from src.wandering.cushion import CushionGraph


log = logging.getLogger("constellax.wandering.critique")


# ---------------------------------------------------------------------------
# The six locked questions
# ---------------------------------------------------------------------------

QUESTIONS: dict[str, str] = {
    "Q1": "Am I doing it correctly?",
    "Q2": "Is this the main thing, or am I deflecting from the anchor?",
    "Q3": "Is what I'm finding real, or am I projecting structure?",
    "Q4": "Did I gain anything from the time spent here?",
    "Q5": "Should I continue, return to anchor, or hand off to a sub-agent?",
    "Q6": "What did I learn that the user actually needs?",
}

assert len(QUESTIONS) == 6, "the six self-critique questions are locked"


#: Critique uses Haiku 4.5 — fast, cheap, structured output, runs many
#: times per agent. Sonnet's depth would be wasted here; the questions
#: have prescribed structure.
CRITIQUE_DOMAIN = "psychology"  # Haiku 4.5 in provider_map
CRITIQUE_CONCEPT = "self_critique_check"


# ---------------------------------------------------------------------------
# Verdict types
# ---------------------------------------------------------------------------


class CritiqueVerdict(str, Enum):
    """What the critique layer recommends.

    CONTINUE — the agent is on track; keep digging
    RETURN_TO_ANCHOR — re-orient; the agent has drifted but the session
        can still be productive
    ABANDON_DIG — close this dig early with an honest-abandonment report;
        what was found wasn't useful, move on
    HAND_OFF — spawn a sub-agent on the promising thread; current agent
        continues elsewhere or wraps up
    """

    CONTINUE = "continue"
    RETURN_TO_ANCHOR = "return_to_anchor"
    ABANDON_DIG = "abandon_dig"
    HAND_OFF = "hand_off"


@dataclass
class CritiqueResult:
    """The outcome of one self-critique check.

    `verdict` is the action recommendation.
    `answers` is the per-question answer dict ({"Q1": "...", "Q2": "...", ...}).
    `red_flags` are the questions where critique flagged a problem.
    `summary` is the agent's one-sentence self-assessment.
    """

    verdict: CritiqueVerdict
    answers: dict[str, str] = field(default_factory=dict)
    red_flags: list[str] = field(default_factory=list)
    summary: str = ""

    def is_red(self) -> bool:
        """True if critique recommends anything other than CONTINUE."""
        return self.verdict != CritiqueVerdict.CONTINUE


# ---------------------------------------------------------------------------
# Critique prompt
# ---------------------------------------------------------------------------


_CRITIQUE_SYSTEM_PROMPT = """\
You are the self-critique layer for one Wandering Room agent.

The agent is in the middle of a research session. The user gave a problem,
the agent has been wandering and digging. RIGHT NOW it's at an iteration
boundary — you check whether to continue, return to anchor, abandon, or
hand off.

You will see:
  - The user's anchor (cushion graph)
  - The agent's current position (what domain it's in)
  - What the agent just found in this iteration
  - The agent's cumulative tokens spent so far

Apply the SIX QUESTIONS:

  Q1: Am I doing it correctly?
  Q2: Is this the main thing, or am I deflecting from the anchor?
  Q3: Is what I'm finding real, or am I projecting structure?
  Q4: Did I gain anything from the time spent here?
  Q5: Should I continue, return to anchor, or hand off to a sub-agent?
  Q6: What did I learn that the user actually needs?

Be honest. The agent is allowed to be wrong; the user benefits more from
an honest "I abandoned this dig" report than from a forced positive spin.
Per Constellax design: the insight happens in the USER's head — our job
is to deliver honest material, not impressive material.

# RED FLAGS to watch for

- Pattern-matching novelty over substance ("interesting" but not advancing
  the anchor) → Q2 / Q4 red
- Projecting structure where it isn't ("I see bounded freedom in this
  toaster review" — really?) → Q3 red
- Diminishing returns ("digged 3 iterations, gained nothing new") → Q4 red
- Lost the anchor's structural shape ("I'm now writing about something
  unrelated") → Q2 red

# WHAT IS A VALID FINDING (do NOT red-flag these)

A cross-domain mapping that is PARTIAL but whose limits are explicitly
named ("this maps in these specific ways; does NOT map in these other
specific ways") is a VALID finding, not a red flag. The "what does not
map" field exists precisely to make partial mappings honest. Do not
flag Q3 (projection) just because the mapping is partial — only flag
Q3 when the mapping is INVENTED (the source genuinely has no
structural connection to the anchor, but the agent forces one).

# VERDICTS

- continue          — on track, keep digging
- return_to_anchor  — drift detected; re-orient and continue session
- abandon_dig      — close this dig early, write honest-abandonment report,
                     move on to next direction
- hand_off         — promising lead but not THIS agent's focus; spawn a
                     sub-agent for it and this agent continues elsewhere

# WHEN TO PICK ABANDON_DIG vs RETURN_TO_ANCHOR vs CONTINUE

abandon_dig is the STRONGEST verdict and the rarest. Use it only when
BOTH Q3 (projection — the analogy was invented, not found) AND Q4
(no gain — nothing transferable was learned) are simultaneously true.
If you choose abandon_dig, you MUST list BOTH "Q3" and "Q4" in
red_flags. Q3 alone (projection but you DID gain something usable
elsewhere) → return_to_anchor with a clean handoff note in summary.
Q4 alone (no gain but the source was real and the mapping was honest)
→ return_to_anchor — re-orient and try a different angle. Neither
Q3 nor Q4 → continue.

# OUTPUT FORMAT

Return ONE JSON object:

{
  "answers": {
    "Q1": "<one sentence>",
    "Q2": "<one sentence>",
    "Q3": "<one sentence>",
    "Q4": "<one sentence>",
    "Q5": "<one sentence>",
    "Q6": "<one sentence>"
  },
  "red_flags": ["Q2", "Q4"],
  "verdict": "continue|return_to_anchor|abandon_dig|hand_off",
  "summary": "<one sentence overall self-assessment>"
}

No prose preamble. No code fences. JUST the JSON.
"""


def build_critique_user_message(
    cushion: CushionGraph,
    agent_position: str,
    latest_finding: str,
    cumulative_tokens: int,
    iterations_so_far: int,
) -> str:
    """Render the inputs to the self-critique LLM call."""
    blocks = [
        "# ANCHOR (the user's problem)",
        cushion.to_anchor_prompt(),
        "\n# AGENT'S CURRENT POSITION",
        agent_position or "(unspecified)",
        "\n# WHAT THE AGENT JUST FOUND (this iteration)",
        latest_finding.strip(),
        f"\n# CUMULATIVE TOKENS SPENT: {cumulative_tokens}",
        f"# ITERATIONS COMPLETED THIS DIG: {iterations_so_far}",
        "\n# YOUR TASK",
        "Apply the six self-critique questions. Return JSON per the spec.",
    ]
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def _extract_json_object(text: str) -> str:
    text = _strip_code_fences(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in critique response")
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
                return text[start : i + 1]
    raise ValueError("unterminated JSON in critique response")


def parse_critique_response(response_text: str) -> CritiqueResult:
    """Parse the critique LLM output. Tolerant of minor schema wobbles.

    Falls back to CONTINUE verdict if the response is unparseable —
    safer to continue than to halt the agent on a flaky LLM response;
    the next iteration's critique will catch the issue if it persists.
    """
    try:
        json_text = _extract_json_object(response_text)
        payload = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("critique response unparseable: %s", e)
        return CritiqueResult(
            verdict=CritiqueVerdict.CONTINUE,
            summary="(critique unparseable — defaulting to continue)",
        )

    if not isinstance(payload, dict):
        return CritiqueResult(verdict=CritiqueVerdict.CONTINUE)

    # Verdict — coerce to enum, default to CONTINUE on garbage.
    verdict_raw = str(payload.get("verdict", "continue")).strip().lower()
    try:
        verdict = CritiqueVerdict(verdict_raw)
    except ValueError:
        log.debug("critique verdict %r not a known enum; defaulting to CONTINUE", verdict_raw)
        verdict = CritiqueVerdict.CONTINUE

    # Answers — accept any string values, ignore extras.
    answers_raw = payload.get("answers", {})
    if not isinstance(answers_raw, dict):
        answers_raw = {}
    answers = {
        k: str(v).strip()
        for k, v in answers_raw.items()
        if k in QUESTIONS and v
    }

    # Red flags — list of question keys
    red_flags_raw = payload.get("red_flags", [])
    if not isinstance(red_flags_raw, list):
        red_flags_raw = []
    red_flags = [str(q).strip() for q in red_flags_raw if str(q).strip() in QUESTIONS]

    summary = str(payload.get("summary", "")).strip()

    return CritiqueResult(
        verdict=verdict,
        answers=answers,
        red_flags=red_flags,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Top-level critique function
# ---------------------------------------------------------------------------


async def run_self_critique(
    cushion: CushionGraph,
    agent_position: str,
    latest_finding: str,
    cumulative_tokens: int,
    iterations_so_far: int,
    client: LLMClient,
) -> CritiqueResult:
    """Run the self-critique check at an iteration boundary.

    Called by the agent loop after each dig iteration. Result's verdict
    tells the loop what to do next (continue / return / abandon / hand off).
    """
    user_message = build_critique_user_message(
        cushion=cushion,
        agent_position=agent_position,
        latest_finding=latest_finding,
        cumulative_tokens=cumulative_tokens,
        iterations_so_far=iterations_so_far,
    )
    response: LLMResponse = await client.call(
        system_prompt=_CRITIQUE_SYSTEM_PROMPT,
        user_message=user_message,
        domain=CRITIQUE_DOMAIN,
        concept=CRITIQUE_CONCEPT,
    )
    if not response.success:
        log.warning("critique LLM call failed: %s", response.error)
        # Fail-open: continue the dig. Next iteration's critique will
        # re-check; we don't halt agents on infra hiccups.
        return CritiqueResult(
            verdict=CritiqueVerdict.CONTINUE,
            summary=f"(critique call failed: {response.error})",
        )

    return parse_critique_response(response.content)


# ---------------------------------------------------------------------------
# r9 Fix #2 — Structural abandonment gate
# ---------------------------------------------------------------------------
#
# r7 had NO structural constraints on the abandon verdict — it trusted
# whatever the LLM wrote. This let model personality leak through: Grok
# eagerly abandoned, Haiku/Gemini refused to abandon. r8 removed the
# entire iter-1 abandonment path to neutralize the leak, but that lost
# the legitimate honest-abandonment escape valve r7 had.
#
# r9 restores iter-1 abandonment under THREE structural layers, all of
# which must pass for ABANDON_DIG to actually terminate at iter-1:
#
#   Layer 1 — Q3+Q4 floor:    abandon_dig requires both Q3 (projection)
#                             AND Q4 (no gain) in red_flags. Q4 alone →
#                             return_to_anchor. Q3 alone → continue.
#                             Neither → continue.
#   Layer 2 — Circuit breaker: if the agent already abandoned ≥ 2 of its
#                             first 3 digs, demote the next ABANDON_DIG
#                             to CONTINUE. Catches the Grok-personality
#                             leak before it accumulates.
#   Layer 3 — Confidence cap: iter-1 abandonments cap report confidence
#                             at MEDIUM regardless of layer-match
#                             ratio. Applied in agent.py at the report-
#                             build step (this module only emits the
#                             verdict; the cap is enforced downstream).
#
# Layers 1 and 2 act on the verdict returned to the agent loop. Layer 3
# is communicated via the GateDecision.confidence_cap field which the
# agent loop respects when building the report.

_CIRCUIT_BREAKER_WINDOW = 3   # examine first N digs
_CIRCUIT_BREAKER_LIMIT  = 2   # if >= LIMIT abandonments in WINDOW, demote next


@dataclass
class GateDecision:
    """Output of enforce_abandon_gate.

    `verdict` is the (possibly demoted) verdict the agent loop should
    consume. `original_verdict` records what the LLM emitted before
    the gate ran — needed for forensics. `gate_action` is a short
    machine-readable code describing which layer fired (e.g.
    "passed", "layer1_demote_to_rta", "layer1_demote_to_continue",
    "layer2_circuit_breaker"). `confidence_cap` is the maximum
    confidence the resulting iter-1 report may ship at — None means
    no cap (most cases); "medium" means apply Layer 3.
    """

    verdict: CritiqueVerdict
    original_verdict: CritiqueVerdict
    gate_action: str
    confidence_cap: str | None = None  # None | "medium" | "low"


def enforce_abandon_gate(
    result: CritiqueResult,
    iteration_so_far: int,
    abandon_history: list[bool] | None = None,
) -> GateDecision:
    """Apply the three-layer structural gate to a critique result.

    Only intervenes when the LLM verdict is ABANDON_DIG AND we are at
    iter-1 (iteration_so_far == 1). All other verdicts pass through
    unchanged (return_to_anchor / continue / hand_off are already
    safe — they do not terminate the dig at iter-1).

    Parameters:
      result            — parsed CritiqueResult from the LLM
      iteration_so_far  — the iteration number critique fired AFTER
                          (1 for the first post-iter-1 check)
      abandon_history   — list of booleans, one per prior dig in this
                          agent's session, True if that dig terminated
                          at iter-1. May be None (treated as empty).

    Returns a GateDecision the agent loop consumes.
    """
    history = abandon_history or []
    original = result.verdict

    # Pass-through for non-abandonment verdicts: nothing to gate.
    if original != CritiqueVerdict.ABANDON_DIG:
        return GateDecision(
            verdict=original,
            original_verdict=original,
            gate_action="passed",
            confidence_cap=None,
        )

    # Pass-through for iter-2+ abandonments: by r9 design iter-2 always
    # runs, so an iter-2 critique can still recommend abandonment but
    # only as advisory (the agent loop already ran the revise step).
    if iteration_so_far >= 2:
        return GateDecision(
            verdict=original,
            original_verdict=original,
            gate_action="passed_iter2",
            confidence_cap=None,
        )

    # Layer 2 — circuit breaker. If the agent already abandoned a
    # majority of its first few digs, downgrade this abandonment to
    # CONTINUE. The check looks at the FIRST N digs in the session
    # (not the last N) so a Grok-personality bias caught early is
    # caught for the rest of the run.
    if len(history) >= _CIRCUIT_BREAKER_WINDOW:
        window = history[:_CIRCUIT_BREAKER_WINDOW]
        if sum(1 for h in window if h) >= _CIRCUIT_BREAKER_LIMIT:
            return GateDecision(
                verdict=CritiqueVerdict.CONTINUE,
                original_verdict=original,
                gate_action="layer2_circuit_breaker",
                confidence_cap=None,
            )

    # Layer 1 — Q3+Q4 floor. abandon_dig requires both flags.
    rf = set(result.red_flags)
    has_q3 = "Q3" in rf
    has_q4 = "Q4" in rf

    if has_q3 and has_q4:
        # All layers passed — true honest abandonment. Layer 3 caps
        # the report's confidence at MEDIUM.
        return GateDecision(
            verdict=CritiqueVerdict.ABANDON_DIG,
            original_verdict=original,
            gate_action="passed",
            confidence_cap="medium",
        )

    if has_q4 and not has_q3:
        # Q4 alone — gained nothing but the source was real. Re-orient.
        return GateDecision(
            verdict=CritiqueVerdict.RETURN_TO_ANCHOR,
            original_verdict=original,
            gate_action="layer1_demote_to_rta",
            confidence_cap=None,
        )

    if has_q3 and not has_q4:
        # Q3 alone — projection but you gained something. Self-correct
        # and keep going (iter-2 will revise).
        return GateDecision(
            verdict=CritiqueVerdict.CONTINUE,
            original_verdict=original,
            gate_action="layer1_demote_to_continue_q3only",
            confidence_cap=None,
        )

    # Neither Q3 nor Q4 — the LLM said abandon_dig without the
    # structural signature. Demote to CONTINUE.
    return GateDecision(
        verdict=CritiqueVerdict.CONTINUE,
        original_verdict=original,
        gate_action="layer1_demote_to_continue_noflags",
        confidence_cap=None,
    )


__all__ = [
    "QUESTIONS",
    "CRITIQUE_DOMAIN",
    "CRITIQUE_CONCEPT",
    "CritiqueVerdict",
    "CritiqueResult",
    "GateDecision",
    "build_critique_user_message",
    "parse_critique_response",
    "run_self_critique",
    "enforce_abandon_gate",
]
