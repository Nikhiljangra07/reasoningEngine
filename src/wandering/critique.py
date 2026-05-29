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

# VERDICTS

- continue          — on track, keep digging
- return_to_anchor  — drift detected; re-orient and continue session
- abandon_dig      — close this dig early, write honest-abandonment report,
                     move on to next direction
- hand_off         — promising lead but not THIS agent's focus; spawn a
                     sub-agent for it and this agent continues elsewhere

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


__all__ = [
    "QUESTIONS",
    "CRITIQUE_DOMAIN",
    "CRITIQUE_CONCEPT",
    "CritiqueVerdict",
    "CritiqueResult",
    "build_critique_user_message",
    "parse_critique_response",
    "run_self_critique",
]
