"""
lead_translator.py — the CHAOS-LAW organ: convert an OPEN GAP (a question the loop
must eventually answer) into a GOAL-FREE STRUCTURAL LEAD the wander can legally chase.

WHY THIS IS THE MOST LAW-SENSITIVE ORGAN:
  The autonomous loop knows, in the JUDGE layer, which checkpoint angle is still open.
  But the wander must NEVER see the question — chaos law. So before any re-dispatch,
  the open gap is translated into anchor/background TERRITORY that points at where an
  answer would structurally live, WITHOUT naming the question, its intent, or any goal.
  The wanderer wanders that territory; the judge (coverage scorer) checks the result
  against the checkpoint. The question never crosses into the anchor.

SAFE BY CONSTRUCTION — THE LEAK TEST IS A HARD GATE:
  Every produced lead passes `leak_check` before it is allowed out. A lead that
  contains a question mark, an interrogative stem, a goal-imperative, or verbatim
  overlap with the gap is REJECTED and re-translated. If no clean lead can be made,
  translate_gap returns clean=False and the caller MUST NOT dispatch — fail-closed.

Standalone (mirrors governor/coverage OpenRouter path). Used by the dispatcher;
unwired in production until the live loop is built.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import httpx

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TRANSLATOR_MODEL = "deepseek/deepseek-v4-pro"

_TRANSLATE_SYSTEM = """You convert an OPEN GAP (a question the research must eventually
answer) into a GOAL-FREE STRUCTURAL LEAD for a wandering explorer.

THE WANDERER MUST NEVER SEE THE QUESTION. Your lead is anchor/background territory
only. It names the STRUCTURAL NEIGHBOURHOOD where an answer would live — the family of
systems, mechanisms, and domains that share the gap's underlying structure — expressed
as memory/territory to wander, NOT as a goal to achieve.

HARD RULES (a violation poisons the whole architecture):
  - NO question marks. NO interrogative phrasing ("how to", "how many", "what governs",
    "when to", "whether to").
  - NO goal or telos. Do NOT say what to find, determine, decide, or answer.
  - NO restatement of the gap. Do not echo its words or its intent.
  - Name ONLY the structural territory: the family of systems/mechanisms that share the
    gap's deep structure, stated as neutral declarative background.

Output ONLY the lead — 1 to 3 declarative sentences, territory-only. No preamble, no
quotes, no question."""

# --- the hard chaos-law gate (mechanical, deterministic) ---
_INTERROGATIVE = re.compile(
    r"\b(how\s+(to|many|much|should|do|does|can)|what\s+(governs|is|are|to|should)"
    r"|when\s+(to|should)|whether\s+to|why\s+(do|does|is))\b", re.I)
_GOAL_VERB = re.compile(
    r"\b(find|determine|answer|decide|figure\s+out|solve|identify|compute|choose|"
    r"discover|work\s+out)\s+(how|what|whether|when|the|a|an|out)\b", re.I)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def leak_check(lead: str, gap_question: str) -> tuple[bool, list[str]]:
    """Hard chaos-law gate. Returns (clean, reasons). Clean ⇔ no reasons."""
    reasons: list[str] = []
    if "?" in lead:
        reasons.append("contains a question mark")
    if _INTERROGATIVE.search(lead):
        reasons.append("interrogative stem present (question leaked)")
    if _GOAL_VERB.search(lead):
        reasons.append("goal-imperative present (telos leaked)")
    lg, qg = _words(lead), _words(gap_question)
    q5 = {tuple(qg[i:i + 5]) for i in range(len(qg) - 4)}
    if any(tuple(lg[i:i + 5]) in q5 for i in range(len(lg) - 4)):
        reasons.append("verbatim 5-gram overlap with the question")
    return (not reasons, reasons)


@dataclass
class LeadResult:
    lead: str = ""
    clean: bool = False
    reasons: list[str] = field(default_factory=list)
    attempts: int = 0
    error: str = ""


async def translate_gap(
    gap_question: str,
    *,
    model: str = TRANSLATOR_MODEL,
    client: httpx.AsyncClient | None = None,
    max_attempts: int = 3,
) -> LeadResult:
    """Translate an open gap into a goal-free lead, GATED by leak_check. Fail-closed:
    if no clean lead is produced in max_attempts, returns clean=False (do NOT dispatch)."""
    if not (os.getenv("OPENROUTER_API_KEY") or "").strip():
        return LeadResult(error="OPENROUTER_API_KEY unset")

    own = client is None
    if own:
        client = httpx.AsyncClient()
    headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
               "Content-Type": "application/json"}
    try:
        last = LeadResult()
        for attempt in range(1, max_attempts + 1):
            # On a retry, tell the model exactly which rule it broke (no question echo).
            nudge = "" if attempt == 1 else \
                f"\n\nYour previous attempt LEAKED ({'; '.join(last.reasons)}). " \
                "Rewrite as pure declarative territory — no interrogative, no goal."
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": _TRANSLATE_SYSTEM + nudge},
                    {"role": "user", "content": f"OPEN GAP:\n{gap_question}"},
                ],
                "temperature": 0.3, "max_tokens": 2000, "usage": {"include": True},
            }
            try:
                r = await client.post(_OPENROUTER_URL, headers=headers, json=body, timeout=90.0)
                r.raise_for_status()
                lead = (r.json()["choices"][0]["message"].get("content") or "").strip().strip('"')
            except Exception as e:
                last = LeadResult(attempts=attempt, error=f"{type(e).__name__}: {e}")
                continue
            clean, reasons = leak_check(lead, gap_question)
            last = LeadResult(lead=lead, clean=clean, reasons=reasons, attempts=attempt)
            if clean:
                return last
        return last  # clean=False → caller must NOT dispatch
    finally:
        if own:
            await client.aclose()


__all__ = ["translate_gap", "leak_check", "LeadResult", "TRANSLATOR_MODEL"]
