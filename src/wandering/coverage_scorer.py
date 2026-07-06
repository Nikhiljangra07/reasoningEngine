"""
coverage_scorer.py — ORGAN 2 of the autonomous orchestrator: the Definition-of-Done
coverage scorer that produces D_t for blend-04's triangulated halt.

WHAT IT IS:
  Given the cushion's REQUIRED ANGLES (the judge-checkpoint sub-questions) and the
  FINDINGS accumulated so far, it returns D_t ∈ [0,1] — the fraction of required
  angles that are structurally addressed — plus a per-angle covered/partial/uncovered
  verdict and the list of still-open angles. This is the automated form of the strict
  hand-grading done against the 5 Cushion-3 sub-questions.

FLOW-NOT-JUDGE — THE LOAD-BEARING CONSTRAINT:
  This organ scores COVERAGE (is each required angle ADDRESSED by some finding?),
  NEVER QUALITY (is the answer good?). The orchestrator governs flow; the human
  judges quality. The classifier is explicitly constrained to presence/addressing
  and must mark "covered" regardless of how good a finding is. If this ever drifts
  into grading answer quality, it has become the in-loop judge the whole architecture
  forbids. D_t is a flow signal, not a verdict on the work.

  Reading the QUESTION here is correct and NOT a chaos-law violation: the wander never
  sees this organ — only the halt logic does. The question is the judge checkpoint;
  measuring coverage against it is exactly the judge layer's job.

SELF-CONTAINED: mirrors governor.py's OpenRouter path (httpx, temp 0, retry). No
coupling to the live pipeline. Used by the body scaffold; unwired in production.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
COVERAGE_MODEL = "deepseek/deepseek-v4-pro"   # cheap, validated; temp 0 for stability

_COVERAGE_SYSTEM = """You are a COVERAGE scorer for an autonomous research loop.

You are given a numbered list of REQUIRED ANGLES (the questions the work must
address) and the FINDINGS produced so far. For EACH required angle, decide ONLY
whether the findings STRUCTURALLY ADDRESS it:
  - "covered"   : at least one finding directly proposes a mechanism or answer for this angle.
  - "partial"   : a finding touches the angle but leaves a NAMED part of it open.
  - "uncovered" : no finding addresses this angle at all.

CRITICAL — YOU SCORE COVERAGE, NOT QUALITY.
  Mark "covered" if the angle is addressed AT ALL — regardless of whether the answer
  is good, correct, novel, or elegant. You are NOT judging how well it is answered,
  only whether an answer is PRESENT. Quality is the human's job, never yours. A weak
  but on-target finding is "covered". A brilliant finding about a DIFFERENT angle does
  not cover this one.

STRICT PRESENCE — A MENTION IS NOT COVERAGE.
  "partial" requires a finding that proposes a mechanism PARTLY answering THIS
  SPECIFIC angle. A mere mention of related terminology, or a mechanism aimed at a
  DIFFERENT angle that merely touches this one's keywords, is "uncovered" — NOT
  "partial". (E.g. a finding about bias *magnitude* that says "consumes wave
  capacity" does NOT address "how many agents to commit" — that is uncovered.)
  This organ feeds a STOP decision, so a false "covered" is the costly error:
  WHEN IN DOUBT BETWEEN "partial" AND "uncovered", CHOOSE "uncovered".

Output ONLY JSON, no prose, no code fences:
{"verdicts":[{"angle":1,"coverage":"covered|partial|uncovered","by":"<short finding ref or reason>"}]}
"""

_WEIGHT = {"covered": 1.0, "partial": 0.5, "uncovered": 0.0}


def parse_required_angles(question: str) -> list[str]:
    """Extract numbered sub-questions '(1) … (2) …' from a cushion question.

    Trims a trailing meta-sentence ("What do X, Y reveal about each?") off the last
    angle. Falls back to the whole question as one angle if none are numbered.
    """
    parts = re.split(r"\(\s*(\d+)\s*\)", question)  # [pre, '1', t1, '2', t2, …]
    angles: list[str] = []
    for i in range(1, len(parts) - 1, 2):
        txt = parts[i + 1].strip()
        # drop a trailing "What do/does … ?" meta-sentence on the final angle
        txt = re.split(r"\s+What d(o|oes)\b", txt)[0]
        angles.append(txt.strip().rstrip(";").strip())
    return angles or [question.strip()]


@dataclass
class CoverageResult:
    d_t: float
    per_angle: list[dict] = field(default_factory=list)   # {idx, angle, coverage, by}
    open_angles: list[str] = field(default_factory=list)
    raw: str = ""
    error: str = ""


def _parse_json(txt: str) -> dict:
    s = txt.strip()
    if "```" in s:
        s = re.sub(r"```[a-z]*", "", s).replace("```", "").strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)  # bracket-walk fallback
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


async def score_coverage(
    angles: list[str],
    findings_text: str,
    *,
    model: str = COVERAGE_MODEL,
    client: httpx.AsyncClient | None = None,
) -> CoverageResult:
    """Score structural coverage of `angles` by `findings_text`. Flow-not-judge."""
    if not (os.getenv("OPENROUTER_API_KEY") or "").strip():
        return CoverageResult(d_t=0.0, error="OPENROUTER_API_KEY unset")

    own = client is None
    if own:
        client = httpx.AsyncClient()
    try:
        angles_block = "\n".join(f"{i+1}. {a}" for i, a in enumerate(angles))
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": _COVERAGE_SYSTEM},
                {"role": "user", "content":
                    f"REQUIRED ANGLES:\n{angles_block}\n\nFINDINGS:\n{findings_text}"},
            ],
            "temperature": 0.0, "max_tokens": 4000, "usage": {"include": True},
        }
        headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                   "Content-Type": "application/json"}
        raw = ""
        for attempt in range(2):
            try:
                r = await client.post(_OPENROUTER_URL, headers=headers, json=body, timeout=90.0)
                r.raise_for_status()
                raw = (r.json()["choices"][0]["message"].get("content") or "").strip()
                if raw:
                    break
            except Exception as e:
                if attempt == 1:
                    return CoverageResult(d_t=0.0, error=f"{type(e).__name__}: {e}")

        verdicts = _parse_json(raw).get("verdicts", [])
        by_idx = {int(v.get("angle", 0)): v for v in verdicts if isinstance(v, dict)}

        per_angle, total, scored, open_angles = [], 0.0, 0, []
        for i, a in enumerate(angles, 1):
            v = by_idx.get(i, {})
            cov = str(v.get("coverage", "uncovered")).lower()
            if cov not in _WEIGHT:
                cov = "uncovered"
            total += _WEIGHT[cov]
            scored += 1
            per_angle.append({"idx": i, "angle": a, "coverage": cov, "by": v.get("by", "")})
            if cov != "covered":
                open_angles.append(f"Q{i}: {a[:60]}")

        d_t = round(total / scored, 3) if scored else 0.0
        return CoverageResult(d_t=d_t, per_angle=per_angle, open_angles=open_angles, raw=raw)
    finally:
        if own:
            await client.aclose()


__all__ = ["parse_required_angles", "score_coverage", "CoverageResult", "COVERAGE_MODEL"]
