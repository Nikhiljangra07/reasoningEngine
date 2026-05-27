"""
InlineClassifier — turns one IterationRecord into a DecisionTraceBundle.

ROLE IN THE PIPELINE
====================
Sits between the raw iteration (Phase 1 — verbatim text + provenance) and
the Neo4j writer (Phase 2a — typed event nodes). The background sweeper
(Phase 3) calls this for every iteration where structured_at IS NULL,
hands the resulting bundle to Neo4jDecisionTraceWriter.write_bundle,
then stamps structured_at.

WHY ONE LLM CALL
================
Decisions, questions, references, and insights are all derivable from the
same two pieces of text (user message + assistant reply). Splitting them
across separate calls would burn 4× the cost for ~no quality gain. The
same single-call pattern that powers IterationMetadataExtractor; we just
extract a different set of fields here.

FAIL-SAFE
=========
On ANY failure (timeout, no API key, malformed JSON, network error) the
classifier returns a bundle with ONLY UserMessage + SystemResponse
populated (those are verbatim from the iteration — no extraction needed).
The other event arrays come back empty. This way the raw conversation is
ALWAYS captured in Decision Trace, even when classification fails — the
sweeper still stamps structured_at, and a future re-extraction can fill
in the typed events if we improve the prompt.

PROVENANCE STAMPING
===================
Every event the classifier produces inherits the iteration's full address
(workspace_id, surface_id, user_id, thread_id, project_id, iteration_id,
ts). The LLM doesn't pick these — they come from the parent iteration.
The LLM only chooses content (text, status, confidence, etc.).

CONFIDENCE
==========
Per the locked architecture: we persist all extracted events but flag
low-confidence ones via the `confidence` field on each event (0.0–1.0).
The retriever can later filter `WHERE confidence >= 0.7` if it wants
strict mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass

from src.core.decision_trace_types import (
    Decision,
    DecisionTraceBundle,
    Insight,
    Question,
    Reference,
    SystemResponse,
    UserMessage,
    new_dt_id,
)
from src.core.thread_types import IterationRecord

log = logging.getLogger("constellax.decision_trace_classifier")


# Default to Gemini 2.5 Flash — same model IterationMetadataExtractor uses,
# same env var operator-controlled. Cost per call is ~$0.0003 for typical
# turns; Haiku 4.5 via OpenRouter is a fine fallback if Gemini is down.
DEFAULT_MODEL = os.environ.get("GRAPHIFY_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TIMEOUT_SEC = 15.0
MAX_INPUT_CHARS = 12_000   # caps the classifier prompt size


# Allowed reference kinds — kept in sync with decision_trace_types.ReferenceKind.
# The classifier may return anything; we coerce to "other" if it picks
# something off-list.
_ALLOWED_REFERENCE_KINDS = {
    "url", "file", "mcp_resource", "memory_id", "code_symbol", "other",
}
_ALLOWED_DECISION_STATUS = {"noted", "committed", "superseded", "rejected"}


# ─── Prompt ──────────────────────────────────────────────────────────

_CLASSIFIER_PROMPT = """You read ONE conversation turn (user message + assistant reply) and extract structured events.

Return ONLY valid JSON with these four arrays (each may be empty):

{
  "decisions": [
    {"text": "...", "status": "noted|committed|superseded|rejected", "confidence": 0.0}
  ],
  "questions": [
    {"text": "...", "resolved": true, "confidence": 0.0}
  ],
  "references": [
    {"kind": "url|file|mcp_resource|memory_id|code_symbol|other",
     "target": "...", "label": "...", "confidence": 0.0}
  ],
  "insights": [
    {"text": "...", "confidence": 0.0}
  ]
}

DEFINITIONS:

- decision: a commitment the user OR system EXPLICITLY made in this turn.
  Good examples: "Going with Neo4j", "We will defer that to Phase 3", "I'll send the report tomorrow".
  Bad examples (NOT decisions): "Maybe we should...", "I'm considering...", "What if we...".
  status:
    "noted"      = soft / tentative
    "committed"  = firm, action-bound
    "superseded" = explicitly replaces a prior decision mentioned earlier
    "rejected"   = explicitly ruled out

- question: an open question raised by either party in this turn.
  resolved=true ONLY if THIS turn's reply gives a concrete answer.
  resolved=false if the question is still open at end of this turn.

- reference: any URL, file path, MCP resource name, prior memory id, or
  code symbol that was EXPLICITLY cited. Don't invent — only what's in
  the text. `kind` must be one of the listed values; `target` is the
  raw reference value; `label` is an optional human-readable name.

- insight: a noteworthy pattern, observation, or framing worth retrieving
  again later. BE CONSERVATIVE — most turns have ZERO insights. Insights
  are different from decisions: a decision is a commitment to act, an
  insight is something noticed about the problem space.

CONFIDENCE SCALE:

  1.0 = explicit in the text (only when wording is unambiguous)
  0.8 = strongly implied
  0.6 = reasonably inferred from context
  0.4 = a stretch
  0.2 = a guess

Do not use 1.0 unless the wording is explicit. Most extractions sit between 0.6 and 0.9.

OUTPUT:
- Strict JSON, no markdown fences, no prose, no comments.
- Empty arrays where nothing of that type was present.
- Do NOT invent content not present in the user/assistant text.
"""


# ─── Internal classification result ─────────────────────────────────

@dataclass
class _ClassifierStats:
    model:       str
    tokens_in:   int
    tokens_out:  int
    latency_ms:  int
    success:     bool
    error:       str | None = None


# ─── InlineClassifier ──────────────────────────────────────────────

class InlineClassifier:
    """One Gemini call → typed Decision Trace events.

    Stateless apart from model/timeout config. Safe to share one instance
    across the process; calls do not mutate instance state."""

    def __init__(self, model: str | None = None, timeout_sec: float = DEFAULT_TIMEOUT_SEC):
        self.model = model or DEFAULT_MODEL
        self.timeout_sec = timeout_sec

    async def classify_iteration(self, iteration: IterationRecord) -> tuple[DecisionTraceBundle, _ClassifierStats]:
        """Classify one IterationRecord. Always returns a bundle with at
        least UserMessage + SystemResponse populated (verbatim). On LLM
        failure, the other arrays are empty and stats.success is False."""
        user_text = (iteration.question or "").strip()
        sys_text = ""
        if iteration.response and iteration.response.synthesizer:
            sys_text = (iteration.response.synthesizer.text or "").strip()

        # Provenance — every event inherits this from the iteration.
        prov = dict(
            iteration_id=iteration.id,
            thread_id=iteration.thread_id,
            workspace_id=iteration.workspace_id or "web",
            surface_id=iteration.surface_id or "chat",
            user_id="",   # filled by caller — Iteration doesn't carry user_id today
            project_id=None,
        )
        # Try to pull user_id from iteration.meta (the trace endpoint stashes
        # it there alongside the iteration record). Falls back to empty
        # string — sweeper can supply a default at write-time if needed.
        user_id = (iteration.meta or {}).get("user_id") or ""
        prov["user_id"] = user_id

        user_ts = iteration.created_at or time.time()
        sys_ts = iteration.completed_at or user_ts

        # ALWAYS construct verbatim UserMessage + SystemResponse. These exist
        # even when the LLM call fails — they're the source of truth for
        # the conversation and don't depend on classification.
        #
        # Deterministic IDs (derived from iteration_id) make sweeper retries
        # idempotent: if the classifier fails on first attempt and we retry
        # later, we don't create duplicate verbatim nodes. The writer's
        # MERGE-by-id then makes the second pass a no-op. Each iteration
        # has at most one UserMessage and one SystemResponse, so the
        # iteration_id alone is sufficient entropy.
        bundle = DecisionTraceBundle(
            iteration_id=iteration.id,
            thread_id=iteration.thread_id,
            user_message=UserMessage(
                id=f"dt-msg-user-{iteration.id}", text=user_text, ts=user_ts, **prov,
            ) if user_text else None,
            system_response=SystemResponse(
                id=f"dt-msg-sys-{iteration.id}", text=sys_text, ts=sys_ts, **prov,
            ) if sys_text else None,
        )

        # If both texts are empty, nothing to classify — return early.
        if not user_text and not sys_text:
            return bundle, _ClassifierStats(
                model=self.model, tokens_in=0, tokens_out=0, latency_ms=0,
                success=False, error="empty input",
            )

        # No API key → return the verbatim-only bundle, log so the operator
        # sees it on first run.
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return bundle, _ClassifierStats(
                model=self.model, tokens_in=0, tokens_out=0, latency_ms=0,
                success=False, error="no GEMINI_API_KEY / GOOGLE_API_KEY",
            )

        prompt_payload = _build_payload(user_text, sys_text)
        start = time.time()
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt_payload,
                    config={
                        "system_instruction": _CLASSIFIER_PROMPT,
                        "temperature": 0.0,         # structured output → deterministic
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json",
                    },
                ),
                timeout=self.timeout_sec,
            )
        except asyncio.TimeoutError:
            log.warning("classifier timed out on iter=%s", iteration.id)
            return bundle, _ClassifierStats(
                model=self.model, tokens_in=0, tokens_out=0,
                latency_ms=int((time.time() - start) * 1000),
                success=False, error="timeout",
            )
        except Exception as e:
            log.warning("classifier LLM call failed: %s", e)
            return bundle, _ClassifierStats(
                model=self.model, tokens_in=0, tokens_out=0,
                latency_ms=int((time.time() - start) * 1000),
                success=False, error=f"{type(e).__name__}: {e}",
            )

        latency_ms = int((time.time() - start) * 1000)
        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0) if usage else 0
        tokens_out = getattr(usage, "candidates_token_count", 0) if usage else 0

        raw = getattr(response, "text", "") or ""
        parsed = _parse_classifier_json(raw)
        if parsed is None:
            log.warning("classifier returned malformed JSON: %s", raw[:200])
            return bundle, _ClassifierStats(
                model=self.model, tokens_in=tokens_in, tokens_out=tokens_out,
                latency_ms=latency_ms, success=False, error="malformed JSON",
            )

        # Translate parsed JSON into typed dataclasses. Provenance comes from
        # the iteration, NOT from the LLM.
        bundle.decisions  = _build_decisions(parsed.get("decisions") or [], prov, sys_ts)
        bundle.questions  = _build_questions(parsed.get("questions") or [], prov, sys_ts)
        bundle.references = _build_references(parsed.get("references") or [], prov, sys_ts)
        bundle.insights   = _build_insights(parsed.get("insights") or [], prov, sys_ts)

        return bundle, _ClassifierStats(
            model=self.model, tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency_ms, success=True,
        )


# ─── Helpers ─────────────────────────────────────────────────────────

def _build_payload(user_text: str, sys_text: str) -> str:
    """Trim to MAX_INPUT_CHARS, with the user message getting the larger
    share (it's usually shorter and more informative for classification)."""
    u_cap = min(len(user_text), MAX_INPUT_CHARS // 3)
    s_cap = max(0, MAX_INPUT_CHARS - u_cap - 200)
    return f"USER MESSAGE:\n{user_text[:u_cap]}\n\nASSISTANT REPLY:\n{sys_text[:s_cap]}"


def _parse_classifier_json(text: str) -> dict | None:
    """Same defensive parser as IterationMetadataExtractor — strict json
    first, then fence-strip, then balanced-object search."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _clamp_conf(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, f))


def _coerce_status(v) -> str:
    s = str(v or "").strip().lower()
    return s if s in _ALLOWED_DECISION_STATUS else "noted"


def _coerce_kind(v) -> str:
    k = str(v or "").strip().lower()
    return k if k in _ALLOWED_REFERENCE_KINDS else "other"


def _build_decisions(raw: list, prov: dict, ts: float) -> list[Decision]:
    out: list[Decision] = []
    for d in raw[:50]:                       # hard cap as defense vs runaway output
        if not isinstance(d, dict):
            continue
        text = str(d.get("text") or "").strip()
        if not text:
            continue
        out.append(Decision(
            id=new_dt_id("decision"), text=text, ts=ts,
            status=_coerce_status(d.get("status")),
            confidence=_clamp_conf(d.get("confidence", 0.5)),
            **prov,
        ))
    return out


def _build_questions(raw: list, prov: dict, ts: float) -> list[Question]:
    out: list[Question] = []
    for q in raw[:50]:
        if not isinstance(q, dict):
            continue
        text = str(q.get("text") or "").strip()
        if not text:
            continue
        out.append(Question(
            id=new_dt_id("question"), text=text, ts=ts,
            resolved=bool(q.get("resolved", False)),
            confidence=_clamp_conf(q.get("confidence", 0.5)),
            **prov,
        ))
    return out


def _build_references(raw: list, prov: dict, ts: float) -> list[Reference]:
    out: list[Reference] = []
    for r in raw[:50]:
        if not isinstance(r, dict):
            continue
        target = str(r.get("target") or "").strip()
        if not target:
            continue
        label = str(r.get("label") or "").strip() or None
        out.append(Reference(
            id=new_dt_id("reference"), kind=_coerce_kind(r.get("kind")),
            target=target, label=label, ts=ts,
            confidence=_clamp_conf(r.get("confidence", 0.5)),
            **prov,
        ))
    return out


def _build_insights(raw: list, prov: dict, ts: float) -> list[Insight]:
    out: list[Insight] = []
    for i in raw[:50]:
        if not isinstance(i, dict):
            continue
        text = str(i.get("text") or "").strip()
        if not text:
            continue
        out.append(Insight(
            id=new_dt_id("insight"), text=text, ts=ts,
            confidence=_clamp_conf(i.get("confidence", 0.5)),
            **prov,
        ))
    return out
