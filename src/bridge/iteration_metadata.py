"""
IterationMetadataExtractor — pulls the seven memory-layer signals out of
a completed iteration in ONE LLM call:

    1. entities           — extracted concepts/people/products/places/metrics
    2. tags               — LLM-generated topic tags (≤5)
    3. domains            — high-level domain classification
    4. user_mode          — exploratory | decisive | stuck | venting | analytical
    5. time_horizon       — immediate | weeks | months | year+
    6. load_bearing_assumption — the framing doing the most work
    7. (entities aggregated up to thread.all_entities for fast filtering)

WHY ONE CALL
============
All seven signals come from the same source material (the user's question +
the engine's response). Splitting into seven calls would burn 7x the cost
for ~no quality gain. One structured-output call returns all of it for
~$0.001–0.003 via Gemini 2.5 Flash.

SAFETY
======
Returns a dataclass with all fields populated; every field defaults to
empty/None on extraction failure. The dispatcher hook treats this as
best-effort: if the metadata call fails, the iteration is still saved,
just with empty memory fields. The pipeline never crashes here.

PROVIDER
========
Default: Gemini 2.5 Flash (whatever GRAPHIFY_GEMINI_MODEL is set to).
Same env var graphify reads, so the operator has one place to control
"which Gemini model handles structured memory extraction."
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass

from src.core.thread_types import Entity

log = logging.getLogger("constellax.metadata_extractor")


DEFAULT_MODEL = os.environ.get("GRAPHIFY_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TIMEOUT_SEC = 20.0
MAX_INPUT_CHARS = 12_000   # enough for question + memo body


@dataclass
class ExtractedMetadata:
    """Output of one extraction call. Every field has a safe default."""
    entities:        list[Entity]
    tags:            list[str]
    domains:         list[str]
    user_mode:       str | None       # one of UserMode literals
    time_horizon:    str | None       # one of TimeHorizon literals
    load_bearing_assumption: str | None
    model:           str
    tokens_in:       int = 0
    tokens_out:      int = 0
    latency_ms:      int = 0
    cost_usd:        float = 0.0
    success:         bool = False
    error:           str | None = None

    @classmethod
    def empty(cls, model: str, error: str | None = None) -> "ExtractedMetadata":
        return cls(
            entities=[], tags=[], domains=[],
            user_mode=None, time_horizon=None, load_bearing_assumption=None,
            model=model, success=False, error=error,
        )


# ─── Prompt + JSON schema (kept inline so the contract is one file) ───

_EXTRACTION_PROMPT = """You analyze a completed reasoning thread and extract structured memory signals.
Return STRICT JSON only — no prose, no markdown fences, no explanation.

The user asked a question and the system produced a response. You must extract:

1. entities: list of concrete things named or implied in the question/response.
   Each entity has {name, kind, salience (0.0–1.0)}.
   kind ∈ {"person", "concept", "product", "place", "time_window", "metric", "other"}
   salience = how central this entity is to the iteration (1.0 = core, 0.3 = passing).
   Cap at 12 entities. Skip generic words ("the", "thing").

2. tags: 3–5 short topic tags in kebab-case (e.g. "startup-strategy", "team-shape", "paid-acquisition").

3. domains: 1–3 high-level domains. Examples: "business", "strategy", "technical", "personal",
   "relationship", "career", "finance", "code", "product".

4. user_mode: ONE of "exploratory" | "decisive" | "stuck" | "venting" | "analytical"
   based on the user's apparent stance in the question.

5. time_horizon: ONE of "immediate" | "weeks" | "months" | "year+"
   based on when the decision/outcome unfolds. Use null if not inferable.

6. load_bearing_assumption: the single phrase or framing in the user's prompt that's
   doing the most work in their reasoning. If it doesn't hold, the verdict collapses.
   One sentence. Use null if the question is too small to have one (e.g. greetings).

OUTPUT FORMAT — return EXACTLY this JSON structure:

{
  "entities": [{"name": "...", "kind": "...", "salience": 0.0}],
  "tags": ["...", "..."],
  "domains": ["..."],
  "user_mode": "...",
  "time_horizon": "..." | null,
  "load_bearing_assumption": "..." | null
}
"""


class IterationMetadataExtractor:
    """Async wrapper. One method: `extract(question, response_text)`."""

    def __init__(self, model: str | None = None, timeout_sec: float = DEFAULT_TIMEOUT_SEC):
        self.model = model or DEFAULT_MODEL
        self.timeout_sec = timeout_sec

    async def extract(self, question: str, response_text: str) -> ExtractedMetadata:
        if not question.strip() and not response_text.strip():
            return ExtractedMetadata.empty(self.model, error="empty input")

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return ExtractedMetadata.empty(self.model, error="no GEMINI_API_KEY")

        prompt = _build_user_message(question, response_text)

        start = time.time()
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "system_instruction": _EXTRACTION_PROMPT,
                        "temperature":        0.0,    # structured output → low temp
                        "max_output_tokens":  2048,
                        "response_mime_type": "application/json",
                    },
                ),
                timeout=self.timeout_sec,
            )
        except asyncio.TimeoutError:
            return ExtractedMetadata.empty(self.model, error="timeout")
        except Exception as e:
            log.warning("metadata extraction call failed: %s", e)
            return ExtractedMetadata.empty(self.model, error=f"{type(e).__name__}: {e}")

        latency_ms = int((time.time() - start) * 1000)
        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", 0) if usage else 0
        tokens_out = getattr(usage, "candidates_token_count", 0) if usage else 0

        raw_text = getattr(response, "text", "") or ""
        parsed = _parse_extraction_json(raw_text)
        if parsed is None:
            return ExtractedMetadata.empty(self.model, error="malformed JSON in response")

        return _build_metadata(
            parsed, self.model, tokens_in, tokens_out, latency_ms,
        )


# ─── Helpers ──────────────────────────────────────────────────────────

def _build_user_message(question: str, response_text: str) -> str:
    q = question[:MAX_INPUT_CHARS // 3]
    r = response_text[:MAX_INPUT_CHARS - len(q) - 200]
    return f"QUESTION:\n{q}\n\nRESPONSE:\n{r}"


def _parse_extraction_json(text: str) -> dict | None:
    """Defensive JSON parser. Tries direct json.loads, then strips code fences,
    then locates the first balanced object. Returns None if all fail."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Last resort: find the first balanced { ... }
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


_VALID_USER_MODES = {"exploratory", "decisive", "stuck", "venting", "analytical"}
_VALID_HORIZONS   = {"immediate", "weeks", "months", "year+"}
_VALID_ENT_KINDS  = {"person", "concept", "product", "place", "time_window", "metric", "other"}


def _build_metadata(
    parsed: dict, model: str, tokens_in: int, tokens_out: int, latency_ms: int,
) -> ExtractedMetadata:
    """Coerce a parsed dict into ExtractedMetadata, filtering invalid values."""
    entities_raw = parsed.get("entities") or []
    entities: list[Entity] = []
    for e in entities_raw[:12] if isinstance(entities_raw, list) else []:
        if not isinstance(e, dict): continue
        name = (e.get("name") or "").strip()
        kind = (e.get("kind") or "other").strip().lower()
        if kind not in _VALID_ENT_KINDS: kind = "other"
        try:
            salience = float(e.get("salience", 1.0))
            salience = max(0.0, min(1.0, salience))
        except (TypeError, ValueError):
            salience = 1.0
        if not name: continue
        entities.append(Entity(name=name, kind=kind, salience=salience))

    tags_raw = parsed.get("tags") or []
    tags = [str(t).strip().lower() for t in tags_raw if isinstance(t, str) and t.strip()][:8]

    domains_raw = parsed.get("domains") or []
    domains = [str(d).strip().lower() for d in domains_raw if isinstance(d, str) and d.strip()][:4]

    user_mode = parsed.get("user_mode")
    if isinstance(user_mode, str): user_mode = user_mode.strip().lower()
    if user_mode not in _VALID_USER_MODES: user_mode = None

    time_horizon = parsed.get("time_horizon")
    if isinstance(time_horizon, str): time_horizon = time_horizon.strip().lower()
    if time_horizon not in _VALID_HORIZONS: time_horizon = None

    lba = parsed.get("load_bearing_assumption")
    if not isinstance(lba, str) or not lba.strip(): lba = None
    else: lba = lba.strip()

    cost = _approximate_cost(model, tokens_in, tokens_out)
    return ExtractedMetadata(
        entities=entities, tags=tags, domains=domains,
        user_mode=user_mode, time_horizon=time_horizon,
        load_bearing_assumption=lba,
        model=model, tokens_in=tokens_in, tokens_out=tokens_out,
        latency_ms=latency_ms, cost_usd=cost, success=True,
    )


def _approximate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Rough cost estimate for the model used. Kept conservative — actual
    Google pricing varies by short/long context tier."""
    # Default to Gemini 2.5 Flash short-context pricing
    in_per_1m = 0.075
    out_per_1m = 0.30
    if "flash-lite" in model:
        in_per_1m, out_per_1m = 0.05, 0.20
    elif "flash-preview" in model or "3-flash" in model:
        in_per_1m, out_per_1m = 0.50, 3.00
    elif "pro" in model:
        in_per_1m, out_per_1m = 1.25, 5.00
    return tokens_in * in_per_1m / 1_000_000 + tokens_out * out_per_1m / 1_000_000
