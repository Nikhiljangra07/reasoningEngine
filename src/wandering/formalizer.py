"""
Formalizer seat — DeepSeek R1 renders each finished blend into testable math.

WHERE IT SITS
-------------
After the blender (Opus 4.8) produces blends, before/independent-of drift &
verify. R1 is JUNIOR to the blender: Opus makes the blend (the creative leap),
R1 only grounds + formalizes it downstream. R1 never invents or rewrites a blend.

This seat is ADDITIVE and NON-INVASIVE: it reads blends and returns a separate
FormalizeReport. It does NOT mutate the Blend objects, so drift / verify / rank
behave identically whether it runs or not.

WHY IT TALKS TO OPENROUTER DIRECTLY (not via LLMClient)
------------------------------------------------------
R1 is a reasoning model: on a hard blend it runs ~170s and emits LaTeX. Two
consequences, both learned live on 2026-06-15 (see scripts/run_formalize.py):
  - LLMClient.TIMEOUT_SECONDS is 150s — it aborts R1 mid-reason every time.
    So we call OpenRouter directly with a 300s timeout.
  - LaTeX backslashes break JSON (`\\frac` is an invalid escape), so R1 returns
    a `@@SECTION@@` delimiter format, parsed by markers — not JSON.
Cost comes straight from OpenRouter's `usage.cost`.

THE LAWS (validated 2026-06-15 on blend-02 + blend-03: faithful math, real
citations, honest `partial`, real falsifiers, qualitative parts flagged not
faked) are the contract R1 formalizes under. Edit them here.

NOT HERE (reserved): the router — when to look up more / when to stop. R1 plans
its reference lookups once (Pass A) and formalizes once (Pass B). Any iterative
"search-more / continue / stop" control belongs to the router, by design.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

from src.bridge.web_search import SearchResult, web_search

log = logging.getLogger("constellax.wandering.formalizer")

# ---------------------------------------------------------------------------
# Seat config
# ---------------------------------------------------------------------------
R1_FORMALIZE_MODEL = "deepseek/deepseek-r1"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/nikhiljangra/reasoningEngine",
}
R1_TIMEOUT_S = 300.0          # R1 reasons ~170s on the big prompt; 150s cap aborts it
R1_LOOKUP_MAX_TOKENS = 800
R1_FORMALIZE_MAX_TOKENS = 8000
R1_RETRIES = 3                # OpenRouter routes R1 across upstreams; some return empty

#: A search function: (query, max_results) -> SearchResult. Defaults to web_search.
SearchFn = Callable[..., Awaitable[SearchResult]]


# ---------------------------------------------------------------------------
# THE LAWS  (the contract R1 formalizes under — vet/edit here)
# ---------------------------------------------------------------------------
LAWS = """You are DeepSeek R1, the JUNIOR formalizer working beside a senior
synthesizer (Claude Opus 4.8). Opus has already produced the BLEND below — the
creative leap. You did NOT create it and you will NOT rewrite, sharpen, or
"improve" the idea. Your one job: formalize this blend into honest mathematics
or physics — as fully and faithfully as the structure honestly allows — so it
can be tested instead of argued about.

THE LAWS — obey every one:

1. SOLVE IT TO YOUR BEST ABILITY. Formalize as much of this blend as genuinely,
   faithfully maps to math or physics — push as far as the structure honestly
   allows; do NOT stop short. There is NO quota and NO cap. Partial is
   acceptable ONLY where the remainder truly does not formalize — and YOU
   decide where that frontier sits, not a fixed percentage. Leave only the
   genuinely non-mathematical parts as prose. Do NOT force a symbol onto
   something that isn't mathematical, and do NOT pad.

2. ABSTAIN HONESTLY. If the blend has no faithful formal form, set
   FORMALIZABLE = no and explain why. A correct abstention is worth MORE than
   invented math. Manufacturing equations to look rigorous is the single worst
   thing you can do here.

3. JUSTIFY AND CITE EVERY BORROWING. For each variable, function, operator, or
   physical law you introduce, prove WHY it belongs: map the blend's own
   element -> the mathematical object and show the derivation, the way a
   mathematician justifies each line. AND whenever you invoke an established
   formula, theorem, or method — from your own knowledge of the literature OR
   from the web references — CITE its source and state (a) why THIS formula
   applies here and (b) how that existing work helps THIS blend. No unexplained
   symbols. No formula pulled from thin air.

4. FULL AUTHORITY, FULLY EARNED. You may use any branch of mathematics or
   physics you need. But every borrowed tool must be warranted by the blend's
   OWN structure — never because it looks impressive. State the assumption
   behind each borrowing and name exactly where it would break.

5. END IN A TEST. The formalization must cash out as something runnable: state
   what it PREDICTS, what computation or observation would CONFIRM it, and what
   would FALSIFY it. If you cannot state a falsifier, say so plainly.

6. REFERENCE USE ONLY — BUT THE WHOLE LITERATURE IS OPEN. Ground your formalism
   in any established mathematics or physics — known formulas, theorems,
   textbook methods, published papers — drawn from your own knowledge of the
   literature OR from the web references provided. This is REFERENCE use: every
   borrowing obeys Law 3 (cite it, say why it applies, say how it helps this
   blend). Do NOT re-research the blend's topic or judge its novelty.

OUTPUT FORMAT — use EXACTLY these section markers, in this order. Write freely
inside each section (LaTeX / backslashes / multi-line math are all fine — there
is no JSON to escape). Emit nothing before @@FORMALIZABLE@@.

@@FORMALIZABLE@@
yes | partial | no

@@FORMAL_CORE@@
The math/physics, with the derivation and per-object justification (Law 3).
Empty if FORMALIZABLE is no.

@@OBJECTS@@
One per line:  symbol :: the blend-element it stands for :: why it belongs

@@CITATIONS@@
One per line:  source :: why it applies here :: how it helps THIS blend

@@PROSE_REMAINDER@@
What you deliberately left qualitative, and why (Law 1).

@@TEST@@
predicts: ...
confirms: ...
falsifies: ...

@@BREAKPOINTS@@
Where the formalism would break (Law 4).

@@CONFIDENCE@@
A single number 0.0-1.0.

@@CAVEAT@@
One line — the thing you are least sure about."""

LOOKUP_PROMPT = """You are about to formalize the BLEND below into mathematics /
physics. FIRST: list UP TO 3 reference lookups that would let you ground the
formalization in ESTABLISHED, real models — formal definitions, standard
equations, canonical forms. Reference material ONLY, e.g.:
  "expected information gain formula Bayesian experimental design"
  "colimit definition category theory"
Do NOT request lookups about the blend's topic, its authors, or whether it is
novel. If you need no lookups, return an empty list.

Return ONE JSON object and nothing else: {"lookups": ["query1", "query2"]}"""


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BlendFormalization:
    blend_id:     str
    formalizable: str            # yes | partial | no | (empty if call failed)
    sections:     dict           # parsed @@SECTION@@ blocks
    raw:          str            # raw R1 Pass-B output (delimiter format)
    lookups:      list[str] = field(default_factory=list)
    cost_usd:     float     = 0.0
    ok:           bool      = False   # False if R1 returned nothing / unparseable

    def to_dict(self) -> dict:
        return {
            "blend_id":     self.blend_id,
            "formalizable": self.formalizable,
            "sections":     self.sections,
            "raw":          self.raw,
            "lookups":      list(self.lookups),
            "cost_usd":     round(self.cost_usd, 4),
            "ok":           self.ok,
        }


@dataclass
class FormalizeReport:
    formalizations: list[BlendFormalization] = field(default_factory=list)
    model:          str   = R1_FORMALIZE_MODEL
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model":          self.model,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "formalizations": [f.to_dict() for f in self.formalizations],
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict | None:
    """json.loads, then a balanced-brace fallback (for the small Pass-A JSON)."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _parse_sections(text: str) -> dict:
    """Split @@SECTION@@ blocks. Leading reasoning before the first marker is
    ignored. Returns {} only if no markers are present at all."""
    if not text or "@@" not in text:
        return {}
    parts = re.split(r"@@([A-Z_]+)@@", text)   # [pre, NAME, body, NAME, body, ...]
    return {parts[i].strip().lower(): parts[i + 1].strip()
            for i in range(1, len(parts) - 1, 2)}


def _blend_dict(b) -> dict:
    """Normalize a Blend object OR an already-serialized dict to the dict shape."""
    return b.to_dict() if hasattr(b, "to_dict") else dict(b)


def _blend_text(b: dict) -> str:
    """The blend core R1 formalizes — thesis/mechanism/structure + the source
    SHAPES (the actual methods), not the full bridge prose. Operates on the
    BlendBatch.to_dict() shape (same shape as a saved blends.json entry)."""
    sel = b.get("selection", {}) or {}
    shapes = [
        f"  - [{c.get('report_id', '?')}] {(c.get('source_shape') or '').strip()}"
        for c in (b.get("source_cards", []) or [])
    ]
    shapes_block = "\n".join(shapes) if shapes else "  (none)"
    return (
        f"BLEND {b.get('blend_id', '?')}\n\n"
        f"THESIS:\n{(b.get('thesis') or '').strip()}\n\n"
        f"MECHANISM:\n{(b.get('mechanism') or '').strip()}\n\n"
        f"EMERGENT STRUCTURE:\n{(b.get('emergent_structure') or '').strip()}\n\n"
        f"ADVANCES CUSHION:\n{(b.get('advances_cushion') or '').strip()}\n\n"
        f"SELECTION TENSION:\n{(sel.get('tension') or '').strip()}\n\n"
        f"SOURCE METHODS (the real machinery each card drew on):\n{shapes_block}\n"
    )


# ---------------------------------------------------------------------------
# R1 call (direct OpenRouter — bypasses LLMClient's 150s cap)
# ---------------------------------------------------------------------------
async def _r1(system: str, user: str, max_tokens: int, model: str, tries: int = R1_RETRIES):
    """Call R1 directly via OpenRouter. Returns (content, cost_usd). Retries on
    empty/error — OpenRouter routes R1 across upstreams, some flaky."""
    last_err = ""
    for t in range(tries):
        try:
            async with httpx.AsyncClient(timeout=R1_TIMEOUT_S) as hc:
                r = await hc.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json={
                    "model": model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "max_tokens": max_tokens,
                })
            data = r.json()
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}: {str(data)[:160]}"
            else:
                ch = (data.get("choices") or [{}])[0]
                content = ((ch.get("message") or {}).get("content")) or ""
                if content.strip():
                    cost = float((data.get("usage") or {}).get("cost", 0.0) or 0.0)
                    return content, cost
                last_err = "empty content"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:140]}"
        if t < tries - 1:
            log.info("formalizer: R1 retry %d/%d (%s)", t + 1, tries - 1, last_err)
            await asyncio.sleep(4.0)
    log.warning("formalizer: R1 failed after %d tries (%s)", tries, last_err)
    return "", 0.0


# ---------------------------------------------------------------------------
# The seat
# ---------------------------------------------------------------------------
async def formalize_one(blend, *, search_fn: SearchFn,
                        model: str = R1_FORMALIZE_MODEL) -> BlendFormalization:
    """Two-pass formalization of one blend: R1 plans reference lookups -> we
    fetch (reference-only) -> R1 formalizes under THE LAWS."""
    b = _blend_dict(blend)
    bid = b.get("blend_id", "?")
    blend_text = _blend_text(b)
    spend = 0.0

    # Pass A — R1 plans its reference lookups
    a_content, a_cost = await _r1(LOOKUP_PROMPT, blend_text, R1_LOOKUP_MAX_TOKENS, model)
    spend += a_cost
    plan = _extract_json(a_content) or {}
    lookups = [q for q in (plan.get("lookups") or []) if isinstance(q, str)][:3]

    # fetch — reference snippets (never raises; DDG fallback if no key)
    refs = []
    for q in lookups:
        try:
            res = await search_fn(q, max_results=3)
            lines = "\n".join(h.as_context_line() for h in res.hits[:3]) or "  (no hits)"
        except Exception as e:  # search must never break the seat
            lines = f"  (search failed: {type(e).__name__})"
        refs.append(f"LOOKUP: {q}\n{lines}")
    refs_block = (
        "\n\nWEB REFERENCES (established models/definitions — Law 6, reference only):\n\n"
        + "\n\n".join(refs)
    ) if refs else "\n\n(No lookups requested.)"

    # Pass B — R1 formalizes under THE LAWS
    b_content, b_cost = await _r1(LAWS, blend_text + refs_block, R1_FORMALIZE_MAX_TOKENS, model)
    spend += b_cost
    sec = _parse_sections(b_content)

    return BlendFormalization(
        blend_id=bid,
        formalizable=(sec.get("formalizable", "") or "").split()[0] if sec.get("formalizable") else "",
        sections=sec,
        raw=b_content,
        lookups=lookups,
        cost_usd=spend,
        ok=bool(sec),
    )


async def formalize_blends(
    blends,
    *,
    model: str = R1_FORMALIZE_MODEL,          # accepted for telemetry/symmetry
    search_fn: SearchFn | None = None,
    on_progress: Callable[[str, dict], None] | None = None,
) -> FormalizeReport:
    """Formalize each blend (sequential — R1 is slow + rate-limited; the retry
    handles flakiness). Accepts Blend objects OR serialized blend dicts. Never
    raises on a single-blend failure: a failed blend is recorded ok=False."""
    search_fn = search_fn or web_search
    report = FormalizeReport(model=model)
    for blend in (blends or []):
        try:
            f = await formalize_one(blend, search_fn=search_fn, model=model)
        except Exception as e:  # one blend dying must not kill the stage
            bid = _blend_dict(blend).get("blend_id", "?")
            log.warning("formalizer: blend %s crashed (%s)", bid, e)
            f = BlendFormalization(blend_id=bid, formalizable="", sections={}, raw="",
                                   lookups=[], cost_usd=0.0, ok=False)
        report.formalizations.append(f)
        report.total_cost_usd += f.cost_usd
        if on_progress:
            on_progress("formalized", {
                "blend_id": f.blend_id, "formalizable": f.formalizable,
                "ok": f.ok, "cost": round(f.cost_usd, 4),
            })
    return report


# ---------------------------------------------------------------------------
# Markdown rendering — LaTeX \( \) -> $…$ and \[ \] -> $$…$$ for KaTeX preview
# ---------------------------------------------------------------------------
def _tex(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"\\\[\s*(.*?)\s*\\\]",
               lambda m: "\n\n$$\n" + m.group(1).strip() + "\n$$\n\n", s, flags=re.DOTALL)
    s = re.sub(r"\\\(\s*(.*?)\s*\\\)",
               lambda m: "$" + m.group(1).strip() + "$", s, flags=re.DOTALL)
    return s


def _as_table(body: str, headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    extra = []
    for ln in (body or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "::" in ln:
            parts = [_tex(p.strip()) for p in ln.split("::")]
            parts += [""] * (len(headers) - len(parts))
            out.append("| " + " | ".join(parts[:len(headers)]) + " |")
        else:
            extra.append(_tex(ln))
    return "\n".join(out) + ("\n\n" + "\n".join(extra) if extra else "")


def render_markdown(report: FormalizeReport) -> str:
    L = ["# R1 Formalizer — results", "",
         f"- **model:** `{report.model}`",
         f"- **total cost:** ${round(report.total_cost_usd, 4)}",
         "", "> Open VS Code preview (⇧⌘V) to render the math.", ""]
    for f in report.formalizations:
        sec = f.sections
        L += ["\n---\n",
              f"## {f.blend_id} — formalizable: **{f.formalizable or 'FAILED'}** "
              f"· confidence {sec.get('confidence', '?')} · ${round(f.cost_usd, 4)}", ""]
        if not f.ok:
            L += ["_R1 returned nothing parseable for this blend (see raw)._", "",
                  "```", (f.raw or "")[:800], "```", ""]
            continue
        if f.lookups:
            L += ["**Reference lookups R1 requested:**"] + [f"- {q}" for q in f.lookups] + [""]
        if sec.get("formal_core"):
            L += ["### Formal core", "", _tex(sec["formal_core"]), ""]
        if sec.get("objects"):
            L += ["### Objects (each justified)", "",
                  _as_table(sec["objects"], ["Symbol", "Maps to (blend element)", "Why it belongs"]), ""]
        if sec.get("citations"):
            L += ["### Citations", "",
                  _as_table(sec["citations"], ["Source", "Why it applies", "How it helps"]), ""]
        if sec.get("prose_remainder"):
            L += ["### Left qualitative (prose, not faked)", "", _tex(sec["prose_remainder"]), ""]
        if sec.get("test"):
            L += ["### Test (predict / confirm / falsify)", "", _tex(sec["test"]), ""]
        if sec.get("breakpoints"):
            L += ["### Where it breaks", "", _tex(sec["breakpoints"]), ""]
        if sec.get("caveat"):
            L += ["### Honest caveat", "", _tex(sec["caveat"]), ""]
    return "\n".join(L)


__all__ = (
    "R1_FORMALIZE_MODEL",
    "BlendFormalization",
    "FormalizeReport",
    "formalize_one",
    "formalize_blends",
    "render_markdown",
)
