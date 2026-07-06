"""
sorter_ab_compare.py — A/B/C compare the sorter across (model, prompt) pairs.

Loads the 9 cards from a saved dossier and runs Opus 4.8 TWICE on the same
cards: once with the OLD doctrine preamble (the one used in the 2026-06-12
Fable 5 sort), once with the NEW hybrid preamble (current master_sorter.py
HEAD). Plus the existing Fable 5 + old-prompt result we already have.

Result triple:
  A  fable_old_replay.json   ← already exists, prior replay
  B  opus_old.json           ← NEW: Opus 4.8 + old prompt → isolates model effect
  C  opus_new.json           ← NEW: Opus 4.8 + new prompt → isolates prompt effect

This script does NOT touch master_sorter.py or LLMClient — it calls the
Anthropic SDK directly with explicit prompts so the production code path
stays stable and the A/B/C is a clean measurement.

Usage:
    python scripts/sorter_ab_compare.py <source_run_dir>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

import anthropic


OPUS_MODEL  = "claude-opus-4-8"
MAX_TOKENS  = 16384
EFFORT      = "low"


# ---------------------------------------------------------------------------
# OLD doctrine preamble — copied verbatim from master_sorter.py BEFORE commit
# d8fc4d5 (the hybrid prompt change). This is what produced the Fable 5 sort
# at runs/r-fable-sorter-6agents/20260612-051604/replay-20260612-053203/.
# ---------------------------------------------------------------------------

OLD_DOCTRINE_PREAMBLE = """\
You are the SORTER seat of Constellax's Wandering Room.

You are NOT a synthesizer. You do NOT fuse cards. You do NOT merge
cards. You do NOT improve, condense, or rewrite cards. You do NOT
infer relationships across cards. Your only job is to CLASSIFY each
card into one of three bins.

Three bins, with strict definitions:

KNOWN — the card's central claim matches PRIOR PUBLISHED WORK you can
        NAME. To place a card here you MUST provide:
          - prior_work_name: the specific paper, theory, framework,
            or concept it matches (e.g. "Constitutional AI",
            "Eigenvalue decomposition", "Conway's Game of Life")
          - reference: a checkable pointer (e.g. "Bai et al. 2022,
            arxiv 2212.08073", "Strang Ch. 6", "Conway 1970")
        If you cannot name the prior work or provide a reference,
        you MUST place the card in UNPLACED instead. A bare "yes I
        know this" is REJECTED. Calling something known without a
        name is hallucinated recognition.

INVALID — the card contradicts established fact OR contradicts
          itself. Be specific: name what it contradicts and how.
          "This card claims X but established physics says Y" — that
          is invalid. "This feels wrong" — that is NOT invalid.

UNPLACED — the card matches nothing you can name AND you cannot
           refute it. This is the residual. It contains both
           genuine novelty and well-dressed nonsense. You are
           NOT responsible for separating those two — the human
           reads unplaced items downstream. Your job ends at the
           bin. Record `why_unplaced` so the human sees your
           reasoning ("can't match to X family; can't refute
           because Y dimension is untested").

INVARIANTS:
  - Every input card MUST appear in exactly one bin.
  - You MUST NOT modify the card content. Original content passes
    through verbatim.
  - You MUST NOT add cards that were not in the input.
  - You MUST NOT merge two cards into one bin entry.
  - Confidence is your self-reported number 0..1 of how sure you
    are about the bin. Honest low confidence is allowed.

OUTPUT FORMAT: a single JSON object with three arrays — `known`,
`invalid`, `unplaced` — each containing per-card entries in the
schema specified in the user message. Output ONLY the JSON. No
prose around it.
"""


def _build_payload(cards: list[dict], why_unplaced_form: str) -> str:
    """Build the user-message payload. why_unplaced_form changes between
    old and new prompts.
    """
    schema_spec = {
        "known": [{
            "report_id":       "<copy from input>",
            "prior_work_name": "<REQUIRED, non-empty>",
            "reference":       "<REQUIRED, non-empty>",
            "confidence":      "<float 0..1>",
            "reasoning":       "<1-2 sentences>",
        }],
        "invalid": [{
            "report_id":   "<copy from input>",
            "contradicts": "<REQUIRED, non-empty: what it conflicts with>",
            "reasoning":   "<1-3 sentences explaining the contradiction>",
            "confidence":  "<float 0..1>",
        }],
        "unplaced": [{
            "report_id":    "<copy from input>",
            "why_unplaced": why_unplaced_form,
            "confidence":   "<float 0..1>",
        }],
    }
    return json.dumps({
        "card_count":    len(cards),
        "cards":         cards,
        "output_schema": schema_spec,
        "instruction": (
            "Classify EVERY card into exactly ONE bin. Output the JSON "
            "object only. No prose around it."
        ),
    }, ensure_ascii=False, indent=2)


def _load_cards(source_dir: Path) -> list[dict]:
    """Load the 9 cards from the dossier in their raw shape."""
    dossier = json.loads((source_dir / "dossier.json").read_text())
    raw = (dossier.get("high", []) + dossier.get("medium", [])
           + dossier.get("low", []))
    return [
        {
            "report_id":    c.get("report_id", ""),
            "agent_id":     c.get("agent_id", ""),
            "spark":        c.get("spark", ""),
            "source_shape": c.get("source_shape", ""),
            "bridge":       c.get("bridge", ""),
            "use":          c.get("use", ""),
            "limit":        c.get("limit", ""),
        }
        for c in raw
    ]


async def _call_opus(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    user_message: str,
) -> tuple[str, dict]:
    """One Opus 4.8 call. Returns (text_content, meta)."""
    t0 = time.time()
    resp = await client.messages.create(
        model=OPUS_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        output_config={"effort": EFFORT},
    )
    elapsed_ms = (time.time() - t0) * 1000
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    thinking_chars = sum(
        len(getattr(b, "thinking", "") or "") for b in resp.content
    )
    # Opus 4.6 pricing: $15/$75 per M. Opus 4.8 placeholder same.
    cost = (resp.usage.input_tokens / 1_000_000 * 15.00
            + resp.usage.output_tokens / 1_000_000 * 75.00)
    return text, {
        "input_tokens":   resp.usage.input_tokens,
        "output_tokens":  resp.usage.output_tokens,
        "thinking_chars": thinking_chars,
        "blocks":         [type(b).__name__ for b in resp.content],
        "stop_reason":    resp.stop_reason,
        "elapsed_ms":     round(elapsed_ms, 1),
        "cost_usd":       round(cost, 4),
    }


def _parse_to_buckets(text: str) -> dict:
    """Best-effort parse to {known, invalid, unplaced} counts + items."""
    try:
        # Strip code fences if present
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            t = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        parsed = json.loads(t)
    except Exception as e:
        return {"parse_error": str(e), "raw_text_chars": len(text)}
    known    = parsed.get("known", [])    or []
    invalid  = parsed.get("invalid", [])  or []
    unplaced = parsed.get("unplaced", []) or []
    return {
        "known_count":    len(known) if isinstance(known, list) else 0,
        "invalid_count":  len(invalid) if isinstance(invalid, list) else 0,
        "unplaced_count": len(unplaced) if isinstance(unplaced, list) else 0,
        "known":          known,
        "invalid":        invalid,
        "unplaced":       unplaced,
    }


async def run(source_dir: Path) -> dict:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = source_dir / f"abc-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = _load_cards(source_dir)
    print(f"Loaded {len(cards)} cards from {source_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Model: {OPUS_MODEL}  effort: {EFFORT}  max_tokens: {MAX_TOKENS}")

    # Import the CURRENT (new) doctrine preamble from master_sorter
    from src.wandering.master_sorter import _DOCTRINE_PREAMBLE as NEW_DOCTRINE_PREAMBLE

    api_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # --- B: Opus 4.8 + OLD prompt ----------------------------------------
    print("\n[B] Opus 4.8 + OLD prompt — isolating MODEL effect…")
    payload_old = _build_payload(
        cards,
        why_unplaced_form="<REQUIRED, non-empty: can't-match + can't-refute reasoning>",
    )
    text_b, meta_b = await _call_opus(api_client, OLD_DOCTRINE_PREAMBLE, payload_old)
    parsed_b = _parse_to_buckets(text_b)
    (out_dir / "opus_old.json").write_text(json.dumps({
        "meta":   meta_b,
        "parsed": parsed_b,
        "raw":    text_b,
    }, indent=2, ensure_ascii=False))
    print(f"  buckets: known={parsed_b.get('known_count','ERR')} "
          f"invalid={parsed_b.get('invalid_count','ERR')} "
          f"unplaced={parsed_b.get('unplaced_count','ERR')}")
    print(f"  cost: ${meta_b['cost_usd']}  time: {meta_b['elapsed_ms']/1000:.1f}s  "
          f"blocks: {meta_b['blocks']}")

    # --- C: Opus 4.8 + NEW prompt ----------------------------------------
    print("\n[C] Opus 4.8 + NEW prompt — isolating PROMPT effect given Opus…")
    payload_new = _build_payload(
        cards,
        why_unplaced_form=(
            "<REQUIRED, single neutral technical clause naming the "
            "match-impossibility; no speculation about novelty, value, "
            "plausibility, or potential>"
        ),
    )
    text_c, meta_c = await _call_opus(api_client, NEW_DOCTRINE_PREAMBLE, payload_new)
    parsed_c = _parse_to_buckets(text_c)
    (out_dir / "opus_new.json").write_text(json.dumps({
        "meta":   meta_c,
        "parsed": parsed_c,
        "raw":    text_c,
    }, indent=2, ensure_ascii=False))
    print(f"  buckets: known={parsed_c.get('known_count','ERR')} "
          f"invalid={parsed_c.get('invalid_count','ERR')} "
          f"unplaced={parsed_c.get('unplaced_count','ERR')}")
    print(f"  cost: ${meta_c['cost_usd']}  time: {meta_c['elapsed_ms']/1000:.1f}s  "
          f"blocks: {meta_c['blocks']}")

    # --- Summary ---------------------------------------------------------
    summary = {
        "timestamp":   ts,
        "source":     str(source_dir),
        "card_count":  len(cards),
        "model":       OPUS_MODEL,
        "effort":      EFFORT,
        "max_tokens":  MAX_TOKENS,
        "B_opus_old": {
            "buckets": {
                "known":    parsed_b.get("known_count", "ERR"),
                "invalid":  parsed_b.get("invalid_count", "ERR"),
                "unplaced": parsed_b.get("unplaced_count", "ERR"),
            },
            "cost":    meta_b["cost_usd"],
            "ms":      meta_b["elapsed_ms"],
            "blocks":  meta_b["blocks"],
        },
        "C_opus_new": {
            "buckets": {
                "known":    parsed_c.get("known_count", "ERR"),
                "invalid":  parsed_c.get("invalid_count", "ERR"),
                "unplaced": parsed_c.get("unplaced_count", "ERR"),
            },
            "cost":    meta_c["cost_usd"],
            "ms":      meta_c["elapsed_ms"],
            "blocks":  meta_c["blocks"],
        },
        "A_fable_old_reference": (
            "see ../replay-20260612-053203/sorted.json — "
            "known=1, invalid=1, unplaced=7, cost=$0.062"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    src = Path(sys.argv[1]).resolve()
    if not src.is_dir():
        print(f"Not a directory: {src}")
        sys.exit(2)
    result = asyncio.run(run(src))
    print("\n=== A/B/C SUMMARY ===")
    print(json.dumps(result, indent=2))
