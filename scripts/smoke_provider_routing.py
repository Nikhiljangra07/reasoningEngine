"""
Smoke test for multi-provider routing.

Fires one tiny call against each provider whose key is set, then prints the
CALL log and the summary block. The "provider" field in each CALL line is
the proof of which key was billed.

Usage: PYTHONPATH=. python3 scripts/smoke_provider_routing.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env so the smoke test sees the same keys as the server
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Pipe obs logs to stdout so the CALL lines are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)

from src.llm.client import ClientMode, LLMClient
from src.llm.provider_map import Provider


# Smallest tractable call per provider — cheap, fast, easy to verify the
# routing without burning real credits.
SMOKE_CALLS = [
    # (label, model_slug, env_var_required_for_direct)
    ("Anthropic  Haiku",      "anthropic/claude-haiku-4-5",   "ANTHROPIC_API_KEY"),
    ("Google     Flash-Lite", "google/gemini-2.5-flash-lite", "GEMINI_API_KEY"),
    ("OpenAI     Nano",       "openai/gpt-5.4-nano",          "OPENAI_API_KEY"),
    # xAI / Grok — uses grok-3-mini (a real, small, cheap xAI model)
    ("xAI        Grok-3-mini","xai/grok-3-mini",              "XAI_API_KEY"),
    # DeepSeek — always goes through OpenRouter (no direct key)
    ("DeepSeek   Flash",      "deepseek/deepseek-v4-flash",   "OPENROUTER_API_KEY"),
]


async def main() -> int:
    client = LLMClient(mode=ClientMode.LIVE)
    print()
    print("─── configured providers ──────────────────────────────")
    for p in Provider:
        present = "✓" if p in client._provider_clients else " "
        print(f"  [{present}] {p.value}")
    print()

    print("─── firing one call per available provider ───────────")
    for label, model, key_env in SMOKE_CALLS:
        if not os.environ.get(key_env, "").strip():
            print(f"  [skip] {label}  — {key_env} not set")
            continue
        try:
            r = await client.call(
                system_prompt="You are a smoke test. Reply with exactly: OK",
                user_message="ping",
                domain="smoke",
                concept="ping",
                model=model,
                max_tokens=8,
                temperature=0.0,
            )
            verdict = "OK" if r.success else f"FAIL ({r.error})"
            print(f"  [{verdict:>4}] {label:<24} {model}")
        except Exception as e:
            print(f"  [FAIL] {label:<24} {model}  → {e!r}")

    print()
    print("─── summary (proof of which key was billed) ──────────")
    client.summarize_calls(request_id="smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
