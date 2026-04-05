"""
LoRa Deep Reasoning Engine — Interactive CLI.

Run from the project root:
    python run.py

Type your problem. LoRa thinks. LoRa speaks.
Type 'quit' to exit.
"""

import asyncio
import os
import sys
import time

# Load .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key] = val

from src.core.types import (
    Direction,
    Domain,
    FrameworkID,
    Problem,
    Variable,
)
from src.llm.client import LLMClient, ClientMode
from src.llm.engine import run_async_formation
from src.llm.speech import generate_speech, extract_speech_input


def parse_problem(text: str) -> Problem:
    """
    Parse user's raw text into a Problem with auto-extracted variables.

    This is a simple extraction — the real intelligence is in
    Chemistry Self-Assembly which reads the raw text via LLM.
    """
    variables = []

    # Auto-extract basic variables from the text
    # (Chemistry's LLM call will do the real analysis)
    sentences = text.replace(".", ". ").split(". ")
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        lower = sentence.lower()

        # Detect direction from language
        negative_signals = [
            "but", "however", "struggle", "fight", "doubt", "fear",
            "worried", "stuck", "hate", "dread", "can't", "don't",
            "terrified", "frustrated", "angry", "exhausted", "unhappy",
            "unfulfilled", "overwhelm", "anxious", "lost", "trapped",
        ]
        positive_signals = [
            "love", "passionate", "dream", "want", "excited", "enjoy",
            "growing", "improving", "opportunity", "happy", "grateful",
        ]

        neg_count = sum(1 for w in negative_signals if w in lower)
        pos_count = sum(1 for w in positive_signals if w in lower)

        if neg_count > pos_count:
            direction = Direction.NEGATIVE
        elif pos_count > neg_count:
            direction = Direction.POSITIVE
        else:
            direction = Direction.NEUTRAL

        # Estimate magnitude from sentence emphasis
        magnitude = 0.6
        if any(w in lower for w in ["every", "always", "never", "completely", "totally", "extremely"]):
            magnitude = 0.85
        elif any(w in lower for w in ["sometimes", "maybe", "slightly", "a bit"]):
            magnitude = 0.4

        variables.append(Variable(
            name=f"user_statement_{i}",
            description=sentence[:200],
            magnitude=magnitude,
            direction=direction,
            confidence=0.8,
            source_framework=FrameworkID.FIRST_PRINCIPLES,
            is_user_stated=True,
        ))

    return Problem(
        statement=text,
        variables=variables[:8],  # cap at 8 to control token usage
    )


async def run_lora(text: str, client: LLMClient) -> None:
    """Run the full LoRa pipeline on a user's problem."""
    problem = parse_problem(text)

    print()
    print("  LoRa is thinking...")
    print(f"  ({len(problem.variables)} variables extracted)")
    print()

    start = time.monotonic()

    # Run the engine (2 iterations for Phase 1)
    engine_result = await run_async_formation(
        problem=problem,
        client=client,
        max_iterations=2,
    )

    engine_time = time.monotonic() - start

    # Extract speech input
    speech_input = extract_speech_input(
        engine_result=engine_result,
        user_original_text=text,
        is_phase_one=True,
        estimated_additional_credits=15.0,
    )

    # Generate the narrated response
    speech_result = await generate_speech(client, speech_input)

    total_time = time.monotonic() - start

    # Display
    print("  " + "─" * 56)
    print()
    print(speech_result.response_text)
    print()

    if speech_result.dig_deeper_prompt:
        print(f"  [{speech_result.dig_deeper_prompt}]")
        print()

    print("  " + "─" * 56)
    summary = client.get_call_summary()
    print(f"  {summary['total_calls']} calls | "
          f"{summary['total_tokens']['total_tokens']} tokens | "
          f"${summary['estimated_cost_usd']:.2f} | "
          f"{total_time:.1f}s")
    print()


async def main():
    # Determine mode
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if has_key:
        mode = ClientMode.LIVE
        mode_label = "LIVE (Sonnet)"
    else:
        mode = ClientMode.MOCK
        mode_label = "MOCK (no API key — set ANTHROPIC_API_KEY in .env for live mode)"

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║         LoRa Deep Reasoning Engine v2               ║")
    print("  ║         5 domains · 63 concepts · Wu Xing           ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  Mode: {mode_label}")
    print("  Type your problem. LoRa will think and respond.")
    print("  Type 'quit' to exit.")
    print()

    client = LLMClient(mode=mode)

    while True:
        try:
            print("  ┌─ Your problem:")
            text = input("  │ ").strip()

            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye.\n")
                break

            # Support multi-line input (end with empty line)
            while True:
                more = input("  │ ").strip()
                if not more:
                    break
                text += " " + more

            # Reset call log for fresh stats per problem
            client.call_log = []

            await run_lora(text, client)

        except KeyboardInterrupt:
            print("\n\n  Goodbye.\n")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")


if __name__ == "__main__":
    # If a problem is passed as command line argument, run once and exit
    if len(sys.argv) > 1:
        problem_text = " ".join(sys.argv[1:])

        async def single_run():
            has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            mode = ClientMode.LIVE if has_key else ClientMode.MOCK

            print()
            print(f"  Mode: {'LIVE (Sonnet)' if has_key else 'MOCK'}")
            print(f"  Problem: {problem_text[:80]}...")

            client = LLMClient(mode=mode)
            await run_lora(problem_text, client)

        asyncio.run(single_run())
    else:
        # Interactive mode
        asyncio.run(main())
