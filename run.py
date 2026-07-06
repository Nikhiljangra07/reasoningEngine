"""
Constellax Reasoning Engine — Interactive CLI.

Run from the project root:
    python run.py

Type your problem. Constellax thinks. Constellax speaks.
Type 'quit' to exit.
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from src.core.types import (
    Direction,
    Domain,
    FrameworkID,
    Problem,
    Variable,
)
from src.llm.client import LLMClient, ClientMode
from src.llm.effort import Effort, iterations_for, normalize_effort
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


async def run_constellax(text: str, client: LLMClient, effort: Effort = Effort.MEDIUM) -> None:
    """Run the full Constellax pipeline on a user's problem."""
    problem = parse_problem(text)

    max_iters = iterations_for(effort)

    print()
    print(f"  Constellax is thinking... (effort={effort.value}, iterations={max_iters})")
    print(f"  ({len(problem.variables)} variables extracted)")
    print()

    start = time.monotonic()

    engine_result = await run_async_formation(
        problem=problem,
        client=client,
        max_iterations=max_iters,
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


def _resolve_mode_and_effort() -> tuple[ClientMode, str, Effort]:
    """Pick LIVE/MOCK and the default effort tier from the environment."""
    has_key = bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if has_key:
        mode = ClientMode.LIVE
        mode_label = "LIVE (OpenRouter)"
    else:
        mode = ClientMode.MOCK
        mode_label = "MOCK (no API key — set OPENROUTER_API_KEY in .env for live mode)"
    effort = normalize_effort((os.environ.get("CONSTELLAX_EFFORT") or os.environ.get("LORA_EFFORT")))
    return mode, mode_label, effort


async def main(effort: Effort):
    mode, mode_label, _ = _resolve_mode_and_effort()

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║         Constellax Reasoning Engine v2              ║")
    print("  ║         5 domains · 63 concepts · Wu Xing           ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  Mode: {mode_label}")
    print(f"  Effort: {effort.value} (iterations={iterations_for(effort)})")
    print("  Type your problem. Constellax will think and respond.")
    print("  Type 'quit' to exit. Type 'effort low|medium|high' to switch tiers.")
    print()

    client = LLMClient(mode=mode)
    current_effort = effort

    while True:
        try:
            print("  ┌─ Your problem:")
            text = input("  │ ").strip()

            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye.\n")
                break

            # Effort switch shortcut: "effort low" / "effort medium" / "effort high"
            lower = text.lower()
            if lower.startswith("effort "):
                tier = lower.split(None, 1)[1].strip()
                current_effort = normalize_effort(tier)
                print(f"  → effort={current_effort.value} "
                      f"(iterations={iterations_for(current_effort)})\n")
                continue

            # Support multi-line input (end with empty line)
            while True:
                more = input("  │ ").strip()
                if not more:
                    break
                text += " " + more

            # Reset call log for fresh stats per problem
            client.call_log = []

            await run_constellax(text, client, effort=current_effort)

        except KeyboardInterrupt:
            print("\n\n  Goodbye.\n")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")


def _parse_cli_args(argv: list[str]) -> tuple[str, Effort]:
    """
    Pull `--effort low|medium|high` out of argv, return (problem_text, effort).

    Effort comes from the flag if present, otherwise from LORA_EFFORT env,
    otherwise DEFAULT_EFFORT (medium).
    """
    effort = normalize_effort((os.environ.get("CONSTELLAX_EFFORT") or os.environ.get("LORA_EFFORT")))
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--effort" and i + 1 < len(argv):
            effort = normalize_effort(argv[i + 1])
            i += 2
            continue
        if token.startswith("--effort="):
            effort = normalize_effort(token.split("=", 1)[1])
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return " ".join(cleaned), effort


if __name__ == "__main__":
    # If a problem is passed as command line argument, run once and exit.
    # Usage: python run.py "my problem" --effort high
    if len(sys.argv) > 1:
        problem_text, cli_effort = _parse_cli_args(sys.argv[1:])

        async def single_run():
            mode, mode_label, _ = _resolve_mode_and_effort()
            print()
            print(f"  Mode: {mode_label}")
            print(f"  Effort: {cli_effort.value} (iterations={iterations_for(cli_effort)})")
            print(f"  Problem: {problem_text[:80]}...")

            client = LLMClient(mode=mode)
            await run_constellax(problem_text, client, effort=cli_effort)

        asyncio.run(single_run())
    else:
        # Interactive mode — effort from env, switchable mid-session via "effort <tier>"
        env_effort = normalize_effort((os.environ.get("CONSTELLAX_EFFORT") or os.environ.get("LORA_EFFORT")))
        asyncio.run(main(env_effort))
