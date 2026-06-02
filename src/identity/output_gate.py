"""
Output gate — strip → lint → regenerate-on-fail → final.

The gate is the runtime counterpart of the doctrine. Every prose
output the user reads should pass through it. The gate:

  1. Strips obvious openers (`voice.strip`).
  2. Runs the structured lint (`voice.lint`).
  3. If the lint blocks, builds a remediation directive and runs the
     supplied regenerate function once.
  4. Re-strips and re-lints the regenerated draft.
  5. Returns a `GatedOutput` with the final text plus telemetry.

The gate does NOT call the LLM directly. The caller supplies a
`regenerate_fn` — a small closure that knows how to call its specific
LLM client with the augmented system prompt. This keeps the gate
free of LLM-client coupling and lets sync/async/mock-LLM call sites
all use the same gate.

Telemetry
=========

Every gate run produces a `GatedOutput` with `attempts`,
`regenerated`, `initial_lint`, and `final_lint` fields. The caller
increments an `identity_regenerate` operational counter when
`regenerated` is True; aggregated, that tells us how often the
doctrine is being violated and whether the runtime header needs to
tighten.

Sync vs async
=============

Two entrypoints — `gate_output_async` for async LLM clients and
`gate_output_sync` for sync paths. Both share the same logic via a
private helper. Most wandering call sites are async; speech.py and
some tests are sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from src.identity.voice.lint import (
    LintContext,
    LintResult,
    build_regenerate_directive,
    lint,
    should_regenerate,
)
from src.identity.voice.strip import strip_openers


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

#: A synchronous regeneration callable. Receives a directive string to
#: append to the system prompt; returns the next draft.
RegenerateSyncFn = Callable[[str], str]

#: An asynchronous regeneration callable. Same contract, async.
RegenerateAsyncFn = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class GatedOutput:
    """Result of running text through the output gate.

    `text` is the final output — already stripped, already lint-checked.
    `regenerated` is True when the lint blocked the initial draft and a
    second draft was produced.
    `attempts` is the number of model calls actually made (1 or 2).
    `initial_lint` is the lint result of the FIRST draft, for telemetry.
    `final_lint` is the lint result of the FINAL text — should always
        be `passed=True` except in the rare case where the regenerate
        ALSO failed (then `final_lint.passed` is False and the caller
        decides what to do with the imperfect output).
    `directive_used` is the regenerate directive that was passed to
        `regenerate_fn`, when regeneration fired — empty otherwise."""

    text:            str
    regenerated:     bool
    attempts:        int
    initial_lint:    LintResult
    final_lint:      LintResult
    directive_used:  str = ""


# ---------------------------------------------------------------------------
# Sync entrypoint
# ---------------------------------------------------------------------------

def gate_output_sync(
    initial_text: str,
    *,
    regenerate_fn: RegenerateSyncFn,
    context: LintContext | None = None,
    max_attempts: int = 2,
) -> GatedOutput:
    """Run text through the identity gate (sync version).

    Parameters
    ----------
    initial_text:
        The model's first draft. Already produced — gate does not
        call the LLM for the initial draft.
    regenerate_fn:
        Callable taking a remediation directive and returning a new
        draft. The caller composes the directive into the system
        prompt of the next LLM call.
    context:
        LintContext carrying `real_goal_surfaced`,
        `map_not_march_strike`, `emotional_state_relevant`. None →
        default context (no flags set).
    max_attempts:
        Total number of attempts permitted (1 disables regenerate,
        2 allows one regenerate). Hard ceiling at 2 — there is no
        case where a third attempt is appropriate; if two strong
        directives both failed, the caller should treat that as a
        signal to investigate the prompt, not to call again.

    Returns
    -------
    A `GatedOutput`."""

    cleaned = strip_openers(initial_text)
    initial = lint(cleaned, context)

    if initial.passed or max_attempts < 2:
        return GatedOutput(
            text=cleaned,
            regenerated=False,
            attempts=1,
            initial_lint=initial,
            final_lint=initial,
        )

    directive = build_regenerate_directive(initial)
    second_raw = regenerate_fn(directive)
    second_cleaned = strip_openers(second_raw)
    second = lint(second_cleaned, context)

    return GatedOutput(
        text=second_cleaned,
        regenerated=True,
        attempts=2,
        initial_lint=initial,
        final_lint=second,
        directive_used=directive,
    )


# ---------------------------------------------------------------------------
# Async entrypoint
# ---------------------------------------------------------------------------

async def gate_output_async(
    initial_text: str,
    *,
    regenerate_fn: RegenerateAsyncFn,
    context: LintContext | None = None,
    max_attempts: int = 2,
) -> GatedOutput:
    """Run text through the identity gate (async version).

    See `gate_output_sync` for parameter semantics. The only
    difference is that `regenerate_fn` is async."""

    cleaned = strip_openers(initial_text)
    initial = lint(cleaned, context)

    if initial.passed or max_attempts < 2:
        return GatedOutput(
            text=cleaned,
            regenerated=False,
            attempts=1,
            initial_lint=initial,
            final_lint=initial,
        )

    directive = build_regenerate_directive(initial)
    second_raw = await regenerate_fn(directive)
    second_cleaned = strip_openers(second_raw)
    second = lint(second_cleaned, context)

    return GatedOutput(
        text=second_cleaned,
        regenerated=True,
        attempts=2,
        initial_lint=initial,
        final_lint=second,
        directive_used=directive,
    )
