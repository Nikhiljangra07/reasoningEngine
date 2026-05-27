"""
memory_injection — builds the markdown "PRIOR MEMORY" block injected into
LLM system prompts so the model has cross-thread / cross-platform context.

ROLE
====
Bridges the MemoryRetriever (Phase 4) and the dispatcher (Phase 4 wiring).
One async function: `build_memory_directive(question, user_id, ...)`. It
runs the retriever, renders the timeline, and wraps it in a directive
header that the system prompt can append verbatim.

DESIGN
======
- One call site: the trace endpoint in server.py builds the directive
  before invoking dispatch().
- The directive is a string — empty string means "no memory available,
  do nothing." The dispatch path appends iff non-empty.
- Failures degrade silently: any exception in the retriever or renderer
  produces an empty directive. The user's question still gets answered;
  the model just lacks past-context awareness for this turn.
- Cost: one Gemini embedding call (~$0.00002) + one Cypher vector query
  per trace. Trivially cheap, runs on every substantive request.

WHAT THE LLM SEES
=================
When memory is found, the model gets a section like:

    ## PRIOR MEMORY
    (Decision Trace recall — the user's accumulated context across all
    threads. Use this to ground your answer in past decisions; never
    fabricate references that aren't here.)

    [render_timeline output — CURRENT THREAD + CROSS-THREAD sections]

When memory is empty/unavailable, the directive is empty string — no
prompt change at all, so the model's behavior is identical to pre-Phase-4.
"""

from __future__ import annotations

import logging

from src.bridge.memory_retriever import MemoryRetriever, render_timeline

log = logging.getLogger("constellax.memory_injection")


_HEADER = (
    "## PRIOR MEMORY\n"
    "(Decision Trace recall — the user's accumulated context across all "
    "threads they've had with Constellax. Use this to ground your answer "
    "in past decisions; never fabricate references that aren't here. "
    "When you cite past decisions, name the thread title so the user "
    "knows which prior conversation you're referencing.)\n"
)


async def build_memory_directive(
    question: str,
    *,
    retriever: MemoryRetriever | None,
    user_id: str | None,
    thread_id: str | None = None,
    project_id: str | None = None,
    workspace_id: str | None = None,
    surface_id: str | None = None,
    k_local: int = 5,
    k_cross: int = 5,
) -> str:
    """Build the markdown PRIOR MEMORY block for one request.

    Returns "" when:
      - the retriever isn't available (Neo4j not configured)
      - user_id is missing (cross-user isolation hard-fail)
      - the retrieve call raises (logged, swallowed)
      - the result is empty (no local turns, no cross-thread matches)

    The caller appends the non-empty return value to the system prompt
    (or to extra_directives). On empty return, the caller does nothing —
    the trace endpoint behaves identically to pre-Phase-4."""
    if retriever is None:
        return ""
    if not user_id:
        # Don't query without a user — would cross-contaminate memory.
        return ""
    try:
        result = await retriever.retrieve(
            question or "",
            user_id=user_id,
            thread_id=thread_id,
            project_id=project_id,
            workspace_id=workspace_id,
            surface_id=surface_id,
            cross_thread=True,
            k_local=k_local,
            k_cross=k_cross,
        )
    except Exception as e:
        log.warning("memory_injection: retrieve failed (%s) — empty directive", e)
        return ""

    if not result.local and not result.cross_thread:
        # Nothing to inject. Don't pad the prompt with an empty section header.
        return ""

    return _HEADER + "\n" + render_timeline(result, show_provenance=True)
