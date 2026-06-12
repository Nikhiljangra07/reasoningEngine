"""
Identity-exempt control-plane registry.

Not every LLM call in the engine produces user-facing prose. Some
calls are pure control-plane: routing decisions, query triage,
structured-JSON visualization payloads, internal wandering critique.
For these, the Singular Path doctrine doesn't apply — the model is
classifying or emitting schema-bound JSON, not speaking to the user.

This module names every such site explicitly. The source-proof test
(`tests/test_identity_source_proof.py`) reads this registry and
exempts the listed (file, prompt-constant) tuples from the
"every-call-must-compose" check. Anything NOT in this registry must
route through `compose_system_prompt`.

Why a registry rather than implicit exemption
=============================================

Implicit exemption ("this call doesn't carry identity because the
output is JSON") drifts. Six months from now, a contributor adds
a new uncomposed call and nobody notices because there is no
forcing function. The registry IS the forcing function:

  - Adding a new call site that isn't composed → source-proof test
    fails until either (a) the call composes, or (b) the
    contributor adds an entry here with a reason.
  - The reason field is required and is read in code review — it
    makes the "why isn't this composed" question impossible to
    skip past.

Adding to this registry
=======================

Only add a site if it is genuinely control-plane:

  - Output is structured JSON consumed by code, not read by user
  - Decision-making call (routing, triage, classification)
  - Internal-only critique / verification / inspection step

If the call produces prose the user reads (or prose downstream
code feeds back to the user), compose it instead. The default
answer to "should this carry the doctrine?" is YES.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExemptSite:
    """One identity-exempt call site.

    `file` is the source path relative to repo root (forward slashes
    on all platforms).
    `prompt_name` is the Python identifier of the prompt constant
    or local variable passed as `system_prompt=...` at the call
    site. The source-proof test matches on this name.
    `reason` is a short clause explaining why this site does not
    need the doctrine header. Required — implicit reasoning rots."""

    file:        str
    prompt_name: str
    reason:      str


CONTROL_PLANE_SITES: tuple[ExemptSite, ...] = (
    ExemptSite(
        file="src/llm/router.py",
        prompt_name="ROUTER_SYSTEM_PROMPT",
        reason=(
            "Formation routing — picks which DOMAINS to activate "
            "via structured JSON. Never reaches the user. Adding the "
            "doctrine header burns tokens on every routing decision "
            "with zero quality impact on the classification output."
        ),
    ),
    ExemptSite(
        file="src/llm/triage.py",
        prompt_name="_GATE_SYSTEM_PROMPT",
        reason=(
            "Triage gate — classifies the user's message into "
            "trivial/direct/direct_plus/deep + extracts MCP needs. "
            "Returns structured JSON consumed by the dispatcher. The "
            "classification itself is internal control flow."
        ),
    ),
    ExemptSite(
        file="src/llm/visualizer.py",
        prompt_name="_VISUAL_GENERATOR_PROMPT",
        reason=(
            "Visualization generator — produces schema-bound JSON "
            "(mermaid, comparison-table, quadrant, score-chart, "
            "flow-graph, knowledge-graph) that the frontend renders. "
            "The rendered chart is user-visible but the prompt is "
            "producing structured data, not natural-language voice."
        ),
    ),
    ExemptSite(
        file="src/wandering/critique.py",
        prompt_name="_CRITIQUE_SYSTEM_PROMPT",
        reason=(
            "Per-iteration wandering self-critique — internal "
            "continue/return/abandon/hand-off decision inside the "
            "agent loop. Output is consumed by the loop's branching "
            "logic, never reaches the user."
        ),
    ),
    ExemptSite(
        file="src/wandering/call_tracker.py",
        prompt_name="system_prompt",
        reason=(
            "AgentScopedLLMClient.call() — pure passthrough wrapper "
            "that forwards the system_prompt parameter verbatim to "
            "the base LLMClient.call. Identity composition happens at "
            "the upstream call site (e.g. _run_dig_iteration, "
            "score_mechanism, score_non_map); composing here would "
            "double-wrap the doctrine header. The wrapper's job is "
            "tagging + per-call audit, not prompt construction."
        ),
    ),
    ExemptSite(
        file="src/wandering/master_synthesizer.py",
        prompt_name="system_prompt",
        reason=(
            "master_synthesizer._call_with_budget() — pure passthrough "
            "helper that wraps client.call with cost-cap enforcement + "
            "per-call audit. The composed doctrine header lives at the "
            "single master_synthesize() call site where "
            "compose_system_prompt(_DOCTRINE_PREAMBLE, mode='master_synthesizer') "
            "runs once and is forwarded into every round. Composing "
            "inside this helper would double-wrap on every R1/R2/R3/R4 "
            "call across both seats."
        ),
    ),
    ExemptSite(
        file="src/wandering/master_sorter.py",
        prompt_name="system_prompt",
        reason=(
            "master_sorter._call_with_budget() — pure passthrough "
            "helper that wraps client.call with cost-cap enforcement + "
            "per-call audit for the sorter tributary. The composed "
            "doctrine header lives at the single master_sort() call site "
            "where compose_system_prompt(_DOCTRINE_PREAMBLE, mode='master_sorter') "
            "runs once and is forwarded into the single sort pass. "
            "Composing inside this helper would double-wrap the doctrine "
            "on every sort call."
        ),
    ),
)


def is_exempt(file: str, prompt_name: str) -> bool:
    """True when (file, prompt_name) matches a registered exempt site."""
    return any(
        s.file == file and s.prompt_name == prompt_name
        for s in CONTROL_PLANE_SITES
    )


def exempt_reason(file: str, prompt_name: str) -> str | None:
    """Return the registered reason for an exempt site, or None when
    the site is not in the registry."""
    for s in CONTROL_PLANE_SITES:
        if s.file == file and s.prompt_name == prompt_name:
            return s.reason
    return None
