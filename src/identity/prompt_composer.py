"""
Prompt composer — single canonical entry point for system prompts.

Every model-facing call site in the engine composes its system prompt
through `compose_system_prompt`. The composer prepends the doctrine
header and tags the local mode. Without this, the identity layer is
decorative — wired into a doctrine doc but absent from the prompts
the model actually receives.

Composition shape
=================

```
{SYSTEM_PROMPT_HEADER}

---

MODE: {mode}

---

{local_prompt}
```

The header is the doctrine the model holds while generating. The
`MODE:` line tells the model which sub-task it is serving (cushion
compose, structural match, dossier synthesis, speech, etc.). The
local prompt carries the mode-specific instructions, format
requirements, output schema, and worked examples.

A call site that omits `mode` gets just header + local. A call site
that passes an empty local gets just header + mode. Both are valid;
the most common usage is all three.

Determinism
===========

`compose_system_prompt` is a pure function. No I/O, no clock, no
randomness. Identical inputs produce identical outputs — important
for prompt caching and for snapshot-comparison tests.
"""

from __future__ import annotations

from src.identity.singular_path import SYSTEM_PROMPT_HEADER


# A single horizontal-rule separator used between header / mode / local.
# Chosen for visual clarity in logged prompts and minimal token cost
# (4 characters + newlines).
_SEP = "\n\n---\n\n"


def compose_system_prompt(local_prompt: str | None, *, mode: str | None = None) -> str:
    """Compose the canonical system prompt for a model-facing call.

    Always prepends `SYSTEM_PROMPT_HEADER`. When `mode` is provided,
    tags the prompt with a `MODE: <mode>` line so the model knows
    which sub-task it is serving. When `local_prompt` is provided,
    appends it under a separator.

    Parameters
    ----------
    local_prompt:
        Mode-specific instructions, format requirements, output
        schema, worked examples. Pass None or "" when the call has
        no local prompt (rare).
    mode:
        Short tag naming the sub-task. Examples: "cushion_compose",
        "structural_match", "wandering_dig", "card_articulation",
        "dossier_synthesis", "speech". Free-form string — kept short
        to minimize token cost.

    Returns
    -------
    A single string suitable for passing as `system_prompt` to the
    LLM client.
    """

    parts: list[str] = [SYSTEM_PROMPT_HEADER.strip()]

    if mode and mode.strip():
        parts.append(f"MODE: {mode.strip()}")

    if local_prompt and local_prompt.strip():
        parts.append(local_prompt.strip())

    return _SEP.join(parts)


def carries_identity_header(system_prompt: str) -> bool:
    """True when the supplied system prompt was composed via
    `compose_system_prompt` (or otherwise begins with the canonical
    header).

    Used by integration tests to prove a call site's system prompt is
    going through the composer. Cheap substring check on a sentinel
    line from the header — does not parse the full prompt."""

    if not system_prompt:
        return False
    sentinel = "You are the reasoning core of Constellax."
    return sentinel in system_prompt[:2000]
