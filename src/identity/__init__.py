"""
The Singular Path — identity layer for the Constellax reasoning core.

This package is the doctrine baked into the engine. It is not a wrapper
applied to outputs at the edges; the disciplines exposed here are
imported and called by the parts of the system that make decisions.

Layout
======

  singular_path.py
      Runtime constants. The SYSTEM_PROMPT_HEADER injected into every
      model-facing call. The internal thinking checklist used by the
      synthesizer (silent — not a visible output format). Public types
      shared across disciplines (Goal, Position, Context).

  disciplines/
      Five plain Python modules, one per discipline. Each exposes
      pure functions / dataclasses operating on plain types. No LLM
      calls live here — the disciplines compute on state and produce
      structured judgements that the synthesizer uses to shape prompts
      and to weight or filter findings.

        goal_supremacy        — discriminate claim vs real goal,
                                surface real-goal probe when the
                                stated goal contradicts other signals
        long_horizon          — project a decision across timeframes;
                                flag compounding vs decaying actions
        opportunity_capture   — six-question test on a surfaced opening
        attachment_detection  — name distortion patterns (sunk cost,
                                identity protection, urgency-as-fear,
                                patience-as-avoidance) without
                                diagnosing the user
        resource_conversion   — evaluate convertibility of constraints,
                                criticism, dead-ends, sunk effort

  sovereignty.py
      The three limits — no execution, no argument, no padding — as
      structured rules. The Map-Not-March counter (two-strike state
      keyed by session + position hash) that forces a switch from
      argument to cartography once the user has restated the same
      position twice.

  voice/
      strip.py   — light opener strip (regex, surgical, fail-closed)
      lint.py    — structured lint: real-goal surfaced? failure mode
                   attached? sovereignty preserved? no padding? When
                   the lint fails, the call site regenerates with a
                   stronger directive rather than rewriting the text.

Integration points (wired by the engine, not by this package)
=============================================================

The engine calls into these utilities at named points. The identity
package itself imports nothing from the engine — it has no upward
dependencies. The engine modules import from here.

  cushion compose      — goal_supremacy.surface_real_goal()
                         attachment_detection.scan()
  dossier build        — goal_supremacy.discriminate()
                         opportunity_capture.test()
  per-agent report     — attachment_detection.scan() for failure_mode
  sub-agent spawning   — opportunity_capture.test()
  synthesis layer      — all five disciplines + voice.lint

Per the Constellax operating rule: read / reason / articulate only.
The disciplines surface judgements; the user makes the move.
"""

from src.identity.singular_path import (
    DOCTRINE_NAME,
    DOCTRINE_VERSION,
    RECOVER_GOAL_PROBE,
    SYSTEM_PROMPT_HEADER,
    THINKING_CHECKLIST,
    Context,
    Goal,
    Position,
)
from src.identity.sovereignty import (
    ANTI_LAWS,
    MAP_NOT_MARCH_THRESHOLD,
    AntiLaw,
    MapNotMarchCounter,
    position_hash,
)
from src.identity.prompt_composer import (
    carries_identity_header,
    compose_system_prompt,
)
from src.identity.output_gate import (
    GatedOutput,
    gate_output_async,
    gate_output_sync,
)
from src.identity.exempt import (
    CONTROL_PLANE_SITES,
    ExemptSite,
    exempt_reason,
    is_exempt,
)

__all__ = [
    "DOCTRINE_NAME",
    "DOCTRINE_VERSION",
    "RECOVER_GOAL_PROBE",
    "SYSTEM_PROMPT_HEADER",
    "THINKING_CHECKLIST",
    "Context",
    "Goal",
    "Position",
    "ANTI_LAWS",
    "MAP_NOT_MARCH_THRESHOLD",
    "AntiLaw",
    "MapNotMarchCounter",
    "position_hash",
    "carries_identity_header",
    "compose_system_prompt",
    "GatedOutput",
    "gate_output_async",
    "gate_output_sync",
    "CONTROL_PLANE_SITES",
    "ExemptSite",
    "exempt_reason",
    "is_exempt",
]
