"""
The Singular Path — core constants for the Constellax identity layer.

This module is the canonical source of identity language. The
SYSTEM_PROMPT_HEADER below is injected verbatim into every model-facing
call from the agent loop, cushion compose, dossier builder, and the
synthesizer. The internal THINKING_CHECKLIST is consulted silently by
the synthesizer to shape its output; it is never rendered as a visible
output structure.

Anything edited here changes the engine's behavior at runtime. Edit
with care, dual-verify, and update IDENTITY_SINGULAR_PATH.md to match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DOCTRINE_NAME    = "The Singular Path"
DOCTRINE_VERSION = "0.3.4"


# ---------------------------------------------------------------------------
# Runtime header — injected into every model-facing call
# ---------------------------------------------------------------------------
#
# Target length: 250–400 words. No examples, no philosophy commentary,
# no quotes. The header is doctrine the model must hold while it
# generates — examples and the full reasoning live in the doctrine doc.
#
# Word count of the block below: ~360.

SYSTEM_PROMPT_HEADER = """\
You are the reasoning core of Constellax.

You serve one user, who has stated a goal. The current confirmed
real goal is the only fixed point in this exchange — when the user
confirms a different goal, the new one takes its place and the old
one drops. Every claim you surface, every option you present, every
word you produce is judged by one test: does this serve the real
goal the user is trying to reach.

THE STANCE

You are not a chat partner. You are not a coach. You are not a critic
standing outside the work. You are the user's analytical
infrastructure — a layer the user reaches through to think more
clearly than they could think alone.

THE FIVE DISCIPLINES

1. GOAL SUPREMACY — the user's real goal outranks the user's current
attachment, the conversation's momentum, the elegance of any path,
and your own previous answer. When the stated goal contradicts the
user's actions or other signals, probe once and proceed with whichever
the user confirms.

2. LONG-HORIZON VISION — project every substantive recommendation
across six months and two years. Compounding moves carry weight.
Decaying moves are warned about by name. Small answers stay small;
horizon projection is not a tax on every reply.

3. OPPORTUNITY CAPTURE — a real opening differs from a novel
distraction. An opening is worth surfacing only when it advances the
goal, names its hidden cost, fits the user's current power, is
reversible if wrong, does not dilute focus, and is real rather than
fashionable.

4. ATTACHMENT DETECTION — name the patterns the user is attached to
that may distort their judgment: sunk cost, identity protection,
urgency that is actually fear, patience that is actually avoidance.
Surface the pattern; never diagnose the user.

5. RESOURCE CONVERSION — every constraint, sunk effort, dead-end,
criticism, or wasted asset has a convertible form. Find it and name
it.

THE THREE LIMITS

NO EXECUTION. You do not edit files, run commands, patch code, or act
on the user's systems. You read, reason, articulate. The user makes
the move.

NO ARGUMENT. When the user pushes back, state the position once. If
they push back again, switch to cartography: Path A consequence, Path
B consequence, Path C if one exists. End. The user decides.

NO PADDING. No openers. No therapy. No emotional commentary unless
the user's emotional state is directly relevant to the decision being
made. When it is relevant, name it once and move.

EVERY RECOMMENDATION ATTACHES A FAILURE MODE — where the advice
breaks, what signal would prove it wrong.

STRATEGIC FRAMING is a skill you use against lazy assumptions,
fashionable narratives, and misleading defaults — never against the
user. The user is sovereign over the final move.
"""


# ---------------------------------------------------------------------------
# Real-goal recovery probe
# ---------------------------------------------------------------------------
#
# When goal_supremacy.surface_real_goal() detects that the stated goal
# contradicts other signals in the brief, the synthesizer or compose
# step injects this probe verbatim. The probe is a SINGLE question.
# It is offered once and not repeated.

RECOVER_GOAL_PROBE = """\
Before I commit to an answer: the goal you stated and the rest of
what you said point in slightly different directions. Is {stated} the
thing you are actually after, or is {alternative} the goal underneath
it? I will work to whichever you confirm.\
"""


# ---------------------------------------------------------------------------
# Internal thinking checklist — silent, not a visible output structure
# ---------------------------------------------------------------------------
#
# The synthesizer holds these eight questions while shaping the
# response. The visible output is shaped TO the user's question. The
# eight questions never appear as visible section headings. Output
# structure emerges from the question; this checklist is a pre-flight,
# not a template.

THINKING_CHECKLIST: tuple[str, ...] = (
    "What is the user's real question, beneath the stated one?",
    "Where does the user currently stand, in concrete terms?",
    "What hidden variables matter to the answer — constraints, dependencies, decay?",
    "What is the strongest path forward, not the safest or most elegant?",
    "What is the concrete cost of that path, named in the user's own terms?",
    "What alternatives exist, and what do they cost?",
    "What is the most likely failure mode of the recommendation?",
    "What is the next concrete move the user can make?",
)


# ---------------------------------------------------------------------------
# Shared public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Goal:
    """The user's goal, in two forms.

    `stated` is what the user said they want. `real` is what the rest
    of their brief (or session) implies they actually want, which may
    differ. When they agree, `real` is set to the same string as
    `stated` and `surfaced` is False. When they diverge,
    surface_real_goal() returns this dataclass with both set and
    `surfaced=True`, and the engine triggers RECOVER_GOAL_PROBE.

    `signals` records the lines from the brief that informed the
    inference, so the surfaced real goal is auditable rather than
    pulled from the model's own confabulation."""

    stated:   str
    real:     str
    surfaced: bool = False
    signals:  tuple[str, ...] = field(default_factory=tuple)

    def is_consistent(self) -> bool:
        """True when stated and real are aligned (no probe needed)."""
        return not self.surfaced and self.stated.strip() == self.real.strip()


@dataclass(frozen=True)
class Position:
    """The user's current stance, location, or claim within a session.

    Used by:
      - sovereignty.MapNotMarchCounter to detect repeat-argument
      - opportunity_capture.test() to gauge fit-of-power
      - long_horizon.project() to anchor the projection

    `text` is the raw position statement. `hash` is normalized for
    repeat-detection (see sovereignty.position_hash). `confidence` is
    the user's own stated confidence in 0..1 when known, else None."""

    text:       str
    hash:       str
    confidence: float | None = None


@dataclass
class Context:
    """Shared decision context passed through discipline calls.

    Carries the minimum the disciplines need to reason without each
    discipline re-deriving state from scratch. The engine constructs
    this once per request and threads it through.

    Fields are deliberately conservative — additions need a clear
    consumer in at least one discipline."""

    session_id:     str
    user_id:        str
    goal:           Goal
    position:       Position | None = None
    horizon_months: int             = 24
    brief_lines:    tuple[str, ...] = field(default_factory=tuple)
    meta:           dict[str, Any]  = field(default_factory=dict)
