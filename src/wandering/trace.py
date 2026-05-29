"""
Decision trace — the audit trail of one agent's wandering.

Every step the agent takes is logged: where it looked, why, what it
encountered, whether it dug or moved on, what it kept, what it discarded.
The trace is internal (the user doesn't read it directly) but preserved
for:

  1. Future-session mining (discarded clues marked `possibly_relevant_elsewhere`
     or `revisit_later` can be checked against future anchors)
  2. Debugging the wandering policy when something feels off
  3. Reverse engineering a session's path if a breakthrough lands and the
     user wants to see HOW the system got there

Per Law 4: traces are read-only artifacts. Wandering Room never edits the
user's project state based on trace contents.

ISOLATION: imports nothing wandering-internal except types. No LLM, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Step kinds — what the agent did at this point in the trace
# ---------------------------------------------------------------------------


class StepKind(str, Enum):
    """What kind of move the agent made at this step."""

    INITIALIZED = "initialized"           # agent created with anchor + budget
    FETCHED = "fetched"                   # pulled content from a source
    MATCHED = "matched"                   # checked content against cushion
    DUG = "dug"                           # entered a multi-iteration deep dive
    REPORTED = "reported"                 # produced an ExplorationReport
    SELF_CRITIQUED = "self_critiqued"     # ran the six-question check
    RETURNED_TO_ANCHOR = "returned_to_anchor"  # critique triggered re-orientation
    SPAWNED_SUBAGENT = "spawned_subagent"  # called the spawn tool
    ABANDONED = "abandoned"               # closed dig early on red critique
    MOVED_ON = "moved_on"                 # picked new direction
    EXHAUSTED = "exhausted"               # time/token budget reached


# ---------------------------------------------------------------------------
# Discard classification — preserve, don't delete
# ---------------------------------------------------------------------------


class DiscardKind(str, Enum):
    """How a clue was discarded. Classified, not deleted.

    The compounding-asset principle: a clue that's off-topic NOW may
    resonate with a FUTURE anchor. Future sessions check the discarded
    shelf against new anchors and may surface previously-discarded clues.
    """

    DISCARDED_FOR_CURRENT_ANCHOR = "discarded_for_current_anchor"
    POSSIBLY_RELEVANT_ELSEWHERE = "possibly_relevant_elsewhere"
    REVISIT_LATER = "revisit_later"


@dataclass
class DiscardedClue:
    """One clue an agent encountered but decided not to pursue.

    Preserved with a classification so future sessions can mine it.
    """

    description: str  # one-line description of the clue
    source_hint: str = ""  # where it was (URL / domain / source title)
    classification: DiscardKind = DiscardKind.DISCARDED_FOR_CURRENT_ANCHOR
    reason: str = ""  # why the agent decided to discard
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Trace step — one entry in the audit log
# ---------------------------------------------------------------------------


@dataclass
class TraceStep:
    """One step in an agent's decision trail.

    Steps are append-only. The trace is the linear story of what happened.
    """

    step_id: int  # monotonic per-agent (0, 1, 2, ...)
    kind: StepKind
    timestamp: float = 0.0

    # Optional fields, varying by kind
    position: str = ""       # current "where am I" (domain / URL / topic)
    rationale: str = ""      # why this step was taken
    detail: str = ""         # short free-form payload
    matched_count: int = 0   # for MATCHED steps
    iterations_used: int = 0  # for DUG steps
    report_id: str = ""      # for REPORTED steps
    subagent_id: str = ""    # for SPAWNED_SUBAGENT steps
    tokens_spent: int = 0    # cumulative at this step


# ---------------------------------------------------------------------------
# Decision trace — the full per-agent audit log
# ---------------------------------------------------------------------------


@dataclass
class DecisionTrace:
    """The full audit log of one agent's session.

    Append-only. Includes the trace steps AND the classified discarded
    shelf. Both are preserved at session end (Neo4j storage, deferred).
    """

    agent_id: str
    anchor_summary: str = ""  # one-line user problem for display
    steps: list[TraceStep] = field(default_factory=list)
    discarded_clues: list[DiscardedClue] = field(default_factory=list)

    # Aggregates filled in as the trace grows
    total_tokens_spent: int = 0
    total_reports_produced: int = 0
    total_subagents_spawned: int = 0

    # Final state
    completion_reason: str = ""  # "exhausted_budget", "completed", "abandoned"
    ended_at: float = 0.0

    def append(self, step: TraceStep) -> None:
        """Add a step to the trace. Sets step_id automatically."""
        step.step_id = len(self.steps)
        if step.tokens_spent > self.total_tokens_spent:
            self.total_tokens_spent = step.tokens_spent
        if step.kind == StepKind.REPORTED:
            self.total_reports_produced += 1
        if step.kind == StepKind.SPAWNED_SUBAGENT:
            self.total_subagents_spawned += 1
        self.steps.append(step)

    def discard(self, clue: DiscardedClue) -> None:
        """Mark a clue as discarded (classified, not deleted)."""
        self.discarded_clues.append(clue)

    def last_step(self) -> TraceStep | None:
        return self.steps[-1] if self.steps else None

    def step_count(self) -> int:
        return len(self.steps)

    def steps_of(self, kind: StepKind) -> list[TraceStep]:
        """Filter steps by kind. Used for quick stats / debugging."""
        return [s for s in self.steps if s.kind == kind]


__all__ = [
    "StepKind",
    "DiscardKind",
    "DiscardedClue",
    "TraceStep",
    "DecisionTrace",
]
