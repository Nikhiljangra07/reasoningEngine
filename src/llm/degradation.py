"""
Graceful Degradation — Three Levels.

Level 1: Agent Timeout or Malformed Response
  Retry once → skip concept → continue with remaining → log failure

Level 2: Full Domain Failure
  All concepts in a domain failed → skip domain in Wu Xing cycles
  → credits not charged → user informed → confidence reduced

Level 3: Multiple Domain Failure (3+ domains down)
  Switch to degraded mode → run whatever's available
  → entire response FREE → free retry token → confidence significantly reduced

CRITICAL: Never expose internal terminology to users.
No "Physics island unreachable." No "Ke cycle failed."
Tell the user what it means for THEM, not what happened internally.

ISOLATION: Pure logic. No domain imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DegradationLevel(Enum):
    """Current degradation level of the engine."""
    NONE = "none"               # everything working
    LEVEL_1 = "level_1"         # some concepts skipped
    LEVEL_2 = "level_2"         # a full domain is down
    LEVEL_3 = "level_3"         # 3+ domains down


@dataclass
class FailureRecord:
    """Record of a single failure."""
    domain: str
    concept: str
    error: str
    attempt: int                # which attempt failed (1 or 2)
    recoverable: bool           # was it recovered by retry?


@dataclass
class DegradationState:
    """Current state of the degradation system."""
    level: DegradationLevel
    failures: list[FailureRecord]
    domains_down: list[str]
    concepts_skipped: list[str]
    confidence_reduction: float     # 0.0 to 1.0 — multiply original confidence by this
    user_message: str               # human-facing message, NO internal terminology
    free_response: bool
    free_retry_issued: bool


class DegradationTracker:
    """
    Tracks failures and determines degradation level.

    Usage:
        tracker = DegradationTracker()
        tracker.record_failure("physics", "first_principles", "timeout")
        state = tracker.get_state()
    """

    def __init__(self):
        self.failures: list[FailureRecord] = []
        self._domain_concept_counts: dict[str, int] = {}
        self._domain_total_concepts: dict[str, int] = {}
        self._domain_failures: dict[str, int] = {}

    def set_domain_concept_count(self, domain: str, total_concepts: int) -> None:
        """Register how many concepts a domain has (to detect full domain failure)."""
        self._domain_total_concepts[domain] = total_concepts
        self._domain_concept_counts[domain] = 0
        self._domain_failures[domain] = 0

    def record_failure(
        self,
        domain: str,
        concept: str,
        error: str,
        attempt: int = 1,
        recoverable: bool = False,
    ) -> None:
        """Record a concept-level failure."""
        self.failures.append(FailureRecord(
            domain=domain,
            concept=concept,
            error=error,
            attempt=attempt,
            recoverable=recoverable,
        ))

        if not recoverable:
            self._domain_failures[domain] = self._domain_failures.get(domain, 0) + 1

    def record_success(self, domain: str, concept: str) -> None:
        """Record a successful concept execution."""
        self._domain_concept_counts[domain] = self._domain_concept_counts.get(domain, 0) + 1

    def get_state(self) -> DegradationState:
        """Calculate current degradation state from all recorded failures."""

        # Identify fully failed domains
        domains_down = []
        for domain, total in self._domain_total_concepts.items():
            failures = self._domain_failures.get(domain, 0)
            successes = self._domain_concept_counts.get(domain, 0)
            if total > 0 and failures >= total and successes == 0:
                domains_down.append(domain)

        # Identify skipped concepts (failed but domain is still up)
        concepts_skipped = [
            f"{f.domain}:{f.concept}"
            for f in self.failures
            if not f.recoverable and f.domain not in domains_down
        ]

        # Determine level
        if len(domains_down) >= 3:
            level = DegradationLevel.LEVEL_3
        elif len(domains_down) >= 1:
            level = DegradationLevel.LEVEL_2
        elif concepts_skipped:
            level = DegradationLevel.LEVEL_1
        else:
            level = DegradationLevel.NONE

        # Confidence reduction
        confidence_map = {
            DegradationLevel.NONE: 1.0,
            DegradationLevel.LEVEL_1: 0.9,      # minor: some angles missing
            DegradationLevel.LEVEL_2: 0.7,       # moderate: a full domain down
            DegradationLevel.LEVEL_3: 0.4,       # significant: limited analysis
        }
        confidence_reduction = confidence_map[level]

        # User message — NO INTERNAL TERMINOLOGY
        user_message = _build_user_message(level, domains_down, concepts_skipped)

        # Free response policy
        free_response = level == DegradationLevel.LEVEL_3
        free_retry = level == DegradationLevel.LEVEL_3

        return DegradationState(
            level=level,
            failures=self.failures,
            domains_down=domains_down,
            concepts_skipped=concepts_skipped,
            confidence_reduction=confidence_reduction,
            user_message=user_message,
            free_response=free_response,
            free_retry_issued=free_retry,
        )


def _build_user_message(
    level: DegradationLevel,
    domains_down: list[str],
    concepts_skipped: list[str],
) -> str:
    """
    Build user-facing message. ZERO internal terminology.
    Tell them what it means for THEM, not what happened internally.
    """
    if level == DegradationLevel.NONE:
        return ""

    if level == DegradationLevel.LEVEL_1:
        return (
            "This response covers your situation thoroughly, though some "
            "angles couldn't be fully explored this time. The core analysis "
            "is solid."
        )

    if level == DegradationLevel.LEVEL_2:
        return (
            "This response covers most of your situation, but some perspectives "
            "couldn't be fully explored this time. Your credits were adjusted "
            "accordingly. The analysis is still valuable — just not as comprehensive "
            "as it could be."
        )

    if level == DegradationLevel.LEVEL_3:
        return (
            "We ran into some issues during this analysis. The response below "
            "is based on limited analysis. This one's on us — no credits charged, "
            "and your next deep analysis is free."
        )

    return ""
