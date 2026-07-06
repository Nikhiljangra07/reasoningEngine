"""
session_state — per-wander shared state across all agents in a session.

Every wander spawns N agents. Without shared state, two agents can fetch
the same URL ten minutes apart and waste their entire budget on duplicate
content; or a strong findSimilar hop discovered by agent P03 can't reach
agent P07 because there's no place to leave it.

This module is the answer: ONE SessionState per Wandering Room session,
passed by reference into every agent, every fetcher, every policy call.

WHAT IT HOLDS (today):
  - visited_urls: set of URLs any agent in this session has already
    fetched. Consult before issuing a new request; skip duplicates.
  - followon_queue: priority queue of URLs that ONE agent picked
    via tier-2 page reading + link scoring, that ANY agent can pull
    on its next outer loop. This is how cross-agent serendipity
    travels.

WHAT IT DELIBERATELY DOES NOT HOLD:
  - Match results from other agents. Per Law 1 (chaos is the feature)
    agents should NOT share findings mid-flight — that turns wandering
    into consensus optimization. Findings are reconciled at synthesis
    time, after every agent's wander is complete.
  - Domain visit counts across agents. Per-agent inverse-visit weighting
    stays per-agent so chaos diversity is preserved. Two agents both
    wandering "physics" is fine; they'll hit different sources.

CONCURRENCY:
  All mutations go through `asyncio.Lock`. Reads are unlocked (Python
  set/list iteration over a snapshot is safe enough at our scale; we're
  not in a high-contention path). The lock cost is small — agents fetch
  every 5-30 seconds, not every millisecond.

ISOLATION:
  No LLM, no I/O, no persistence. Pure in-memory dataclass.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit


log = logging.getLogger("constellax.wandering.session_state")


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Return a canonical form of `url` for dedup comparison.

    Lower-cases scheme + host, strips trailing slash on path-only URLs,
    drops fragments. Query strings are preserved (often material:
    `?id=42` vs `?id=43` are different pages). Empty input returns "".

    This is intentionally light — we don't want two URLs that the user
    would consider "the same page" to be treated as different. Anything
    more aggressive (stripping utm_*, sorting query keys) risks the
    opposite mistake on rare sites where order matters.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()

    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    path = parts.path or ""
    # Drop trailing slash on the root path; keep elsewhere because some
    # CMSes serve different content at `/a/` vs `/a`.
    if path == "/":
        path = ""
    query = parts.query
    fragment = ""  # always dropped
    return urlunsplit((scheme, netloc, path, query, fragment))


# ---------------------------------------------------------------------------
# Follow-on queue item
# ---------------------------------------------------------------------------


@dataclass
class FollowonItem:
    """One URL queued for follow-on fetching.

    Created when an agent's tier-2 page read surfaces a high-quality link
    (via the same matcher LLM scoring link anchor + surrounding text), or
    when Exa.findSimilar produces a strong neighbor.

    `score` is a 0..1 relative priority — higher is dequeued first. We
    keep it simple: per-source heuristics decide the score (link-from-
    tier-2 vs findSimilar-from-hit) and the queue sorts on it.

    `parent_url` is the URL the link came from — used in the trace for
    audit and to avoid follow-on cycles (if A links to B and B links
    back to A, the second follow-on is a no-op via visited_urls).

    `origin` distinguishes the path that produced the item: useful for
    diagnostics (`"link"` vs `"findsimilar"`) and for budget caps if we
    later want to bound either source independently.
    """

    url: str
    score: float = 0.5
    parent_url: str = ""
    origin: str = "link"  # "link" | "findsimilar"

    def __post_init__(self) -> None:
        self.url = normalize_url(self.url)
        self.parent_url = normalize_url(self.parent_url)
        # Clamp to [0,1] so the priority queue stays well-behaved.
        if self.score < 0.0:
            self.score = 0.0
        elif self.score > 1.0:
            self.score = 1.0


# ---------------------------------------------------------------------------
# Agent noticeboard (WANDER_AGENT_NOTICEBOARD test scaffold — June 2026)
# ---------------------------------------------------------------------------
#
# Append-only heads-up board. After a successful dig, an agent posts a short
# notice — domain, match strength, one-sentence finding, direction pointed —
# so other agents can SEE what's been covered before picking their next
# domain. Informational only: notices do NOT enter dig content (the dig
# itself remains independent per Law 1) and do NOT trigger cross-agent
# critique. The intent is coverage awareness, not mid-flight consensus.
#
# Safety properties (vs the rejected real-time chat design):
#   - Notices are POSTED only after a dig completes — agents can't see
#     peers' in-progress findings.
#   - Notices are READ only at next-domain-pick time — they influence
#     which domain an agent walks into, not what bridges it finds there.
#   - Each report remains an independent sample for the synthesizer.
#   - The noticeboard is append-only; nothing gets edited or retracted.
#
# Failure modes to watch for (will surface in A/B against r6):
#   - Premature claim: thin dig posts "I covered X" → others avoid X →
#     under-coverage. Mitigation: match_strength in the notice so others
#     can judge value; notices are informational, not exclusive.
#   - Domain crowding inverse: agents over-correct toward never-covered
#     domains, missing complementary dives. Mitigation: domain-pick
#     policy treats notices as soft hints, not hard exclusions.


@dataclass
class AgentNotice:
    """One heads-up notice posted to the session noticeboard.

    Posted by an agent after a successful dig produces a report, before
    the agent picks its next domain. Read by other agents at their own
    next-domain-pick step. Pure informational — never enters dig content.
    """

    agent_id:        str
    domain:          str
    match_strength:  str   # "weak" | "moderate" | "strong" — derived from total_matched_nodes
    summary:         str   # one-sentence finding (≤ 200 chars)
    principle:       str   # load-bearing principle (≤ 200 chars)
    direction:       str   # where this points other agents (≤ 200 chars)
    timestamp:       float  # epoch seconds when posted

    def to_dict(self) -> dict[str, str | float]:
        return {
            "agent_id":       self.agent_id,
            "domain":         self.domain,
            "match_strength": self.match_strength,
            "summary":        self.summary,
            "principle":      self.principle,
            "direction":      self.direction,
            "timestamp":      self.timestamp,
        }

    def render_short(self) -> str:
        """One-line rendering for inclusion in another agent's context."""
        return (
            f"[{self.agent_id} | {self.domain} | {self.match_strength}] "
            f"{self.summary}"
            + (f" Principle: {self.principle}" if self.principle else "")
            + (f" Direction: {self.direction}" if self.direction else "")
        )


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Per-wander shared state. ONE instance per Wandering Room session,
    passed by reference to every agent.

    Agents are async coroutines that touch the same instance — all
    mutations go through `_lock`. Reads of `visited_urls` membership and
    queue peek are unlocked (single-step operations on Python primitives
    are atomic-enough at our scale).

    Lifecycle:
      - Created in runtime.run_wandering_session() before agents spawn.
      - Lives in memory for the duration of the wander.
      - Discarded after the session ends. NOT persisted across wanders
        (that would defeat Law 1 — every wander starts fresh).
    """

    session_id: str = ""
    visited_urls: set[str] = field(default_factory=set)
    followon_queue: list[FollowonItem] = field(default_factory=list)

    # Bounded queue so a runaway page producing 50 links doesn't starve
    # the chaos walker. Capacity is much larger than typical use; the
    # priority sort keeps the best near the head.
    MAX_FOLLOWON_QUEUE: int = 50

    # WANDER_AGENT_NOTICEBOARD test scaffold (June 2026). Append-only list
    # of AgentNotice posted by agents at next-domain-pick boundaries.
    # Empty when the noticeboard feature flag is off — no behavior change
    # from baseline in that case. Informational only; never enters dig
    # content. Bounded to prevent unbounded growth on long runs.
    noticeboard: list["AgentNotice"] = field(default_factory=list)
    MAX_NOTICEBOARD_ENTRIES: int = 200

    # CONSTELLAX_GOVERNOR (Stage-2 flow governor, June 2026). The governor is a
    # session-level observer (src/wandering/governor.py) that watches findings
    # arrive and may seize FLOW — halting the wander early on a confirmed CLOSE.
    # SessionState stays PURE: it holds only a halt flag + reason + an optional
    # per-notice callback ref. ALL governor logic (LLM edge detection, the
    # decide table) lives in the governor module, never here. Defaults below
    # mean zero behavior change when the flag is off (no callback, halt False).
    governor_halt: bool = False
    governor_halt_reason: str = ""
    # async callback: (AgentNotice) -> Awaitable[None]. Set by runtime when the
    # governor is active; fired fire-and-forget from post_notice (off-lock).
    _on_notice: object = field(default=None, repr=False)

    # Internal lock — never expose to callers; use the public helpers.
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # --- visited URL helpers ---

    def has_visited(self, url: str) -> bool:
        """Membership check on `visited_urls`. Cheap, unlocked.

        Used by fetchers before issuing a request. False negatives are
        impossible (the set monotonically grows). False positives could
        theoretically happen during a concurrent `mark_visited` — but the
        outcome is "skip this URL", which is exactly what we want anyway.
        """
        if not url:
            return False
        return normalize_url(url) in self.visited_urls

    async def mark_visited(self, url: str) -> bool:
        """Record `url` as visited. Returns True if newly added, False if
        already present.

        Always lock — protects against two agents racing to mark the same
        URL where the second one would otherwise also fetch it.
        """
        if not url:
            return False
        normalized = normalize_url(url)
        async with self._lock:
            if normalized in self.visited_urls:
                return False
            self.visited_urls.add(normalized)
            return True

    # --- follow-on queue helpers ---

    async def enqueue_followon(self, item: FollowonItem) -> bool:
        """Add `item` to the follow-on queue. Returns True if added, False
        if rejected (duplicate URL, already-visited, or capacity).

        We dedup against `visited_urls` AND against already-queued URLs —
        no point queuing a URL we've already wandered to or queued.

        The queue is kept SORTED by score descending so the next dequeue
        is O(1). Insertion is O(N) on the queue length but N is bounded
        by MAX_FOLLOWON_QUEUE.
        """
        if not item.url:
            return False
        if item.url in self.visited_urls:
            return False
        async with self._lock:
            # Re-check inside the lock — visited_urls could have grown
            # between the unlocked check above and now.
            if item.url in self.visited_urls:
                return False
            # Dedup against already-queued URLs.
            for existing in self.followon_queue:
                if existing.url == item.url:
                    return False
            self.followon_queue.append(item)
            self.followon_queue.sort(key=lambda fi: fi.score, reverse=True)
            if len(self.followon_queue) > self.MAX_FOLLOWON_QUEUE:
                # Drop the lowest-priority overflow. The queue is sorted
                # descending so .pop() removes the worst entry.
                self.followon_queue.pop()
            return True

    async def pop_followon(self) -> FollowonItem | None:
        """Remove and return the highest-priority follow-on item, or None
        if the queue is empty.

        Caller is responsible for marking the URL visited if it issues
        a fetch.
        """
        async with self._lock:
            if not self.followon_queue:
                return None
            return self.followon_queue.pop(0)

    def peek_followon_count(self) -> int:
        """Unlocked queue size. Used by policy to decide whether to
        consult the queue this turn."""
        return len(self.followon_queue)

    # --- noticeboard helpers (WANDER_AGENT_NOTICEBOARD test scaffold) ---

    async def post_notice(self, notice: "AgentNotice") -> bool:
        """Append `notice` to the noticeboard. Returns True on success.

        Bounded — if the board would exceed MAX_NOTICEBOARD_ENTRIES, the
        OLDEST notice is dropped (FIFO). Newer notices reflect the
        cohort's most recent coverage and are more useful to readers.

        Called by agents after each successful dig produces a report.
        Pure write — no read or LLM call. Cheap.
        """
        if notice is None or not notice.agent_id:
            return False
        async with self._lock:
            self.noticeboard.append(notice)
            if len(self.noticeboard) > self.MAX_NOTICEBOARD_ENTRIES:
                self.noticeboard.pop(0)
            cb = self._on_notice
        # Fire the governor observer OFF-lock, fire-and-forget — it must never
        # block the posting agent. The governor owns all LLM/decision logic and
        # guards its own state; we only schedule it. Failures are the
        # governor's to swallow (it wraps observe in try/except).
        if cb is not None:
            try:
                asyncio.get_running_loop().create_task(cb(notice))
            except Exception as e:  # no running loop / scheduling failure — non-fatal
                log.warning("governor observer not scheduled (ignored): %s", e)
        return True

    def recent_notices(
        self, *, n: int = 10, exclude_agent_id: str = "",
    ) -> list["AgentNotice"]:
        """Return the most recent up-to-`n` notices, optionally excluding
        notices posted by `exclude_agent_id` (so an agent doesn't read
        its own posts as cohort signal).

        Unlocked — reading a snapshot of an append-only list is safe at
        our concurrency level. Returns a fresh list so callers can't
        mutate the live board through the result.
        """
        if not self.noticeboard:
            return []
        # Walk from the tail so we get most-recent first; filter out
        # self-posts as we go; stop after collecting n.
        out: list["AgentNotice"] = []
        for entry in reversed(self.noticeboard):
            if exclude_agent_id and entry.agent_id == exclude_agent_id:
                continue
            out.append(entry)
            if len(out) >= n:
                break
        return out

    def peek_noticeboard_count(self) -> int:
        """Unlocked board size. Useful for diagnostics."""
        return len(self.noticeboard)


__all__ = [
    "AgentNotice",
    "FollowonItem",
    "SessionState",
    "normalize_url",
]
