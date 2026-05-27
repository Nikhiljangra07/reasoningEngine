"""
DecisionTraceSweeper — background task that turns raw iterations into a
structured Decision Trace, automatically, on a 30-minute idle barrier.

ROLE
====
Phase 1 stamps every Iteration with workspace_id / surface_id /
structured_at=NULL. Phase 2a built the typed event nodes + writer; 2b
built the InlineClassifier; 2c built the markdown decision-log parser.
This file is the orchestrator that connects them:

    for each iteration that's been idle > 30 minutes and not yet structured:
        bundle = classifier.classify_iteration(iteration)
        writer.write_bundle(bundle)
        store.stamp_structured(iter.id, now)

It runs as a FastAPI background task — one sweep every N minutes,
batch size 50, never blocks the request path.

WHY NOT INLINE (during /api/v2/trace)?
======================================
1. LLM cost — classifier adds ~$0.0003/turn. Doing this synchronously
   inflates request latency by ~2-3 seconds. Async sweeper amortizes.
2. Idle barrier — we want to structure the CONVERSATION AS A WHOLE
   (multiple turns the user has just had), not each turn in isolation.
   The 30-min idle wait gives natural conversation boundaries.
3. User-visible failure isolation — if Gemini is down, the trace
   endpoint still returns a response. Memory structuring catches up
   when the LLM recovers.

SAFETY
======
- Atomic writes — each iteration's bundle commits in one transaction.
- Stamp last — `structured_at` set AFTER bundle write, so crashes
  leave the iteration unstructured and the next sweep retries.
- Idempotent — verbatim node IDs are deterministic (dt-msg-user-<iter_id>,
  dt-msg-sys-<iter_id>), so MERGE on retry is a no-op for UserMessage
  and SystemResponse. Classifier-extracted nodes get fresh uuids each
  attempt, but those only get written on the FIRST successful pass
  (subsequent passes are blocked by structured_at IS NULL filter).
- Per-iteration error isolation — if classification or writing fails on
  one iteration, we log + skip. The rest of the batch still processes.

POLICY
======
- Default idle threshold: 1800 seconds (30 minutes).
- Default batch size: 50 iterations per sweep.
- Default interval: 300 seconds (5 minutes between sweeps).
- ALWAYS write the bundle (even verbatim-only on classifier failure) —
  the verbatim text is the source of truth and should land in the graph
  regardless of whether typed extraction succeeded.
- ALWAYS stamp structured_at on successful write — even when the
  classifier failed and we wrote only verbatim. Re-extraction is a
  future operator-driven action, not part of the normal sweep.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from src.bridge.neo4j_backend import (
    Neo4jDecisionTraceWriter,
    Neo4jThreadStore,
)
from src.llm.decision_trace_classifier import InlineClassifier

log = logging.getLogger("constellax.decision_trace_sweeper")


DEFAULT_IDLE_SEC = 1800        # 30 minutes
DEFAULT_BATCH_SIZE = 50
DEFAULT_INTERVAL_SEC = 300     # 5 minutes between sweeps


@dataclass
class SweepStats:
    """What one sweep_once() pass did. Logged at INFO if any work happened,
    DEBUG otherwise. The sweeper accumulates these across calls; the
    server can expose them via a debug endpoint."""
    processed:           int = 0   # iterations the sweeper looked at
    structured:          int = 0   # iterations successfully written + stamped
    classifier_failures: int = 0   # classifier returned success=False (we still wrote verbatim)
    write_failures:      int = 0   # writer.write_bundle raised — we DID NOT stamp
    stamp_failures:      int = 0   # stamp_structured returned False after write — rare
    errors:              list[str] = field(default_factory=list)

    def merge(self, other: "SweepStats") -> None:
        """Accumulate another sweep's counts into this one."""
        self.processed += other.processed
        self.structured += other.structured
        self.classifier_failures += other.classifier_failures
        self.write_failures += other.write_failures
        self.stamp_failures += other.stamp_failures
        self.errors.extend(other.errors[:10])  # cap to last 10 to avoid unbounded growth


# ─── The sweeper ─────────────────────────────────────────────────────

class DecisionTraceSweeper:
    """Background task that auto-structures iterations 30 minutes after
    they go idle. Compose with a Neo4jThreadStore, an InlineClassifier,
    and a Neo4jDecisionTraceWriter — the sweeper doesn't build any of
    them itself, keeping wiring explicit at the server layer."""

    def __init__(
        self,
        store: Neo4jThreadStore,
        classifier: InlineClassifier,
        writer: Neo4jDecisionTraceWriter,
        *,
        idle_threshold_sec: int = DEFAULT_IDLE_SEC,
        batch_size: int = DEFAULT_BATCH_SIZE,
        interval_sec: int = DEFAULT_INTERVAL_SEC,
    ) -> None:
        self.store = store
        self.classifier = classifier
        self.writer = writer
        self.idle_threshold_sec = idle_threshold_sec
        self.batch_size = batch_size
        self.interval_sec = interval_sec
        # Cumulative stats across the process lifetime — useful for a debug endpoint
        self.lifetime_stats = SweepStats()
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    # ─── One sweep ────────────────────────────────────────────────

    async def sweep_once(self) -> SweepStats:
        """Find unstructured iterations past the idle threshold and process
        each. Returns the stats for this single sweep (not cumulative).

        Never raises — every per-iteration failure is caught and logged.
        The only way this returns a partial result is if the initial
        query itself raises (Neo4j unreachable), in which case stats are
        empty and the error is in stats.errors."""
        stats = SweepStats()
        try:
            iter_ids = await self.store.find_unstructured_iteration_ids(
                idle_sec=self.idle_threshold_sec, limit=self.batch_size,
            )
        except Exception as e:
            log.warning("sweep: find_unstructured_iteration_ids failed: %s", e)
            stats.errors.append(f"query_failed: {type(e).__name__}: {e}")
            return stats

        if not iter_ids:
            return stats

        log.info("sweep: %d iteration(s) ready for structuring", len(iter_ids))
        for iter_id in iter_ids:
            stats.processed += 1
            await self._process_one(iter_id, stats)

        # Cumulative update
        self.lifetime_stats.merge(stats)
        log.info(
            "sweep done: processed=%d structured=%d cls_fail=%d write_fail=%d stamp_fail=%d",
            stats.processed, stats.structured,
            stats.classifier_failures, stats.write_failures, stats.stamp_failures,
        )
        return stats

    async def _process_one(self, iter_id: str, stats: SweepStats) -> None:
        """Process a single iteration. Catches every exception and logs;
        the outer loop is never blocked by one bad iteration."""
        try:
            iteration = await self.store.get_iteration(iter_id)
        except Exception as e:
            log.warning("sweep: get_iteration(%s) failed: %s", iter_id, e)
            stats.errors.append(f"get:{iter_id}: {type(e).__name__}")
            return
        if iteration is None:
            log.debug("sweep: iter %s vanished between query and load", iter_id)
            return

        # Hydrate user_id from the parent thread. The sweeper does the
        # lookup here so the classifier doesn't need a separate Thread
        # round-trip. If the thread is missing (shouldn't happen but
        # defensive), continue with empty user_id — the writer still
        # produces correctly-shaped nodes, just without that filter
        # dimension.
        user_id = ""
        try:
            if iteration.thread_id:
                thread = await self.store.get_thread(iteration.thread_id)
                if thread is not None and thread.user_id:
                    user_id = thread.user_id
        except Exception as e:
            log.debug("sweep: get_thread for iter %s failed (continuing): %s", iter_id, e)
        # The classifier reads user_id from iteration.meta — stash it there
        # for this pass without mutating the persisted record.
        if iteration.meta is None:
            iteration.meta = {}
        iteration.meta["user_id"] = user_id

        # Classify (LLM call). Returns a bundle even on failure (verbatim only).
        try:
            bundle, cls_stats = await self.classifier.classify_iteration(iteration)
        except Exception as e:
            log.warning("sweep: classifier crash on iter %s: %s", iter_id, e)
            stats.errors.append(f"classify:{iter_id}: {type(e).__name__}")
            return
        if not cls_stats.success:
            # Verbatim still written; log so the operator sees recurring failures
            stats.classifier_failures += 1
            log.info(
                "sweep: classifier failed on iter %s (%s) — writing verbatim only",
                iter_id, cls_stats.error,
            )

        # Write the bundle. Writer is idempotent (MERGE-based), so a retry
        # on the next sweep produces no duplicates.
        try:
            await self.writer.write_bundle(bundle)
        except Exception as e:
            stats.write_failures += 1
            log.warning("sweep: writer failed on iter %s: %s", iter_id, e)
            stats.errors.append(f"write:{iter_id}: {type(e).__name__}")
            return  # DO NOT stamp — let the next sweep retry

        # Stamp structured_at LAST. Crashes between write and stamp leave
        # the iteration unstructured; idempotent writer makes retry safe.
        try:
            stamped = await self.store.stamp_structured(iter_id, time.time())
            if stamped:
                stats.structured += 1
            else:
                stats.stamp_failures += 1
                log.warning("sweep: stamp returned False for iter %s", iter_id)
        except Exception as e:
            stats.stamp_failures += 1
            log.warning("sweep: stamp failed on iter %s: %s", iter_id, e)
            stats.errors.append(f"stamp:{iter_id}: {type(e).__name__}")

    # ─── The run loop ─────────────────────────────────────────────

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Loop calling sweep_once every interval_sec until stop_event is set.

        Use this as the background coroutine for FastAPI:
            sweeper._task = asyncio.create_task(sweeper.run(stop_event))

        Each iteration:
          1. Wait `interval_sec` (interruptible by stop_event)
          2. Run one sweep
          3. Repeat

        The FIRST sweep also waits `interval_sec` — we don't want to
        block server startup; the first ~30 minutes of operation typically
        have nothing for the sweeper anyway."""
        self._stop_event = stop_event or asyncio.Event()
        log.info(
            "sweeper: starting (idle=%ds, batch=%d, interval=%ds)",
            self.idle_threshold_sec, self.batch_size, self.interval_sec,
        )
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.interval_sec,
                )
                # If we get here, stop was set during the wait — exit cleanly
                break
            except asyncio.TimeoutError:
                pass  # normal: interval elapsed without stop signal

            if self._stop_event.is_set():
                break

            try:
                await self.sweep_once()
            except Exception as e:
                # sweep_once itself shouldn't raise (it catches internally),
                # but the outer guard makes the run loop unconditionally
                # robust to any unexpected failure mode.
                log.exception("sweeper: sweep_once raised — continuing: %s", e)

        log.info("sweeper: stopped")

    def stop(self) -> None:
        """Request the run loop to exit at its next interval boundary."""
        if self._stop_event is not None:
            self._stop_event.set()


# ─── Factory ─────────────────────────────────────────────────────────

def build_sweeper(
    store: Neo4jThreadStore,
    *,
    idle_threshold_sec: int | None = None,
    batch_size: int | None = None,
    interval_sec: int | None = None,
) -> DecisionTraceSweeper:
    """Construct a DecisionTraceSweeper sharing the store's driver. Builds
    its own InlineClassifier (cheap, stateless) and reuses the driver
    for the Neo4jDecisionTraceWriter (avoids spinning up a second
    connection pool).

    The three optional overrides are typically only used in tests — the
    production defaults (30 min / 50 / 5 min) match the locked architecture."""
    classifier = InlineClassifier()
    writer = Neo4jDecisionTraceWriter(store._driver, database=store._database)
    return DecisionTraceSweeper(
        store=store, classifier=classifier, writer=writer,
        idle_threshold_sec=idle_threshold_sec or DEFAULT_IDLE_SEC,
        batch_size=batch_size or DEFAULT_BATCH_SIZE,
        interval_sec=interval_sec or DEFAULT_INTERVAL_SEC,
    )
