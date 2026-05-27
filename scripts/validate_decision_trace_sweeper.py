"""
validate_decision_trace_sweeper.py — Phase 3 regression test.

Exercises the full sweeper pipeline end-to-end against live Aura without
waiting 30 real minutes:

  1. Create a Thread + Iteration with completed_at = (now - 2 hours),
     structured_at = NULL.
  2. Build a sweeper with idle_threshold_sec=1 (so the synthetic iteration
     immediately qualifies).
  3. Call sweep_once() directly. Verify:
       - The iteration was processed (stats.processed == 1)
       - The bundle was written to Neo4j (UserMessage + SystemResponse
         exist with dt-msg-user-<iter_id> / dt-msg-sys-<iter_id> ids)
       - structured_at was stamped (timestamp present, > completed_at)
  4. Call sweep_once() again — verify it's a no-op (the iteration is
     now structured, so the WHERE clause excludes it).
  5. Cleanup.

USAGE
=====
  cd ~/Desktop/reasoningEngine
  source .venv/bin/activate
  python scripts/validate_decision_trace_sweeper.py

EXIT CODES
==========
  0 — all checks passed
  1 — Neo4j connection or env missing
  3 — assertion failed
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.bridge.decision_trace_sweeper import build_sweeper
from src.bridge.neo4j_backend import (
    build_neo4j_thread_store_from_env,
    init_schema,
)
from src.core.thread_types import (
    IterationRecord,
    Segment,
    SegmentedResponse,
    ThreadRecord,
)


def _g(s): return f"\033[32m{s}\033[0m"
def _r(s): return f"\033[31m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"


async def main() -> int:
    print(_b("\n=== Decision Trace Sweeper validation ===\n"))

    store = build_neo4j_thread_store_from_env()
    if store is None:
        print(_r("  ✗ Could not build Neo4j store"))
        return 1
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    print(_g(f"  ✓ Driver ready (database={db})"))

    await init_schema(store._driver, database=db)

    # 1) Plant a synthetic iteration that already qualifies for sweeping
    suf = uuid.uuid4().hex[:8]
    user_id = f"u-sweep-{suf}"
    thread = ThreadRecord(
        id=f"thr-sweep-{suf}",
        user_id=user_id,
        workspace_id="cursor",
        title="Phase 3 sweeper test",
    )
    # completed_at = now - 7200 (2 hours ago) so any idle_threshold > 1 matches
    completed_at = time.time() - 7200
    iteration = IterationRecord(
        id=f"itr-sweep-{suf}",
        thread_id=thread.id,
        sequence_num=1,
        workspace_id="cursor",
        surface_id="map-room",
        question="Should we commit to Aura Pro at 8GB once we cross 1000 active users?",
        response=SegmentedResponse(
            overall_confidence="high",
            synthesizer=Segment(
                text="Yes. At 1000+ users you're roughly 10-15% of an 8GB instance's headroom — comfortable, with clear room to grow. Commit.",
                confidence="high", delivered_at=completed_at,
            ),
        ),
        created_at=completed_at - 1,
        completed_at=completed_at,
        structured_at=None,            # explicit — this is what the sweeper looks for
    )
    await store.save_thread(thread)
    await store.save_iteration(iteration)
    print(_g(f"  ✓ planted iteration {iteration.id} completed_at=now-2h structured_at=NULL"))

    # 2) Build a sweeper with idle_threshold_sec=1 so the synthetic record
    #    qualifies immediately. We invoke sweep_once() directly, not run().
    sweeper = build_sweeper(store, idle_threshold_sec=1, batch_size=10)

    # 3) First sweep — should process and structure the iteration
    stats = await sweeper.sweep_once()
    print(f"\n  first sweep stats:")
    print(f"    processed:           {stats.processed}")
    print(f"    structured:          {stats.structured}")
    print(f"    classifier_failures: {stats.classifier_failures}")
    print(f"    write_failures:      {stats.write_failures}")
    print(f"    stamp_failures:      {stats.stamp_failures}")

    failed = False
    if stats.processed < 1:
        print(_r(f"  ✗ sweeper didn't process the planted iteration (processed={stats.processed})"))
        failed = True
    if stats.structured < 1:
        print(_r(f"  ✗ sweeper didn't structure the iteration (structured={stats.structured})"))
        failed = True
    if stats.write_failures > 0:
        print(_r(f"  ✗ unexpected write failures: {stats.errors}"))
        failed = True

    # 4) Verify Neo4j state — structured_at stamped, dt-msg-user node exists with deterministic id
    async with store._driver.session(database=db) as s:
        r = await s.run(
            "MATCH (i:Iteration {id: $id}) "
            "RETURN i.structured_at AS sa, i.completed_at AS ca",
            id=iteration.id,
        )
        rec = await r.single()
        if rec is None or rec["sa"] is None:
            print(_r("  ✗ structured_at was not stamped"))
            failed = True
        elif rec["sa"] <= rec["ca"]:
            print(_r(f"  ✗ structured_at ({rec['sa']}) not after completed_at ({rec['ca']})"))
            failed = True
        else:
            print(_g(f"  ✓ structured_at stamped: {rec['sa']:.2f} (was NULL before sweep)"))

        # Deterministic UserMessage / SystemResponse IDs
        for expected_id, label in [
            (f"dt-msg-user-{iteration.id}", "UserMessage"),
            (f"dt-msg-sys-{iteration.id}", "SystemResponse"),
        ]:
            r = await s.run(
                f"MATCH (n:DecisionTrace:{label} {{id: $id}}) RETURN n.text AS text",
                id=expected_id,
            )
            rec = await r.single()
            if rec is None:
                print(_r(f"  ✗ {label} not found by deterministic id ({expected_id})"))
                failed = True
            else:
                print(_g(f"  ✓ {label} present: {rec['text'][:60]!r}"))

        # Sweeper-extracted typed events — count by type
        r = await s.run(
            "MATCH (i:Iteration {id: $id})-[r]->(n:DecisionTrace) "
            "RETURN type(r) AS rel, count(n) AS c ORDER BY rel",
            id=iteration.id,
        )
        rels = {rec["rel"]: rec["c"] async for rec in r}
        print(f"  iteration relationships: {rels}")

    # 5) Second sweep — should be a no-op (structured_at IS NULL filter excludes the iter)
    stats2 = await sweeper.sweep_once()
    if stats2.processed != 0:
        print(_r(f"  ✗ second sweep should have been a no-op, processed={stats2.processed}"))
        failed = True
    else:
        print(_g(f"  ✓ second sweep correctly skipped already-structured iteration"))

    # 6) Cleanup — remove typed events, iteration, thread
    async with store._driver.session(database=db) as s:
        await s.run(
            "MATCH (n:DecisionTrace) WHERE n.iteration_id = $iid DETACH DELETE n",
            iid=iteration.id,
        )
    await store.delete_thread(thread.id)
    print(_g("  ✓ cleanup complete"))

    await store.close()
    if failed:
        print(_r("\n=== RESULT: FAILED ===\n"))
        return 3
    print(_g("\n=== RESULT: ALL CHECKS PASSED ===\n"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
