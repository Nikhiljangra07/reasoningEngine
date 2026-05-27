"""
validate_decision_trace_writer.py — Phase 2a regression test.

Verifies:
  - init_schema picks up the new Decision Trace constraints + indexes
  - Neo4jDecisionTraceWriter.write_bundle commits all 6 event types
  - Dual labels (:DecisionTrace:<Type>) are stamped on every node
  - Typed relationships from Iteration (MADE_DECISION, RAISED_QUESTION, etc.)
    point at the right nodes
  - SUPERSEDES edges resolve correctly between Decisions
  - write_bundle is idempotent (re-running produces same counts, no dup nodes)

USAGE
=====
  cd ~/Desktop/reasoningEngine
  source .venv/bin/activate
  python scripts/validate_decision_trace_writer.py

EXIT CODES
==========
  0 — all checks passed
  1 — connection or env failure
  2 — schema init failed
  3 — any assertion failed
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

from src.bridge.neo4j_backend import (
    Neo4jDecisionTraceWriter,
    build_neo4j_thread_store_from_env,
    init_schema,
)
from src.core.decision_trace_types import (
    Decision,
    DecisionTraceBundle,
    Insight,
    Question,
    Reference,
    SystemResponse,
    UserMessage,
    new_dt_id,
)
from src.core.thread_types import IterationRecord, ThreadRecord


def _g(s): return f"\033[32m{s}\033[0m"
def _r(s): return f"\033[31m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"


async def main() -> int:
    print(_b("\n=== Decision Trace Writer parity validation ===\n"))

    store = build_neo4j_thread_store_from_env()
    if store is None:
        print(_r("  ✗ Could not build Neo4j store. Check NEO4J_URI / NEO4J_PASSWORD."))
        return 1
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    print(_g(f"  ✓ Driver ready (database={db})"))

    try:
        await init_schema(store._driver, database=db)
        print(_g("  ✓ Schema init OK (Phase 2a constraints + indexes applied)"))
    except Exception as e:
        print(_r(f"  ✗ Schema init failed: {e}"))
        await store.close()
        return 2

    writer = Neo4jDecisionTraceWriter(store._driver, database=db)

    suf = uuid.uuid4().hex[:8]
    thr = ThreadRecord(id=f"thr-parity-2a-{suf}", user_id=f"u-2a-{suf}",
                       workspace_id="cursor", title="Phase 2a parity")
    it = IterationRecord(
        id=f"itr-parity-2a-{suf}", thread_id=thr.id, sequence_num=1,
        workspace_id="cursor", surface_id="map-room",
        created_at=time.time(), completed_at=time.time(),
    )
    await store.save_thread(thr)
    await store.save_iteration(it)

    ts = time.time()
    common = dict(
        iteration_id=it.id, thread_id=thr.id,
        workspace_id="cursor", surface_id="map-room",
        user_id=thr.user_id, project_id=None, ts=ts,
    )
    older_id = new_dt_id("decision")
    older = Decision(id=older_id, text="Use FalkorDB", status="superseded", confidence=0.8, **common)
    newer = Decision(id=new_dt_id("decision"), text="Migrate to Neo4j",
                     status="committed", confidence=0.95, supersedes=older_id, **common)
    bundle = DecisionTraceBundle(
        iteration_id=it.id, thread_id=thr.id,
        user_message=UserMessage(id=new_dt_id("user_message"), text="Should we move to Neo4j?", **common),
        system_response=SystemResponse(id=new_dt_id("system_response"),
                                       text="Yes, the GDS + vector wins are real.", **common),
        decisions=[older, newer],
        questions=[Question(id=new_dt_id("question"), text="What is the cost trajectory?",
                            resolved=False, confidence=0.9, **common)],
        references=[Reference(id=new_dt_id("reference"), kind="url",
                              target="https://neo4j.com/pricing", label="Neo4j Pricing",
                              confidence=1.0, **common)],
        insights=[Insight(id=new_dt_id("insight"),
                          text="Vector + graph in one DB collapses two services.",
                          confidence=0.85, **common)],
    )

    counts = await writer.write_bundle(bundle)
    expected = {"user_message":1, "system_response":1, "decision":2,
                "question":1, "reference":1, "insight":1}
    if counts != expected:
        print(_r(f"  ✗ counts mismatch: got {counts}, expected {expected}"))
        return 3
    print(_g(f"  ✓ write_bundle counts: {counts}"))

    failed = False
    async with store._driver.session(database=db) as s:
        # Dual labels on Decision nodes
        r = await s.run(
            "MATCH (n:DecisionTrace:Decision {iteration_id: $iid}) "
            "RETURN n.id AS id, labels(n) AS labels",
            iid=it.id,
        )
        rows = [dict(rec) async for rec in r]
        if len(rows) != 2:
            print(_r(f"  ✗ expected 2 :Decision rows, got {len(rows)}"))
            failed = True
        for row in rows:
            if set(row["labels"]) != {"DecisionTrace", "Decision"}:
                print(_r(f"  ✗ wrong labels: {row['labels']}"))
                failed = True
        if not failed:
            print(_g("  ✓ Dual labels (:DecisionTrace:Decision) confirmed on all Decisions"))

        # SUPERSEDES edge
        r = await s.run(
            "MATCH (new:Decision)-[:SUPERSEDES]->(old:Decision) "
            "WHERE new.iteration_id = $iid AND new.text = $new_text "
            "RETURN old.text AS old_text",
            iid=it.id, new_text="Migrate to Neo4j",
        )
        rec = await r.single()
        if rec is None or rec["old_text"] != "Use FalkorDB":
            print(_r("  ✗ SUPERSEDES edge missing or wrong"))
            failed = True
        else:
            print(_g("  ✓ SUPERSEDES edge points 'Migrate to Neo4j' → 'Use FalkorDB'"))

        # Iteration relationships
        r = await s.run(
            "MATCH (i:Iteration {id: $iid})-[r]->(n:DecisionTrace) "
            "RETURN type(r) AS rel, count(n) AS c ORDER BY rel",
            iid=it.id,
        )
        rels = {rec["rel"]: rec["c"] async for rec in r}
        expected_rels = {
            "HAS_USER_MESSAGE": 1, "HAS_SYSTEM_RESPONSE": 1, "MADE_DECISION": 2,
            "RAISED_QUESTION": 1, "CITED": 1, "RECORDED_INSIGHT": 1,
        }
        if rels != expected_rels:
            print(_r(f"  ✗ relationships mismatch: got {rels}, expected {expected_rels}"))
            failed = True
        else:
            print(_g(f"  ✓ All 6 typed relationships from Iteration present: {rels}"))

    # Idempotency
    counts2 = await writer.write_bundle(bundle)
    async with store._driver.session(database=db) as s:
        rec = await (await s.run(
            "MATCH (n:DecisionTrace) WHERE n.iteration_id = $iid RETURN count(n) AS c",
            iid=it.id,
        )).single()
        if rec["c"] != 7:
            print(_r(f"  ✗ idempotency: expected 7 nodes after re-write, got {rec['c']}"))
            failed = True
        else:
            print(_g(f"  ✓ Idempotency: re-write counts {counts2}; total nodes still {rec['c']}"))

    # Cleanup
    async with store._driver.session(database=db) as s:
        await s.run(
            "MATCH (n:DecisionTrace) WHERE n.iteration_id = $iid DETACH DELETE n",
            iid=it.id,
        )
    await store.delete_thread(thr.id)
    print(_g("  ✓ Cleanup complete"))

    await store.close()
    if failed:
        print(_r("\n=== RESULT: FAILED ===\n"))
        return 3
    print(_g("\n=== RESULT: ALL CHECKS PASSED ===\n"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
