"""
validate_neo4j_parity.py — pre-cutover sanity check for the FalkorDB → Neo4j
migration.

WHAT THIS DOES
==============
1. Connects to Neo4j using NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD.
2. Runs schema init (creates constraints + vector index).
3. Writes a synthetic ThreadRecord + 2 IterationRecords against BOTH
   the Neo4j backend and an InMemory backend.
4. Reads back from both and diffs the results.
5. Cleans up the synthetic data on success.

WHEN TO RUN
===========
Before flipping CONSTELLAX_DB_BACKEND=neo4j in production. After you've:
  1. Provisioned Aura Free.
  2. Set NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in your .env.
  3. Installed the neo4j driver (pip install -r requirements.txt).

USAGE
=====
  cd ~/Desktop/reasoningEngine
  python3 scripts/validate_neo4j_parity.py

EXIT CODES
==========
  0 — all checks passed; Neo4j is ready for cutover
  1 — connection failure or env vars missing
  2 — schema init failed
  3 — parity check failed (read-back differs from in-memory)

This script writes under a unique project_id (`parity_check_<ts>`) so it
will not interfere with real data. It cleans up before exiting on success.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any

# Make src/ imports work when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bridge.neo4j_backend import (
    Neo4jThreadStore,
    build_neo4j_thread_store_from_env,
    init_schema,
)
from src.bridge.thread_store import InMemoryThreadStore
from src.core.thread_types import (
    Entity,
    IterationRecord,
    MemoryContext,
    SegmentedResponse,
    Segment,
    ThreadRecord,
    DEFAULT_UNCERTAINTY_DISCLAIMER,
)


# Try to load .env so the script picks up local credentials.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"
def _bold(s: str) -> str:  return f"\033[1m{s}\033[0m"
def _yellow(s: str) -> str:return f"\033[33m{s}\033[0m"


# ─── Build synthetic test data ────────────────────────────────────────

def make_test_fixtures(project_id: str) -> tuple[ThreadRecord, list[IterationRecord]]:
    now = time.time()
    thread_id = f"thr-parity-{uuid.uuid4().hex[:8]}"
    user_id = f"user-parity-{uuid.uuid4().hex[:8]}"

    thread = ThreadRecord(
        id=thread_id,
        user_id=user_id,
        project_id=project_id,
        workspace_id="parity-test",
        title="Parity check thread",
        created_at=now,
        updated_at=now,
        status="active",
        all_entities=["Neo4j", "FalkorDB", "Cypher"],
        all_tags=["migration", "graph-db"],
        all_domains=["infrastructure"],
        iteration_count=2,
        last_route="deep",
        last_confidence="moderate",
        aggregate_time_ms=1234,
        aggregate_cost_usd=0.04,
        perspectives_run=3,
    )

    iters = []
    for i in range(2):
        iter_id = f"itr-parity-{uuid.uuid4().hex[:8]}-{i}"
        iters.append(IterationRecord(
            id=iter_id,
            thread_id=thread_id,
            sequence_num=i + 1,
            question=f"Test question {i + 1}",
            status="done",
            created_at=now + i,
            completed_at=now + i + 0.5,
            entities=[Entity(name="Neo4j", kind="product"), Entity(name="Cypher", kind="language")],
            tags=["migration", "graph-db"],
            domains=["infrastructure"],
            embedding=[0.1, 0.2, 0.3] + [0.0] * (1536 - 3),  # 1536-dim default
            embedding_model="parity-fake-model",
            response=SegmentedResponse(
                overall_confidence="moderate",
                synthesizer=Segment(text=f"Synth {i+1}", confidence="moderate", delivered_at=now+i+0.4),
                opinion=None,
                prospects=None,
            ),
            memory_context=MemoryContext(),
        ))
    return thread, iters


# ─── Single-backend write + read sequence ─────────────────────────────

async def write_and_read(store: Any, thread: ThreadRecord, iters: list[IterationRecord]) -> dict:
    """Round-trip data through a store, return a result snapshot for diffing."""
    # Save iterations FIRST so the thread.iteration_ids are populated
    # before save_thread. Mirror real flow from thread_persistence.py.
    for it in iters:
        await store.save_iteration(it)
    thread.iteration_ids = [it.id for it in iters]
    await store.save_thread(thread)

    fetched_thread = await store.get_thread(thread.id)
    fetched_iters = await store.list_iterations_for_thread(thread.id)
    by_entity = await store.find_threads_mentioning_entity("Neo4j")
    by_tag = await store.find_threads_by_tag("migration")
    listed = await store.list_threads(user_id=thread.user_id, limit=10)

    return {
        "thread_found": fetched_thread is not None,
        "thread_title": fetched_thread.title if fetched_thread else None,
        "thread_status": fetched_thread.status if fetched_thread else None,
        "thread_all_entities_count": len(fetched_thread.all_entities) if fetched_thread else 0,
        "iter_count": len(fetched_iters),
        "iter_sequence_order": [it.sequence_num for it in fetched_iters],
        "iter_ids": sorted(it.id for it in fetched_iters),
        "by_entity_has_thread": thread.id in by_entity,
        "by_tag_has_thread": thread.id in by_tag,
        "list_threads_has_thread": any(t.id == thread.id for t in listed),
    }


# ─── Main parity routine ──────────────────────────────────────────────

async def main() -> int:
    print(_bold("\n=== Neo4j ↔ InMemory parity validation ===\n"))

    # ─── Step 1: build Neo4j store from env ───────────────────────────
    print("Step 1: connecting to Neo4j Aura...")
    neo4j_store = build_neo4j_thread_store_from_env()
    if neo4j_store is None:
        print(_red("  ✗ Could not build Neo4j store."))
        print("    Required env vars: NEO4J_URI, NEO4J_PASSWORD (NEO4J_USERNAME optional, defaults to 'neo4j').")
        print("    Add them to your .env file:")
        print("      NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io")
        print("      NEO4J_USERNAME=neo4j")
        print("      NEO4J_PASSWORD=...")
        return 1

    uri = os.environ.get("NEO4J_URI", "").strip()
    print(_green(f"  ✓ Driver constructed for {uri}"))

    # ─── Step 2: schema init ───────────────────────────────────────────
    # Read database name from env (Aura Free uses the instance ID as the db
    # name, not the standard "neo4j"). init_schema's default of "neo4j" is
    # wrong for Aura Free — every DDL statement would land in the wrong
    # database and silently get swallowed by the warning-only error handler.
    print("\nStep 2: initializing schema (constraints + vector index)...")
    try:
        dim = int(os.environ.get("NEO4J_EMBEDDING_DIM", "1536"))
        database = os.environ.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
        await init_schema(neo4j_store._driver, database=database, embedding_dim=dim)
        print(_green(f"  ✓ Schema ready (database={database}, vector dim={dim})"))
    except Exception as e:
        print(_red(f"  ✗ Schema init failed: {e}"))
        await neo4j_store.close()
        return 2

    # ─── Step 3: parity check ─────────────────────────────────────────
    print("\nStep 3: write fixtures to both backends and diff read-back...")
    project_id = f"parity_check_{int(time.time())}"
    inmem = InMemoryThreadStore()

    try:
        # Two separate fixtures so the same IDs don't collide across stores
        thread_a, iters_a = make_test_fixtures(project_id)
        thread_b, iters_b = make_test_fixtures(project_id)

        inmem_result  = await write_and_read(inmem, thread_a, iters_a)
        neo4j_result = await write_and_read(neo4j_store, thread_b, iters_b)
    except Exception as e:
        print(_red(f"  ✗ Write/read raised: {e}"))
        await neo4j_store.close()
        return 3

    # ─── Step 4: diff ────────────────────────────────────────────────
    # We can't compare ids directly (different per backend), but every
    # other shape-and-presence claim should match exactly.
    shape_keys = [
        "thread_found", "thread_status", "thread_all_entities_count",
        "iter_count", "iter_sequence_order",
        "by_entity_has_thread", "by_tag_has_thread", "list_threads_has_thread",
    ]
    mismatches: list[str] = []
    for k in shape_keys:
        a, b = inmem_result.get(k), neo4j_result.get(k)
        if a != b:
            mismatches.append(f"  - {k}: inmem={a!r}  neo4j={b!r}")

    print()
    if not mismatches:
        print(_green("  ✓ All parity checks passed:"))
        for k in shape_keys:
            print(f"    {k}: {inmem_result[k]!r}")
    else:
        print(_red("  ✗ Parity FAILED:"))
        for m in mismatches:
            print(_red(m))

    # ─── Step 5: cleanup ────────────────────────────────────────────
    print("\nStep 4: cleaning up test data...")
    try:
        await neo4j_store.delete_thread(thread_b.id)
        print(_green(f"  ✓ Removed Neo4j thread {thread_b.id}"))
    except Exception as e:
        print(_yellow(f"  ⚠ Cleanup hiccup (non-fatal): {e}"))

    await neo4j_store.close()

    if mismatches:
        print(_red("\n=== RESULT: PARITY FAILED — do not flip the env var yet. ===\n"))
        return 3
    print(_green("\n=== RESULT: ALL CHECKS PASSED — Neo4j is ready for cutover. ===\n"))
    print("Next step: set CONSTELLAX_DB_BACKEND=neo4j in your runtime env to switch over.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
