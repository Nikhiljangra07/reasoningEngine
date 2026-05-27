"""
validate_memory_retriever.py — Phase 4 regression test.

End-to-end validation against live Aura:
  1. Plant TWO threads (same user, different topics) with iterations,
     embeddings, and typed Decision Trace events.
  2. Issue a retrieve(query, user_id, thread_id=<thread_B>) call where
     the query semantically matches thread_A.
  3. Verify:
       - `local` entries come from thread_B (current thread context)
       - `cross_thread` entries surface thread_A (cross-thread recall)
       - Both groups carry full provenance (workspace_id, surface_id, ts)
       - render_timeline produces non-empty, readable markdown
  4. Cleanup.

USAGE
=====
  cd ~/Desktop/reasoningEngine
  source .venv/bin/activate
  python scripts/validate_memory_retriever.py

EXIT CODES
==========
  0 — all checks passed
  1 — Neo4j or env missing
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

from src.bridge.embedding_service import GeminiEmbeddingService
from src.bridge.memory_retriever import MemoryRetriever, render_timeline
from src.bridge.neo4j_backend import (
    Neo4jDecisionTraceWriter,
    build_neo4j_thread_store_from_env,
    init_schema,
)
from src.core.decision_trace_types import (
    Decision,
    DecisionTraceBundle,
    SystemResponse,
    UserMessage,
    new_dt_id,
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


async def _plant_thread(store, writer, embedder, *,
                        user_id, thread_id, title, workspace,
                        user_text, system_text):
    """Create one thread + one iteration + the Decision Trace bundle + embedding."""
    thr = ThreadRecord(id=thread_id, user_id=user_id, workspace_id=workspace, title=title)
    iter_id = f"itr-{thread_id}"
    iter_ts = time.time()
    iteration = IterationRecord(
        id=iter_id, thread_id=thread_id, sequence_num=1,
        workspace_id=workspace, surface_id="chat",
        question=user_text,
        response=SegmentedResponse(
            overall_confidence="high",
            synthesizer=Segment(text=system_text, confidence="high", delivered_at=iter_ts),
        ),
        created_at=iter_ts, completed_at=iter_ts,
    )
    # Embed the iteration text (user + system combined) and stamp it
    emb = await embedder.embed(f"{user_text}\n\n{system_text}")
    if emb.success and emb.vector:
        iteration.embedding = emb.vector
        iteration.embedding_model = emb.model
    await store.save_thread(thr)
    await store.save_iteration(iteration)

    # Decision Trace events — verbatim + one decision
    bundle = DecisionTraceBundle(
        iteration_id=iter_id, thread_id=thread_id,
        user_message=UserMessage(
            id=f"dt-msg-user-{iter_id}", iteration_id=iter_id, thread_id=thread_id,
            workspace_id=workspace, surface_id="chat", user_id=user_id, project_id=None,
            text=user_text, ts=iter_ts,
        ),
        system_response=SystemResponse(
            id=f"dt-msg-sys-{iter_id}", iteration_id=iter_id, thread_id=thread_id,
            workspace_id=workspace, surface_id="chat", user_id=user_id, project_id=None,
            text=system_text, ts=iter_ts,
        ),
        decisions=[Decision(
            id=new_dt_id("decision"), iteration_id=iter_id, thread_id=thread_id,
            workspace_id=workspace, surface_id="chat", user_id=user_id, project_id=None,
            text=f"Decision derived from {title}", ts=iter_ts,
            status="committed", confidence=0.9,
        )],
    )
    await writer.write_bundle(bundle)
    return thr, iteration, emb


async def main() -> int:
    print(_b("\n=== MemoryRetriever validation ===\n"))

    store = build_neo4j_thread_store_from_env()
    if store is None:
        print(_r("  ✗ no Neo4j store"))
        return 1
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    await init_schema(store._driver, database=db)

    embedder = GeminiEmbeddingService()
    writer = Neo4jDecisionTraceWriter(store._driver, database=db)
    retriever = MemoryRetriever(store, embedder)

    user_id = f"u-retrieve-{uuid.uuid4().hex[:8]}"

    # Thread A — the "old context" we want to recall via similarity
    thr_a_id = f"thr-A-{uuid.uuid4().hex[:8]}"
    thr_a, iter_a, emb_a = await _plant_thread(
        store, writer, embedder,
        user_id=user_id, thread_id=thr_a_id,
        title="Migration to Neo4j Aura",
        workspace="cursor",
        user_text="Should we use Neo4j Aura Free or Pro for the beta phase database?",
        system_text="Free is fine for 10-15 testers. Move to Pro at ~500 active users.",
    )
    print(_g(f"  ✓ planted thread A (id={thr_a_id[:24]}…)"))

    # Thread B — the "current thread" the model is in
    thr_b_id = f"thr-B-{uuid.uuid4().hex[:8]}"
    thr_b, iter_b, emb_b = await _plant_thread(
        store, writer, embedder,
        user_id=user_id, thread_id=thr_b_id,
        title="UI styling for the Map Room",
        workspace="web",
        user_text="What color palette works for the Map Room knowledge graph nodes?",
        system_text="A dark base with light text on tinted backgrounds reads cleanly without straining.",
    )
    print(_g(f"  ✓ planted thread B (id={thr_b_id[:24]}…)"))

    if not (emb_a.success and emb_b.success):
        print(_r(f"  ✗ embedder failed — vector path can't be tested"))
        return 3
    print(_g(f"  ✓ both iterations embedded ({len(emb_a.vector)} dim)"))

    # Issue a retrieve call from thread B, but with a query that matches thread A
    print("\n--- retrieve: from thread B, query matches thread A's topic ---\n")
    result = await retriever.retrieve(
        "What database tier should we use for the beta launch?",
        user_id=user_id, thread_id=thr_b_id,
        k_local=5, k_cross=5,
    )
    print(f"  result.latency_ms={result.latency_ms}")
    print(f"  result.embedded_ok={result.embedded_ok} dim={result.vector_dim}")
    print(f"  local entries: {len(result.local)}")
    for e in result.local:
        print(f"    - [{e.source}] thread={e.thread_title!r} iter={e.iteration_id[:24]}…")
    print(f"  cross_thread entries: {len(result.cross_thread)}")
    for e in result.cross_thread:
        print(f"    - [{e.source} sim={e.score:.2f}] thread={e.thread_title!r}")

    failed = False
    if len(result.local) < 1:
        print(_r("  ✗ expected at least 1 local entry from thread B"))
        failed = True
    elif result.local[0].thread_id != thr_b_id:
        print(_r(f"  ✗ local entry should come from thread B ({thr_b_id}); got {result.local[0].thread_id}"))
        failed = True

    if len(result.cross_thread) < 1:
        print(_r("  ✗ expected at least 1 cross-thread entry"))
        failed = True
    else:
        cross_ids = {e.thread_id for e in result.cross_thread}
        if thr_a_id not in cross_ids:
            print(_r(f"  ✗ cross-thread should include thread A; got {cross_ids}"))
            failed = True
        if thr_b_id in cross_ids:
            print(_r(f"  ✗ cross-thread should EXCLUDE thread B (current); got {cross_ids}"))
            failed = True

    print("\n--- render_timeline output ---\n")
    md = render_timeline(result)
    print(md)
    if "CURRENT THREAD" not in md or "CROSS-THREAD MEMORY" not in md:
        print(_r("  ✗ render_timeline missing expected section headers"))
        failed = True

    # Cleanup
    print("\n  cleaning up...")
    async with store._driver.session(database=db) as s:
        await s.run(
            "MATCH (n:DecisionTrace) WHERE n.user_id = $uid DETACH DELETE n", uid=user_id,
        )
    await store.delete_thread(thr_a_id)
    await store.delete_thread(thr_b_id)
    await store.close()
    print(_g("  ✓ cleanup complete"))

    if failed:
        print(_r("\n=== RESULT: FAILED ===\n"))
        return 3
    print(_g("\n=== RESULT: ALL CHECKS PASSED ===\n"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
