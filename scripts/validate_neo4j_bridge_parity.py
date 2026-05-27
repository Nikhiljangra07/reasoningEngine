"""
validate_neo4j_bridge_parity.py — pre-cutover sanity check for the
ConversationStore + DecisionAnchor migration from Redis to Neo4j.

WHAT THIS DOES
==============
1. Connects to Neo4j using NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD.
2. Runs schema init (creates constraints for the bridge entities too).
3. Writes a synthetic DecisionAnchor, Session, Iteration, TurningPoint,
   DecisionLink against BOTH the Neo4j backends and the in-memory
   backends.
4. Reads back from both and diffs the protocol-level CRUD results.
5. Cleans up the synthetic data on success.

This is a sibling of `validate_neo4j_parity.py` (which covers
ThreadStore). Run BOTH before flipping CONSTELLAX_DB_BACKEND=neo4j
for the full conversation-store cutover.

USAGE
=====
  cd ~/Desktop/reasoningEngine
  source .venv/bin/activate
  python scripts/validate_neo4j_bridge_parity.py

EXIT CODES
==========
  0 — all checks passed; bridge backends are ready for cutover
  1 — connection failure or env vars missing
  2 — schema init failed
  3 — parity check failed (read-back differs from in-memory)
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
    Neo4jAnchorBackend,
    Neo4jConversationBackend,
    build_neo4j_driver_from_env,
    init_schema,
)
from src.bridge.redis_backend import (
    InMemoryAnchorBackend,
    InMemoryConversationBackend,
)
from src.bridge.types import (
    CodeRef,
    DecisionAnchor,
    DecisionLink,
    Iteration as BridgeIteration,
    Session,
    TurningPoint,
)


def _green(s): return f"\033[32m{s}\033[0m"
def _red(s):   return f"\033[31m{s}\033[0m"
def _bold(s):  return f"\033[1m{s}\033[0m"
def _yellow(s):return f"\033[33m{s}\033[0m"


# ─── Synthetic fixtures ───────────────────────────────────────────────

def make_fixtures(project_id: str) -> dict:
    suffix = uuid.uuid4().hex[:8]
    now = time.time()
    decision = DecisionAnchor(
        id=f"D-parity-{suffix}",
        title="Parity test decision",
        rationale="Verifying Neo4jAnchorBackend round-trip.",
        evidence=["fact one", "fact two"],
        status="OPEN",
        created_at=now,
        code_refs=[CodeRef(file_path="src/foo.py", line_start=10, line_end=20, symbol_name="bar", symbol_type="function")],
        tags=["migration", "graph-db"],
    )
    session = Session(
        id=f"S-parity-{suffix}",
        project_id=project_id,
        title="Parity test session",
        started_at=now,
        iteration_count=0,
    )
    iteration = BridgeIteration(
        id=f"I-parity-{suffix}",
        session_id=session.id,
        sequence_num=1,
        user_text="hello?",
        engine_response="hi there",
        created_at=now,
        route="trivial", effort="low",
    )
    turning_point = TurningPoint(
        id=f"T-parity-{suffix}",
        session_id=session.id,
        iteration_id=iteration.id,
        title="Direction shift",
        description="We pivoted from X to Y.",
        triggered_by_decisions=[decision.id],
        led_to_decisions=[],
        created_at=now,
    )
    link = DecisionLink(
        id=f"L-parity-{suffix}",
        project_id=project_id,
        from_decision_id=decision.id,
        to_decision_id=decision.id,  # self-link (degenerate but legal)
        link_type="depends_on",
        rationale="Self-loop for the parity test only.",
        created_at=now,
    )
    return {
        "decision": decision,
        "session": session,
        "iteration": iteration,
        "turning_point": turning_point,
        "link": link,
    }


# ─── Write + read sequence — works for both backend pairs ─────────────

async def write_and_read(anchor_backend, conv_backend, project_id: str, fx: dict) -> dict:
    # Anchors
    await anchor_backend.put(project_id, fx["decision"])
    # Conversation entities
    await conv_backend.put(project_id, "sessions", fx["session"])
    await conv_backend.put(project_id, "iterations", fx["iteration"])
    await conv_backend.put(project_id, "turning_points", fx["turning_point"])
    await conv_backend.put(project_id, "decision_links", fx["link"])

    # Reads
    got_decision  = await anchor_backend.get(project_id, fx["decision"].id)
    got_session   = await conv_backend.get(project_id, "sessions", fx["session"].id)
    got_iter      = await conv_backend.get(project_id, "iterations", fx["iteration"].id)
    got_tp        = await conv_backend.get(project_id, "turning_points", fx["turning_point"].id)
    got_link      = await conv_backend.get(project_id, "decision_links", fx["link"].id)

    listed_dec    = await anchor_backend.list_for_project(project_id)
    listed_sess   = await conv_backend.list_for_project(project_id, "sessions")
    listed_iter   = await conv_backend.list_for_project(project_id, "iterations")
    listed_tp     = await conv_backend.list_for_project(project_id, "turning_points")
    listed_link   = await conv_backend.list_for_project(project_id, "decision_links")

    # Status update path (anchor-only)
    status_ok     = await anchor_backend.update_status(project_id, fx["decision"].id, "SETTLED")
    after_status  = await anchor_backend.get(project_id, fx["decision"].id)

    return {
        "decision_found":      got_decision is not None,
        "decision_title":      got_decision.title if got_decision else None,
        "decision_tags_count": len(got_decision.tags) if got_decision else 0,
        "decision_code_refs":  len(got_decision.code_refs) if got_decision else 0,
        "session_found":       got_session is not None,
        "session_title":       got_session.title if got_session else None,
        "iteration_found":     got_iter is not None,
        "iteration_seq":       got_iter.sequence_num if got_iter else None,
        "iteration_route":     got_iter.route if got_iter else None,
        "tp_found":            got_tp is not None,
        "tp_title":            got_tp.title if got_tp else None,
        "tp_triggered_count":  len(got_tp.triggered_by_decisions) if got_tp else 0,
        "link_found":          got_link is not None,
        "link_type":           got_link.link_type if got_link else None,
        "listed_decisions":    len(listed_dec),
        "listed_sessions":     len(listed_sess),
        "listed_iterations":   len(listed_iter),
        "listed_turning":      len(listed_tp),
        "listed_links":        len(listed_link),
        "status_update_ok":    status_ok,
        "status_after_update": after_status.status if after_status else None,
    }


# ─── Main ─────────────────────────────────────────────────────────────

async def main() -> int:
    print(_bold("\n=== Neo4j bridge ↔ InMemory parity validation ===\n"))

    print("Step 1: building Neo4j driver from env...")
    built = build_neo4j_driver_from_env()
    if built is None:
        print(_red("  ✗ Could not build Neo4j driver."))
        print("    Required env vars: NEO4J_URI, NEO4J_PASSWORD (NEO4J_USERNAME optional).")
        return 1
    driver, database = built
    print(_green(f"  ✓ Driver constructed for database={database}"))

    # Step 2 — schema init (idempotent; safe to re-run).
    print("\nStep 2: initializing schema (constraints + indexes for bridge entities)...")
    try:
        dim = int(os.environ.get("NEO4J_EMBEDDING_DIM", "1536"))
        await init_schema(driver, database=database, embedding_dim=dim)
        print(_green("  ✓ Schema ready"))
    except Exception as e:
        print(_red(f"  ✗ Schema init failed: {e}"))
        await driver.close()
        return 2

    # Build backends (sharing the one driver)
    neo4j_anchors = Neo4jAnchorBackend(driver, database=database)
    neo4j_conv    = Neo4jConversationBackend(driver, database=database)
    inmem_anchors = InMemoryAnchorBackend()
    inmem_conv    = InMemoryConversationBackend()

    # Step 3 — write + read both backends with separate fixtures and compare.
    print("\nStep 3: write fixtures to both backends and diff read-back...")
    project_id = f"bridge_parity_{int(time.time())}"
    try:
        fx_inmem = make_fixtures(project_id)
        fx_neo4j = make_fixtures(project_id)
        inmem_result = await write_and_read(inmem_anchors, inmem_conv, project_id, fx_inmem)
        neo4j_result = await write_and_read(neo4j_anchors, neo4j_conv, project_id, fx_neo4j)
    except Exception as e:
        print(_red(f"  ✗ Write/read raised: {e}"))
        import traceback; traceback.print_exc()
        await driver.close()
        return 3

    # Step 4 — diff.
    shape_keys = [
        "decision_found", "decision_tags_count", "decision_code_refs",
        "session_found", "iteration_found", "iteration_seq", "iteration_route",
        "tp_found", "tp_triggered_count",
        "link_found", "link_type",
        "listed_decisions", "listed_sessions", "listed_iterations",
        "listed_turning", "listed_links",
        "status_update_ok", "status_after_update",
    ]
    mismatches = []
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

    # Step 5 — cleanup (Neo4j side only; inmem is per-process).
    print("\nStep 4: cleaning up Neo4j test data...")
    try:
        await neo4j_anchors.delete(project_id, fx_neo4j["decision"].id)
        for et, key in [
            ("sessions", "session"), ("iterations", "iteration"),
            ("turning_points", "turning_point"), ("decision_links", "link"),
        ]:
            await neo4j_conv.delete(project_id, et, fx_neo4j[key].id)
        print(_green("  ✓ Cleanup complete"))
    except Exception as e:
        print(_yellow(f"  ⚠ Cleanup hiccup (non-fatal): {e}"))

    await driver.close()

    if mismatches:
        print(_red("\n=== RESULT: PARITY FAILED — do not flip the env var yet. ===\n"))
        return 3
    print(_green("\n=== RESULT: ALL CHECKS PASSED — bridge backends ready for cutover. ===\n"))
    print("Next step: ensure CONSTELLAX_DB_BACKEND=neo4j is set, then restart server.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
