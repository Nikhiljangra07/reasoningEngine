"""
validate_neo4j_anchor_wiring.py — Phase 5 regression test.

End-to-end validation that BridgeClient(mode="live", anchor_backend=...)
actually persists DecisionAnchors to Neo4j across process boundaries.

Two stages:
  1. STAGE A — write
     Build a Neo4jAnchorBackend from env, construct a live BridgeClient
     with a unique project_id, store two DecisionAnchors (one OPEN, one
     SETTLED). Confirm get_decision / list / find_similar all return them
     within the same process.

  2. STAGE B — read back through a FRESH driver + adapter
     Close the first driver. Build a SECOND backend (new connection pool)
     and confirm the anchors written in Stage A are still there. This is
     the "process restart" simulation — proves Neo4j is the source of
     truth, not in-process state.

Also re-validates the half-live refusal:
  - BridgeClient(mode="live") without anchor_backend MUST raise ValueError.
  - This is the guard against silently degrading to in-memory in production.

USAGE
=====
  cd ~/Desktop/reasoningEngine
  source .venv/bin/activate
  python scripts/validate_neo4j_anchor_wiring.py

EXIT CODES
==========
  0 — all checks passed
  1 — Neo4j env vars missing (NEO4J_URI / NEO4J_PASSWORD unset)
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

from src.bridge import BridgeClient
from src.bridge.neo4j_backend import (
    Neo4jAnchorBackend,
    build_neo4j_driver_from_env,
)
from src.bridge.types import CodeRef, DecisionAnchor


def _g(s): return f"\033[32m{s}\033[0m"
def _r(s): return f"\033[31m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"


def _make_decision(decision_id: str, title: str, status: str = "OPEN") -> DecisionAnchor:
    return DecisionAnchor(
        id=decision_id,
        title=title,
        rationale=(
            "Phase 5 validation — anchor persisted via Neo4jAnchorBackend "
            "and recalled through BridgeClient(mode='live')."
        ),
        evidence=["scripts/validate_neo4j_anchor_wiring.py"],
        status=status,
        created_at=time.time(),
        code_refs=[
            CodeRef(
                file_path="src/bridge/client.py",
                line_start=39,
                line_end=120,
                symbol_name="BridgeClient",
                symbol_type="class",
            ),
        ],
        tags=["phase-5", "anchor-persistence", "neo4j"],
    )


async def main() -> int:
    print(_b("\n=== Neo4jAnchorBackend wiring validation ===\n"))

    # ── Guard 0: refusal to build half-live ─────────────────────────────
    try:
        BridgeClient(repo_root=".", mode="live")
    except ValueError as e:
        if "anchor_backend" not in str(e):
            print(_r(f"  ✗ wrong error message: {e}"))
            return 3
        print(_g("  ✓ live mode without anchor_backend → ValueError (good)"))
    else:
        print(_r("  ✗ live mode without anchor_backend did NOT raise — silent in-memory degrade is unsafe"))
        return 3

    # ── Build the first driver / backend ────────────────────────────────
    built = build_neo4j_driver_from_env()
    if built is None:
        print(_r("  ✗ NEO4J_URI / NEO4J_PASSWORD missing — cannot validate"))
        return 1
    driver_a, database = built
    backend_a = Neo4jAnchorBackend(driver_a, database=database, owns_driver=False)
    print(_g(f"  ✓ built Neo4jAnchorBackend (database={database})"))

    project_id = f"proj-phase5-{uuid.uuid4().hex[:8]}"
    print(f"  using project_id={project_id}")

    bridge_a = BridgeClient(
        repo_root=".",
        mode="live",
        project_id=project_id,
        anchor_backend=backend_a,
    )
    print(_g(f"  ✓ BridgeClient(mode='live') constructed with injected backend"))

    # ── STAGE A — write + same-process recall ────────────────────────────
    d_open = _make_decision("dt-phase5-open", "Phase 5 open anchor", status="OPEN")
    d_settled = _make_decision("dt-phase5-settled", "Phase 5 settled anchor", status="SETTLED")
    await bridge_a.store_decision(d_open)
    await bridge_a.store_decision(d_settled)
    print(_g("  ✓ wrote 2 anchors via live BridgeClient"))

    # In-process readback
    fetched = await bridge_a.get_decision(d_open.id)
    if fetched is None or fetched.title != d_open.title:
        print(_r(f"  ✗ same-process get_decision returned {fetched!r}"))
        await driver_a.close()
        return 3
    print(_g("  ✓ same-process get_decision round-tripped"))

    touched = await bridge_a.get_decisions_touching_file("src/bridge/client.py")
    if len(touched) < 2:
        print(_r(f"  ✗ same-process get_decisions_touching_file returned {len(touched)}, expected ≥ 2"))
        await driver_a.close()
        return 3
    print(_g(f"  ✓ same-process get_decisions_touching_file → {len(touched)} hit(s)"))

    # ── Close first driver to simulate restart ──────────────────────────
    await driver_a.close()
    print(_g("  ✓ closed first driver (simulating process restart)"))

    # ── STAGE B — fresh driver, same project_id, must see prior writes ──
    built2 = build_neo4j_driver_from_env()
    if built2 is None:
        print(_r("  ✗ failed to rebuild driver"))
        return 3
    driver_b, _ = built2
    backend_b = Neo4jAnchorBackend(driver_b, database=database, owns_driver=False)
    bridge_b = BridgeClient(
        repo_root=".",
        mode="live",
        project_id=project_id,
        anchor_backend=backend_b,
    )
    print(_g("  ✓ fresh driver + fresh BridgeClient bound to same project_id"))

    after_restart = await bridge_b.get_decision(d_open.id)
    if after_restart is None:
        print(_r("  ✗ post-restart get_decision returned None — anchor did NOT persist"))
        await driver_b.close()
        return 3
    if after_restart.status != "OPEN" or after_restart.title != d_open.title:
        print(_r(f"  ✗ post-restart anchor mismatch: {after_restart}"))
        await driver_b.close()
        return 3
    print(_g("  ✓ anchor survived restart (status + title intact)"))

    # update_status — exercises the read-mutate-write path
    await bridge_b.update_decision_status(d_open.id, "SETTLED")
    updated = await bridge_b.get_decision(d_open.id)
    if updated is None or updated.status != "SETTLED":
        print(_r(f"  ✗ update_decision_status did not persist: {updated}"))
        await driver_b.close()
        return 3
    print(_g("  ✓ update_decision_status persisted via Neo4j"))

    # find_similar_decisions — keyword path (default scorer)
    similar = await bridge_b.find_similar_decisions("phase 5 anchor settled", k=5)
    if not similar:
        print(_r("  ✗ find_similar_decisions returned no hits"))
        await driver_b.close()
        return 3
    print(_g(f"  ✓ find_similar_decisions → {len(similar)} hit(s)"))

    # ── Project isolation: a different project_id must see NOTHING ──────
    bridge_other = BridgeClient(
        repo_root=".",
        mode="live",
        project_id=f"proj-other-{uuid.uuid4().hex[:8]}",
        anchor_backend=backend_b,
    )
    leak = await bridge_other.get_decision(d_open.id)
    if leak is not None:
        print(_r(f"  ✗ project isolation breached: other project saw {leak.id}"))
        await driver_b.close()
        return 3
    print(_g("  ✓ project isolation holds (other project_id sees nothing)"))

    # ── Cleanup ─────────────────────────────────────────────────────────
    print("\n  cleaning up...")
    async with driver_b.session(database=database) as s:
        await s.run(
            "MATCH (d:DecisionAnchor {project_id: $pid}) DETACH DELETE d",
            pid=project_id,
        )
    await driver_b.close()
    print(_g("  ✓ cleanup complete"))

    print(_g("\n=== RESULT: ALL CHECKS PASSED ===\n"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
