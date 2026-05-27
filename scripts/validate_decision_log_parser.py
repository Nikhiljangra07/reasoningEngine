"""
validate_decision_log_parser.py — Phase 2c regression test.

Walks the full Phase 2c flow:
  1. Build a synthetic markdown decision log (CLAUDE.md style).
  2. Parse it via MarkdownDecisionLogExtractor → list[DecisionAnchor].
  3. Persist each anchor via Neo4jAnchorBackend (the bridge backend we
     migrated earlier, parity-verified).
  4. Read back via list_for_project, verify shape.
  5. Re-parse + re-put — verify IDs are stable (no duplicates created).
  6. Cleanup.

USAGE
=====
  cd ~/Desktop/reasoningEngine
  source .venv/bin/activate
  python scripts/validate_decision_log_parser.py

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
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.bridge.decision_log_parser import MarkdownDecisionLogExtractor
from src.bridge.neo4j_backend import (
    Neo4jAnchorBackend,
    build_neo4j_driver_from_env,
    init_schema,
)


def _g(s): return f"\033[32m{s}\033[0m"
def _r(s): return f"\033[31m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"


SAMPLE_MD = """
# Constellax — Decision Log

## OVERVIEW
This section is intentionally non-decision-bearing.

## DECISIONS

### Migrate to Neo4j Aura
- status: settled
- rationale: GDS + native vector indexes consolidate two services
- date: 2026-05-27
- tags: db, migration, infra
- evidence:
    - 18 parity checks pass
    - Aura Free fits beta workload (10-15 testers)
    - Single auth + backup story

### Build Map Room visualizer
- status: settled
- rationale: User shouldn't see raw nodes / searches
- date: 2026-05-25
- tags: ui, visualization

### Pursue Decision Trace infrastructure
- status: open
- rationale: Foundational memory pipeline
- date: 2026-05-27
- tags: memory, architecture

## NOTES
Unrelated trailing prose — should not appear in output.
"""


async def main() -> int:
    print(_b("\n=== Markdown Decision Log → Neo4j parity validation ===\n"))

    built = build_neo4j_driver_from_env()
    if built is None:
        print(_r("  ✗ Could not build Neo4j driver — set NEO4J_URI / NEO4J_PASSWORD"))
        return 1
    driver, database = built
    print(_g(f"  ✓ Driver ready (database={database})"))

    try:
        await init_schema(driver, database=database)
        print(_g("  ✓ Schema init OK"))
    except Exception as e:
        print(_r(f"  ✗ Schema init failed: {e}"))
        await driver.close()
        return 1

    project_id = f"decision_log_parity_{uuid.uuid4().hex[:6]}"
    parser = MarkdownDecisionLogExtractor()
    anchors = parser.parse(SAMPLE_MD, project_id=project_id)
    print(f"  parsed {len(anchors)} anchors:")
    for a in anchors:
        print(f"    {a.id}  {a.title!r}  status={a.status} tags={a.tags} evidence={len(a.evidence)}")

    failed = False
    if len(anchors) != 3:
        print(_r(f"  ✗ expected 3 decisions, got {len(anchors)}"))
        failed = True

    backend = Neo4jAnchorBackend(driver, database=database)

    # 1) Initial write — all anchors land in Neo4j
    for a in anchors:
        await backend.put(project_id, a)
    read_back = await backend.list_for_project(project_id)
    if len(read_back) != len(anchors):
        print(_r(f"  ✗ read-back count mismatch: stored {len(anchors)}, read {len(read_back)}"))
        failed = True
    else:
        print(_g(f"  ✓ {len(anchors)} anchors round-tripped through Neo4j"))

    # 2) Idempotency — re-parse + re-put should NOT create duplicates
    anchors2 = parser.parse(SAMPLE_MD, project_id=project_id)
    for a in anchors2:
        await backend.put(project_id, a)
    read_back2 = await backend.list_for_project(project_id)
    if len(read_back2) != len(anchors):
        print(_r(f"  ✗ idempotency broken: after re-parse+re-put, found {len(read_back2)} anchors (expected {len(anchors)})"))
        failed = True
    else:
        print(_g(f"  ✓ Idempotency: re-parse + re-put produces no duplicates"))

    # 3) Stable IDs — first and second parse pass produce identical IDs
    if [a.id for a in anchors] != [b.id for b in anchors2]:
        print(_r("  ✗ stable-id invariant broken: re-parse produced different IDs"))
        failed = True
    else:
        print(_g("  ✓ Stable IDs across re-parse"))

    # 4) Cleanup
    async with driver.session(database=database) as s:
        await s.run(
            "MATCH (d:DecisionAnchor) WHERE d.project_id = $pid DETACH DELETE d",
            pid=project_id,
        )
    print(_g("  ✓ Cleanup complete"))
    await driver.close()

    if failed:
        print(_r("\n=== RESULT: FAILED ===\n"))
        return 3
    print(_g("\n=== RESULT: ALL CHECKS PASSED ===\n"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
