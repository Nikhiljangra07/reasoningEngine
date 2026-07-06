"""Autonomous-pipeline cross-cycle / cross-run memory, backed by the shared
Constellax Neo4j Aura instance.

SAFETY CONTRACT — every method honors it:
  * FAIL-OPEN. Neo4j is an ACCELERATOR, never a dependency. The JSON artifacts
    under runs/ remain the source of truth. If the driver can't build, the
    instance is paused (AuraDB Free auto-pauses after ~3 idle days), or any query
    throws, this module logs and degrades to a NO-OP — it never raises into the
    pipeline and never hangs it (every DB op is timeout-bounded).
  * NAMESPACED. Writes only :AutonRun / :AutonCard nodes and their OWN vector
    index. It never touches the existing ThreadStore / DecisionTrace / CushionNode
    schema and performs NO DELETE on anything it didn't create.
  * ADDITIVE SCHEMA. Constraints / indexes use IF NOT EXISTS — safe on every call.

Phase 4 wires the WRITE path (record each cycle's cards) + provides recall. The
dispatcher/drift-checker READ wiring lands in a later phase; this module already
exposes recall_similar() / recall_by_cushion() for them.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

log = logging.getLogger("auton_memory")

_CARD_LABEL = "AutonCard"
_RUN_LABEL = "AutonRun"
_GAP_LABEL = "AutonGap"
_CARD_VECTOR_INDEX = "auton_card_embedding_idx"
# Timeouts so a paused / unreachable Aura can never stall a run.
_CONNECT_TIMEOUT_S = float(os.environ.get("AUTON_MEMORY_CONNECT_TIMEOUT_S", "8"))
_OP_TIMEOUT_S = float(os.environ.get("AUTON_MEMORY_OP_TIMEOUT_S", "15"))


class AutonMemory:
    """Fail-open Neo4j memory for the autonomous pipeline. Construct via
    `from_env()`; call `connect_and_init()` once; then `record_cycle()` per cycle
    and `recall_similar()` / `recall_by_cushion()` to read. A no-op instance
    (enabled=False) behaves identically minus persistence — callers need no
    branching."""

    def __init__(self, driver, database: str, *, embedding_dim: int = 1536,
                 fresh: bool = False) -> None:
        self._driver = driver
        self._database = database
        self._dim = embedding_dim
        self.fresh = fresh                # ignore PRIOR memory on recall (still records)
        self.enabled = driver is not None

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def from_env(cls, *, fresh: bool = False) -> "AutonMemory":
        """Build from NEO4J_* env vars. Returns a no-op instance (enabled=False)
        if creds are missing or the driver can't construct — never raises."""
        try:
            dim = int(os.environ.get("NEO4J_EMBEDDING_DIM", "1536") or "1536")
        except ValueError:
            dim = 1536
        built = None
        try:
            from src.bridge.neo4j_backend import build_neo4j_driver_from_env
            built = build_neo4j_driver_from_env()
        except Exception as e:  # import or driver-construct failure
            log.warning("[auton_memory] driver build failed: %s — no-op mode", e)
        if not built:
            return cls(None, "neo4j", embedding_dim=dim, fresh=fresh)
        driver, database = built
        return cls(driver, database, embedding_dim=dim, fresh=fresh)

    # ── lifecycle ───────────────────────────────────────────────────────
    async def connect_and_init(self) -> bool:
        """Verify connectivity (timeout-bounded) + create the Auton* schema
        (idempotent). On ANY failure → disable (no-op) and return False."""
        if not self.enabled:
            return False
        try:
            await asyncio.wait_for(self._init_schema(), timeout=_CONNECT_TIMEOUT_S)
            log.info("[auton_memory] connected + schema ready (db=%s, dim=%d)",
                     self._database, self._dim)
            return True
        except Exception as e:
            log.warning("[auton_memory] connect/init failed (%s) — NO-OP mode; "
                        "JSON remains source of truth", e)
            self.enabled = False
            return False

    async def _init_schema(self) -> None:
        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"CREATE CONSTRAINT auton_card_id IF NOT EXISTS "
                f"FOR (c:{_CARD_LABEL}) REQUIRE c.card_id IS UNIQUE")
            await session.run(
                f"CREATE CONSTRAINT auton_run_id IF NOT EXISTS "
                f"FOR (r:{_RUN_LABEL}) REQUIRE r.run_id IS UNIQUE")
            await session.run(
                f"CREATE CONSTRAINT auton_gap_id IF NOT EXISTS "
                f"FOR (g:{_GAP_LABEL}) REQUIRE g.gap_id IS UNIQUE")
            await session.run(
                f"CREATE VECTOR INDEX {_CARD_VECTOR_INDEX} IF NOT EXISTS "
                f"FOR (c:{_CARD_LABEL}) ON c.embedding "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(self._dim)}, "
                f"`vector.similarity_function`: 'cosine'}}}}")

    async def close(self) -> None:
        if self._driver is not None:
            try:
                await self._driver.close()
            except Exception:
                pass

    # ── write ───────────────────────────────────────────────────────────
    async def record_cycle(self, *, run_id: str, cushion_hash: str, cycle: int,
                           cards, embeddings: list | None = None) -> int:
        """Persist this cycle's cards as :AutonCard nodes under an :AutonRun.
        `embeddings`: optional list aligned 1:1 to `cards` (None entries stored
        without a vector → simply absent from the vector index). Fail-open:
        returns the count written (0 on any failure) and NEVER raises."""
        if not self.enabled or not cards:
            return 0
        rows = []
        for i, c in enumerate(cards):
            rid = str(getattr(c, "report_id", "") or i)
            emb = None
            if embeddings and i < len(embeddings) and embeddings[i]:
                emb = [float(x) for x in embeddings[i]]
            rows.append({
                "card_id": f"{run_id}:{cycle}:{rid}",
                "report_id": rid,
                "bridge": str(getattr(c, "bridge", "") or "")[:4000],
                "cycle": int(cycle),
                "embedding": emb,
            })
        try:
            async def _tx(tx):
                await tx.run(
                    f"MERGE (r:{_RUN_LABEL} {{run_id: $run_id}}) "
                    f"SET r.cushion_hash = $cushion_hash, r.last_cycle = $cycle",
                    run_id=run_id, cushion_hash=cushion_hash, cycle=int(cycle))
                await tx.run(
                    f"UNWIND $rows AS row "
                    f"MERGE (c:{_CARD_LABEL} {{card_id: row.card_id}}) "
                    f"SET c.report_id = row.report_id, c.bridge = row.bridge, "
                    f"    c.cycle = row.cycle, c.run_id = $run_id, "
                    f"    c.cushion_hash = $cushion_hash, c.embedding = row.embedding "
                    f"WITH c MATCH (r:{_RUN_LABEL} {{run_id: $run_id}}) "
                    f"MERGE (r)-[:HAS_CARD]->(c)",
                    rows=rows, run_id=run_id, cushion_hash=cushion_hash)
            async with self._driver.session(database=self._database) as session:
                await asyncio.wait_for(session.execute_write(_tx), timeout=_OP_TIMEOUT_S)
            return len(rows)
        except Exception as e:
            log.warning("[auton_memory] record_cycle failed (%s) — skipped, run continues", e)
            return 0

    # ── read ────────────────────────────────────────────────────────────
    async def recall_similar(self, *, embedding, k: int = 8,
                             cushion_hash: str | None = None,
                             same_cushion_only: bool = False) -> list:
        """Vector recall of prior cards (cross-run by default). fresh=True → []
        (ignore prior memory). Returns [(score, row_dict), …]. Fail-open → []."""
        if not self.enabled or self.fresh or not embedding:
            return []
        try:
            async def _run():
                async with self._driver.session(database=self._database) as session:
                    result = await session.run(
                        f"CALL db.index.vector.queryNodes($idx, $k, $embedding) "
                        f"YIELD node, score "
                        f"RETURN node.card_id AS card_id, node.bridge AS bridge, "
                        f"       node.cushion_hash AS cushion_hash, node.cycle AS cycle, "
                        f"       node.run_id AS run_id, score",
                        idx=_CARD_VECTOR_INDEX, k=int(k), embedding=[float(x) for x in embedding])
                    return [r.data() async for r in result]
            rows = await asyncio.wait_for(_run(), timeout=_OP_TIMEOUT_S)
            if same_cushion_only and cushion_hash:
                rows = [r for r in rows if r.get("cushion_hash") == cushion_hash]
            return [(float(r.get("score") or 0.0), r) for r in rows]
        except Exception as e:
            log.warning("[auton_memory] recall_similar failed (%s) — returning []", e)
            return []

    async def record_gaps(self, *, run_id: str, cushion_hash: str, cycle: int,
                          gaps: list) -> int:
        """Persist dispatched gap KEYS so future cycles/runs don't re-chase them.
        `gaps`: list of (normalized_key, original_text). Fail-open → returns count
        written (0 on failure), never raises."""
        if not self.enabled or not gaps:
            return 0
        rows = [{"gap_id": f"{cushion_hash}:{k}", "gap_key": k, "text": str(t)[:500],
                 "cushion_hash": cushion_hash, "run_id": run_id, "cycle": int(cycle)}
                for k, t in gaps if k]
        if not rows:
            return 0
        try:
            async def _tx(tx):
                await tx.run(
                    f"UNWIND $rows AS row "
                    f"MERGE (g:{_GAP_LABEL} {{gap_id: row.gap_id}}) "
                    f"SET g.gap_key = row.gap_key, g.text = row.text, "
                    f"    g.cushion_hash = row.cushion_hash, g.run_id = row.run_id, "
                    f"    g.cycle = row.cycle", rows=rows)
            async with self._driver.session(database=self._database) as session:
                await asyncio.wait_for(session.execute_write(_tx), timeout=_OP_TIMEOUT_S)
            return len(rows)
        except Exception as e:
            log.warning("[auton_memory] record_gaps failed (%s) — skipped", e)
            return 0

    async def recall_gap_keys(self, *, cushion_hash: str,
                              exclude_run_id: str | None = None) -> set:
        """Normalized keys of gaps ALREADY dispatched for this cushion (cross-run).
        Conservative exact-key dedup — the dispatcher drops only gaps whose key
        matches. fresh=True → empty. Fail-open → empty set."""
        if not self.enabled or self.fresh:
            return set()
        try:
            async def _run():
                async with self._driver.session(database=self._database) as session:
                    result = await session.run(
                        f"MATCH (g:{_GAP_LABEL} {{cushion_hash: $h}}) "
                        f"WHERE $excl IS NULL OR g.run_id <> $excl "
                        f"RETURN g.gap_key AS k",
                        h=cushion_hash, excl=exclude_run_id)
                    return [r["k"] async for r in result]
            keys = await asyncio.wait_for(_run(), timeout=_OP_TIMEOUT_S)
            return {k for k in keys if k}
        except Exception as e:
            log.warning("[auton_memory] recall_gap_keys failed (%s) — returning empty set", e)
            return set()

    async def recall_by_cushion(self, *, cushion_hash: str, limit: int = 50,
                                exclude_run_id: str | None = None) -> list:
        """Text-free recall: prior cards for this cushion (cross-run), newest
        cycle first. The no-embedding fallback for dedup. Fail-open → []."""
        if not self.enabled or self.fresh:
            return []
        try:
            async def _run():
                async with self._driver.session(database=self._database) as session:
                    result = await session.run(
                        f"MATCH (c:{_CARD_LABEL} {{cushion_hash: $h}}) "
                        f"WHERE $excl IS NULL OR c.run_id <> $excl "
                        f"RETURN c.card_id AS card_id, c.bridge AS bridge, "
                        f"       c.cycle AS cycle, c.run_id AS run_id "
                        f"ORDER BY c.cycle DESC LIMIT $lim",
                        h=cushion_hash, lim=int(limit), excl=exclude_run_id)
                    return [r.data() async for r in result]
            return await asyncio.wait_for(_run(), timeout=_OP_TIMEOUT_S)
        except Exception as e:
            log.warning("[auton_memory] recall_by_cushion failed (%s) — returning []", e)
            return []


__all__ = ["AutonMemory"]
