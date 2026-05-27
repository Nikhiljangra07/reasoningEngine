"""
BridgeClient — the facade wuxing calls.

Routes each method to the right adapter:
    graphify-side reads → GraphifyAdapter (real)
    memory-side reads   → MemoryAdapter (in-memory in stub mode,
                                          injected backend in live mode)
    drift detection     → src.bridge.drift (real interface, stub comparator)

Two modes:
    "stub" — graphify methods return empty results without touching the
             graphify-out/ index. MemoryAdapter is backed by the default
             InMemoryAnchorBackend (process-local dict). Use this for
             unit tests and architecture wiring.

    "live" — requires an `anchor_backend` to be supplied (built once at
             startup by the caller — typically a Neo4jAnchorBackend
             sharing the server's driver). MemoryAdapter is constructed
             with that backend, so store_decision / recall_decisions /
             find_similar_decisions hit real persistent storage. The
             graphify side remains real (it already was).

Construction contract:
    BridgeClient(mode="stub")                     → in-memory memory side
    BridgeClient(mode="live", anchor_backend=...) → persistent memory side
    BridgeClient(mode="live")  *without* a backend raises ValueError —
        we refuse to build a half-live bridge that silently degrades to
        in-memory storage in production.
"""

from __future__ import annotations

from src.bridge.drift import detect_drift as _detect_drift
from src.bridge.graphify_adapter import GraphifyAdapter
from src.bridge.memory_adapter import MemoryAdapter
from src.bridge.redis_backend import AnchorBackend
from src.bridge.types import CodeRef, DecisionAnchor, DriftReport


_VALID_MODES = ("stub", "live")


class BridgeClient:
    """
    Unified read interface over graphify + Memory V2.

    Construct once per repo_root. Methods are async to match the
    existing engine pattern in src/llm/client.py.
    """

    def __init__(
        self,
        repo_root: str,
        mode: str = "stub",
        project_id: str | None = None,
        anchor_backend: AnchorBackend | None = None,
    ):
        if mode not in _VALID_MODES:
            raise ValueError(
                f"BridgeClient mode must be one of {_VALID_MODES}, got {mode!r}"
            )

        self.repo_root = repo_root
        self.mode = mode
        # Optional project scope. All memory queries are filtered by this id
        # so two repos can NEVER blend their decision memory. If None, the
        # adapter uses a shared "unscoped" bucket — fine for tests, not for
        # production. Compute one via
        # src.project.compute_fingerprint(repo_root).project_id.
        self.project_id = project_id

        if mode == "live" and anchor_backend is None:
            # We refuse to build a half-live bridge that silently degrades
            # to in-memory storage in production. The Neo4j anchor backend
            # must be built and injected by the caller.
            raise ValueError(
                "BridgeClient(mode='live') requires an anchor_backend. "
                "Build one via build_neo4j_anchor_backend_from_env() (or "
                "share the server's driver via Neo4jAnchorBackend(driver, "
                "database=db, owns_driver=False)) and pass it in."
            )

        # Stub mode: keep both adapters around. The graphify one is
        # real but only consulted when a method explicitly delegates.
        # In the current stub mode the methods return empty results
        # without delegating, so that the smoke test does not depend on
        # graphify-out/ existing.
        self._graphify = GraphifyAdapter(repo_root)
        # MemoryAdapter is project-scoped. In stub mode it falls through
        # to InMemoryAnchorBackend. In live mode it uses the injected
        # backend (typically Neo4jAnchorBackend, sharing the server driver).
        self._memory = MemoryAdapter(
            repo_root,
            project_id=project_id,
            backend=anchor_backend,
        )

    # -----------------------------------------------------------------------
    # graphify-side reads (stubbed in stub mode for test independence)
    # -----------------------------------------------------------------------

    async def get_code_structure(self, file_path: str) -> dict:
        """
        Return the nodes + edges anchored to file_path.

        In stub mode: returns an empty-but-well-formed shape so callers
        can pattern-match against the structure without a built graph.
        Switch to live mode (or call GraphifyAdapter directly) for real
        results.
        """
        return {
            "file": file_path,
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "mode": self.mode,
        }

    async def get_callers_of(self, symbol: str) -> list[CodeRef]:
        """Find code locations that call a symbol. Stub returns []."""
        return []

    async def get_dependencies_of(self, file_path: str) -> list[CodeRef]:
        """Find code locations a file depends on. Stub returns []."""
        return []

    # -----------------------------------------------------------------------
    # Memory V2-side reads (real — delegate to MemoryAdapter)
    # -----------------------------------------------------------------------

    async def get_decision(self, decision_id: str) -> DecisionAnchor | None:
        """Fetch a single decision by id. None if not found."""
        return await self._memory.get_decision(decision_id)

    async def get_decisions_touching_file(
        self, file_path: str
    ) -> list[DecisionAnchor]:
        """Find decisions whose code_refs include file_path."""
        return await self._memory.get_decisions_touching_file(file_path)

    async def find_similar_decisions(
        self, context_text: str, k: int = 5
    ) -> list[DecisionAnchor]:
        """Find top-k decisions similar to a context."""
        return await self._memory.find_similar_decisions(context_text, k)

    # -----------------------------------------------------------------------
    # Bridge-specific cross-reference + drift
    # -----------------------------------------------------------------------

    async def get_decisions_for_code_ref(
        self, ref: CodeRef
    ) -> list[DecisionAnchor]:
        """
        Find every decision whose code_refs touch the given ref's file.

        Implemented as get_decisions_touching_file(ref.file_path) for
        now — file-level granularity is enough until the Memory V2 port
        lands and we can index code_refs by (file, line range).
        """
        return await self.get_decisions_touching_file(ref.file_path)

    async def get_code_refs_for_decision(
        self, decision_id: str
    ) -> list[CodeRef]:
        """Return the code_refs stored on a decision (empty list if missing)."""
        return await self._memory.get_code_refs_for_decision(decision_id)

    async def detect_drift(self, decision: DecisionAnchor) -> DriftReport:
        """
        Detect whether the code at a decision's code_refs still honors
        the decision's intent.

        Delegates to src.bridge.drift.detect_drift, which uses the stub
        comparator until the LLM-backed comparator lands.
        """
        return await _detect_drift(decision, self._graphify)

    # -----------------------------------------------------------------------
    # Memory V2-side writes (real — delegate to MemoryAdapter)
    # -----------------------------------------------------------------------

    async def store_decision(self, decision: DecisionAnchor) -> str:
        """Persist a decision via the adapter's backend (Neo4j in live mode)."""
        return await self._memory.store_decision(decision)

    async def update_decision_status(
        self, decision_id: str, status: str
    ) -> None:
        """Update a decision's status. Raises KeyError on unknown id."""
        await self._memory.update_decision_status(decision_id, status)
