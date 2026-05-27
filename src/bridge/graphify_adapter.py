"""
Graphify adapter — real wrapper around the vendored graphify graph.

Reads graphify-out/graph.json directly. The graph file is the output of
`graphify extract .` (or `/graphify .` inside an AI assistant). Format:
node_link JSON with nodes + edges (a.k.a. links), where nodes carry
source_file/label/line metadata and edges carry a context tag like
"call", "import", "field", "parameter_type", "return_type",
"generic_arg".

Why direct JSON parsing rather than importing graphify.serve:
- Avoids pulling networkx as a hard dependency of the bridge.
- The graphify graph.json schema is stable and small enough to parse
  with plain dict ops.
- Keeps the adapter dependency-free; if graphify changes its internal
  query helpers, this wrapper is unaffected.

If graphify-out/graph.json does not exist, methods raise FileNotFoundError
with the exact CLI command needed to build it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from src.bridge.types import CodeRef


# Default location relative to repo root. Matches the graphify CLI default.
_GRAPH_RELPATH = "graphify-out/graph.json"

# Edge context labels we recognise (from graphify.serve._CONTEXT_HINTS).
_CTX_CALL = "call"
_CTX_IMPORT = "import"


class GraphifyAdapter:
    """
    Real adapter over a graphify graph.json file.

    Construct once per repo_root. The graph is loaded lazily on the
    first query so that adapter instantiation is cheap and instantiating
    it without a built graph is not an immediate failure (the failure
    happens on first read, with a clear message).

    All methods are async to match the engine's async patterns even
    though the underlying JSON read is synchronous.
    """

    def __init__(self, repo_root: str, graph_relpath: str = _GRAPH_RELPATH):
        self.repo_root = Path(repo_root).resolve()
        self.graph_path = self.repo_root / graph_relpath
        self._graph: dict[str, Any] | None = None
        self._nodes_by_id: dict[str, dict[str, Any]] | None = None
        self._nodes_by_file: dict[str, list[dict[str, Any]]] | None = None
        self._edges_out: dict[str, list[dict[str, Any]]] | None = None
        self._edges_in: dict[str, list[dict[str, Any]]] | None = None

    # -----------------------------------------------------------------------
    # Graph loading / indexing
    # -----------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the graph and build indices on first use."""
        if self._graph is not None:
            return

        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"graphify graph not found at {self.graph_path}. "
                f"Build it with: cd {self.repo_root} && graphify extract ."
            )

        try:
            data = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"graphify graph at {self.graph_path} is corrupted ({e}). "
                "Rebuild with: graphify extract . --force"
            ) from e

        # graphify uses either "links" or "edges" depending on version.
        edges = data.get("links") or data.get("edges") or []
        nodes = data.get("nodes") or []

        self._graph = data

        nodes_by_id: dict[str, dict[str, Any]] = {}
        nodes_by_file: dict[str, list[dict[str, Any]]] = {}
        for n in nodes:
            nid = n.get("id")
            if nid is None:
                continue
            nodes_by_id[nid] = n
            sf = n.get("source_file")
            if sf:
                nodes_by_file.setdefault(sf, []).append(n)

        edges_out: dict[str, list[dict[str, Any]]] = {}
        edges_in: dict[str, list[dict[str, Any]]] = {}
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            if src is None or tgt is None:
                continue
            edges_out.setdefault(src, []).append(e)
            edges_in.setdefault(tgt, []).append(e)

        self._nodes_by_id = nodes_by_id
        self._nodes_by_file = nodes_by_file
        self._edges_out = edges_out
        self._edges_in = edges_in

    # -----------------------------------------------------------------------
    # Public queries (async wrappers around sync JSON ops)
    # -----------------------------------------------------------------------

    async def get_code_structure(self, file_path: str) -> dict:
        """
        Return the nodes + outgoing edges anchored to a single file.

        Shape:
            {
                "file": "<file_path>",
                "nodes": [ <raw graphify node dict>, ... ],
                "edges": [ <raw graphify edge dict>, ... ],
                "node_count": int,
                "edge_count": int,
            }

        Yields control once via asyncio.sleep(0) so this method composes
        cleanly with concurrent agent calls.
        """
        await asyncio.sleep(0)
        self._ensure_loaded()
        assert self._nodes_by_file is not None and self._edges_out is not None

        file_nodes = self._nodes_by_file.get(file_path, [])
        file_edges: list[dict[str, Any]] = []
        for n in file_nodes:
            nid = n.get("id")
            if nid is None:
                continue
            file_edges.extend(self._edges_out.get(nid, []))

        return {
            "file": file_path,
            "nodes": file_nodes,
            "edges": file_edges,
            "node_count": len(file_nodes),
            "edge_count": len(file_edges),
        }

    async def get_callers_of(self, symbol: str) -> list[CodeRef]:
        """
        Find code locations that CALL a symbol.

        Resolves the symbol to one or more graphify nodes (matching on
        node id or label), then follows incoming edges with context ==
        "call" back to caller nodes, and returns each caller's source
        location as a CodeRef.

        Returns an empty list if the symbol is not in the graph.
        """
        await asyncio.sleep(0)
        self._ensure_loaded()
        assert (
            self._nodes_by_id is not None
            and self._edges_in is not None
        )

        target_ids = self._resolve_symbol_to_ids(symbol)
        if not target_ids:
            return []

        seen: set[str] = set()
        refs: list[CodeRef] = []
        for tid in target_ids:
            for edge in self._edges_in.get(tid, []):
                if edge.get("context") != _CTX_CALL:
                    continue
                src_id = edge.get("source")
                if src_id is None or src_id in seen:
                    continue
                seen.add(src_id)
                src_node = self._nodes_by_id.get(src_id)
                if src_node is None:
                    continue
                ref = self._node_to_coderef(src_node)
                if ref is not None:
                    refs.append(ref)
        return refs

    async def get_dependencies_of(self, file_path: str) -> list[CodeRef]:
        """
        Find code locations a given file DEPENDS ON.

        Follows outgoing edges from every node in the file (any context,
        not just "import") and returns deduplicated CodeRefs to the
        target nodes that live in other files.

        Returns an empty list if the file has no nodes in the graph.
        """
        await asyncio.sleep(0)
        self._ensure_loaded()
        assert (
            self._nodes_by_id is not None
            and self._nodes_by_file is not None
            and self._edges_out is not None
        )

        seen: set[str] = set()
        refs: list[CodeRef] = []
        for n in self._nodes_by_file.get(file_path, []):
            nid = n.get("id")
            if nid is None:
                continue
            for edge in self._edges_out.get(nid, []):
                tgt_id = edge.get("target")
                if tgt_id is None or tgt_id in seen:
                    continue
                tgt_node = self._nodes_by_id.get(tgt_id)
                if tgt_node is None:
                    continue
                # Skip same-file targets — those are internal structure,
                # not dependencies.
                if tgt_node.get("source_file") == file_path:
                    continue
                seen.add(tgt_id)
                ref = self._node_to_coderef(tgt_node)
                if ref is not None:
                    refs.append(ref)
        return refs

    async def get_node(self, node_id: str) -> dict | None:
        """Return the raw graphify node dict for an id, or None."""
        await asyncio.sleep(0)
        self._ensure_loaded()
        assert self._nodes_by_id is not None
        return self._nodes_by_id.get(node_id)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _resolve_symbol_to_ids(self, symbol: str) -> list[str]:
        """
        Map a symbol name to graphify node ids.

        Match priority: exact id → exact label → normalized label →
        endswith on id. Returns all nodes that match the strongest
        available tier.

        Normalized label matching handles graphify v0.8.x's convention of
        decorating method labels as ".store_decision()" — strip the
        leading dot and trailing parens before comparing.
        """
        assert self._nodes_by_id is not None
        if symbol in self._nodes_by_id:
            return [symbol]

        # Exact label
        exact_label = [
            nid for nid, n in self._nodes_by_id.items()
            if n.get("label") == symbol
        ]
        if exact_label:
            return exact_label

        # Normalized label — strip leading "." and trailing "()"
        normalized = [
            nid for nid, n in self._nodes_by_id.items()
            if _normalize_label(n.get("label")) == symbol
        ]
        if normalized:
            return normalized

        # Suffix match on id (handles "Module.func" → "func")
        suffix = [
            nid for nid, n in self._nodes_by_id.items()
            if nid.endswith("." + symbol)
            or nid.endswith(":" + symbol)
            or nid.endswith("_" + symbol)
        ]
        return suffix

    def _node_to_coderef(self, node: dict) -> CodeRef | None:
        """
        Convert a graphify node dict into a CodeRef.

        Returns None if the node has no source_file (e.g. a pure concept
        node from a doc-only extraction).

        Line numbers can arrive in any of these forms across graphify
        versions: numeric `line_start`/`line_end`/`line` (older), or a
        single string `source_location` like "L42" or "L42-58" (v0.8.x).
        We accept all of them.
        """
        sf = node.get("source_file")
        if not sf:
            return None

        line_start, line_end = self._extract_lines(node)
        return CodeRef(
            file_path=sf,
            line_start=line_start,
            line_end=line_end,
            symbol_name=_normalize_label(node.get("label")) or node.get("label"),
            symbol_type=node.get("kind") or node.get("type"),
        )

    @classmethod
    def _extract_lines(cls, node: dict) -> tuple[int, int]:
        """
        Best-effort line range extraction across schema versions.

        Falls back to (1, 1) if nothing parsable is present.
        """
        # Older schema: numeric fields
        start = cls._safe_int(node.get("line_start") or node.get("line"))
        end = cls._safe_int(
            node.get("line_end") or node.get("line_start") or node.get("line")
        )

        # v0.8.x: source_location like "L42" or "L42-58"
        if start is None:
            loc = node.get("source_location")
            if isinstance(loc, str) and loc.startswith("L"):
                stripped = loc[1:]
                if "-" in stripped:
                    a, b = stripped.split("-", 1)
                    start = cls._safe_int(a)
                    end = cls._safe_int(b) if end is None else end
                else:
                    start = cls._safe_int(stripped)

        if start is None:
            start = 1
        if end is None or end < start:
            end = start
        return start, end

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


def _normalize_label(label: Any) -> str | None:
    """
    Strip graphify's decorative label syntax for symbol matching.

    Examples:
        ".store_decision()" → "store_decision"
        "MemoryAdapter"      → "MemoryAdapter"
        "main()"             → "main"
        None                 → None
    """
    if not isinstance(label, str):
        return None
    s = label.strip()
    if s.startswith("."):
        s = s[1:]
    if s.endswith("()"):
        s = s[:-2]
    return s or None
