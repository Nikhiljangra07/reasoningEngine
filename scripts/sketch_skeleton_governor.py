"""
sketch_skeleton_governor.py — SCAFFOLD (test harness, NOT wired into the pipeline).

Tests the skeleton-governor idea on REAL data, ZERO spend, by replaying a finished
run's blends as the skeleton graph. Composes the pieces we settled on:

  • blend-04 controller (decide: CLOSE / ALARM / REALLOCATE / HOLD)
        — reused from sketch_governance_controller.py (single source of truth)
  • the EDGE DETECTOR = the BLENDER ITSELF: a blend's source_card_ids are edges,
        and emergent_structure (vs an empty 'merge') = emergence vs reinforcement.
        No invented primitive — this is the repo-native notion of cross-connection.
  • blend-02 completeness (a giant connected component) + a bridge = the shape s(t).
  • Halo overlay: audit.json blind spots shown beside the structure (the Halo→governor
        link, in its read-only/observe form for the replay).

WHAT THIS TESTS (static replay, all MEASURED from existing data):
  skeleton graph, % cohering, giant component, bridges/hubs, emergence/reinforcement
  split, outliers (the divergence reserve), Halo overlay.

WHAT IT HONESTLY DOES NOT TEST:
  the TREND half — divergence trend Δd (blend-04) and the sterility doubt meter
  (blend-01) are RATE signals; they need a LIVE ROUNDS run, not a static snapshot.
  So the verdict here is shown CONDITIONAL on the (unmeasured) trend, never faked.

Touches nothing live. No API calls. Reads one run dir.

Usage: PYTHONPATH=. python scripts/sketch_skeleton_governor.py runs/r-collision/20260615-212736
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sketch_governance_controller import decide   # reuse the validated controller logic


# --- graph helpers (tiny, over card-id nodes) ------------------------------
def _components(nodes: list[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    parent = {x: x for x in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict[str, set[str]] = {}
    for x in nodes:
        comps.setdefault(find(x), set()).add(x)
    return list(comps.values())


def _bridges(nodes: list[str], edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    base = len(_components(nodes, edges))
    out = []
    for e in edges:
        if len(_components(nodes, [x for x in edges if x != e])) > base:
            out.append(e)
    return out


def _blends(col: dict) -> list[dict]:
    b = col.get("blends")
    return b["blends"] if isinstance(b, dict) else (b or [])


def _halo_spots(section) -> list:
    if not isinstance(section, dict):
        return []
    for k in ("blind_spots", "blindspots", "spots", "findings", "holes", "gaps"):
        v = section.get(k)
        if isinstance(v, list):
            return v
    return []


def _spot_text(s) -> str:
    if isinstance(s, str):
        return s
    if isinstance(s, dict):
        for k in ("blind_spot", "description", "text", "summary", "gap", "title", "what"):
            if s.get(k):
                return str(s[k])
        return json.dumps(s)[:160]
    return str(s)


# --- the scaffold ----------------------------------------------------------
def main(run_dir: Path) -> None:
    col = json.loads((run_dir / "collision.json").read_text())
    dos = json.loads((run_dir / "dossier.json").read_text())
    au = json.loads((run_dir / "audit.json").read_text())

    cards = (dos.get("high") or []) + (dos.get("medium") or []) + (dos.get("low") or [])
    nodes = [c["report_id"] for c in cards]
    n = len(nodes)

    # --- EDGES = the blender's blends (repo-native cross-connection) --------
    edge_emergence: dict[frozenset, bool] = {}
    card_blend_count: dict[str, int] = {}
    for b in _blends(col):
        sids = b.get("source_card_ids", [])
        emergence = bool((b.get("emergent_structure") or "").strip())  # empty == merge == reinforcement
        for c in sids:
            card_blend_count[c] = card_blend_count.get(c, 0) + 1
        for a, bb in combinations(sids, 2):
            key = frozenset((a, bb))
            edge_emergence[key] = edge_emergence.get(key, False) or emergence
    edges = [tuple(k) for k in edge_emergence]

    # --- STRUCTURAL cockpit (all measured) ---------------------------------
    cohered = set(card_blend_count)
    comps = _components(nodes, edges)
    giant = max(comps, key=len) if comps else set()
    nontrivial = sorted([c for c in comps if len(c) > 1], key=len, reverse=True)
    hubs = {c: k for c, k in card_blend_count.items() if k > 1}
    bridges = _bridges(nodes, edges)
    n_emergence = sum(1 for v in edge_emergence.values() if v)
    n_reinforce = sum(1 for v in edge_emergence.values() if not v)
    outliers = [x for x in nodes if x not in cohered]

    pct_cohering = 100 * len(cohered) / n
    giant_cov = 100 * len(giant) / n

    W = 74
    print("=" * W)
    print("SKELETON-GOVERNOR SCAFFOLD — static replay (zero spend)")
    print(f"source: {run_dir.name}")
    print("=" * W)

    print("\n[ STRUCTURE — measured from the blender's edges ]")
    print(f"  cards (nodes):            {n}")
    print(f"  edges (blend pairs):      {len(edges)}   emergence={n_emergence} reinforcement={n_reinforce}")
    print(f"  % cohering:               {pct_cohering:.0f}%   ({len(cohered)}/{n} cards in ≥1 blend)")
    print(f"  components (size>1):      {[len(c) for c in nontrivial]}")
    print(f"  giant component:          {len(giant)} cards  (coverage {giant_cov:.0f}% of n)")
    print(f"  hubs (cards in >1 blend): {hubs or 'none'}")
    print(f"  bridges (load-bearing):   {len(bridges)} -> {[tuple(b) for b in bridges] if bridges else 'none'}")
    print(f"  outliers (divergence reserve): {len(outliers)} cards unblended")
    if giant:
        print(f"  giant component members:  {sorted(giant)}")

    # --- VERDICT, conditional on the (unmeasured) trend --------------------
    print("\n[ GOVERNOR VERDICT — shape is measured; trend Δd is NOT (needs live rounds) ]")
    EPS = 0.5
    for f in (0.25, 0.50):
        shape = len(giant) >= f * n
        need = int(f * n + 0.999)  # cards needed for a skeleton at this f
        print(f"  at f={f:.2f} (skeleton needs giant ≥ {need} cards):  shape s(t) = {shape}")
        for label, dd in (("contracting", -1.0), ("stable", 0.0), ("expanding", +1.0)):
            print(f"      if trend={label:11} → {decide(dd, shape, EPS).action}")

    # --- HALO overlay (the Halo→governor link, read-only here) -------------
    print("\n[ HALO OVERLAY — what the blind-spot finder says is missing ]")
    for sect in ("cards_audit", "blends_audit"):
        spots = _halo_spots(au.get(sect))
        print(f"  {sect}: {len(spots)} blind spot(s)")
        for s in spots[:2]:
            print(f"     • {_spot_text(s)[:150]}")

    # --- honest scope note -------------------------------------------------
    print("\n[ SCOPE — what this run validated vs what it did not ]")
    print("  ✓ measured: skeleton graph, cohering %, giant component, hub/bridge, "
          "emergence/reinforcement, outliers, Halo overlay")
    print("  ✗ NOT here: divergence trend Δd and the sterility doubt meter (rate signals)")
    print("            → those need the LIVE ROUNDS run; verdict above is conditional on trend.")
    print("=" * W)


if __name__ == "__main__":
    rd = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if not rd or not rd.is_dir():
        raise SystemExit("pass a run dir, e.g. runs/r-collision/20260615-212736")
    main(rd)
