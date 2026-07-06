"""
sketch_governance_controller.py — ROUGH WIRING (thinking artifact, NOT wired).

Makes blend-04 runnable: "divergence collapse is a two-faced signal." This is
R1's formalization (run 20260615-212736, formalize.md) rendered as code,
DECOUPLED from the live pipeline, so we can watch the autonomous-router
controller behave before deciding to build it for real. Touches nothing in
src/. Zero spend — illustrative inputs only.

THE CONTROLLER (R1's formalization of blend-04):
  Two signals govern the wander swarm of N agents with states x_i(t):
    • divergence    d(t) = ½ Σ_{(i,j)∈E} w_ij · ||x_i − x_j||²   (Lyapunov consensus energy)
      trend Δd(t) = d(t) − d(t−1):  contracting (Δd<−ε) | stable (|Δd|≤ε) | expanding
    • shape         s(t) = 1 iff a connected component covers ≥ f·n nodes AND
      contains a BRIDGE edge   (percolation giant-component + bridge theory)
  Decision table — the two-faced reading of contraction:
    contracting & shape    → CLOSE      genuine convergence: stop, emit the skeleton
    contracting & NO shape → ALARM      PREMATURE collapse: re-inject divergence
    stable      & shape    → REALLOCATE skeleton forming: pour agents into it
    else                   → HOLD       keep wandering
  Falsifier (R1): closes when only one condition holds, or fails to close when both.
  Cites: Zeng 2017 (Lyapunov consensus), percolation theory, bridge (graph theory).

WHY IT'S THE CUSHION'S ANSWER: CLOSE = "when to stop", ALARM = "what triggers a
re-route", REALLOCATE = "how to dynamically allocate agents" — the three
self-governance sub-questions, in one controller.

HOME for the real thing: the autonomous ROUTER (Nikhil's reserved Stage-2 work).
This is a CANDIDATE mechanism for it — not the router, and not a judge: it
governs FLOW (stop/continue/allocate), never quality. Human stays the judge.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- signal 1: divergence — Lyapunov consensus energy ----------------------
def divergence(states: dict[int, tuple[float, ...]],
               edges: list[tuple[int, int]],
               weights: dict[tuple[int, int], float] | None = None) -> float:
    """d(t) = ½ Σ_{(i,j)∈edges} w_ij · ||x_i − x_j||².

    REAL WIRING — x_i = embedding of agent i's running finding
    (gemini-embedding-001, already in the memory pipeline); edges = which agents
    are 'in contact' (contribution-board peers). A cheaper proxy already exists:
    the wander's novelty memory / contribution-board overlap IS an inverse
    divergence signal. Here: illustrative 2-D vectors.
    """
    w = weights or {}
    total = 0.0
    for (i, j) in edges:
        d2 = sum((a - b) ** 2 for a, b in zip(states[i], states[j]))
        total += w.get((i, j), 1.0) * d2
    return 0.5 * total


# --- signal 2: structural shape — giant component + bridge -----------------
def _components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict[int, list[int]] = {}
    for x in range(n):
        comps.setdefault(find(x), []).append(x)
    return list(comps.values())


def has_shape(n: int, edges: list[tuple[int, int]], f: float = 0.5) -> bool:
    """s(t): a component covering ≥ f·n nodes that CONTAINS a bridge edge
    (an edge whose removal disconnects that component).

    REAL WIRING — nodes = wander findings/fragments; edges = semantic / shared-
    entity cross-links between them (entity extraction already in the memory
    pipeline). The bridge = the load-bearing link the skeleton hangs on.
    """
    comps = _components(n, edges)
    if not comps:
        return False
    giant = max(comps, key=len)
    if len(giant) < f * n:
        return False
    gset = set(giant)
    g_edges = [(a, b) for (a, b) in edges if a in gset and b in gset]
    base = len(_components(n, edges))
    for e in g_edges:                       # is any giant-edge a bridge?
        if len(_components(n, [x for x in edges if x != e])) > base:
            return True
    return False


# --- the controller: blend-04's truth table --------------------------------
@dataclass
class Decision:
    action: str   # CLOSE | ALARM | REALLOCATE | HOLD
    why: str


def decide(delta_d: float, shape: bool, eps: float) -> Decision:
    contracting = delta_d < -eps
    stable = abs(delta_d) <= eps
    if contracting and shape:
        return Decision("CLOSE", "converged AND structured — stop, emit the skeleton")
    if contracting and not shape:
        return Decision("ALARM", "premature collapse — re-inject divergence (twist framing / spawn fresh angles)")
    if stable and shape:
        return Decision("REALLOCATE", "skeleton forming — pour agents into the giant component")
    return Decision("HOLD", "keep wandering")


# --- blend-01 doubt meter: sterility with K-consecutive hysteresis ----------
def sterility_series(r_series, k=2):
    """blend-01's sterility doubt meter, de-twitched with K-consecutive hysteresis.

    Raw sterile at round t:  r_t > 0  AND  ṙ_t < 0  AND  r̈_t ≤ 0
    (structure-formation positive but DECELERATING — about to stall).

    A single noisy dip raw-fires falsely (R1's own caveat: "ṙ ≤ 0 may be too strict
    for noisy systems"). So the ACTIONABLE 're-inject' signal (`confirmed`) only fires
    once raw-sterile has held for K consecutive rounds — filtering transient dips,
    confirming genuine death. Validated on the tempo-video curve [4,8,9,2,5,6,6,5,4,2]:
    K=2 filters the round-4 dip and confirms the real decline at rounds 9-10.

    Returns one dict per round: {round, r, rdot, rddot, raw_sterile, streak, confirmed}.
    """
    rdot = [r_series[i] - r_series[i - 1] for i in range(1, len(r_series))]
    rddot = [rdot[i] - rdot[i - 1] for i in range(1, len(rdot))]
    out, streak = [], 0
    for t in range(len(r_series)):
        d = rdot[t - 1] if t >= 1 else None
        dd = rddot[t - 2] if t >= 2 else None
        raw = bool(t >= 2 and r_series[t] > 0 and d < 0 and dd <= 0)
        streak = streak + 1 if raw else 0
        out.append({"round": t + 1, "r": r_series[t], "rdot": d, "rddot": dd,
                    "raw_sterile": raw, "streak": streak, "confirmed": raw and streak >= k})
    return out


# --- worked demo: drive the controller through all 4 regimes ----------------
def _demo() -> None:
    EPS, F = 0.5, 0.75   # giant must cover ≥3 of 4 nodes to count as a skeleton
    chain = [(0, 1), (1, 2), (2, 3)]          # path graph: every edge a bridge, giant=all 4
    ring = [(0, 1), (1, 2), (2, 3), (3, 0)]   # cycle: connected but NO bridge
    split = [(0, 1), (2, 3)]                   # two pairs: no giant component

    scenarios = [
        # (label, states@t-1, states@t, edges-for-d, n, shape-graph)
        ("converged + skeleton",
         {0:(0,0),1:(3,0),2:(0,3),3:(3,3)}, {0:(0,0),1:(0.4,0),2:(0,0.4),3:(0.4,0.4)}, chain, 4, chain),
        ("converged, NO skeleton",
         {0:(0,0),1:(3,0),2:(0,3),3:(3,3)}, {0:(0,0),1:(0.4,0),2:(0,0.4),3:(0.4,0.4)}, chain, 4, split),
        ("stable + skeleton",
         {0:(0,0),1:(1,0),2:(0,1),3:(1,1)}, {0:(0,0),1:(1.05,0),2:(0,1.05),3:(1.05,1.05)}, chain, 4, chain),
        ("still expanding",
         {0:(0,0),1:(0.5,0),2:(0,0.5),3:(0.5,0.5)}, {0:(0,0),1:(2,0),2:(0,2),3:(2,2)}, chain, 4, chain),
    ]

    print("blend-04 governance controller — ε=%.2f  f=%.2f\n" % (EPS, F))
    print(f"{'scenario':24} {'d(t-1)':>7} {'d(t)':>7} {'Δd':>7} {'shape':>6}  → action")
    print("-" * 78)
    for label, prev, cur, d_edges, n, shape_g in scenarios:
        d0 = divergence(prev, d_edges)
        d1 = divergence(cur, d_edges)
        dd = d1 - d0
        sh = has_shape(n, shape_g, F)
        dec = decide(dd, sh, EPS)
        print(f"{label:24} {d0:7.2f} {d1:7.2f} {dd:7.2f} {str(sh):>6}  → {dec.action:11} ({dec.why})")


if __name__ == "__main__":
    _demo()
