"""
governor — the live flow-governor for the Wandering Room (Stage-2 autonomous router, first cut).

WHAT IT IS
----------
A SESSION-LEVEL observer that watches findings arrive on the shared noticeboard
and governs FLOW — never quality. It owns one authority in this cut: HALT the
wander early when the swarm has genuinely converged (CLOSE). The other two
readings (ALARM = premature collapse, REALLOCATE = pour agents into the skeleton)
are emitted as ADVISORY records because the re-inject / dynamic-spawn mechanisms
do not exist in the live loop yet — this module never fabricates them.

This is blend-04's controller ("divergence collapse is a two-faced signal") and
blend-01's sterility doubt meter, rendered live, riding on the mini-blender edge
detector (DeepSeek V4 Pro — smoke-test winner: 83% true-edge recall, 17% false
edges; also the wander seat, so the front half runs on one model family).

FLOW NOT JUDGE
--------------
The only decisions are CLOSE | ALARM | REALLOCATE | HOLD — all about whether the
wander keeps running, never about whether a finding is good. The human stays the
sole judge of quality. This is the router's job (Nikhil's reserved Stage-2 work),
and it governs flow only.

CHAOS LAW
---------
The governor reads the noticeboard (PRESENT findings) and compares finding-to-
finding via the mini-blender. It NEVER sees the cushion question, and never feeds
anything back into an agent's wander anchor. Edge detection is structure-only:
"do these two findings share a deep mechanism?" — the future question never enters.

LIVE SIGNALS (no embeddings)
----------------------------
blend-04's decide table needs a divergence trend Δd (Lyapunov, embeddings) and a
structural shape s(t). This cut expresses the table in signals VALIDATED live on
real arrival order, with NO embedding dependency:
  • "contracting" ≈ the emergence-rate confirmed-sterile signal (structure-
    formation dying — the swarm has stopped diverging into new structure).
  • "shape" = the skeleton's giant connected component WITH a bridge (computed
    cheaply from the mini-blender's emergence edges — no embeddings).
Mapping (faithful to blend-04's two-faced reading):
  sterile & shape    → CLOSE       converged AND structured  → HALT (acted on)
  sterile & NO shape → ALARM       premature collapse        → advisory
  fertile & shape    → REALLOCATE  skeleton forming          → advisory
  else               → HOLD        keep wandering
Only CLOSE is acted on (halt). Conservative by construction: strict bridge-edge
shape rarely fires, so the governor errs toward keep-wandering, never toward a
premature halt.

ISOLATION
---------
Self-contained in src/. Owns its own httpx client to OpenRouter (the exact
validated mini-blender path). If OPENROUTER_API_KEY is unset, the governor
disables itself (logs once, never halts) — fail-safe. Holds an asyncio.Lock for
its own skeleton state; never blocks an agent (observe is fired off-lock,
fire-and-forget, by SessionState.post_notice).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("constellax.wandering.governor")


# ---------------------------------------------------------------------------
# Mini-blender edge detector. Model chosen by smoke test (4-way, scripts/
# sketch_miniblender_*.py) on the same gold edges: DeepSeek V4 Pro won —
# 83% true-edge recall, only 17% false edges (qwen 67%/25%, haiku 100%/92%
# blob, kimi-k2.6 17% + ~19s/probe dead). DeepSeek is also the cheapest and
# is ALREADY the wander seat, so the front half (wander + edge detection)
# runs on one model family. Swapped from qwen/qwen3.6-35b-a3b on this basis.
# ---------------------------------------------------------------------------

MINIBLENDER_MODEL = "deepseek/deepseek-v4-pro"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_EDGE_SYSTEM = (
    "You are a STRICT edge detector in a reasoning engine. Given two research findings, "
    "classify their structural relation. Be CONSERVATIVE — most random pairs are unrelated.\n"
    '- "emergence": ONLY if you can NAME a specific shared deep mechanism or tension that '
    "bridges them across different surfaces — a genuine cross-connection that creates something "
    "neither states alone. The bar is HIGH; if you cannot name the precise shared structure, do not use this.\n"
    '- "reinforcement": they restate the SAME idea — near-duplicate, same point.\n'
    '- "unrelated": no specific structural connection. THIS IS THE DEFAULT. If the link is '
    "vague, generic, or merely topical (same broad subject), answer unrelated.\n"
    "Judge STRUCTURE, not topic overlap. When unsure, answer unrelated.\n"
    'Output ONLY compact JSON: {"relation":"emergence|reinforcement|unrelated","confidence":0.0-1.0}'
)


def _parse_relation(txt: str) -> str:
    s = (txt or "").strip()
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    try:
        return str(json.loads(s).get("relation", "?"))
    except Exception:
        return "PARSE_FAIL"


# ---------------------------------------------------------------------------
# Controller math (promoted from scripts/sketch_governance_controller.py)
# ---------------------------------------------------------------------------


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
    """s(t): a component covering >= f*n nodes that CONTAINS a bridge edge
    (an edge whose removal disconnects that component). The bridge is the
    load-bearing link the skeleton hangs on.

    Strict bridge-edge (not articulation-point) by design: this is the
    CONSERVATIVE reading — it rarely fires, so the governor errs toward
    keep-wandering. Loosening to "bridge OR articulation point" is the
    planned refinement once halt behavior is observed on live runs.
    """
    if n <= 0 or not edges:
        return False
    comps = _components(n, edges)
    giant = max(comps, key=len)
    if len(giant) < f * n:
        return False
    gset = set(giant)
    g_edges = [(a, b) for (a, b) in edges if a in gset and b in gset]
    base = len(_components(n, edges))
    for e in g_edges:
        if len(_components(n, [x for x in edges if x != e])) > base:
            return True
    return False


def sterility_series(r_series: list[int], k: int = 2) -> list[dict]:
    """blend-01's sterility doubt meter, de-twitched with K-consecutive hysteresis.

    Raw sterile at round t:  r_t > 0 AND r-dot < 0 AND r-ddot <= 0
    (structure-formation positive but DECELERATING — about to stall).
    The ACTIONABLE 'confirmed' signal only fires once raw-sterile has held for K
    consecutive rounds — filtering transient dips (validated on the tempo curve:
    K=2 filtered the round-4/5/7/10 dips and confirmed the real declines).
    """
    rdot = [r_series[i] - r_series[i - 1] for i in range(1, len(r_series))]
    rddot = [rdot[i] - rdot[i - 1] for i in range(1, len(rdot))]
    out: list[dict] = []
    streak = 0
    for t in range(len(r_series)):
        d = rdot[t - 1] if t >= 1 else None
        dd = rddot[t - 2] if t >= 2 else None
        raw = bool(t >= 2 and r_series[t] > 0 and d is not None and d < 0
                   and dd is not None and dd <= 0)
        streak = streak + 1 if raw else 0
        out.append({"round": t + 1, "r": r_series[t], "rdot": d, "rddot": dd,
                    "raw_sterile": raw, "streak": streak, "confirmed": raw and streak >= k})
    return out


@dataclass
class Decision:
    action: str   # CLOSE | ALARM | REALLOCATE | HOLD
    why: str


def decide_live(*, sterile_confirmed: bool, shape: bool) -> Decision:
    """blend-04's two-faced reading in validated live signals (no embeddings)."""
    if sterile_confirmed and shape:
        return Decision("CLOSE", "converged AND structured — halt, emit the skeleton")
    if sterile_confirmed and not shape:
        return Decision("ALARM", "premature collapse — re-inject divergence (advisory; mechanism not yet live)")
    if (not sterile_confirmed) and shape:
        return Decision("REALLOCATE", "skeleton forming — pour agents into the giant component (advisory; mechanism not yet live)")
    return Decision("HOLD", "keep wandering")


# ---------------------------------------------------------------------------
# The live governor
# ---------------------------------------------------------------------------


def _notice_text(notice) -> str:
    """finding text the mini-blender sees — PRESENT findings only, never the question."""
    return (f"[{getattr(notice, 'domain', '?') or '?'}] {getattr(notice, 'summary', '') or ''} "
            f"| principle: {getattr(notice, 'principle', '') or ''} "
            f"| direction: {getattr(notice, 'direction', '') or ''}")[:700]


@dataclass
class WanderGovernor:
    """Session-level flow governor. One per wander, attached to SessionState.

    Lifecycle:
      - Created in runtime when CONSTELLAX_GOVERNOR=1.
      - SessionState.post_notice fires `observe(notice)` per arrival (off-lock).
      - On a confirmed CLOSE, sets session_state.governor_halt — agents exit
        gracefully at the top of their next loop.
      - runtime calls `governance_record()` for governance.json and `aclose()`
        at session end.
    """

    session_state: object = None
    batch: int = 4                  # findings per emergence-rate "round" (matches the bench)
    cap_per_finding: int = 3        # new x frontier probes per arrival (bounds cost)
    max_probes: int = 150           # hard ceiling on total mini-blender calls (bounds cost ~$0.3)
    hysteresis_k: int = 2           # K-consecutive sterility confirmation
    shape_f: float = 0.5            # giant component must cover >= f * findings

    # --- live skeleton state (guarded by _lock) ---
    _findings: list[tuple[str, str]] = field(default_factory=list)   # (id, text) in arrival order
    _index_of: dict[str, int] = field(default_factory=dict)
    _edges: set[frozenset] = field(default_factory=set)
    _frontier: list[str] = field(default_factory=list)               # ids with >= 1 emergence edge
    _checked: set[frozenset] = field(default_factory=set)
    _round_emergence: int = 0       # emergence count accumulating in the current round
    _since_round: int = 0           # findings seen since the last round boundary
    _r_series: list[int] = field(default_factory=list)
    _probes_used: int = 0
    _confirm_probes: int = 0        # extra reverse-direction probes (bidirectional edge confirm)
    _decisions: list[dict] = field(default_factory=list)
    _halted: bool = False
    _disabled_reason: str = ""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _sem: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(8), repr=False)

    def __post_init__(self) -> None:
        if not (os.getenv("OPENROUTER_API_KEY") or "").strip():
            self._disabled_reason = "OPENROUTER_API_KEY unset"
            log.warning("[governor] disabled: %s (will observe nothing, never halt)", self._disabled_reason)

    # --- mini-blender probe (validated path) ---
    async def _probe(self, a_text: str, b_text: str) -> str:
        if self._client is None:
            self._client = httpx.AsyncClient()
        body = {
            "model": MINIBLENDER_MODEL,
            "messages": [{"role": "system", "content": _EDGE_SYSTEM},
                         {"role": "user", "content": f"Finding A:\n{a_text}\n\nFinding B:\n{b_text}"}],
            "temperature": 0.0, "max_tokens": 2000, "usage": {"include": True},
        }
        headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                   "Content-Type": "application/json"}
        async with self._sem:
            for attempt in range(2):
                try:
                    r = await self._client.post(_OPENROUTER_URL, headers=headers, json=body, timeout=90.0)
                    r.raise_for_status()
                    d = r.json()
                    content = (d["choices"][0]["message"].get("content") or "").strip()
                    if content:
                        return _parse_relation(content)
                except Exception:
                    if attempt == 1:
                        return "ERR"
        return "EMPTY"

    async def _probe_bidir(self, a_text: str, b_text: str) -> str:
        """Position-bias-hardened edge judgment. The single-direction probe is
        order-sensitive (LLM pairwise judges show documented position bias). So:
        probe (A,B); only an "emergence" CLAIM is worth confirming, so confirm it
        with the reverse (B,A) and accept the edge ONLY if BOTH directions agree.
        Disagreement → "unrelated" (conservative, matching the governor's bias).

        LAZY by design: the reverse probe fires only on emergence candidates (a
        minority, since the detector defaults to unrelated), so precision rises at
        near-minimal extra cost. Temp stays 0 — NO self-consistency (that would
        need temp>0 and ~k× cost; deliberately excluded)."""
        rel = await self._probe(a_text, b_text)
        if rel != "emergence":
            return rel                      # cheap path: non-edges need no confirm
        self._confirm_probes += 1           # single-threaded asyncio: plain += is safe
        rev = await self._probe(b_text, a_text)
        return "emergence" if rev == "emergence" else "unrelated"

    # --- the per-finding entrypoint (fired off-lock by post_notice) ---
    async def observe(self, notice) -> None:
        """Called once per arriving noticeboard finding. Bounded edge detection
        against the skeleton frontier, then re-evaluate the flow decision."""
        if self._disabled_reason or self._halted:
            return
        try:
            nid = f"{getattr(notice, 'agent_id', '?')}-{int(getattr(notice, 'timestamp', 0))}-{len(self._findings)}"
            ntext = _notice_text(notice)

            # pick bounded candidate probes: new finding x recent frontier
            async with self._lock:
                if nid in self._index_of:
                    return
                self._index_of[nid] = len(self._findings)
                self._findings.append((nid, ntext))
                self._since_round += 1
                candidates: list[str] = []
                for fid in reversed(self._frontier):
                    k = frozenset((nid, fid))
                    if k in self._checked:
                        continue
                    candidates.append(fid)
                    if len(candidates) >= self.cap_per_finding:
                        break
                # if frontier is empty (early), seed against the most recent prior findings
                if not candidates:
                    for fid, _ in reversed(self._findings[:-1]):
                        k = frozenset((nid, fid))
                        if k in self._checked:
                            continue
                        candidates.append(fid)
                        if len(candidates) >= self.cap_per_finding:
                            break
                budget_left = max(0, self.max_probes - self._probes_used)
                candidates = candidates[:budget_left]
                self._probes_used += len(candidates)
                text_by_id = dict(self._findings)

            # run probes OFF-lock (network) — never block an agent or the board
            results = await asyncio.gather(
                *[self._probe_bidir(ntext, text_by_id[fid]) for fid in candidates],
                return_exceptions=True,
            )

            # merge results back under the lock
            async with self._lock:
                for fid, rel in zip(candidates, results):
                    self._checked.add(frozenset((nid, fid)))
                    if rel == "emergence":
                        self._edges.add(frozenset((nid, fid)))
                        self._round_emergence += 1
                        for x in (nid, fid):
                            if x in self._frontier:
                                self._frontier.remove(x)
                            self._frontier.append(x)
                # close a round every `batch` arrivals → extend the emergence-rate curve
                if self._since_round >= self.batch:
                    self._r_series.append(self._round_emergence)
                    self._round_emergence = 0
                    self._since_round = 0
                    self._evaluate_locked()
        except Exception as e:  # never let the governor crash a wander
            log.warning("[governor] observe failed (ignored): %s", e)

    def _evaluate_locked(self) -> None:
        """Re-run the doubt meter + shape + decide on the current skeleton.
        Caller holds _lock. Sets governor_halt on a confirmed CLOSE."""
        series = sterility_series(self._r_series, k=self.hysteresis_k)
        sterile_confirmed = bool(series and series[-1]["confirmed"])
        # build integer-indexed edges for the structural test
        n = len(self._findings)
        idx = {fid: i for i, (fid, _) in enumerate(self._findings)}
        iedges = [(idx[a], idx[b]) for e in self._edges for a, b in [tuple(e)]
                  if a in idx and b in idx]
        shape = has_shape(n, iedges, self.shape_f)
        dec = decide_live(sterile_confirmed=sterile_confirmed, shape=shape)
        rec = {
            "round": len(self._r_series),
            "r_series": list(self._r_series),
            "sterile_confirmed": sterile_confirmed,
            "shape": shape,
            "edges": len(self._edges),
            "findings": n,
            "probes_used": self._probes_used,
            "action": dec.action,
            "why": dec.why,
            "ts": time.time(),
        }
        self._decisions.append(rec)
        log.info("[governor] round %d: r=%s sterile=%s shape=%s -> %s",
                 rec["round"], self._r_series, sterile_confirmed, shape, dec.action)
        if dec.action == "CLOSE" and not self._halted:
            self._halted = True
            ss = self.session_state
            if ss is not None:
                try:
                    ss.governor_halt = True
                    ss.governor_halt_reason = f"CLOSE: {dec.why}"
                except Exception:
                    pass
            log.info("[governor] CLOSE — seizing flow, halting wander (%s)", dec.why)

    # --- skeleton-gap exposure (Phase 6: feeds the dispatcher) ---
    def _isolated_findings(self) -> list[str]:
        """Findings with NO emergence edge — the loose threads the mini-blender
        could NOT integrate into the skeleton. These are the structural GAPS the
        dispatcher targets: territory where connective tissue is still missing.
        (Read-only; safe to call any time.)"""
        in_skeleton: set[str] = set()
        for e in self._edges:
            in_skeleton |= set(e)
        return [txt for (nid, txt) in self._findings if nid not in in_skeleton and txt]

    def skeleton_gaps(self, *, limit: int = 12) -> dict:
        """Structural-gap summary for the dispatcher: the isolated findings (loose
        threads) + how fragmented the skeleton is (component_count > 1 means the
        mini-blender is still missing bridges between clusters)."""
        n = len(self._findings)
        idx = {fid: i for i, (fid, _) in enumerate(self._findings)}
        iedges = [(idx[a], idx[b]) for e in self._edges for a, b in [tuple(e)]
                  if a in idx and b in idx]
        comps = _components(n, iedges) if n else []
        iso = self._isolated_findings()
        return {
            "isolated_findings": [t[:240] for t in iso][:limit],
            "isolated_count": len(iso),
            "component_count": len(comps),
        }

    # --- artifact + teardown ---
    def governance_record(self) -> dict:
        rec = {
            "enabled": not bool(self._disabled_reason),
            "disabled_reason": self._disabled_reason,
            "halted": self._halted,
            "model": MINIBLENDER_MODEL,
            "params": {"batch": self.batch, "cap_per_finding": self.cap_per_finding,
                       "max_probes": self.max_probes, "hysteresis_k": self.hysteresis_k,
                       "shape_f": self.shape_f},
            "findings_seen": len(self._findings),
            "edges_found": len(self._edges),
            "probes_used": self._probes_used,
            "confirm_probes": self._confirm_probes,   # reverse-direction bidirectional confirms
            "bidirectional": True,
            "emergence_rate_series": list(self._r_series),
            "decisions": list(self._decisions),
            "final_action": self._decisions[-1]["action"] if self._decisions else "HOLD",
        }
        try:
            rec["skeleton_gaps"] = self.skeleton_gaps()
        except Exception as e:  # never let gap-extraction break the record
            log.warning("[governor] skeleton_gaps failed (ignored): %s", e)
            rec["skeleton_gaps"] = {"isolated_findings": [], "isolated_count": 0, "component_count": 0}
        return rec

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None


__all__ = [
    "WanderGovernor",
    "Decision",
    "decide_live",
    "has_shape",
    "sterility_series",
    "MINIBLENDER_MODEL",
]
