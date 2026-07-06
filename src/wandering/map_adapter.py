"""
Wandering Room → Map Room adapter.

Pure-data mapping: takes a SessionResult + its built Dossier and emits the
Map Room's Memo wire shape (snake_case dict). The Memo carries a single
`knowledge-graph` visual whose nodes/edges express the dossier as a
graph — pursuit at the center, cards radiating out, clusters/contradictions/
opportunity_paths as additional structure.

Why a graph: a WanderingReport IS graph-shaped (Pursuit → Cards via partial
matches, Clusters → Cards, Cards ↔ Cards via contradictions, Cards → Paths
via support). The Map Room's existing KnowledgeGraphSpec renderer (Cytoscape,
auto-layout) renders this natively — no information loss, no impedance
mismatch with the Thinking Room's Memo shape.

ISOLATION: this adapter NEVER reads or writes the master :Thread / :Iteration
graph. It only takes wandering-namespace data and emits a JSON-serializable
dict for the HTTP response.

DETERMINISTIC: no LLM calls, no async, no network. Same input → same output.
"""

from __future__ import annotations

from typing import Any

from src.wandering.dossier import Dossier
from src.wandering.report import Confidence
from src.wandering.runtime import SessionResult


# Truncation budgets keep the graph readable. Cytoscape can render long
# labels, but the visual gets noisy past ~80 chars per node and ~60 per
# cluster/path bubble.
_NODE_LABEL_MAX     = 80
_BUBBLE_LABEL_MAX   = 60
_PURSUIT_LABEL_MAX  = 80
_REASONING_BODY_MAX = 240


def session_to_memo(session: SessionResult, dossier: Dossier) -> dict[str, Any]:
    """Convert a wandering session's dossier into Map Room Memo wire shape.

    Caller is responsible for calling `await build_dossier(session, client)`
    first; this adapter is intentionally sync + pure so it stays trivial to
    test and impossible to misuse (no hidden LLM cost).
    """
    cards = (
        list(dossier.high.cards)
        + list(dossier.medium.cards)
        + list(dossier.low.cards)
    )
    card_by_id = {c.report_id: c for c in cards}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # ── Root node: the user's pursuit ─────────────────────────────────
    pursuit_text = (
        session.cushion.raw_input.problem.content
        or session.cushion.actual.summary
        or "(no pursuit stated)"
    )
    nodes.append({
        "id":    "pursuit",
        "label": _truncate(pursuit_text, _PURSUIT_LABEL_MAX),
        "kind":  "decision",
    })

    # ── Card nodes + spokes from pursuit ──────────────────────────────
    # LOW confidence cards render as "claim" (dashed/dimmer) — they're
    # the breakthrough zone but the most tentative. MEDIUM/HIGH render as
    # "concept". Match strength rides the edge label as a percentage.
    for card in cards:
        node_label = _truncate(
            card.spark or card.domain or card.report_id,
            _NODE_LABEL_MAX,
        )
        kind = "claim" if card.confidence == Confidence.LOW else "concept"
        nodes.append({
            "id":    card.report_id,
            "label": node_label,
            "kind":  kind,
        })
        match_pct = int(round(_clamp01(card.match_strength) * 100))
        edges.append({
            "source":   "pursuit",
            "target":   card.report_id,
            "relation": card.domain or "match",
            "label":    f"{match_pct}%",
        })

    # ── Clusters: each becomes a concept bubble linked to its members ─
    synthesis = dossier.synthesis
    for i, cluster in enumerate(synthesis.clusters):
        cluster_id = f"cluster-{i + 1}"
        nodes.append({
            "id":    cluster_id,
            "label": _truncate(cluster.label, _BUBBLE_LABEL_MAX),
            "kind":  "concept",
        })
        for card_id in cluster.card_ids:
            # Silently skip card_ids the cluster references but don't exist
            # on the dossier — synthesizer can hallucinate ids; rendering
            # a dangling edge would crash Cytoscape.
            if card_id in card_by_id:
                edges.append({
                    "source":   cluster_id,
                    "target":   card_id,
                    "relation": "contains",
                })

    # ── Contradictions: edges between the two card_ids they reference ─
    # We DON'T add a node — contradictions are inherently between two
    # specific cards, so the edge tells the whole story. Layout-wise,
    # this means contradicting cards are pulled toward each other by
    # Cytoscape's force layout, which is exactly what we want.
    for contradiction in synthesis.contradictions:
        ids = contradiction.card_ids
        if len(ids) < 2:
            continue
        # Pair the first two referenced cards (synthesizer convention).
        source_id, target_id = ids[0], ids[1]
        if source_id in card_by_id and target_id in card_by_id:
            edges.append({
                "source":   source_id,
                "target":   target_id,
                "relation": "contradicts",
                "label":    "tension",
            })

    # ── Opportunity paths: outcome node + supporting edges from cards ─
    for j, path in enumerate(synthesis.opportunity_paths):
        path_id = f"path-{j + 1}"
        nodes.append({
            "id":    path_id,
            "label": _truncate(path.description, _BUBBLE_LABEL_MAX),
            "kind":  "outcome",
        })
        for card_id in path.supporting_card_ids:
            if card_id in card_by_id:
                edges.append({
                    "source":   card_id,
                    "target":   path_id,
                    "relation": "supports",
                })

    # ── Reasoning column on the right of Map Room ─────────────────────
    # Top insights reference card_ids; resolve to card text so the user
    # reads the spark, not a wander-P01-001-style id.
    reasoning = _build_reasoning(synthesis.top_insights, card_by_id)

    open_questions = [
        {"question": q, "answer": ""}
        for q in synthesis.open_questions[:6]
        if isinstance(q, str) and q.strip()
    ]

    verdict_line = (
        synthesis.recommended_next_direction.strip()
        or "Research dossier — no single verdict; explore the partial matches."
    )
    verdict_body = _compose_verdict_body(dossier)

    return {
        "verdict_line":   verdict_line,
        "verdict_body":   verdict_body,
        "confidence":     _aggregate_confidence(dossier),
        "reasoning":      reasoning,
        "alternatives":   [],
        "falsifiers":     [],
        "open_questions": open_questions,
        "visuals": [
            {
                "type":   "knowledge-graph",
                "title":  "Dossier map",
                "layout": "cose",
                "nodes":  nodes,
                "edges":  edges,
            }
        ],
    }


# ─── helpers ──────────────────────────────────────────────────────────


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _clamp01(x: float) -> float:
    if not isinstance(x, (int, float)):
        return 0.0
    if x != x:  # NaN check without importing math
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _build_reasoning(
    top_insights: list[str],
    card_by_id: dict[str, Any],
) -> list[dict[str, str]]:
    """Resolve top_insights (card_ids OR free-form strings) into reasoning
    rows the Memo's right column renders. Cap at 5 — the Map Room's
    reasoning column gets noisy past that."""
    out: list[dict[str, str]] = []
    for insight in top_insights:
        if not isinstance(insight, str) or not insight.strip():
            continue
        card = card_by_id.get(insight.strip())
        if card is not None:
            out.append({
                "title": (card.domain or "Insight").strip() or "Insight",
                "body":  _truncate(card.spark or "", _REASONING_BODY_MAX),
            })
        else:
            out.append({
                "title": "",
                "body":  _truncate(insight, _REASONING_BODY_MAX),
            })
        if len(out) >= 5:
            break
    return out


def _aggregate_confidence(dossier: Dossier) -> str:
    """Map the card distribution to a Map Room confidence band.

    The Map Room confidence band was designed for Thinking Room verdicts.
    For a wandering dossier we use it as a coarse signal of "how strong
    is the body of partial matches" — HIGH band dominates the meter,
    otherwise the largest band wins. Empty dossier → low (honest)."""
    high_n = len(dossier.high.cards)
    med_n  = len(dossier.medium.cards)
    low_n  = len(dossier.low.cards)
    total  = high_n + med_n + low_n
    if total == 0:
        return "low"
    # HIGH cards exist? Lead with them — they're the rare strong-overlap finds.
    if high_n >= max(med_n, low_n):
        return "high"
    if med_n >= low_n:
        return "moderate"
    return "low"


def _compose_verdict_body(dossier: Dossier) -> str:
    """Short factual one-liner under the verdict — counts the dossier."""
    high_n = len(dossier.high.cards)
    med_n  = len(dossier.medium.cards)
    low_n  = len(dossier.low.cards)
    total  = high_n + med_n + low_n

    parts: list[str] = []
    if total == 0:
        parts.append("No partial matches surfaced")
    else:
        parts.append(
            f"{total} partial match{'es' if total != 1 else ''} surfaced "
            f"({low_n} low · {med_n} medium · {high_n} high)"
        )
    cluster_n = len(dossier.synthesis.clusters)
    if cluster_n:
        parts.append(f"{cluster_n} cluster{'s' if cluster_n != 1 else ''}")
    contradiction_n = len(dossier.synthesis.contradictions)
    if contradiction_n:
        parts.append(
            f"{contradiction_n} contradiction{'s' if contradiction_n != 1 else ''}"
        )
    opportunity_n = len(dossier.synthesis.opportunity_paths)
    if opportunity_n:
        parts.append(
            f"{opportunity_n} opportunity path"
            f"{'s' if opportunity_n != 1 else ''}"
        )
    return " · ".join(parts) + "."


__all__ = ["session_to_memo"]
