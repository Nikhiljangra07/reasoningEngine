"""
trust — domain-based source weighting for confidence assignment.

CRITICAL CONSTRAINT (Law 3): trust scoring is a TIEBREAKER, never a
gatekeeper. We never SUPPRESS a strong match because the source domain
is "untrusted" — a HIGH match from a random Substack stays HIGH, because
Law 3 says partial matches in unexpected places are the breakthrough zone.

Trust only PROMOTES borderline matches from trusted sources. A LOW match
on arxiv.org with reasonable evidence can promote to MEDIUM. That's it.

The implementation is intentionally a small lookup table — no learned
weights, no per-author scoring, no PageRank. Three rules:

  1. Peer-reviewed venues (arxiv, *.edu, *.gov) earn an upward tiebreak.
  2. Encyclopedic / curated venues (wikipedia, *.org) earn a smaller bump.
  3. Everything else is neutral.

If we add more sources later, append to TRUST_WEIGHTS — never remove the
"trust is additive, never suppressive" invariant.

ISOLATION: pure function over a URL string. No I/O, no LLM.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from src.wandering.report import Confidence


log = logging.getLogger("constellax.wandering.trust")


# ---------------------------------------------------------------------------
# Weight table
# ---------------------------------------------------------------------------

# Multiplicative weight on the borderline-promotion rule. Higher means
# "more likely to promote a borderline match." We don't multiply scores
# directly — we apply discrete promotions (LOW→MEDIUM, MEDIUM→HIGH) when
# the weight exceeds a threshold and the base match was at least
# minimally present.
#
# Suffix entries match the host's right-most label sequence; literal
# domain entries match exact host. Order in the table doesn't matter —
# we always pick the strongest matching weight.

LITERAL_DOMAIN_WEIGHTS: dict[str, float] = {
    "arxiv.org": 1.20,
    "wikipedia.org": 1.10,
    "en.wikipedia.org": 1.10,
    "scholar.google.com": 1.15,
    "ncbi.nlm.nih.gov": 1.15,
    "pubmed.ncbi.nlm.nih.gov": 1.15,
}

SUFFIX_WEIGHTS: list[tuple[str, float]] = [
    (".edu", 1.10),
    (".gov", 1.10),
    (".ac.uk", 1.10),
    (".edu.au", 1.10),
    (".ox.ac.uk", 1.15),
    (".cam.ac.uk", 1.15),
    (".org", 1.05),
]

DEFAULT_WEIGHT = 1.00


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def _host_of(url: str) -> str:
    """Lowercase host from a URL, empty string if unparseable.

    Strips leading "www." so e.g. www.wikipedia.org → wikipedia.org.
    """
    if not url:
        return ""
    try:
        host = urlsplit(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def domain_weight(url: str) -> float:
    """Return the trust weight for `url`'s host. Defaults to 1.0.

    Lookup order:
      1. Exact host match in LITERAL_DOMAIN_WEIGHTS
      2. Longest matching suffix in SUFFIX_WEIGHTS (longer suffix wins)
      3. DEFAULT_WEIGHT (neutral)
    """
    host = _host_of(url)
    if not host:
        return DEFAULT_WEIGHT
    if host in LITERAL_DOMAIN_WEIGHTS:
        return LITERAL_DOMAIN_WEIGHTS[host]
    # Longest-suffix-wins so .ox.ac.uk (1.15) beats .ac.uk (1.10).
    best = (DEFAULT_WEIGHT, 0)
    for suffix, weight in SUFFIX_WEIGHTS:
        if host.endswith(suffix) and len(suffix) > best[1]:
            best = (weight, len(suffix))
    return best[0]


# ---------------------------------------------------------------------------
# Confidence adjustment — the only mutation we apply
# ---------------------------------------------------------------------------

# Promotion threshold: a weight above this triggers a one-step promotion
# on borderline cases. 1.10 is the floor across the table — anything at
# the "neutral" weight (1.00) never triggers.
PROMOTION_THRESHOLD = 1.10


def adjust_confidence(
    base: Confidence,
    *,
    url: str,
    total_matched_nodes: int,
) -> Confidence:
    """Apply trust-based one-step promotion to a borderline confidence.

    Rules:
      - If base is HIGH, no change. Trust never adds to already-strong.
      - If base is MEDIUM, no change. (See DESIGN NOTE below.)
      - If base is LOW and weight >= PROMOTION_THRESHOLD and signal
        (matched_nodes >= 1), promote to MEDIUM.
      - Otherwise unchanged.

    Trust NEVER demotes. A HIGH-match from a random blog stays HIGH.

    DESIGN NOTE — why MEDIUM → HIGH is no longer allowed:
    The earlier rule promoted MEDIUM → HIGH on `total_matched_nodes >= 1`
    plus a trusted domain. The problem: `total_matched_nodes` is an
    absolute count, but the cushion's constellation size is decided per
    problem by the LLM (some problems map to 6 nodes, others to 30).
    "1 matched node" means very different things across constellations,
    so an absolute floor cannot identify "borderline" reliably. Rather
    than thread `constellation_size` through this pure function — which
    re-architects the function for one rule — we just remove the
    rule. The base confidence already encodes structural match quality
    via ratios at the matcher layer. Domain trust gives a one-step
    nudge on weak matches (LOW → MEDIUM, "this is a weak signal but
    from a cited source — worth surfacing") and stops there. The
    two-step lift from "borderline" to "decision-grade" based on
    domain alone was the over-promotion.

    Pure function — no I/O, no logging side effects.
    """
    if base == Confidence.HIGH:
        return base
    if base == Confidence.MEDIUM:
        return base
    if total_matched_nodes < 1:
        return base

    weight = domain_weight(url)
    if weight < PROMOTION_THRESHOLD:
        return base

    if base == Confidence.LOW:
        return Confidence.MEDIUM
    return base


__all__ = [
    "LITERAL_DOMAIN_WEIGHTS",
    "SUFFIX_WEIGHTS",
    "DEFAULT_WEIGHT",
    "PROMOTION_THRESHOLD",
    "domain_weight",
    "adjust_confidence",
]
