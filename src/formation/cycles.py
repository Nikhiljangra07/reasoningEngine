"""
Wu Xing Cycle Definitions — The Traffic Rules.

Outer Layer: governs how the five domains interact with each other.
Two cycles run SIMULTANEOUSLY:

Sheng (Generating / Construction — clockwise):
  Each domain's output feeds into the next domain as input.
  Philosophy → Chemistry �� Physics → Mathematics → Psychology → Philosophy

Ke (Controlling / Deconstruction — pentagram):
  Each domain challenges a DIFFERENT domain than the one it feeds.
  Physics checks Psychology       (Earth dams Water)
  Psychology checks Chemistry     (Water extinguishes Fire)
  Chemistry checks Mathematics    (Fire melts Metal)
  Mathematics checks Philosophy   (Metal chops Wood)
  Philosophy checks Physics       (Wood penetrates Earth)

When fewer than 5 domains are active, the cycles run on the available
subset. Missing domain slots are skipped.

ISOLATION: Imports ONLY from src.core.types.
"""

from __future__ import annotations

from src.core.types import Domain


# ---------------------------------------------------------------------------
# Sheng Cycle (Generating / Construction)
# ---------------------------------------------------------------------------
# Order: Philosophy → Chemistry → Physics → Mathematics → Psychology → (loop)
# Each domain's output feeds the next as input.

SHENG_ORDER: list[Domain] = [
    Domain.PHILOSOPHY,      # Wood
    Domain.CHEMISTRY,       # Fire
    Domain.PHYSICS,         # Earth
    Domain.MATHEMATICS,     # Metal
    Domain.PSYCHOLOGY,      # Water
]


def get_sheng_order(active_domains: list[Domain]) -> list[Domain]:
    """
    Get the Sheng cycle order for the currently active domains.
    Skips domains that aren't active. Preserves the cycle order.
    """
    return [d for d in SHENG_ORDER if d in active_domains]


def get_sheng_upstream(domain: Domain, active_domains: list[Domain]) -> Domain | None:
    """
    Get which domain feeds INTO this domain in the Sheng cycle.

    Philosophy feeds Chemistry.
    Chemistry feeds Physics.
    Physics feeds Mathematics.
    Mathematics feeds Psychology.
    Psychology feeds Philosophy.

    Returns None if the upstream domain is not active.
    """
    order = get_sheng_order(active_domains)
    if domain not in order:
        return None

    idx = order.index(domain)
    upstream_idx = (idx - 1) % len(order)
    return order[upstream_idx]


# ---------------------------------------------------------------------------
# Ke Cycle (Controlling / Deconstruction)
# ---------------------------------------------------------------------------
# Each domain challenges a DIFFERENT domain than the one it feeds.
# This structurally prevents echo chambers.

KE_PAIRS: dict[Domain, Domain] = {
    Domain.PHYSICS: Domain.PSYCHOLOGY,          # Earth dams Water
    Domain.PSYCHOLOGY: Domain.CHEMISTRY,        # Water extinguishes Fire
    Domain.CHEMISTRY: Domain.MATHEMATICS,       # Fire melts Metal
    Domain.MATHEMATICS: Domain.PHILOSOPHY,      # Metal chops Wood
    Domain.PHILOSOPHY: Domain.PHYSICS,          # Wood penetrates Earth
}

# Reverse lookup: who checks ME?
KE_CHECKED_BY: dict[Domain, Domain] = {v: k for k, v in KE_PAIRS.items()}


def get_ke_target(challenger: Domain) -> Domain | None:
    """Get which domain this challenger checks in the Ke cycle."""
    return KE_PAIRS.get(challenger)


def get_ke_challenger(target: Domain) -> Domain | None:
    """Get which domain checks this target in the Ke cycle."""
    return KE_CHECKED_BY.get(target)


def get_active_ke_pairs(
    active_domains: list[Domain],
) -> list[tuple[Domain, Domain]]:
    """
    Get all active Ke cycle pairs for the current domain set.
    Only returns pairs where BOTH challenger and target are active.
    """
    pairs = []
    for challenger, target in KE_PAIRS.items():
        if challenger in active_domains and target in active_domains:
            pairs.append((challenger, target))
    return pairs
