"""
============================================================================
  CONTROL ROOM  —  Constellax Wandering Room run configuration
============================================================================

This is the ONE place you set the direction for a wandering run.
Open this file, change the knobs below, save, then run:

    python scripts/run_fable_sorter_6agents.py

Nothing else to touch. The runner reads this file and obeys it.

Built 2026-06-13 — single-user (Nikhil) experimental config. Not a
production surface; no user-facing UI. When/if this goes to distribution,
these knobs become the per-user request body.
============================================================================
"""

# ---------------------------------------------------------------------------
# DOMAINS — where the models are allowed to wander.
# ---------------------------------------------------------------------------
# Leave EMPTY ([]) to unleash the full 60+ domain palette (the default
# broad cross-domain spread). Put domain names here to RESTRICT the
# wander to ONLY those domains.
#
# Valid names (from policy.SEED_DOMAINS), pick any subset:
#   physics  mathematics  biology  chemistry  ecology  neuroscience
#   astronomy  logic  computer_science  information_theory  psychology
#   sociology  anthropology  economics  linguistics  history  philosophy
#   religion  mythology  literature  music  film  theater  visual_arts
#   poetry  dance  architecture  ... (see policy.py for the full list)
#
# Example — only math + physics:
#     WANDER_DOMAINS = ["physics", "mathematics"]
# Example — unleash everything:
#     WANDER_DOMAINS = []
# ---------------------------------------------------------------------------
WANDER_DOMAINS = ["physics", "mathematics"]


# ---------------------------------------------------------------------------
# MODE — how the agents wander.
# ---------------------------------------------------------------------------
#   "multi_pendulum"  → N parallel agents, no sub-agents (MEDIUM, cheaper)
#   "absolute_chaos"  → N parallel agents, EACH spawns up to 2 follow-up
#                       sub-agents on its strongest finds (HIGH, deeper,
#                       more cards, costs more). This is "complete chaos".
#   "triple_pendulum" → one sequential chain of sub-agents (LOW, cheapest)
# ---------------------------------------------------------------------------
WANDER_MODE = "absolute_chaos"


# ---------------------------------------------------------------------------
# AGENTS — how many root agents to unleash.
# ---------------------------------------------------------------------------
# In absolute_chaos, the effective count is higher because each root may
# spawn up to 2 sub-agents on HIGH-confidence finds.
# ---------------------------------------------------------------------------
WANDER_AGENTS = 6


# ---------------------------------------------------------------------------
# MODEL — which model the wandering agents run on.
# ---------------------------------------------------------------------------
# All agents use this one model (uniform cohort). Swap to vary.
#   "anthropic/claude-sonnet-4-6"   ← current default
#   "anthropic/claude-opus-4-8"     ← stronger, pricier
#   "deepseek/deepseek-v4-pro"      ← different RLHF lineage
# ---------------------------------------------------------------------------
WANDER_MODEL = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# SORTER MODEL — which model classifies the cards into known/invalid/unplaced.
# ---------------------------------------------------------------------------
# Fable 5 is currently access-gated by Anthropic; Opus 4.8 is the
# redirect target and runs with the tightened sorter doctrine.
#   "anthropic/claude-fable-5"   ← gated right now (404)
#   "anthropic/claude-opus-4-8"  ← current working sorter
# ---------------------------------------------------------------------------
SORTER_MODEL = "anthropic/claude-opus-4-8"


# ===========================================================================
# Helpers — the runner imports these. You don't need to edit below this line.
# ===========================================================================

def as_dict() -> dict:
    """Return the control-room settings as a plain dict for logging."""
    return {
        "WANDER_DOMAINS": list(WANDER_DOMAINS),
        "WANDER_MODE":    WANDER_MODE,
        "WANDER_AGENTS":  WANDER_AGENTS,
        "WANDER_MODEL":   WANDER_MODEL,
        "SORTER_MODEL":   SORTER_MODEL,
    }


def seed_domains_env() -> str:
    """Comma-separated domain string for the WANDER_SEED_DOMAINS env var.
    Empty string when WANDER_DOMAINS is empty (full palette)."""
    return ",".join(WANDER_DOMAINS)
