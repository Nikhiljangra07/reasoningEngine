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
WANDER_DOMAINS = []  # broadened 2026-06-15 for the Cushion 2 run — full palette
                     # (reach the CS prior art the question asks for + metaphorical math/physics)


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
WANDER_AGENTS = 7  # Cushion 3 (2026-06-16): self-design run, 7 root agents


# ---------------------------------------------------------------------------
# MODEL — which model the wandering agents run on.
# ---------------------------------------------------------------------------
# All agents use this one model (uniform cohort). Swap to vary.
#   "anthropic/claude-sonnet-4-6"   ← current default (the wanderers' seat)
#   "anthropic/claude-opus-4-8"     ← stronger, pricier
#   "deepseek/deepseek-v4-pro"      ← different RLHF lineage
#
# Decided 2026-06-14 (Nikhil's spec): ONLY the wander runs on DeepSeek V4 Pro.
# Sorter (Sonnet), blender (Opus), drift-checker (Sonnet), halo (Sonnet) keep
# their original models — the DeepSeek change is deliberately ISOLATED to this
# one seat so its effect reads cleanly against the all-Sonnet/Opus baseline.
# NOTE on cost: DeepSeek V4 Pro is $1.74/$3.48 per M vs Sonnet $3/$15 — ~0.58x
# on input, NOT "1/5"; on the input-heavy wander it only ~halves cost. (A prior
# comment overstated the saving.) Also: WANDER_MODEL only sets the ROOT agents
# — the dig sub-agents + per-card articulation still route through
# runtime.MODE_DEFAULTS (Sonnet-heavy); run 20260614-015743 showed ~$2 of the
# wander stayed on Sonnet despite this knob. Fix that separately if real wander
# savings are wanted.
# ---------------------------------------------------------------------------
WANDER_MODEL = "deepseek/deepseek-v4-pro"


# ---------------------------------------------------------------------------
# SORTER MODEL — classifies cards into known/invalid/unplaced, now with web
# search (sorter_verify gathers evidence; the sorter bins against it).
# ---------------------------------------------------------------------------
# Nikhil's spec (2026-06-14): the sorter STAYS Sonnet — only the wander changed.
# Data point: run 20260614-015743 briefly put the sorter on DeepSeek and it
# binned all 29 cards "unplaced" vs Sonnet's all-"known" on comparable evidence.
# Kept for comparison, but Sonnet is the intended sorter.
#   "anthropic/claude-sonnet-4-6"  ← current (intended)
#   "deepseek/deepseek-v4-pro"     ← the 015743 data-point run
# ---------------------------------------------------------------------------
SORTER_MODEL = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# BLENDER MODEL — the collision seat (brick 2, not yet wired).
# ---------------------------------------------------------------------------
# The most cognitively demanding job: mindfully pick 2-4 cards that move
# toward the cushion and BLEND (not merge) them into a new candidate.
# Nikhil's spec (2026-06-14): the blender STAYS Opus 4.8 — the proven collision
# seat. Data point: run 20260614-015743 briefly put it on DeepSeek and it held
# up (4 coherent blends, same 3-novel/1-adjacent split, $0.05 vs Opus $0.74).
# Kept for comparison, but Opus is the intended blender.
#   "anthropic/claude-opus-4-8" ← current (intended, the proven blender)
#   "deepseek/deepseek-v4-pro"  ← the 015743 data-point run
# ---------------------------------------------------------------------------
BLENDER_MODEL = "anthropic/claude-opus-4-8"


# ---------------------------------------------------------------------------
# DRIFT-CHECKER MODEL — the supervisor seat (brick 3, not yet wired).
# ---------------------------------------------------------------------------
# Watches the blender; flags only when it drifts off the cushion toward a
# different problem. 2026-06-14: now Sonnet 4.6 — the quality eye on the
# pipeline, and a DIFFERENT lineage from the DeepSeek workers it supervises
# (the independent-eye principle, preserved even after the cost sweep).
#   "anthropic/claude-sonnet-4-6"  ← current (quality supervisor)
#   "deepseek/deepseek-v4-pro"     ← if cost on this seat ever matters
# ---------------------------------------------------------------------------
DRIFT_CHECKER_MODEL = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# AUDITOR MODEL — the halo auditor (blend-03, Phase 1 observer).
# ---------------------------------------------------------------------------
# Audits each layer (cushion / cards / blends) for blind spots + slack.
# This IS the hole-finder we proved is the quality bottleneck — if blind-spot
# quality matters more than cost, this is the seat to put a stronger model on.
#   "anthropic/claude-sonnet-4-6"  ← current: the quality seat on the bottleneck
#   "deepseek/deepseek-v4-pro"     ← cheaper, if blind-spot quality holds
# 2026-06-14: Nikhil chose Sonnet here — the auditor IS the hole-finder, so
# the one quality seat is best spent on it (drift-checker is also Sonnet).
# ---------------------------------------------------------------------------
AUDITOR_MODEL = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# RANKER MODEL — the final alignment pass (quality_ranker, the stage-1 closer).
# ---------------------------------------------------------------------------
# Runs LAST: ranks the verified blends by advancement toward the CUSHION
# (primary) + which halo blind spots they resolve (secondary), protecting
# blends that open a NEW gap. RANKS, never deletes — a mis-rank loses nothing.
# A judgment task → Sonnet (the strong eye). One cheap call (~$0.05).
#   "anthropic/claude-sonnet-4-6"  ← current (the judgment seat)
#   "deepseek/deepseek-v4-pro"     ← cheaper, if alignment quality holds
# ---------------------------------------------------------------------------
RANKER_MODEL = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# FORMALIZER MODEL — the R1 junior formalizer seat (post-blend stage).
# ---------------------------------------------------------------------------
# After the blender, DeepSeek R1 renders each finished blend into testable
# mathematics — justified + cited + falsifiable, abstaining where a blend isn't
# mathematical. R1 is JUNIOR to Opus: it formalizes, never re-blends. Run as a
# separate back-half stage (scripts/run_formalize.py), like the ranker/halo.
# Validated 2026-06-15 on blend-02 + blend-03 (faithful math, real citations,
# honest 'partial', real falsifiers). Routes via OpenRouter (no direct key).
#   "deepseek/deepseek-r1"  ← apex math/reasoning seat
# ---------------------------------------------------------------------------
R1_FORMALIZE_MODEL = "deepseek/deepseek-r1"


# ---------------------------------------------------------------------------
# CONTRIBUTION BOARD — additive, positive-sum dig directive (overlap reduction)
# ---------------------------------------------------------------------------
# When True, sets WANDER_CONTRIBUTION_BOARD=1 for the run: each wander agent
# reads what peers have already surfaced and is told to add the deeper/MISSING
# layer — go deeper, find the transferable structure beneath what's covered.
# Explicitly NOT competitive ("add what they missed," never "beat them"), with
# an anti-inflation guard. Chaos-safe — reads only the PAST (posted findings),
# never steers the walk.
#
# Default False so the corrected-config baseline stays a CLEAN single-variable
# run. Flip to True to isolate the board's effect in its own run (don't stack
# it on top of the sorter/blender fix in the same run — one change at a time).
# 2026-06-14: ON for the board A/B vs baseline 20260614-035615 — ONLY change.
CONTRIBUTION_BOARD = True


# ===========================================================================
# Helpers — the runner imports these. You don't need to edit below this line.
# ===========================================================================

def as_dict() -> dict:
    """Return the control-room settings as a plain dict for logging."""
    return {
        "WANDER_DOMAINS":      list(WANDER_DOMAINS),
        "WANDER_MODE":         WANDER_MODE,
        "WANDER_AGENTS":       WANDER_AGENTS,
        "WANDER_MODEL":        WANDER_MODEL,
        "SORTER_MODEL":        SORTER_MODEL,
        "BLENDER_MODEL":       BLENDER_MODEL,
        "DRIFT_CHECKER_MODEL": DRIFT_CHECKER_MODEL,
        "AUDITOR_MODEL":       AUDITOR_MODEL,
        "RANKER_MODEL":        RANKER_MODEL,
        "R1_FORMALIZE_MODEL":  R1_FORMALIZE_MODEL,
        "CONTRIBUTION_BOARD":  CONTRIBUTION_BOARD,
    }


def seed_domains_env() -> str:
    """Comma-separated domain string for the WANDER_SEED_DOMAINS env var.
    Empty string when WANDER_DOMAINS is empty (full palette)."""
    return ",".join(WANDER_DOMAINS)
