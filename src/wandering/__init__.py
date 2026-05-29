"""
Wandering Room — Constellax's research mode for people mid-build on hard concepts.

Wandering Room sends multiple AI agents into bounded chaotic exploration across
ALL human knowledge (not just adjacent domains), holding the user's problem as
an immutable anchor. Agents recognize partial structural matches as exploration
triggers (not as termination signals — metal detector, not Shazam), dig deeper
into resonant zones, and return confidence-scored cited mini-reports.

The user does the synthesis. Constellax creates the conditions for the user's
Heisenberg moment; it does not deliver insights.

See `docs/wandering_room_plan.md` for the complete specification.

# The Seven Laws governing this module

  1. Chaos is the feature. Domain unconstrained. No optimization of the walk.
  2. Anchor never moves. Insight in user's head. Reports are residue.
  3. Partial matches are the breakthrough zone. LOW confidence is gold.
  4. Read/reason/explain. No edits, no commits, no deploys.
  5. Feel the concept before scoping the build.
  6. Honest doubt > performative confidence.
  7. Match on structural essence, not topical surface.

# Module layout

Anchor (cushion):
  cushion.py    — types (CushionField, CushionInput, CushionLayer, CushionGraph)
  composer.py   — Sonnet-based extraction of three-layer cushion from four-field brief

Engine:
  matching.py   — metal detector mechanic: cushion vs content match scoring
  policy.py     — chaos walk + anchor pull + drift detection
  critique.py   — self-critique six metacognitive questions per iteration
  agent.py      — one wandering pendulum (state + loop)
  runtime.py    — multi-agent orchestration (Triple/Multi/Absolute modes)

Output:
  report.py     — ExplorationReport with mandatory what_does_not_map
  trace.py      — DecisionTrace + DiscardedClue classification
  articulate.py — Spark/Source Shape/Bridge/Use/Limit/Confidence card
  synthesis.py  — cross-report clustering + contradictions + paths
  dossier.py    — final user-facing artifact (3 confidence bands + synthesis)

# Wiring deferred to later phases

  - API endpoint (POST /api/v2/wandering/...)
  - Neo4j persistence for cushion + reports + traces + discarded shelf
  - Tavily/Notion/IDE real fetchers (stub_fetcher used in tests)
  - Sub-agent spawning runtime mechanics (mode shape locked; execution Phase 7)
  - Frontend visualization (Map mode, dossier rendering)
"""

# ---- Anchor ----
from src.wandering.cushion import (
    CushionField,
    CushionGraph,
    CushionInput,
    CushionLayer,
    SkipReason,
)

# ---- Output types ----
from src.wandering.report import (
    Confidence,
    ExplorationReport,
    LayerMatch,
    SourceCitation,
)
from src.wandering.trace import (
    DecisionTrace,
    DiscardKind,
    DiscardedClue,
    StepKind,
    TraceStep,
)

# ---- Engine ----
from src.wandering.matching import (
    MatchResult,
    iterations_for_match,
)
from src.wandering.policy import (
    NextMove,
    detect_drift,
    next_move,
    pick_next_domain,
)
from src.wandering.critique import (
    CritiqueResult,
    CritiqueVerdict,
    QUESTIONS as CRITIQUE_QUESTIONS,
)
from src.wandering.agent import (
    AgentBudget,
    AgentState,
    FetchResult,
    stub_fetcher,
)
from src.wandering.runtime import (
    SessionResult,
    WanderingConfig,
    WanderingMode,
)

# ---- Articulation + synthesis + dossier ----
from src.wandering.articulate import ArticulatedCard
from src.wandering.synthesis import (
    Contradiction,
    InsightCluster,
    OpportunityPath,
    SynthesisMap,
)
from src.wandering.dossier import (
    ConfidenceBand,
    Dossier,
    DossierMetadata,
)

# ---- Sub-agent spawning ----
from src.wandering.subagent import (
    MAX_CHAIN_DEPTH,
    SpawnRequest,
    SpawnResult,
    should_spawn,
    run_subagent,
)

# ---- Real fetcher (wires to web_search) ----
from src.wandering.fetcher import web_search_fetcher

# ---- Persistence ----
from src.wandering.persistence import (
    InMemoryWanderingStore,
    Neo4jWanderingStore,
    WanderingStore,
    build_wandering_store_from_env,
)

__all__ = [
    # Anchor
    "CushionField",
    "CushionGraph",
    "CushionInput",
    "CushionLayer",
    "SkipReason",
    # Output types
    "Confidence",
    "ExplorationReport",
    "LayerMatch",
    "SourceCitation",
    "DecisionTrace",
    "DiscardKind",
    "DiscardedClue",
    "StepKind",
    "TraceStep",
    # Engine
    "MatchResult",
    "iterations_for_match",
    "NextMove",
    "detect_drift",
    "next_move",
    "pick_next_domain",
    "CritiqueResult",
    "CritiqueVerdict",
    "CRITIQUE_QUESTIONS",
    "AgentBudget",
    "AgentState",
    "FetchResult",
    "stub_fetcher",
    "SessionResult",
    "WanderingConfig",
    "WanderingMode",
    # Articulation + synthesis + dossier
    "ArticulatedCard",
    "Contradiction",
    "InsightCluster",
    "OpportunityPath",
    "SynthesisMap",
    "ConfidenceBand",
    "Dossier",
    "DossierMetadata",
]
