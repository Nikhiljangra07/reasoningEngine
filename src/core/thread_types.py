"""
Thread-layer data types for Constellax's memory pipeline.

This is the master schema. Nothing in here knows about databases — these are
plain dataclasses with `to_payload()` / `from_payload()` methods that serialize
to versioned JSON. The wire format is decoupled from the in-memory shape so
the schema can evolve without breaking stored records.

LAYERING
========
Three nested levels:

    ThreadRecord  ─── one conversation thread (sidebar entry)
        │
        └── IterationRecord  ─── one Q&A turn within the thread
                │
                └── SegmentedResponse  ─── the 3-segment delivery
                        ├── synthesizer    (direct answer)
                        ├── opinion        (peer commentary + multi-perspective)
                        └── prospects      (conditional forecasts + uncertainty disclaimer)

PHILOSOPHY
==========
1. Every record carries `schema_version: int`. Old records remain readable
   forever; new code branches on version when interpreting them.
2. Every record carries `meta: dict` — an open escape hatch. New fields land
   here before being promoted to first-class.
3. New fields are always optional (default None / empty list). Old records
   continue to deserialize cleanly.
4. The wire format flows through `to_payload()` / `from_payload()` — never
   `asdict()`. This lets us refactor internal shape without breaking storage.
5. Memory-layer fields (embedding, entities, tags, user_mode, etc.) live at
   the iteration level. The thread aggregates them denormally for fast lookup.

THE UNCERTAINTY DISCLAIMER
==========================
ProspectsSegment.uncertainty_disclaimer is a REQUIRED string field (not
optional). The frontend MUST render it verbatim. Default text per Nikhil
on 2026-05-25: "These are insights, not predictions. You're the one big
brain — this is the bigger picture based on what you've shared this session."

This is intentional product policy: Constellax is not a fortune-teller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# ─── Constants ────────────────────────────────────────────────────────

# Current schema version. Bump when shape changes in a non-additive way.
# Additive changes (new optional fields) do NOT require a version bump.
SCHEMA_VERSION = 1

# Required disclaimer text on every prospects segment. Frontend renders it
# verbatim, regardless of model output.
DEFAULT_UNCERTAINTY_DISCLAIMER = (
    "These are insights, not predictions. You're the one big brain — "
    "this is the bigger picture based on what you've shared this session."
)


# ─── Enums (as Literal types — easier to extend, JSON-safe) ───────────

ThreadStatus    = Literal["active", "ended", "archived"]
IterationStatus = Literal["pending", "done", "failed"]
SegmentKind     = Literal["synthesizer", "opinion", "prospects"]
Confidence      = Literal["high", "moderate", "low"]
EpistemicFlag   = Literal["strong", "hedge", "weak"]
Route           = Literal["trivial", "direct", "direct_plus", "deep"]
Effort          = Literal["low", "medium", "high"]
TimeHorizon     = Literal["immediate", "weeks", "months", "year+"]
UserMode        = Literal["exploratory", "decisive", "stuck", "venting", "analytical"]


# ─── Supporting types ─────────────────────────────────────────────────

@dataclass
class Attachment:
    """A file/image/CSV/MCP reference the user attached to the question."""
    kind:    str                       # "image" | "file" | "csv" | "mcp_ref" | "memory_recall"
    name:    str
    size:    int | None = None
    content: str | None = None         # base64 for blobs; plain text for text files
    mime:    str | None = None
    meta:    dict = field(default_factory=dict)


@dataclass
class TriageSnapshot:
    """Frozen output of the triage gate at the moment this iteration ran."""
    route:              Route
    recommended_effort: Effort
    risk_flags:         list[str] = field(default_factory=list)
    why:                str = ""
    classifier_mode:    str = ""       # "live" | "live_failed_fallback_mock" | etc


@dataclass
class BudgetSnapshot:
    """Frozen budget state at iteration completion."""
    iterations:     int = 0
    wall_time_sec:  float = 0.0
    cost_usd:       float = 0.0
    mcp_calls:      int = 0
    breached:       bool = False
    breach_reason:  str = ""


@dataclass
class ModelCall:
    """One LLM call made during this iteration — for cost auditing + provenance."""
    purpose:    str                    # "triage" | "synthesizer" | "opinion" | "prospects" | "extract_memo" | "embedding" | "metadata_extraction"
    model:      str                    # "gemini-2.5-flash" | "claude-sonnet-4-6" | "gemini-text-embedding-004"
    backend:    str                    # "gemini" | "claude" | "openai" | etc.
    tokens_in:  int = 0
    tokens_out: int = 0
    cost_usd:   float = 0.0
    latency_ms: int = 0


@dataclass
class GraphQuery:
    """One graphify query made during reasoning — for provenance."""
    method:        str                 # "get_callers_of" | "get_dependencies_of" | "get_code_structure"
    arg:           str
    result_count:  int = 0


@dataclass
class ProvenanceRecord:
    """What tools / models / data produced this iteration. Audit trail."""
    model_calls:        list[ModelCall] = field(default_factory=list)
    graphify_queries:   list[GraphQuery] = field(default_factory=list)
    mcps_invoked:       list[str] = field(default_factory=list)
    capability_state:   dict = field(default_factory=dict)
    backend_versions:   dict = field(default_factory=dict)  # {"graphify": "0.8.18", "engine": "...", ...}


# ─── Memory-layer fields (the pattern-matching primitives) ────────────

@dataclass
class Entity:
    """An extracted concept, person, product, or place from the question/response."""
    name:   str                        # "Maya", "Q3 launch", "paid acquisition"
    kind:   str                        # "person" | "concept" | "product" | "place" | "time_window" | "metric" | "other"
    salience: float = 1.0              # 0.0–1.0, how central to the iteration
    # Memory pane classification — one of FACT / PATTERN / VALUE / CONTEXT
    # / TENSION / INTEREST. Optional; older records and entities the
    # extractor hasn't classified yet leave this None and the UI falls
    # back to deriving a category from the entity's `kind`.
    category: str | None = None
    # User-set pin flag — pinned entities survive forever (the workspace's
    # 90-day staleness sweep skips them) and surface at the top of the
    # Memory pane. Default False keeps every legacy record unaffected.
    pinned:   bool = False
    meta:     dict = field(default_factory=dict)


@dataclass
class HiddenVariable:
    """A load-bearing thing the user didn't see — surfaced in the Opinion segment."""
    name:        str
    why_hidden:  str                   # explanation of why user might miss it
    impact:      str                   # how it changes the analysis


@dataclass
class PerspectiveNote:
    """A framework's contribution to the opinion segment (summarized)."""
    framework:   str                   # FrameworkID name as string
    domain:      str                   # "physics" | "mathematics" | "psychology" | "philosophy" | "chemistry"
    angle:       str                   # 1–2 sentence summary of this lens's contribution
    weight:      float = 1.0


# ─── Prospects (the future-insights segment, with conditional structure) ─

@dataclass
class ProspectBranch:
    """One conditional outcome scenario. ALL fields required — confidence is mandatory."""
    condition:  str                    # "If you go paid-first..."
    outcome:    str                    # "...CAC pressure compounds by week 6 unless ..."
    confidence: Confidence             # "high" | "moderate" | "low" — REQUIRED
    horizon:    TimeHorizon | None = None  # optional time-window of the outcome


# ─── Map Room artifacts ───────────────────────────────────────────────

@dataclass
class VisualBlock:
    """A visualization spec — Mermaid, Vega-Lite, comparison-table, etc."""
    kind:   str                        # "mermaid" | "vega-lite" | "comparison-table"
    title:  str | None = None
    spec:   Any = None                 # type-specific payload (string for mermaid, dict for vega-lite, etc.)
    meta:   dict = field(default_factory=dict)


@dataclass
class WalkStep:
    """One numbered step in the Map Room's 'walk me through' panel."""
    number:    int
    title:     str
    body:      str
    refs:      list[str] = field(default_factory=list)  # references to nodes/edges in visuals[]


@dataclass
class UserNote:
    """A note the user took inside the Map Room."""
    id:          str
    text:        str
    created_at:  float
    anchor_ref:  str | None = None     # optional: which visual/step this note is anchored to


@dataclass
class MapRoomArtifacts:
    """Everything the Map Room renders + everything the user produced there."""
    visuals:           list[VisualBlock] = field(default_factory=list)
    walkthrough_steps: list[WalkStep] = field(default_factory=list)
    user_notes:        list[UserNote] = field(default_factory=list)


# ─── Mid-stream interjection (v2 streaming feature; field reserved now) ─

@dataclass
class Interjection:
    """A mid-stream addition from the user between segments."""
    at_segment: SegmentKind            # which segment the interjection arrived during/after
    text:       str
    timestamp:  float
    applied:    bool = False           # did the engine actually incorporate it?


# ─── The 3-segment response ───────────────────────────────────────────

@dataclass
class Segment:
    """Base shape — direct-answer synthesizer uses this directly."""
    text:           str
    confidence:     Confidence | None = None
    delivered_at:   float = 0.0
    latency_ms:     int = 0
    model_used:     str = ""           # e.g. "gemini-2.5-flash"
    tokens_in:      int = 0
    tokens_out:     int = 0
    meta:           dict = field(default_factory=dict)


@dataclass
class OpinionSegment(Segment):
    """The peer-commentary segment. Surfaces multi-perspective + hidden variables."""
    peer_commentary:        str = ""                                  # the validation-then-pushback opener
    perspectives:           list[PerspectiveNote] = field(default_factory=list)
    hidden_variables:       list[HiddenVariable] = field(default_factory=list)


@dataclass
class ProspectsSegment(Segment):
    """The conditional-forecast segment. Mandatory uncertainty disclaimer."""
    branches: list[ProspectBranch] = field(default_factory=list)
    uncertainty_disclaimer: str = DEFAULT_UNCERTAINTY_DISCLAIMER
    # ↑ REQUIRED. Frontend renders verbatim. Always non-empty.


# ─── Outcome tracking (filled in later, after user reports back) ──────

@dataclass
class OutcomeRecord:
    """How an iteration's recommendation actually played out. Posted later."""
    reported_at:      float
    outcome_text:     str              # free-form user report
    followed_advice:  bool | None = None
    accuracy:         Confidence | None = None  # user's self-assessment of how right we were
    surprise_factor:  str | None = None         # what we missed
    meta:             dict = field(default_factory=dict)


# ─── The Segmented Response (top of the response stack) ───────────────

@dataclass
class SegmentedResponse:
    """The 3-segment delivery. Always populated together; segments are filled
    in order during streaming but the record is written once on completion."""
    overall_confidence: Confidence

    synthesizer: Segment | None = None
    opinion:     OpinionSegment | None = None
    prospects:   ProspectsSegment | None = None

    map_room:           MapRoomArtifacts = field(default_factory=MapRoomArtifacts)
    user_interjections: list[Interjection] = field(default_factory=list)

    # The "name the load-bearing assumption" output — promoted to a structured
    # field per Nikhil's spec. This is a CRITICAL graph edge — every time the
    # user assumed X, what happened later?
    load_bearing_assumption: str | None = None

    # Raw memo dict (verdict_line, verdict_body, reasoning, alternatives,
    # falsifiers, open_questions, visuals). Carried verbatim so the Map
    # Room can rehydrate structured fields on a fresh-tab fetch via
    # /api/v2/thread/{id}/full. Without this, the persistence boundary
    # strips structure and the Map Room falls back to parsing the
    # synthesizer prose — which loses visuals entirely.
    memo: dict | None = None


# ─── Memory context (what past data influenced THIS iteration) ────────

@dataclass
class MemoryContext:
    """Memory recall trace — what was pulled in as context for this iteration."""
    recalled_iteration_ids: list[str] = field(default_factory=list)  # past iterations cited
    recalled_decision_ids:  list[str] = field(default_factory=list)  # DecisionAnchor IDs surfaced
    similarity_scores:      dict = field(default_factory=dict)        # iter_id → score
    graphify_queries:       list[GraphQuery] = field(default_factory=list)


# ─── IterationRecord — the heart of the schema ────────────────────────

@dataclass
class IterationRecord:
    """One Q&A turn within a thread. Carries everything the system knows about
    this interaction — what was asked, what was generated, what memory was
    used, what it cost, and the extracted patterns for future pattern matching."""

    # ─── Schema versioning + escape hatch ──────────────
    schema_version: int = SCHEMA_VERSION
    meta:           dict = field(default_factory=dict)

    # ─── Identity ──────────────
    id:                  str = ""
    thread_id:           str = ""
    sequence_num:        int = 0
    parent_iteration_id: str | None = None       # for branched continuations

    # ─── Provenance (workspace_id is denormalized from Thread for direct query) ─
    # workspace_id is bound to the thread (one workspace per thread). It's
    # denormalized here so retrievers can filter iterations by platform
    # without a graph join through Thread.
    #
    # surface_id can vary PER iteration within the same thread — the user
    # may toggle between chat / map-room / wandering-room mid-conversation,
    # so each turn records which surface was active when it happened.
    # Defaults to None for legacy payloads written before this field existed;
    # new requests default to "web" / "chat" at the API boundary.
    workspace_id: str | None = None       # "cursor"|"claude"|"web"|"map-room"|... (denormalized)
    surface_id:   str | None = None       # "chat" (default) | "map-room" | "wandering-room"

    # ─── Input ──────────────
    question:        str = ""
    attachments:     list[Attachment] = field(default_factory=list)
    effort_picked:   Effort = "medium"
    mcps_selected:   list[str] = field(default_factory=list)

    # ─── Triage + engine artifacts ──────────────
    triage:  TriageSnapshot | None = None
    engine:  dict = field(default_factory=dict)
    # ^ EngineArtifacts as a raw dict. Variables/Perspectives are already serialized
    # by src/llm/serializers (serialize_perspectives, serialize_formation_plan).
    # Storing as opaque dict here avoids importing the entire engine type tree.

    # ─── The response (3-segment) ──────────────
    response: SegmentedResponse | None = None

    # ─── Memory pattern-matching primitives ──────────────
    # These are the seven additives that turn this from a conversation log into
    # an actual memory system.

    embedding:      list[float] | None = None    # vector for similarity recall (None until generated)
    embedding_model: str | None = None            # which model produced it (so we can migrate later)

    entities:       list[Entity] = field(default_factory=list)        # extracted concepts/people/products
    tags:           list[str] = field(default_factory=list)            # LLM-generated topic tags
    domains:        list[str] = field(default_factory=list)            # high-level domain classification

    user_mode:      UserMode | None = None        # exploratory/decisive/stuck/venting/analytical
    time_horizon:   TimeHorizon | None = None     # decision's time-window

    # Memory recall: what past data influenced THIS iteration's reasoning
    memory_context: MemoryContext = field(default_factory=MemoryContext)

    # ─── Outcome (populated later, post-user-feedback) ──────────────
    outcome_followup: OutcomeRecord | None = None

    # ─── Telemetry ──────────────
    budget:     BudgetSnapshot = field(default_factory=BudgetSnapshot)
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)

    # ─── Lifecycle ──────────────
    status:       IterationStatus = "pending"
    error:        dict | None = None
    created_at:   float = 0.0
    completed_at: float | None = None

    # ─── Structured-memory flag ──────────────
    # None = this iteration has not yet been processed by the Decision Trace
    # snapshot pipeline. A float timestamp = the moment the sweeper extracted
    # typed sub-nodes (Decision/Question/Reference/Insight) and committed them
    # to Neo4j. The sweeper only ever sets this; it never reads through this
    # field to decide whether to skip — it queries it with a WHERE clause.
    #
    # Idempotency: set LAST in the sweeper's transaction, after all typed
    # nodes are MERGE'd. A crash mid-sweep leaves the iteration unstructured
    # so the next sweep picks it up cleanly.
    structured_at: float | None = None

    # ─── Serialization (the wire-format insulator) ──────────────

    def to_payload(self) -> dict:
        """Serialize to a versioned dict suitable for FalkorDB/Postgres/etc.
        Always emits a `schema_version` field; new readers branch on it."""
        return _to_jsonable(self)

    @classmethod
    def from_payload(cls, raw: dict) -> "IterationRecord":
        """Deserialize from a dict, handling old schema versions defensively.
        Unknown fields land in `meta` rather than crashing the load."""
        return _iteration_from_payload(raw)


# ─── ThreadRecord — the conversation container ────────────────────────

@dataclass
class ThreadRecord:
    """One conversation thread. Lives in exactly one user/project/workspace
    triple. Iterations are stored separately (by id); this record aggregates."""

    # ─── Schema versioning + escape hatch ──────────────
    schema_version: int = SCHEMA_VERSION
    meta:           dict = field(default_factory=dict)

    # ─── Identity ──────────────
    id:           str = ""
    user_id:      str | None = None              # null = guest / anonymous
    project_id:   str | None = None              # which codebase / problem domain
    workspace_id: str | None = None              # "claude" | "cursor" | "codex" | "antigravity" | "web" | str

    # ─── Display ──────────────
    title:    str = ""                            # auto-derived from first question; user can rename
    summary:  str | None = None                   # auto-generated after N>1 iterations

    # ─── Lifecycle ──────────────
    created_at:   float = 0.0
    updated_at:   float = 0.0
    ended_at:     float | None = None
    status:       ThreadStatus = "active"

    # ─── Contents (denormalized for fast list queries) ──────────────
    iteration_ids: list[str] = field(default_factory=list)
    iteration_count:     int = 0
    last_route:          Route | None = None
    last_confidence:     Confidence | None = None

    # ─── Cross-references (graph-layer projection) ──────────────
    decision_anchor_ids: list[str] = field(default_factory=list)
    turning_point_ids:   list[str] = field(default_factory=list)
    related_thread_ids:  list[str] = field(default_factory=list)  # via embedding similarity, populated lazily

    # ─── Aggregated memory signals (for fast filter/search) ──────────────
    # Union of all iteration entities/tags/domains. Lets sidebar filter by
    # "all threads mentioning Maya" without scanning every iteration.
    all_entities: list[str] = field(default_factory=list)
    all_tags:     list[str] = field(default_factory=list)
    all_domains:  list[str] = field(default_factory=list)

    # ─── Cost / time / perspective rollups (for dashboard + history) ─────
    # Accumulated across every iteration on this thread. Updated by the
    # persistence task after each iteration save; older records that
    # predate these fields read 0 and the UI renders the existing
    # "—" placeholder gracefully.
    aggregate_time_ms:   int   = 0
    aggregate_cost_usd:  float = 0.0
    perspectives_run:    int   = 0   # sum of frameworks that actually fired

    def to_payload(self) -> dict:
        return _to_jsonable(self)

    @classmethod
    def from_payload(cls, raw: dict) -> "ThreadRecord":
        return _thread_from_payload(raw)


# ─── Wire-format serialization helpers ────────────────────────────────

def _to_jsonable(obj: Any) -> Any:
    """Recursively convert a dataclass tree into JSON-safe primitives.
    Plays the role of `asdict()` but routes through a layer we control,
    so we can intercept future format changes (e.g. omit deprecated fields,
    add wire-format aliases) without touching the dataclass definitions."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def _coerce_segment(raw: dict | None, segment_class: type) -> Any:
    """Build a Segment subclass from a dict, ignoring unknown fields gracefully."""
    if not raw:
        return None
    known = {f.name for f in segment_class.__dataclass_fields__.values()}
    safe = {k: v for k, v in raw.items() if k in known}
    obj = segment_class(**{k: v for k, v in safe.items() if not isinstance(v, list)})
    # Re-attach list fields with proper element types where applicable
    for k, v in safe.items():
        if isinstance(v, list):
            setattr(obj, k, _coerce_list_field(segment_class, k, v))
    return obj


def _coerce_list_field(cls: type, field_name: str, raw_list: list) -> list:
    """Best-effort: convert a list of dicts back to dataclass instances
    when the target field type is a known dataclass list."""
    mappers = {
        "perspectives":     PerspectiveNote,
        "hidden_variables": HiddenVariable,
        "branches":         ProspectBranch,
    }
    cls_target = mappers.get(field_name)
    if cls_target is None:
        return raw_list
    out = []
    for item in raw_list:
        if isinstance(item, dict):
            known = {f.name for f in cls_target.__dataclass_fields__.values()}
            out.append(cls_target(**{k: v for k, v in item.items() if k in known}))
        else:
            out.append(item)
    return out


def _coerce_dataclass(raw: dict | None, cls: type) -> Any:
    """Generic dict → dataclass coercion that drops unknown fields."""
    if not raw:
        return cls() if hasattr(cls, "__dataclass_fields__") else None
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in raw.items() if k in known})


def _iteration_from_payload(raw: dict) -> IterationRecord:
    """Deserialize an IterationRecord, branching on schema_version when needed."""
    if not isinstance(raw, dict):
        raise ValueError(f"IterationRecord payload must be dict, got {type(raw)}")

    version = raw.get("schema_version", 1)
    # When SCHEMA_VERSION advances past 1, add `if version == 1: upgrade(...)` here.

    # Pull out nested structured fields and rebuild them as dataclass instances.
    response_raw = raw.get("response")
    response_obj: SegmentedResponse | None = None
    if response_raw:
        response_obj = SegmentedResponse(
            overall_confidence=response_raw.get("overall_confidence", "moderate"),
            synthesizer=_coerce_segment(response_raw.get("synthesizer"), Segment),
            opinion=_coerce_segment(response_raw.get("opinion"), OpinionSegment),
            prospects=_coerce_segment(response_raw.get("prospects"), ProspectsSegment),
            map_room=_coerce_dataclass(response_raw.get("map_room"), MapRoomArtifacts),
            user_interjections=[
                _coerce_dataclass(i, Interjection) for i in (response_raw.get("user_interjections") or [])
            ],
            load_bearing_assumption=response_raw.get("load_bearing_assumption"),
            memo=response_raw.get("memo") if isinstance(response_raw.get("memo"), dict) else None,
        )

    triage_raw = raw.get("triage")
    triage_obj = _coerce_dataclass(triage_raw, TriageSnapshot) if triage_raw else None

    budget_obj = _coerce_dataclass(raw.get("budget"), BudgetSnapshot)
    provenance_obj = _coerce_dataclass(raw.get("provenance"), ProvenanceRecord)
    memory_context_obj = _coerce_dataclass(raw.get("memory_context"), MemoryContext)

    outcome_raw = raw.get("outcome_followup")
    outcome_obj = _coerce_dataclass(outcome_raw, OutcomeRecord) if outcome_raw else None

    entities = [_coerce_dataclass(e, Entity) for e in (raw.get("entities") or [])]
    attachments = [_coerce_dataclass(a, Attachment) for a in (raw.get("attachments") or [])]

    # Known top-level field names
    iteration_fields = {f.name for f in IterationRecord.__dataclass_fields__.values()}
    # Unknown fields → meta escape hatch
    unknown = {k: v for k, v in raw.items() if k not in iteration_fields}
    meta = dict(raw.get("meta") or {})
    if unknown:
        meta.setdefault("_unknown_fields", {}).update(unknown)

    return IterationRecord(
        schema_version=version,
        meta=meta,
        id=raw.get("id", ""),
        thread_id=raw.get("thread_id", ""),
        sequence_num=raw.get("sequence_num", 0),
        parent_iteration_id=raw.get("parent_iteration_id"),
        # Phase 1 fields — default None for legacy payloads (cutover smoke
        # tests, anything stored before Phase 1) so deserialization stays
        # forward-compatible. The API boundary applies real defaults
        # ("web" / "chat") on inbound requests.
        workspace_id=raw.get("workspace_id"),
        surface_id=raw.get("surface_id"),
        question=raw.get("question", ""),
        attachments=attachments,
        effort_picked=raw.get("effort_picked", "medium"),
        mcps_selected=list(raw.get("mcps_selected") or []),
        triage=triage_obj,
        engine=raw.get("engine") or {},
        response=response_obj,
        embedding=raw.get("embedding"),
        embedding_model=raw.get("embedding_model"),
        entities=entities,
        tags=list(raw.get("tags") or []),
        domains=list(raw.get("domains") or []),
        user_mode=raw.get("user_mode"),
        time_horizon=raw.get("time_horizon"),
        memory_context=memory_context_obj,
        outcome_followup=outcome_obj,
        budget=budget_obj,
        provenance=provenance_obj,
        status=raw.get("status", "done"),
        error=raw.get("error"),
        created_at=raw.get("created_at", 0.0),
        completed_at=raw.get("completed_at"),
        structured_at=raw.get("structured_at"),
    )


def _thread_from_payload(raw: dict) -> ThreadRecord:
    """Deserialize a ThreadRecord, branching on schema_version when needed."""
    if not isinstance(raw, dict):
        raise ValueError(f"ThreadRecord payload must be dict, got {type(raw)}")

    version = raw.get("schema_version", 1)

    thread_fields = {f.name for f in ThreadRecord.__dataclass_fields__.values()}
    unknown = {k: v for k, v in raw.items() if k not in thread_fields}
    meta = dict(raw.get("meta") or {})
    if unknown:
        meta.setdefault("_unknown_fields", {}).update(unknown)

    return ThreadRecord(
        schema_version=version,
        meta=meta,
        id=raw.get("id", ""),
        user_id=raw.get("user_id"),
        project_id=raw.get("project_id"),
        workspace_id=raw.get("workspace_id"),
        title=raw.get("title", ""),
        summary=raw.get("summary"),
        created_at=raw.get("created_at", 0.0),
        updated_at=raw.get("updated_at", 0.0),
        ended_at=raw.get("ended_at"),
        status=raw.get("status", "active"),
        iteration_ids=list(raw.get("iteration_ids") or []),
        iteration_count=raw.get("iteration_count", 0),
        last_route=raw.get("last_route"),
        last_confidence=raw.get("last_confidence"),
        decision_anchor_ids=list(raw.get("decision_anchor_ids") or []),
        turning_point_ids=list(raw.get("turning_point_ids") or []),
        related_thread_ids=list(raw.get("related_thread_ids") or []),
        all_entities=list(raw.get("all_entities") or []),
        all_tags=list(raw.get("all_tags") or []),
        all_domains=list(raw.get("all_domains") or []),
    )
