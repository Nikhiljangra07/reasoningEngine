# Wandering Room — Complete Plan (V0.2)

**Status:** Concept locked. Awaiting Phase 0 kickoff.
**Last updated:** 2026-05-29
**Authoritative source for:** intake form, anchor structure, wandering modes, agent architecture, output format, build phases.

---

## Part I — Mission & Laws

### Mission

Wandering Room is Constellax's research mode for people **mid-build on hard concepts**. Not Q&A. Not life decisions. Concept work, project advancement, breaking bottlenecks in active research/architecture/product design.

**What it does:** Simulates the human creative cognition pattern — holding a problem in mind while encountering unrelated material, recognizing partial structural matches, extracting threads, carrying them back.

**What it does NOT do:** Deliver insights. Constellax delivers the *conditions* for the user's Heisenberg moment. Reports are residue. The insight happens in the user's brain.

**Comparison:**
- Perplexity / ChatGPT / Gemini Deep Research → linear depth, single domain, return conclusion
- Wandering Room → chaotic breadth across all human knowledge, anchor-tethered, return partial-match leads for user synthesis

### The Seven Laws (governing Claude's behavior on this project)

1. **Chaos is the feature.** Domain is unconstrained — agents wander across ALL human knowledge. Anchor pull is on structural essence, not topical proximity. Wuxing's own origin (5000-year-old Taoist philosophy → modern reasoning architecture) is the proof and precedent.
2. **The anchor never moves. The insight happens in the user's head.** Optimize for surface tension, not comprehensiveness.
3. **Partial matches are the breakthrough zone, not the long tail.** LOW confidence is where the Heisenberg lives.
4. **Read / reason / explain / map / remember / suggest. No acts.** No edits, no commits, no deploys. Property line, not feature flag.
5. **Feel the concept before scoping the build.**
6. **Honest doubt beats performative confidence.**
7. **Match on structural essence, not topical surface.** Cushion graph is (force)-(tension)-(force) primitives, not (entity)-(relation)-(entity) triples.

(Saved as persistent memory at `feedback_wandering_room_laws.md`.)

### Permission Boundary

| Allowed | Forbidden |
|---|---|
| Read | Edit user files |
| Reason | Run commands |
| Map | Commit code |
| Explain | Deploy |
| Remember | Delete data |
| Suggest | Silent state changes |
| Export (with approval) | |

---

## Part II — The Anchor

### Three-Layer Cushion Graph

Because AI lacks the human consciousness that fluidly shifts between abstraction levels, we EXPLICITLY construct three representations of the same problem and hand all three to the agents. Each layer is independently matchable.

| Layer | What it captures | Example for "AI agent control" problem |
|---|---|---|
| **Actual** | Literal, concrete description | "AI agents wandering the internet for inspirations to a user's research problem" |
| **Essence** | Structural-dynamic pattern (forces, tensions, constraints, cycles) | "Bounded freedom. Productive constraint. Anchored chaos." |
| **Mechanism** | Causal primitive (the abstract operating logic) | "Any system aiming for unpredictable creative output under resource limits requires soft structural constraint, not hard limiting constraint" |

Within each layer: 3–8 sub-nodes form that layer's metal-detector graph. Total cushion: 9–24 nodes across three layers.

**Match scoring:** Surface entity overlap = small/zero weight. Essence + mechanism resonance = the whole signal. An agent finding 0 surface match but 4/5 essence + 5/5 mechanism = HIGH confidence breakthrough.

### Four-Field Intake Form

Sonnet extracts the three-layer cushion from this form. User answers four questions; system builds the structural representation.

| Field | Prompt | Captures |
|---|---|---|
| **1. The Problem** | "What are you trying to figure out? Describe it concretely — the specific thing you're wrestling with." | Actual problem |
| **2. The Context** | "Where does this problem sit? Is it standalone or part of a bigger system you're building? And what brings you here now — what made this the question?" | System context + origin |
| **3. The Vision** | "Where are you trying to go? What are you building toward, even roughly? This helps agents recognize material that resonates with your trajectory, not just your current state." | Future vision |
| **4. Your Current Map** | "What angles, threads, or domains are you already considering? Even rough hunches help — the agents start from your partial map and chaos-walk outward from there." | Initial inspirations + related domains (+ auto-enriched from project memory) |

**Auto-enrichment of Current Map:** The "current map" field is automatically enriched from the user's project memory graph. The wandering agents (and decision room agents) already have access to the user's project state — recent threads, current architecture, ongoing decisions. When building the cushion, we transparently pull relevant project context to enrich whatever the user supplies. Behind the scenes, no permission required — it's the user's own memory. This means even a sparse "current map" field gets the benefit of full project context for cushion construction.

### Skip + Follow-Up + Warning Protocol

Every field is skippable, but skipping triggers a reflection prompt and an honest warning. The warning is field-specific — tells the user what's specifically lost.

| Field | Cost of skipping |
|---|---|
| Problem | Empty actual layer → no anchor → results catastrophically thin |
| Context | Cushion loses dimensional richness → agents miss where the problem fits |
| Vision | One matching surface lost → Heisenberg-zone hits become rarer |
| Current Map | Cold start → weaker essence seeds → broader, weaker connections |

**Pattern:** "Are you sure you don't have even a rough sense of X? Even a sentence helps." → if still skipped → "Without this, [specific consequence]. Continue?"

---

## Part III — The Engine

### Three Wandering Modes (LOW / MED / HIGH effort)

| Mode | Structure | Agents | Duration | Cost (est.) |
|---|---|---|---|---|
| **Triple Pendulum** (LOW) | One chain, sub-agents extend it sequentially. Agent 1 → 10m → Agent 2 → 17m → Agent 3 → 30m | 1 agent + 2-3 sub-agents | 10–15 min | low |
| **Multi Pendulum** (MEDIUM) | One anchor, N parallel agents, no sub-agents. Star pattern | 5 agents | 20–30 min | medium |
| **Absolute Chaos** (HIGH) | One anchor, N agents, each can spawn sub-agents on HIGH matches. Star + recursive branches | 10 agents | 30–60 min | high |

### Per-Agent State

Each agent receives:

```json
{
  "agent_id": "P03",
  "model": "DeepSeek V4 Pro | Haiku 4.5 | Sonnet 4.6",
  "anchor": "<three-layer cushion graph>",
  "domain_or_direction": "starting position (random or assigned)",
  "distance_budget_tokens": 30000,
  "time_budget": "30 minutes",
  "iteration_count_at_match": "2 + match_count, capped at 5",
  "spawn_subagent_tool": "<available in Triple/Absolute Chaos>"
}
```

### Pattern Matching — Metal Detector Mechanic

At every step:
1. Agent fetches content from current position (Tavily / Notion / IDE / domain-specific)
2. Encodes content as a graph
3. Computes overlap against EACH of the three cushion layers
4. **Any node match in ANY layer triggers exploration** (metal detector keeps beeping)
5. Match strength → iteration count (locked at start of dig)
6. Dig produces a cited mini-report
7. Agent returns to anchor space, picks next direction (chaotic walk)

**Iteration scaling (locked at start, no mid-iteration rescaling):**
- 1 node match → 3 iterations of dig
- 2 nodes → 4 iterations
- 3 nodes → 5 iterations
- 4-5 nodes → 5 iterations (capped)

Formula: `iterations = min(2 + match_count, 5)`

### Self-Critique Layer (NEW — sits inside Wuxing's umbrella)

Each agent runs six metacognitive questions at iteration boundaries:

1. Am I doing it correctly?
2. Is this the main thing, or am I deflecting from the anchor?
3. Is what I'm finding real, or am I projecting structure?
4. Did I gain anything from the time spent here?
5. Should I continue, return to anchor, or hand off to a sub-agent?
6. What did I learn that the user actually needs?

If self-critique fires red → return to anchor + re-orient, OR close current dig early with an "honest abandonment" report (still valuable — classified as `discarded_for_current_anchor`).

### Wuxing Soft Supervision

Wuxing 5-domain engine = the bridge above the engine room. Reads reports as they come in, downscores ones that don't pass contradiction check, surfaces redirections in synthesis. **Does NOT hard-interrupt agents.** Wandering Room agents keep their freedom; Wuxing influences via scoring, not control.

### Sub-Agent Spawning — Tool Call, NOT MCP

Each agent receives the `spawn_subagent` tool:

```
spawn_subagent(
  focus_area: str,          # what the sub-agent should pursue
  distance_budget: int,     # token budget for the chain
  iteration_count: int,     # 2-5 iterations
  inherits_anchor: True,    # cushion graph passed down
)
```

**Trigger:** Auto-spawn on HIGH match (Absolute Chaos mode) + user-requested via "Dig deeper on report #N" button. Excluded: agent self-elect (would make credit burn unpredictable).

MCPs are for external system access. Sub-agent spawning is internal runtime orchestration → tool call is the right primitive.

### Distance Mechanics (deferred but designed)

Three composite metrics behind one user-facing knob:

- **Primary**: token budget per chain (what we actually control; maps to cost)
- **Secondary**: semantic drift (cosine distance from cushion embedding — triggers return-to-anchor when crossed)
- **Tertiary**: chain depth (max sub-agent recursion per mode)

User-facing UX shows ONE knob — "Pendulum Length" or "Exploration Range" — slider 10–100. Internally maps to (token budget, drift threshold, depth cap) per mode.

(Detailed distance specification is deferred — the real engineering question. Build the structure first; tune the metric later.)

---

## Part IV — Output & Memory

### Report Structure (each exploration produces one)

```json
{
  "report_id": "wander-P03-004",
  "agent_id": "P03",
  "anchor_summary": "<one-line user problem>",
  "domain_explored": "music recognition / jazz / Newton mechanics / etc.",
  "source_locations": [
    {"title": "...", "url": "...", "used_for": "structural comparison"}
  ],
  "matched_layers": {
    "actual": "0/4",
    "essence": "4/5",
    "mechanism": "5/5"
  },
  "match_ratio_summary": "0 surface, 4 essence, 5 mechanism",
  "confidence": "HIGH (structural axes)",
  "exploration_summary": "What I found, in human terms",
  "advancement": "How this advances the anchor",
  "what_does_not_map": "MANDATORY — where the analogy breaks (validates non-empty)",
  "next_lead": "Where to dig further if user wants more"
}
```

**`what_does_not_map` is load-bearing.** Empty submissions fail schema validation and re-prompt the agent. This is the anti-LLM-bullshit primitive that makes partial matches honestly usable.

### Articulation Layer (distinct from Synthesis)

Every report is rendered into this six-field user-facing format:

```
Spark:        What was noticed
Source Shape: What structure it came from
Bridge:       How it maps to the user's problem
Use:          What the user can do with it
Limit:        Where the analogy breaks
Confidence:   How strongly to treat it
```

### Synthesis Layer (Sonnet reads all reports)

Produces:
- Top insights (HIGH confidence)
- Clusters (groups of related partial-matches)
- Contradictions (reports that point opposite directions)
- Opportunity paths
- Open questions
- Recommended next direction
- "What would change the verdict"

### Dossier UI

```
Research Dossier — "[user's anchor summary]"
Time: 30 min   Agents: 10   Reports: 27

█ HIGH confidence (4 reports)
▓ MEDIUM confidence (11 reports)
░ LOW confidence (12 reports) — the Heisenberg zone

[ Each report expandable to Spark/Source Shape/Bridge/Use/Limit/Confidence ]
[ Export to Notion ] [ Discard low-confidence ] [ Dig deeper on report #N ]
```

**LOW confidence is surfaced prominently, NOT buried.** Heisenberg zone. Per Law 3.

### Trace Persistence

The full agent decision trail is preserved per agent:
- Where it looked
- Why it went there
- What it encountered
- Which nodes matched (per layer)
- What it kept
- What it discarded
- Why it returned to anchor

**Discarded clues are classified, NOT deleted:**

| Classification | Meaning |
|---|---|
| `discarded_for_current_anchor` | Off-topic for THIS anchor |
| `possibly_relevant_elsewhere` | Could matter for a different anchor |
| `revisit_later` | Worth checking again with more context |

Future sessions can mine prior discards when new anchor has structural overlap. This is the compounding asset — Constellax accumulates a private archive of "ideas surfaced but not used yet."

### Memory (canonical vs export)

| Layer | Role | Storage |
|---|---|---|
| Graph memory | Canonical — anchors, fingerprints, reports, bridges, traces | Neo4j |
| Vector memory | Canonical — semantic search across stored reports | Embeddings store |
| Internal trace logs | Canonical — full agent decision trails | Filesystem / Neo4j |
| Notion | EXPORT ONLY — polished artifacts for user's workspace | Notion API |

**Notion ≠ memory.** Notion can be ripped out without losing canonical state.

---

## Part V — Build Phases

| Phase | Scope | Duration | Key files |
|---|---|---|---|
| **0** | Cushion encoder + brief composer (four-field form + Sonnet extraction of three-layer cushion) | 3–4 days | `src/wandering/cushion.py`, `src/wandering/composer.py` |
| **1** | Single agent, chaotic walk, decision trace | 3 days | `src/wandering/agent.py`, `src/wandering/policy.py` |
| **2** | Multi-agent parallel + drift mechanics + distance enforcement | 4 days | `src/wandering/runtime.py`, scoring pipeline |
| **3** | Self-critique layer + Wuxing supervision integration | 3 days | `src/wandering/critique.py` |
| **4** | Evaluation + Articulation layers (Spark/Bridge/etc.) | 3 days | `src/wandering/articulate.py` |
| **5** | Synthesis layer + research dossier output | 3 days | extends `src/llm/speech.py` |
| **6** | Visual layer (React Flow primary, Vega-Lite secondary) | 4 days | frontend repo |
| **7** | Sub-pendulum spawn + dig-deeper + Notion export | 3 days | `src/wandering/subpendulum.py` + Notion adapter |

**Total:** ~26 working days, ~5 calendar weeks of focused build.

Phase 0 alone is a usable feature even before wandering exists — "Constellax helps you turn a vague problem into a structured research brief with a cushion graph" is a meaningful slice.

---

## Part VI — Model Strategy

| Role | Model | Why |
|---|---|---|
| Triage / cheap validation | Haiku 4.5 | Fast, structured |
| Anchor building (form → cushion) | Sonnet 4.6 | Three-layer extraction needs depth |
| Bulk wandering agents | DeepSeek V4 Pro | Strong reasoning, cheap, tool use |
| Lightweight scout agents | Haiku 4.5 | Fast iteration |
| Self-critique scoring | Haiku 4.5 | Structured cross-check |
| Wuxing supervision | (existing per-domain assignment) | unchanged |
| Articulation (Spark/Bridge/etc.) | Sonnet 4.6 | Prose quality matters at output |
| Synthesis | Sonnet 4.6 | Combining 30 reports needs depth |

**User can override via Cursor-style model picker** in advanced mode. Default mix per session: ~3 Sonnet / 2 DeepSeek / 5 Haiku (configurable).

---

## Part VII — Pricing (Deferred — see [payment_system_standby.md](payment_system_standby.md))

Per-session credit-based, NOT subscription. Wandering Room is a discrete ceremonial action — different unit economics from chat. Estimated cost per default 30-min session: ~$10. Credit packs with healthy margin. Per-session model picker shows estimated cost up front.

---

## Part VIII — Open Questions (deferred to their implementation phase)

1. **Distance metric calibration** — exact thresholds for token budget, semantic drift, chain depth. Resolve in Phase 2.
2. **30-minute UX wait** — live progress map vs walk-away-and-come-back vs hybrid. Resolve in Phase 6 (UI).
3. **Brief composer mode** — pure form vs conversational intake with Sonnet probing vs hybrid. Resolve in Phase 0 implementation.

---

## What's Locked vs What's Open

**LOCKED:**
- Mission, Laws, Permission boundary
- Three-layer cushion (Actual / Essence / Mechanism)
- Four-field intake form + skip/warning protocol
- Three wandering modes mapped to LOW/MED/HIGH
- Metal detector matching mechanic
- Match strength → iteration scaling (committed at start, no rescaling)
- Sub-agent spawning via tool call (not MCP)
- Self-critique 6 questions
- Wuxing as soft supervisor
- Report structure + mandatory `what_does_not_map`
- Articulation format (Spark/Source Shape/Bridge/Use/Limit/Confidence)
- Trace persistence with classification
- Notion as export, not memory
- Build phase ordering

**OPEN (deferred to implementation):**
- Distance metric specifics
- UX during the wait
- Conversational intake vs pure form

**NEXT:** Phase 0 kickoff — cushion encoder + brief composer. Awaiting user direction to begin.
