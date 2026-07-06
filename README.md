# Constellax

An autonomous **divergent-reasoning engine**: instead of asking one model to answer a hard question,
Constellax sends goal-blind agents wandering across distant knowledge domains for structural analogies,
then fuses what they find — across model families — into grounded, testable proposals.

> **Project status — honest version.** Constellax as a product is **in the architecture phase**; parts
> of the system (memory pipeline, credit/payment scaffolding, product routing) are built but not
> complete. **The core subsystem — the Wandering Room — is built, has run end-to-end on a real
> published open research problem, and is documented with full evidence.** That run, warts and all, is
> this repo's centerpiece. Start with:
>
> - ⭐ [`RESEARCH_CASE_STUDY.md`](RESEARCH_CASE_STUDY.md) — the complete end-to-end run record
> - ⭐ [`PIPELINE_ASSESSMENT.md`](PIPELINE_ASSESSMENT.md) — a deliberately balanced strengths *and*
>   weaknesses audit
> - [`CLAUDE.md`](CLAUDE.md) — the dated decision log (every architecture call and why)

## The Wandering Room (the part that works)

The pipeline takes a structured **cushion** (problem / context / vision / hunches / question), parses
the question into sub-angles, and runs an autonomous multi-cycle loop:

```
cushion ──► goal-BLIND wandering agents          (search distant domains for structural analogies;
               │                                  a three-tier "chaos law" leak-gate strips any goal
               │                                  signal — fails closed, enforced in code)
               ▼
        governor · shepherd · halo               (flow control, drift detection, blind-spot audit —
               │                                  goal-AWARE judges steering up to 4 cycles)
               ▼
        cross-lineage blender                    (Anthropic Opus + DeepSeek R1 — two model families;
               │                                  agreement across lineages is the signal)
               ▼
        grounding stage                          (proposals → explicit math → falsifiable toy model)
```

The load-bearing original idea: **generators never see the goal; judges do.** Convergence-by-prompt is
the failure mode this architecture exists to prevent, and the blindness is mechanically enforced
(regex + n-gram leak gates that drop unlaunderable leads), not aspirational.

### What happened when we pointed it at a real open problem

Target: the coordination-vs-correlation detection gap in *Multi-Agent Risks from Advanced AI*
(Hammond et al., Cooperative AI Foundation, 2025). Four cycles, 218 cards, coverage 0 → 0.8, $24.35,
4h14m. The pipeline **did not hallucinate**, converged on an interventionist separation criterion,
grounded it in do-calculus, and **validated it in a self-checking toy model** (separates coordination
from common-cause correlation, 0.345 vs 0.0001).

**And the honest headline:** the result was *correct but not novel* — a single frontier-model prompt
matched it in ~30 seconds for ~2¢. The full account of why, what that means, and where the wander
underperformed its own design (topic clustering, ~1 keeper per 25 discarded cards) is in the
assessment doc. That evaluation — running the comparison and publishing the unflattering answer — is
the part of this project I'd defend hardest.

## What's genuinely here

- **~139K lines of Python** across 345 files; **43 test files (~18.5K lines)** covering the dispatcher,
  identity system, wandering engine, and bridge
- Fail-open degradation on every external dependency (Neo4j, search, judges) — the pipeline runs with
  reduced signal instead of dying; zero bare `except:` in the codebase
- Honest cost accounting (per-model pricing, true cumulative budget caps) and lossless cancellation
  (a wander cut at its ceiling keeps its finished work)
- Judge position-bias hardening (bidirectional probing before an emergence edge is accepted)
- A memory layer (Neo4j graph + embeddings) and `/api/v2` routing — **work in progress**
- FastAPI server (`server.py`, 22 endpoints) + CLI (`run.py`) + minimal web UI

## What's NOT done (also the honest version)

- The full product around the engine — memory consolidation, credits, multi-user routing — is
  scaffolded, not finished.
- The wander's divergence is weaker than designed: it clusters on topic more than it should.
- No demonstrated *novel* result yet. The engine's edge can only appear on problems whose answer is not
  already latent in a frontier model's weights — finding and testing such a problem is the open front.

## Provenance

Constellax began inside the author's earlier project (**LoRa**, an analytical-reasoning product) and
was split out as an independent system. Two pieces were deliberately ported rather than rewritten, and
their docstrings say so:

- `src/auth/supabase_auth.py` — a Python port of LoRa's TypeScript Supabase JWT middleware
- `src/bridge/` — an adapter mirroring LoRa's Memory-V2 shapes during the transition

Legacy `LORA_`-prefixed environment variables still work as fallbacks; the canonical names are now
`CONSTELLAX_*` (see `.env.example`).

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # set ANTHROPIC_API_KEY (+ optional OPENROUTER/EXA/NEO4J for full signal)
python server.py            # FastAPI on :8100, minimal UI at /
python run.py               # CLI mode
PYTHONPATH=. .venv/bin/python tests/test_wandering_engine.py   # tests are plain-python runners,
                                                               # NOT pytest — see tests/README.md
```

Autonomous wandering runs are launched from `scripts/` (see `scripts/control_room.py` for the knobs);
run artifacts land under `runs/` — the case study's artifacts are preserved there.
