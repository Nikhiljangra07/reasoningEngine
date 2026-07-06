# r-fable-sorter-6agents

Each timestamped subdirectory holds one complete run of the wandering room
with the **master sorter** tributary (Fable 5, single-pass classification)
as the master tier. The synthesizer (Opus 4.6 + GPT-5.4) is NOT run on these.

## Files per run (`runs/r-fable-sorter-6agents/<timestamp>/`)

| File | What it holds |
|---|---|
| `cushion.json` | The composed `CushionGraph` — three-layer structural representation extracted from the user's 3-component intake (pursuit / vision / hunches). |
| `session.json` | Every `ExplorationReport` and `DecisionTrace` from the 6 wandering agents, plus token usage and per-call telemetry. |
| `dossier.json` | The full assembled Dossier — confidence-banded cards + synthesis map + the sorter's output. This is the user-facing artifact. |
| `sorted.json` | Just the `SortedReport` extracted from the dossier, for fast comparison across runs. Three buckets (known / invalid / unplaced) + parser_demotions + dropped_report_ids. |
| `run_meta.json` | Run config and observed metrics: model mix, agent count, time/cost, durations per phase, environment, code revision. The audit trail. |
| `cushion_input.json` | The exact `CushionInput` text passed in (pursuit/vision/hunches). Source of truth for reproducibility. |
| `run.log` | Captured stdout/stderr from the runner. |

## Comparison protocol

When comparing two runs (e.g. run A on 6 agents vs run B on 12 agents):

1. **Bucket distribution** — diff the count of (known / invalid / unplaced) between runs. The interesting axis is `unplaced` — the candidate-gold bucket.
2. **Parser demotions** — high `parser_demotions` = the model is calling things "known" without naming them. Tracks hallucinated recognition.
3. **Dropped cards** — non-empty `dropped_report_ids` means the sorter lost cards. Should be zero.
4. **`known` references** — read the `prior_work_name` + `reference` fields. Are they real published works? This is the model's recognition quality.
5. **`unplaced` reasoning** — read `why_unplaced`. Genuine novelty has a different texture than well-dressed nonsense; the sorter's reasoning is your reading guide, not the verdict.
6. **Cost & latency** — `run_meta.json.total_cost_usd` and per-phase durations show what the architecture costs at this scale.

## Subdir naming

`<timestamp>` is local-time `YYYYMMDD-HHMMSS` so runs sort chronologically.
Subdirs are NEVER overwritten — each run gets its own dir.
