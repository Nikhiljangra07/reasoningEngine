# Constellax — Demo (interview replay)

A **self-contained, offline** replay of the real autonomous run on the AI
coordination-vs-correlation problem. Built to show employers without running the
live pipeline (which takes ~30 min/cycle and costs money). Leads with honesty so it
survives scrutiny.

## How to open
Just **double-click `index.html`** — it opens in any browser, works offline, needs no
server. (`data.js` is loaded as a local script, not a network fetch, so `file://`
works fine.) On a screen-share it auto-plays the run timeline once; the **▶ Replay**
button re-runs the animation.

## What it shows (in order)
the open problem + source paper → the 4-cycle run with coverage climbing 0→0.8 →
cross-domain analogy cards (Stoicism, Buddhism, stat-mech…) → the 4 synthesized
blends → the criterion Ψ + its math → a self-validating toy-model table → an
adversarial novelty check → an honest "what it shows / what it doesn't" panel.

## Safety / provenance
- Isolated: touches **none** of the production UI (`../web/`), server, or pipeline.
- **No fabricated data.** `data.js` is generated from the on-disk run artifacts.

## Regenerate the data
```
python3 build_demo.py        # re-reads runs/auton-c4/run-20260618-044258/* → data.js
```

Companion docs (repo root): `RESEARCH_CASE_STUDY.md` (the full written record),
`PIPELINE_ASSESSMENT.md` (strengths/weaknesses audit).
