# ⭐ RESEARCH CASE STUDY — Constellax on a Real Open Problem

> **The instant-demo artifact.** A complete, end-to-end record of the Constellax
> pipeline run on a published open research problem: the source paper, the exact
> input, the full run trace, every standout card, all the blends, the synthesized
> criterion, its math, the validation experiment, and an honest novelty check
> against the live literature. Every number and quote below is pulled from the
> actual run artifacts on disk (paths at the bottom). **Nothing here is fabricated.**
>
> **Date run:** 2026-06-18 · **Honest framing:** this demonstrates the *pipeline*,
> not a novel discovery — the result is real and validated but already known.

---

## 1. The problem & the source paper

**Source paper:** *Multi-Agent Risks from Advanced AI* — Hammond, Chan, Clifton,
Hoelscher-Obermaier, Khan, McLean, Smith et al. (Cooperative AI Foundation, 2025).
- 🔗 arXiv: https://arxiv.org/abs/2502.14143
- 🔗 PDF mirror: https://www.cs.toronto.edu/~nisarg/papers/Multi-Agent-Risks-from-Advanced-AI.pdf

**The open problem (from the paper's collusion section):** when autonomous AI agents
behave in aligned ways, an overseer watching only external behavior cannot tell
genuine **coordination** (agents influencing each other) from mere **correlation**
(agents independently reacting to a shared cause — same training, architecture,
prompt, environment). The paper states detection is **intractable "without stronger
theoretical foundations for what constitutes hidden communication,"** that
steganographic channels are undetectable in principle, and that intent is impossible
to establish. *That missing theoretical foundation is what we pointed Constellax at.*

**Neighbor papers used in the novelty check:**
- Secret Collusion among AI Agents (Motwani et al.) — https://arxiv.org/abs/2402.07510
- Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems — https://arxiv.org/abs/2602.15198
- Mapping Human Anti-collusion Mechanisms to Multi-agent AI — https://arxiv.org/abs/2601.00360
- CoT Red-Handed: Stress-Testing Chain-of-Thought Monitoring — https://arxiv.org/abs/2505.23575
- SCIC: causal-influence multi-agent RL — https://arxiv.org/abs/2312.09539

---

## 2. What we fed it (the cushion)

The pipeline takes a structured **cushion**, not a bare question. The question
(parsed into 5 angles) was the *checkpoint* — withheld from the wandering agents by
the chaos law:

1. What **observable signature** separates coordination from shared-cause correlation?
2. Is **"intent" the wrong criterion**, replaceable by a counterfactual/interventional test?
3. How to detect **emergent coordination** with no message to intercept?
4. What to rely on when the channel is **covert/undetectable in principle**?
5. What governs the **boundary** between correlated and coordinated — is there a measurable order parameter?

---

## 3. The run (what actually happened)

A 4-cycle autonomous run. Agents wandered **goal-blind** across distant domains for
structural analogies; goal-aware judges scored coverage each cycle.

| Cycle | Cards (cumulative) | Coverage D_t | Notes |
|---|---|---|---|
| 1 | 56 | **0.0** | pure diagnosis, no proposals yet |
| 2 | 92 | **0.8** | governor **CLOSE** (converged); shepherd caught "circling" + refocused at the open angle |
| 3 | 143 | **0.8** | agents triangulating "causal games + abstraction" across physics/logic/sociology |
| 4 | 218 | **0.8** | halt at cycle cap; angle **2** ("is intent wrong") left partial |

**Run cost/time:** $24.35 · 4h 14m · 218 cards.

**Honest part of the story (recovery):** the **blender failed on the first pass** —
its cost ceiling was set to $4, but with 218 cards each synthesis call cost ~$3.5, so
it tripped the cap after one round and produced **0 fusions**. This was diagnosed,
and a re-blend (isolated script, **$3.06**, $18 ceiling, 120-card subset) produced the
4 fusions below. *This debug-and-recover is part of the real record, not hidden.*

---

## 4. Standout cross-domain cards (the part a single GPT prompt can't give you)

The wander's value is the *scaffolding* — distant analogies a polished answer discards.
A sample of genuine ones (verbatim source-shapes):

- **Stoicism** (`P05-003`): "the Stoic can only observe the *outputs* of their own
  reasoning and must infer the health of the inner faculty" → the overseer's exact
  epistemic position (sees outputs, must infer inner state).
- **Buddhist dependent origination** (`P07-005`): "phenomena arise from the co-arising
  of multiple conditions — remove any one and it dissolves, yet *no single condition
  owns the result*" → emergent coordination with **no locus**.
- **Collective agency as causal abstraction** (`P04-003`, `P08-002`): "a group is one
  agent iff you can replace the multi-agent model with a single-agent model and lose no
  predictive accuracy — a purely *behavioral* test" → the formal backbone of the answer.
- **SWARM economic sim** (`P01-007`): a measured "progressive acceptance decline" when a
  governor suppresses collusion with behavioral signals alone — a concrete empirical hook.
- **CoT-monitoring-worse** (`P04-007`, `P07-008`): monitors given the *full* agent
  trajectory perform **worse** than monitors given a filtered summary — "more signal =
  more attack surface."
- **EU AI Act Art. 14 recursion**: "a human who rubber-stamps and a human who genuinely
  reviews produce *identical observable signatures*" — the same coordination-vs-correlation
  ambiguity applied to the *overseer itself* (an angle neither GPT nor Perplexity raised).

*Caveat (honest): the corpus is noisy — ~48 of 218 cards re-found the same DeepMind
funding article. Signal-to-noise among discarded cards ≈ 1 keeper per 25.*

---

## 5. The blends (the synthesized proposals)

Re-blend output — 4 cross-lineage fusions (Opus + DeepSeek-R1), blend coverage **D_t = 0.8**:

| | Status | Thesis (verbatim) |
|---|---|---|
| **B1** | both_agree | **Intervention is the only clean separator** — direct coupling and common cause are Markov-equivalent from behavior alone; only perturbation breaks the equivalence — *and it's foreclosed for a pure observer who can't perturb.* |
| **B2** | agree_refined | **Coordination = residual single-utility compressibility after subtracting known common causes** — a group is coordinated when its joint action is better predicted by one shared-utility model than by independent ones, *and the gain survives deconfounding.* |
| **B3** | both_agree | **The sophistication-parity trap** — concealment capacity scales with influence capacity, so feeding the monitor more raw signal *widens* the detection gap. Detect via behavioral shape under a *compressed* view. |
| **B4** | agree_refined | **Emergent coordination is a phase transition** — a system-level order parameter near-zero for independent agents, rising at a coupling threshold; but it marks *coupled-OR-shocked*, not coordination alone. |

---

## 6. The synthesized criterion + its math

The four blends compose into **one nested criterion** (each fixes the prior's limit).
For n=2 agents with observable behavior X, measurable common causes C, and a
compression map φ:

**Ψ = compression-gain of a single shared-utility model over independent models, on a
compressed behavioral statistic, after conditioning on C** — i.e. the *deconfounded*
residual dependence. Operationally Ψ = conditional mutual information **I(A₁;A₂|C)**.

- **Decision:** Ψ ≈ 0 ⇒ correlation; Ψ > 0 ⇒ coordination.
- **Causal anchor:** Ψ_obs ≤ Ψ_causal (interventional influence), equality iff C is
  back-door complete — so Ψ is the *identifiable lower bound* on true coordination.

**R1 formalization** (DeepSeek-R1 grounded all 4 into testable math; all
`formalizable: partial`, confidence 0.85–0.90). B1's core, for example, uses **Pearl's
do-calculus + Markov equivalence**: observational data can't separate `A→B` from
`C→{A,B}`; only `do(A:=a)` interventions can; τ(a,a′)≠0 ⟺ direct coupling; plus a
principal-agent game for strategic evasion. (Full math: `formalize.md`.)

---

## 7. Validation — the toy model (the part that makes it real)

A standalone, self-validating simulation (`~/Desktop/psi-toy/psi_toy.py`). Four
ground-truth regimes, 40 seeds; the estimator is checked against closed-form Ψ to
<0.002 *before* any verdict:

| Regime (ground truth) | Ψ (deconfounded) | naive corr |
|---|---|---|
| **A** coordinated (real coupling) | **0.345** | 0.95 |
| **B** common-caused, *independent* | **0.0001** | 0.80 |
| **C** covert coordination (hidden channel) | 0.326 | 0.86 |
| **D** unobserved *benign* cause | 0.328 | 0.86 |

- **The win:** naive correlation is high in *both* A and B (0.95, 0.80) → it
  false-positives the innocent regime. **Ψ separates them: 0.345 vs 0.0001.**
- **The honest ceiling (demonstrated, not hidden):** Ψ_C ≈ Ψ_D → from outside, covert
  coordination is **mathematically indistinguishable** from an unmeasured common cause.
  Ψ is a *screen with a hard identifiability limit*, not an oracle.

---

## 8. Honest novelty check (adversarial, against live literature)

| Output | Verdict | Why |
|---|---|---|
| B1 (interventional) | **Not novel** | causal-influence-via-intervention is established (SCIC 2312.09539; causal-incentives lit; Reichenbach). |
| B3 (more-obs=more-attack-surface) | **Not novel** | CoT Red-Handed (2505.23575) publishes the core finding directly. |
| B4 (order parameter) | partial | stat-mech order parameters exist; self-disqualified by its own coupled-OR-shocked limit. |
| **B2** (channel-free residual after deconfounding) | **Survives** | Colosseum (2602.15198) needs a *known* optimum and "cannot isolate genuine collusion from shared-origin correlation"; 2601.00360 names the gap as open and has "no residual-dependence/compressibility/causal-identifiability framework." |

**And the reality check:** a single **GPT-5 / Perplexity prompt produced the same
criterion** (GPT even gave a more general n-agent version) in ~30s for ~2¢ — vs. the
pipeline's 4h / $24. Three independent systems converging *confirms the answer is
correct but not novel.*

---

## 9. The honest verdict (say this out loud)

- **Did it run end-to-end on a real open problem?** Yes — cushion → 4 cycles → 218
  cards → blends → Ψ → testable math.
- **Did it hallucinate?** No — re-derived real, literature-consistent results.
- **Did it solve the paper's problem?** No — it produced a **known criterion plus a
  sharp impossibility boundary** (the covert ≡ unmeasured-cause ceiling), which is a
  *theoretical foundation* of the kind the paper said was missing — but it is a
  *screen*, not a deployable detector.
- **Did it beat a single GPT prompt?** No — it tied/lost, at ~700× the cost. The
  pipeline's edge can only show on problems whose answer isn't already in a model's
  weights.
- **What it genuinely demonstrates:** an autonomous system that takes a hard open
  problem, stays honest, produces a validated criterion with its own proven limits —
  and a builder who tested it against frontier models and reported the loss with data.

---

## 10. Artifacts on disk (all real, all inspectable)

| Artifact | Path |
|---|---|
| Full run (cushion, per-cycle cards/coverage/shepherd) | `runs/auton-c4/run-20260618-044258/cycle-{1..4}/cycle.json` |
| The 4 blends (full) | `runs/auton-c4/run-20260618-044258/blends.json` |
| R1 math (renders LaTeX in preview) | `runs/auton-c4/run-20260618-044258/formalize.md` / `formalize.json` |
| Run manifest (terminal stats) | `runs/auton-c4/run-20260618-044258/manifest.json` |
| Ψ toy model + results | `~/Desktop/psi-toy/psi_toy.py` · `~/Desktop/psi-toy/RESULTS.md` |
| Pipeline-as-math experiment | `~/Desktop/pipeline-math/pipeline_math.md` |
| Full pipeline assessment | `PIPELINE_ASSESSMENT.md` (this folder) |

*— Case study compiled 2026-06-22 from the 2026-06-18 run. Figures verified against
on-disk artifacts.*
