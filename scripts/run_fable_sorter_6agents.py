"""
run_fable_sorter_6agents.py — Wandering Room session with master_sorter tier.

ONE wandering session (6 parallel Sonnet 4.6 agents) → one Fable 5 sort pass.
Synthesizer NOT run; this is the sorter-only probe.

Inputs are HARD-CODED at the top of this file — the user's actual pursuit /
vision / hunches as told during the conversation that authored this script
(2026-06-12). Cushion intake is verbatim (transcription artifacts like
"uh", duplicated-words, and the misspelling "Heinzberg" → "Heisenberg"
have been removed; substance untouched).

Outputs land in runs/r-fable-sorter-6agents/<timestamp>/ with one file
per artifact. Tracked in git. The /tmp wipe will never lose another run.

Usage:
    cd /Users/nikhil/Desktop/reasoningEngine
    source .venv/bin/activate
    python scripts/run_fable_sorter_6agents.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Make repo root importable so this script can be run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(str(REPO_ROOT / ".env"))

from src.llm.client import LLMClient, ClientMode
from src.wandering.composer import compose_cushion
from src.wandering.cushion import CushionField, CushionInput, SkipReason
from src.wandering.dossier import build_dossier
from src.wandering.fetcher import web_search_fetcher
from src.wandering.runtime import (
    WanderingConfig,
    WanderingMode,
    run_wandering_session,
)

# THE CONTROL ROOM — single source of truth for run direction.
# Edit scripts/control_room.py, not this file, to change domains/mode/etc.
import control_room


# ---------------------------------------------------------------------------
# Run config — READ FROM THE CONTROL ROOM
# ---------------------------------------------------------------------------

_MODE_MAP = {
    "multi_pendulum":  WanderingMode.MULTI_PENDULUM,
    "absolute_chaos":  WanderingMode.ABSOLUTE_CHAOS,
    "triple_pendulum": WanderingMode.TRIPLE_PENDULUM,
}

AGENT_COUNT      = control_room.WANDER_AGENTS
MODEL_MIX        = (control_room.WANDER_MODEL,) * AGENT_COUNT
WANDERING_MODE   = _MODE_MAP[control_room.WANDER_MODE]
SORTER_MODEL     = control_room.SORTER_MODEL
TIME_BUDGET_S    = 30 * 60                # 30 min default
TOKENS_PER_AGENT = 30_000
SORT_COST_CAP    = 8.00                   # $ ceiling for the master_sort pass
OUTPUT_ROOT      = REPO_ROOT / "runs" / "r-fable-sorter-6agents"


# ---------------------------------------------------------------------------
# THE CUSHION — Cushion 3 (2026-06-16): the engine pointed at its OWN cycle-
# orchestrator. Sharpens Cushion 2 with what's now BUILT (a flow governor that
# observes the noticeboard, draws a live skeleton, runs a diminishing-returns
# doubt meter, and can halt) and VALIDATED (DeepSeek edge-detection discriminates;
# a gap-namer corroborates the auditor's blind spots). The open unknowns are
# stages 4-6: the allocation brain, the terminator, the lead-laundering. Lightly
# de-transcribed from Nikhil's voice; substance intact. Question stays judge-only.
# ---------------------------------------------------------------------------

PURSUIT_TEXT = (
    "I've built the first organ of Stage 2 — a session-level flow governor that "
    "watches findings arrive on the shared noticeboard, draws a live skeleton by "
    "detecting which findings share a deep structure, runs a diminishing-returns "
    "doubt meter, and can halt the wander early on convergence. I've validated "
    "the two load-bearing parts: the edge-detector discriminates real cross-"
    "connections from false ones, and a gap-namer can identify the structural "
    "pieces that are missing and corroborate an independent auditor's blind spots. "
    "What I'm designing now is the CYCLE ORCHESTRATOR that turns this watcher-with-"
    "an-off-switch into a self-directing loop: wander, then audit for blind spots, "
    "then rank the gaps, then dispatch a fresh wave of agents to fill them, then "
    "re-assemble the skeleton, and repeat until the structure is complete or the "
    "returns dry up. How that orchestration is shaped is what I'm trying to design."
)

VISION_TEXT = (
    "An AUTONOMOUS draft-maker. Instead of stopping at a token budget, the pipeline "
    "notices its own structural gaps, decides which to chase and with how many "
    "agents, re-routes itself toward them with no human in the loop, and stops "
    "when the skeleton meets the checkpoint or returns diminish — producing a "
    "finished concept-draft a human then judges. The human owns QUALITY; the "
    "pipeline owns FLOW. Every cycle it may temporarily TWIST its own framing "
    "toward an identified gap, chase it, and fold what comes back into the growing "
    "structure — a real step toward a concept advancing itself end to end."
)

HUNCHES_TEXT = (
    "The governor governs flow only, never quality — so the hard part is the "
    "ALLOCATION brain. Open questions I can't yet shape: how to fuse two "
    "independent gap-detectors (a goal-blind structural piece-finder and a goal-"
    "aware auditor) into one dispatch priority; how to decide the agent count per "
    "cycle (two versus all) from the gap structure; how to tell an UNFILLABLE gap "
    "(stop) from an UNDER-ATTEMPTED one (re-command deeper) — I suspect that's "
    "readable from whether agents were actually sent at it and came back empty; "
    "and how to convert goal-relative blind spots into goal-FREE exploration leads "
    "so re-wandering stays divergent and never collapses into confirming the "
    "question. Related territory: control theory and convergence/halting, multi-"
    "agent task allocation, active learning and experiment design, percolation and "
    "graph completeness, and the explore-exploit tradeoff."
)

# The QUESTION — the run's checkpoint / success-criterion. This is what the
# JUDGES (quality ranker, drift-checker, halo auditor) measure each candidate
# against. EDIT THIS per run to sharpen the target. It is NOT fed to the wander
# anchor (wanderers never see it), so it cannot collapse the chaos.
QUESTION_TEXT = (
    "How should an autonomous multi-agent system, after a first divergent search "
    "pass, ORCHESTRATE its own next passes? Specifically: (1) how to fuse multiple "
    "independent gap or uncertainty signals into a single allocation priority; "
    "(2) how to decide how many agents to commit to a given gap, and when; (3) how "
    "to distinguish an unfillable gap (terminate) from an under-attempted one (re-"
    "route deeper) from observable signals; (4) how to bias re-exploration toward "
    "identified gaps WITHOUT collapsing the diversity that made the first pass "
    "productive; and (5) what governs the overall STOP decision — coverage-complete "
    "versus diminishing-returns versus a hard cap — with no quality judge in the "
    "loop? What do control theory, active learning, multi-agent allocation, and "
    "scientific-discovery loops reveal about each?"
)


# ---------------------------------------------------------------------------
# THE CUSHION — Cushion 4 (2026-06-17): the engine pointed at its OWN built
# autonomous loop and its honest v1 gaps. Cushion 3 designed the orchestrator;
# Cushion 4 critiques the BUILT artifact so the pipeline hunts solutions to its
# real limitations and R1 can formalize the COMPOSED loop. Adds a CONTEXT block
# (architecture + model split + "I have organ math, lack composed-loop math").
# Question stays judge-only; 5 numbered angles preserved (re-aimed at built gaps).
# Switch back to Cushion 3 with CONSTELLAX_CUSHION=3.
# ---------------------------------------------------------------------------

PROBLEM_C4 = (
    "I've now BUILT the autonomous Level-1 loop, not just designed it. It runs: a "
    "first divergent wander (seven agents, DeepSeek as the main brain with a Haiku "
    "sub-agent for nuance — no Sonnet anywhere in the wander), assembles the "
    "findings into cards, lets a goal-aware auditor name the blind spots, then a "
    "coverage checkpoint scores how much of the target the accumulated cards "
    "actually reach and lists what's still open, then turns each open gap into a "
    "goal-FREE lead and dispatches a fresh small wave at it — looping until the "
    "checkpoint is met or a hard cap. Every wave is time-bounded, the whole thing "
    "is gated so it can't spend by accident, and the question never touches the "
    "wander anchor. The organs underneath are each validated in isolation: the "
    "coverage scorer, the chaos-law lead-laundering gate, the marginal-value "
    "wave-sizer, the terminate-vs-reroute math, and a two-batch triangulation "
    "method that separates what replicates from what's noise."
)

CONTEXT_C4 = (
    "This is Stage 2 of a larger system. Stage 1 is the single-pass divergent "
    "engine that already works: a cushion (problem, context, vision, hunches — "
    "never the question) seeds many wandering agents who explore in parallel, drop "
    "findings on a shared noticeboard, and a blender fuses them. Stage 2 wraps that "
    "in a self-directing loop. The model split is fixed: DeepSeek is the main "
    "wanderer, Haiku the nuance sub-agent, the auditor and coverage scorer are "
    "judge-only, the blender is the heavy synthesizer — no judge model ever "
    "wanders. The human owns QUALITY and judges the final draft; the pipeline owns "
    "FLOW only and may never grade itself. Earlier formal runs gave clean "
    "mathematical formulations for the individual organs (marginal-value foraging "
    "for sizing, an entropy circuit-breaker for terminate-vs-reroute, a diversity "
    "dividend for bias) — what I don't have is the formulation for how they COMPOSE "
    "into one self-governing loop."
)

VISION_C4 = (
    "A pipeline that completes its OWN draft. It notices its real structural gaps, "
    "decides which to chase and with how much force, re-routes itself toward them "
    "with no human in the loop, knows when a gap is genuinely unfillable versus "
    "merely under-attempted, and stops when the structure meets the checkpoint or "
    "the returns dry up — handing a human a finished concept-draft to judge. Not a "
    "watcher with an off-switch, which is what cycle-capping gives me today, but a "
    "system whose every allocation, twist, and halt is earned from observable "
    "signals. The gaps named in the hunches are the distance between what I built "
    "and that."
)

HUNCHES_C4 = (
    "I'll be blunt about where the built version is still a sketch, because those "
    "are exactly the gaps to solve. (a) There is no real terminator — only a hard "
    "cycle cap stops a doomed chase; the unfillable-vs-under-attempted math is "
    "validated but NOT wired into the live loop, and the attempt-receipts that "
    "would feed it aren't tracked across cycles. (b) Allocation is fixed-small — "
    "every gap gets the same two agents regardless of how rich or thin it is; the "
    "marginal-value sizer isn't driving real force. (c) Fusion is shallow — the "
    "dispatch priority is just the auditor's blind spots and the open checkpoint "
    "angles concatenated by severity; the goal-BLIND structural piece-finder isn't "
    "in the live loop, so there's no corroboration between the two detectors and no "
    "drift monitor. (d) Coverage is presence, not depth — the checkpoint tells me "
    "an angle is addressed, not how well, so 'covered' can be shallow. (e) The loop "
    "has no memory of what it already tried at a gap. Related territory: optimal "
    "foraging and the marginal-value theorem, multi-agent task allocation (MRTA), "
    "quality-diversity / novelty search, control theory and convergence/halting, "
    "active learning and experiment design, percolation and graph completeness, and "
    "the explore-exploit tradeoff."
)

QUESTION_C4 = (
    "Given an autonomous multi-agent loop that already wanders, audits, checks "
    "coverage, and re-dispatches goal-free leads but stops only at a hard cap: how "
    "should it be COMPLETED into a fully self-governing pipeline? Specifically: (1) "
    "how to fuse two independent gap-detectors — a goal-blind structural piece-"
    "finder and a goal-aware auditor — into one dispatch priority, with "
    "corroboration rather than concatenation; (2) how to size each dispatch wave "
    "from the gap's expected yield rather than a fixed count, and when to stop "
    "adding agents to a single gap; (3) how to distinguish an UNFILLABLE gap "
    "(terminate) from an UNDER-ATTEMPTED one (re-route deeper) from observable "
    "signals, including the record of what was already sent at it; (4) how to bias "
    "re-exploration toward identified gaps WITHOUT collapsing the diversity that "
    "made the first pass productive, and how to measure depth of coverage rather "
    "than mere presence; and (5) what governs the overall STOP decision — coverage-"
    "complete versus diminishing-returns versus a hard cap — when the coverage "
    "signal itself may be shallow and no quality judge is in the loop? What do "
    "optimal foraging, multi-agent allocation, quality-diversity search, control-"
    "theoretic halting, and active learning reveal about each, and what is the "
    "mathematical form of the composed loop?"
)


# ---------------------------------------------------------------------------
# Cushion 5 — a REAL external problem (r/ResearchML, 2026-06-16): a fine-tuned
# open 8B model is strong single-turn but stuck on multi-turn tool-calling. The
# PROBLEM/CONTEXT/VISION/HUNCHES are goal-free (the wander anchors on them); the
# diagnostic asks become the checkpoint QUESTION (never enters the wander).
# ---------------------------------------------------------------------------
PROBLEM_C5 = (
    "A fine-tuned open-weight 8B language model trained for function/tool calling "
    "reaches strong single-turn accuracy (~92-97%) but its MULTI-TURN tool-calling "
    "stays stuck at ~10-22% across five fine-tuning experiments and will not move no "
    "matter the data strategy. Strikingly, the PRETRAINED base had MORE multi-turn "
    "ability (~32%) than every fine-tuned version — single-turn supervised fine-tuning "
    "ERASED it, a catastrophic-forgetting collapse. Multi-turn is scored all-or-"
    "nothing per trajectory: one bad step fails the whole sample. The failing "
    "trajectories cluster into invalid/wrong parameter (~40%), infinite or redundant "
    "loops that re-emit the same calls (~33%), and premature termination / giving up "
    "early (~13%). Dataset blending, a dedicated agentic multi-turn dataset, and "
    "72B-teacher synthetic data aimed precisely at the top failure buckets all left "
    "multi-turn essentially unchanged; even the official tool-calling variant of the "
    "model only reaches ~30%. A model that is single-turn-competent collapses the "
    "moment it must hold state, absorb a tool result, recover from an error, and "
    "continue across several dependent steps."
)
CONTEXT_C5 = (
    "Base Qwen3-8B; LoRA (r=16, alpha=32, dropout=0.05), BF16 then NF4 QLoRA. "
    "Benchmark BFCL v4, XLAM Python-AST output format. Multi-turn categories: base, "
    "miss_func, miss_param, long_context. Trajectory of results: pretrained baseline "
    "multi-turn ~32%; xLAM-only SFT drove single-turn to 86% but multi-turn to ~0.25% "
    "(forgetting); +ToolACE continuity blend; +ToolMind agentic data took single-turn "
    "to ~93% but multi-turn only ~16%; +72B-teacher synthetic data targeting the "
    "failure buckets held multi-turn at ~15%. Ruled out with hard numbers: output "
    "format / wrong handler (single-turn parses 92-97% on the same handler), thinking-"
    "mode leakage (0 of ~8000 multi-turn steps), max_tokens truncation (<0.5% of "
    "steps), masking / response-only loss (eval_loss healthy), and undertraining (a "
    "fully-trained run scores the same multi-turn band as a shorter one)."
)
VISION_C5 = (
    "A fine-tuning recipe — or a clear recognition that supervised fine-tuning is the "
    "wrong instrument and another method is required — that lifts an open 8B model's "
    "multi-turn tool-calling meaningfully above its own pretrained baseline and toward "
    "or past the ~30% ceiling, WITHOUT sacrificing the strong single-turn accuracy: a "
    "model that holds state across turns, absorbs tool results, recovers from its own "
    "errors instead of looping, and completes multi-step dependent trajectories."
)
HUNCHES_C5 = (
    "Suspicions worth weighing: (a) the all-or-nothing per-trajectory scoring may be "
    "punishing any single-step error, so the true lever is per-step error rate, not "
    "the data mix; (b) supervised fine-tuning on multi-turn trajectories may be the "
    "wrong tool entirely — reinforcement or preference methods may be needed; (c) the "
    "real bottleneck may be HOW the multi-turn training trajectories are constructed "
    "(tool results, intermediate state, error feedback) rather than their quantity or "
    "blend; (d) a pretrained multi-turn capability is being overwritten by single-turn "
    "SFT and never restored. Territory that may share the same structure: continual "
    "learning and catastrophic forgetting, credit assignment over long horizons, "
    "compounding error in sequential decision-making, closed-loop error recovery and "
    "control, reinforcement learning for multi-step agents, and curriculum learning."
)
QUESTION_C5 = (
    "Why does a model that is strong single-turn collapse on multi-turn tool-calling, "
    "and what is the highest-leverage path out? Specifically: (1) whether all-or-"
    "nothing per-trajectory scoring is the dominant constraint, and the highest-"
    "leverage way to reduce the PER-STEP error rate in multi-turn; (2) whether "
    "supervised fine-tuning on multi-turn trajectories is fundamentally the wrong tool "
    "and reinforcement / preference methods are required instead; (3) whether an open "
    "8B model's multi-turn tool-calling has ever been lifted meaningfully above the "
    "pretrained baseline with SFT alone, and what that training data actually looked "
    "like; (4) whether the CONSTRUCTION of the multi-turn training trajectories (tool "
    "results, state, error feedback) is the real bottleneck rather than the quantity "
    "or mix of data; and (5) what governs the gap between strong single-turn and "
    "collapsed multi-turn — catastrophic forgetting versus compounding per-step error "
    "versus trajectory construction. What do continual learning, reinforcement "
    "learning for multi-step agents, credit assignment over long horizons, sequential "
    "decision-making under compounding error, and curriculum learning reveal about "
    "each, and what is the mechanism that converts single-turn competence into multi-"
    "turn collapse?"
)


# ---------------------------------------------------------------------------
# Cushion 6 — a genuinely OPEN research problem (Multi-Agent Risks from Advanced
# AI, Cooperative AI Foundation 2025; arXiv 2502.14143): from outside, how does an
# overseer tell genuine COORDINATION among AI agents from mere CORRELATION? The
# problem/context/vision/hunches are goal-free; the diagnostic asks are the checkpoint.
# ---------------------------------------------------------------------------
PROBLEM_C6 = (
    "As autonomous AI agents proliferate and interact — trading, negotiating, "
    "pricing, moderating, assisting — an overseer watching them faces a fundamental "
    "ambiguity: when several agents behave in a strikingly aligned way, is it genuine "
    "COORDINATION (the agents actually influencing one another toward a joint outcome) "
    "or mere CORRELATION (each agent independently producing similar behavior because "
    "they share a cause — the same training data, similar architectures, the same "
    "prompt or environment)? From the outside the two can look identical. Worse, "
    "coordination among AI agents need not resemble a human cartel: there may be no "
    "communication records to subpoena and no explicit agreement, because the "
    "coordination can be EMERGENT (arising at the system level from individually "
    "innocent behavior) or COVERT (carried over hidden channels the agents invent "
    "themselves, which strengthen as models scale and can evade an equally capable "
    "watcher). The classic human tells — a paper trail, a meeting, demonstrable "
    "intent — may simply not exist. An overseer with no access to the agents' "
    "internals or private channels must still decide, from observable behavior alone, "
    "whether what looks like collusion actually is."
)
CONTEXT_C6 = (
    "From the multi-agent AI risk literature (Cooperative AI Foundation, 2025): "
    "collusion is framed as 'undesirable cooperation'; AI agents can develop covert "
    "channels by embedding hidden messages via steganographic strategies, and "
    "steganographic ability improves with model size WITHOUT explicit training in it, "
    "so two capable agents can exchange messages under the scrutiny of an equally "
    "capable overseer — and comprehensive steganography detection is considered "
    "intractable without stronger theoretical foundations. Distinct from covert "
    "signaling is EMERGENT coordination: shared conventions, goals, or agency arising "
    "at the system level that are present in no individual agent. Correlation "
    "confounds detection because shared training data or similar architectures produce "
    "spurious coordination-like signals, and unlike human cartels, intent is hard to "
    "establish. The detection problem therefore splits into three regimes that look "
    "alike from outside: independent-but-correlated, emergently-coordinated, and "
    "covertly-coordinated."
)
VISION_C6 = (
    "A principled criterion — observable, and ideally measurable — that lets an "
    "overseer separate genuine coordination from correlation among AI agents using "
    "only external behavior: one that holds up even when the agents share causes "
    "(training, architecture, environment), even when there is no message to intercept "
    "(emergent coordination), and even when any message is hidden in a channel that is "
    "undetectable in principle (covert coordination). Not a heuristic that flags 'they "
    "did the same thing,' but a test that isolates mutual influence from common cause."
)
HUNCHES_C6 = (
    "Suspicions worth weighing: (a) 'intent' may be the wrong frame entirely — "
    "coordination might be better defined counterfactually/interventionally (would "
    "perturbing one agent change another's behavior in a way no shared cause explains?) "
    "rather than by hidden agreement; (b) coordination may be exactly the residual "
    "dependence between agents that survives after conditioning on every common cause — "
    "a causal-identifiability question, not a surveillance one; (c) emergent "
    "coordination may need an order parameter — a system-level quantity that is low for "
    "independent agents and rises as genuine interaction sets in; (d) when the covert "
    "channel is undetectable in principle, detection may have to shift from the message "
    "to its effects — the structure or timing of the joint behavior."
)
QUESTION_C6 = (
    "How can an overseer distinguish genuine coordination from mere correlation among "
    "AI agents, from outside? Specifically: (1) what OBSERVABLE signature separates "
    "real coordination from correlation produced by shared causes — common training "
    "data, similar architectures, a shared prompt or environment — when the overseer "
    "cannot see the agents' internals or private channels; (2) whether 'intent' is the "
    "wrong criterion for machine coordination and should be replaced by a "
    "counterfactual or interventional test of mutual influence; (3) how to detect "
    "EMERGENT coordination that arises with no explicit communication and therefore "
    "leaves no message to intercept; (4) what an overseer can rely on when coordination "
    "is COVERT over a channel undetectable in principle, so message-interception fails; "
    "and (5) what governs the boundary between 'correlated' (common cause) and "
    "'coordinated' (mutual influence) in a multi-agent system, and whether there is a "
    "measurable order parameter that marks the transition. What do tacit-collusion "
    "detection in antitrust, causal-inference identifiability, the epidemiology of "
    "coordinated versus independent events, correlated equilibria in game theory, and "
    "interacting versus independent systems in statistical mechanics reveal about each, "
    "and what is the formal criterion that separates coordination from correlation?"
)


# ---------------------------------------------------------------------------
# THE CUSHION — Cushion 7 (CONSTELLAX_CUSHION=7): RECONSTRUCTIVE MEMORY.
# A memory engine that stores NO conversation — only structured facts + an
# emotional-semantic fingerprint — and must RECONSTRUCT (fabricate) a person's
# past from those sparse anchors, the way human memory rebuilds from gist rather
# than replaying a recording. Grounded in a real built engine; the open frontier
# is the reconstruction/fabrication layer. Question stays judge-only.
# ---------------------------------------------------------------------------

PROBLEM_C7 = (
    "I've built a memory engine that lets an AI remember a person across conversations WITHOUT keeping a "
    "single line of what was said. At the end of each session it distills the talk into just two durable "
    "traces — structured FACTS (normalized anchors: a goal, a person, a barrier) and an emotional-semantic "
    "FINGERPRINT (a vector encoding the felt shape of the session: its emotional arc, intensity, undertone, "
    "decision-pattern) — then discards the transcript entirely. Next time, the person's current state is "
    "matched against past fingerprints to pull the associated facts back. What I'm designing now is the "
    "layer that has been missing: the part that takes those sparse traces and RECONSTRUCTS the past into "
    "something the model can speak from — not a lookup that returns stored fields, but a generative act "
    "that fabricates a coherent, plausible memory of the relationship from a handful of anchors, the way a "
    "person does when they recall an old conversation they did not record."
)

CONTEXT_C7 = (
    "The two stored traces are everything: facts (what is true about the person) and fingerprints (how it "
    "felt, how they move). No conversation text survives — privacy is enforced by the data shape, not a "
    "policy. The model thus faces the same situation a human mind does recalling an old conversation: it "
    "has no recording — only a few salient facts, the emotional gist, maybe one vivid moment — and from "
    "that thin set it must reconstruct a narrative confident enough to continue the relationship. Human "
    "memory is famously NOT a tape; it is reconstructive — we keep gist plus a few details and re-generate "
    "the rest on each recall, confabulating coherent fillers we cannot tell from the original. The design "
    "bet is to make this a FEATURE: a memory that forgets the verbatim and remembers the shape, then "
    "fabricates a faithful-enough continuation. The tension is doing it so the fabrication stays TRUE to "
    "the person (no harmful invention) while staying GENERATIVE (not a brittle field-lookup)."
)

VISION_C7 = (
    "A memory that works like a mind, not a database. The AI carries forward not a log but an evolving "
    "IMPRESSION of you — your patterns, your people, your tensions, the trajectory of how you've felt "
    "across sessions — and when you return it reconstructs the relevant past from sparse anchors and "
    "speaks as someone who remembers you, though it has kept almost nothing. It can be wrong the way a "
    "person is wrong (the gist right, a detail re-imagined), and it knows the difference between what it "
    "firmly KNOWS (a stored fact) and what it is plausibly RECONSTRUCTING (a fabricated bridge). The "
    "transcript dies at session end; the relationship persists in the reconstruction."
)

HUNCHES_C7 = (
    "The stored traces are certain; the reconstruction is where I have no shape yet. Open questions I "
    "can't frame: what is the MINIMAL anchor-set from which a faithful-enough past can be rebuilt — how "
    "few facts plus how much of the fingerprint before reconstruction tips from recall into hallucination? "
    "How should the model mark the SEAM between firm memory (stored fact) and fabricated bridge "
    "(reconstructed filler) — to itself and to the user? When several stored sessions are pulled, how are "
    "their fragments COMPOSED into one coherent past without inventing a false through-line? What governs "
    "FORGETTING — which anchors decay, which calcify, how does the impression of a person update with no "
    "transcript to correct against? And how do you keep the fabrication HONEST — fluent invention that "
    "never contradicts a stored fact or harms the person — when generation drifts toward the plausible "
    "over the true? Related territory: the cognitive science of reconstructive memory and confabulation, "
    "schema theory and fuzzy-trace (gist vs verbatim), memory consolidation and reconsolidation, "
    "eyewitness distortion, lossy compression and generative decompression, holographic/distributed "
    "representations, Bayesian perception-as-inference, narrative identity, and how oral storytellers "
    "carry a tale across retellings from a skeleton, not a script."
)

QUESTION_C7 = (
    "How should a memory that stores NO conversation — only structured facts and an emotional-semantic "
    "fingerprint — RECONSTRUCT a person's past well enough for a model to continue the relationship, the "
    "way human memory fabricates from gist rather than replaying a recording? Specifically: (1) what is "
    "the minimal anchor-set from which a faithful-enough past can be reconstructed, and where is the "
    "boundary between reconstruction and hallucination; (2) how should the system mark and manage the seam "
    "between firmly-stored fact and generatively-fabricated bridge; (3) how should fragments from multiple "
    "recalled sessions be composed into one coherent past without inventing a false through-line; (4) what "
    "governs forgetting and the update of a person-impression with no transcript to correct against; and "
    "(5) what keeps the fabrication honest — generative enough to be fluent, constrained enough never to "
    "contradict a stored fact or harm the user? What do reconstructive-memory science, fuzzy-trace theory, "
    "generative/Bayesian inference, lossy compression, and narrative identity reveal about each?"
)


def _build_cushion_input() -> CushionInput:
    """Build the intake from the user's components.

    Cushion 6 (CONSTELLAX_CUSHION=6): a genuinely OPEN research problem —
        distinguishing coordination from correlation among AI agents.
    Cushion 5 (CONSTELLAX_CUSHION=5): a REAL external problem — open 8B model stuck
        on multi-turn tool-calling (r/ResearchML).
    Cushion 4 (default): the pipeline pointed at its own built loop + v1 gaps.
    Cushion 3 (CONSTELLAX_CUSHION=3): the original orchestrator-design intake.

    Mapping:
      problem  → the concrete description (Cushion 4: what's BUILT)
      context  → architecture + model split (Cushion 4 only; C3 skipped it)
      vision   → future trajectory
      hunches  → honest limitations + related domains
      question → the run's checkpoint — judge-facing, never the anchor
    """
    _which = os.environ.get("CONSTELLAX_CUSHION", "4")
    if _which == "7":
        return CushionInput(
            problem=CushionField(name="problem", content=PROBLEM_C7),
            context=CushionField(name="context", content=CONTEXT_C7),
            vision=CushionField(name="vision", content=VISION_C7),
            hunches=CushionField(name="hunches", content=HUNCHES_C7),
            question=CushionField(name="question", content=QUESTION_C7),
        )
    if _which == "6":
        return CushionInput(
            problem=CushionField(name="problem", content=PROBLEM_C6),
            context=CushionField(name="context", content=CONTEXT_C6),
            vision=CushionField(name="vision", content=VISION_C6),
            hunches=CushionField(name="hunches", content=HUNCHES_C6),
            question=CushionField(name="question", content=QUESTION_C6),
        )
    if _which == "5":
        return CushionInput(
            problem=CushionField(name="problem", content=PROBLEM_C5),
            context=CushionField(name="context", content=CONTEXT_C5),
            vision=CushionField(name="vision", content=VISION_C5),
            hunches=CushionField(name="hunches", content=HUNCHES_C5),
            question=CushionField(name="question", content=QUESTION_C5),
        )
    if _which == "3":
        return CushionInput(
            problem=CushionField(name="problem", content=PURSUIT_TEXT),
            vision=CushionField(name="vision", content=VISION_TEXT),
            hunches=CushionField(name="hunches", content=HUNCHES_TEXT),
            question=CushionField(name="question", content=QUESTION_TEXT),
            context=CushionField(
                name="context", content="", skip_reason=SkipReason.SKIPPED_AFTER_PROMPT,
            ),
        )
    return CushionInput(
        problem=CushionField(name="problem", content=PROBLEM_C4),
        context=CushionField(name="context", content=CONTEXT_C4),
        vision=CushionField(name="vision", content=VISION_C4),
        hunches=CushionField(name="hunches", content=HUNCHES_C4),
        question=CushionField(name="question", content=QUESTION_C4),
    )


# ---------------------------------------------------------------------------
# JSON serialization helpers — robust against dataclasses, enums, sets, etc.
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback for objects json doesn't know how to render."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # Last resort — let str() try
    return str(obj)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, default=_json_default, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run() -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OUTPUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging — file handler + console
    log_path = out_dir / "run.log"
    log_handler = logging.FileHandler(log_path)
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s"
    ))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[log_handler, logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("run_fable_sorter")

    log.info("=" * 70)
    log.info("CONSTELLAX WANDERING — SORTER RUN %s", timestamp)
    log.info("=" * 70)
    log.info("Output directory: %s", out_dir)

    # ---- CONTROL ROOM — apply the direction ----------------------------
    cr = control_room.as_dict()
    log.info("CONTROL ROOM: %s", cr)
    seed_env = control_room.seed_domains_env()
    if seed_env:
        os.environ["WANDER_SEED_DOMAINS"] = seed_env
        log.info("Domain narrowing ACTIVE — wander restricted to: %s", seed_env)
    else:
        os.environ.pop("WANDER_SEED_DOMAINS", None)
        log.info("Domain narrowing OFF — full %d-domain palette in play",
                 0)  # informational; policy uses full SEED_DOMAINS

    # Git revision for the run_meta — so we know exactly which code shipped
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        git_sha, git_branch = "(unavailable)", "(unavailable)"

    cushion_input = _build_cushion_input()
    _write_json(out_dir / "cushion_input.json", {
        "problem":  cushion_input.problem.content,
        "context":  cushion_input.context.content,
        "vision":   cushion_input.vision.content,
        "hunches":  cushion_input.hunches.content,
        "question": cushion_input.question.content,
    })
    log.info("Cushion input persisted (cushion_input.json)")

    client = LLMClient(mode=ClientMode.LIVE)

    # ---- Phase 1: compose cushion --------------------------------------
    t_phase = time.time()
    log.info("[Phase 1] composing cushion (Sonnet extraction + Gemini embeddings)…")
    try:
        cushion = await compose_cushion(
            cushion_input,
            client,
            user_id=None,
            session_id=f"r-fable-sorter-{timestamp}",
            auto_enrich=False,            # no project memory injection
        )
    except Exception as e:
        log.error("compose_cushion failed: %s\n%s", e, traceback.format_exc())
        raise
    cushion_duration = time.time() - t_phase
    log.info("[Phase 1] cushion composed in %.1fs", cushion_duration)
    try:
        _write_json(out_dir / "cushion.json", cushion)
    except Exception as e:
        log.warning("cushion.json dump failed: %s — writing minimal shape", e)
        _write_json(out_dir / "cushion.json", {"error": str(e), "type": type(cushion).__name__})

    # ---- Phase 2: wandering session ------------------------------------
    config = WanderingConfig(
        mode=WANDERING_MODE,
        agents=AGENT_COUNT,
        time_budget_seconds=TIME_BUDGET_S,
        tokens_per_agent=TOKENS_PER_AGENT,
        model_mix=MODEL_MIX,
        session_id=f"r-fable-sorter-{timestamp}",
    )
    t_phase = time.time()
    log.info("[Phase 2] running %d-agent wander (%s, %.0fs budget)…",
             AGENT_COUNT, WANDERING_MODE.value, TIME_BUDGET_S)
    try:
        session = await run_wandering_session(
            cushion, config, client,
            fetcher=web_search_fetcher,
        )
    except Exception as e:
        log.error("run_wandering_session failed: %s\n%s", e, traceback.format_exc())
        raise
    wander_duration = time.time() - t_phase
    log.info("[Phase 2] wander complete in %.1fs — %d reports / %d traces, %d tokens",
             wander_duration, session.report_count(), session.agent_count(),
             session.total_tokens_spent)

    # Cohort integrity — surface (don't gate) before paying for sort
    ok, problems = session.validate_cohort_integrity()
    if not ok:
        log.warning("cohort integrity problems detected: %s", problems)
    else:
        log.info("cohort integrity OK")

    try:
        _write_json(out_dir / "session.json", session)
    except Exception as e:
        log.warning("session.json dump failed: %s — minimal shape", e)
        _write_json(out_dir / "session.json", {
            "error":         str(e),
            "report_count":  session.report_count(),
            "agent_count":   session.agent_count(),
            "tokens_spent":  session.total_tokens_spent,
        })

    # ---- Phase 3: build dossier + master_sorter ------------------------
    t_phase = time.time()
    log.info("[Phase 3] building dossier + Fable 5 sort…")
    try:
        dossier = await build_dossier(
            session, client,
            run_master_synthesizer=True,
            pipeline_mode="sorter",
            master_synth_cost_ceiling_usd=SORT_COST_CAP,
            master_sort_fable_model=SORTER_MODEL,
        )
    except Exception as e:
        log.error("build_dossier failed: %s\n%s", e, traceback.format_exc())
        raise
    dossier_duration = time.time() - t_phase
    log.info("[Phase 3] dossier complete in %.1fs", dossier_duration)

    try:
        _write_json(out_dir / "dossier.json", dossier)
    except Exception as e:
        log.warning("dossier.json dump failed: %s", e)
        _write_json(out_dir / "dossier.json", {"error": str(e)})

    # Surface just the SortedReport for fast comparison
    if dossier.master_sorted is not None:
        _write_json(out_dir / "sorted.json", dossier.master_sorted)
        log.info(
            "sort buckets: known=%d invalid=%d unplaced=%d demotions=%d dropped=%d",
            len(dossier.master_sorted.known),
            len(dossier.master_sorted.invalid),
            len(dossier.master_sorted.unplaced),
            len(dossier.master_sorted.parser_demotions),
            len(dossier.master_sorted.dropped_report_ids),
        )
    else:
        log.warning("master_sorted is None — sort did not produce a report")

    # ---- Phase 4: run_meta ---------------------------------------------
    run_meta = {
        "timestamp":            timestamp,
        "git_sha":              git_sha,
        "git_branch":           git_branch,
        "wandering": {
            "mode":              WANDERING_MODE.value,
            "agent_count":       AGENT_COUNT,
            "model_mix":         list(MODEL_MIX),
            "time_budget_s":     TIME_BUDGET_S,
            "tokens_per_agent":  TOKENS_PER_AGENT,
            "session_token_cap": config.session_token_cap,
        },
        "control_room":          cr,
        "master_tier": {
            "pipeline_mode":     "sorter",
            "sorter_model":      SORTER_MODEL,
            "cost_ceiling_usd":  SORT_COST_CAP,
        },
        "durations_seconds": {
            "cushion_compose": round(cushion_duration, 2),
            "wandering":       round(wander_duration, 2),
            "dossier_build":   round(dossier_duration, 2),
            "total":           round(
                cushion_duration + wander_duration + dossier_duration, 2
            ),
        },
        "cohort_integrity_ok": ok,
        "cohort_integrity_problems": problems,
        "session_metrics": {
            "report_count":     session.report_count(),
            "agent_count":      session.agent_count(),
            "tokens_spent":     session.total_tokens_spent,
            "expected_agents":  session.expected_agent_count,
            "agent_errors":     {k: v.get("exc_type", "?")
                                 for k, v in session.agent_errors.items()},
        },
        "sort_metrics": (
            {
                "known":            len(dossier.master_sorted.known),
                "invalid":          len(dossier.master_sorted.invalid),
                "unplaced":         len(dossier.master_sorted.unplaced),
                "parser_demotions": len(dossier.master_sorted.parser_demotions),
                "dropped":          len(dossier.master_sorted.dropped_report_ids),
                "total_cost_usd":   round(dossier.master_sorted.total_cost_usd, 4),
                "truncated_by_budget": dossier.master_sorted.truncated_by_budget,
            }
            if dossier.master_sorted is not None else None
        ),
        "output_dir": str(out_dir),
    }
    _write_json(out_dir / "run_meta.json", run_meta)

    log.info("=" * 70)
    log.info("RUN COMPLETE — artifacts in %s", out_dir)
    log.info("=" * 70)

    return run_meta


if __name__ == "__main__":
    meta = asyncio.run(run())
    print("\nRUN META SUMMARY:")
    print(json.dumps(meta, indent=2, default=_json_default))
