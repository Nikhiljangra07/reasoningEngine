"""
Constellation Interpreter — chaos-amplifying score-profile generator.

DOCTRINE
========
The Wandering Room is a cohort architecture. A single session runs 4-10
agents in parallel; the user's mind is the integrator that resolves the
cohort's collective output. This interpreter is a LEAF utility called
once per (content, cushion) pair — many times per session, across many
agents.

Per-agent precision is NOT the objective. Cohort-level recall on rare
structural matches is. A high per-agent F1 actively suppresses chaos —
it collapses the channel disagreement that IS the breakthrough zone
the user's wander is built to surface.

This file used to be a relevance classifier with a cheap-first ladder,
a structural-foothold gate, and a non-map veto. It scored F1=0.927 on
a fixture. It was wrong. Industry-standard rigor in a per-agent
classifier is exactly the optimization the Seven Laws reject:

  LAW 1  chaos is feature, not noise           → no pre-filtering
  LAW 2  insight in user's head, not system    → no vetoes
  LAW 3  LOW band = breakthrough zone          → surface partial signal
  LAW 7  structural essence over surface       → dynamics carry verdict

The interpreter is now a SCORE-PROFILE GENERATOR with PER-AGENT BIAS
DIVERSITY. Each call deterministically picks one of five decision rules
from a hash of (url, content_hash). Different agents in the same wander
naturally land on different rules because they fetch different URLs —
the cohort spreads across the bias population for free, without needing
the agent layer to know about modes.

ARCHITECTURE
============
Seven channels still run per call:

  1. VECTOR     — cosine between content fingerprint and cushion nodes
  2. OVERLAP    — count of cushion nodes the fingerprint touches
  3. ROLE       — structural role match (degraded neutral until wired)
  4. MECHANISM  — shared causal pattern, judged by Haiku
  5. EVIDENCE   — URL-domain credibility heuristic
  6. NOVELTY    — has this content been seen this session?
  7. NON-MAP    — where does the analogy fail, articulated by Haiku

ALL channels run on every non-empty fingerprint. The cheap-first ladder
is GONE — pre-filtering on cheap channels suppresses exactly the
candidates whose mechanism-channel signal would have been the
breakthrough. Cost: ~2x LLM calls per session vs. legacy. Within the
multi-agent absorption budget; the cohort architecture licenses it.

BIAS MODES
==========
Five decision rules. Each call picks one deterministically:

  aggressive          — any single positive surfaces; multi-positive DIGs
  conservative        — ≥3 positives + mechanism for DIG
  mechanism_only      — trust structural channels; ignore surface entirely
  non_map_amplifier   — distortion (non_map) is itself signal, not veto
  random              — stochastic ε-accept of low-score candidates

Mode selection is deterministic-but-distributed: same (url, content_hash)
always picks the same mode (debuggable, reproducible), but different
URLs land on different modes (cohort diversity for free).

DISAGREEMENT AS SIGNAL — EMPIRICALLY REVERSED
=============================================
The cross-channel std-dev is computed every call and exposed as
`verdict.disagreement`. The original prior was that high disagreement
marked the breakthrough zone (channels can't collapse → user's pattern
lives at the boundary).

The 30-pair adversarial sweep proved the opposite, decisively:
  - real cross-domain analogies cluster at LOW disagreement (mean
    0.211, max 0.217) — channels CONVERGE on a real structural pattern
  - surface-similar non-analogies cluster at HIGH disagreement (mean
    0.380, max 0.420) — channels FIGHT because the pair is noise some
    channels are catching and others aren't

So high disagreement is a free, accurate non-match detector. Every
decider that gates on `disagreement >= DISAGREEMENT_HEISENBERG` routes
to SAVE_FOR_LATER (never DIG). When channels are fighting, the
candidate is preserved for the user to inspect but does NOT burn dig
budget. Low disagreement with partial signal remains the breakthrough
zone the LOW band surfaces.

This is the doctrine-vs-data correction made on 2026-06-01 after the
initial Heisenberg-as-DIG rule was shown to be wired backwards against
its own measured signal.

NON-MAP IS NOT A VETO
=====================
The legacy interpreter treated "non_map didn't find a failure mode" as
suspicious-surface-match and voted SKIP. That's a Law 2 violation — the
system pre-deciding for the user. The new interpreter keeps non_map as
a positive signal in every mode (a found failure mode = real analogy)
and never uses its absence as a veto. The `non_map_amplifier` mode
explicitly inverts the legacy view: HIGH non_map score AMPLIFIES the
candidate's DIG-worthiness rather than disqualifying it.

SHIM CONTRACT
=============
`agent.py:_verdict_to_match_result` reads only:
  - verdict.decision     ∈ {dig, save_for_later, skip}
  - verdict.matched_nodes
  - verdict.reason

New fields on InterpreterVerdict (bias_mode, disagreement,
channel_score_profile) are additive with defaults. The shim is
unaffected.

Original Phase 4 created 2026-06-01. Chaos-amp recalibration 2026-06-01.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from src.identity import compose_system_prompt
from src.llm.client import LLMClient, LLMResponse
from src.wandering.cushion import CushionGraph, CushionNode
from src.wandering.fingerprint import ContentFingerprint


log = logging.getLogger("constellax.wandering.interpreter")


# ---------------------------------------------------------------------------
# Constants — channel thresholds + bias mode population
# ---------------------------------------------------------------------------

#: Cosine threshold for the vector channel to count as "positive".
#: Same value as legacy (0.6) — this isn't a tuned classifier knob, it's
#: a sensible cut between near-zero noise and meaningful cosine signal.
VECTOR_THRESHOLD = 0.6

#: Per-channel "positive" thresholds (used by ChannelScores.positive_count
#: and by every bias-mode decider).
ROLE_POSITIVE = 0.5
MECHANISM_POSITIVE = 0.5
EVIDENCE_POSITIVE = 0.5

#: Cross-channel-variance threshold above which the candidate routes to
#: SAVE_FOR_LATER (never DIG) — the empirical non-match detector.
#: Measured 2026-06-01 against the 30-pair adversarial sweep:
#:   - should-match pairs: min=0.195, mean=0.211, max=0.217
#:   - should-not-match pairs: min=0.330, mean=0.380, max=0.420
#: 0.30 cleanly separates the two clusters. High disagreement = channels
#: fighting = pair is probably surface noise, not structural rhyme.
#: Preserved as SAVE_FOR_LATER so the user can still inspect; never DIG
#: because dig budget is finite and would be wasted on a likely non-match.
DISAGREEMENT_HEISENBERG = 0.30

#: Stochastic-accept floor in `random` mode. When deterministic noise
#: exceeds this, accept a single-positive verdict as DIG, or accept a
#: zero-positive verdict as SAVE_FOR_LATER. This is the "release the
#: valve" mechanism — not free noise, but capped license for the
#: cohort to surface long-tail breakthrough candidates.
RANDOM_DIG_NOISE_FLOOR = 0.55
RANDOM_SAVE_NOISE_FLOOR = 0.75

#: Haiku route for the LLM-mediated channels.
JUDGE_DOMAIN = "psychology"
MECHANISM_CONCEPT = "interpreter_mechanism_judge"
NON_MAP_CONCEPT = "interpreter_non_map_judge"

#: The bias mode population — order matters for deterministic selection
#: (hash % len), so do not reorder without intent.
BIAS_MODES: tuple[str, ...] = (
    "aggressive",
    "conservative",
    "mechanism_only",
    "non_map_amplifier",
    "random",
)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class ChannelScores:
    """Per-channel score for one (fingerprint, cushion) judgment.

    All scores are floats in [0, 1] except `overlap` (count) and
    `cushion_node_distances` (auxiliary diagnostic map). Every bias
    mode operates on this struct; the deciders differ only in how
    they READ these values, never in what's measured.
    """

    vector:    float = 0.0
    overlap:   int   = 0
    role:      float = 0.5
    mechanism: float = 0.0
    evidence:  float = 0.0
    novelty:   float = 1.0
    non_map:   float = 0.0
    cushion_node_distances: dict[str, float] = field(default_factory=dict)

    def positive_count(self) -> int:
        """How many channels crossed their positive threshold. Used by
        deciders that aggregate, not by mechanism_only or
        non_map_amplifier."""
        count = 0
        if self.vector >= VECTOR_THRESHOLD:
            count += 1
        if self.overlap >= 1:
            count += 1
        if self.role >= ROLE_POSITIVE:
            count += 1
        if self.mechanism >= MECHANISM_POSITIVE:
            count += 1
        if self.evidence >= EVIDENCE_POSITIVE:
            count += 1
        if self.novelty >= 1.0:
            count += 1
        if self.non_map >= 1.0:
            count += 1
        return count

    def profile_vector(self) -> list[float]:
        """Normalized vector of all channel scores — used by the
        disagreement computation and exposed on InterpreterVerdict for
        downstream dossier work."""
        return [
            self.vector,
            min(1.0, self.overlap / 3.0),  # 3+ touched nodes saturates
            self.role,
            self.mechanism,
            self.evidence,
            self.novelty,
            self.non_map,
        ]


Decision = Literal["dig", "skip", "save_for_later"]


@dataclass
class InterpreterVerdict:
    """The final judgment the interpreter returns for one content piece.

    Additive fields (bias_mode, disagreement, channel_score_profile)
    have defaults so the shim in agent.py is unaffected. Downstream
    dossier work can read them when ready.
    """

    decision:      Decision
    scores:        ChannelScores
    matched_nodes: list[str]
    failure_mode:  str = ""
    reason:        str = ""
    #: Which bias mode produced this verdict — populated by interpret()
    bias_mode:     str = ""
    #: Cross-channel std-dev; >= DISAGREEMENT_HEISENBERG = chaos zone
    disagreement:  float = 0.0
    #: Snapshot of the 7-channel score profile (post-normalization)
    channel_score_profile: list[float] = field(default_factory=list)


@dataclass
class SessionState:
    """Per-session state the interpreter consults.

    Used by the novelty channel to detect content the wander already
    saw earlier in the same session. Updated by `mark_seen()` AFTER a
    verdict — the interpreter doesn't mark on its own.
    """

    seen_content_hashes: set[str] = field(default_factory=set)
    seen_urls:           set[str] = field(default_factory=set)

    def mark_seen(self, fingerprint: ContentFingerprint) -> None:
        if fingerprint.content_hash:
            self.seen_content_hashes.add(fingerprint.content_hash)
        if fingerprint.url:
            self.seen_urls.add(fingerprint.url)

    def is_novel(self, fingerprint: ContentFingerprint) -> bool:
        if fingerprint.content_hash and fingerprint.content_hash in self.seen_content_hashes:
            return False
        if fingerprint.url and fingerprint.url in self.seen_urls:
            return False
        return True


# ---------------------------------------------------------------------------
# Cheap channels — pure functions, no I/O
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def score_vector(
    fingerprint: ContentFingerprint,
    cushion: CushionGraph,
) -> tuple[float, list[CushionNode], dict[str, float]]:
    """Vector channel — cosine between fingerprint embedding and cushion
    node embeddings."""
    if not fingerprint.embedding:
        return 0.0, [], {}

    all_distances: dict[str, float] = {}
    candidates: list[tuple[float, CushionNode]] = []

    for layer in cushion.layers():
        records = layer.node_records
        if not records:
            continue
        for rec in records:
            if not rec.embedding:
                continue
            sim = _cosine(fingerprint.embedding, rec.embedding)
            all_distances[rec.text] = sim
            if sim >= VECTOR_THRESHOLD:
                candidates.append((sim, rec))

    candidates.sort(key=lambda t: t[0], reverse=True)
    touched_nodes = [c[1] for c in candidates]
    max_cosine = candidates[0][0] if candidates else (
        max(all_distances.values()) if all_distances else 0.0
    )
    return max_cosine, touched_nodes, all_distances


def score_overlap(touched_nodes: list[CushionNode]) -> int:
    return len(touched_nodes)


def score_role(
    touched_nodes: list[CushionNode],
    fingerprint: ContentFingerprint,  # noqa: ARG001 — Phase 4 placeholder
) -> float:
    """Role channel — degraded neutral (0.5) until role topology is
    wired into both cushion and content. Stays at 0.5 (does not bias
    deciders) when neutral."""
    if not touched_nodes:
        return 0.0
    return 0.5


_TRUSTED_DOMAINS = {
    "wikipedia.org", "wikiquote.org", "wikisource.org",
    "arxiv.org", "ssrn.com", "nature.com", "science.org", "cell.com",
    "pnas.org", "thelancet.com", "nejm.org", "plos.org",
    "stanford.edu", "mit.edu", "harvard.edu", "berkeley.edu",
    "ox.ac.uk", "cam.ac.uk", "ucl.ac.uk",
    "britannica.com", "smithsonianmag.com", "nationalgeographic.com",
    "nytimes.com", "washingtonpost.com", "bbc.com", "reuters.com",
    "economist.com", "theatlantic.com", "newyorker.com",
}

_SUSPICIOUS_DOMAINS = {
    "pinterest.com", "medium.com",
    "ezinearticles.com", "buzzfeed.com",
}


def score_evidence(fingerprint: ContentFingerprint) -> float:
    """Evidence channel — domain credibility heuristic. One signal of
    seven; modes are free to ignore it (mechanism_only does).
    """
    domain = (fingerprint.domain or "").lower().strip()
    if not domain:
        return 0.5
    if ":" in domain:
        domain = domain.split(":", 1)[0]

    for trusted in _TRUSTED_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return 1.0

    if domain.endswith(".edu") or domain.endswith(".gov"):
        return 0.7

    for sus in _SUSPICIOUS_DOMAINS:
        if domain == sus or domain.endswith("." + sus):
            return 0.3

    return 0.5


def score_novelty(
    fingerprint: ContentFingerprint,
    session_state: SessionState | None,
) -> float:
    if session_state is None:
        return 1.0
    return 1.0 if session_state.is_novel(fingerprint) else 0.0


# ---------------------------------------------------------------------------
# LLM channels — mechanism + non_map
# ---------------------------------------------------------------------------


_MECHANISM_SYSTEM_PROMPT = """\
You are Constellax's mechanism-match judge.

You are given:
  1. A user's CUSHION — a structural pattern describing their problem
     (concrete entities, essence-layer dynamics, and underlying causal
     mechanism).
  2. A piece of CONTENT (described by its structural fingerprint).

Your single judgment: does the content operate by the SAME CAUSAL
MECHANISM as the cushion? Not the same surface topic, not the same
keywords — the same underlying causal pattern.

# CRITICAL DISTINCTION

A mechanism is a CAUSAL primitive: "X → Y because Z." If you can write
the same causal sentence with both the cushion's variables and the
content's variables, they share a mechanism. If you can't, they don't.

Examples of shared mechanism:
  - "fixed structure enables variation" — same mechanism in jazz
    improvisation and AI agent control
  - "positive feedback amplifies small inputs" — same mechanism in
    debt spirals and political polarization
  - "tacit knowledge transmits only via proximity" — same mechanism in
    apprenticeship and mentorship

Examples of shared topic but DIFFERENT mechanism:
  - Compound interest (exponential growth) and traffic timing
    (constraint optimization) — both involve "rates" but different math
  - Career change (irreversible commitment) and printer drivers
    (interface compatibility) — both involve "switching" but different
    forces

# OUTPUT FORMAT

Return ONE valid JSON object:

{
  "score": <float 0.0 to 1.0>,
  "shared_mechanism": "<one short sentence naming the causal primitive, or empty if score < 0.5>"
}

Score conventions:
  0.0-0.3 — clearly different mechanism
  0.4-0.6 — partial overlap or unclear
  0.7-1.0 — shared mechanism, can be stated as one causal primitive

No prose preamble. No code fences. JUST the JSON object.
"""


_NON_MAP_SYSTEM_PROMPT = """\
You are Constellax's non-map judge.

A real cross-domain analogy is NOT identical to the cushion — it shares
structure at the essence/mechanism layer, but the surface, scale,
mechanism details, or scope inevitably DIFFER. Naming exactly where the
analogy breaks is a sign of structural honesty.

You are given:
  1. A user's CUSHION pattern.
  2. A piece of CONTENT (its fingerprint).

Your job: identify ONE specific way this content's pattern does NOT map
onto the cushion. Not "they're completely different" (that's just no
match). One SPECIFIC place where the analogy stops working — a scope
difference, a mechanism detail that's inverted, a scale mismatch, a
variable that exists on one side but not the other.

# WHY THIS MATTERS

If you can find a meaningful break point, the rest is probably a real
analogy. If you can't find any break point (the analogy seems perfect),
that's suspicious — it usually means the analogy is empty or surface-
level rather than structurally deep.

# OUTPUT FORMAT

Return ONE valid JSON object:

{
  "failure_mode": "<one sentence naming where the analogy specifically breaks, or empty if no meaningful break point exists>",
  "found": <true if a meaningful failure mode was identified, false if the analogy seems too perfect or is just empty>
}

Examples:

Content: jazz improvisation vs Cushion: AI agent control
  failure_mode: "Jazz improvisation operates in continuous real-time
                 audio with no recovery from a bad note, while AI agents
                 can backtrack and retry within a budget."
  found: true

Content: tax brackets vs Cushion: AI agent control
  failure_mode: ""
  found: false  (no real analogy here; the question doesn't apply)

No prose preamble. No code fences. JUST the JSON object.
"""


def _strip_code_fences(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    return fenced.group(1).strip() if fenced else text.strip()


def _extract_json_object(text: str) -> str:
    text = _strip_code_fences(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON in judge response")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    raise ValueError("unterminated JSON in judge response")


def _build_judge_payload(
    fingerprint: ContentFingerprint,
    touched_nodes: list[CushionNode],
    cushion: CushionGraph,
) -> str:
    blocks = ["# CUSHION (the user's problem structure)"]
    for layer in (cushion.essence, cushion.mechanism):
        if layer.summary:
            blocks.append(f"\n## {layer.name.upper()}")
            blocks.append(layer.summary)
        if layer.nodes:
            blocks.append("Nodes: " + ", ".join(layer.nodes[:6]))

    blocks.append("\n# CONTENT FINGERPRINT")
    if fingerprint.domain:
        blocks.append(f"Source domain: {fingerprint.domain}")
    if fingerprint.phrases:
        blocks.append("Structural phrases:")
        for p in fingerprint.phrases:
            blocks.append(f"  - {p}")
    else:
        blocks.append("(no fingerprint phrases extracted)")

    if touched_nodes:
        blocks.append("\n# Cushion nodes the vector channel already matched:")
        for n in touched_nodes[:5]:
            blocks.append(f"  - {n.layer}: {n.text}")

    return "\n".join(blocks)


async def score_mechanism(
    fingerprint: ContentFingerprint,
    touched_nodes: list[CushionNode],
    cushion: CushionGraph,
    client: LLMClient,
) -> float:
    user_message = _build_judge_payload(fingerprint, touched_nodes, cushion)
    try:
        response: LLMResponse = await client.call(
            system_prompt=compose_system_prompt(
                _MECHANISM_SYSTEM_PROMPT, mode="interpreter_mechanism"),
            user_message=user_message,
            domain=JUDGE_DOMAIN,
            concept=MECHANISM_CONCEPT,
        )
    except Exception as e:
        log.warning("mechanism judge raised: %s", e)
        return 0.0

    if not response.success or not response.content:
        log.warning("mechanism judge failed: %s", response.error or "no content")
        return 0.0

    try:
        json_text = _extract_json_object(response.content)
        payload = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("mechanism judge response unparseable: %s", e)
        return 0.0

    if not isinstance(payload, dict):
        return 0.0
    try:
        score = float(payload.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


async def score_non_map(
    fingerprint: ContentFingerprint,
    touched_nodes: list[CushionNode],
    cushion: CushionGraph,
    client: LLMClient,
) -> tuple[float, str]:
    """Non-map channel — Haiku identifies a specific failure mode.

    Returns (score, failure_mode_text). score is 1.0 if a meaningful
    failure mode was found, 0.0 otherwise.

    NOTE on chaos amp: the score is no longer used as a veto by any
    decider. High non_map is a positive signal (real analogy with
    honest break point). Low non_map is one of seven channels —
    weighted differently per mode but never a SKIP override.
    """
    user_message = _build_judge_payload(fingerprint, touched_nodes, cushion)
    try:
        response: LLMResponse = await client.call(
            system_prompt=compose_system_prompt(
                _NON_MAP_SYSTEM_PROMPT, mode="interpreter_non_map"),
            user_message=user_message,
            domain=JUDGE_DOMAIN,
            concept=NON_MAP_CONCEPT,
        )
    except Exception as e:
        log.warning("non_map judge raised: %s", e)
        return 0.0, ""

    if not response.success or not response.content:
        log.warning("non_map judge failed: %s", response.error or "no content")
        return 0.0, ""

    try:
        json_text = _extract_json_object(response.content)
        payload = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("non_map judge response unparseable: %s", e)
        return 0.0, ""

    if not isinstance(payload, dict):
        return 0.0, ""

    found = bool(payload.get("found", False))
    failure_mode = str(payload.get("failure_mode") or "").strip()
    if found and failure_mode:
        return 1.0, failure_mode
    return 0.0, failure_mode


# ---------------------------------------------------------------------------
# Bias modes — per-agent rule diversity via deterministic selection
# ---------------------------------------------------------------------------


def _stable_hash(*parts: str) -> int:
    """Stable cross-process hash. Python's built-in hash() is salted
    per-process; we need reproducibility across runs for debugging.
    SHA-256 of the joined parts, take first 8 bytes as unsigned int.
    """
    joined = "\x1f".join(p or "" for p in parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _pick_bias_mode(fingerprint: ContentFingerprint) -> str:
    """Deterministic-but-distributed bias-mode selection.

    Same (url, content_hash) → same mode every time (debug + repro).
    Different URLs naturally land on different modes; an agent fetching
    20 URLs in a session hits ~all 5 modes ~4 times each → cohort gets
    per-agent rule diversity for free with no agent-layer change.

    When both url and content_hash are empty (degenerate input), falls
    back to `aggressive` rather than randomizing — degenerate fingerprints
    are usually empty or near-empty, and aggressive's high-recall
    posture matches what we want when signal is sparse.
    """
    url = fingerprint.url or ""
    ch = fingerprint.content_hash or ""
    if not url and not ch:
        return "aggressive"
    seed = _stable_hash(url, ch)
    return BIAS_MODES[seed % len(BIAS_MODES)]


def _channel_variance(scores: ChannelScores) -> float:
    """Cross-channel std-dev — the disagreement signal.

    High variance = channels strongly disagree about whether this
    content matches the cushion. Disagreement IS the Heisenberg signal:
    the system cannot collapse to a verdict because the structural
    pattern lives at the boundary where surface and dynamics diverge.
    Used by every decider to bias toward DIG / SAVE_FOR_LATER on
    high-variance candidates.
    """
    values = scores.profile_vector()
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _decide_aggressive(
    scores: ChannelScores,
    disagreement: float,
) -> tuple[Decision, str]:
    """High-recall mode. Any positive signal surfaces. Strong single
    channel OR multi-positive DIGs. High channel-disagreement routes to
    SAVE (empirically: high disagreement = non-match, see DOCTRINE)."""
    pos = scores.positive_count()
    # Empirical non-match detector — high cross-channel disagreement
    # means the channels are fighting, which the data shows is the
    # signature of a surface-noise pair, NOT a breakthrough zone.
    if disagreement >= DISAGREEMENT_HEISENBERG:
        return "save_for_later", (
            f"aggressive: high disagreement={disagreement:.2f} ≥ "
            f"{DISAGREEMENT_HEISENBERG} — channels fighting, likely "
            f"non-match (pos={pos}). Preserve as SAVE, do not burn dig."
        )
    if pos >= 2 or scores.mechanism >= 0.6 or scores.non_map >= 1.0:
        return "dig", (
            f"aggressive: pos={pos}, mech={scores.mechanism:.2f}, "
            f"non_map={scores.non_map:.1f}"
        )
    if pos >= 1:
        return "save_for_later", f"aggressive: 1 positive (pos={pos})"
    return "skip", "aggressive: zero positives across 7 channels"


def _decide_conservative(
    scores: ChannelScores,
    disagreement: float,
) -> tuple[Decision, str]:
    """High-precision mode. ≥3 positives AND mechanism for DIG. The
    cohort balance — without this mode the cohort would over-surface."""
    pos = scores.positive_count()
    # Heisenberg still surfaces, just at lower commitment.
    if disagreement >= DISAGREEMENT_HEISENBERG and pos >= 2:
        return "save_for_later", (
            f"conservative: heisenberg disagreement={disagreement:.2f} "
            f"with pos={pos} → preserve, don't dig"
        )
    if pos >= 3 and scores.mechanism >= MECHANISM_POSITIVE:
        return "dig", (
            f"conservative: {pos}/7 positives + mech={scores.mechanism:.2f} "
            f"→ multi-channel convergence with structural signal"
        )
    if pos >= 2:
        return "save_for_later", (
            f"conservative: {pos}/7 positives — preserve for later iteration"
        )
    return "skip", f"conservative: {pos}/7 positives — below threshold"


def _decide_mechanism_only(
    scores: ChannelScores,
    disagreement: float,
) -> tuple[Decision, str]:
    """Surface is noise; trust only structural channels (mechanism +
    non_map). Catches cross-domain analogies the surface-weighted modes
    suppress. This is the chaos cohort's structural-essence anchor."""
    if scores.mechanism >= 0.6 and scores.non_map >= 1.0:
        return "dig", (
            f"mechanism_only: mech={scores.mechanism:.2f} + non_map=1.0 "
            f"(real analogy, honest break point)"
        )
    if scores.mechanism >= 0.4 or scores.non_map >= 1.0:
        return "save_for_later", (
            f"mechanism_only: mech={scores.mechanism:.2f}, "
            f"non_map={scores.non_map:.1f} — partial structural signal"
        )
    if disagreement >= DISAGREEMENT_HEISENBERG:
        return "save_for_later", (
            f"mechanism_only: weak structural signal but disagreement="
            f"{disagreement:.2f} → preserve as Heisenberg candidate"
        )
    return "skip", (
        f"mechanism_only: mech={scores.mechanism:.2f}, "
        f"non_map={scores.non_map:.1f} — no structural signal"
    )


def _decide_non_map_amplifier(
    scores: ChannelScores,
    disagreement: float,
) -> tuple[Decision, str]:
    """Distortion IS signal — but it needs an ANCHOR.

    Haiku can hallucinate a 'failure mode' for almost anything, so
    non_map=1.0 alone is not enough. The corroboration rule: non_map=1
    DIGs only when there is also at least one structural reading
    (mechanism > 0 OR vector ≥ threshold OR overlap ≥ 1). Pure
    non_map=1 with zero anchor preserves as SAVE_FOR_LATER — distortion
    is signal not veto, but it is not promotion either.

    High disagreement routes to SAVE (empirical non-match detector).
    """
    pos = scores.positive_count()
    has_structural_anchor = (
        scores.mechanism > 0.0
        or scores.vector >= VECTOR_THRESHOLD
        or scores.overlap >= 1
    )
    # Heisenberg inversion — high disagreement = non-match, not breakthrough.
    if disagreement >= DISAGREEMENT_HEISENBERG:
        return "save_for_later", (
            f"non_map_amp: high disagreement={disagreement:.2f} ≥ "
            f"{DISAGREEMENT_HEISENBERG} — channels fighting. "
            f"Preserve as SAVE (pos={pos}, non_map={scores.non_map:.1f})."
        )
    if scores.non_map >= 1.0 and has_structural_anchor:
        return "dig", (
            f"non_map_amp: non_map=1.0 + structural anchor "
            f"(mech={scores.mechanism:.2f}, vec={scores.vector:.2f}, "
            f"overlap={scores.overlap}) — real analogy with honest break"
        )
    if pos >= 2 and (scores.mechanism > 0.0 or scores.vector >= VECTOR_THRESHOLD):
        return "dig", (
            f"non_map_amp: pos={pos} with structural anchor "
            f"(mech={scores.mechanism:.2f}, vec={scores.vector:.2f})"
        )
    if pos >= 1 or scores.non_map >= 1.0:
        return "save_for_later", (
            f"non_map_amp: signal without anchor — preserve "
            f"(pos={pos}, non_map={scores.non_map:.1f}, "
            f"structural_anchor={has_structural_anchor})"
        )
    return "skip", "non_map_amp: zero signal across channels"


def _decide_random(
    scores: ChannelScores,
    disagreement: float,
    fingerprint: ContentFingerprint,
) -> tuple[Decision, str]:
    """Stochastic acceptance — the explicit chaos valve, released into
    BREADTH (SAVE_FOR_LATER), not DEPTH (DIG).

    The chaos-amp doctrine says randomness creates breakthroughs. But
    the data (random mode sweep precision = 57% at DIG) shows the valve
    was opened too wide: stochastic DIG promotion was producing
    coin-flip-quality breakthrough surfaces. The fix:

      - DIG requires pos ≥ 3, OR (pos ≥ 2 AND mechanism ≥ POSITIVE).
        Bare pos ≥ 2 was the largest random-mode FP contributor.
      - Stochastic acceptance now releases the candidate into
        SAVE_FOR_LATER (breadth surfacing), not into DIG.
      - Mechanism corroboration required for any stochastic DIG.
      - High disagreement routes early to SAVE (Heisenberg empirical
        non-match detector).

    Reproducibility preserved: same (url, content_hash) → same
    decision every time. Different agents fetching different URLs land
    on different deterministic noise values, so the cohort still
    spreads across the stochastic envelope.
    """
    pos = scores.positive_count()
    noise = (_stable_hash(
        fingerprint.content_hash or "", fingerprint.url or "",
    ) % 1000) / 1000.0

    # Heisenberg inversion — high disagreement = non-match, not breakthrough.
    if disagreement >= DISAGREEMENT_HEISENBERG:
        return "save_for_later", (
            f"random: high disagreement={disagreement:.2f} ≥ "
            f"{DISAGREEMENT_HEISENBERG} — chaos with discipline: "
            f"preserve, do not dig (pos={pos}, noise={noise:.2f})"
        )

    # Tightened DIG — requires multi-positive convergence OR a 2-positive
    # case anchored by mechanism. Bare pos≥2 was the FP-pipe; the
    # mechanism requirement is the discipline that keeps the valve from
    # being mushy.
    if pos >= 3 or (pos >= 2 and scores.mechanism >= MECHANISM_POSITIVE):
        return "dig", (
            f"random: pos={pos}, mech={scores.mechanism:.2f}, "
            f"noise={noise:.2f} → multi-positive convergence with structural signal"
        )

    # Stochastic acceptance now promotes to SAVE_FOR_LATER (breadth),
    # not to DIG (depth). DIG can still fire from this branch ONLY when
    # there is a corroborating mechanism — a corroborated stochastic
    # case is closer to a structural rhyme than a true random catch.
    if pos >= 1 and noise > RANDOM_DIG_NOISE_FLOOR and scores.mechanism > 0.0:
        return "dig", (
            f"random: 1 positive + mech={scores.mechanism:.2f} + "
            f"noise={noise:.2f} > {RANDOM_DIG_NOISE_FLOOR:.2f} → "
            f"corroborated stochastic dig"
        )

    # Any positive signal, OR a stochastic acceptance, OR an honest
    # non_map distortion → surface as SAVE so the cohort can see it.
    if (pos >= 1
            or noise > RANDOM_SAVE_NOISE_FLOOR
            or scores.non_map >= 1.0):
        return "save_for_later", (
            f"random: pos={pos}, noise={noise:.2f}, "
            f"non_map={scores.non_map:.1f} → preserve (breadth)"
        )

    return "skip", (
        f"random: zero signal even with stochastic floor "
        f"(pos={pos}, noise={noise:.2f}, threshold={RANDOM_SAVE_NOISE_FLOOR:.2f})"
    )


def _decide(
    scores: ChannelScores,
    disagreement: float,
    bias_mode: str,
    fingerprint: ContentFingerprint,
) -> tuple[Decision, str]:
    """Dispatcher — routes to the bias-mode decider.

    The five modes intentionally have DIFFERENT thresholds and
    DIFFERENT channel weightings. Cohort-level diversity emerges from
    different agents (fetching different URLs) deterministically
    landing on different modes — not from any single decider being
    'correct'.
    """
    if bias_mode == "aggressive":
        return _decide_aggressive(scores, disagreement)
    if bias_mode == "conservative":
        return _decide_conservative(scores, disagreement)
    if bias_mode == "mechanism_only":
        return _decide_mechanism_only(scores, disagreement)
    if bias_mode == "non_map_amplifier":
        return _decide_non_map_amplifier(scores, disagreement)
    if bias_mode == "random":
        return _decide_random(scores, disagreement, fingerprint)
    # Unknown mode falls back to aggressive — high recall is the safer
    # failure direction in a cohort architecture.
    log.warning("unknown bias mode %r, falling back to aggressive", bias_mode)
    return _decide_aggressive(scores, disagreement)


# ---------------------------------------------------------------------------
# Aggregator — interpret()
# ---------------------------------------------------------------------------


async def interpret(
    fingerprint: ContentFingerprint,
    cushion: CushionGraph,
    *,
    client: LLMClient,
    session_state: SessionState | None = None,
    skip_llm_channels: bool = False,
    bias_mode: str | None = None,
) -> InterpreterVerdict:
    """Run all seven channels and produce a final verdict.

    ALL channels run on every non-empty fingerprint — no cheap-first
    ladder. Cheap channels are computed synchronously, then mechanism +
    non_map run in parallel via asyncio.gather.

    `skip_llm_channels=True` short-circuits the LLM calls — useful for
    unit tests exercising only the cheap path.

    `bias_mode` overrides the deterministic per-(url, content_hash)
    selection. None (default) = auto-pick. Pass a specific mode for
    targeted testing or for callers that want to enforce a specific
    posture (e.g., 'conservative' for a verification pass).

    Failure handling:
      - Empty fingerprint (no embedding, no phrases) → SKIP
      - Any individual channel raises → that channel scores 0.0; the
        verdict still computes from the remaining channels.
    """
    scores = ChannelScores()
    matched_nodes_text: list[str] = []

    # Pick the bias mode FIRST so empty-fingerprint skips still record
    # which mode would have run (useful for cohort-level analytics).
    mode = bias_mode or _pick_bias_mode(fingerprint)

    # Early-exit: empty or missing fingerprint.
    if not fingerprint.embedding and not fingerprint.phrases:
        return InterpreterVerdict(
            decision="skip",
            scores=scores,
            matched_nodes=[],
            reason="empty fingerprint (no embedding, no phrases)",
            bias_mode=mode,
            disagreement=0.0,
            channel_score_profile=scores.profile_vector(),
        )

    # === Cheap channels (synchronous) ===

    vec, touched, distances = score_vector(fingerprint, cushion)
    scores.vector = vec
    scores.cushion_node_distances = distances
    scores.overlap = score_overlap(touched)
    scores.role = score_role(touched, fingerprint)
    scores.evidence = score_evidence(fingerprint)
    scores.novelty = score_novelty(fingerprint, session_state)
    matched_nodes_text = [n.text for n in touched]

    # === LLM channels (parallel) — no gate. Chaos amp licenses this. ===

    failure_mode = ""
    if not skip_llm_channels:
        mech_task = asyncio.create_task(
            score_mechanism(fingerprint, touched, cushion, client),
        )
        non_map_task = asyncio.create_task(
            score_non_map(fingerprint, touched, cushion, client),
        )
        mech_result, non_map_result = await asyncio.gather(
            mech_task, non_map_task, return_exceptions=True,
        )

        if isinstance(mech_result, BaseException):
            log.warning("mechanism channel raised: %s", mech_result)
            scores.mechanism = 0.0
        else:
            scores.mechanism = float(mech_result)

        if isinstance(non_map_result, BaseException):
            log.warning("non_map channel raised: %s", non_map_result)
            scores.non_map = 0.0
        else:
            nm_score, failure_mode = non_map_result
            scores.non_map = float(nm_score)

    disagreement = _channel_variance(scores)
    decision, reason = _decide(scores, disagreement, mode, fingerprint)

    return InterpreterVerdict(
        decision=decision,
        scores=scores,
        matched_nodes=matched_nodes_text,
        failure_mode=failure_mode,
        reason=f"[{mode}] {reason}",
        bias_mode=mode,
        disagreement=disagreement,
        channel_score_profile=scores.profile_vector(),
    )


__all__ = [
    "VECTOR_THRESHOLD",
    "ROLE_POSITIVE",
    "MECHANISM_POSITIVE",
    "EVIDENCE_POSITIVE",
    "DISAGREEMENT_HEISENBERG",
    "RANDOM_DIG_NOISE_FLOOR",
    "RANDOM_SAVE_NOISE_FLOOR",
    "BIAS_MODES",
    "ChannelScores",
    "Decision",
    "InterpreterVerdict",
    "SessionState",
    "score_vector",
    "score_overlap",
    "score_role",
    "score_evidence",
    "score_novelty",
    "score_mechanism",
    "score_non_map",
    "interpret",
]
