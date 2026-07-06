"""Regression test suite for the chaos-amp interpreter.

Covers the three calibration fixes landed 2026-06-01:
  1. Heisenberg signal INVERTED — high disagreement -> SAVE_FOR_LATER,
     never DIG (empirical non-match detector).
  2. non_map_amplifier mode requires structural corroboration — non_map=1
     alone -> SAVE; with anchor (mechanism>0 OR vector>=THRESH OR
     overlap>=1) -> DIG.
  3. random mode DIG threshold tightened — pos>=3 OR (pos>=2 AND
     mechanism>=POSITIVE); stochastic acceptance releases to SAVE
     (breadth), not DIG (depth).

Also covers:
  - All 5 bias modes reachable + deterministic selection
  - Cohort diversity (different URLs land on different modes)
  - Channel variance computation correctness
  - Shim contract preservation (additive verdict fields)
  - interpret() end-to-end via skip_llm_channels (no API calls)
  - Empty fingerprint short-circuit
  - Mode override via bias_mode kwarg

No LLM or embedding calls in this suite. Decision-path tests use
hand-crafted ChannelScores; end-to-end tests use skip_llm_channels=True.
"""
from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from src.wandering.interpreter import (
    BIAS_MODES,
    DISAGREEMENT_HEISENBERG,
    MECHANISM_POSITIVE,
    RANDOM_DIG_NOISE_FLOOR,
    RANDOM_SAVE_NOISE_FLOOR,
    VECTOR_THRESHOLD,
    ChannelScores,
    InterpreterVerdict,
    SessionState,
    _channel_variance,
    _decide,
    _decide_aggressive,
    _decide_conservative,
    _decide_mechanism_only,
    _decide_non_map_amplifier,
    _decide_random,
    _pick_bias_mode,
    interpret,
)
from src.wandering.fingerprint import ContentFingerprint


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def mk_fp(url: str = "", content_hash: str = "", embedding=None, phrases=()):
    """Minimal ContentFingerprint for testing."""
    return ContentFingerprint(
        content_hash=content_hash, url=url, domain="",
        phrases=tuple(phrases),
        phrases_combined=" ".join(phrases),
        embedding=embedding,
    )


def mk_cushion():
    """Minimal CushionGraph for end-to-end tests."""
    from src.wandering.cushion import (
        CushionField, CushionGraph, CushionInput, CushionLayer,
    )
    return CushionGraph(
        actual=CushionLayer(name="actual", nodes=["agent under constraint"], summary="A"),
        essence=CushionLayer(name="essence", nodes=["bounded freedom"], summary="E"),
        mechanism=CushionLayer(name="mechanism", nodes=["structural constraint"], summary="M"),
        raw_input=CushionInput(
            problem=CushionField(name="problem", content="test"),
            context=CushionField(name="context", content=""),
            vision=CushionField(name="vision", content=""),
            hunches=CushionField(name="hunches", content=""),
        ),
    )


# ---------------------------------------------------------------------------
# Channel variance (disagreement signal)
# ---------------------------------------------------------------------------


class TestChannelVariance:
    """The disagreement signal is the empirical non-match detector."""

    def test_true_zero_signal_has_zero_variance(self):
        # All channels explicitly zero — not the dataclass defaults
        # (which set novelty=1.0 and role=0.5 and thus produce
        # high variance even on "empty" content).
        scores = ChannelScores(
            vector=0.0, overlap=0, role=0.0, mechanism=0.0,
            evidence=0.0, novelty=0.0, non_map=0.0,
        )
        dis = _channel_variance(scores)
        assert dis == pytest.approx(0.0, abs=1e-9)

    def test_default_fingerprint_state_has_high_variance(self):
        # Real first-encounter content has novelty=1.0 (unseen) + role=0.5
        # (touched_nodes empty) + zeros elsewhere. This naturally produces
        # high cross-channel variance — which is correct: a fingerprint
        # with NO structural signal beyond "novel" is exactly the noise
        # case Heisenberg-SAVE should catch.
        scores = ChannelScores()  # defaults
        dis = _channel_variance(scores)
        assert dis >= DISAGREEMENT_HEISENBERG, (
            f"default-state disagreement {dis:.3f} should hit Heisenberg "
            f"threshold ({DISAGREEMENT_HEISENBERG}) — empty signal is "
            f"the canonical 'channels disagree' case"
        )

    def test_one_channel_screaming_others_silent_is_high_variance(self):
        scores = ChannelScores(
            vector=0.95, overlap=3, role=0.0, mechanism=0.0,
            evidence=0.0, novelty=0.0, non_map=0.0,
        )
        dis = _channel_variance(scores)
        assert dis >= DISAGREEMENT_HEISENBERG, (
            f"one-channel-screaming disagreement {dis:.3f} "
            f"should be above threshold"
        )

    def test_consistent_moderate_signal_is_low_variance(self):
        scores = ChannelScores(
            vector=0.5, overlap=1, role=0.5, mechanism=0.5,
            evidence=0.5, novelty=0.5, non_map=0.0,
        )
        dis = _channel_variance(scores)
        assert dis < DISAGREEMENT_HEISENBERG


# ---------------------------------------------------------------------------
# Bias mode selection — deterministic + cohort-diverse
# ---------------------------------------------------------------------------


class TestBiasModeSelection:
    """Cohort diversity for free via deterministic hash(url, content_hash)."""

    def test_all_five_modes_reachable_over_population(self):
        seen = Counter()
        for i in range(200):
            fp = mk_fp(url=f"https://example.com/p{i}", content_hash=f"h{i}")
            seen[_pick_bias_mode(fp)] += 1
        for mode in BIAS_MODES:
            assert mode in seen, f"mode {mode!r} never selected in 200 trials"

    def test_deterministic_same_fingerprint_same_mode(self):
        fp = mk_fp(url="https://example.com/x", content_hash="abc123")
        modes = {_pick_bias_mode(fp) for _ in range(10)}
        assert len(modes) == 1, f"non-deterministic mode selection: {modes}"

    def test_cohort_diversity_across_urls(self):
        # 8 different URLs should produce >= 3 distinct modes
        modes = set()
        for url in [f"https://site{i}.com/page" for i in range(8)]:
            modes.add(_pick_bias_mode(mk_fp(url=url, content_hash="shared")))
        assert len(modes) >= 3, f"insufficient cohort diversity: {modes}"

    def test_degenerate_empty_fingerprint_defaults_to_aggressive(self):
        assert _pick_bias_mode(mk_fp(url="", content_hash="")) == "aggressive"


# ---------------------------------------------------------------------------
# CALIBRATION FIX 1 — Heisenberg signal inverted
# ---------------------------------------------------------------------------


class TestHeisenbergInversion:
    """High cross-channel disagreement empirically marks NON-MATCHES.
    Should never DIG in any mode; must SAVE_FOR_LATER.

    Adversarial sweep evidence:
      should-match pairs:     min=0.195, mean=0.211, max=0.217
      should-not-match pairs: min=0.330, mean=0.380, max=0.420
    """

    def _high_disagreement_scores(self) -> ChannelScores:
        # One channel screaming, others silent — high variance ~0.44
        return ChannelScores(
            vector=0.95, overlap=3, role=0.0, mechanism=0.0,
            evidence=0.0, novelty=0.0, non_map=0.0,
        )

    def test_aggressive_high_disagreement_never_digs(self):
        scores = self._high_disagreement_scores()
        dis = _channel_variance(scores)
        assert dis >= DISAGREEMENT_HEISENBERG
        decision, reason = _decide_aggressive(scores, dis)
        assert decision == "save_for_later", (
            f"aggressive Heisenberg case must SAVE, got {decision}: {reason}"
        )

    def test_non_map_amplifier_high_disagreement_never_digs(self):
        # Even with non_map=1.0, high disagreement -> SAVE
        scores = ChannelScores(
            vector=0.95, overlap=0, role=0.0, mechanism=0.0,
            evidence=0.0, novelty=0.0, non_map=1.0,
        )
        dis = _channel_variance(scores)
        assert dis >= DISAGREEMENT_HEISENBERG
        decision, _ = _decide_non_map_amplifier(scores, dis)
        assert decision == "save_for_later"

    def test_random_high_disagreement_never_digs(self):
        scores = self._high_disagreement_scores()
        dis = _channel_variance(scores)
        fp = mk_fp(url="https://x/y", content_hash="z")
        decision, _ = _decide_random(scores, dis, fp)
        assert decision == "save_for_later"

    def test_conservative_high_disagreement_routes_to_save(self):
        # Conservative was already correct; verify it stays correct.
        scores = ChannelScores(
            vector=0.95, overlap=3, role=0.0, mechanism=0.0,
            evidence=0.0, novelty=0.0, non_map=0.0,
        )
        dis = _channel_variance(scores)
        assert dis >= DISAGREEMENT_HEISENBERG
        decision, _ = _decide_conservative(scores, dis)
        assert decision in ("save_for_later", "skip")


# ---------------------------------------------------------------------------
# CALIBRATION FIX 2 — non_map_amplifier requires structural corroboration
# ---------------------------------------------------------------------------


class TestNonMapAmplifierCorroboration:
    """non_map=1.0 alone is not enough — Haiku can hallucinate failure
    modes. DIG requires non_map=1 AND a structural anchor.

    Tests pass disagreement=0.0 explicitly so the corroboration check
    is isolated from the Heisenberg-SAVE early-return path. In
    production both paths exist; we test them separately.
    """

    def test_pure_non_map_no_anchor_saves_does_not_dig(self):
        scores = ChannelScores(
            vector=0.40, overlap=0, role=0.0, mechanism=0.0,
            evidence=0.5, novelty=1.0, non_map=1.0,
        )
        # Force disagreement=0 to isolate the corroboration path
        decision, reason = _decide_non_map_amplifier(scores, disagreement=0.0)
        assert decision == "save_for_later", (
            f"pure non_map (no anchor) must SAVE, got {decision}: {reason}"
        )

    def test_non_map_with_mechanism_anchor_digs(self):
        scores = ChannelScores(
            vector=0.40, overlap=0, role=0.0, mechanism=0.5,
            evidence=0.5, novelty=1.0, non_map=1.0,
        )
        decision, _ = _decide_non_map_amplifier(scores, disagreement=0.0)
        assert decision == "dig"

    def test_non_map_with_vector_anchor_digs(self):
        scores = ChannelScores(
            vector=0.65, overlap=0, role=0.0, mechanism=0.0,
            evidence=0.5, novelty=1.0, non_map=1.0,
        )
        decision, _ = _decide_non_map_amplifier(scores, disagreement=0.0)
        assert decision == "dig"

    def test_non_map_with_overlap_anchor_digs(self):
        scores = ChannelScores(
            vector=0.50, overlap=2, role=0.5, mechanism=0.0,
            evidence=0.5, novelty=1.0, non_map=1.0,
        )
        decision, _ = _decide_non_map_amplifier(scores, disagreement=0.0)
        assert decision == "dig"

    def test_non_map_with_heisenberg_disagreement_saves_regardless(self):
        # Even with non_map=1 + anchor, high disagreement -> SAVE.
        scores = ChannelScores(
            vector=0.65, overlap=0, role=0.0, mechanism=0.5,
            evidence=0.0, novelty=1.0, non_map=1.0,
        )
        decision, _ = _decide_non_map_amplifier(scores, disagreement=0.45)
        assert decision == "save_for_later"


# ---------------------------------------------------------------------------
# CALIBRATION FIX 3 — random mode threshold tightened
# ---------------------------------------------------------------------------


class TestRandomModeThreshold:
    """Random mode used to DIG on pos>=2 alone — 57% precision.
    Now requires pos>=3 OR (pos>=2 AND mechanism>=POSITIVE).
    Stochastic acceptance releases to SAVE, not DIG."""

    def test_pos_2_alone_no_mechanism_does_not_dig(self):
        # Two positives but no mechanism — used to DIG, now must SAVE.
        # Force disagreement=0 to isolate the threshold path.
        scores = ChannelScores(
            vector=0.65, overlap=0, role=0.0, mechanism=0.0,
            evidence=0.0, novelty=1.0, non_map=0.0,
        )
        fp = mk_fp(url="https://test/a", content_hash="aaa")
        decision, _ = _decide_random(scores, disagreement=0.0, fingerprint=fp)
        assert decision != "dig", (
            f"random must NOT DIG on pos=2 without mechanism, got {decision}"
        )

    def test_pos_2_with_mechanism_digs(self):
        scores = ChannelScores(
            vector=0.65, overlap=0, role=0.0, mechanism=0.55,
            evidence=0.0, novelty=1.0, non_map=0.0,
        )
        fp = mk_fp(url="https://test/b", content_hash="bbb")
        decision, _ = _decide_random(scores, disagreement=0.0, fingerprint=fp)
        assert decision == "dig"

    def test_pos_3_alone_digs(self):
        scores = ChannelScores(
            vector=0.65, overlap=1, role=0.5, mechanism=0.0,
            evidence=0.5, novelty=1.0, non_map=0.0,
        )
        fp = mk_fp(url="https://test/c", content_hash="ccc")
        decision, _ = _decide_random(scores, disagreement=0.0, fingerprint=fp)
        assert decision == "dig"

    def test_pure_non_map_in_random_does_not_dig(self):
        # non_map=1.0 as the ONLY positive — used to DIG random
        # (pos>=2 or non_map>=1), now must SAVE.
        scores = ChannelScores(
            vector=0.40, overlap=0, role=0.0, mechanism=0.0,
            evidence=0.4, novelty=0.0, non_map=1.0,
        )
        # Sanity-check the fixture has pos=1 (non_map only)
        assert scores.positive_count() == 1
        fp = mk_fp(url="https://test/d", content_hash="ddd")
        decision, _ = _decide_random(scores, disagreement=0.0, fingerprint=fp)
        assert decision != "dig", (
            f"random must NOT DIG on pure non_map (pos=1), got {decision}"
        )

    def test_random_heisenberg_disagreement_saves_regardless(self):
        # Even with strong signal, high disagreement -> SAVE in random.
        scores = ChannelScores(
            vector=0.95, overlap=3, role=0.5, mechanism=0.6,
            evidence=0.5, novelty=1.0, non_map=0.0,
        )
        fp = mk_fp(url="https://test/h", content_hash="heisenberg")
        decision, _ = _decide_random(scores, disagreement=0.45, fingerprint=fp)
        assert decision == "save_for_later"

    def test_random_is_reproducible(self):
        scores = ChannelScores(
            vector=0.45, overlap=0, role=0.3, mechanism=0.3,
            evidence=0.4, novelty=1.0, non_map=0.0,
        )
        fp = mk_fp(url="https://test/repro", content_hash="repro_hash")
        d1, _ = _decide_random(scores, disagreement=0.0, fingerprint=fp)
        d2, _ = _decide_random(scores, disagreement=0.0, fingerprint=fp)
        assert d1 == d2


# ---------------------------------------------------------------------------
# Mode-specific path verification (mechanism_only + conservative)
# ---------------------------------------------------------------------------


class TestMechanismOnlyPath:
    """The pure-dynamics path that the structural_foothold gate used to
    block. mechanism+non_map alone must reach DIG."""

    def test_pure_mechanism_plus_non_map_digs(self):
        scores = ChannelScores(
            vector=0.40, overlap=0, role=0.0, mechanism=0.7,
            evidence=0.5, novelty=1.0, non_map=1.0,
        )
        dis = _channel_variance(scores)
        decision, _ = _decide_mechanism_only(scores, dis)
        assert decision == "dig"

    def test_weak_structural_signal_saves(self):
        scores = ChannelScores(
            vector=0.40, overlap=0, role=0.0, mechanism=0.5,
            evidence=0.5, novelty=1.0, non_map=0.0,
        )
        dis = _channel_variance(scores)
        decision, _ = _decide_mechanism_only(scores, dis)
        assert decision == "save_for_later"


class TestConservativePath:
    """Conservative is the precision-tight backbone of the cohort.
    It already scored 100% precision in the adversarial sweep."""

    def test_3_positives_plus_mechanism_digs(self):
        scores = ChannelScores(
            vector=0.65, overlap=1, role=0.5, mechanism=0.5,
            evidence=0.0, novelty=0.0, non_map=0.0,
        )
        dis = _channel_variance(scores)
        if dis >= DISAGREEMENT_HEISENBERG:
            pytest.skip("fixture has Heisenberg-level disagreement")
        decision, _ = _decide_conservative(scores, dis)
        assert decision == "dig"

    def test_2_positives_only_saves(self):
        scores = ChannelScores(
            vector=0.65, overlap=1, role=0.0, mechanism=0.0,
            evidence=0.0, novelty=0.0, non_map=0.0,
        )
        dis = _channel_variance(scores)
        if dis >= DISAGREEMENT_HEISENBERG:
            pytest.skip("fixture has Heisenberg-level disagreement")
        decision, _ = _decide_conservative(scores, dis)
        assert decision == "save_for_later"


# ---------------------------------------------------------------------------
# Dispatcher + interpret() end-to-end (no LLM)
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_unknown_mode_falls_back_to_aggressive(self):
        scores = ChannelScores()
        fp = mk_fp(url="x", content_hash="y")
        dis = _channel_variance(scores)
        d_unknown, _ = _decide(scores, dis, "made_up_mode", fp)
        d_aggressive, _ = _decide_aggressive(scores, dis)
        assert d_unknown == d_aggressive


class TestInterpretEndToEnd:
    """interpret() with skip_llm_channels=True — no API calls."""

    def test_empty_fingerprint_skips(self):
        async def run():
            cushion = mk_cushion()
            fp = mk_fp()  # no embedding, no phrases
            return await interpret(fp, cushion, client=None,
                                   session_state=SessionState(),
                                   skip_llm_channels=True)

        verdict = asyncio.run(run())
        assert verdict.decision == "skip"
        assert verdict.bias_mode in BIAS_MODES
        assert len(verdict.channel_score_profile) == 7

    def test_populated_verdict_has_all_additive_fields(self):
        async def run():
            cushion = mk_cushion()
            fp = mk_fp(url="https://e2e/test", content_hash="e2e",
                       phrases=("bounded freedom", "structural constraint"))
            return await interpret(fp, cushion, client=None,
                                   session_state=SessionState(),
                                   skip_llm_channels=True)

        verdict = asyncio.run(run())
        assert verdict.bias_mode in BIAS_MODES
        assert 0.0 <= verdict.disagreement <= 1.0
        assert len(verdict.channel_score_profile) == 7
        assert verdict.decision in ("dig", "save_for_later", "skip")

    def test_bias_mode_override_respected(self):
        async def run():
            cushion = mk_cushion()
            fp = mk_fp(url="https://test/o", content_hash="override",
                       phrases=("anything",))
            return await interpret(fp, cushion, client=None,
                                   session_state=SessionState(),
                                   skip_llm_channels=True,
                                   bias_mode="conservative")

        verdict = asyncio.run(run())
        assert verdict.bias_mode == "conservative"


# ---------------------------------------------------------------------------
# Shim contract — additive fields, agent.py still works
# ---------------------------------------------------------------------------


class TestShimContract:
    """agent.py:_verdict_to_match_result reads only decision /
    matched_nodes / reason. Verifies that the additive fields
    (bias_mode, disagreement, channel_score_profile) don't break it."""

    def test_legacy_shape_verdict_still_constructible(self):
        """A verdict built with only the pre-rewrite required fields
        (no bias_mode, disagreement, channel_score_profile) must still
        be a valid InterpreterVerdict."""
        scores = ChannelScores(vector=0.7, overlap=2, mechanism=0.6)
        verdict = InterpreterVerdict(
            decision="dig",
            scores=scores,
            matched_nodes=["essence node"],
            reason="legacy shape",
        )
        # New fields default to safe values
        assert verdict.bias_mode == ""
        assert verdict.disagreement == 0.0
        assert verdict.channel_score_profile == []
        # Required fields readable
        assert verdict.decision == "dig"
        assert verdict.matched_nodes == ["essence node"]

    def test_shim_consumes_only_required_fields(self):
        from src.wandering import agent as _agent
        scores = ChannelScores(vector=0.7, overlap=2)
        verdict = InterpreterVerdict(
            decision="dig",
            scores=scores,
            matched_nodes=["bounded freedom"],
            reason="[mechanism_only] shim test",
            bias_mode="mechanism_only",
            disagreement=0.18,
            channel_score_profile=scores.profile_vector(),
        )
        cushion = mk_cushion()
        match_result = _agent._verdict_to_match_result(verdict, cushion)
        # Shim produces a valid MatchResult — uses only decision /
        # matched_nodes / reason
        assert match_result.dig_iterations >= 3
        assert "interpreter:dig" in match_result.raw_response
        assert match_result.total_matched_nodes >= 1
