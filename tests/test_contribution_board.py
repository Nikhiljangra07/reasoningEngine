"""
Tests for the wander contribution board (WANDER_CONTRIBUTION_BOARD).

The board injects an additive, positive-sum "add the deeper/missing layer"
block into the dig prompt, built from peers' already-posted notices. These
tests pin: the flag defaults OFF, the builder's empty/blank handling, the
additive (non-competitive) framing, and — the integration that matters —
that the block actually reaches the dig user-message and that the legacy
prompt is unchanged when the board is off.
"""

from __future__ import annotations

import pytest

from src.llm.client import LLMResponse
from src.wandering.agent import (
    _build_contribution_block,
    _run_dig_iteration,
    _use_contribution_board,
)


class _Notice:
    def __init__(self, summary, principle=""):
        self.summary = summary
        self.principle = principle


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("WANDER_CONTRIBUTION_BOARD", raising=False)
    assert _use_contribution_board() is False
    monkeypatch.setenv("WANDER_CONTRIBUTION_BOARD", "1")
    assert _use_contribution_board() is True


def test_builder_empty_and_blank_summaries():
    assert _build_contribution_block([]) == ""              # no peers -> legacy prompt
    assert _build_contribution_block([_Notice("   ", "p")]) == ""  # blank summary skipped


def test_builder_renders_additive_not_competitive_frame():
    b = _build_contribution_block([_Notice("loops drive isolation->distrust", "leverage points")])
    assert "CONTRIBUTION BOARD" in b
    assert "loops drive isolation->distrust" in b and "leverage points" in b
    # additive + anti-inflation, NOT competitive
    assert "Contribute what is MISSING" in b
    assert "NOT a competition" in b
    assert "worthless" not in b.lower() and "beat" not in b.lower()


class _StubClient:
    def __init__(self):
        self.last_user = None

    async def call(self, **kw):
        self.last_user = kw.get("user_message", "")
        return LLMResponse(
            content='{"exploration_summary":"s","advancement":"a","what_does_not_map":"m"}',
            input_tokens=1, output_tokens=1, latency_ms=1.0, success=True, model="x",
        )


class _Cushion:
    def to_anchor_prompt(self):
        return "ANCHOR TEXT"


class _Fetched:
    title = "T"; url = "u"; domain_hint = "physics"; body = "B"


class _Match:
    total_matched_nodes = 2
    matches: dict = {}
    dig_iterations = 1


@pytest.mark.asyncio
async def test_block_reaches_dig_prompt_and_legacy_untouched():
    c = _StubClient()
    # with a block -> it appears in the dig user-message
    await _run_dig_iteration(_Cushion(), _Fetched(), _Match(), 0, c,
                             contribution_block="THE BOARD BLOCK")
    assert "THE BOARD BLOCK" in c.last_user
    assert "ANCHOR TEXT" in c.last_user

    # default (no block) -> legacy prompt, board text absent
    await _run_dig_iteration(_Cushion(), _Fetched(), _Match(), 0, c)
    assert "THE BOARD BLOCK" not in c.last_user
    assert "ANCHOR TEXT" in c.last_user
