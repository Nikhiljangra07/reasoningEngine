"""
Tests for the blender (src/wandering/blender.py).

Fully offline: a stub LLM client returns crafted JSON. No network, no
credits. Covers the happy path, the blend-not-merge flag (empty emergent
structure), unknown-card-id dropping, the insufficient-input guard, and
payload construction (bins + cushion reach the model).
"""

from __future__ import annotations

import json

import pytest

from src.llm.client import LLMResponse
from src.wandering.articulate import ArticulatedCard
from src.wandering.report import Confidence
from src.wandering.blender import (
    Blend,
    BlendBatch,
    blend_cards,
    _build_blend_payload,
)


class StubClient:
    def __init__(self, content: str, success: bool = True):
        self._content = content
        self._success = success
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            content=self._content, input_tokens=300, output_tokens=200,
            latency_ms=1.0, success=self._success,
            model=kwargs.get("model", ""),
            error=None if self._success else "stub_failure",
        )


class _Problem:
    def __init__(self, content): self.content = content
class _RawInput:
    def __init__(self, content): self.problem = _Problem(content)
class _CushionShim:
    def __init__(self, content): self.raw_input = _RawInput(content)


def make_card(rid: str, spark: str = "seed") -> ArticulatedCard:
    return ArticulatedCard(
        report_id=rid, spark=spark, source_shape="domain X",
        bridge=f"bridge for {rid}", use="do something", limit="breaks somewhere",
        confidence=Confidence.MEDIUM,
    )


def _blend_json(blends):
    return json.dumps({"blends": blends})


@pytest.mark.asyncio
async def test_blend_happy_path():
    cards = [make_card("r1"), make_card("r2"), make_card("r3")]
    content = _blend_json([{
        "source_card_ids": ["r1", "r2"],
        "why_these_cards": "they sit in tension",
        "spark": "a stray thought",
        "motive": "advance the cushion",
        "tension": "r1 pulls structural, r2 pulls temporal",
        "discovery_path": "r1 asserts X; r2 asserts Y; reconciling them needs Z; Z is the thesis",
        "thesis": "a genuinely new claim C",
        "mechanism": "C runs because of the joint",
        "emergent_structure": "C predicts decay that neither r1 nor r2 contains",
        "advances_cushion": "gives the soft-anchor a measurable signal",
        "confidence": 0.8,
    }])
    client = StubClient(content)
    batch = await blend_cards(
        cushion=None, cards=cards, bins_by_id={"r1": "known", "r2": "unplaced"},
        client=client, model="anthropic/claude-opus-4-8",
    )
    assert isinstance(batch, BlendBatch)
    assert len(batch.blends) == 1
    b = batch.blends[0]
    assert b.source_card_ids == ["r1", "r2"]
    assert len(b.source_cards) == 2                  # provenance snapshots attached
    assert b.emergent_structure.startswith("C predicts")
    assert b.selection.tension.startswith("r1 pulls")
    assert b.selection.discovery_path.startswith("r1 asserts X")   # reverse-engineerable genealogy
    assert b.confidence == 0.8
    # round-trips
    d = batch.to_dict()
    assert d["blends"][0]["thesis"] == "a genuinely new claim C"


@pytest.mark.asyncio
async def test_empty_emergent_structure_flagged_as_possible_merge():
    cards = [make_card("r1"), make_card("r2")]
    content = _blend_json([{
        "source_card_ids": ["r1", "r2"],
        "thesis": "looks like a list of both",
        "emergent_structure": "",   # the merge tell
        "confidence": 0.5,
    }])
    batch = await blend_cards(cushion=None, cards=cards, client=StubClient(content))
    # kept (human judges) but flagged
    assert len(batch.blends) == 1
    assert any(n["reason"] == "empty_emergent_structure_possible_merge" for n in batch.parser_notes)


@pytest.mark.asyncio
async def test_unknown_card_ids_dropped_and_blend_pruned():
    cards = [make_card("r1"), make_card("r2")]
    content = _blend_json([
        {"source_card_ids": ["r1", "ghost"], "emergent_structure": "x"},   # 1 valid -> dropped
        {"source_card_ids": ["r1", "r2"], "emergent_structure": "y"},      # 2 valid -> kept
    ])
    batch = await blend_cards(cushion=None, cards=cards, client=StubClient(content))
    assert len(batch.blends) == 1
    assert batch.blends[0].source_card_ids == ["r1", "r2"]
    assert any(n["reason"] == "unknown_card_id" for n in batch.parser_notes)
    assert any(n["reason"] == "blend_dropped_too_few_valid_cards" for n in batch.parser_notes)


@pytest.mark.asyncio
async def test_insufficient_cards_no_llm_call():
    client = StubClient("{}")
    batch = await blend_cards(cushion=None, cards=[make_card("r1")], client=client)
    assert batch.blends == []
    assert client.calls == []     # no spend on < 2 cards


def test_payload_includes_bins_and_cushion():
    cards = [make_card("r1"), make_card("r2")]
    payload = _build_blend_payload(
        _CushionShim("MY REAL PROBLEM"), cards, {"r1": "known", "r2": "unplaced"},
    )
    assert "MY REAL PROBLEM" in payload
    assert '"bin": "known"' in payload
    assert '"bin": "unplaced"' in payload
    assert "emergent_structure" in payload   # schema demands it
