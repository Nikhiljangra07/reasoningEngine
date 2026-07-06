"""
Tests for the quality ranker (src/wandering/quality_ranker.py) — the final
alignment pass. Offline: a stub client returns ranking JSON.

Pins the load-bearing guarantees: every blend is kept (never deleted),
higher advancement ranks higher, a new-gap opener is protected, and the pass
fails soft (keeps everything in input order) on an LLM error.
"""

from __future__ import annotations

import json

import pytest

from src.llm.client import LLMResponse
from src.wandering.blender import Blend
from src.wandering.quality_ranker import QualityRanking, rank_blends


class StubClient:
    def __init__(self, content: str, success: bool = True):
        self._content = content
        self._success = success

    async def call(self, **kw) -> LLMResponse:
        return LLMResponse(content=self._content, input_tokens=100, output_tokens=60,
                           latency_ms=1.0, success=self._success, model=kw.get("model", ""),
                           error=None if self._success else "stub_fail")


def _blend(bid, thesis="T"):
    return Blend(blend_id=bid, source_card_ids=["r1", "r2"], thesis=thesis)


class _Spot:
    def __init__(self, layer, text):
        self.layer = layer
        self.blind_spot = text


@pytest.mark.asyncio
async def test_empty_input_no_call():
    r = await rank_blends(cushion="c", blends=[], blind_spots=[], client=StubClient("{}"))
    assert r.ranked == []


@pytest.mark.asyncio
async def test_ranks_by_advancement_and_keeps_all():
    content = json.dumps({"rankings": [
        {"blend_id": "blend-01", "advancement": 0.2, "blind_spots_addressed": [], "opens_new_gap": ""},
        {"blend_id": "blend-02", "advancement": 0.9, "blind_spots_addressed": ["G1"], "opens_new_gap": ""},
    ]})
    blends = [_blend("blend-01"), _blend("blend-02")]
    r = await rank_blends(cushion="advance my concept", blends=blends,
                          blind_spots=[_Spot("cushion", "no advancement metric")],
                          novelty_by_id={"blend-01": "novel", "blend-02": "adjacent"},
                          client=StubClient(content))
    # all kept
    assert {x.blend_id for x in r.ranked} == {"blend-01", "blend-02"}
    # higher advancement (02) ranks first
    assert r.ranked[0].blend_id == "blend-02" and r.ranked[0].rank == 1
    assert r.ranked[1].blend_id == "blend-01"
    # novelty bin carried through
    assert r.ranked[0].novelty_bin == "adjacent"
    json.dumps(r.to_dict())  # serializable


@pytest.mark.asyncio
async def test_new_gap_is_surfaced_not_scored():
    # equal advancement -> TIED rank. opens_new_gap is SURFACED on the record,
    # but does NOT change the order (it is no longer a scoring bonus).
    content = json.dumps({"rankings": [
        {"blend_id": "blend-01", "advancement": 0.5, "opens_new_gap": ""},
        {"blend_id": "blend-02", "advancement": 0.5, "opens_new_gap": "a slot nobody named"},
    ]})
    r = await rank_blends(cushion="c", blends=[_blend("blend-01"), _blend("blend-02")],
                          blind_spots=[], client=StubClient(content))
    # equal advancement within TIE_EPSILON -> both share rank 1, both flagged tied
    assert {x.rank for x in r.ranked} == {1}
    assert all(x.tied for x in r.ranked)
    # the new gap is still surfaced for the human to read
    surfaced = {x.blend_id: x.opens_new_gap for x in r.ranked}
    assert surfaced["blend-02"] == "a slot nobody named"


@pytest.mark.asyncio
async def test_gap_count_does_not_change_rank():
    # blend-01 resolves THREE gaps but advances less; blend-02 resolves none but
    # advances more. Advancement alone orders -> blend-02 must rank first.
    content = json.dumps({"rankings": [
        {"blend_id": "blend-01", "advancement": 0.4, "blind_spots_addressed": ["G1", "G2", "G3"]},
        {"blend_id": "blend-02", "advancement": 0.7, "blind_spots_addressed": []},
    ]})
    r = await rank_blends(cushion="c", blends=[_blend("blend-01"), _blend("blend-02")],
                          blind_spots=[], client=StubClient(content))
    assert r.ranked[0].blend_id == "blend-02" and r.ranked[0].rank == 1
    assert r.ranked[1].blend_id == "blend-01"
    # the gaps are still surfaced on the lower-ranked blend
    assert r.ranked[1].blind_spots_addressed == ["G1", "G2", "G3"]


@pytest.mark.asyncio
async def test_unassessed_blend_kept_not_deleted():
    # LLM only assessed one of two -> the other is KEPT, ranked last
    content = json.dumps({"rankings": [{"blend_id": "blend-01", "advancement": 0.8}]})
    r = await rank_blends(cushion="c", blends=[_blend("blend-01"), _blend("blend-02")],
                          blind_spots=[], client=StubClient(content))
    ids = {x.blend_id for x in r.ranked}
    assert ids == {"blend-01", "blend-02"}              # nothing deleted
    assert r.ranked[-1].blend_id == "blend-02"          # unassessed ranked last
    assert any(n["reason"] == "blend_unassessed_kept_last" for n in r.parser_notes)


@pytest.mark.asyncio
async def test_fail_soft_keeps_everything():
    r = await rank_blends(cushion="c", blends=[_blend("blend-01"), _blend("blend-02")],
                          blind_spots=[], client=StubClient("", success=False))
    assert r.ok is False
    assert [x.blend_id for x in r.ranked] == ["blend-01", "blend-02"]  # input order, all kept
