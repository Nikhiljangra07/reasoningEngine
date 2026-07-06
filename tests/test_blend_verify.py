"""
Tests for blend verification (src/wandering/blend_verify.py) — the sorter's
second seat with the 4th bin (adjacent).

Fully offline: stub client returns crafted JSON for BOTH the query-extraction
and the verdict call (same content works for both since each parses its own
keys); fake search returns crafted hits.
"""

from __future__ import annotations

import json

import pytest

from src.bridge.web_search import SearchHit, SearchResult
from src.llm.client import LLMResponse
from src.wandering.blender import Blend
from src.wandering.blend_verify import (
    BlendVerificationReport,
    gather_blend_evidence,
    verify_blends,
)


class ScriptedClient:
    """Returns queued contents in order, one per .call()."""
    def __init__(self, contents: list[str]):
        self._contents = contents
        self.i = 0
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        content = self._contents[min(self.i, len(self._contents) - 1)]
        self.i += 1
        return LLMResponse(content=content, input_tokens=200, output_tokens=80,
                           latency_ms=1.0, success=True, model=kwargs.get("model", ""))


async def hit_search(query: str) -> SearchResult:
    return SearchResult(query=query, provider="fake", latency_ms=1,
                        hits=[SearchHit(title="Cousin method", snippet="similar idea", url="http://paper/1")])


async def empty_search(query: str) -> SearchResult:
    return SearchResult(query=query, provider="fake", latency_ms=1, hits=[], error="no_hits")


def make_blend(bid: str) -> Blend:
    return Blend(blend_id=bid, source_card_ids=["r1", "r2"],
                 thesis=f"thesis {bid}", mechanism="it runs like so",
                 emergent_structure="the new part")


@pytest.mark.asyncio
async def test_gather_blend_evidence_searches_every_blend():
    blends = [make_blend("blend-01"), make_blend("blend-02")]
    queries = json.dumps({"queries": {"blend-01": ["q1", "q2"], "blend-02": ["q3"]}})
    client = ScriptedClient([queries])
    ledger = await gather_blend_evidence(blends=blends, client=client, search_fn=hit_search)
    assert set(ledger.per_card) == {"blend-01", "blend-02"}
    assert ledger.total_queries == 3
    assert ledger.total_hits == 3
    assert ledger.per_card["blend-01"].searched is True


@pytest.mark.asyncio
async def test_verify_bins_known_adjacent_novel():
    blends = [make_blend("blend-01"), make_blend("blend-02"), make_blend("blend-03")]
    queries = json.dumps({"queries": {b.blend_id: ["q"] for b in blends}})
    verdict = json.dumps({"verdicts": [
        {"blend_id": "blend-01", "bin": "known", "references": [{"title": "T", "url": "http://paper/1"}],
         "resemblance": "duplicates it", "reasoning": "exact match"},
        {"blend_id": "blend-02", "bin": "adjacent", "references": [{"title": "T", "url": "http://paper/2"}],
         "resemblance": "a cousin exists", "still_new": "the twist", "reasoning": "partial"},
        {"blend_id": "blend-03", "bin": "novel", "references": [],
         "reasoning": "nothing resembled it", "confidence": 0.7},
    ]})
    client = ScriptedClient([queries, verdict])
    report = await verify_blends(cushion=None, blends=blends, client=client, search_fn=hit_search)
    assert isinstance(report, BlendVerificationReport)
    assert [b.blend_id for b in report.known] == ["blend-01"]
    assert [b.blend_id for b in report.adjacent] == ["blend-02"]
    assert [b.blend_id for b in report.novel] == ["blend-03"]
    assert report.adjacent[0].still_new == "the twist"


@pytest.mark.asyncio
async def test_known_without_reference_demoted_to_novel():
    blends = [make_blend("blend-01")]
    queries = json.dumps({"queries": {"blend-01": ["q"]}})
    # claims KNOWN but cites no url → must be demoted (can't claim prior art w/o evidence)
    verdict = json.dumps({"verdicts": [
        {"blend_id": "blend-01", "bin": "known", "references": [], "reasoning": "i just know it"},
    ]})
    client = ScriptedClient([queries, verdict])
    report = await verify_blends(cushion=None, blends=blends, client=client, search_fn=hit_search)
    assert report.known == []
    assert [b.blend_id for b in report.novel] == ["blend-01"]
    assert any(n["reason"] == "claim_without_reference_demoted_to_novel" for n in report.parser_notes)


@pytest.mark.asyncio
async def test_empty_input_no_calls():
    client = ScriptedClient(["{}"])
    report = await verify_blends(cushion=None, blends=[], client=client, search_fn=hit_search)
    assert report.input_blend_count == 0
    assert client.calls == []


def test_doctrine_is_balanced_not_certainty_biased():
    """Guard against the certainty bias the all-adjacent run exposed: the
    doctrine must NOT default to adjacent or demote on mere component
    resemblance, and MUST carry the same-MOVE test that lets real novelty
    surface. Tension/friction is the signal, not noise to be smoothed."""
    from src.wandering.blend_verify import _BLEND_VERIFY_DOCTRINE as D
    # the old thumb-on-the-scale lines are gone
    assert "Default to ADJACENT over NOVEL" not in D
    assert "ANY real structural resemblance demotes" not in D
    # the rebalanced spine is present
    assert "same-MOVE test" in D or "same-MOVE" in D
    assert "TOO SMOOTH" in D and "TOO CHAOTIC" in D
    assert "Novelty lives in the TRANSFER, not the components" in D
    assert "Do NOT default to any bin" in D


@pytest.mark.asyncio
async def test_evidence_trail_persisted_on_report():
    blends = [make_blend("blend-01")]
    queries = json.dumps({"queries": {"blend-01": ["q"]}})
    verdict = json.dumps({"verdicts": [{"blend_id": "blend-01", "bin": "novel", "reasoning": "x"}]})
    client = ScriptedClient([queries, verdict])
    report = await verify_blends(cushion=None, blends=blends, client=client, search_fn=empty_search)
    # the full evidence ledger rides on the report for the human to audit
    assert report.evidence is not None
    d = report.to_dict()
    assert d["evidence"]["total_queries"] == 1
