"""
Tests for the sorter verification engine (src/wandering/sorter_verify.py)
and its consumption by master_sort.

Fully offline: a stub LLM client returns crafted JSON, an injected
search_fn returns crafted hits. No network, no API keys, no credits.

Covers:
  - happy path: extracted queries → searches → populated EvidenceLedger
  - heuristic fallback when the extractor returns no queries
  - every card is searched even when the extraction call fails outright
  - master_sort's verified doctrine vs the memory-only path
  - the dossier serializes the evidence trail
"""

from __future__ import annotations

import json

import pytest

from src.bridge.web_search import SearchHit, SearchResult
from src.llm.client import LLMResponse
from src.wandering.articulate import ArticulatedCard
from src.wandering.report import Confidence
from src.wandering import master_sorter
from src.wandering.master_sorter import _build_sort_payload, _DOCTRINE_VERIFIED, _DOCTRINE_PREAMBLE
from src.wandering.sorter_verify import (
    CardEvidence,
    EvidenceLedger,
    gather_evidence,
)


# ─── fakes ─────────────────────────────────────────────────────────────


class StubClient:
    """Minimal LLMClient stand-in: every .call returns the same content."""

    def __init__(self, content: str, success: bool = True):
        self._content = content
        self._success = success
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            content=self._content,
            input_tokens=120,
            output_tokens=40,
            latency_ms=1.0,
            success=self._success,
            model=kwargs.get("model", ""),
            error=None if self._success else "stub_failure",
        )


class RaisingClient:
    async def call(self, **kwargs) -> LLMResponse:
        raise RuntimeError("boom")


def _hit_search(_query: str):
    async def _inner(query: str) -> SearchResult:
        return SearchResult(
            query=query,
            hits=[SearchHit(title="A real paper", snippet="maps X to Y", url="http://arxiv.org/abs/1")],
            provider="fake",
            latency_ms=1,
        )
    return _inner


async def hit_search(query: str) -> SearchResult:
    return SearchResult(
        query=query,
        hits=[SearchHit(title="A real paper", snippet="maps X to Y", url="http://arxiv.org/abs/1")],
        provider="fake",
        latency_ms=1,
    )


async def empty_search(query: str) -> SearchResult:
    return SearchResult(query=query, hits=[], provider="fake", error="no_hits", latency_ms=1)


def make_card(rid: str) -> ArticulatedCard:
    return ArticulatedCard(
        report_id=rid,
        spark="Markov chains appear in dialogue modeling",
        source_shape="probability theory",
        bridge="conversational trust decays like a Markov chain state",
        use="model when trust collapses",
        limit="the memoryless assumption is wrong for grudges",
        confidence=Confidence.MEDIUM,
    )


# ─── tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_populates_ledger():
    cards = [make_card("r1"), make_card("r2")]
    extraction = json.dumps({"queries": {"r1": ["q1a", "q1b"], "r2": ["q2a"]}})
    client = StubClient(extraction)

    ledger = await gather_evidence(
        cushion=None, cards=cards, client=client,
        query_model="anthropic/claude-sonnet-4-6", search_fn=hit_search,
    )

    assert isinstance(ledger, EvidenceLedger)
    assert ledger.extraction_ok is True
    assert set(ledger.per_card) == {"r1", "r2"}
    assert ledger.per_card["r1"].queries == ["q1a", "q1b"]
    assert ledger.per_card["r2"].queries == ["q2a"]
    # 2 + 1 = 3 queries, each returns 1 hit
    assert ledger.total_queries == 3
    assert ledger.total_hits == 3
    assert ledger.per_card["r1"].found_anything is True
    assert ledger.per_card["r1"].searched is True
    # round-trips to a plain dict
    d = ledger.to_dict()
    assert d["total_hits"] == 3
    assert d["per_card"]["r1"]["hits"][0]["url"] == "http://arxiv.org/abs/1"


@pytest.mark.asyncio
async def test_heuristic_fallback_when_extractor_returns_no_queries():
    cards = [make_card("r1")]
    # valid JSON but no queries for the card → heuristic fallback
    client = StubClient(json.dumps({"queries": {}}))

    ledger = await gather_evidence(
        cushion=None, cards=cards, client=client, search_fn=hit_search,
    )

    ev = ledger.per_card["r1"]
    assert ev.searched is True
    assert len(ev.queries) == 1            # one heuristic query
    assert "heuristic fallback" in ev.note
    # heuristic query built from source_shape + spark
    assert "probability theory" in ev.queries[0]


@pytest.mark.asyncio
async def test_every_card_searched_even_when_extraction_crashes():
    cards = [make_card("r1"), make_card("r2")]
    ledger = await gather_evidence(
        cushion=None, cards=cards, client=RaisingClient(), search_fn=hit_search,
    )
    # extraction blew up → all cards fall back to heuristic, all searched
    assert ledger.extraction_ok is False
    assert all(ledger.per_card[r].searched for r in ("r1", "r2"))
    assert ledger.total_queries == 2


@pytest.mark.asyncio
async def test_empty_search_records_searched_but_no_hits():
    cards = [make_card("r1")]
    client = StubClient(json.dumps({"queries": {"r1": ["q1"]}}))
    ledger = await gather_evidence(
        cushion=None, cards=cards, client=client, search_fn=empty_search,
    )
    ev = ledger.per_card["r1"]
    assert ev.searched is True
    assert ev.found_anything is False
    assert ledger.total_hits == 0
    assert len(ledger.search_errors) == 1   # error recorded for the human


@pytest.mark.asyncio
async def test_empty_input_no_calls():
    client = StubClient("{}")
    ledger = await gather_evidence(cushion=None, cards=[], client=client, search_fn=hit_search)
    assert ledger.total_queries == 0
    assert ledger.per_card == {}
    assert client.calls == []   # no LLM call fired on empty input


def test_payload_injects_evidence_and_changes_ref_hint():
    cards = [make_card("r1")]
    ledger = EvidenceLedger()
    ev = CardEvidence(report_id="r1", queries=["q1"], searched=True)
    from src.wandering.sorter_verify import EvidenceHit
    ev.hits = [EvidenceHit(query="q1", title="Paper", url="http://x", snippet="s", provider="fake")]
    ledger.per_card["r1"] = ev

    verified = _build_sort_payload(None, cards, None, web_evidence=ledger)
    assert '"evidence"' in verified
    assert "COPIED FROM" in verified
    assert "http://x" in verified

    plain = _build_sort_payload(None, cards, None, web_evidence=None)
    assert '"evidence"' not in plain
    assert "COPIED FROM" not in plain


def test_verified_doctrine_distinct_from_memory_doctrine():
    # The two doctrines must actually differ — the verified one names the
    # evidence; the memory one tells the model to match training memory.
    assert "EVIDENCE" in _DOCTRINE_VERIFIED
    assert "search engine" in _DOCTRINE_VERIFIED
    assert _DOCTRINE_VERIFIED != _DOCTRINE_PREAMBLE
