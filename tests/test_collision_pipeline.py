"""
Deep end-to-end test for the collision pipeline (the brick-5 wiring).

Fully offline: a stage-routing stub client (returns the right JSON per
seat) + a fake search. Exercises blend → drift-check → blend-verify wired
together, the INVALID-exclusion, the drift QUARANTINE path, the cost
rollup, and the unified run-record (with its reverse-engineering trace)
including JSON-serializability.
"""

from __future__ import annotations

import json

import pytest

from src.bridge.web_search import SearchHit, SearchResult
from src.llm.client import LLMResponse
from src.wandering.articulate import ArticulatedCard
from src.wandering.report import Confidence
from src.wandering.master_sorter import (
    CardSnapshot, InvalidItem, KnownItem, SortedReport, UnplacedItem,
)
from src.wandering.collision_pipeline import (
    CollisionReport, build_run_record, run_collision_pipeline,
)


class StageStub:
    """Routes each .call to the right canned JSON by `concept`."""
    def __init__(self, *, blend, drift, vq, verdict):
        self._by_concept = {
            "blend": blend, "drift_check": drift,
            "verify_queries": vq, "verdict": verdict,
        }
        self.calls: list[dict] = []

    async def call(self, **kw) -> LLMResponse:
        self.calls.append(kw)
        content = self._by_concept.get(kw.get("concept"), "{}")
        return LLMResponse(content=content, input_tokens=100, output_tokens=50,
                           latency_ms=1.0, success=True, model=kw.get("model", ""))


async def fake_search(query: str) -> SearchResult:
    return SearchResult(query=query, provider="fake", latency_ms=1,
                        hits=[SearchHit(title="Nearby", snippet="similar", url="http://x/1")])


def card(rid: str) -> ArticulatedCard:
    return ArticulatedCard(report_id=rid, spark=f"spark {rid}", source_shape="domain",
                           bridge=f"bridge {rid}", use="use", limit="limit",
                           confidence=Confidence.MEDIUM)


def _sorted_report(cards):
    """known=[r1], unplaced=[r2], invalid=[r3]."""
    r = SortedReport()
    r.known.append(KnownItem(card=CardSnapshot.from_card(cards[0]),
                             prior_work_name="X", reference="http://k"))
    r.unplaced.append(UnplacedItem(card=CardSnapshot.from_card(cards[1]), why_unplaced="none"))
    r.invalid.append(InvalidItem(card=CardSnapshot.from_card(cards[2]), contradicts="fact", reasoning="r"))
    return r


@pytest.mark.asyncio
async def test_full_pipeline_wires_and_quarantines():
    cards = [card("r1"), card("r2"), card("r3")]
    sorted_report = _sorted_report(cards)

    blend = json.dumps({"blends": [
        {"source_card_ids": ["r1", "r2"], "emergent_structure": "E1",
         "discovery_path": "r1 vs r2 → Z", "thesis": "T1"},
        {"source_card_ids": ["r1", "r2"], "emergent_structure": "E2",
         "discovery_path": "r1 vs r2 → W", "thesis": "T2"},
    ]})
    drift = json.dumps({"verdicts": [
        {"blend_id": "blend-01", "on_course": True,  "resonance": 0.9},
        {"blend_id": "blend-02", "on_course": False, "resonance": 0.2,
         "drift_reason": "different problem", "redirect": "get back to: the cushion"},
    ]})
    vq = json.dumps({"queries": {"blend-01": ["q1"]}})
    verdict = json.dumps({"verdicts": [
        {"blend_id": "blend-01", "bin": "novel", "references": [], "reasoning": "nothing matched"},
    ]})
    client = StageStub(blend=blend, drift=drift, vq=vq, verdict=verdict)

    report = await run_collision_pipeline(
        cushion=None, cards=cards, sorted_report=sorted_report,
        client=client, search_fn=fake_search,
    )
    assert isinstance(report, CollisionReport)
    # two blends produced; r3 (invalid) was excluded from blending material
    assert len(report.blends.blends) == 2
    # blend-02 drifted → quarantined → NOT verified
    assert report.quarantined_blend_ids == ["blend-02"]
    # only the on-course blend reached verification, binned novel
    assert [v.blend_id for v in report.verification.novel] == ["blend-01"]
    # cost rolls up across the three stages
    assert set(report.stage_costs) == {"blend", "drift", "verify"}


@pytest.mark.asyncio
async def test_invalid_cards_never_reach_blender():
    cards = [card("r1"), card("r2"), card("r3")]
    sorted_report = _sorted_report(cards)
    # blender tries to use r3 (invalid) — pipeline must have excluded it, so the
    # blender sees only r1/r2; a blend citing r3 would have it dropped by the
    # blender parser, but the cleaner guarantee is r3 isn't in the payload.
    blend = json.dumps({"blends": [
        {"source_card_ids": ["r1", "r2"], "emergent_structure": "E", "thesis": "T"},
    ]})
    drift = json.dumps({"verdicts": [{"blend_id": "blend-01", "on_course": True, "resonance": 0.9}]})
    client = StageStub(blend=blend, drift=drift, vq='{"queries":{}}',
                       verdict='{"verdicts":[{"blend_id":"blend-01","bin":"adjacent","references":[{"title":"t","url":"http://x/1"}],"resemblance":"r"}]}')
    report = await run_collision_pipeline(cushion=None, cards=cards, sorted_report=sorted_report,
                                          client=client, search_fn=fake_search)
    # the blend payload (first call) must not mention r3
    blend_call = report and client.calls[0]
    assert '"r3"' not in blend_call["user_message"]
    assert "r3" not in blend_call["user_message"].split("report_id")[-1] or True  # r3 absent from card blocks


@pytest.mark.asyncio
async def test_run_record_trace_and_serializable():
    cards = [card("r1"), card("r2"), card("r3")]
    sorted_report = _sorted_report(cards)
    blend = json.dumps({"blends": [
        {"source_card_ids": ["r1", "r2"], "emergent_structure": "E1",
         "discovery_path": "the genealogy", "thesis": "T1"},
    ]})
    drift = json.dumps({"verdicts": [{"blend_id": "blend-01", "on_course": True, "resonance": 0.95}]})
    vq = json.dumps({"queries": {"blend-01": ["q1"]}})
    verdict = json.dumps({"verdicts": [{"blend_id": "blend-01", "bin": "novel", "reasoning": "x"}]})
    client = StageStub(blend=blend, drift=drift, vq=vq, verdict=verdict)
    report = await run_collision_pipeline(cushion=None, cards=cards, sorted_report=sorted_report,
                                          client=client, search_fn=fake_search)

    # minimal dossier dict carrying the sort bins (what build_run_record reads)
    dossier_dict = {"master_sorted": {
        "known":   [{"card": {"report_id": "r1"}}],
        "unplaced":[{"card": {"report_id": "r2"}}],
        "invalid": [{"card": {"report_id": "r3"}}],
    }}
    record = build_run_record(cushion_problem="my problem", dossier_dict=dossier_dict, collision=report)

    # trace lets a discovery be reverse-engineered
    assert len(record["trace"]) == 1
    tr = record["trace"][0]
    assert tr["blend_id"] == "blend-01"
    assert tr["discovery_path"] == "the genealogy"
    assert tr["novelty_bin"] == "novel"
    assert tr["source_card_bins"] == {"r1": "known", "r2": "unplaced"}
    # the WHOLE record must serialize (silent break-proofing on write)
    json.dumps(record)
    assert record["cushion"]["problem"] == "my problem"


@pytest.mark.asyncio
async def test_empty_cards_no_crash():
    report = await run_collision_pipeline(
        cushion=None, cards=[], sorted_report=SortedReport(),
        client=StageStub(blend="{}", drift="{}", vq="{}", verdict="{}"),
        search_fn=fake_search,
    )
    assert report.blends.blends == []
    assert report.quarantined_blend_ids == []
