"""
Tests for the halo auditor (src/wandering/halo_auditor.py) — blend-03 Phase 1.

Fully offline: stub client returns blind-spot JSON. Covers per-layer audits,
the bounded/ranked top-N guard, fail-open, the full three-layer report, and
serialization.
"""

from __future__ import annotations

import json

import pytest

from src.llm.client import LLMResponse
from src.wandering.articulate import ArticulatedCard
from src.wandering.report import Confidence
from src.wandering.blender import Blend
from src.wandering.halo_auditor import (
    MAX_BLIND_SPOTS,
    AuditReport,
    audit_blends,
    audit_cards,
    audit_cushion,
    run_halo_audit,
)


class StubClient:
    def __init__(self, content: str, success: bool = True):
        self._content = content
        self._success = success
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(content=self._content, input_tokens=200, output_tokens=120,
                           latency_ms=1.0, success=self._success, model=kwargs.get("model", ""),
                           error=None if self._success else "stub_failure")


def _bs(n, sev="medium"):
    return json.dumps({"blind_spots": [
        {"blind_spot": f"missing thing {i}", "why_it_matters": "threatens the cushion",
         "severity": sev, "suggested_angle": "look here"} for i in range(n)
    ]})


def card(rid):
    return ArticulatedCard(report_id=rid, spark="s", source_shape="physics",
                           bridge="a bridge claim", use="u", limit="l", confidence=Confidence.MEDIUM)


@pytest.mark.asyncio
async def test_audit_cushion_parses_and_tags_layer():
    a = await audit_cushion(cushion="advance my concept", client=StubClient(_bs(3)))
    assert a.layer == "cushion"
    assert len(a.blind_spots) == 3
    assert all(b.layer == "cushion" for b in a.blind_spots)
    assert a.blind_spots[0].suggested_angle == "look here"


@pytest.mark.asyncio
async def test_blind_spots_are_bounded_and_severity_ranked():
    # 9 spots, mixed severity -> capped at MAX, high first
    content = json.dumps({"blind_spots":
        [{"blind_spot": f"low {i}", "severity": "low"} for i in range(4)] +
        [{"blind_spot": f"high {i}", "severity": "high"} for i in range(5)]})
    a = await audit_cushion(cushion="c", client=StubClient(content))
    assert len(a.blind_spots) == MAX_BLIND_SPOTS
    # all the kept ones should be the high-severity ones (ranked first)
    assert all(b.severity == "high" for b in a.blind_spots)


@pytest.mark.asyncio
async def test_audit_cards_and_blends():
    ac = await audit_cards(cushion="c", cards=[card("r1"), card("r2")], client=StubClient(_bs(2)))
    assert ac.layer == "cards" and len(ac.blind_spots) == 2
    ab = await audit_blends(cushion="c", blends=[Blend(blend_id="b1", source_card_ids=["r1"], thesis="T")],
                            client=StubClient(_bs(2)))
    assert ab.layer == "blends" and len(ab.blind_spots) == 2


@pytest.mark.asyncio
async def test_fail_open_marks_not_ok():
    a = await audit_cushion(cushion="c", client=StubClient("", success=False))
    assert a.ok is False
    assert a.blind_spots == []
    assert "failed" in a.note


@pytest.mark.asyncio
async def test_run_halo_audit_all_three_layers():
    report = await run_halo_audit(
        cushion="advance my concept",
        cards=[card("r1"), card("r2")],
        blends=[Blend(blend_id="b1", source_card_ids=["r1", "r2"], thesis="T")],
        client=StubClient(_bs(2)),
    )
    assert isinstance(report, AuditReport)
    assert report.cushion_audit and report.cards_audit and report.blends_audit
    assert len(report.all_blind_spots()) == 6     # 2 per layer x 3 layers
    d = report.to_dict()
    json.dumps(d)                                  # serializable
    assert d["cards_audit"]["layer"] == "cards"


@pytest.mark.asyncio
async def test_run_halo_audit_skips_empty_layers():
    report = await run_halo_audit(cushion="c", cards=[], blends=[], client=StubClient(_bs(1)))
    assert report.cushion_audit is not None       # cushion always audited
    assert report.cards_audit is None             # no cards -> no card audit
    assert report.blends_audit is None
