"""
Tests for the drift-checker (src/wandering/drift_checker.py).

Fully offline: stub LLM client returns crafted JSON. Covers the happy
path, the low-resonance belt-and-braces flag, unjudged-blend defaulting,
fail-open on LLM error, and the empty-input guard.
"""

from __future__ import annotations

import json

import pytest

from src.llm.client import LLMResponse
from src.wandering.blender import Blend
from src.wandering.drift_checker import (
    DriftReport,
    RESONANCE_DRIFT_FLOOR,
    check_drift,
)


class StubClient:
    def __init__(self, content: str, success: bool = True):
        self._content = content
        self._success = success
        self.calls: list[dict] = []

    async def call(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(
            content=self._content, input_tokens=200, output_tokens=80,
            latency_ms=1.0, success=self._success, model=kwargs.get("model", ""),
            error=None if self._success else "stub_failure",
        )


def make_blend(bid: str) -> Blend:
    return Blend(blend_id=bid, source_card_ids=["r1", "r2"], thesis=f"thesis {bid}",
                 advances_cushion="advances it", emergent_structure="emergent x")


def _verdicts(vs):
    return json.dumps({"verdicts": vs})


@pytest.mark.asyncio
async def test_happy_path_on_course_and_drift():
    blends = [make_blend("blend-01"), make_blend("blend-02")]
    content = _verdicts([
        {"blend_id": "blend-01", "on_course": True,  "resonance": 0.9, "drift_reason": "", "redirect": ""},
        {"blend_id": "blend-02", "on_course": False, "resonance": 0.5,
         "drift_reason": "solves a generic search problem, not the cushion",
         "redirect": "get back to: the user's vague-anchor case"},
    ])
    report = await check_drift(cushion=None, blends=blends, client=StubClient(content))
    assert isinstance(report, DriftReport)
    assert report.on_course_ids == ["blend-01"]
    assert report.drifting_ids == ["blend-02"]
    v = report.verdict_for("blend-02")
    assert v.on_course is False
    assert "generic search" in v.drift_reason
    assert v.redirect.startswith("get back to")


@pytest.mark.asyncio
async def test_low_resonance_forces_drift_flag_even_if_model_says_on_course():
    blends = [make_blend("blend-01")]
    # model claims on_course but resonance is below the floor → flagged
    content = _verdicts([{"blend_id": "blend-01", "on_course": True,
                          "resonance": RESONANCE_DRIFT_FLOOR - 0.1}])
    report = await check_drift(cushion=None, blends=blends, client=StubClient(content))
    assert report.drifting_ids == ["blend-01"]
    assert report.verdict_for("blend-01").on_course is False


@pytest.mark.asyncio
async def test_unjudged_blend_defaults_on_course():
    blends = [make_blend("blend-01"), make_blend("blend-02")]
    # model only judged blend-01; blend-02 must default on_course (stay out of the way)
    content = _verdicts([{"blend_id": "blend-01", "on_course": True, "resonance": 0.8}])
    report = await check_drift(cushion=None, blends=blends, client=StubClient(content))
    assert "blend-02" in report.on_course_ids
    assert any(n["reason"] == "blend_unjudged_defaulted_on_course" for n in report.parser_notes)


@pytest.mark.asyncio
async def test_fail_open_on_llm_error():
    blends = [make_blend("blend-01"), make_blend("blend-02")]
    report = await check_drift(cushion=None, blends=blends, client=StubClient("", success=False))
    # supervisor must never silently kill the blender's work
    assert set(report.on_course_ids) == {"blend-01", "blend-02"}
    assert report.drifting_ids == []
    assert any(n["reason"] == "llm_call_failed_defaulted_on_course" for n in report.parser_notes)


@pytest.mark.asyncio
async def test_empty_input_no_call():
    client = StubClient("{}")
    report = await check_drift(cushion=None, blends=[], client=client)
    assert report.verdicts == []
    assert client.calls == []
