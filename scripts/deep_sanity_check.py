"""
deep_sanity_check.py — the FINAL offline check for the collision pipeline.

Hunts for SILENT breakage that unit tests can miss:
  1. import every wandering module (cycle / import-error detection)
  2. py_compile every pipeline source + script (syntax)
  3. import every runner/replay script (their imports resolve)
  4. every component report.to_dict() is JSON-serializable (write-path safety)
  5. run the whole collision pipeline end-to-end on stubs + serialize the
     unified run record (the documentation environment actually produces)
  6. the dam is intact (master_synthesizer path still wired in build_dossier)
  7. the two fixes did not regress (balanced verifier doctrine + discovery_path)

Offline. No API keys, no network, no spend. Exits non-zero on any failure.

Usage:
    PYTHONPATH=. python scripts/deep_sanity_check.py
"""

from __future__ import annotations

import asyncio
import importlib
import json
import py_compile
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PASS, FAIL = 0, 0
FAILURES: list[str] = []


def check(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        FAIL += 1
        FAILURES.append(f"{name}: {type(e).__name__}: {e}")
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# ── 1. import every wandering module ───────────────────────────────────
def _imports():
    wd = REPO_ROOT / "src" / "wandering"
    for py in sorted(wd.glob("*.py")):
        if py.stem == "__init__":
            continue
        importlib.import_module(f"src.wandering.{py.stem}")


# ── 2. py_compile every source + script ────────────────────────────────
def _compile():
    targets = list((REPO_ROOT / "src" / "wandering").glob("*.py"))
    targets += [
        REPO_ROOT / "scripts" / s for s in (
            "control_room.py", "run_collision_pipeline.py", "replay_verified_sort.py",
            "replay_blend.py", "replay_drift.py", "replay_blend_verify.py",
        )
    ]
    for t in targets:
        py_compile.compile(str(t), doraise=True)


# ── 3. import the runner/replay scripts (imports resolve) ──────────────
def _import_scripts():
    for mod in ("control_room", "run_collision_pipeline", "replay_blend"):
        importlib.import_module(mod)


# ── 4. report dataclasses serialize ────────────────────────────────────
def _serializable():
    from src.wandering.blender import BlendBatch
    from src.wandering.drift_checker import DriftReport
    from src.wandering.blend_verify import BlendVerificationReport
    from src.wandering.sorter_verify import EvidenceLedger
    from src.wandering.master_sorter import SortedReport
    from src.wandering.collision_pipeline import CollisionReport
    for obj in (BlendBatch(), DriftReport(), BlendVerificationReport(),
                EvidenceLedger(), SortedReport(), CollisionReport()):
        json.dumps(obj.to_dict())


# ── 5. full mocked pipeline + run record ───────────────────────────────
def _e2e():
    from src.bridge.web_search import SearchHit, SearchResult
    from src.llm.client import LLMResponse
    from src.wandering.articulate import ArticulatedCard
    from src.wandering.report import Confidence
    from src.wandering.master_sorter import (
        CardSnapshot, InvalidItem, KnownItem, SortedReport, UnplacedItem)
    from src.wandering.collision_pipeline import build_run_record, run_collision_pipeline

    class Stub:
        def __init__(self, m): self.m = m
        async def call(self, **kw):
            return LLMResponse(content=self.m.get(kw.get("concept"), "{}"),
                               input_tokens=10, output_tokens=5, latency_ms=1,
                               success=True, model=kw.get("model", ""))

    async def fsearch(q):
        return SearchResult(query=q, provider="fake", latency_ms=1,
                            hits=[SearchHit(title="t", snippet="s", url="http://x/1")])

    def c(rid):
        return ArticulatedCard(report_id=rid, spark="s", source_shape="d", bridge="b",
                               use="u", limit="l", confidence=Confidence.MEDIUM)
    cards = [c("r1"), c("r2"), c("r3")]
    sr = SortedReport()
    sr.known.append(KnownItem(card=CardSnapshot.from_card(cards[0]), prior_work_name="X", reference="k"))
    sr.unplaced.append(UnplacedItem(card=CardSnapshot.from_card(cards[1]), why_unplaced="n"))
    sr.invalid.append(InvalidItem(card=CardSnapshot.from_card(cards[2]), contradicts="f", reasoning="r"))

    stub = Stub({
        "blend": json.dumps({"blends": [{"source_card_ids": ["r1", "r2"],
                  "emergent_structure": "E", "discovery_path": "path", "thesis": "T"}]}),
        "drift_check": json.dumps({"verdicts": [{"blend_id": "blend-01", "on_course": True, "resonance": 0.9}]}),
        "verify_queries": json.dumps({"queries": {"blend-01": ["q"]}}),
        "verdict": json.dumps({"verdicts": [{"blend_id": "blend-01", "bin": "novel", "reasoning": "x"}]}),
    })

    async def go():
        rep = await run_collision_pipeline(cushion=None, cards=cards, sorted_report=sr,
                                           client=stub, search_fn=fsearch)
        rec = build_run_record(cushion_problem="p", dossier_dict={"master_sorted": sr.to_dict()}, collision=rep)
        json.dumps(rec)
        assert rec["trace"][0]["discovery_path"] == "path"
        assert rec["trace"][0]["novelty_bin"] == "novel"
        assert rep.quarantined_blend_ids == []
    asyncio.run(go())


# ── 6. the dam: synthesizer path still wired ───────────────────────────
def _dam():
    import inspect
    from src.wandering import dossier as d
    from src.wandering.master_synthesizer import master_synthesize  # noqa: F401
    src = inspect.getsource(d.build_dossier)
    assert 'pipeline_mode == "sorter"' in src
    assert 'pipeline_mode == "synthesizer"' in src
    assert "master_synthesize(" in src


# ── 7. the two fixes did not regress ───────────────────────────────────
def _fixes():
    from src.wandering.blend_verify import _BLEND_VERIFY_DOCTRINE as D
    assert "Default to ADJACENT over NOVEL" not in D, "certainty bias returned"
    assert "same-MOVE" in D and "TOO SMOOTH" in D, "balance framing missing"
    from src.wandering.blender import SelectionRationale
    assert "discovery_path" in SelectionRationale.__dataclass_fields__, "discovery_path missing"


# ── 8. halo auditor: 3 checkpoints run + serialize (observer layer) ────
def _halo():
    from src.llm.client import LLMResponse
    from src.wandering.articulate import ArticulatedCard
    from src.wandering.report import Confidence
    from src.wandering.blender import Blend
    from src.wandering.halo_auditor import (
        AuditReport, audit_blends, audit_cards, audit_cushion)

    class Stub:
        async def call(self, **kw):
            return LLMResponse(
                content=json.dumps({"blind_spots": [
                    {"blind_spot": "x", "why_it_matters": "y",
                     "severity": "high", "suggested_angle": "z"}]}),
                input_tokens=10, output_tokens=5, latency_ms=1,
                success=True, model=kw.get("model", ""))

    def c(rid):
        return ArticulatedCard(report_id=rid, spark="s", source_shape="d", bridge="b",
                               use="u", limit="l", confidence=Confidence.MEDIUM)

    async def go():
        cu = await audit_cushion(cushion="advance my concept", client=Stub(), model="m")
        ca = await audit_cards(cushion="c", cards=[c("r1"), c("r2")], client=Stub(), model="m")
        bl = await audit_blends(
            cushion="c", blends=[Blend(blend_id="blend-01", source_card_ids=["r1"], thesis="T")],
            client=Stub(), model="m")
        rep = AuditReport(cushion_audit=cu, cards_audit=ca, blends_audit=bl, model="m",
                          total_cost_usd=cu.cost_usd + ca.cost_usd + bl.cost_usd)
        json.dumps(rep.to_dict())
        assert (cu.layer, ca.layer, bl.layer) == ("cushion", "cards", "blends")
        assert len(rep.all_blind_spots()) == 3
    asyncio.run(go())


# ── 9. the halo is wired INLINE into the live runner (not just post-hoc) ─
def _halo_wired():
    import inspect
    import run_collision_pipeline as rcp
    src = inspect.getsource(rcp.run)
    for needle in ('audit_cushion(', 'audit_cards(', 'audit_blends(',
                   '_safe_halo(', 'control_room.AUDITOR_MODEL'):
        assert needle in src, f"halo wiring missing from runner: {needle}"


# ── 10. quality ranker runs, keeps every blend, ranks by advancement ───
def _ranker():
    from src.llm.client import LLMResponse
    from src.wandering.blender import Blend
    from src.wandering.quality_ranker import rank_blends

    class Stub:
        async def call(self, **kw):
            return LLMResponse(content=json.dumps({"rankings": [
                {"blend_id": "blend-01", "advancement": 0.9, "blind_spots_addressed": ["G1"]},
                {"blend_id": "blend-02", "advancement": 0.3}]}),
                input_tokens=10, output_tokens=5, latency_ms=1, success=True, model="m")

    async def go():
        r = await rank_blends(
            cushion="c",
            blends=[Blend(blend_id="blend-01", source_card_ids=["r1"], thesis="T"),
                    Blend(blend_id="blend-02", source_card_ids=["r2"], thesis="U")],
            blind_spots=[], client=Stub())
        assert len(r.ranked) == 2 and r.ranked[0].blend_id == "blend-01"  # higher adv, all kept
        json.dumps(r.to_dict())
    asyncio.run(go())


# ── 11. quality ranker wired as the FINAL pass in the runner ───────────
def _ranker_wired():
    import inspect
    import run_collision_pipeline as rcp
    src = inspect.getsource(rcp.run)
    for needle in ('rank_blends(', 'control_room.RANKER_MODEL', 'quality.json'):
        assert needle in src, f"ranker wiring missing from runner: {needle}"


if __name__ == "__main__":
    print("\nDEEP SANITY CHECK — collision pipeline\n" + "=" * 50)
    check("1. all wandering modules import (no cycles)", _imports)
    check("2. all sources + scripts compile", _compile)
    check("3. runner/replay scripts import", _import_scripts)
    check("4. component reports JSON-serialize", _serializable)
    check("5. full mocked pipeline + run record", _e2e)
    check("6. dam intact (synthesizer path wired)", _dam)
    check("7. balance fix + discovery_path not regressed", _fixes)
    check("8. halo auditor runs 3 checkpoints + serializes", _halo)
    check("9. halo wired INLINE into live runner", _halo_wired)
    check("10. quality ranker runs + keeps every blend", _ranker)
    check("11. quality ranker wired as final pass", _ranker_wired)
    print("=" * 50)
    print(f"{PASS} passed, {FAIL} failed")
    if FAIL:
        print("\nFAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL DEEP CHECKS GREEN ✓")
