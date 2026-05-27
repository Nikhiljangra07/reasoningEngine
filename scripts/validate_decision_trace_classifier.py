"""
validate_decision_trace_classifier.py — Phase 2b regression test.

Verifies the InlineClassifier turns a synthetic IterationRecord into a
correctly-shaped DecisionTraceBundle. Makes one real Gemini Flash call
per fixture (~$0.0003 each).

USAGE
=====
  cd ~/Desktop/reasoningEngine
  source .venv/bin/activate
  python scripts/validate_decision_trace_classifier.py

EXIT CODES
==========
  0 — all checks passed (or graceful no-API-key fallback)
  1 — Gemini key configured but classification failed when it shouldn't
  2 — fail-safe path broken (empty input not handled correctly)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.core.thread_types import IterationRecord, SegmentedResponse, Segment
from src.llm.decision_trace_classifier import InlineClassifier


def _g(s): return f"\033[32m{s}\033[0m"
def _r(s): return f"\033[31m{s}\033[0m"
def _b(s): return f"\033[1m{s}\033[0m"
def _y(s): return f"\033[33m{s}\033[0m"


def _print_bundle(label, bundle, stats):
    print(f"\n--- {label} ---")
    print(f"  stats: success={stats.success} latency={stats.latency_ms}ms model={stats.model}")
    if stats.error:
        print(f"  error: {stats.error}")
    if bundle.user_message:
        print(f"  UserMessage: {bundle.user_message.text[:80]!r}")
    if bundle.system_response:
        print(f"  SystemResponse: {bundle.system_response.text[:80]!r}")
    print(f"  decisions ({len(bundle.decisions)}):")
    for d in bundle.decisions:
        print(f"    - [{d.status} conf={d.confidence:.2f}] {d.text[:80]!r}")
    print(f"  questions ({len(bundle.questions)}):")
    for q in bundle.questions:
        print(f"    - [resolved={q.resolved} conf={q.confidence:.2f}] {q.text[:80]!r}")
    print(f"  references ({len(bundle.references)}):")
    for r in bundle.references:
        print(f"    - [{r.kind} conf={r.confidence:.2f}] {r.target[:80]!r}")
    print(f"  insights ({len(bundle.insights)}):")
    for i in bundle.insights:
        print(f"    - [conf={i.confidence:.2f}] {i.text[:80]!r}")


async def main() -> int:
    print(_b("\n=== Decision Trace Classifier validation ===\n"))

    classifier = InlineClassifier()
    print(f"Classifier model: {classifier.model}\n")

    # Fixture 1: a clear decision-bearing turn — should extract Decision + Question + Reference
    iter1 = IterationRecord(
        id="itr-2b-fix1", thread_id="thr-2b-test", sequence_num=1,
        workspace_id="cursor", surface_id="chat",
        question="Let's go with Neo4j Aura Free for the beta phase — can it handle 10-15 testers? Check https://neo4j.com/aura/.",
        response=SegmentedResponse(
            overall_confidence="high",
            synthesizer=Segment(
                text="Yes, comfortably. Aura Free is 200K nodes / 400K relationships — at 10-15 testers you're nowhere near the cap. I'd recommend committing to it for beta.",
                confidence="high", delivered_at=time.time(),
            ),
        ),
        created_at=time.time(), completed_at=time.time() + 1,
        meta={"user_id": "u-2b"},
    )
    b1, s1 = await classifier.classify_iteration(iter1)
    _print_bundle("Fixture 1 — decision-bearing turn", b1, s1)

    # Fixture 2: an open-question turn — Question only, no Decision
    iter2 = IterationRecord(
        id="itr-2b-fix2", thread_id="thr-2b-test", sequence_num=2,
        workspace_id="cursor", surface_id="map-room",
        question="What's the latency difference between Neo4j Aura Free and Pro?",
        response=SegmentedResponse(
            overall_confidence="moderate",
            synthesizer=Segment(
                text="I don't have specific benchmark numbers. Pro instances typically run on dedicated hardware (1-32GB RAM) so they're consistently faster on cold queries, but for the working-set-fits-in-RAM case both are comparable.",
                confidence="moderate", delivered_at=time.time(),
            ),
        ),
        created_at=time.time(), completed_at=time.time() + 1,
        meta={"user_id": "u-2b"},
    )
    b2, s2 = await classifier.classify_iteration(iter2)
    _print_bundle("Fixture 2 — open-question turn", b2, s2)

    # Fixture 3: empty input — fail-safe verbatim path
    iter3 = IterationRecord(
        id="itr-2b-fix3", thread_id="thr-2b-test", sequence_num=3,
        workspace_id="cursor", surface_id="chat",
        question="", response=None,
    )
    b3, s3 = await classifier.classify_iteration(iter3)
    print("\n--- Fixture 3 — empty input ---")
    print(f"  stats: success={s3.success} error={s3.error}")
    print(f"  user_message={b3.user_message}, system_response={b3.system_response}")

    # Assertions: cheap shape checks. Quality checks (was the extraction sensible?)
    # are left to the operator inspecting the output.
    fail = False
    if b3.user_message is not None or b3.system_response is not None:
        print(_r("  ✗ empty input should produce empty messages"))
        fail = True
    if b3.decisions or b3.questions or b3.references or b3.insights:
        print(_r("  ✗ empty input should produce no extracted events"))
        fail = True

    # If the API key is present, fixtures 1 and 2 should have succeeded.
    have_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if have_key:
        for label, stats, bundle in [("fixture 1", s1, b1), ("fixture 2", s2, b2)]:
            if not stats.success:
                print(_r(f"  ✗ {label}: classification failed with key present: {stats.error}"))
                fail = True
            if bundle.user_message is None or bundle.system_response is None:
                print(_r(f"  ✗ {label}: missing verbatim messages"))
                fail = True
    else:
        print(_y("  ⚠ no GEMINI_API_KEY / GOOGLE_API_KEY — only fail-safe path tested"))

    if fail:
        print(_r("\n=== RESULT: FAILED ===\n"))
        return 1 if have_key else 2
    print(_g("\n=== RESULT: PASSED ===\n"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
