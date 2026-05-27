"""
Drift detection.

A decision says: "we chose request-shape hash + orgId for idempotency
keys (D-014)". The code at refund.ts:17 used to honor that. If a later
PR changed it to use orgId alone, the code has drifted from the
decision. detect_drift() catches this.

The comparison itself is a semantic question — does the current code
*honor the intent* of the decision? — which needs an LLM call. That
comparator is a separate task; this file ships with a stub comparator
that returns "no_action" so the rest of the pipeline can be wired up.

When the real comparator lands, it will call src/llm/client.py with the
SYNTHESIZER role (or a new "drift_comparator" role) and consume:
    - decision.title
    - decision.rationale
    - decision.evidence
    - the current code text at code_ref
and return:
    - is_drifted: bool
    - drift_description: str (human-readable)
    - confidence: float (0–1)
    - suggested_action: "reconcile" | "supersede" | "no_action"
"""

from __future__ import annotations

from typing import Protocol

from src.bridge.graphify_adapter import GraphifyAdapter
from src.bridge.types import CodeRef, DecisionAnchor, DriftReport


class DriftComparator(Protocol):
    """
    Pluggable comparator interface.

    The real implementation will be an async function backed by an LLM
    call. The stub implementation below has the same signature so it
    can be swapped in without touching detect_drift().
    """

    async def __call__(
        self, decision: DecisionAnchor, code_ref: CodeRef, current_code: str
    ) -> tuple[bool, str, float, str]:
        ...


async def stub_comparator(
    decision: DecisionAnchor, code_ref: CodeRef, current_code: str
) -> tuple[bool, str, float, str]:
    """
    Stub comparator. Always returns "no drift, no action, zero confidence".

    TODO: replace with an LLM call via src/llm/client.py. The real
    implementation should:
        1. Render a system prompt that names the decision (id, title,
           rationale) and asks: "Does the following code still honor
           this decision?"
        2. Pass current_code as the user message.
        3. Parse the JSON response into the (is_drifted, description,
           confidence, suggested_action) tuple this contract returns.

    Returning zero confidence here means callers will not mistake the
    stub's silence for actual agreement.
    """
    return (
        False,
        "comparator not yet implemented",
        0.0,
        "no_action",
    )


async def detect_drift_for_ref(
    decision: DecisionAnchor,
    code_ref: CodeRef,
    graphify: GraphifyAdapter,
    comparator: DriftComparator = stub_comparator,
) -> DriftReport:
    """
    Run drift detection against a single code_ref.

    Pulls the current code at code_ref (via graphify for now —
    eventually a real file read or a graphify node-content lookup) and
    invokes the comparator. Wraps the result in a DriftReport.
    """
    # TODO: replace this with a real file-text read when the comparator
    # is implemented. For now we pass an empty string — the stub
    # comparator ignores it anyway, and live runs will fail loudly when
    # the real comparator tries to reason about empty content.
    current_code = await _fetch_code_at_ref(graphify, code_ref)

    is_drifted, description, confidence, action = await comparator(
        decision, code_ref, current_code
    )

    return DriftReport(
        decision=decision,
        code_ref=code_ref,
        is_drifted=is_drifted,
        drift_description=description,
        confidence=confidence,
        suggested_action=action,
    )


async def detect_drift(
    decision: DecisionAnchor,
    graphify: GraphifyAdapter,
    comparator: DriftComparator = stub_comparator,
) -> DriftReport:
    """
    Run drift detection across all of a decision's code_refs.

    Returns a single DriftReport summarising the overall verdict. The
    top-level is_drifted is True if ANY per-ref check came back as
    drifted; the code_ref on the returned report is the first drifted
    ref, or the first ref overall if nothing drifted.

    Decisions with zero code_refs return a no-op report with a clear
    description rather than raising — a "floating" decision is itself a
    signal worth surfacing.
    """
    if not decision.code_refs:
        return DriftReport(
            decision=decision,
            code_ref=CodeRef(file_path="", line_start=1, line_end=1),
            is_drifted=False,
            drift_description=(
                "decision has no code_refs — cannot check for drift. "
                "Add code_refs to anchor this decision to source locations."
            ),
            confidence=0.0,
            suggested_action="no_action",
        )

    per_ref_reports: list[DriftReport] = []
    for ref in decision.code_refs:
        per_ref_reports.append(
            await detect_drift_for_ref(decision, ref, graphify, comparator)
        )

    drifted = [r for r in per_ref_reports if r.is_drifted]
    if drifted:
        # Summarize against the first drifted ref; preserve all per-ref
        # detail on per_ref_reports for callers that need the breakdown.
        first = drifted[0]
        return DriftReport(
            decision=decision,
            code_ref=first.code_ref,
            is_drifted=True,
            drift_description=first.drift_description,
            confidence=first.confidence,
            suggested_action=first.suggested_action,
            per_ref_reports=per_ref_reports,
        )

    first = per_ref_reports[0]
    return DriftReport(
        decision=decision,
        code_ref=first.code_ref,
        is_drifted=False,
        drift_description=first.drift_description,
        confidence=first.confidence,
        suggested_action=first.suggested_action,
        per_ref_reports=per_ref_reports,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_code_at_ref(
    graphify: GraphifyAdapter, code_ref: CodeRef
) -> str:
    """
    Fetch the current source text at a code_ref.

    TODO: implement when the real comparator lands. Options:
        (a) Direct file read at repo_root/code_ref.file_path, slicing
            line_start..line_end. Simple, but couples the bridge to the
            filesystem layout — fine for local dev, fragile for hosted use.
        (b) Look up the graphify node and read a node-attached snippet
            if graphify stores one. Decouples from the filesystem.
        (c) Hybrid: try (b), fall back to (a).

    For now returns empty string. The stub comparator ignores this.
    """
    return ""
