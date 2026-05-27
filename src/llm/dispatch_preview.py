"""
Dispatch preview — pre-flight cost estimation + formation visibility.

WHY THIS EXISTS
    The reasoningEngine already has 5 layers of adaptive dispatch:
        1. Triage gate    → 4 routes (TRIVIAL/DIRECT/DIRECT_PLUS/DEEP)
        2. Chemistry router → picks active_domains + concepts_per_domain
        3. Le Chatelier's → skips low-scrutiny domains per iteration
        4. Convergence    → breaks the loop early when results stabilize
        5. Budget tracker → hard caps on iter / cost / wall-time / MCP calls

    What was MISSING was visibility and pre-flight estimation. The frontend
    couldn't see what would actually run, and the user couldn't see expected
    cost before LLM-heavy work fired. This module closes that gap.

WHAT IT DOES
    `preview_dispatch(text, client, ...)` runs ONLY the cheap upstream
    classifier(s) — triage (~$0.0002) and the formation router for DEEP
    routes (~$0.002) — and returns:

        - route + recommended_effort + risk flags (from triage)
        - formation_plan (active_domains, concepts_per_domain, ...) for DEEP
        - deterministic cost_breakdown — computed from PRICING table and
          per-call token estimates, NOT from the LLM's own guess

    Total cost of preview itself: < $0.003 (vs full dispatch which can run
    $0.10–$1.00). Use it to gate expensive work behind user confirmation
    when the estimate is high.

WHAT IT DOES NOT DO
    - Does NOT change the existing dispatch() flow. Purely additive.
    - Does NOT execute the engine. Stops after the router (or earlier).
    - Does NOT mutate any storage. No conversation records written.
    - Does NOT make cost decisions. Returns the estimate; the caller
      (UI or middleware) decides whether to proceed.

TOKEN + COST ESTIMATE METHODOLOGY
    Each call type has empirical token estimates (input + output) measured
    from production traces. Cost = (in_tokens × in_price + out_tokens ×
    out_price) / 1_000_000, using PRICING from provider_map.

    Estimates are intentionally CONSERVATIVE (rounded up). If actual cost
    comes in under the estimate, that's the desired direction. The budget
    tracker still enforces the hard cap regardless.

    To adjust: edit `_CALL_PROFILES` below. Each profile names a call type,
    the model resolver path, and median in/out tokens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.llm.client import LLMClient
from src.llm.effort import Effort, normalize_effort, iterations_for
from src.llm.provider_map import (
    DOMAIN_MODELS,
    GATING_MODEL,
    KE_CRITIC_MODELS,
    ROUTER_MODEL,
    SYNTHESIZER_MODEL,
    get_pricing,
)
from src.llm.router import LLMFormationPlan, route_problem
from src.llm.triage import Route, TriageResult, triage

# Engine result types are referenced only in type hints + duck-typed
# attribute access in the serializer below. Guarded import keeps the
# dispatch_preview module's runtime import graph cheap.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.llm.engine import EngineResult, Trajectory


# ---------------------------------------------------------------------------
# Per-call profiles — input/output token estimates per call type.
# Tuned to be conservative (slight overestimate). Edit here if real traffic
# diverges meaningfully.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CallProfile:
    """One call's expected size + model."""
    label: str
    model: str
    in_tokens: int
    out_tokens: int

    def cost_usd(self) -> float:
        in_price, out_price = get_pricing(self.model)
        return (self.in_tokens * in_price + self.out_tokens * out_price) / 1_000_000


_CALL_PROFILES: dict[str, CallProfile] = {
    # Triage gate — Gemini Flash-Lite, structured JSON output
    "triage":     CallProfile("triage",     GATING_MODEL,     in_tokens=600,  out_tokens=250),
    # Chemistry router — Gemini Flash, structured JSON output
    "router":     CallProfile("router",     ROUTER_MODEL,     in_tokens=1200, out_tokens=600),
    # DIRECT route — single Sonnet answer
    "direct":     CallProfile("direct",     SYNTHESIZER_MODEL, in_tokens=800,  out_tokens=600),
    # DIRECT_PLUS route — Sonnet answer with memory context injected
    "direct_plus": CallProfile("direct_plus", SYNTHESIZER_MODEL, in_tokens=1500, out_tokens=600),
    # Final narration in DEEP route — Sonnet, full output context
    "speech":     CallProfile("speech",     SYNTHESIZER_MODEL, in_tokens=2000, out_tokens=800),
}


def _avg_domain_call_profile() -> CallProfile:
    """
    A typical domain call. Domains use different models (4× Sonnet,
    1× DeepSeek V4 Pro on math). The estimate uses Sonnet 4.6 pricing
    as the conservative default — math runs on DeepSeek which is cheaper,
    so this overestimates slightly (which is fine for safety).
    """
    return CallProfile("domain", DOMAIN_MODELS["physics"], in_tokens=1500, out_tokens=1000)


def _avg_ke_critic_profile() -> CallProfile:
    """
    A typical Ke critic call. 5 pairs map to a mix of Sonnet/Haiku/Gemini
    Pro models. We average using Sonnet pricing as the conservative default.
    """
    return CallProfile("ke_critic", SYNTHESIZER_MODEL, in_tokens=800, out_tokens=400)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """
    Deterministic cost estimate, broken out per phase. Sum of `phases`
    equals `total_usd`. All amounts in USD, rounded to 5 decimals.

    Phases that don't apply to the route have value 0.0 — the dict is
    always the same shape so the frontend can render a fixed table.
    """
    total_usd: float
    estimated_llm_calls: int
    estimated_seconds: float
    phases: dict[str, float] = field(default_factory=dict)


@dataclass
class DispatchPreview:
    """
    Output of `preview_dispatch()`. Includes the triage result, the
    formation plan (if DEEP), and the deterministic cost estimate.
    """
    route: str
    recommended_effort: str
    triage_why: str
    risk_flags: list[str]
    interrupt: bool
    mcps_needed: list[dict]
    formation_plan: dict | None
    cost_breakdown: CostBreakdown
    classifier_mode: str


# ---------------------------------------------------------------------------
# Cost estimation — deterministic, no LLM call needed
# ---------------------------------------------------------------------------

def _seconds_estimate(num_calls: int) -> float:
    """
    Rough wall-time estimate.

    Each non-trivial LLM call is ~1.5s p50. Calls within an iteration
    fan out in parallel via call_batch, so the iteration's wall time is
    bounded by the slowest call in the batch + overhead.

    Conservative formula: 0.5s base + 1.2s × max-parallel-depth.
    For preview we approximate by treating LLM calls as half-parallel.
    """
    return round(0.5 + 1.2 * (num_calls / 2.0), 2)


def estimate_cost_breakdown(
    route: Route,
    effort: Effort,
    num_active_domains: int = 5,
    num_ke_pairs: int = 5,
    converged_iterations: int | None = None,
) -> CostBreakdown:
    """
    Compute a deterministic cost estimate for a dispatch decision.

    Parameters:
        route: Route from triage
        effort: Effort tier (only relevant for DEEP)
        num_active_domains: domains the router activated (default 5 = all)
        num_ke_pairs: Ke critic pairs that will run (default 5)
        converged_iterations: optional override for iteration count. When
            None, uses iterations_for(effort) as worst-case (no convergence).

    Returns a CostBreakdown with per-phase amounts and a total.
    """
    phases: dict[str, float] = {
        "triage": 0.0,
        "router": 0.0,
        "direct": 0.0,
        "domain_calls": 0.0,
        "ke_calls": 0.0,
        "speech": 0.0,
    }
    num_calls = 0

    # Triage runs for every route (the entry-point classifier).
    triage_p = _CALL_PROFILES["triage"]
    phases["triage"] = triage_p.cost_usd()
    num_calls += 1

    if route == Route.TRIVIAL:
        # Canned response. No additional LLM calls.
        pass

    elif route == Route.DIRECT:
        direct_p = _CALL_PROFILES["direct"]
        phases["direct"] = direct_p.cost_usd()
        num_calls += 1

    elif route == Route.DIRECT_PLUS:
        direct_plus_p = _CALL_PROFILES["direct_plus"]
        phases["direct"] = direct_plus_p.cost_usd()
        num_calls += 1

    elif route == Route.DEEP:
        # Router fires first (single call).
        router_p = _CALL_PROFILES["router"]
        phases["router"] = router_p.cost_usd()
        num_calls += 1

        # Main loop: each iteration runs domains + Ke critics in parallel.
        iters = converged_iterations if converged_iterations is not None else iterations_for(effort)
        domain_p = _avg_domain_call_profile()
        ke_p     = _avg_ke_critic_profile()

        phases["domain_calls"] = round(iters * num_active_domains * domain_p.cost_usd(), 5)
        phases["ke_calls"]     = round(iters * num_ke_pairs * ke_p.cost_usd(),         5)
        num_calls += iters * (num_active_domains + num_ke_pairs)

        # Final narration is one Sonnet call.
        speech_p = _CALL_PROFILES["speech"]
        phases["speech"] = speech_p.cost_usd()
        num_calls += 1

    # Round phases to 5 decimals after computation
    phases = {k: round(v, 5) for k, v in phases.items()}
    total = round(sum(phases.values()), 5)

    return CostBreakdown(
        total_usd=total,
        estimated_llm_calls=num_calls,
        estimated_seconds=_seconds_estimate(num_calls),
        phases=phases,
    )


# ---------------------------------------------------------------------------
# Formation plan serialization
# ---------------------------------------------------------------------------

def serialize_formation_plan(plan: LLMFormationPlan | None) -> dict | None:
    """
    Convert an LLMFormationPlan dataclass into a JSON-safe dict for API
    responses. Returns None if `plan` is None (TRIVIAL/DIRECT/DIRECT_PLUS
    routes don't have a formation plan).
    """
    if plan is None:
        return None
    return {
        "active_domains": [d.value for d in plan.active_domains],
        "concepts_per_domain": dict(plan.concepts_per_domain),
        "estimated_agent_count": plan.estimated_agent_count,
        "estimated_iterations": plan.estimated_iterations,
        "estimated_credit_cost": plan.estimated_credit_cost,
        "problem_complexity": plan.problem_complexity,
        "reasoning": plan.reasoning,
    }


# ---------------------------------------------------------------------------
# Trajectory serialization — engine output → UI card shape
# ---------------------------------------------------------------------------
#
# A Trajectory is the engine's per-finding output: a root cause, the
# evidence chain that revealed it, the bias it was hidden behind, the
# projected consequences if ignored, and the source domains whose
# perspectives converged on it.
#
# The UI renders each Trajectory as a "card" with four fields:
#   angle     — short label (which perspectives produced this finding)
#   title     — declarative sentence (the variable's description)
#   body      — readable consequence framing (cost_if_ignored, then
#               first projected consequence as fallback)
#   reasoning — full trace shown when the user expands the card
#
# This serializer is the ONLY place that maps engine internals to the
# UI's card contract. Touching the UI card shape means touching here.

def _format_enum(value: object) -> str:
    """Best-effort string representation for an Enum field."""
    inner = getattr(value, "value", value)
    return str(inner)


def _humanize_token(s: str) -> str:
    """Internal identifiers like 'first_principles' or 'The_Synthetic_Wall'
    leak straight from the engine. Replace underscores with spaces so the
    UI doesn't read like a stack trace."""
    return s.replace("_", " ").strip() if s else s


def _serialize_trajectory(t: "Trajectory") -> dict:
    rc = t.root_cause
    var = rc.variable

    # Title — sentence-shaped statement of the root cause.
    title = _humanize_token((var.description or var.name or "").strip())
    if not title:
        title = "Finding"

    # Body — try a real consequence; otherwise fall back to the bias
    # that hid the finding (one short sentence). If neither exists,
    # leave body empty so the UI just shows title + reasoning trace.
    # No more templated "consequences compound" footer.
    body = ""
    if t.consequences:
        c = t.consequences[0]
        timeframe = (c.timeframe or "").strip()
        body = (
            f"{c.description.strip()} ({timeframe})."
            if timeframe else c.description.strip()
        )
    elif rc.bias_that_hid_it:
        body = f"Hidden by: {_humanize_token(rc.bias_that_hid_it)}."

    # Angle — which source domains converged. Humanize the
    # framework names so the UI sees "first principles" not
    # "first_principles".
    angle = " · ".join(_humanize_token(d) for d in t.source_domains if d).lower()
    if not angle:
        angle = "perspective"

    # Reasoning — the trace shown when the user expands the card.
    # Built from the structured fields the engine already produced.
    # All internal identifiers humanized so the trace reads cleanly.
    parts: list[str] = []
    if rc.evidence_chain:
        parts.append(
            "Evidence: "
            + " → ".join(_humanize_token(e) for e in rc.evidence_chain)
        )
    if rc.bias_that_hid_it:
        parts.append(f"Bias that hid it: {_humanize_token(rc.bias_that_hid_it)}")
    if t.consequences:
        cons_lines = []
        for c in t.consequences:
            cons_lines.append(
                f"  · {c.description.strip()} "
                f"(in {c.timeframe.strip() or 'unspecified'}, "
                f"severity={_format_enum(c.severity)}, "
                f"p={c.probability:.0%})"
            )
        parts.append("Projected consequences:\n" + "\n".join(cons_lines))
    if rc.frameworks_that_agree:
        fmt = ", ".join(_humanize_token(_format_enum(f)) for f in rc.frameworks_that_agree)
        parts.append(f"Frameworks aligned: {fmt}")
    parts.append(f"Confidence: {t.confidence:.0%}")

    reasoning = "\n\n".join(parts)

    return {
        "angle": angle,
        "title": title,
        "body": body,
        "reasoning": reasoning,
    }


def serialize_perspectives(engine_result: "EngineResult | None") -> list[dict]:
    """
    Convert the engine's trajectories into the UI's card array.

    Returns an empty list when the engine didn't run (TRIVIAL / DIRECT /
    DIRECT_PLUS routes) — callers can safely render the result_text
    advisor voice alone in that case.

    The wire shape is `[{angle, title, body, reasoning}, ...]` and is
    the single source of truth for the UI's `ResultCardData[]` contract.
    """
    if engine_result is None:
        return []
    return [_serialize_trajectory(t) for t in engine_result.trajectories]


def _triage_to_dict(triage_result: TriageResult) -> dict:
    """Helper to serialize the triage portion of a preview."""
    return {
        "route": triage_result.route.value,
        "recommended_effort": triage_result.recommended_effort.value,
        "interrupt": triage_result.interrupt,
        "risk_flags": list(triage_result.risk_flags),
        "mcps_needed": [
            {"name": m.name, "why": m.why, "required": m.required}
            for m in triage_result.mcps_needed
        ],
        "why": triage_result.why,
        "classifier_mode": triage_result.classifier_mode,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def preview_dispatch(
    text: str,
    client: LLMClient,
    user_effort: Effort | str | None = Effort.MEDIUM,
    force_triage_result: TriageResult | None = None,
) -> DispatchPreview:
    """
    Run the cheap upstream classifier(s) and return a dispatch preview.

    Total cost of this preview: < $0.003 worst case (triage + router).
    The caller can decide whether to proceed with the full dispatch
    based on the cost_breakdown.

    Parameters:
        text: user question (same input as dispatch())
        client: LLMClient — used for both triage and the formation router
        user_effort: optional effort hint. Triage may override.
        force_triage_result: skip live classification (testing convenience)

    Returns DispatchPreview. Never raises on classifier failure; falls
    back to mock classifier with explicit `classifier_mode` tag.
    """
    # Step 1 — Triage (1 cheap call, or 0 in mock mode).
    if force_triage_result is not None:
        triage_result = force_triage_result
    else:
        triage_result = await triage(text, client=client)

    # Effort: user hint OR triage recommendation. The dispatcher uses the
    # max of these in DEEP routes, so we mirror that here for estimation.
    user_eff = normalize_effort(user_effort)
    effort = (
        triage_result.recommended_effort
        if iterations_for(triage_result.recommended_effort) >= iterations_for(user_eff)
        else user_eff
    )

    # Step 2 — Router (DEEP only; 1 cheap Gemini Flash call).
    formation_plan: LLMFormationPlan | None = None
    num_active_domains = 5
    num_ke_pairs = 5

    if triage_result.route == Route.DEEP:
        try:
            formation_plan = await route_problem(client, text)
            num_active_domains = max(1, len(formation_plan.active_domains))
            # Ke pairs scale with active domains. KE_CRITIC_MODELS has 5
            # canonical pairs; if fewer domains are active, fewer pairs fire.
            num_ke_pairs = _count_active_ke_pairs(formation_plan.active_domains)
        except Exception:  # noqa: BLE001
            # If the router fails, fall back to a default plan estimate.
            # The real dispatch path would also degrade gracefully here.
            formation_plan = None

    # Step 3 — Deterministic cost estimate.
    cost = estimate_cost_breakdown(
        route=triage_result.route,
        effort=effort,
        num_active_domains=num_active_domains,
        num_ke_pairs=num_ke_pairs,
    )

    return DispatchPreview(
        route=triage_result.route.value,
        recommended_effort=effort.value,
        triage_why=triage_result.why,
        risk_flags=list(triage_result.risk_flags),
        interrupt=triage_result.interrupt,
        mcps_needed=[
            {"name": m.name, "why": m.why, "required": m.required}
            for m in triage_result.mcps_needed
        ],
        formation_plan=serialize_formation_plan(formation_plan),
        cost_breakdown=cost,
        classifier_mode=triage_result.classifier_mode,
    )


def _count_active_ke_pairs(active_domains) -> int:
    """
    Count how many of the 5 canonical Ke critic pairs are active given the
    set of active domains. Both endpoints of a pair must be in
    active_domains for the pair to fire.
    """
    active_set = {d.value for d in active_domains}
    count = 0
    for (challenger, target) in KE_CRITIC_MODELS.keys():
        if challenger in active_set and target in active_set:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Serialization helper for the preview itself (used by the FastAPI route)
# ---------------------------------------------------------------------------

def preview_to_dict(preview: DispatchPreview) -> dict:
    """Convert a DispatchPreview into a JSON-safe dict for API responses."""
    return {
        "route": preview.route,
        "recommended_effort": preview.recommended_effort,
        "triage_why": preview.triage_why,
        "risk_flags": list(preview.risk_flags),
        "interrupt": preview.interrupt,
        "mcps_needed": list(preview.mcps_needed),
        "formation_plan": preview.formation_plan,
        "cost_breakdown": asdict(preview.cost_breakdown),
        "classifier_mode": preview.classifier_mode,
    }
