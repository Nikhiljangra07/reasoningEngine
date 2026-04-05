"""
Credit-Based Payment System.

Users purchase credits. Each deep reasoning request consumes credits
based on ACTUAL compute used, not estimates.

credit_cost = base_cost + (active_domains × domain_cost) + (iterations_run × iteration_cost)

Failure policy:
  - Domain fails → that domain's credits NOT charged
  - 2 domains fail → those credits refunded, user informed
  - 3+ domains fail → entire response FREE + next request free

ISOLATION: Pure calculations. No domain imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Credit pricing (these are abstract credits, not USD)
BASE_COST = 2.0            # minimum charge for any request (covers triage + validation)
DOMAIN_COST = 1.5          # per domain that actually ran
ITERATION_COST = 1.0       # per iteration completed
KE_COST = 0.5              # per Ke challenge pair that ran

# Failure thresholds
PARTIAL_FAILURE_THRESHOLD = 2    # 2 domain failures → refund those domains
FULL_FAILURE_THRESHOLD = 3       # 3+ failures → entire response free


@dataclass
class CreditEstimate:
    """Pre-execution credit estimate shown to user before they confirm."""
    estimated_total: float
    base_cost: float
    domain_cost: float
    iteration_cost: float
    ke_cost: float
    active_domains: int
    estimated_iterations: int
    estimated_ke_pairs: int


@dataclass
class CreditInvoice:
    """Post-execution credit calculation based on ACTUAL compute."""
    actual_total: float
    base_cost: float
    domain_cost: float
    iteration_cost: float
    ke_cost: float
    domains_ran: int
    domains_failed: int
    iterations_ran: int
    ke_pairs_ran: int
    refund: float
    free_retry_issued: bool
    breakdown: dict[str, float]


def estimate_credits(
    active_domains: int,
    estimated_iterations: int,
    estimated_ke_pairs: int,
) -> CreditEstimate:
    """
    Estimate credits BEFORE execution. Shown to user for confirmation.

    "This analysis is estimated at X credits. Proceed?"
    """
    domain_total = active_domains * DOMAIN_COST
    iteration_total = estimated_iterations * ITERATION_COST
    ke_total = estimated_ke_pairs * estimated_iterations * KE_COST
    total = BASE_COST + domain_total + iteration_total + ke_total

    return CreditEstimate(
        estimated_total=total,
        base_cost=BASE_COST,
        domain_cost=domain_total,
        iteration_cost=iteration_total,
        ke_cost=ke_total,
        active_domains=active_domains,
        estimated_iterations=estimated_iterations,
        estimated_ke_pairs=estimated_ke_pairs,
    )


def calculate_invoice(
    domains_ran: int,
    domains_failed: int,
    iterations_ran: int,
    ke_pairs_ran: int,
    is_phase_one_only: bool = False,
) -> CreditInvoice:
    """
    Calculate credits AFTER execution based on ACTUAL compute.

    Failed domains don't charge. 3+ failures = free.
    """
    # Base cost always applies (triage + validation ran)
    base = BASE_COST

    # Domain cost: only for domains that succeeded
    successful_domains = domains_ran - domains_failed
    domain_total = successful_domains * DOMAIN_COST

    # Iteration cost
    iteration_total = iterations_ran * ITERATION_COST

    # Ke cost
    ke_total = ke_pairs_ran * KE_COST

    # Subtotal before failure policy
    subtotal = base + domain_total + iteration_total + ke_total

    # Failure policy
    refund = 0.0
    free_retry = False

    if domains_failed >= FULL_FAILURE_THRESHOLD:
        # 3+ domains failed → entire response free + free retry
        refund = subtotal
        free_retry = True
    elif domains_failed >= PARTIAL_FAILURE_THRESHOLD:
        # 2 domains failed → refund those domain costs
        refund = domains_failed * DOMAIN_COST

    actual_total = max(subtotal - refund, 0.0)

    # Phase 1 discount (user didn't ask for full analysis)
    if is_phase_one_only:
        actual_total *= 0.6  # 40% discount for quick batch

    breakdown = {
        "base": base,
        "domains": domain_total,
        "iterations": iteration_total,
        "ke_challenges": ke_total,
        "refund": -refund,
    }

    return CreditInvoice(
        actual_total=actual_total,
        base_cost=base,
        domain_cost=domain_total,
        iteration_cost=iteration_total,
        ke_cost=ke_total,
        domains_ran=domains_ran,
        domains_failed=domains_failed,
        iterations_ran=iterations_ran,
        ke_pairs_ran=ke_pairs_ran,
        refund=refund,
        free_retry_issued=free_retry,
        breakdown=breakdown,
    )


def format_credit_display(
    estimate: CreditEstimate | None = None,
    invoice: CreditInvoice | None = None,
) -> str:
    """
    Format credit info for user-facing display.
    Never expose internal terminology.
    """
    if estimate:
        return (
            f"This analysis is estimated at {estimate.estimated_total:.1f} credits. "
            f"({estimate.active_domains} domains, ~{estimate.estimated_iterations} iterations)"
        )

    if invoice:
        msg = f"Analysis complete. {invoice.actual_total:.1f} credits used."

        if invoice.refund > 0:
            msg += f" ({invoice.refund:.1f} credits refunded due to incomplete analysis.)"

        if invoice.free_retry_issued:
            msg += " This one's on us — no credits charged, and your next analysis is free."

        return msg

    return ""
