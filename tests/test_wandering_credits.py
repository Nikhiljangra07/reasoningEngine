"""
Tests for the credit ledger underneath Wandering Room.

Covers:
  - tokens_to_credits rounding & boundary math
  - CreditService basic operations (balance, grant, reserve, commit, release)
  - reserve-commit symmetry: net spend = used; refunded = budgeted - used
  - reserve-release: full refund, no ledger entry written
  - Insufficient-balance raises before any state mutation
  - Concurrent reserves on the same user don't overspend
  - Different users run in parallel (no cross-user deadlock)
  - grant_starter is idempotent (fires once per user)
  - Reservation closure: double-commit / double-release raises ValueError
  - find_reservation_for_session / is_open helpers
  - transactions ordering (newest first), warning_level thresholds
  - account_summary payload shape

Pure-logic tests against InMemoryCreditStore. No network, no Neo4j.

Run:
  PYTHONPATH=. .venv/bin/python tests/test_wandering_credits.py
"""

from __future__ import annotations

import asyncio
import sys
import time

from src.wandering.credits import (
    CommitResult,
    CreditAccount,
    CreditService,
    CreditTx,
    CreditTxKind,
    DANGER_THRESHOLD,
    DEFAULT_PACKS,
    FREE_STARTER_CREDITS,
    InMemoryCreditStore,
    InsufficientCredits,
    Reservation,
    SUBSCRIPTION_MONTHLY_GRANT,
    TOKENS_PER_CREDIT,
    WARNING_THRESHOLD,
    tokens_to_credits,
)


# ─── Mini test harness ─────────────────────────────────────────────────

PASSED = 0
FAILED = 0
ERRORS: list[tuple[str, str]] = []


def test(name: str):
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    global PASSED, FAILED
    name = getattr(fn, "_test_name", fn.__name__)
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILED += 1
        ERRORS.append((name, f"FAIL: {e}"))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, f"ERROR: {type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


# ─── Fixture builders ─────────────────────────────────────────────────


def _fresh_service() -> CreditService:
    return CreditService(InMemoryCreditStore())


# ─── tokens_to_credits ────────────────────────────────────────────────


@test("MATH.1 tokens_to_credits rounds up to nearest credit")
def test_tokens_round_up():
    assert tokens_to_credits(0) == 0
    assert tokens_to_credits(1) == 1                 # 1 token → 1 credit (round up)
    assert tokens_to_credits(9_999) == 1
    assert tokens_to_credits(10_000) == 1            # exactly one credit
    assert tokens_to_credits(10_001) == 2            # boundary crossing
    assert tokens_to_credits(60_000) == 6            # Triple Pendulum default
    assert tokens_to_credits(150_000) == 15          # Multi Pendulum default
    assert tokens_to_credits(400_000) == 40          # Absolute Chaos default


@test("MATH.2 tokens_to_credits clamps non-positive to zero")
def test_tokens_non_positive():
    assert tokens_to_credits(-1) == 0
    assert tokens_to_credits(-100_000) == 0


# ─── Pack table sanity ────────────────────────────────────────────────


@test("PACKS.1 default pack table matches the credit economy design")
def test_default_packs_shape():
    by_slug = {p.slug: p for p in DEFAULT_PACKS}
    assert "starter" in by_slug
    assert "builder" in by_slug
    assert "researcher" in by_slug
    starter = by_slug["starter"]
    builder = by_slug["builder"]
    researcher = by_slug["researcher"]
    # Discount curve: $/credit must drop as packs get bigger.
    assert starter.cents_per_credit > builder.cents_per_credit
    assert builder.cents_per_credit > researcher.cents_per_credit
    # Builder ≈ 9% cheaper than starter; researcher ≈ 17% cheaper.
    starter_rate = starter.cents_per_credit
    assert builder.cents_per_credit    < starter_rate * 0.93
    assert researcher.cents_per_credit < starter_rate * 0.86


# ─── Empty state ──────────────────────────────────────────────────────


@test("INIT.1 balance for unknown user is 0")
async def test_unknown_user_balance():
    svc = _fresh_service()
    bal = await svc.balance("ghost-user")
    assert bal == 0


@test("INIT.2 has_account is false until grant_starter fires")
async def test_no_account_until_grant():
    svc = _fresh_service()
    assert (await svc.has_account("alice")) is False
    await svc.grant_starter("alice")
    assert (await svc.has_account("alice")) is True


# ─── Starter grant ────────────────────────────────────────────────────


@test("STARTER.1 grant_starter gives FREE_STARTER_CREDITS to new account")
async def test_starter_grant_amount():
    svc = _fresh_service()
    tx = await svc.grant_starter("alice")
    assert tx is not None
    assert tx.kind == CreditTxKind.STARTER_GRANT
    assert tx.delta == FREE_STARTER_CREDITS
    assert await svc.balance("alice") == FREE_STARTER_CREDITS


@test("STARTER.2 grant_starter is idempotent (no double-grant)")
async def test_starter_grant_idempotent():
    svc = _fresh_service()
    first  = await svc.grant_starter("alice")
    second = await svc.grant_starter("alice")
    third  = await svc.grant_starter("alice")
    assert first is not None
    assert second is None
    assert third  is None
    # Balance stayed at the starter amount.
    assert await svc.balance("alice") == FREE_STARTER_CREDITS


@test("STARTER.3 grant_starter does NOT fire if user already has any account")
async def test_starter_no_fire_if_account_open():
    svc = _fresh_service()
    # Open the account via a TOPUP, NOT a starter grant.
    await svc.grant(
        user_id="bob", amount=20,
        kind=CreditTxKind.TOPUP, note="manual seed",
    )
    # Now starter_grant should refuse — account exists.
    tx = await svc.grant_starter("bob")
    assert tx is None
    assert await svc.balance("bob") == 20


# ─── Generic grant ────────────────────────────────────────────────────


@test("GRANT.1 TOPUP increments balance and lifetime_purchased")
async def test_grant_topup():
    svc = _fresh_service()
    await svc.grant("carol", 50, CreditTxKind.TOPUP, ref_id="stripe_abc")
    summary = await svc.account_summary("carol")
    assert summary["balance"] == 50
    assert summary["lifetime_purchased"] == 50
    assert summary["lifetime_granted"]   == 0


@test("GRANT.2 SUBSCRIPTION_GRANT increments lifetime_granted (not purchased)")
async def test_grant_subscription():
    svc = _fresh_service()
    await svc.grant(
        "dave", SUBSCRIPTION_MONTHLY_GRANT,
        CreditTxKind.SUBSCRIPTION_GRANT, note="Pro monthly",
    )
    summary = await svc.account_summary("dave")
    assert summary["balance"] == SUBSCRIPTION_MONTHLY_GRANT
    assert summary["lifetime_purchased"] == 0
    assert summary["lifetime_granted"]   == SUBSCRIPTION_MONTHLY_GRANT


@test("GRANT.3 grant() rejects non-grant kinds")
async def test_grant_rejects_charge():
    svc = _fresh_service()
    raised = False
    try:
        await svc.grant("e", 5, CreditTxKind.CHARGE)
    except ValueError:
        raised = True
    assert raised, "expected ValueError on non-grant kind"


@test("GRANT.4 grant() rejects non-positive amounts")
async def test_grant_rejects_zero():
    svc = _fresh_service()
    raised_zero = raised_neg = False
    try:
        await svc.grant("e", 0, CreditTxKind.TOPUP)
    except ValueError:
        raised_zero = True
    try:
        await svc.grant("e", -3, CreditTxKind.TOPUP)
    except ValueError:
        raised_neg = True
    assert raised_zero and raised_neg


# ─── Reserve / commit / release ───────────────────────────────────────


@test("RESERVE.1 reserve deducts from spendable balance via hold")
async def test_reserve_holds():
    svc = _fresh_service()
    await svc.grant("frank", 50, CreditTxKind.TOPUP)
    res = await svc.reserve("frank", 15, ref_id="wsess-1")
    assert res.held_credits == 15
    # Spendable is now 35; persisted still 50.
    assert await svc.balance("frank") == 35
    assert await svc.held("frank")    == 15


@test("RESERVE.2 reserve with insufficient credits raises InsufficientCredits")
async def test_reserve_insufficient():
    svc = _fresh_service()
    await svc.grant("ghost", 10, CreditTxKind.TOPUP)
    raised = False
    try:
        await svc.reserve("ghost", 11, ref_id="w-x")
    except InsufficientCredits as ic:
        raised = True
        assert ic.balance == 10
        assert ic.needed  == 11
    assert raised
    # Balance unchanged after failed reserve.
    assert await svc.balance("ghost") == 10
    assert await svc.held("ghost")    == 0


@test("RESERVE.3 reserve sums against active holds (no overspend)")
async def test_reserve_against_holds():
    svc = _fresh_service()
    await svc.grant("hera", 30, CreditTxKind.TOPUP)
    await svc.reserve("hera", 20, ref_id="w-1")
    # Spendable is now 10. Asking for 15 must fail.
    raised = False
    try:
        await svc.reserve("hera", 15, ref_id="w-2")
    except InsufficientCredits:
        raised = True
    assert raised
    # But 10 is still allowed (exact match).
    res2 = await svc.reserve("hera", 10, ref_id="w-2")
    assert res2.held_credits == 10
    assert await svc.balance("hera") == 0
    assert await svc.held("hera")    == 30


@test("COMMIT.1 commit charges actual tokens (rounded up) and refunds rest")
async def test_commit_charges_and_refunds():
    svc = _fresh_service()
    await svc.grant("ian", 50, CreditTxKind.TOPUP)
    res = await svc.reserve("ian", 15, ref_id="w-commit")
    # Wander actually spent 120,000 tokens = 12 credits.
    result = await svc.commit(res.reservation_id, actual_tokens=120_000)
    assert isinstance(result, CommitResult)
    assert result.budgeted == 15
    assert result.used     == 12
    assert result.refunded == 3
    assert result.balance_after == 50 - 12  # 38
    # And the spendable balance reflects that.
    assert await svc.balance("ian") == 38
    assert await svc.held("ian")    == 0


@test("COMMIT.2 commit caps used at held when wander overshot budget")
async def test_commit_caps_at_held():
    svc = _fresh_service()
    await svc.grant("jen", 20, CreditTxKind.TOPUP)
    res = await svc.reserve("jen", 6, ref_id="w-overshoot")
    # Wander overshot: spent 100k tokens = 10 credits, but only 6 held.
    result = await svc.commit(res.reservation_id, actual_tokens=100_000)
    assert result.used     == 6     # capped
    assert result.refunded == 0
    assert result.balance_after == 14
    assert await svc.balance("jen") == 14


@test("COMMIT.3 commit with zero tokens releases everything")
async def test_commit_zero_tokens():
    svc = _fresh_service()
    await svc.grant("kim", 20, CreditTxKind.TOPUP)
    res = await svc.reserve("kim", 10, ref_id="w-zero")
    result = await svc.commit(res.reservation_id, actual_tokens=0)
    assert result.used     == 0
    assert result.refunded == 10
    assert result.balance_after == 20
    assert await svc.balance("kim") == 20


@test("RELEASE.1 release returns full hold and writes NO ledger entry")
async def test_release_full_refund():
    svc = _fresh_service()
    await svc.grant("leo", 30, CreditTxKind.TOPUP)
    pre_txs = await svc.transactions("leo")
    res = await svc.reserve("leo", 15, ref_id="w-cancel")
    assert await svc.balance("leo") == 15  # held
    result = await svc.release(res.reservation_id)
    assert result.budgeted == 15
    assert result.used     == 0
    assert result.refunded == 15
    assert result.balance_after == 30
    # The ledger should still only have the original TOPUP — no
    # release/reserve entries cluttering the audit trail.
    post_txs = await svc.transactions("leo")
    assert len(post_txs) == len(pre_txs)


@test("CLOSE.1 double-commit raises ValueError")
async def test_double_commit():
    svc = _fresh_service()
    await svc.grant("mia", 20, CreditTxKind.TOPUP)
    res = await svc.reserve("mia", 10, ref_id="w-1")
    await svc.commit(res.reservation_id, actual_tokens=50_000)
    raised = False
    try:
        await svc.commit(res.reservation_id, actual_tokens=50_000)
    except ValueError:
        raised = True
    assert raised


@test("CLOSE.2 commit-then-release raises ValueError")
async def test_commit_then_release():
    svc = _fresh_service()
    await svc.grant("nia", 20, CreditTxKind.TOPUP)
    res = await svc.reserve("nia", 10, ref_id="w-1")
    await svc.commit(res.reservation_id, actual_tokens=50_000)
    raised = False
    try:
        await svc.release(res.reservation_id)
    except ValueError:
        raised = True
    assert raised


@test("CLOSE.3 release-then-commit raises ValueError")
async def test_release_then_commit():
    svc = _fresh_service()
    await svc.grant("oli", 20, CreditTxKind.TOPUP)
    res = await svc.reserve("oli", 10, ref_id="w-1")
    await svc.release(res.reservation_id)
    raised = False
    try:
        await svc.commit(res.reservation_id, actual_tokens=10_000)
    except ValueError:
        raised = True
    assert raised


@test("CLOSE.4 is_open reflects reservation state")
async def test_is_open():
    svc = _fresh_service()
    await svc.grant("pia", 20, CreditTxKind.TOPUP)
    res = await svc.reserve("pia", 5, ref_id="w-1")
    assert svc.is_open(res.reservation_id) is True
    await svc.commit(res.reservation_id, actual_tokens=10_000)
    assert svc.is_open(res.reservation_id) is False
    assert svc.is_open("nonexistent-id")   is False


@test("CLOSE.5 find_reservation_for_session locates open holds")
async def test_find_for_session():
    svc = _fresh_service()
    await svc.grant("quinn", 30, CreditTxKind.TOPUP)
    res = await svc.reserve("quinn", 10, ref_id="wsess-target")
    found = svc.find_reservation_for_session("wsess-target")
    assert found is not None
    assert found.reservation_id == res.reservation_id
    # After commit, no longer found.
    await svc.commit(res.reservation_id, actual_tokens=10_000)
    assert svc.find_reservation_for_session("wsess-target") is None


# ─── Concurrency ──────────────────────────────────────────────────────


@test("CONC.1 concurrent reserves on same user serialize (no overspend)")
async def test_concurrent_reserves_same_user():
    svc = _fresh_service()
    await svc.grant("racer", 50, CreditTxKind.TOPUP)

    # Three concurrent reserves of 20 each. Only two can succeed
    # (40 total), the third must raise InsufficientCredits.
    async def try_reserve(idx: int):
        try:
            res = await svc.reserve("racer", 20, ref_id=f"w-{idx}")
            return ("ok", res)
        except InsufficientCredits as ic:
            return ("nope", ic.balance)

    results = await asyncio.gather(
        try_reserve(1), try_reserve(2), try_reserve(3),
    )
    successes = [r for r in results if r[0] == "ok"]
    failures  = [r for r in results if r[0] == "nope"]
    assert len(successes) == 2, f"expected 2 successes, got {len(successes)}"
    assert len(failures)  == 1
    # Balance after: 50 - 40 = 10 spendable, 40 held.
    assert await svc.balance("racer") == 10
    assert await svc.held("racer")    == 40


@test("CONC.2 reserves on different users run in parallel without interference")
async def test_concurrent_different_users():
    svc = _fresh_service()
    # Three users each with their own balance.
    await svc.grant("u1", 30, CreditTxKind.TOPUP)
    await svc.grant("u2", 30, CreditTxKind.TOPUP)
    await svc.grant("u3", 30, CreditTxKind.TOPUP)

    async def reserve_for(user: str):
        return await svc.reserve(user, 20, ref_id=f"w-{user}")

    results = await asyncio.gather(
        reserve_for("u1"), reserve_for("u2"), reserve_for("u3"),
    )
    assert all(r is not None for r in results)
    # Each user has 10 spendable, 20 held — they didn't drain each other.
    for u in ("u1", "u2", "u3"):
        assert await svc.balance(u) == 10
        assert await svc.held(u)    == 20


# ─── Ledger / summary ────────────────────────────────────────────────


@test("TX.1 transactions returns newest first")
async def test_tx_ordering():
    svc = _fresh_service()
    await svc.grant("sam", 10, CreditTxKind.TOPUP,    note="first")
    await asyncio.sleep(0.01)
    await svc.grant("sam",  5, CreditTxKind.ADMIN_GRANT, note="second")
    await asyncio.sleep(0.01)
    res = await svc.reserve("sam", 8, ref_id="w-x")
    await svc.commit(res.reservation_id, actual_tokens=70_000)  # 7 credits

    txs = await svc.transactions("sam", limit=10)
    # The CHARGE for the wander should be on top; STARTER/SUBSCRIPTION/TOPUP
    # entries below. Only grants and charges appear (no reserve/release noise).
    assert txs[0].kind == CreditTxKind.CHARGE
    kinds = [t.kind for t in txs]
    assert CreditTxKind.ADMIN_GRANT in kinds
    assert CreditTxKind.TOPUP       in kinds
    # No reserve/release pollution in the ledger.
    for t in txs:
        assert t.kind in {
            CreditTxKind.TOPUP,
            CreditTxKind.ADMIN_GRANT,
            CreditTxKind.CHARGE,
        }


@test("TX.2 balance_after on the CHARGE entry matches final spendable")
async def test_balance_after_on_charge():
    svc = _fresh_service()
    await svc.grant("tia", 30, CreditTxKind.TOPUP)
    res = await svc.reserve("tia", 15, ref_id="w-tia")
    await svc.commit(res.reservation_id, actual_tokens=120_000)  # 12 credits
    txs = await svc.transactions("tia")
    charge = next(t for t in txs if t.kind == CreditTxKind.CHARGE)
    assert charge.delta == -12
    assert charge.balance_after == 18  # 30 - 12


@test("SUMMARY.1 account_summary exposes the chip-render shape")
async def test_summary_shape():
    svc = _fresh_service()
    await svc.grant("uma", 40, CreditTxKind.TOPUP)
    res = await svc.reserve("uma", 15, ref_id="w-uma")
    summary = await svc.account_summary("uma")
    expected_keys = {
        "balance", "held", "persisted_balance",
        "warning_level", "warning_threshold", "danger_threshold",
        "lifetime_purchased", "lifetime_granted", "lifetime_spent",
        "tokens_per_credit", "has_account",
    }
    assert set(summary.keys()) >= expected_keys, (
        f"missing keys: {expected_keys - set(summary.keys())}"
    )
    assert summary["balance"]            == 25   # 40 - 15 held
    assert summary["held"]               == 15
    assert summary["persisted_balance"]  == 40
    assert summary["lifetime_purchased"] == 40
    assert summary["tokens_per_credit"]  == TOKENS_PER_CREDIT
    assert summary["has_account"]        is True


@test("WARN.1 warning_level thresholds map correctly")
async def test_warning_levels():
    svc = _fresh_service()
    # neutral: above WARNING_THRESHOLD
    assert svc.warning_level(WARNING_THRESHOLD + 1) == "neutral"
    # warning: at or below WARNING_THRESHOLD, above DANGER_THRESHOLD
    assert svc.warning_level(WARNING_THRESHOLD)     == "warning"
    assert svc.warning_level(DANGER_THRESHOLD + 1)  == "warning"
    # danger: at or below DANGER_THRESHOLD
    assert svc.warning_level(DANGER_THRESHOLD)      == "danger"
    assert svc.warning_level(0)                     == "danger"


@test("WARN.2 warning_level integrates with account_summary")
async def test_warning_level_in_summary():
    svc = _fresh_service()
    await svc.grant("vee", 100, CreditTxKind.TOPUP)
    s1 = await svc.account_summary("vee")
    assert s1["warning_level"] == "neutral"
    # Burn down to 12 (between thresholds: warning).
    res1 = await svc.reserve("vee", 88, ref_id="w-vee-1")
    await svc.commit(res1.reservation_id, actual_tokens=880_000)
    s2 = await svc.account_summary("vee")
    assert s2["balance"] == 12
    assert s2["warning_level"] == "warning"
    # Burn down to 5 (danger zone).
    res2 = await svc.reserve("vee", 7, ref_id="w-vee-2")
    await svc.commit(res2.reservation_id, actual_tokens=70_000)
    s3 = await svc.account_summary("vee")
    assert s3["balance"] == 5
    assert s3["warning_level"] == "danger"


# ─── End-to-end happy path ────────────────────────────────────────────


@test("E2E.1 new user → starter grant → wander → cancel → new wander → complete")
async def test_e2e_lifecycle():
    """The full first-day-of-use story:
       - User shows up; we grant 10 starter credits.
       - They try a Multi Pendulum wander (15 credits) — denied, top up.
       - They buy a Starter pack (50 credits).
       - They run a LOW wander (6 credits), cancel after 2 credits used.
       - They run a MED wander (15 credits) to completion at 12 credits used.
       - Final balance reconciles."""
    svc = _fresh_service()
    user = "first-day-user"

    # 1. Starter grant.
    tx = await svc.grant_starter(user)
    assert tx is not None
    assert await svc.balance(user) == FREE_STARTER_CREDITS  # 10

    # 2. Try to run a MED wander (15 credits) — should fail at 10 starter.
    raised = False
    try:
        await svc.reserve(user, 15, ref_id="wsess-attempt-1")
    except InsufficientCredits as ic:
        raised = True
        assert ic.balance == FREE_STARTER_CREDITS  # 10
        assert ic.needed  == 15
    assert raised

    # 3. Top up with Starter pack: +50.
    await svc.grant(user, 50, CreditTxKind.TOPUP, ref_id="stripe_evt_1")
    assert await svc.balance(user) == 60

    # 4. Run a LOW wander (6 credits). Test the cancel-before-work path
    #    via release() — full refund, no ledger entry written.
    low_res = await svc.reserve(user, 6, ref_id="wsess-low")
    assert await svc.balance(user) == 54
    low_result = await svc.release(low_res.reservation_id)
    assert low_result.refunded == 6
    assert await svc.balance(user) == 60

    # 4b. Realistic in-flight cancel: reserve, commit at partial tokens.
    in_flight = await svc.reserve(user, 6, ref_id="wsess-low-2")
    in_flight_result = await svc.commit(
        in_flight.reservation_id, actual_tokens=18_000,  # 2 credits
    )
    assert in_flight_result.used     == 2
    assert in_flight_result.refunded == 4
    assert await svc.balance(user) == 58

    # 5. Run a MED wander (15 credits) to completion at 120k tokens (12 credits).
    med_res = await svc.reserve(user, 15, ref_id="wsess-med")
    assert await svc.balance(user) == 43
    med_result = await svc.commit(
        med_res.reservation_id, actual_tokens=120_000,
    )
    assert med_result.used     == 12
    assert med_result.refunded == 3
    assert await svc.balance(user) == 46

    # 6. Sanity check the account summary at the end.
    summary = await svc.account_summary(user)
    assert summary["balance"]            == 46
    assert summary["lifetime_purchased"] == 50
    assert summary["lifetime_granted"]   == FREE_STARTER_CREDITS  # 10
    assert summary["lifetime_spent"]     == 2 + 12  # both wanders combined


# ─── Run all ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Collect all module-level functions decorated with @test.
    tests = [
        v for v in globals().values()
        if callable(v) and hasattr(v, "_test_name")
    ]
    print(f"\nRunning {len(tests)} credit ledger tests...\n")
    for t in tests:
        run_test(t)
    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED:
        print("\nFailures:")
        for name, err in ERRORS:
            print(f"  {name}: {err}")
        sys.exit(1)
    sys.exit(0)
