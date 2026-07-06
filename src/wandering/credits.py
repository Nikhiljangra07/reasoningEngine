"""
Credit ledger — the money layer that sits underneath Wandering Room.

Two-tier model for Constellax:

  1. Subscription (Constellax Pro) — gates Thinking Map mode, includes a
     monthly Wandering credit grant as a sampler.

  2. Pay-as-you-go credit packs — purchased separately when a user is
     actively wandering. The infrastructure cost of running 10 agents
     over an hour (Absolute Chaos) is real; bundling it into a flat
     subscription would either underprice heavy users or overprice
     casual ones.

Architecture (today): all accounting lives on our side. The Stripe
side (top-up checkout, webhook) is intentionally NOT wired yet — the
service is designed so the future Stripe webhook handler is a thin
call into `grant(user_id, amount, kind=TOPUP, ref_id=stripe_session_id)`.

Bookkeeping model
=================
Reserve-and-commit, with reservations held purely in process memory:

  - Persisted `balance` is the authoritative spendable amount. It
    moves ONLY on real settlements (grants, charges).
  - In-flight wanders hold credits via in-memory reservations.
    `available_balance` = `persisted_balance` - sum(active holds).
  - On wander completion, we write ONE ledger entry: CHARGE for the
    actual credits used (rounded up from tokens-spent).
  - On wander cancellation, NO ledger entry is written — the hold
    simply evaporates, leaving balance untouched.

Why this model:
  - The user's ledger reads naturally:
      Topup +50      Stripe purchase
      Charge -12     wander abc
      Topup +50      Stripe purchase
      Charge -40     wander xyz
    No noise from reserve/release pairs. Easy to render and reconcile.
  - Cancelled wanders don't pollute the audit trail with no-op entries.
  - Server restarts during wanders drop in-memory holds. That's OK
    because wanders themselves don't survive restart (the worker dies
    and jobs.py transitions them to failed on next startup). The held
    credits return naturally because no CHARGE was ever written.

Per Law 4: writes are scoped to the CREDIT namespace.

Operations
==========
  balance(user_id)                    -> int       (spendable, holds subtracted)
  reserve(user_id, amount, ref_id)    -> Reservation  (raises InsufficientCredits)
  commit(reservation, actual_tokens)  -> CommitResult (writes CHARGE entry)
  release(reservation)                -> None         (drops the hold, no entry)
  grant(user_id, amount, kind, ref)   -> CreditTx     (writes additive entry)
  grant_starter(user_id)              -> CreditTx?    (idempotent first-time grant)
  transactions(user_id, limit)        -> list[CreditTx]
  has_account(user_id)                -> bool

Invariants
==========
  - Balance is never negative. reserve() raises before deducting.
  - sum(deltas in ledger) == account.balance, always.
  - All operations are atomic per-user (asyncio.Lock).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


log = logging.getLogger("constellax.credits")


# ---------------------------------------------------------------------------
# Tunables — single source of truth for the credit economy
# ---------------------------------------------------------------------------

# 1 credit = N tokens of agent work. The conversion is the single dial
# that decides "how many wanders does a starter pack buy."
TOKENS_PER_CREDIT: int = 10_000

# Free starter grant for brand-new accounts. 10 credits = one full
# Triple Pendulum wander (6 cr) plus 4 left over, OR two near-complete
# attempts. Generous enough that the user gets a real taste of the
# dossier before deciding to top up, without giving away a full
# Multi Pendulum (15 cr) for free.
FREE_STARTER_CREDITS: int = 10

# Warning thresholds for the UI credit chip. Maps to mode costs:
#   15 = below the cost of one Multi Pendulum wander (15 credits).
#   10 = approaching the bottom; one Triple Pendulum wander left.
# Below the user's actual budget at /session time we hard-block (402).
WARNING_THRESHOLD: int = 15
DANGER_THRESHOLD:  int = 10

# Subscription monthly grant (Constellax Pro perk). One Multi Pendulum
# wander OR two Triple Pendulum wanders per month, or save up.
SUBSCRIPTION_MONTHLY_GRANT: int = 15


def tokens_to_credits(tokens: int) -> int:
    """Round-up tokens → credits. Always non-negative. Round up so we
    never under-charge ourselves (the user gets at most 1 token short
    of a full credit's worth as a freebie before we count it)."""
    if tokens <= 0:
        return 0
    return (tokens + TOKENS_PER_CREDIT - 1) // TOKENS_PER_CREDIT


@dataclass(frozen=True)
class CreditPack:
    """A purchasable bundle of credits. Pricing is USD cents to avoid
    float drift. Unused until Stripe plugs in; exposed via API so the
    frontend can render the topup modal even before billing is live."""
    slug:        str
    label:       str
    credits:     int
    price_cents: int

    @property
    def cents_per_credit(self) -> float:
        if self.credits <= 0:
            return 0.0
        return self.price_cents / self.credits

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug":             self.slug,
            "label":            self.label,
            "credits":          self.credits,
            "price_cents":      self.price_cents,
            "cents_per_credit": round(self.cents_per_credit, 2),
        }


# Pack tier table. Discount curve nudges bulk top-ups without being
# predatory: 9% off at Builder, 17% off at Researcher.
DEFAULT_PACKS: tuple[CreditPack, ...] = (
    CreditPack(slug="starter",    label="Starter",    credits=50,  price_cents=1000),  # $10.00
    CreditPack(slug="builder",    label="Builder",    credits=110, price_cents=2000),  # $20.00
    CreditPack(slug="researcher", label="Researcher", credits=300, price_cents=5000),  # $50.00
)


# ---------------------------------------------------------------------------
# Credit transaction types
# ---------------------------------------------------------------------------


class CreditTxKind(str, Enum):
    """The reason a ledger entry exists. The set of kinds is small on
    purpose — only real balance movements get entries."""

    # Additive (delta > 0)
    STARTER_GRANT      = "starter_grant"        # first-time grant
    TOPUP              = "topup"                # Stripe one-time purchase
    SUBSCRIPTION_GRANT = "subscription_grant"   # monthly Pro perk
    ADMIN_GRANT        = "admin_grant"          # support/test grant

    # Subtractive (delta < 0)
    CHARGE             = "charge"               # wander completed, settle


@dataclass(frozen=True)
class CreditTx:
    """One row in the ledger. Every persisted balance movement
    corresponds to exactly one CreditTx."""

    tx_id:         str
    user_id:       str
    kind:          CreditTxKind
    delta:         int       # signed
    balance_after: int       # cached for O(1) renders
    ts:            float
    ref_id:        str = ""  # session_id of wander, stripe_session_id, etc.
    note:          str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_id":         self.tx_id,
            "kind":          self.kind.value,
            "delta":         self.delta,
            "balance_after": self.balance_after,
            "ts":            self.ts,
            "ref_id":        self.ref_id,
            "note":          self.note,
        }


@dataclass
class CreditAccount:
    """Cached running totals. The ledger is the source of truth; this
    is the O(1) read surface."""

    user_id:             str
    balance:             int   = 0
    lifetime_purchased:  int   = 0   # from TOPUP only
    lifetime_granted:    int   = 0   # from STARTER/SUBSCRIPTION/ADMIN
    lifetime_spent:      int   = 0   # from CHARGE
    created_at:          float = 0.0


@dataclass
class Reservation:
    """An in-memory hold on credits for an in-flight wander. NOT
    persisted — lives only in CreditService._reservations.

    Lifecycle:
        reserve()  →  hold registered, persisted_balance untouched
        commit()   →  CHARGE entry written for actual usage; hold dropped
        release()  →  hold dropped silently; no entry

    Closed reservations cannot be committed or released again.
    """
    reservation_id: str
    user_id:        str
    held_credits:   int       # how many credits the user agreed to put at risk
    ref_id:         str       # session_id of the wander
    created_at:     float
    closed:         bool = False


@dataclass(frozen=True)
class CommitResult:
    """What commit() returns — the credit breakdown the UI shows the
    user after a wander completes or aborts.

    `refunded` is the credits that went back into the user's spendable
    balance (the unused portion of the hold). It's computed as
    `budgeted - used`.
    """
    budgeted:      int    # the originally-held credits
    used:          int    # what we actually charged (rounded up from tokens)
    refunded:      int    # held - used (never negative)
    balance_after: int    # the user's spendable balance after settlement


# ---------------------------------------------------------------------------
# Storage protocol
# ---------------------------------------------------------------------------


class CreditStore(Protocol):
    """All methods async. Failures NEVER raise — implementations log
    and return False/None like WanderingStore."""

    async def get_account(self, user_id: str) -> CreditAccount | None: ...
    async def upsert_account(self, account: CreditAccount) -> bool:    ...
    async def append_tx(self, tx: CreditTx) -> bool:                   ...
    async def list_transactions(
        self, user_id: str, limit: int = 20,
    ) -> list[CreditTx]: ...
    async def has_any_account(self, user_id: str) -> bool:             ...


# ---------------------------------------------------------------------------
# In-memory backend — dev + tests
# ---------------------------------------------------------------------------


class InMemoryCreditStore:
    """Dicts only. Lost on process restart. Dev/tests only."""

    def __init__(self) -> None:
        self._accounts: dict[str, CreditAccount] = {}
        self._txs:      dict[str, list[CreditTx]] = {}  # newest first

    async def get_account(self, user_id: str) -> CreditAccount | None:
        return self._accounts.get(user_id)

    async def upsert_account(self, account: CreditAccount) -> bool:
        # Defensive copy — store should not share mutable state with caller.
        self._accounts[account.user_id] = CreditAccount(
            user_id=account.user_id,
            balance=account.balance,
            lifetime_purchased=account.lifetime_purchased,
            lifetime_granted=account.lifetime_granted,
            lifetime_spent=account.lifetime_spent,
            created_at=account.created_at,
        )
        return True

    async def append_tx(self, tx: CreditTx) -> bool:
        self._txs.setdefault(tx.user_id, []).insert(0, tx)
        return True

    async def list_transactions(self, user_id: str, limit: int = 20) -> list[CreditTx]:
        if limit <= 0:
            return []
        return list(self._txs.get(user_id, []))[:limit]

    async def has_any_account(self, user_id: str) -> bool:
        return user_id in self._accounts


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------


CREDIT_SCHEMA_CYPHER = """
CREATE CONSTRAINT credit_account_user_id IF NOT EXISTS
  FOR (a:CreditAccount) REQUIRE a.user_id IS UNIQUE;

CREATE INDEX credit_tx_user_id IF NOT EXISTS
  FOR (t:CreditTx) ON (t.user_id);

CREATE INDEX credit_tx_ts IF NOT EXISTS
  FOR (t:CreditTx) ON (t.ts);
""".strip()


class Neo4jCreditStore:
    """Neo4j-backed credit ledger.

    Schema:
      (User)-[:HOLDS]->(CreditAccount {user_id, balance, lifetime_*})
      (CreditAccount)-[:LEDGER]->(CreditTx {tx_id, kind, delta, ...})
    """

    def __init__(self, driver: Any, database: str = "neo4j") -> None:
        self._driver        = driver
        self._database      = database
        self._schema_inited = False

    async def init_schema(self) -> bool:
        if self._schema_inited:
            return True
        try:
            async with self._driver.session(database=self._database) as session:
                for stmt in CREDIT_SCHEMA_CYPHER.split(";\n"):
                    s = stmt.strip()
                    if s:
                        await session.run(s)
            self._schema_inited = True
            return True
        except Exception as e:
            log.warning("Neo4jCreditStore.init_schema failed: %s", e)
            return False

    async def get_account(self, user_id: str) -> CreditAccount | None:
        cypher = """
        MATCH (a:CreditAccount {user_id: $user_id})
        RETURN a.user_id            AS user_id,
               a.balance            AS balance,
               a.lifetime_purchased AS lifetime_purchased,
               a.lifetime_granted   AS lifetime_granted,
               a.lifetime_spent     AS lifetime_spent,
               a.created_at         AS created_at
        """
        try:
            async with self._driver.session(database=self._database) as sess:
                result = await sess.run(cypher, user_id=user_id)
                row = await result.single()
                if not row:
                    return None
                return CreditAccount(
                    user_id=str(row["user_id"]),
                    balance=int(row["balance"] or 0),
                    lifetime_purchased=int(row["lifetime_purchased"] or 0),
                    lifetime_granted=int(row["lifetime_granted"] or 0),
                    lifetime_spent=int(row["lifetime_spent"] or 0),
                    created_at=float(row["created_at"] or 0.0),
                )
        except Exception as e:
            log.warning("Neo4jCreditStore.get_account failed for %s: %s", user_id, e)
            return None

    async def upsert_account(self, account: CreditAccount) -> bool:
        cypher = """
        MERGE (a:CreditAccount {user_id: $user_id})
        ON CREATE SET a.created_at = $created_at
        SET a.balance            = $balance,
            a.lifetime_purchased = $lifetime_purchased,
            a.lifetime_granted   = $lifetime_granted,
            a.lifetime_spent     = $lifetime_spent
        WITH a
        MERGE (u:User {user_id: $user_id})
        MERGE (u)-[:HOLDS]->(a)
        """
        try:
            async with self._driver.session(database=self._database) as sess:
                await sess.run(
                    cypher,
                    user_id=account.user_id,
                    balance=account.balance,
                    lifetime_purchased=account.lifetime_purchased,
                    lifetime_granted=account.lifetime_granted,
                    lifetime_spent=account.lifetime_spent,
                    created_at=account.created_at,
                )
            return True
        except Exception as e:
            log.warning("Neo4jCreditStore.upsert_account failed for %s: %s",
                        account.user_id, e)
            return False

    async def append_tx(self, tx: CreditTx) -> bool:
        cypher = """
        MATCH (a:CreditAccount {user_id: $user_id})
        CREATE (t:CreditTx {
            tx_id:         $tx_id,
            user_id:       $user_id,
            kind:          $kind,
            delta:         $delta,
            balance_after: $balance_after,
            ts:            $ts,
            ref_id:        $ref_id,
            note:          $note
        })
        CREATE (a)-[:LEDGER]->(t)
        """
        try:
            async with self._driver.session(database=self._database) as sess:
                await sess.run(
                    cypher,
                    user_id=tx.user_id,
                    tx_id=tx.tx_id,
                    kind=tx.kind.value,
                    delta=tx.delta,
                    balance_after=tx.balance_after,
                    ts=tx.ts,
                    ref_id=tx.ref_id,
                    note=tx.note,
                )
            return True
        except Exception as e:
            log.warning("Neo4jCreditStore.append_tx failed for %s/%s: %s",
                        tx.user_id, tx.tx_id, e)
            return False

    async def list_transactions(self, user_id: str, limit: int = 20) -> list[CreditTx]:
        if limit <= 0:
            return []
        cypher = """
        MATCH (a:CreditAccount {user_id: $user_id})-[:LEDGER]->(t:CreditTx)
        RETURN t.tx_id         AS tx_id,
               t.kind          AS kind,
               t.delta         AS delta,
               t.balance_after AS balance_after,
               t.ts            AS ts,
               t.ref_id        AS ref_id,
               t.note          AS note
        ORDER BY t.ts DESC
        LIMIT $limit
        """
        try:
            async with self._driver.session(database=self._database) as sess:
                result = await sess.run(cypher, user_id=user_id, limit=limit)
                rows = await result.data()
        except Exception as e:
            log.warning("Neo4jCreditStore.list_transactions failed for %s: %s",
                        user_id, e)
            return []

        out: list[CreditTx] = []
        for r in rows:
            try:
                kind = CreditTxKind(r.get("kind", ""))
            except ValueError:
                continue
            out.append(CreditTx(
                tx_id=str(r.get("tx_id", "")),
                user_id=user_id,
                kind=kind,
                delta=int(r.get("delta", 0)),
                balance_after=int(r.get("balance_after", 0)),
                ts=float(r.get("ts", 0.0)),
                ref_id=str(r.get("ref_id", "") or ""),
                note=str(r.get("note", "") or ""),
            ))
        return out

    async def has_any_account(self, user_id: str) -> bool:
        cypher = "MATCH (a:CreditAccount {user_id: $user_id}) RETURN count(a) AS c"
        try:
            async with self._driver.session(database=self._database) as sess:
                result = await sess.run(cypher, user_id=user_id)
                row = await result.single()
                return bool(row and int(row["c"] or 0) > 0)
        except Exception as e:
            log.warning("Neo4jCreditStore.has_any_account failed for %s: %s",
                        user_id, e)
            return False


# ---------------------------------------------------------------------------
# CreditService — the business logic on top of the store
# ---------------------------------------------------------------------------


class InsufficientCredits(Exception):
    """Raised by reserve() when available balance < amount. Carries the
    gap so the API layer can render a useful 402 payload."""
    def __init__(self, balance: int, needed: int) -> None:
        super().__init__(f"insufficient_credits: balance={balance}, needed={needed}")
        self.balance = balance
        self.needed  = needed


class CreditService:
    """Single entry point for all credit operations.

    Per-user locking ensures concurrent reserve/commit/release on the
    same user serialize cleanly. Different users run in parallel.

    Reservations live in this process's memory; the ledger is durable.
    """

    def __init__(self, store: CreditStore) -> None:
        self._store        = store
        self._reservations: dict[str, Reservation] = {}
        self._locks:        dict[str, asyncio.Lock] = {}

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock

    def _active_holds_for(self, user_id: str) -> int:
        """Sum of in-flight reservations for this user. Excludes closed."""
        return sum(
            r.held_credits
            for r in self._reservations.values()
            if r.user_id == user_id and not r.closed
        )

    async def _persisted_balance(self, user_id: str) -> int:
        acc = await self._store.get_account(user_id)
        return acc.balance if acc is not None else 0

    async def _ensure_account(self, user_id: str) -> CreditAccount:
        """Fetch the account, creating an empty one if absent."""
        acc = await self._store.get_account(user_id)
        if acc is None:
            acc = CreditAccount(
                user_id=user_id,
                balance=0,
                lifetime_purchased=0,
                lifetime_granted=0,
                lifetime_spent=0,
                created_at=time.time(),
            )
            await self._store.upsert_account(acc)
        return acc

    async def _write_movement(
        self,
        user_id: str,
        kind: CreditTxKind,
        delta: int,
        ref_id: str = "",
        note:   str = "",
    ) -> CreditTx:
        """Apply a real balance movement: update CreditAccount and append
        a CreditTx. Caller MUST hold the per-user lock."""
        acc = await self._ensure_account(user_id)
        new_balance = acc.balance + delta
        if new_balance < 0:
            # Should never happen — callers must pre-check via reserve().
            raise RuntimeError(
                f"credit balance would go negative: user={user_id} "
                f"balance={acc.balance} delta={delta}"
            )
        acc.balance = new_balance
        if kind == CreditTxKind.TOPUP:
            acc.lifetime_purchased += abs(delta)
        elif kind in (
            CreditTxKind.STARTER_GRANT,
            CreditTxKind.SUBSCRIPTION_GRANT,
            CreditTxKind.ADMIN_GRANT,
        ):
            acc.lifetime_granted += abs(delta)
        elif kind == CreditTxKind.CHARGE:
            acc.lifetime_spent += abs(delta)

        await self._store.upsert_account(acc)

        tx = CreditTx(
            tx_id=str(uuid.uuid4()),
            user_id=user_id,
            kind=kind,
            delta=delta,
            balance_after=acc.balance,
            ts=time.time(),
            ref_id=ref_id,
            note=note,
        )
        await self._store.append_tx(tx)
        return tx

    # --- Public API ----------------------------------------------------

    async def balance(self, user_id: str) -> int:
        """Spendable balance — persisted balance minus active in-flight
        holds. This is what /credits/balance returns to the UI."""
        persisted = await self._persisted_balance(user_id)
        return max(0, persisted - self._active_holds_for(user_id))

    async def held(self, user_id: str) -> int:
        """Credits currently reserved against in-flight wanders."""
        return self._active_holds_for(user_id)

    def warning_level(self, balance: int) -> str:
        """UI hint: 'neutral' | 'warning' | 'danger'."""
        if balance <= DANGER_THRESHOLD:
            return "danger"
        if balance <= WARNING_THRESHOLD:
            return "warning"
        return "neutral"

    async def has_account(self, user_id: str) -> bool:
        """True if user has ever opened a credit account. Used to gate
        the starter grant so it fires at most once per user."""
        return await self._store.has_any_account(user_id)

    async def account_summary(self, user_id: str) -> dict[str, Any]:
        """Everything the UI needs to render the header chip + balance
        detail panel in one call."""
        acc = await self._store.get_account(user_id)
        held = self._active_holds_for(user_id)
        persisted = acc.balance if acc else 0
        spendable = max(0, persisted - held)
        return {
            "balance":            spendable,
            "held":               held,
            "persisted_balance":  persisted,
            "warning_level":      self.warning_level(spendable),
            "warning_threshold":  WARNING_THRESHOLD,
            "danger_threshold":   DANGER_THRESHOLD,
            "lifetime_purchased": acc.lifetime_purchased if acc else 0,
            "lifetime_granted":   acc.lifetime_granted   if acc else 0,
            "lifetime_spent":     acc.lifetime_spent     if acc else 0,
            "tokens_per_credit":  TOKENS_PER_CREDIT,
            "has_account":        acc is not None,
        }

    async def transactions(self, user_id: str, limit: int = 20) -> list[CreditTx]:
        return await self._store.list_transactions(user_id, limit=limit)

    async def grant_starter(self, user_id: str) -> CreditTx | None:
        """Idempotent: gives FREE_STARTER_CREDITS to brand-new accounts.
        Returns None if the user already has an account."""
        lock = self._lock_for(user_id)
        async with lock:
            if await self._store.has_any_account(user_id):
                return None
            return await self._write_movement(
                user_id=user_id,
                kind=CreditTxKind.STARTER_GRANT,
                delta=FREE_STARTER_CREDITS,
                ref_id="",
                note=f"Welcome to Constellax. {FREE_STARTER_CREDITS} credits to explore.",
            )

    async def grant(
        self,
        user_id: str,
        amount: int,
        kind: CreditTxKind,
        ref_id: str = "",
        note:   str = "",
    ) -> CreditTx:
        """Add credits to a user. Used by the Stripe webhook (TOPUP),
        subscription job (SUBSCRIPTION_GRANT), admin tool (ADMIN_GRANT),
        and grant_starter() (STARTER_GRANT, via this path internally)."""
        if kind not in (
            CreditTxKind.STARTER_GRANT,
            CreditTxKind.TOPUP,
            CreditTxKind.SUBSCRIPTION_GRANT,
            CreditTxKind.ADMIN_GRANT,
        ):
            raise ValueError(f"grant() called with non-grant kind: {kind}")
        if amount <= 0:
            raise ValueError(f"grant amount must be positive, got {amount}")

        lock = self._lock_for(user_id)
        async with lock:
            return await self._write_movement(
                user_id=user_id,
                kind=kind,
                delta=amount,
                ref_id=ref_id,
                note=note,
            )

    async def reserve(
        self,
        user_id: str,
        amount: int,
        ref_id: str,
        note: str = "",
    ) -> Reservation:
        """Atomically hold `amount` credits for the wander identified by
        `ref_id`. Raises InsufficientCredits if spendable balance < amount.

        Returns the Reservation handle. Caller MUST eventually call
        either commit() or release() on it.
        """
        if amount < 0:
            raise ValueError(f"reserve amount must be >= 0, got {amount}")

        lock = self._lock_for(user_id)
        async with lock:
            persisted = await self._persisted_balance(user_id)
            held      = self._active_holds_for(user_id)
            spendable = max(0, persisted - held)
            if spendable < amount:
                raise InsufficientCredits(balance=spendable, needed=amount)

            reservation = Reservation(
                reservation_id=str(uuid.uuid4()),
                user_id=user_id,
                held_credits=amount,
                ref_id=ref_id,
                created_at=time.time(),
                closed=False,
            )
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    async def commit(
        self,
        reservation_id: str,
        actual_tokens: int,
        note: str = "",
    ) -> CommitResult:
        """Convert a reservation into a final charge.

        Writes a CHARGE entry for tokens_to_credits(actual_tokens),
        capped at the held amount. The unused portion silently goes
        back to the user's spendable balance (no ledger entry — the
        hold simply evaporates).

        Idempotent guard: closed reservations raise ValueError.
        """
        reservation = self._reservations.get(reservation_id)
        if reservation is None:
            raise ValueError(f"reservation not found: {reservation_id}")
        if reservation.closed:
            raise ValueError(f"reservation already closed: {reservation_id}")

        lock = self._lock_for(reservation.user_id)
        async with lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None or reservation.closed:
                raise ValueError(f"reservation closed concurrently: {reservation_id}")

            budgeted = reservation.held_credits
            used     = min(tokens_to_credits(actual_tokens), budgeted)
            refunded = budgeted - used

            if used > 0:
                await self._write_movement(
                    user_id=reservation.user_id,
                    kind=CreditTxKind.CHARGE,
                    delta=-used,
                    ref_id=reservation.ref_id,
                    note=note or (
                        f"Wander {reservation.ref_id[:8]}: "
                        f"{actual_tokens:,} tokens"
                    ),
                )

            reservation.closed = True
            # We keep the closed reservation in the dict briefly so
            # debugging/inspection can see what happened. Long-running
            # processes should prune closed reservations periodically,
            # but a few thousand entries is harmless.

            spendable_after = await self.balance(reservation.user_id)
            return CommitResult(
                budgeted=budgeted,
                used=used,
                refunded=refunded,
                balance_after=spendable_after,
            )

    async def release(self, reservation_id: str) -> CommitResult:
        """Cancel a reservation outright — drop the hold without writing
        any ledger entry. Returns a CommitResult with used=0,
        refunded=budgeted for the cancel-confirmation UI.

        Idempotent guard: closed reservations raise ValueError. Callers
        that aren't sure whether commit() was already called should
        check `is_open()` first.
        """
        reservation = self._reservations.get(reservation_id)
        if reservation is None:
            raise ValueError(f"reservation not found: {reservation_id}")
        if reservation.closed:
            raise ValueError(f"reservation already closed: {reservation_id}")

        lock = self._lock_for(reservation.user_id)
        async with lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None or reservation.closed:
                raise ValueError(f"reservation closed concurrently: {reservation_id}")

            budgeted = reservation.held_credits
            reservation.closed = True
            spendable_after = await self.balance(reservation.user_id)
            return CommitResult(
                budgeted=budgeted,
                used=0,
                refunded=budgeted,
                balance_after=spendable_after,
            )

    def is_open(self, reservation_id: str) -> bool:
        """Cheap check — true if the reservation exists and is open.
        Used by the abort path to handle 'already committed by worker'
        races without raising."""
        r = self._reservations.get(reservation_id)
        return r is not None and not r.closed

    def reservation(self, reservation_id: str) -> Reservation | None:
        """Read-only access. Returns the (possibly closed) reservation,
        or None if unknown."""
        return self._reservations.get(reservation_id)

    def find_reservation_for_session(self, session_id: str) -> Reservation | None:
        """Look up the open reservation for a wander by its session_id.
        Used at /abort time — the abort endpoint knows session_id, not
        reservation_id."""
        for r in self._reservations.values():
            if r.ref_id == session_id and not r.closed:
                return r
        return None


# ---------------------------------------------------------------------------
# Factory — build the service from env
# ---------------------------------------------------------------------------

_CREDIT_SERVICE: CreditService | None = None


def build_credit_service_from_env() -> CreditService:
    """Build the singleton CreditService, picking InMemoryCreditStore or
    Neo4jCreditStore based on CONSTELLAX_DB_BACKEND. Matches the
    WanderingStore factory pattern."""
    import os
    backend = os.environ.get("CONSTELLAX_DB_BACKEND", "").strip().lower()
    if backend == "neo4j":
        try:
            from src.bridge.neo4j_backend import build_neo4j_driver_from_env
            result = build_neo4j_driver_from_env()
            if result is None:
                log.warning(
                    "CONSTELLAX_DB_BACKEND=neo4j but driver build returned None; "
                    "falling back to in-memory credits"
                )
                return CreditService(InMemoryCreditStore())
            driver, database = result
            store = Neo4jCreditStore(driver=driver, database=database)
            log.info("CreditService: Neo4j backend ready (database=%s)", database)
            return CreditService(store)
        except Exception as e:
            log.warning(
                "Neo4jCreditStore build failed (%s); falling back to in-memory",
                e,
            )
    return CreditService(InMemoryCreditStore())


def get_credit_service() -> CreditService:
    """Process-singleton accessor. Lazy. Same pattern as get_store()."""
    global _CREDIT_SERVICE
    if _CREDIT_SERVICE is None:
        _CREDIT_SERVICE = build_credit_service_from_env()
    return _CREDIT_SERVICE


def _set_credit_service(service: CreditService) -> None:
    """Test helper: inject a specific service."""
    global _CREDIT_SERVICE
    _CREDIT_SERVICE = service


__all__ = [
    "TOKENS_PER_CREDIT",
    "FREE_STARTER_CREDITS",
    "WARNING_THRESHOLD",
    "DANGER_THRESHOLD",
    "SUBSCRIPTION_MONTHLY_GRANT",
    "DEFAULT_PACKS",
    "tokens_to_credits",
    "CreditPack",
    "CreditTxKind",
    "CreditTx",
    "CreditAccount",
    "Reservation",
    "CommitResult",
    "CreditStore",
    "InMemoryCreditStore",
    "Neo4jCreditStore",
    "CreditService",
    "InsufficientCredits",
    "build_credit_service_from_env",
    "get_credit_service",
    "_set_credit_service",
]
