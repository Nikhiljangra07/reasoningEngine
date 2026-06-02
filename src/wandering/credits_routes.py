"""
API endpoints for the credit ledger.

Mounted at `/api/v2/credits`. Sibling to `/api/v2/wandering` — the
wandering routes call into CreditService for reserve/commit/release;
these routes are the user-facing read surface plus the admin/grant
paths.

Endpoints
---------
  GET  /balance
    response: { balance, held, persisted_balance, warning_level,
                warning_threshold, danger_threshold, lifetime_*,
                tokens_per_credit, has_account, starter_granted }
    purpose: header chip + balance detail panel. Idempotent.
             Fires grant_starter() the first time a user hits this
             endpoint, so the balance returned already includes the
             FREE_STARTER_CREDITS grant for brand-new users.

  GET  /transactions?limit=20
    response: { transactions: [...] }
    purpose: render the credits ledger view.

  GET  /packs
    response: { packs: [...], starter_credits, monthly_grant, tokens_per_credit }
    purpose: the topup modal. CTAs are stub until Stripe plugs in.

  POST /grant_starter
    response: { granted: bool, tx?: {...}, balance: int }
    purpose: explicit grant_starter trigger (mainly for tests / admin).
             /balance fires this implicitly on first read.

  POST /admin/grant
    body: { user_id, amount, kind, note? }
    response: { tx: {...}, balance: int }
    purpose: env-gated dev tool for me to credit test users. Requires
             CONSTELLAX_ADMIN_TOKEN in the X-Admin-Token header.

  POST /topup (stub)
    response: 501 { error: "billing_not_ready", packs: [...] }
    purpose: placeholder until Stripe is wired. Returns the pack list
             so the frontend has something to render in the modal.

Auth
----
Same dual-mode as wandering: authenticated requests get verified user_id
from Supabase JWT; guest requests pass body.user_id.

Per Law 4: writes scoped to the CREDIT namespace.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.auth.supabase_auth import get_effective_user_id
from src.wandering.credits import (
    CreditService,
    CreditTxKind,
    DEFAULT_PACKS,
    FREE_STARTER_CREDITS,
    SUBSCRIPTION_MONTHLY_GRANT,
    TOKENS_PER_CREDIT,
    get_credit_service,
)


log = logging.getLogger("constellax.credits.routes")


def _service() -> CreditService:
    """Indirection so tests can swap in a service with `_set_credit_service`."""
    return get_credit_service()


def _resolve_user_id(req: Request, body: dict[str, Any] | None = None) -> str:
    """Extract the effective user_id. Supabase JWT wins; falls back to
    body.user_id for guest paths. Empty string is permitted and treated
    as a guest with no persistent account — credit calls become no-ops
    for that caller."""
    body_user = ""
    if body is not None:
        body_user = str(body.get("user_id", "") or "").strip()
    return get_effective_user_id(req, body_user) or ""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Build the credits APIRouter. Mount via:
        app.include_router(get_router(), prefix='/api/v2/credits')."""
    router = APIRouter()

    @router.get("/balance")
    async def get_balance(request: Request) -> JSONResponse:
        """Return the user's credit balance. Fires grant_starter() for
        brand-new users so the response already reflects the starter
        grant on first read."""
        user_id = _resolve_user_id(request)
        if not user_id:
            # Guest with no identity — return a stub so the UI can
            # gracefully render "sign in to track credits" state without
            # erroring out.
            return JSONResponse(content={
                "balance":            0,
                "held":               0,
                "persisted_balance":  0,
                "warning_level":      "neutral",
                "warning_threshold":  0,
                "danger_threshold":   0,
                "lifetime_purchased": 0,
                "lifetime_granted":   0,
                "lifetime_spent":     0,
                "tokens_per_credit":  TOKENS_PER_CREDIT,
                "has_account":        False,
                "starter_granted":    False,
                "guest":              True,
            })

        svc = _service()
        starter_granted = False
        # Auto-fire starter grant on first balance read so new users
        # see credits immediately. Idempotent — only fires once per user.
        if not await svc.has_account(user_id):
            try:
                tx = await svc.grant_starter(user_id)
                if tx is not None:
                    starter_granted = True
            except Exception as e:
                log.warning("auto starter grant failed for %s: %s", user_id, e)

        summary = await svc.account_summary(user_id)
        summary["starter_granted"] = starter_granted
        summary["guest"] = False
        return JSONResponse(content=summary)

    @router.get("/transactions")
    async def list_transactions(request: Request) -> JSONResponse:
        """Return the user's recent credit ledger entries."""
        user_id = _resolve_user_id(request)
        if not user_id:
            return JSONResponse(content={"transactions": [], "guest": True})

        limit_raw = request.query_params.get("limit", "20")
        try:
            limit = max(1, min(200, int(limit_raw)))
        except (TypeError, ValueError):
            limit = 20

        svc = _service()
        txs = await svc.transactions(user_id, limit=limit)
        return JSONResponse(content={
            "transactions": [t.to_dict() for t in txs],
        })

    @router.get("/packs")
    async def list_packs(request: Request) -> JSONResponse:
        """Return the pack tier table. Surfaces the same numbers the
        topup modal renders — single source of truth."""
        return JSONResponse(content={
            "packs":              [p.to_dict() for p in DEFAULT_PACKS],
            "starter_credits":    FREE_STARTER_CREDITS,
            "monthly_grant":      SUBSCRIPTION_MONTHLY_GRANT,
            "tokens_per_credit":  TOKENS_PER_CREDIT,
            "billing_ready":      False,
            "billing_status_msg": "Top-up billing coming soon. Reach out for early access credits.",
        })

    @router.post("/grant_starter")
    async def grant_starter(request: Request) -> JSONResponse:
        """Explicit trigger for the starter grant. Idempotent — returns
        granted=false if the user already has an account."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        user_id = _resolve_user_id(request, body)
        if not user_id:
            raise HTTPException(status_code=401, detail="user_id_required")

        svc = _service()
        tx = await svc.grant_starter(user_id)
        balance = await svc.balance(user_id)
        if tx is None:
            return JSONResponse(content={
                "granted": False,
                "reason":  "already_granted",
                "balance": balance,
            })
        return JSONResponse(content={
            "granted": True,
            "tx":      tx.to_dict(),
            "balance": balance,
        })

    @router.post("/admin/grant")
    async def admin_grant(request: Request) -> JSONResponse:
        """Env-gated admin tool. Requires X-Admin-Token matching the
        CONSTELLAX_ADMIN_TOKEN env var. If the env var is not set the
        endpoint is permanently locked."""
        admin_token = os.environ.get("CONSTELLAX_ADMIN_TOKEN", "").strip()
        if not admin_token:
            raise HTTPException(
                status_code=403,
                detail="admin_grant_disabled",
            )
        provided = request.headers.get("X-Admin-Token", "").strip()
        if not provided or provided != admin_token:
            # Constant-time-ish comparison via str equality on equal-length
            # strings is fine for an admin tool with single-attacker risk.
            raise HTTPException(status_code=403, detail="forbidden")

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid_json")

        target_user = str(body.get("user_id", "") or "").strip()
        if not target_user:
            raise HTTPException(status_code=400, detail="user_id_required")

        try:
            amount = int(body.get("amount", 0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="amount_required")
        if amount <= 0:
            raise HTTPException(status_code=400, detail="amount_must_be_positive")

        kind_raw = str(body.get("kind", "admin_grant") or "admin_grant").strip()
        try:
            kind = CreditTxKind(kind_raw)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid_kind:{kind_raw}")

        note = str(body.get("note", "") or "").strip()
        ref_id = str(body.get("ref_id", "") or "").strip()

        svc = _service()
        try:
            tx = await svc.grant(
                user_id=target_user,
                amount=amount,
                kind=kind,
                ref_id=ref_id,
                note=note or f"admin grant via /admin/grant",
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        balance = await svc.balance(target_user)
        return JSONResponse(content={
            "tx":      tx.to_dict(),
            "balance": balance,
        })

    @router.post("/topup")
    async def topup_stub(request: Request) -> JSONResponse:
        """Stub. Returns 501 with the pack list so the frontend topup
        modal can render even before Stripe is wired. When Stripe lands,
        this endpoint replaces with a real Checkout Session create."""
        return JSONResponse(
            status_code=501,
            content={
                "error":   "billing_not_ready",
                "message": "Top-up via Stripe is not yet enabled. Reach out to support for early access credits.",
                "packs":   [p.to_dict() for p in DEFAULT_PACKS],
            },
        )

    return router


__all__ = ["get_router"]
