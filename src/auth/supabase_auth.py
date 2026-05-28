"""
Supabase Auth — JWT verification + identity resolution for Constellax.

ROLE
====
Constellax has historically run on anonymous identity: the frontend
generates a per-browser UUID in localStorage (`identity.ts`) and sends
it as `user_id` in the request body. That's enough for separating
conversation history *between browsers* but it isn't a real identity
— anyone editing localStorage becomes anyone. Per-user OAuth tokens
or any other privacy-sensitive state cannot safely sit on top of it.

This module adds the JWT-verification seam. Verified Supabase tokens
(HS256, signed with SUPABASE_JWT_SECRET) attach a `VerifiedUser` to
`request.state`; the helper `get_effective_user_id` prefers that
verified id over the body-supplied one, and `require_auth` is a
dependency endpoints can opt into when they need to enforce.

NON-BREAKING BY DESIGN (Phase 3A)
=================================
At the time of this commit:
  - `auth_middleware` runs on every request, but only ATTACHES state
    when a valid JWT is present. Missing/invalid tokens flow through
    with `request.state.verified_user = None`.
  - No endpoint calls `require_auth` yet — that's Phase 3C, after the
    frontend can send JWTs.
  - When SUPABASE_JWT_SECRET is unset (current default), `verify_jwt`
    returns None for every input, so behavior is exactly as before:
    body-supplied user_id is the only identity signal.

This module mirrors the LoRa backend's `src/server/auth/supabaseAuth.ts`
pattern. Same algorithm (HS256), same leeway, same priority order
(JWT sub > body user_id > anonymous).

SECURITY NOTES
==============
  - HS256 means the same secret signs AND verifies. NEVER expose
    SUPABASE_JWT_SECRET to the frontend; it's a server-only secret.
  - The leeway window (30s) covers clock skew between Supabase and
    our server; smaller would reject legitimate tokens, larger would
    extend the window for replay.
  - We require `exp` and `sub` claims via PyJWT's `options.require`
    — missing either is an automatic failure.
  - Verification failures log at DEBUG, never WARN/ERROR, because a
    malformed/expired token is normal user behavior (refresh flow
    in progress) and shouldn't pollute production logs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

try:
    import jwt  # PyJWT
    _PYJWT_AVAILABLE = True
except ImportError:
    _PYJWT_AVAILABLE = False
    jwt = None  # type: ignore[assignment]

from fastapi import HTTPException, Request


log = logging.getLogger("constellax.auth")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPABASE_JWT_ALG = "HS256"
SUPABASE_JWT_LEEWAY_SEC = 30


def _get_secret() -> str:
    """Read SUPABASE_JWT_SECRET fresh each call.

    Tests mutate os.environ; reading at import time would mean tests can
    never enable auth after the module loads. Cheap to re-read each call.
    """
    return os.environ.get("SUPABASE_JWT_SECRET", "").strip()


def is_auth_configured() -> bool:
    """True when SUPABASE_JWT_SECRET is set AND PyJWT is installed."""
    return _PYJWT_AVAILABLE and bool(_get_secret())


# ---------------------------------------------------------------------------
# VerifiedUser record
# ---------------------------------------------------------------------------


@dataclass
class VerifiedUser:
    """A user whose identity was just verified against SUPABASE_JWT_SECRET.

    Attached to `request.state.verified_user` by the middleware. Endpoint
    code reads it via `get_effective_user_id` (best-effort) or
    `require_auth` (strict).
    """
    user_id: str            # claims["sub"] — the Supabase user UUID
    email: str = ""         # claims["email"] if present
    role: str = ""          # claims["role"] (e.g. "authenticated") if present
    raw_claims: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JWT verification
# ---------------------------------------------------------------------------


def verify_jwt(token: str) -> VerifiedUser | None:
    """Verify a Supabase HS256 JWT. Returns None on any failure.

    Failure modes that return None (never raise):
      - PyJWT not installed
      - SUPABASE_JWT_SECRET not configured
      - Empty / whitespace-only token
      - Bad signature, expired, missing required claims, wrong alg
      - Malformed token (not a JWT at all)
    """
    if not _PYJWT_AVAILABLE:
        return None
    secret = _get_secret()
    if not secret:
        return None
    if not isinstance(token, str) or not token.strip():
        return None

    try:
        claims = jwt.decode(  # type: ignore[union-attr]
            token,
            secret,
            algorithms=[SUPABASE_JWT_ALG],
            leeway=SUPABASE_JWT_LEEWAY_SEC,
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as e:  # type: ignore[union-attr]
        log.debug("JWT verification failed: %s", e)
        return None
    except Exception as e:
        # Catch-all for anything PyJWT didn't anticipate (malformed
        # base64, decoding errors). Never let auth-path issues become
        # 500s — fail closed instead.
        log.debug("JWT verification raised unexpected exception: %s", e)
        return None

    sub = str(claims.get("sub", "") or "").strip()
    if not sub:
        return None

    return VerifiedUser(
        user_id=sub,
        email=str(claims.get("email", "") or "").strip(),
        role=str(claims.get("role", "") or "").strip(),
        raw_claims=claims,
    )


# ---------------------------------------------------------------------------
# Request integration
# ---------------------------------------------------------------------------


def extract_bearer_token(request: Request) -> str:
    """Pull the bearer token from the Authorization header. Empty on miss."""
    header = (
        request.headers.get("authorization")
        or request.headers.get("Authorization")
        or ""
    )
    header = header.strip()
    if len(header) < 8 or header[:7].lower() != "bearer ":
        return ""
    return header[7:].strip()


async def auth_middleware(request: Request, call_next):
    """Per-request: attach VerifiedUser to request.state if a valid JWT
    is present, else attach None.

    Wire into FastAPI via `app.middleware("http")(auth_middleware)`.
    Never rejects — gating is per-endpoint via `require_auth`.
    """
    token = extract_bearer_token(request)
    request.state.verified_user = verify_jwt(token) if token else None
    return await call_next(request)


def get_verified_user(request: Request) -> VerifiedUser | None:
    """Best-effort read of the verified user attached by middleware.

    Returns None when middleware didn't run (test contexts) or when no
    valid JWT was attached on this request.
    """
    return getattr(request.state, "verified_user", None)


def get_effective_user_id(
    request: Request,
    body_user_id: Any | None = None,
) -> str | None:
    """Resolve the user_id for this request, with JWT taking precedence.

    Priority order:
      1. Verified JWT sub claim (set by `auth_middleware`)
      2. Body-supplied user_id string (anonymous / legacy path)
      3. None — no identity at all

    `body_user_id` is accepted as `Any` because callers pull it from
    untrusted request JSON; non-string values fall through to None.

    Phase 3A contract: endpoints that already use body user_id keep
    working unchanged. Phase 3C will switch the fallback off.
    """
    verified = get_verified_user(request)
    if verified is not None:
        return verified.user_id
    if isinstance(body_user_id, str) and body_user_id.strip():
        return body_user_id.strip()
    return None


def require_auth(request: Request) -> VerifiedUser:
    """FastAPI dependency: assert verified identity, else 401.

    Use as `user = Depends(require_auth)` on endpoints that should
    refuse anonymous traffic. Not wired anywhere yet — that flip is
    Phase 3C, after the frontend can send JWTs.

    Raises HTTPException(401) on missing/invalid JWT.
    """
    verified = get_verified_user(request)
    if verified is None:
        # Detail message intentionally generic — don't leak whether
        # we got no token vs invalid token vs expired token.
        raise HTTPException(
            status_code=401,
            detail={"error": "authentication required"},
        )
    return verified


__all__ = [
    "VerifiedUser",
    "SUPABASE_JWT_ALG",
    "SUPABASE_JWT_LEEWAY_SEC",
    "is_auth_configured",
    "verify_jwt",
    "extract_bearer_token",
    "auth_middleware",
    "get_verified_user",
    "get_effective_user_id",
    "require_auth",
]
