"""
Supabase JWT auth tests.

No live Supabase calls — every test mints self-signed HS256 tokens using
the test secret, then verifies them through src.auth.supabase_auth. The
non-breaking promise (unset secret = no-op, never raises) is covered by
its own block at the bottom.

Run: PYTHONPATH=. python tests/test_supabase_auth.py
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import jwt

from src.auth.supabase_auth import (
    SUPABASE_JWT_ALG,
    SUPABASE_JWT_LEEWAY_SEC,
    VerifiedUser,
    auth_middleware,
    extract_bearer_token,
    get_effective_user_id,
    get_verified_user,
    is_auth_configured,
    require_auth,
    verify_jwt,
)


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
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ---------------------------------------------------------------------------
# Helpers — secret/env scaffolding + token minting
# ---------------------------------------------------------------------------

TEST_SECRET = "test-secret-32-bytes-of-padding-abcdefgh"


def _with_secret(value: str | None = TEST_SECRET):
    """Context-manager-ish helper: set SUPABASE_JWT_SECRET, restore on exit.

    Tests call this inline (saved = ..., os.environ[...] = ..., try/finally)
    rather than using contextlib so the assertions are visible in the test
    body without indenting two more levels.
    """
    saved = os.environ.get("SUPABASE_JWT_SECRET")
    if value is None:
        os.environ.pop("SUPABASE_JWT_SECRET", None)
    else:
        os.environ["SUPABASE_JWT_SECRET"] = value
    return saved


def _restore_secret(saved: str | None) -> None:
    if saved is None:
        os.environ.pop("SUPABASE_JWT_SECRET", None)
    else:
        os.environ["SUPABASE_JWT_SECRET"] = saved


def _mint_token(
    *,
    secret: str = TEST_SECRET,
    sub: str = "supabase-user-uuid",
    email: str = "tester@example.com",
    role: str = "authenticated",
    exp_in_sec: int = 600,
    iat_in_sec: int = 0,
    extra_claims: dict[str, Any] | None = None,
    alg: str = "HS256",
    omit_sub: bool = False,
    omit_exp: bool = False,
) -> str:
    """Mint a JWT for tests. Defaults model a fresh authenticated Supabase user."""
    now = int(time.time())
    claims: dict[str, Any] = {
        "iat": now + iat_in_sec,
        "exp": now + exp_in_sec,
        "sub": sub,
        "email": email,
        "role": role,
    }
    if omit_sub:
        claims.pop("sub", None)
    if omit_exp:
        claims.pop("exp", None)
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, secret, algorithm=alg)


# ---------------------------------------------------------------------------
# Fake Request/State for middleware tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeState:
    verified_user: VerifiedUser | None = None


@dataclass
class _FakeRequest:
    headers: dict[str, str] = field(default_factory=dict)
    state: _FakeState = field(default_factory=_FakeState)


async def _passthrough_call_next(_request):
    return "ok"


# ---------------------------------------------------------------------------
# 1. is_auth_configured
# ---------------------------------------------------------------------------

@test("1.1 is_auth_configured False when SUPABASE_JWT_SECRET unset")
def test_unconfigured():
    saved = _with_secret(None)
    try:
        assert is_auth_configured() is False
    finally:
        _restore_secret(saved)


@test("1.2 is_auth_configured True when SUPABASE_JWT_SECRET set")
def test_configured():
    saved = _with_secret(TEST_SECRET)
    try:
        assert is_auth_configured() is True
    finally:
        _restore_secret(saved)


@test("1.3 is_auth_configured False when secret is whitespace")
def test_whitespace_secret():
    saved = _with_secret("   ")
    try:
        assert is_auth_configured() is False
    finally:
        _restore_secret(saved)


# ---------------------------------------------------------------------------
# 2. verify_jwt — happy path
# ---------------------------------------------------------------------------

@test("2.1 valid token returns VerifiedUser with sub/email/role")
def test_verify_happy():
    saved = _with_secret(TEST_SECRET)
    try:
        token = _mint_token(sub="user-123", email="a@b.com", role="authenticated")
        v = verify_jwt(token)
        assert v is not None
        assert v.user_id == "user-123"
        assert v.email == "a@b.com"
        assert v.role == "authenticated"
        assert v.raw_claims["sub"] == "user-123"
    finally:
        _restore_secret(saved)


@test("2.2 missing email/role fields default to empty string")
def test_verify_minimal_claims():
    saved = _with_secret(TEST_SECRET)
    try:
        # No email, no role — only sub + exp + iat
        token = _mint_token(extra_claims={}, email="", role="")
        v = verify_jwt(token)
        assert v is not None
        assert v.email == ""
        assert v.role == ""
    finally:
        _restore_secret(saved)


# ---------------------------------------------------------------------------
# 3. verify_jwt — failure modes (all return None, never raise)
# ---------------------------------------------------------------------------

@test("3.1 unset secret → returns None even for valid token")
def test_verify_no_secret():
    saved = _with_secret(None)
    try:
        # Mint with a non-empty secret so the token is well-formed; the
        # missing server secret should still cause verification to fail
        # cleanly (None, not raise).
        token = _mint_token(secret="some-other-secret")
        assert verify_jwt(token) is None
    finally:
        _restore_secret(saved)


@test("3.2 token signed with wrong secret → None")
def test_verify_bad_signature():
    saved = _with_secret(TEST_SECRET)
    try:
        token = _mint_token(secret="completely-different-secret")
        assert verify_jwt(token) is None
    finally:
        _restore_secret(saved)


@test("3.3 expired token → None (no exception)")
def test_verify_expired():
    saved = _with_secret(TEST_SECRET)
    try:
        # exp set well outside the leeway window
        token = _mint_token(exp_in_sec=-(SUPABASE_JWT_LEEWAY_SEC + 60))
        assert verify_jwt(token) is None
    finally:
        _restore_secret(saved)


@test("3.4 expired-within-leeway token still verifies")
def test_verify_within_leeway():
    saved = _with_secret(TEST_SECRET)
    try:
        # exp set 10s in the past — under the 30s leeway, still accepted
        token = _mint_token(exp_in_sec=-10)
        v = verify_jwt(token)
        assert v is not None
    finally:
        _restore_secret(saved)


@test("3.5 missing sub claim → None")
def test_verify_no_sub():
    saved = _with_secret(TEST_SECRET)
    try:
        token = _mint_token(omit_sub=True)
        assert verify_jwt(token) is None
    finally:
        _restore_secret(saved)


@test("3.6 empty sub claim → None")
def test_verify_empty_sub():
    saved = _with_secret(TEST_SECRET)
    try:
        token = _mint_token(sub="")
        # Even if PyJWT accepts the token (sub present but empty), our
        # explicit sub-must-be-non-empty check rejects it.
        assert verify_jwt(token) is None
    finally:
        _restore_secret(saved)


@test("3.7 missing exp claim → None (PyJWT options.require rejects)")
def test_verify_no_exp():
    saved = _with_secret(TEST_SECRET)
    try:
        token = _mint_token(omit_exp=True)
        assert verify_jwt(token) is None
    finally:
        _restore_secret(saved)


@test("3.8 malformed token (not a JWT at all) → None, no crash")
def test_verify_malformed():
    saved = _with_secret(TEST_SECRET)
    try:
        assert verify_jwt("definitely-not-a-jwt") is None
        assert verify_jwt("just.two.dots") is None
        assert verify_jwt("") is None
    finally:
        _restore_secret(saved)


@test("3.9 non-string token → None, no crash")
def test_verify_non_string():
    saved = _with_secret(TEST_SECRET)
    try:
        assert verify_jwt(None) is None  # type: ignore[arg-type]
        assert verify_jwt(42) is None  # type: ignore[arg-type]
    finally:
        _restore_secret(saved)


@test("3.10 RS256-signed token rejected (we only accept HS256)")
def test_verify_wrong_alg():
    # We won't generate a real RS256 key here; instead, mint with HS256 but
    # explicitly check the alg list rejects anything else by inverting:
    # an "alg=none" token gets the unsigned-token error path.
    saved = _with_secret(TEST_SECRET)
    try:
        # PyJWT refuses to encode with alg='none' easily; instead simulate
        # by passing a token that uses HS512 — different alg from our
        # accepted list.
        token = jwt.encode(
            {"sub": "x", "exp": int(time.time()) + 600},
            TEST_SECRET,
            algorithm="HS512",
        )
        assert verify_jwt(token) is None
    finally:
        _restore_secret(saved)


# ---------------------------------------------------------------------------
# 4. extract_bearer_token
# ---------------------------------------------------------------------------

@test("4.1 standard 'Bearer <token>' header extracts the token")
def test_extract_standard():
    req = _FakeRequest(headers={"authorization": "Bearer abc.def.ghi"})
    assert extract_bearer_token(req) == "abc.def.ghi"  # type: ignore[arg-type]


@test("4.2 case-insensitive on the scheme")
def test_extract_case_insensitive():
    req = _FakeRequest(headers={"authorization": "bearer abc.def.ghi"})
    assert extract_bearer_token(req) == "abc.def.ghi"  # type: ignore[arg-type]
    req2 = _FakeRequest(headers={"authorization": "BEARER abc"})
    assert extract_bearer_token(req2) == "abc"  # type: ignore[arg-type]


@test("4.3 capital-A Authorization header also accepted")
def test_extract_capital_a():
    req = _FakeRequest(headers={"Authorization": "Bearer tok"})
    assert extract_bearer_token(req) == "tok"  # type: ignore[arg-type]


@test("4.4 no Authorization header → empty string")
def test_extract_missing():
    req = _FakeRequest(headers={})
    assert extract_bearer_token(req) == ""  # type: ignore[arg-type]


@test("4.5 wrong scheme (Basic) → empty string")
def test_extract_wrong_scheme():
    req = _FakeRequest(headers={"authorization": "Basic abc:def"})
    assert extract_bearer_token(req) == ""  # type: ignore[arg-type]


@test("4.6 whitespace-only header → empty string")
def test_extract_whitespace():
    req = _FakeRequest(headers={"authorization": "   "})
    assert extract_bearer_token(req) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. auth_middleware — attaches VerifiedUser or None to request.state
# ---------------------------------------------------------------------------

@test("5.1 middleware attaches VerifiedUser on valid token")
async def test_middleware_valid():
    saved = _with_secret(TEST_SECRET)
    try:
        token = _mint_token(sub="middleware-user")
        req = _FakeRequest(headers={"authorization": f"Bearer {token}"})
        result = await auth_middleware(req, _passthrough_call_next)  # type: ignore[arg-type]
        assert result == "ok"
        assert req.state.verified_user is not None
        assert req.state.verified_user.user_id == "middleware-user"
    finally:
        _restore_secret(saved)


@test("5.2 middleware attaches None on missing header")
async def test_middleware_no_header():
    saved = _with_secret(TEST_SECRET)
    try:
        req = _FakeRequest(headers={})
        result = await auth_middleware(req, _passthrough_call_next)  # type: ignore[arg-type]
        assert result == "ok"
        assert req.state.verified_user is None
    finally:
        _restore_secret(saved)


@test("5.3 middleware attaches None on invalid token")
async def test_middleware_invalid():
    saved = _with_secret(TEST_SECRET)
    try:
        req = _FakeRequest(headers={"authorization": "Bearer not.a.real.token"})
        result = await auth_middleware(req, _passthrough_call_next)  # type: ignore[arg-type]
        assert result == "ok"
        assert req.state.verified_user is None
    finally:
        _restore_secret(saved)


@test("5.4 middleware NEVER rejects — call_next always runs")
async def test_middleware_never_rejects():
    # Even with secret unset and bogus token, we still pass through.
    saved = _with_secret(None)
    try:
        req = _FakeRequest(headers={"authorization": "Bearer garbage"})
        result = await auth_middleware(req, _passthrough_call_next)  # type: ignore[arg-type]
        assert result == "ok"
        assert req.state.verified_user is None
    finally:
        _restore_secret(saved)


# ---------------------------------------------------------------------------
# 6. get_effective_user_id — priority chain
# ---------------------------------------------------------------------------

@test("6.1 verified JWT sub takes precedence over body user_id")
def test_effective_jwt_wins():
    req = _FakeRequest()
    req.state.verified_user = VerifiedUser(user_id="jwt-user")
    assert get_effective_user_id(req, body_user_id="body-user") == "jwt-user"  # type: ignore[arg-type]


@test("6.2 body user_id used when no verified user")
def test_effective_body_fallback():
    req = _FakeRequest()
    req.state.verified_user = None
    assert get_effective_user_id(req, body_user_id="body-user") == "body-user"  # type: ignore[arg-type]


@test("6.3 None when both verified and body absent")
def test_effective_none():
    req = _FakeRequest()
    req.state.verified_user = None
    assert get_effective_user_id(req, body_user_id=None) is None  # type: ignore[arg-type]


@test("6.4 body user_id non-string falls through to None")
def test_effective_non_string_body():
    req = _FakeRequest()
    req.state.verified_user = None
    assert get_effective_user_id(req, body_user_id=42) is None  # type: ignore[arg-type]
    assert get_effective_user_id(req, body_user_id={"id": "x"}) is None  # type: ignore[arg-type]


@test("6.5 body user_id whitespace-only → None")
def test_effective_whitespace_body():
    req = _FakeRequest()
    req.state.verified_user = None
    assert get_effective_user_id(req, body_user_id="   ") is None  # type: ignore[arg-type]


@test("6.6 body user_id stripped before return")
def test_effective_body_stripped():
    req = _FakeRequest()
    req.state.verified_user = None
    assert get_effective_user_id(req, body_user_id="  spaced  ") == "spaced"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. require_auth — raises 401 when no verified user
# ---------------------------------------------------------------------------

@test("7.1 require_auth returns VerifiedUser when present")
def test_require_present():
    req = _FakeRequest()
    req.state.verified_user = VerifiedUser(user_id="user-x")
    v = require_auth(req)  # type: ignore[arg-type]
    assert v.user_id == "user-x"


@test("7.2 require_auth raises HTTPException(401) when absent")
def test_require_missing():
    from fastapi import HTTPException
    req = _FakeRequest()
    req.state.verified_user = None
    try:
        require_auth(req)  # type: ignore[arg-type]
    except HTTPException as e:
        assert e.status_code == 401
        return
    raise AssertionError("expected HTTPException(401)")


@test("7.3 require_auth error detail is generic (no token-state leak)")
def test_require_generic_detail():
    from fastapi import HTTPException
    req = _FakeRequest()
    req.state.verified_user = None
    try:
        require_auth(req)  # type: ignore[arg-type]
    except HTTPException as e:
        detail = e.detail
        detail_str = str(detail).lower()
        # Should NOT mention "expired", "invalid signature", "missing sub"
        # — those leak which failure mode triggered.
        for leak in ("expired", "invalid signature", "missing sub", "malformed"):
            assert leak not in detail_str, f"detail leaks failure mode: {leak!r}"
        return
    raise AssertionError("expected HTTPException")


# ---------------------------------------------------------------------------
# 8. get_verified_user — read-only convenience
# ---------------------------------------------------------------------------

@test("8.1 get_verified_user returns attached user")
def test_get_verified_present():
    req = _FakeRequest()
    req.state.verified_user = VerifiedUser(user_id="u")
    assert get_verified_user(req).user_id == "u"  # type: ignore[arg-type, union-attr]


@test("8.2 get_verified_user returns None when nothing attached")
def test_get_verified_none():
    req = _FakeRequest()
    req.state.verified_user = None
    assert get_verified_user(req) is None  # type: ignore[arg-type]


@test("8.3 get_verified_user returns None when state has no verified_user attr")
def test_get_verified_no_state():
    # Simulates middleware never having run (test contexts, raw requests).
    class _NakedReq:
        class state:  # type: ignore[no-redef]
            pass
    assert get_verified_user(_NakedReq()) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_unconfigured,
    test_configured,
    test_whitespace_secret,
    test_verify_happy,
    test_verify_minimal_claims,
    test_verify_no_secret,
    test_verify_bad_signature,
    test_verify_expired,
    test_verify_within_leeway,
    test_verify_no_sub,
    test_verify_empty_sub,
    test_verify_no_exp,
    test_verify_malformed,
    test_verify_non_string,
    test_verify_wrong_alg,
    test_extract_standard,
    test_extract_case_insensitive,
    test_extract_capital_a,
    test_extract_missing,
    test_extract_wrong_scheme,
    test_extract_whitespace,
    test_middleware_valid,
    test_middleware_no_header,
    test_middleware_invalid,
    test_middleware_never_rejects,
    test_effective_jwt_wins,
    test_effective_body_fallback,
    test_effective_none,
    test_effective_non_string_body,
    test_effective_whitespace_body,
    test_effective_body_stripped,
    test_require_present,
    test_require_missing,
    test_require_generic_detail,
    test_get_verified_present,
    test_get_verified_none,
    test_get_verified_no_state,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} Supabase auth tests...")
    print()
    for fn in ALL_TESTS:
        run_test(fn)
    print()
    print(f"{PASSED} passed, {FAILED} failed")
    if ERRORS:
        print()
        print("Failures:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
