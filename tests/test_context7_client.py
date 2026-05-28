"""
Context7Client + context7 handler tests.

No live Context7 calls — every test injects a fake httpx transport via
the constructor.

Run: PYTHONPATH=. python tests/test_context7_client.py
"""

from __future__ import annotations

import asyncio
import os

import httpx

from src.bridge.context7_client import (
    Context7Client,
    build_context7_client_from_env,
    make_context7_handler,
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


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: dict[str, httpx.Response]):
        self._responses = responses
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self._responses:
            return self._responses[path]
        return httpx.Response(404, json={"message": "Not Found"})


def _client(responses: dict[str, httpx.Response] | None = None, api_key: str | None = None):
    transport = _FakeTransport(responses or {})
    c = Context7Client(api_key=api_key, transport=transport)
    c._test_transport = transport  # type: ignore[attr-defined]
    return c


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

@test("1.1 no api_key → no Authorization header")
def test_no_api_key_header():
    c = Context7Client()
    assert "Authorization" not in c._headers


@test("1.2 api_key set → Authorization header present")
def test_api_key_header():
    c = Context7Client(api_key="ctx7_xyz")
    assert c._headers["Authorization"] == "Bearer ctx7_xyz"


@test("1.3 api_root rstrips trailing slash")
def test_api_root_strip():
    c = Context7Client(api_root="https://example.com/api/v1/")
    assert c._api_root == "https://example.com/api/v1"


# ---------------------------------------------------------------------------
# 2. HTTP status handling
# ---------------------------------------------------------------------------

@test("2.1 200 JSON → ok=True with parsed data")
async def test_200_json():
    c = _client({
        "/api/v1/search": httpx.Response(
            200, json={"results": [{"library_id": "stripe/stripe-python"}]},
        ),
    })
    res = await c.search_libraries("stripe")
    assert res.ok is True
    assert res.data["results"][0]["library_id"] == "stripe/stripe-python"


@test("2.2 200 text/plain → wrapped as {'text': ...}")
async def test_200_text():
    c = _client({
        "/api/v1/vercel/next.js": httpx.Response(
            200, text="# Next.js docs\n\nThe React framework.",
            headers={"Content-Type": "text/plain"},
        ),
    })
    res = await c.get_library_docs("vercel/next.js")
    assert res.ok is True
    assert "Next.js docs" in res.data["text"]


@test("2.3 404 → ok=False 'not found'")
async def test_404():
    c = _client({})
    res = await c.get_library_docs("does/not-exist")
    assert res.ok is False
    assert res.status == 404


@test("2.4 429 → ok=False 'rate limit exhausted'")
async def test_429():
    c = _client({
        "/api/v1/stripe/stripe-python": httpx.Response(429, json={}),
    })
    res = await c.get_library_docs("stripe/stripe-python")
    assert res.ok is False
    assert "rate limit" in res.error


@test("2.5 401 → ok=False auth/permission error")
async def test_401():
    c = _client({
        "/api/v1/search": httpx.Response(401, json={}),
    })
    res = await c.search_libraries("anything")
    assert res.ok is False
    assert res.status == 401


# ---------------------------------------------------------------------------
# 3. get_library_docs params
# ---------------------------------------------------------------------------

@test("3.1 get_library_docs sends topic + tokens query params")
async def test_get_docs_params():
    c = _client({
        "/api/v1/stripe/stripe-python": httpx.Response(
            200, json={"text": "..."},
        ),
    })
    await c.get_library_docs("stripe/stripe-python", topic="idempotency", tokens=1500)
    req = c._test_transport.requests[-1]
    assert req.url.params["topic"] == "idempotency"
    assert req.url.params["tokens"] == "1500"


@test("3.2 tokens clamped at upper bound (_MAX_TOKENS)")
async def test_tokens_clamp():
    c = _client({
        "/api/v1/stripe/stripe-python": httpx.Response(200, json={"text": "..."}),
    })
    await c.get_library_docs("stripe/stripe-python", tokens=99999)
    req = c._test_transport.requests[-1]
    # _MAX_TOKENS = 8000 in the module
    assert int(req.url.params["tokens"]) <= 8000


# ---------------------------------------------------------------------------
# 4. Handler — explicit library_id
# ---------------------------------------------------------------------------

@test("4.1 handler with library_id fetches docs directly")
async def test_handler_explicit_library_id():
    c = _client({
        "/api/v1/stripe/stripe-python": httpx.Response(
            200, json={"text": "# Stripe Python SDK\n\nIdempotency keys are..."},
        ),
    })
    handler = make_context7_handler(c)
    out = await handler(
        {"library_id": "stripe/stripe-python", "topic": "idempotency"},
        purpose="docs lookup",
    )
    assert out["action"] == "get_library_docs"
    assert "Idempotency keys" in out["text"]
    assert out["library_id"] == "stripe/stripe-python"


# ---------------------------------------------------------------------------
# 5. Handler — search → fetch flow
# ---------------------------------------------------------------------------

@test("5.1 handler searches when only query provided, then fetches docs")
async def test_handler_search_then_fetch():
    c = _client({
        "/api/v1/search": httpx.Response(
            200, json={"results": [{"library_id": "vercel/next.js"}]},
        ),
        "/api/v1/vercel/next.js": httpx.Response(
            200, json={"text": "# Next.js docs"},
        ),
    })
    handler = make_context7_handler(c)
    out = await handler({"query": "next.js"}, purpose="learn nextjs")
    assert out["action"] == "search_then_get_docs"
    assert out["library_id"] == "vercel/next.js"
    assert "Next.js docs" in out["text"]


@test("5.2 handler uses purpose as fallback search query when no args")
async def test_handler_purpose_fallback():
    c = _client({
        "/api/v1/search": httpx.Response(
            200, json={"results": [{"library_id": "anthropics/anthropic-sdk-python"}]},
        ),
        "/api/v1/anthropics/anthropic-sdk-python": httpx.Response(
            200, json={"text": "Anthropic SDK docs"},
        ),
    })
    handler = make_context7_handler(c)
    out = await handler({}, purpose="how do I use the anthropic python sdk?")
    assert "Anthropic SDK docs" in out["text"]


@test("5.3 handler raises when no query and no purpose")
async def test_handler_no_input():
    c = _client({})
    handler = make_context7_handler(c)
    try:
        await handler({}, purpose="")
    except RuntimeError as e:
        assert "no library_id" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError when no input")


@test("5.4 search returns empty → handler surfaces 'no match' instead of crashing")
async def test_handler_search_no_match():
    c = _client({
        "/api/v1/search": httpx.Response(
            200, json={"results": []},
        ),
    })
    handler = make_context7_handler(c)
    out = await handler({"query": "zzz-no-match"}, purpose="x")
    assert out["action"] == "search_libraries"
    assert "no matching library" in out["text"].lower() or "zzz-no-match" in out["text"]


# ---------------------------------------------------------------------------
# 6. Handler error path
# ---------------------------------------------------------------------------

@test("6.1 handler raises RuntimeError on 404 → fire_mcp turns into ok=False")
async def test_handler_404_raises():
    c = _client({})  # everything 404s
    handler = make_context7_handler(c)
    try:
        await handler({"library_id": "x/y"}, purpose="x")
    except RuntimeError as e:
        assert "not found" in str(e).lower() or "failed" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError on 404")


# ---------------------------------------------------------------------------
# 7. Env builder
# ---------------------------------------------------------------------------

@test("7.1 build_context7_client_from_env returns None when CONTEXT7_ENABLED unset")
def test_env_disabled():
    saved = os.environ.pop("CONTEXT7_ENABLED", None)
    try:
        assert build_context7_client_from_env() is None
    finally:
        if saved is not None:
            os.environ["CONTEXT7_ENABLED"] = saved


@test("7.2 build_context7_client_from_env returns client when CONTEXT7_ENABLED=1")
def test_env_enabled():
    saved_enabled = os.environ.get("CONTEXT7_ENABLED")
    saved_root = os.environ.get("CONTEXT7_API_ROOT")
    os.environ["CONTEXT7_ENABLED"] = "1"
    os.environ["CONTEXT7_API_ROOT"] = "https://example.com/api/v1"
    try:
        c = build_context7_client_from_env()
        assert c is not None
        assert c._api_root == "https://example.com/api/v1"
    finally:
        if saved_enabled is None:
            os.environ.pop("CONTEXT7_ENABLED", None)
        else:
            os.environ["CONTEXT7_ENABLED"] = saved_enabled
        if saved_root is None:
            os.environ.pop("CONTEXT7_API_ROOT", None)
        else:
            os.environ["CONTEXT7_API_ROOT"] = saved_root


@test("7.3 'false' / '0' / 'no' explicitly disable")
def test_env_falsey_values():
    saved = os.environ.get("CONTEXT7_ENABLED")
    try:
        for val in ("", "0", "false", "no", "off"):
            os.environ["CONTEXT7_ENABLED"] = val
            assert build_context7_client_from_env() is None, f"expected None for {val!r}"
    finally:
        if saved is None:
            os.environ.pop("CONTEXT7_ENABLED", None)
        else:
            os.environ["CONTEXT7_ENABLED"] = saved


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_no_api_key_header,
    test_api_key_header,
    test_api_root_strip,
    test_200_json,
    test_200_text,
    test_404,
    test_429,
    test_401,
    test_get_docs_params,
    test_tokens_clamp,
    test_handler_explicit_library_id,
    test_handler_search_then_fetch,
    test_handler_purpose_fallback,
    test_handler_no_input,
    test_handler_search_no_match,
    test_handler_404_raises,
    test_env_disabled,
    test_env_enabled,
    test_env_falsey_values,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} Context7 client tests...")
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
