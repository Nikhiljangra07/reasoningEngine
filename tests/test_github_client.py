"""
GitHubClient + github handler tests.

No live GitHub calls — every test passes a custom httpx transport so the
contracts (auth header, 200/404/401/403 handling, summary rendering,
heuristic dispatch from purpose text) are exercised without touching the
network.

Run: PYTHONPATH=. python tests/test_github_client.py
"""

from __future__ import annotations

import asyncio
import os

import httpx

from src.bridge.github_client import (
    GitHubClient,
    build_github_client_from_env,
    make_github_handler,
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
# Fake httpx transport — captures the request, returns scripted responses.
# Passed into GitHubClient via the `transport=` constructor arg, so each
# test gets a fresh transport with no leakage between tests.
# ---------------------------------------------------------------------------


class _FakeTransport(httpx.AsyncBaseTransport):
    """Returns the response in `responses` matching the request URL path."""

    def __init__(self, responses: dict[str, httpx.Response]):
        self._responses = responses
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self._responses:
            return self._responses[path]
        return httpx.Response(404, json={"message": "Not Found"})


def _client(responses: dict[str, httpx.Response] | None = None, token: str = "t"):
    """Construct a GitHubClient bound to a fresh fake transport."""
    transport = _FakeTransport(responses or {})
    c = GitHubClient(token=token, transport=transport)
    c._test_transport = transport  # type: ignore[attr-defined]
    return c


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

@test("1.1 GitHubClient with token sets Authorization header")
def test_token_header_set():
    c = GitHubClient(token="ghp_abc")
    assert c._headers.get("Authorization") == "Bearer ghp_abc"


@test("1.2 GitHubClient without token omits Authorization header")
def test_no_token_no_header():
    c = GitHubClient(token=None)
    assert "Authorization" not in c._headers


@test("1.3 GitHubClient strips whitespace from token")
def test_token_strip():
    c = GitHubClient(token="  ghp_xyz \n")
    assert c._headers["Authorization"] == "Bearer ghp_xyz"


# ---------------------------------------------------------------------------
# 2. HTTP status handling
# ---------------------------------------------------------------------------

@test("2.1 200 response → GitHubResult.ok=True with data")
async def test_200_ok():
    c = _client({
        "/repos/owner/repo/pulls/42": httpx.Response(
            200, json={"number": 42, "state": "open", "title": "Fix x"},
        ),
    })
    res = await c.get_pr("owner", "repo", 42)
    assert res.ok is True
    assert res.status == 200
    assert res.data["number"] == 42


@test("2.2 404 → ok=False with 'not found'")
async def test_404():
    c = _client({})  # nothing matches → 404 fallback
    res = await c.get_pr("owner", "repo", 99999)
    assert res.ok is False
    assert res.status == 404
    assert "not found" in res.error.lower()


@test("2.3 401 → ok=False with token-rejected error")
async def test_401():
    c = _client({
        "/repos/o/r": httpx.Response(401, json={}),
    })
    res = await c.get_repo("o", "r")
    assert res.ok is False
    assert res.status == 401
    assert "token rejected" in res.error


@test("2.4 403 + rate-limit header → ok=False with rate-limit message")
async def test_403_rate_limit():
    c = _client({
        "/repos/o/r": httpx.Response(
            403, json={}, headers={"X-RateLimit-Remaining": "0"},
        ),
    })
    res = await c.get_repo("o", "r")
    assert res.ok is False
    assert "rate limit" in res.error


# ---------------------------------------------------------------------------
# 3. Endpoint paths + params
# ---------------------------------------------------------------------------

@test("3.1 search_issues builds /search/issues with q + per_page")
async def test_search_issues_path():
    c = _client({
        "/search/issues": httpx.Response(
            200, json={"items": [], "total_count": 0},
        ),
    })
    res = await c.search_issues("hello", per_page=3)
    assert res.ok is True
    req = c._test_transport.requests[-1]
    assert req.url.path == "/search/issues"
    assert req.url.params["q"] == "hello"
    assert req.url.params["per_page"] == "3"


@test("3.2 list_recent_commits caps per_page at 30")
async def test_list_recent_commits_per_page_cap():
    c = _client({
        "/repos/o/r/commits": httpx.Response(200, json=[]),
    })
    await c.list_recent_commits("o", "r", per_page=100)
    req = c._test_transport.requests[-1]
    assert req.url.params["per_page"] == "30"


# ---------------------------------------------------------------------------
# 4. Handler — explicit args dispatch
# ---------------------------------------------------------------------------

@test("4.1 handler with action=get_pr dispatches to client.get_pr")
async def test_handler_explicit_get_pr():
    c = _client({
        "/repos/stripe/stripe/pulls/482": httpx.Response(
            200, json={
                "number": 482, "state": "open", "title": "Add idempotency keys",
                "body": "Body text.", "merged_at": None,
            },
        ),
    })
    handler = make_github_handler(c)
    out = await handler(
        {"action": "get_pr", "owner": "stripe", "repo": "stripe", "number": 482},
        purpose="check PR",
    )
    assert "PR #482" in out["text"]
    assert "idempotency" in out["text"]
    assert out["action"] == "get_pr"


@test("4.2 handler with action=get_repo returns repo summary")
async def test_handler_get_repo():
    c = _client({
        "/repos/anthropics/sdk": httpx.Response(
            200, json={
                "full_name": "anthropics/sdk",
                "default_branch": "main",
                "language": "Python",
                "description": "The official SDK",
                "topics": ["llm", "ai"],
            },
        ),
    })
    handler = make_github_handler(c)
    out = await handler(
        {"action": "get_repo", "owner": "anthropics", "repo": "sdk"},
        purpose="learn about repo",
    )
    assert "anthropics/sdk" in out["text"]
    assert "Python" in out["text"]


# ---------------------------------------------------------------------------
# 5. Handler — heuristic dispatch from purpose text (no structured args)
# ---------------------------------------------------------------------------

@test("5.1 handler extracts owner/repo + #N from purpose, fetches PR")
async def test_handler_heuristic_pr():
    c = _client({
        "/repos/stripe/stripe-python/pulls/482": httpx.Response(
            200, json={
                "number": 482, "state": "open", "title": "Fix race",
                "body": "...", "merged_at": None,
            },
        ),
    })
    handler = make_github_handler(c)
    out = await handler({}, purpose="what is the status of PR #482 in stripe/stripe-python?")
    assert "PR #482" in out["text"]
    assert out["action"] == "get_pr"


@test("5.2 handler extracts owner/repo only → fetches repo metadata")
async def test_handler_heuristic_repo():
    c = _client({
        "/repos/anthropics/anthropic-sdk-python": httpx.Response(
            200, json={
                "full_name": "anthropics/anthropic-sdk-python",
                "default_branch": "main",
                "language": "Python",
                "description": "Official SDK",
            },
        ),
    })
    handler = make_github_handler(c)
    out = await handler({}, purpose="what is anthropics/anthropic-sdk-python like?")
    assert "anthropics/anthropic-sdk-python" in out["text"]
    assert out["action"] == "get_repo"


@test("5.3 handler with no extractable repo falls back to issue search")
async def test_handler_heuristic_search():
    c = _client({
        "/search/issues": httpx.Response(
            200, json={
                "items": [
                    {
                        "number": 7,
                        "title": "Bug in webhook validation",
                        "state": "open",
                        "body": "Repro steps...",
                        "repository_url": "https://api.github.com/repos/stripe/stripe",
                    },
                ],
                "total_count": 1,
            },
        ),
    })
    handler = make_github_handler(c)
    out = await handler({}, purpose="any open issues about webhook validation?")
    assert "search" in out["text"].lower()
    assert "webhook validation" in out["text"]
    assert out["action"] == "search_issues"


# ---------------------------------------------------------------------------
# 6. Handler error path
# ---------------------------------------------------------------------------

@test("6.1 handler raises RuntimeError on 404 → fire_mcp turns into ok=False")
async def test_handler_404_raises():
    c = _client({})  # everything 404s
    handler = make_github_handler(c)
    try:
        await handler(
            {"action": "get_pr", "owner": "x", "repo": "y", "number": 1},
            purpose="x",
        )
    except RuntimeError as e:
        assert "not found" in str(e).lower() or "failed" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError on 404")


# ---------------------------------------------------------------------------
# 7. Env builder
# ---------------------------------------------------------------------------

@test("7.1 build_github_client_from_env returns None when token unset")
def test_env_builder_none():
    saved = os.environ.pop("GITHUB_TOKEN", None)
    try:
        assert build_github_client_from_env() is None
    finally:
        if saved is not None:
            os.environ["GITHUB_TOKEN"] = saved


@test("7.2 build_github_client_from_env returns client when token set")
def test_env_builder_returns_client():
    saved = os.environ.get("GITHUB_TOKEN")
    os.environ["GITHUB_TOKEN"] = "test_ghp_abc"
    try:
        client = build_github_client_from_env()
        assert client is not None
        assert client._token == "test_ghp_abc"
    finally:
        if saved is None:
            os.environ.pop("GITHUB_TOKEN", None)
        else:
            os.environ["GITHUB_TOKEN"] = saved


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_token_header_set,
    test_no_token_no_header,
    test_token_strip,
    test_200_ok,
    test_404,
    test_401,
    test_403_rate_limit,
    test_search_issues_path,
    test_list_recent_commits_per_page_cap,
    test_handler_explicit_get_pr,
    test_handler_get_repo,
    test_handler_heuristic_pr,
    test_handler_heuristic_repo,
    test_handler_heuristic_search,
    test_handler_404_raises,
    test_env_builder_none,
    test_env_builder_returns_client,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} GitHub client tests...")
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
