"""
PlaywrightClient + playwright handler tests.

No real browser launches — every test injects a FakePageProvider via
the make_playwright_handler factory. PlaywrightClient's real browser
path is tested only at the smoke level (it imports playwright lazily,
so importing this module doesn't require the package).

Run: PYTHONPATH=. python tests/test_playwright_client.py
"""

from __future__ import annotations

import asyncio
import os

from src.bridge.playwright_client import (
    PageSnapshot,
    PlaywrightClient,
    _extract_url_from_purpose,
    _format_snapshot_text,
    _is_safe_url,
    build_playwright_client_from_env,
    make_playwright_handler,
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


class _FakePageProvider:
    """Records calls + returns pre-canned snapshots keyed by exact URL.

    Falls back to a default ok-snapshot when no specific entry matches,
    so a test that doesn't care about the URL can just `_FakePageProvider()`.
    """

    def __init__(self, *, snapshots: dict[str, PageSnapshot] | None = None,
                 default: PageSnapshot | None = None) -> None:
        self.snapshots = snapshots or {}
        self.default = default
        self.calls: list[str] = []

    async def open_url(self, url: str) -> PageSnapshot:
        self.calls.append(url)
        if url in self.snapshots:
            return self.snapshots[url]
        if self.default is not None:
            return self.default
        return PageSnapshot(
            ok=True, url=url, final_url=url, title="Default", text="ok", status=200,
        )


# ---------------------------------------------------------------------------
# 1. URL safety guard
# ---------------------------------------------------------------------------

@test("1.1 https URL is safe")
def test_safe_https():
    ok, _ = _is_safe_url("https://example.com/path")
    assert ok is True


@test("1.2 http URL is safe")
def test_safe_http():
    ok, _ = _is_safe_url("http://example.com")
    assert ok is True


@test("1.3 file:// rejected")
def test_reject_file():
    ok, reason = _is_safe_url("file:///etc/passwd")
    assert ok is False
    assert "scheme" in reason


@test("1.4 javascript: rejected")
def test_reject_js():
    ok, reason = _is_safe_url("javascript:alert(1)")
    assert ok is False
    assert "scheme" in reason


@test("1.5 localhost rejected")
def test_reject_localhost():
    ok, reason = _is_safe_url("http://localhost:8000/")
    assert ok is False
    assert "local" in reason.lower()


@test("1.6 127.0.0.1 rejected (loopback)")
def test_reject_loopback():
    ok, reason = _is_safe_url("http://127.0.0.1/")
    assert ok is False


@test("1.7 RFC1918 private IP rejected")
def test_reject_private_ip():
    for ip in ("http://10.0.0.1/", "http://192.168.1.1/", "http://172.16.0.5/"):
        ok, _ = _is_safe_url(ip)
        assert ok is False, f"expected {ip} to be rejected"


@test("1.8 empty URL rejected")
def test_reject_empty():
    ok, _ = _is_safe_url("")
    assert ok is False
    ok2, _ = _is_safe_url("   ")
    assert ok2 is False


@test("1.9 URL without host rejected")
def test_reject_no_host():
    ok, _ = _is_safe_url("https://")
    assert ok is False


# ---------------------------------------------------------------------------
# 2. URL extraction from purpose text
# ---------------------------------------------------------------------------

@test("2.1 extracts first http URL from prose")
def test_extract_http():
    url = _extract_url_from_purpose("Go look at http://example.com please")
    assert url == "http://example.com"


@test("2.2 extracts first https URL from prose")
def test_extract_https():
    url = _extract_url_from_purpose("Verify https://asklora.io/manifesto renders")
    assert url == "https://asklora.io/manifesto"


@test("2.3 no URL → empty string")
def test_extract_none():
    assert _extract_url_from_purpose("just chatting") == ""
    assert _extract_url_from_purpose("") == ""


@test("2.4 trims trailing punctuation-ish boundaries")
def test_extract_trim():
    # _URL_IN_TEXT_RE stops at whitespace + < > " ' ) — period stays.
    url = _extract_url_from_purpose("see (https://example.com/x)")
    assert url == "https://example.com/x"


# ---------------------------------------------------------------------------
# 3. Snapshot formatting
# ---------------------------------------------------------------------------

@test("3.1 ok snapshot renders title + status + body")
def test_format_ok():
    snap = PageSnapshot(
        ok=True, url="https://example.com", final_url="https://example.com/x",
        title="Example", text="hello world", status=200,
    )
    text = _format_snapshot_text(snap)
    assert "URL: https://example.com/x" in text
    assert "Title: Example" in text
    assert "HTTP status: 200" in text
    assert "hello world" in text


@test("3.2 not-ok snapshot renders error")
def test_format_error():
    snap = PageSnapshot(ok=False, url="https://example.com", error="navigation failed: timeout")
    text = _format_snapshot_text(snap)
    assert "browser error" in text
    assert "timeout" in text


@test("3.3 empty body falls back to placeholder")
def test_format_empty_body():
    snap = PageSnapshot(
        ok=True, url="https://example.com", final_url="https://example.com",
        title="", text="", status=204,
    )
    text = _format_snapshot_text(snap)
    assert "empty page body" in text


# ---------------------------------------------------------------------------
# 4. Handler — URL from args
# ---------------------------------------------------------------------------

@test("4.1 handler reads url from args")
async def test_handler_args_url():
    fake = _FakePageProvider(snapshots={
        "https://example.com": PageSnapshot(
            ok=True, url="https://example.com", final_url="https://example.com",
            title="Hi", text="welcome", status=200,
        ),
    })
    handler = make_playwright_handler(fake)
    out = await handler({"url": "https://example.com"}, purpose="ignore me")
    assert out["action"] == "open_url"
    assert out["title"] == "Hi"
    assert out["status"] == 200
    assert "welcome" in out["text"]
    assert fake.calls == ["https://example.com"]


@test("4.2 handler whitespace-trims url")
async def test_handler_trims():
    fake = _FakePageProvider()
    handler = make_playwright_handler(fake)
    out = await handler({"url": "  https://example.com  "}, purpose="")
    # Default snapshot returns url=passed-in (trimmed) on the call.
    assert fake.calls == ["https://example.com"]
    assert out["action"] == "open_url"


# ---------------------------------------------------------------------------
# 5. Handler — URL from purpose fallback
# ---------------------------------------------------------------------------

@test("5.1 handler extracts URL from purpose when args has none")
async def test_handler_purpose_url():
    fake = _FakePageProvider()
    handler = make_playwright_handler(fake)
    out = await handler({}, purpose="check https://asklora.io/manifesto then report")
    assert fake.calls == ["https://asklora.io/manifesto"]
    assert out["url"] == "https://asklora.io/manifesto"


@test("5.2 args.url wins over purpose URL")
async def test_handler_args_beats_purpose():
    fake = _FakePageProvider()
    handler = make_playwright_handler(fake)
    await handler(
        {"url": "https://primary.example/"},
        purpose="also https://secondary.example mentioned here",
    )
    assert fake.calls == ["https://primary.example/"]


@test("5.3 no URL anywhere → RuntimeError")
async def test_handler_no_url():
    fake = _FakePageProvider()
    handler = make_playwright_handler(fake)
    try:
        await handler({}, purpose="")
    except RuntimeError as e:
        assert "no url" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError when no url")


@test("5.4 args.url empty string falls back to purpose")
async def test_handler_empty_args_url():
    fake = _FakePageProvider()
    handler = make_playwright_handler(fake)
    out = await handler({"url": ""}, purpose="please open https://example.com")
    assert fake.calls == ["https://example.com"]
    assert out["url"] == "https://example.com"


@test("5.5 args.url non-string falls back to purpose")
async def test_handler_non_string_args_url():
    fake = _FakePageProvider()
    handler = make_playwright_handler(fake)
    out = await handler(
        {"url": 42},  # type: ignore[dict-item]
        purpose="open https://example.com",
    )
    assert fake.calls == ["https://example.com"]
    assert out["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# 6. Handler — error propagation
# ---------------------------------------------------------------------------

@test("6.1 not-ok snapshot raises RuntimeError with the reason")
async def test_handler_snapshot_error():
    fake = _FakePageProvider(snapshots={
        "https://broken.example/": PageSnapshot(
            ok=False, url="https://broken.example/",
            error="navigation failed: net::ERR_NAME_NOT_RESOLVED",
        ),
    })
    handler = make_playwright_handler(fake)
    try:
        await handler({"url": "https://broken.example/"}, purpose="x")
    except RuntimeError as e:
        assert "ERR_NAME_NOT_RESOLVED" in str(e) or "failed" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError on not-ok snapshot")


@test("6.2 URL safety violation also reaches handler as RuntimeError (real client path)")
async def test_handler_unsafe_url_via_real_client():
    # The real PlaywrightClient calls _is_safe_url first and returns
    # ok=False — handler should turn that into a RuntimeError without
    # ever launching a browser. We use the real PlaywrightClient here
    # (not a fake) because this exercises the safety integration.
    client = PlaywrightClient()
    handler = make_playwright_handler(client)
    try:
        await handler({"url": "file:///etc/passwd"}, purpose="")
    except RuntimeError as e:
        assert "scheme" in str(e).lower() or "file" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError on unsafe url")


# ---------------------------------------------------------------------------
# 7. Real PlaywrightClient smoke (no browser launch)
# ---------------------------------------------------------------------------

@test("7.1 PlaywrightClient constructs with defaults")
def test_client_defaults():
    c = PlaywrightClient()
    assert c._headless is True
    assert c._timeout_ms == 20000
    assert c._browser_name == "chromium"


@test("7.2 PlaywrightClient clamps low timeouts up to 1s")
def test_client_min_timeout():
    c = PlaywrightClient(timeout_ms=10)
    assert c._timeout_ms == 1000


@test("7.3 PlaywrightClient.open_url rejects unsafe URL without import attempt")
async def test_client_unsafe_url_no_import():
    c = PlaywrightClient()
    snap = await c.open_url("file:///etc/passwd")
    assert snap.ok is False
    assert "scheme" in snap.error.lower()


# ---------------------------------------------------------------------------
# 8. Env builder
# ---------------------------------------------------------------------------

@test("8.1 build_playwright_client_from_env returns None when PLAYWRIGHT_ENABLED unset")
def test_env_disabled():
    saved = os.environ.pop("PLAYWRIGHT_ENABLED", None)
    try:
        assert build_playwright_client_from_env() is None
    finally:
        if saved is not None:
            os.environ["PLAYWRIGHT_ENABLED"] = saved


@test("8.2 build_playwright_client_from_env returns client when enabled")
def test_env_enabled():
    saved = os.environ.get("PLAYWRIGHT_ENABLED")
    os.environ["PLAYWRIGHT_ENABLED"] = "1"
    try:
        c = build_playwright_client_from_env()
        assert c is not None
        assert isinstance(c, PlaywrightClient)
    finally:
        if saved is None:
            os.environ.pop("PLAYWRIGHT_ENABLED", None)
        else:
            os.environ["PLAYWRIGHT_ENABLED"] = saved


@test("8.3 falsey values all disable")
def test_env_falsey():
    saved = os.environ.get("PLAYWRIGHT_ENABLED")
    try:
        for val in ("", "0", "false", "no", "off"):
            os.environ["PLAYWRIGHT_ENABLED"] = val
            assert build_playwright_client_from_env() is None, f"expected None for {val!r}"
    finally:
        if saved is None:
            os.environ.pop("PLAYWRIGHT_ENABLED", None)
        else:
            os.environ["PLAYWRIGHT_ENABLED"] = saved


@test("8.4 PLAYWRIGHT_BROWSER picks browser name")
def test_env_browser_name():
    saved_enabled = os.environ.get("PLAYWRIGHT_ENABLED")
    saved_browser = os.environ.get("PLAYWRIGHT_BROWSER")
    os.environ["PLAYWRIGHT_ENABLED"] = "1"
    os.environ["PLAYWRIGHT_BROWSER"] = "firefox"
    try:
        c = build_playwright_client_from_env()
        assert c is not None
        assert c._browser_name == "firefox"
    finally:
        if saved_enabled is None:
            os.environ.pop("PLAYWRIGHT_ENABLED", None)
        else:
            os.environ["PLAYWRIGHT_ENABLED"] = saved_enabled
        if saved_browser is None:
            os.environ.pop("PLAYWRIGHT_BROWSER", None)
        else:
            os.environ["PLAYWRIGHT_BROWSER"] = saved_browser


@test("8.5 unknown PLAYWRIGHT_BROWSER falls back to chromium")
def test_env_unknown_browser():
    saved_enabled = os.environ.get("PLAYWRIGHT_ENABLED")
    saved_browser = os.environ.get("PLAYWRIGHT_BROWSER")
    os.environ["PLAYWRIGHT_ENABLED"] = "1"
    os.environ["PLAYWRIGHT_BROWSER"] = "konqueror"
    try:
        c = build_playwright_client_from_env()
        assert c is not None
        assert c._browser_name == "chromium"
    finally:
        if saved_enabled is None:
            os.environ.pop("PLAYWRIGHT_ENABLED", None)
        else:
            os.environ["PLAYWRIGHT_ENABLED"] = saved_enabled
        if saved_browser is None:
            os.environ.pop("PLAYWRIGHT_BROWSER", None)
        else:
            os.environ["PLAYWRIGHT_BROWSER"] = saved_browser


@test("8.6 PLAYWRIGHT_HEADLESS=0 disables headless")
def test_env_headless_off():
    saved_enabled = os.environ.get("PLAYWRIGHT_ENABLED")
    saved_headless = os.environ.get("PLAYWRIGHT_HEADLESS")
    os.environ["PLAYWRIGHT_ENABLED"] = "1"
    os.environ["PLAYWRIGHT_HEADLESS"] = "0"
    try:
        c = build_playwright_client_from_env()
        assert c is not None
        assert c._headless is False
    finally:
        if saved_enabled is None:
            os.environ.pop("PLAYWRIGHT_ENABLED", None)
        else:
            os.environ["PLAYWRIGHT_ENABLED"] = saved_enabled
        if saved_headless is None:
            os.environ.pop("PLAYWRIGHT_HEADLESS", None)
        else:
            os.environ["PLAYWRIGHT_HEADLESS"] = saved_headless


@test("8.7 bad PLAYWRIGHT_TIMEOUT_MS falls back to default")
def test_env_bad_timeout():
    saved_enabled = os.environ.get("PLAYWRIGHT_ENABLED")
    saved_t = os.environ.get("PLAYWRIGHT_TIMEOUT_MS")
    os.environ["PLAYWRIGHT_ENABLED"] = "1"
    os.environ["PLAYWRIGHT_TIMEOUT_MS"] = "not-a-number"
    try:
        c = build_playwright_client_from_env()
        assert c is not None
        assert c._timeout_ms == 20000
    finally:
        if saved_enabled is None:
            os.environ.pop("PLAYWRIGHT_ENABLED", None)
        else:
            os.environ["PLAYWRIGHT_ENABLED"] = saved_enabled
        if saved_t is None:
            os.environ.pop("PLAYWRIGHT_TIMEOUT_MS", None)
        else:
            os.environ["PLAYWRIGHT_TIMEOUT_MS"] = saved_t


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_safe_https,
    test_safe_http,
    test_reject_file,
    test_reject_js,
    test_reject_localhost,
    test_reject_loopback,
    test_reject_private_ip,
    test_reject_empty,
    test_reject_no_host,
    test_extract_http,
    test_extract_https,
    test_extract_none,
    test_extract_trim,
    test_format_ok,
    test_format_error,
    test_format_empty_body,
    test_handler_args_url,
    test_handler_trims,
    test_handler_purpose_url,
    test_handler_args_beats_purpose,
    test_handler_no_url,
    test_handler_empty_args_url,
    test_handler_non_string_args_url,
    test_handler_snapshot_error,
    test_handler_unsafe_url_via_real_client,
    test_client_defaults,
    test_client_min_timeout,
    test_client_unsafe_url_no_import,
    test_env_disabled,
    test_env_enabled,
    test_env_falsey,
    test_env_browser_name,
    test_env_unknown_browser,
    test_env_headless_off,
    test_env_bad_timeout,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} Playwright client tests...")
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
