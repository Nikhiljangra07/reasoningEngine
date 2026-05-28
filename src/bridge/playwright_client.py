"""
Playwright (browser automation) client + MCP handler.

ROLE
====
Drives a real browser to fetch a URL and extract its rendered content.
The fundamental difference from web_search: Playwright executes
JavaScript, so SPAs and JS-heavy pages render fully before we read
them. web_search hits an HTTP search index; this hits the actual page.

USE CASES
=========
"What does asklora.io look like right now?" — verify the live page.
"This URL throws a CSP error; can you see what's failing?" — inspect.
"Does this site have a /pricing page?" — navigate and report.

Constellax is a thinking partner, not a code editor — Playwright here
is READ-ONLY: navigate + extract text/title/status. No clicks, no
form-fills, no network interception. If we ever need richer actions,
that's a deliberate scope bump.

OPT-IN — PLAYWRIGHT_ENABLED
===========================
Same honest-registry contract as context7 / github (until that one
got rolled back): default OFF. Set PLAYWRIGHT_ENABLED=1 (or any
truthy value) to enable. When unset, the handler is NOT registered
and the `browser` capability stays MISSING — dispatcher surfaces a
clean missing-capability offer if triage requests browser use.

The Playwright Python package itself is an OPTIONAL dependency.
build_playwright_client_from_env() does the import inside the
function so the module can be imported on systems without playwright
installed (returns None in that case with a log line).

TESTABILITY
===========
The McpHandler is built against a small PageProvider protocol
(`open_url(url) -> PageSnapshot`), not the Playwright API directly.
Real PlaywrightClient implements that protocol via the real package;
unit tests inject a FakePageProvider with pre-canned snapshots.
This is the same pattern Context7Client uses for httpx transports —
the wrapper is honest about being thin and easy to fake.

SAFETY
======
URL validation rejects non-http(s) schemes and (best-effort) blocks
common SSRF targets (localhost, 127.x, RFC1918 private ranges, link-
local). It's not a hardened firewall — operators on Railway should
still treat the browser as a network-egress capability.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse


log = logging.getLogger("constellax.playwright")


_DEFAULT_TIMEOUT_MS = 20_000     # 20s per navigation
_DEFAULT_BROWSER = "chromium"    # cheapest to ship — Playwright supports "firefox" / "webkit" too
_DEFAULT_TEXT_CHAR_CAP = 6_000   # cap on extracted text for prompt injection
_WAIT_UNTIL = "domcontentloaded"  # don't wait for trackers/idle networks


@dataclass
class PageSnapshot:
    """Result of one navigation attempt. Never raises — ok=False is the
    failure signal so the handler can convert to a clean RuntimeError
    for fire_mcp.

    `text` is the rendered body text (post-JS), already truncated to
    _DEFAULT_TEXT_CHAR_CAP. `links` is a small sample of href targets so
    the synthesizer can suggest follow-on navigation if useful.
    """
    ok: bool
    url: str
    final_url: str = ""
    title: str = ""
    text: str = ""
    status: int = 0
    error: str = ""
    links: list[str] = field(default_factory=list)


class PageProvider(Protocol):
    """The handler depends only on this — easy to fake in tests."""
    async def open_url(self, url: str) -> PageSnapshot: ...


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------

_PRIVATE_HOSTNAMES = {"localhost", "broadcasthost", "ip6-localhost", "ip6-loopback"}


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Best-effort SSRF guard. Returns (ok, reason_if_not).

    Rejects non-http(s) schemes outright. Resolves the host to a
    literal IP only when the URL already contains one (no DNS lookup
    here — Playwright will do its own resolution, and DNS-rebinding
    SSRF protection at the bridge level is genuinely hard; this layer
    catches obvious mistakes like file:// or http://127.0.0.1).
    """
    if not isinstance(url, str) or not url.strip():
        return False, "empty url"
    try:
        parsed = urlparse(url.strip())
    except Exception as e:
        return False, f"unparsable url ({e})"
    if parsed.scheme not in ("http", "https"):
        return False, f"refusing non-http(s) scheme: {parsed.scheme!r}"
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "url missing host"
    if host in _PRIVATE_HOSTNAMES:
        return False, f"refusing local hostname: {host!r}"
    # If the host looks like a literal IP, classify it.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast:
            return False, f"refusing private/loopback IP: {host}"
    return True, ""


# ---------------------------------------------------------------------------
# Real Playwright implementation
# ---------------------------------------------------------------------------


class PlaywrightClient:
    """PageProvider backed by real Playwright. Lazy-imports the package
    inside open_url so the module loads on systems without playwright."""

    def __init__(
        self,
        *,
        headless: bool = True,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        browser_name: str = _DEFAULT_BROWSER,
        text_char_cap: int = _DEFAULT_TEXT_CHAR_CAP,
    ) -> None:
        self._headless = headless
        self._timeout_ms = max(1_000, int(timeout_ms))
        self._browser_name = browser_name
        self._text_char_cap = max(500, int(text_char_cap))

    async def open_url(self, url: str) -> PageSnapshot:
        ok, reason = _is_safe_url(url)
        if not ok:
            return PageSnapshot(ok=False, url=url, error=reason)

        # Lazy-import — keeps `import server` working even when playwright
        # is not installed (only matters at handler-fire time).
        try:
            from playwright.async_api import async_playwright  # type: ignore[import-not-found]
        except ImportError:
            return PageSnapshot(
                ok=False, url=url,
                error="playwright not installed (pip install playwright && playwright install chromium)",
            )

        try:
            async with async_playwright() as pw:
                browser_factory = getattr(pw, self._browser_name, None)
                if browser_factory is None:
                    return PageSnapshot(
                        ok=False, url=url,
                        error=f"unknown browser: {self._browser_name!r}",
                    )
                browser = await browser_factory.launch(headless=self._headless)
                try:
                    context = await browser.new_context()
                    page = await context.new_page()
                    page.set_default_timeout(self._timeout_ms)
                    try:
                        response = await page.goto(url, wait_until=_WAIT_UNTIL)
                    except Exception as e:
                        return PageSnapshot(
                            ok=False, url=url, error=f"navigation failed: {e}",
                        )
                    status = response.status if response else 0
                    final_url = page.url
                    title = await page.title()
                    body_text = await page.evaluate(
                        "() => document.body ? document.body.innerText : ''"
                    )
                    links = await page.evaluate(
                        """() => Array.from(document.querySelectorAll('a[href]'))
                                .slice(0, 25)
                                .map(a => a.href)
                                .filter(h => h.startsWith('http'))"""
                    )
                    text = _truncate(str(body_text or ""), self._text_char_cap)
                    return PageSnapshot(
                        ok=True,
                        url=url,
                        final_url=str(final_url or url),
                        title=str(title or "")[:200],
                        text=text,
                        status=int(status),
                        links=[str(h) for h in (links or [])][:25],
                    )
                finally:
                    await browser.close()
        except Exception as e:
            return PageSnapshot(
                ok=False, url=url, error=f"playwright runtime error: {e}",
            )


# ---------------------------------------------------------------------------
# MCP handler
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


_URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>\"'\)]+", re.IGNORECASE)


def _extract_url_from_purpose(purpose: str) -> str:
    """Pull the first http(s) URL out of a free-text purpose string."""
    if not purpose:
        return ""
    m = _URL_IN_TEXT_RE.search(purpose)
    return m.group(0) if m else ""


def _format_snapshot_text(snap: PageSnapshot) -> str:
    """Render a snapshot as the prompt-injectable text field. Includes
    the title + status + body excerpt; the synthesizer reads this as
    "what the live page actually looks like."""
    if not snap.ok:
        return f"(browser error: {snap.error})"
    parts: list[str] = []
    parts.append(f"URL: {snap.final_url or snap.url}")
    if snap.title:
        parts.append(f"Title: {snap.title}")
    if snap.status:
        parts.append(f"HTTP status: {snap.status}")
    parts.append("")
    parts.append(snap.text or "(empty page body)")
    return "\n".join(parts)


def make_playwright_handler(client: PageProvider):
    """Return an async McpHandler bound to `client`.

    Args contract (all optional):
      {
        "url": "https://example.com"   # primary input
      }

    Fallback: scan `purpose` for an http(s) URL. If neither yields a
    URL, raise — fire_mcp surfaces that as a clean handler failure
    (ok=False) instead of opening a browser to nowhere.
    """

    async def playwright_handler(args: dict, purpose: str) -> dict:
        url = (args.get("url") if args else None) or ""
        if not isinstance(url, str) or not url.strip():
            url = _extract_url_from_purpose(purpose)
        url = (url or "").strip()
        if not url:
            raise RuntimeError(
                "Playwright handler: no url in args, no url in purpose"
            )

        snap = await client.open_url(url)
        if not snap.ok:
            raise RuntimeError(f"Playwright open_url failed: {snap.error}")

        return {
            "text":      _format_snapshot_text(snap),
            "url":       snap.url,
            "final_url": snap.final_url,
            "title":     snap.title,
            "status":    snap.status,
            "links":     snap.links,
            "action":    "open_url",
        }

    return playwright_handler


# ---------------------------------------------------------------------------
# Env builder
# ---------------------------------------------------------------------------


def build_playwright_client_from_env() -> PlaywrightClient | None:
    """Build a PlaywrightClient when PLAYWRIGHT_ENABLED is truthy, else None.

    Honors:
      PLAYWRIGHT_ENABLED        — gate (truthy enables)
      PLAYWRIGHT_BROWSER        — chromium | firefox | webkit (default chromium)
      PLAYWRIGHT_HEADLESS       — default 1 (truthy)
      PLAYWRIGHT_TIMEOUT_MS     — per-navigation timeout (default 20000)
      PLAYWRIGHT_TEXT_CHAR_CAP  — extracted-text cap (default 6000)

    Returns None — and the wiring layer skips registering the handler —
    when PLAYWRIGHT_ENABLED is unset/false. Honest-registry contract:
    `browser` stays MISSING unless a handler is actually wired.
    """
    enabled = os.environ.get("PLAYWRIGHT_ENABLED", "").strip().lower()
    if enabled in ("", "0", "false", "no", "off"):
        return None

    browser = (os.environ.get("PLAYWRIGHT_BROWSER", "") or _DEFAULT_BROWSER).strip().lower()
    if browser not in ("chromium", "firefox", "webkit"):
        log.warning(
            "PLAYWRIGHT_BROWSER=%r is not chromium/firefox/webkit; defaulting to chromium",
            browser,
        )
        browser = _DEFAULT_BROWSER

    headless_raw = os.environ.get("PLAYWRIGHT_HEADLESS", "1").strip().lower()
    headless = headless_raw not in ("0", "false", "no", "off")

    try:
        timeout_ms = int(os.environ.get("PLAYWRIGHT_TIMEOUT_MS", str(_DEFAULT_TIMEOUT_MS)))
    except (TypeError, ValueError):
        timeout_ms = _DEFAULT_TIMEOUT_MS

    try:
        text_cap = int(os.environ.get("PLAYWRIGHT_TEXT_CHAR_CAP", str(_DEFAULT_TEXT_CHAR_CAP)))
    except (TypeError, ValueError):
        text_cap = _DEFAULT_TEXT_CHAR_CAP

    return PlaywrightClient(
        headless=headless,
        timeout_ms=timeout_ms,
        browser_name=browser,
        text_char_cap=text_cap,
    )


__all__ = [
    "PageProvider",
    "PageSnapshot",
    "PlaywrightClient",
    "make_playwright_handler",
    "build_playwright_client_from_env",
]
