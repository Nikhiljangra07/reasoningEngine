"""
Context7 (library docs) client + MCP handler.

ROLE
====
Provides on-demand library and API documentation so the reasoning engine
can answer "how do I use library X?" or "what's the contract of feature Y
in version Z?" with current, citable docs instead of training-cutoff
guesses.

Context7 (https://context7.com, by Upstash) hosts a free-tier docs index
of thousands of OSS libraries. Their MCP server (npx @upstash/context7-mcp)
shells out to a small HTTP API that we hit directly here in Python — no
Node.js subprocess, no MCP-over-stdio transport. The handler interface
matches the McpHandler shape registered through src.mcp_router so it
flows through fire_mcp uniformly with every other capability.

OPT-IN — CONTEXT7_ENABLED
=========================
Context7's basic endpoints don't require an API key, but Constellax
refuses to make outbound calls without an explicit operator decision.
Set CONTEXT7_ENABLED=1 (or any truthy value) to enable. When unset, the
handler is NOT registered and the `docs` capability stays MISSING — the
dispatcher surfaces a missing-capability offer when triage requests doc
lookup. Same honest-registry contract as github_client.

CONFIGURABLE API ROOT
=====================
Their endpoint paths may evolve. CONTEXT7_API_ROOT (default
`https://context7.com/api/v1`) lets operators override without code
changes. Robust failure handling: bad endpoint, network error, or
schema drift all become structured ok=False results, never crashes.

ENDPOINTS USED
==============
  GET {api_root}/search?query=<text>            — find a library_id
  GET {api_root}/{library_id}?topic=<text>      — fetch focused docs
                                                  &tokens=<n> (optional)

Token budget guard caps requested doc tokens at 8000 by default — keeps a
runaway library response from eating the prompt budget. Configurable per
handler call via args["tokens"].
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx


log = logging.getLogger("constellax.context7")


_DEFAULT_API_ROOT = "https://context7.com/api/v1"
_DEFAULT_TIMEOUT_S = 12.0
_DEFAULT_USER_AGENT = "constellax-reasoning-engine/1.0"
_DEFAULT_TOKENS = 4000
_MAX_TOKENS = 8000
_MAX_SUMMARY_CHARS = 6000   # cap on final text injected into prompt


@dataclass
class Context7Result:
    """Outcome of one Context7 API call.

    ok=True for any 2xx with parsable body. 404 (library not found) is
    ok=False with a clear message so the handler can surface "no docs
    match" without crashing.
    """
    ok: bool
    status: int
    data: Any = None
    error: str = ""


class Context7Client:
    """Async, read-only Context7 client. Construct once per process."""

    def __init__(
        self,
        *,
        api_root: str = _DEFAULT_API_ROOT,
        api_key: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        user_agent: str = _DEFAULT_USER_AGENT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_root = api_root.rstrip("/")
        self._timeout_s = timeout_s
        self._headers = {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }
        # API key is optional today (public endpoints) but supported so
        # future paid-tier accounts can plug in without code changes.
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._transport = transport

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Context7Result:
        url = f"{self._api_root}{path}" if path.startswith("/") else f"{self._api_root}/{path}"
        client_kwargs: dict[str, Any] = {"timeout": self._timeout_s}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        try:
            async with httpx.AsyncClient(**client_kwargs) as http:
                resp = await http.get(url, headers=self._headers, params=params or {})
        except httpx.TimeoutException:
            return Context7Result(ok=False, status=0, error="context7 request timed out")
        except httpx.RequestError as e:
            return Context7Result(ok=False, status=0, error=f"context7 request failed: {e}")

        if resp.status_code == 200:
            # Some Context7 endpoints return JSON, others may return text
            # with `Content-Type: text/plain`. Handle both.
            ctype = resp.headers.get("Content-Type", "").lower()
            if "application/json" in ctype:
                try:
                    return Context7Result(ok=True, status=200, data=resp.json())
                except ValueError:
                    return Context7Result(
                        ok=False, status=200,
                        error="context7 returned malformed JSON",
                    )
            # Text response — wrap into a dict with a `text` field so the
            # caller can treat both shapes uniformly.
            return Context7Result(
                ok=True, status=200, data={"text": resp.text},
            )

        if resp.status_code == 404:
            return Context7Result(ok=False, status=404, error="not found")
        if resp.status_code in (401, 403):
            return Context7Result(
                ok=False, status=resp.status_code,
                error=f"context7 auth/permission denied ({resp.status_code})",
            )
        if resp.status_code == 429:
            return Context7Result(
                ok=False, status=429, error="context7 rate limit exhausted",
            )
        return Context7Result(
            ok=False, status=resp.status_code,
            error=f"context7 HTTP {resp.status_code}",
        )

    async def search_libraries(self, query: str) -> Context7Result:
        """Find a library_id from a free-text query (e.g. 'stripe python sdk')."""
        return await self._get("/search", params={"query": query})

    async def get_library_docs(
        self,
        library_id: str,
        *,
        topic: str | None = None,
        tokens: int | None = None,
    ) -> Context7Result:
        """Fetch focused docs for a library by id.

        `topic` narrows the docs to a specific feature (e.g. 'idempotency').
        `tokens` caps the response length; defaults to _DEFAULT_TOKENS and
        is clamped to _MAX_TOKENS.
        """
        params: dict[str, Any] = {}
        if topic:
            params["topic"] = topic
        params["tokens"] = max(1, min(int(tokens or _DEFAULT_TOKENS), _MAX_TOKENS))
        # library_id is treated as the path segment — caller can pass
        # `vercel/next.js` or similar slash-separated identifiers, which
        # Context7 expects as a single path component.
        path = "/" + library_id.lstrip("/")
        return await self._get(path, params=params)


# ---------------------------------------------------------------------------
# MCP handler
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int = _MAX_SUMMARY_CHARS) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _extract_text_from_payload(data: Any) -> str:
    """Context7 responses vary in shape — normalize to a text string."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        # Common shapes we've seen / expect:
        #   {"text": "..."}                       — plain text wrapper
        #   {"docs": "..."} / {"content": "..."}  — alternate keys
        #   {"snippets": [{"text": "..."}, ...]}  — structured chunks
        for key in ("text", "docs", "content", "documentation"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v
        snippets = data.get("snippets") or data.get("results")
        if isinstance(snippets, list):
            parts = []
            for s in snippets:
                if isinstance(s, dict):
                    t = s.get("text") or s.get("content") or s.get("snippet")
                    if isinstance(t, str):
                        parts.append(t)
            if parts:
                return "\n\n".join(parts)
    if isinstance(data, list):
        # List of results from /search — render as a brief index.
        parts = []
        for item in data:
            if isinstance(item, dict):
                title = item.get("name") or item.get("library_id") or item.get("title") or "?"
                desc = item.get("description") or item.get("summary") or ""
                parts.append(f"- {title}" + (f" — {desc}" if desc else ""))
        if parts:
            return "Available libraries:\n" + "\n".join(parts)
    return ""


def make_context7_handler(client: Context7Client):
    """Return an async McpHandler bound to `client`.

    Args contract (all optional):
      {
        "library_id": "vercel/next.js"      # explicit library; skips search
        "query":      "stripe python sdk",  # text to search libraries by
        "topic":      "idempotency keys",   # narrow the docs
        "tokens":     2000                  # cap on response tokens
      }

    Coarse fallback when no args: use `purpose` as the search query, take
    the first hit, fetch its docs.
    """

    async def context7_handler(args: dict, purpose: str) -> dict:
        library_id = (args.get("library_id") if args else None) or ""
        query = (args.get("query") if args else None) or ""
        topic = (args.get("topic") if args else None) or None
        tokens = args.get("tokens") if args else None

        # Path 1: explicit library_id → fetch docs directly
        if library_id:
            res = await client.get_library_docs(
                library_id, topic=topic, tokens=tokens,
            )
            if not res.ok:
                raise RuntimeError(f"Context7 get_library_docs failed: {res.error}")
            text = _truncate(_extract_text_from_payload(res.data))
            return {
                "text": text or "(empty docs response)",
                "data": res.data,
                "library_id": library_id,
                "action": "get_library_docs",
            }

        # Path 2: query → search → fetch first hit's docs
        search_q = query or purpose.strip()
        if not search_q:
            raise RuntimeError("Context7 handler: no library_id, no query, no purpose")

        search_res = await client.search_libraries(search_q)
        if not search_res.ok:
            raise RuntimeError(f"Context7 search failed: {search_res.error}")

        # The search payload shape is best-effort: try to find a usable id.
        hit_id: str | None = None
        data = search_res.data
        if isinstance(data, dict):
            # {"results": [{"library_id": "..."}]} or {"libraries": [...]}
            for key in ("results", "libraries", "hits"):
                items = data.get(key)
                if isinstance(items, list) and items:
                    first = items[0]
                    if isinstance(first, dict):
                        hit_id = (
                            first.get("library_id")
                            or first.get("id")
                            or first.get("name")
                        )
                        if hit_id:
                            break
        elif isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                hit_id = first.get("library_id") or first.get("id") or first.get("name")

        if not hit_id:
            # Surface the raw search result instead of failing — better
            # than silently swallowing useful info.
            text = _truncate(_extract_text_from_payload(data))
            return {
                "text": text or f"No matching library for `{search_q}`.",
                "data": data,
                "action": "search_libraries",
            }

        docs_res = await client.get_library_docs(hit_id, topic=topic, tokens=tokens)
        if not docs_res.ok:
            raise RuntimeError(
                f"Context7 fetched library_id={hit_id!r} but get_library_docs "
                f"failed: {docs_res.error}"
            )
        text = _truncate(_extract_text_from_payload(docs_res.data))
        return {
            "text": text or "(empty docs response)",
            "data": docs_res.data,
            "library_id": hit_id,
            "action": "search_then_get_docs",
        }

    return context7_handler


def build_context7_client_from_env() -> Context7Client | None:
    """Build a Context7Client when CONTEXT7_ENABLED is truthy, else None.

    Honors CONTEXT7_API_ROOT (default https://context7.com/api/v1) and
    optional CONTEXT7_API_KEY (paid-tier hook — not required today).

    Returns None — and the wiring layer skips registering the handler —
    when CONTEXT7_ENABLED is unset/false. Same honest-registry contract
    as GitHub.
    """
    enabled = os.environ.get("CONTEXT7_ENABLED", "").strip().lower()
    if enabled in ("", "0", "false", "no", "off"):
        return None
    api_root = os.environ.get("CONTEXT7_API_ROOT", "").strip() or _DEFAULT_API_ROOT
    api_key = os.environ.get("CONTEXT7_API_KEY", "").strip() or None
    return Context7Client(api_root=api_root, api_key=api_key)
