"""
GitHub read-only client + MCP handler.

ROLE
====
Provides Constellax with structured read access to GitHub state — PRs,
issues, repo contents, recent commits — so the reasoning engine can
ground answers about real repository state instead of guessing.

Constellax never writes. This module exposes only GET endpoints; mutation
methods are absent by design and refused at construction time.

WIRING
======
1. `GitHubClient` — thin async httpx wrapper over api.github.com.
   Auth via GITHUB_TOKEN (Personal Access Token, classic or fine-grained
   with `repo` read scope for private repos; public repos need no scope
   beyond authenticated rate limit).

2. `make_github_handler(client)` — returns an McpHandler compatible with
   src.mcp_router. The handler reads the request's purpose text and dispatches
   to a sensible GitHub call (issue/PR search by default; explicit args
   override). Returns a dict with a `text` summary for prompt injection
   and `data` for structured access.

3. Server wiring (server.py): if GITHUB_TOKEN is set at startup, build the
   client, build the handler, register it on the module-level
   McpHandlerRegistry, and mark the `github` capability AVAILABLE in the
   registry. When the env var is absent, the registry stays MISSING and
   the dispatcher surfaces a missing-capability offer when triage requests
   github context.

RATE LIMITS
===========
Authenticated GitHub API: 5000 req/hour. The client logs the remaining
budget on every call (X-RateLimit-Remaining header). At ~50K msgs/month
× ~0.2 github calls/msg = ~14 calls/hour expected — comfortably under.

ON 429 / 5xx the client returns a structured failure dict rather than
raising; the handler surfaces it as ok=False so the dispatcher's
missing-capability offer path fires.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx


log = logging.getLogger("constellax.github")


_API_ROOT = "https://api.github.com"
_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_USER_AGENT = "constellax-reasoning-engine/1.0"


# Conservative cap on response sizes injected into LLM prompts.
# Keeps a runaway repo (e.g., 1000-file listing) from poisoning the
# prompt; the LLM gets the most relevant slice + a "truncated" note.
_MAX_ITEMS_DEFAULT = 5
_MAX_BODY_CHARS = 1200       # per issue/PR body
_MAX_FILE_CHARS = 8000       # per file content


@dataclass
class GitHubResult:
    """Outcome of one GitHub API call.

    `ok` is True for any 2xx response. 404 (resource not found) is also
    considered ok=False with a clear error — the handler can format a
    "no such PR" response instead of crashing.
    """
    ok: bool
    status: int
    data: Any = None
    error: str = ""


class GitHubClient:
    """Async, read-only GitHub REST client. Construct once per process."""

    def __init__(
        self,
        token: str | None = None,
        *,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        user_agent: str = _DEFAULT_USER_AGENT,
        api_root: str = _API_ROOT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # token=None is allowed but heavily rate-limited (60 req/hour
        # unauthenticated). In server context the wiring layer refuses
        # to register the handler when no token is set, so this branch
        # only matters for tests / local exploration.
        self._token = (token or "").strip() or None
        self._timeout_s = timeout_s
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": user_agent,
        }
        if self._token:
            self._headers["Authorization"] = f"Bearer {self._token}"
        self._api_root = api_root.rstrip("/")
        # Optional httpx transport for tests — production callers leave
        # this None and get the default real network transport.
        self._transport = transport

    # -----------------------------------------------------------------------
    # Core HTTP — shared by all read methods
    # -----------------------------------------------------------------------

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> GitHubResult:
        url = f"{self._api_root}{path}" if path.startswith("/") else f"{self._api_root}/{path}"
        client_kwargs: dict[str, Any] = {"timeout": self._timeout_s}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        try:
            async with httpx.AsyncClient(**client_kwargs) as http:
                resp = await http.get(url, headers=self._headers, params=params or {})
        except httpx.TimeoutException:
            return GitHubResult(ok=False, status=0, error="github request timed out")
        except httpx.RequestError as e:
            return GitHubResult(ok=False, status=0, error=f"github request failed: {e}")

        # Surface remaining quota for observability — useful for the
        # health dashboard if/when it picks this up later.
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            log.debug("github rate-limit remaining=%s url=%s", remaining, url)

        if resp.status_code == 200:
            try:
                return GitHubResult(ok=True, status=200, data=resp.json())
            except ValueError:
                return GitHubResult(
                    ok=False, status=200,
                    error="github returned non-JSON 200 body",
                )
        if resp.status_code == 404:
            return GitHubResult(ok=False, status=404, error="not found")
        if resp.status_code == 401:
            return GitHubResult(ok=False, status=401, error="github token rejected (401)")
        if resp.status_code == 403:
            # GitHub uses 403 for rate-limit AND for permission denied.
            # Disambiguate by header presence.
            if resp.headers.get("X-RateLimit-Remaining") == "0":
                return GitHubResult(ok=False, status=403, error="github rate limit exhausted")
            return GitHubResult(ok=False, status=403, error="github permission denied")
        return GitHubResult(
            ok=False, status=resp.status_code,
            error=f"github HTTP {resp.status_code}",
        )

    # -----------------------------------------------------------------------
    # Read endpoints (no writes — by design)
    # -----------------------------------------------------------------------

    async def get_repo(self, owner: str, repo: str) -> GitHubResult:
        """Repo metadata: description, default branch, language, topics."""
        return await self._get(f"/repos/{owner}/{repo}")

    async def get_pr(self, owner: str, repo: str, number: int) -> GitHubResult:
        return await self._get(f"/repos/{owner}/{repo}/pulls/{number}")

    async def get_issue(self, owner: str, repo: str, number: int) -> GitHubResult:
        return await self._get(f"/repos/{owner}/{repo}/issues/{number}")

    async def search_issues(
        self,
        query: str,
        *,
        per_page: int = _MAX_ITEMS_DEFAULT,
    ) -> GitHubResult:
        """Issue+PR search. Supports GitHub's q= grammar (e.g. 'repo:owner/r is:open')."""
        per_page = max(1, min(per_page, 30))
        return await self._get(
            "/search/issues",
            params={"q": query, "per_page": per_page},
        )

    async def list_recent_commits(
        self,
        owner: str,
        repo: str,
        *,
        per_page: int = _MAX_ITEMS_DEFAULT,
    ) -> GitHubResult:
        per_page = max(1, min(per_page, 30))
        return await self._get(
            f"/repos/{owner}/{repo}/commits",
            params={"per_page": per_page},
        )

    async def get_file_contents(
        self,
        owner: str,
        repo: str,
        path: str,
        *,
        ref: str | None = None,
    ) -> GitHubResult:
        # `path` is repo-relative. ref is branch/tag/sha; default is default branch.
        params: dict[str, Any] = {}
        if ref:
            params["ref"] = ref
        return await self._get(f"/repos/{owner}/{repo}/contents/{path}", params=params or None)


# ---------------------------------------------------------------------------
# MCP handler — wraps the client into the McpHandler shape.
#
# Coarse but useful contract: if `args` contains structured fields (owner,
# repo, action, number, etc.), the handler dispatches precisely. Otherwise,
# it falls back to searching issues+PRs across GitHub using the
# `purpose` text as the query. Lets the system function on day one even
# before triage learns to populate structured args.
# ---------------------------------------------------------------------------


# "owner/repo" → ("owner", "repo"). Defensive against trailing periods /
# commas the LLM might tack on.
_OWNER_REPO_RE = re.compile(r"\b([A-Za-z0-9][\w.-]*)/([A-Za-z0-9][\w.-]+?)(?=[\s.,;:!?\"')\]]|$)")
_ISSUE_NUMBER_RE = re.compile(r"#(\d+)|(?:issue|pr|pull request)\s+#?(\d+)", re.I)


def _truncate(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _summarize_issue(item: dict) -> str:
    """Render a single issue/PR record into a one-paragraph human summary."""
    kind = "PR" if "pull_request" in item else "issue"
    repo_full = (item.get("repository_url") or "").rsplit("/", 2)[-2:]
    repo_label = "/".join(repo_full) if len(repo_full) == 2 else "?"
    title = item.get("title") or "(untitled)"
    number = item.get("number") or "?"
    state = item.get("state") or "?"
    body = _truncate(item.get("body") or "", _MAX_BODY_CHARS)
    head = f"[{repo_label}#{number} — {kind} {state}] {title}"
    return head + ("\n" + body if body else "")


def _summarize_pr(data: dict) -> str:
    title = data.get("title") or "(untitled)"
    state = data.get("state") or "?"
    merged = data.get("merged_at") is not None
    merge_tag = " (merged)" if merged else ""
    body = _truncate(data.get("body") or "", _MAX_BODY_CHARS)
    head = f"PR #{data.get('number')} — {state}{merge_tag}: {title}"
    return head + ("\n" + body if body else "")


def _summarize_issue_single(data: dict) -> str:
    title = data.get("title") or "(untitled)"
    state = data.get("state") or "?"
    body = _truncate(data.get("body") or "", _MAX_BODY_CHARS)
    head = f"Issue #{data.get('number')} — {state}: {title}"
    return head + ("\n" + body if body else "")


def _summarize_repo(data: dict) -> str:
    full = data.get("full_name") or "?"
    desc = data.get("description") or ""
    lang = data.get("language") or ""
    branch = data.get("default_branch") or "?"
    topics = data.get("topics") or []
    pieces = [f"{full} — default branch {branch}"]
    if lang:
        pieces.append(f"language: {lang}")
    if desc:
        pieces.append(f"description: {desc}")
    if topics:
        pieces.append(f"topics: {', '.join(topics[:6])}")
    return "\n".join(pieces)


def _summarize_search(items: list[dict], total: int, query: str) -> str:
    n = len(items)
    head = f"GitHub search for `{query}` — {total} total, showing {n}"
    if not items:
        return head + " (no matches)"
    body = "\n\n".join(_summarize_issue(it) for it in items)
    return head + "\n\n" + body


def _extract_owner_repo(text: str) -> tuple[str, str] | None:
    m = _OWNER_REPO_RE.search(text or "")
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    # Filter out obvious false positives: file paths like "src/handlers"
    # tend to have lowercase-only second segments without dots; real repo
    # names also typically don't end in source-file extensions.
    if repo.endswith((".py", ".ts", ".tsx", ".js", ".md", ".json")):
        return None
    return owner, repo


def _extract_issue_number(text: str) -> int | None:
    m = _ISSUE_NUMBER_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def make_github_handler(client: GitHubClient):
    """Return an async McpHandler bound to `client`.

    The handler inspects (args, purpose) and dispatches to the right
    GitHub call. Always returns a dict with at least `text`; on failure
    raises a RuntimeError so fire_mcp converts it to ok=False with the
    error in blocked_reason.
    """

    async def github_handler(args: dict, purpose: str) -> dict:
        action = (args.get("action") or "").strip().lower() if args else ""
        owner = args.get("owner") if args else None
        repo = args.get("repo") if args else None
        number = args.get("number") if args else None
        query = args.get("query") if args else None

        # 1) Explicit get_pr / get_issue / get_repo when fully specified.
        if action == "get_pr" and owner and repo and number:
            res = await client.get_pr(owner, repo, int(number))
            if not res.ok:
                raise RuntimeError(f"GitHub get_pr failed: {res.error}")
            return {"text": _summarize_pr(res.data), "data": res.data, "action": "get_pr"}

        if action == "get_issue" and owner and repo and number:
            res = await client.get_issue(owner, repo, int(number))
            if not res.ok:
                raise RuntimeError(f"GitHub get_issue failed: {res.error}")
            return {"text": _summarize_issue_single(res.data), "data": res.data, "action": "get_issue"}

        if action == "get_repo" and owner and repo:
            res = await client.get_repo(owner, repo)
            if not res.ok:
                raise RuntimeError(f"GitHub get_repo failed: {res.error}")
            return {"text": _summarize_repo(res.data), "data": res.data, "action": "get_repo"}

        if action == "search_issues" and query:
            res = await client.search_issues(query)
            if not res.ok:
                raise RuntimeError(f"GitHub search_issues failed: {res.error}")
            items = (res.data or {}).get("items") or []
            total = (res.data or {}).get("total_count") or len(items)
            return {
                "text": _summarize_search(items, total, query),
                "data": items,
                "action": "search_issues",
            }

        # 2) Heuristic dispatch from purpose text — no structured args.
        purpose_text = purpose or ""
        repo_match = _extract_owner_repo(purpose_text)
        issue_num = _extract_issue_number(purpose_text)

        if repo_match and issue_num:
            # User mentioned owner/repo + #N — assume it's a PR or issue.
            # Try PR first (covers most "what's PR #N status" queries);
            # the API returns 404 cleanly if it's actually an issue.
            owner_, repo_ = repo_match
            res = await client.get_pr(owner_, repo_, issue_num)
            if res.ok:
                return {"text": _summarize_pr(res.data), "data": res.data, "action": "get_pr"}
            # Fall through to issue lookup
            res = await client.get_issue(owner_, repo_, issue_num)
            if res.ok:
                return {"text": _summarize_issue_single(res.data), "data": res.data, "action": "get_issue"}
            raise RuntimeError(f"GitHub: no PR or issue #{issue_num} in {owner_}/{repo_}")

        if repo_match:
            # owner/repo mentioned but no number — fetch repo metadata.
            owner_, repo_ = repo_match
            res = await client.get_repo(owner_, repo_)
            if res.ok:
                return {"text": _summarize_repo(res.data), "data": res.data, "action": "get_repo"}
            raise RuntimeError(f"GitHub get_repo failed for {owner_}/{repo_}: {res.error}")

        # 3) Last resort — search the issue+PR corpus for the purpose text.
        search_q = query or purpose_text.strip()
        if not search_q:
            raise RuntimeError("GitHub handler: no query and no owner/repo could be extracted")
        res = await client.search_issues(search_q)
        if not res.ok:
            raise RuntimeError(f"GitHub search fallback failed: {res.error}")
        items = (res.data or {}).get("items") or []
        total = (res.data or {}).get("total_count") or len(items)
        return {
            "text": _summarize_search(items, total, search_q),
            "data": items,
            "action": "search_issues",
        }

    return github_handler


def build_github_client_from_env(env_var: str = "GITHUB_TOKEN") -> GitHubClient | None:
    """Build a GitHubClient from GITHUB_TOKEN, or return None when unset.

    Returning None is the contract the wiring layer uses to decide
    whether to register the handler and flip the capability registry
    status to AVAILABLE.
    """
    token = os.environ.get(env_var, "").strip()
    if not token:
        return None
    return GitHubClient(token=token)
