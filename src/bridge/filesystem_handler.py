"""
Filesystem MCP handler — reads from per-request `attached_files` payload.

ROLE
====
The backend NEVER reads the user's local filesystem directly. The IDE
extension (constellax-ui in the VSCode host) reads files via the IDE's
own scoped file APIs and attaches them to the request body. This handler
consumes that attached payload and folds the file contents into the
prompt as an MCP CONTEXT block.

Why this shape:
  - Zero attack surface on the server (no path traversal possible —
    the server has no file-system access path at all).
  - The IDE's existing user-consent model handles the "is the user OK
    with us reading these files?" question. The MCP picker in
    constellax-ui surfaces filesystem as an opt-in toggle; when toggled
    off, the extension simply doesn't attach anything.
  - Per-request scoping: each request brings its own files; no
    cross-request state, no shared cache.

REQUEST PAYLOAD CONTRACT
========================
Server.py extracts `attached_files` from the request JSON body and
passes them to make_filesystem_handler(). Expected shape:

    "attached_files": [
        {
            "path":     "src/auth.py",            # required, repo-relative or absolute
            "content":  "def login(): ...",       # required, the file body
            "language": "python"                  # optional, code-fence hint
        },
        ...
    ]

Handler args (optional):
    {"path": "src/auth.py"}    # narrow to one of the attached files

When no args["path"] is set, the handler returns ALL attached files
concatenated. When `path` is provided, only the matching attachment is
rendered. No match raises RuntimeError → fire_mcp turns into ok=False.

LIMITS
======
Each file's content is truncated to FILE_MAX_CHARS to keep the prompt
budget bounded; the rendered block is further capped by TOTAL_MAX_CHARS.
Both are constants you can tune below as production usage grows.
"""

from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger("constellax.filesystem")


FILE_MAX_CHARS = 12_000
TOTAL_MAX_CHARS = 40_000


def _truncate(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "\n…[truncated]"


def _normalize_files(attached_files: Any) -> list[dict[str, Any]]:
    """Validate + filter the request's attached_files payload.

    Drops entries missing required fields rather than raising — a single
    bad attachment shouldn't poison an otherwise-valid request.
    """
    if not isinstance(attached_files, list):
        return []
    valid: list[dict[str, Any]] = []
    for item in attached_files:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content")
        if not isinstance(path, str) or not path.strip():
            continue
        if not isinstance(content, str):
            continue
        valid.append({
            "path": path.strip(),
            "content": content,
            "language": (item.get("language") or "").strip() or None,
        })
    return valid


def _render_files(files: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    running = 0
    for f in files:
        fence = f.get("language") or ""
        content = _truncate(f.get("content") or "", FILE_MAX_CHARS)
        block = f"### {f['path']}\n```{fence}\n{content}\n```"
        if running + len(block) > TOTAL_MAX_CHARS and parts:
            parts.append("\n…[remaining files omitted to fit prompt budget]")
            break
        parts.append(block)
        running += len(block)
    return "\n\n".join(parts)


def make_filesystem_handler(attached_files: Any):
    """Return a request-scoped McpHandler that reads from attached_files.

    `attached_files` is captured in the closure — each handler instance
    is bound to one request's payload. The factory does the validation
    once so per-call work is just dict lookup + rendering.
    """
    files = _normalize_files(attached_files)

    async def filesystem_handler(args: dict, purpose: str) -> dict:
        # Narrow to a specific path when args ask for it.
        target = (args.get("path") if args else None) or ""
        if target:
            matching = [f for f in files if f["path"] == target]
            if not matching:
                # Surface a useful diagnostic instead of crashing —
                # the LLM can explain "you asked for X but it wasn't
                # in the attached set."
                attached_names = [f["path"] for f in files]
                raise RuntimeError(
                    f"Filesystem handler: path {target!r} not in attached "
                    f"files (attached: {attached_names})"
                )
            selected = matching
        else:
            selected = files

        if not selected:
            raise RuntimeError("Filesystem handler: no files attached to this request")

        text = _render_files(selected)
        return {
            "text": text,
            "data": {
                "files": [{"path": f["path"], "size": len(f["content"])} for f in selected],
                "total_attached": len(files),
                "rendered": len(selected),
            },
            "action": "read_attached_files",
        }

    return filesystem_handler


def normalize_attached_files(attached_files: Any) -> list[dict[str, Any]]:
    """Public form of _normalize_files for server-side request validation.

    Server.py uses this to (a) decide whether to flip the filesystem
    capability to AVAILABLE and (b) cheaply validate the request payload
    before constructing the handler.
    """
    return _normalize_files(attached_files)
