"""
pdf_extractor — extract math/code/figure-aware markdown from a PDF URL.

Jina Reader handles PDFs, but its output garbles LaTeX equations and
drops figure references, which matters when wandering agents stumble
onto arXiv physics/math papers or user-uploaded research reports. F1 in
WANDERING_ROOM_FUTURE_WORK.md.

Approach: Anthropic's Messages API natively accepts PDF documents as
base64 input. We download the PDF over httpx, b64-encode, and ask Claude
Sonnet to render it as structured markdown — preserving math as inline
or display LaTeX, code blocks as fenced markdown, and figures as
italicized brief descriptions.

  - No new vendor surface — already paying for Claude.
  - No new SDK dependency — httpx + json + base64 (all stdlib + already
    in our requirements.txt).
  - Soft-fails to Jina via the dispatching caller in extractors.py.
  - Feature-flagged on `ANTHROPIC_API_KEY` presence (optional in .env).
    Without the key, the dispatcher routes everything through Jina.

GUARDED BUDGETS:
  - Max PDF size: 32 MB (Anthropic's documented limit per request).
  - Max pages: roughly 100, also Anthropic's limit. We don't try to
    page-count before sending — the API returns a clean error if too
    large, which we soft-fail.
  - Default timeout: 60 seconds. PDFs are big; reasoning over them is
    slower than a normal Sonnet call.

ISOLATION: imports stdlib + httpx only. No matcher, no agent state, no
wandering-specific imports.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx


log = logging.getLogger("constellax.wandering.pdf_extractor")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANTHROPIC_API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_PDF_MODEL = os.environ.get(
    "ANTHROPIC_PDF_MODEL",
    "claude-sonnet-4-5",
).strip() or "claude-sonnet-4-5"

# Hard ceiling matching Anthropic's documented per-request PDF limit.
# Pages-as-images get tokenized at ~2k tokens per page; a 100-page paper
# is ~200k tokens which approaches the context window. We let Anthropic
# enforce the page limit and just bound the byte count here.
MAX_PDF_BYTES = 32 * 1024 * 1024  # 32 MB

# Output cap on the extracted markdown. Same shape as Jina's: long
# enough to capture structural content, short enough to keep the
# matcher's prompt budget honest. ~5k tokens is comfortable.
MAX_OUTPUT_CHARS = 20_000

DEFAULT_DOWNLOAD_TIMEOUT = 20.0
DEFAULT_API_TIMEOUT      = 60.0
MAX_OUTPUT_TOKENS        = 8192  # Claude Sonnet typical max for one response


# ---------------------------------------------------------------------------
# Result type — mirrors extractors.ExtractResult so the dispatching
# caller can return either path through the same interface.
# ---------------------------------------------------------------------------


@dataclass
class PdfExtractResult:
    url:        str
    body:       str = ""
    chars:      int = 0
    pages:      int = 0  # informational only; Anthropic doesn't always report
    error:      str | None = None
    latency_ms: int = 0
    ok:         bool = False


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def is_pdf_url(url: str) -> bool:
    """Cheap content-type guess from a URL alone.

    True when the URL path ends in `.pdf` (case-insensitive). Some hosts
    serve PDFs through extension-less URLs (arxiv abstract pages, e.g.),
    so a False here doesn't mean "not a PDF" — it means we couldn't tell
    from the URL string alone. The caller (extractors.py) can fall back
    to Jina; if Jina returns an opaque body, the matcher will treat it
    as no-match and the agent moves on.
    """
    if not url:
        return False
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        return False
    return path.endswith(".pdf")


def is_available() -> bool:
    """True when ANTHROPIC_API_KEY is configured. Cheap diagnostic — the
    extractors.py dispatcher reads this to decide whether to attempt
    Claude PDF input or skip straight to Jina."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------


async def _download_pdf(
    url: str, *, timeout_sec: float = DEFAULT_DOWNLOAD_TIMEOUT,
) -> tuple[bytes | None, str | None]:
    """Fetch the PDF bytes for `url`. Returns (bytes, None) on success
    or (None, error_str) on any failure.

    Hard-caps at MAX_PDF_BYTES to keep us from accidentally streaming a
    multi-GB PDF into memory. Truncation IS a real possibility for large
    bibliographies, but the wandering use case prioritises math/code
    content from the FIRST pages — truncated tail is acceptable.
    """
    try:
        async with httpx.AsyncClient(
            timeout=timeout_sec,
            follow_redirects=True,
            headers={"User-Agent": "Constellax-Wandering/1.0"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.content
    except httpx.TimeoutException:
        return None, "download_timeout"
    except httpx.HTTPStatusError as e:
        return None, f"download_http_{e.response.status_code}"
    except Exception as e:
        log.warning("pdf download failed for %s: %s", url, e)
        return None, f"download_{type(e).__name__}"

    if not data:
        return None, "download_empty"
    if len(data) > MAX_PDF_BYTES:
        log.info(
            "pdf %s exceeds MAX_PDF_BYTES (%d > %d); truncating",
            url, len(data), MAX_PDF_BYTES,
        )
        data = data[:MAX_PDF_BYTES]
    return data, None


# ---------------------------------------------------------------------------
# Anthropic API call
# ---------------------------------------------------------------------------


_PDF_EXTRACT_SYSTEM = (
    "You are a research PDF extractor for Constellax's Wandering Room. "
    "Render the user-supplied PDF as clean markdown for downstream "
    "structural-match analysis.\n\n"
    "RULES:\n"
    "- Preserve all mathematical equations as LaTeX. Inline: $...$. "
    "Display: $$...$$. Do not paraphrase math into prose.\n"
    "- Preserve code blocks as fenced markdown with the language hinted "
    "when obvious (```python, ```julia, etc.).\n"
    "- Render figures and diagrams as brief italicised one-line "
    "captions: *Figure: <what it shows>*. Do not invent details.\n"
    "- Preserve section headings (h2/h3) and paragraph breaks.\n"
    "- Skip running headers, page numbers, and reference-only pages "
    "(citations at the end of a paper are noise for our use).\n"
    "- Stay faithful to the source — no commentary, no summary, no "
    "interpretation. The downstream LLM does that.\n\n"
    "Output ONLY the rendered markdown. No preamble, no code fences "
    "wrapping the whole document. Start with the title or first "
    "section heading."
)


async def _extract_via_anthropic(
    pdf_bytes: bytes,
    *,
    timeout_sec: float = DEFAULT_API_TIMEOUT,
    model: str = ANTHROPIC_PDF_MODEL,
) -> tuple[str | None, str | None]:
    """Call Anthropic Messages API with the PDF as base64 input.

    Returns (markdown_body, None) on success, (None, error_str) on
    failure. Uses the standard `document` content block — supported on
    Claude Sonnet 4.x and later. The system prompt above instructs the
    model to preserve math, code, and figure captions verbatim.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None, "no_api_key"

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type":      "application/json",
    }
    body = {
        "model":       model,
        "max_tokens":  MAX_OUTPUT_TOKENS,
        "system":      _PDF_EXTRACT_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type":       "base64",
                            "media_type": "application/pdf",
                            "data":       base64.standard_b64encode(pdf_bytes).decode("ascii"),
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract this PDF as faithful markdown following "
                            "the system rules. Begin output now."
                        ),
                    },
                ],
            }
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.post(ANTHROPIC_API, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException:
        return None, "api_timeout"
    except httpx.HTTPStatusError as e:
        # 401 invalid key, 413 too large, 429 rate limit, 5xx upstream.
        # All soft-fail.
        status = e.response.status_code
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            detail = ""
        log.warning("anthropic pdf extract %s: %s", status, detail[:200])
        return None, f"api_http_{status}"
    except Exception as e:
        log.warning("anthropic pdf extract failed: %s", e)
        return None, f"api_{type(e).__name__}"

    # The response is a Messages-API payload. Content is a list of
    # blocks; we concatenate text blocks.
    blocks = payload.get("content") or []
    parts: list[str] = []
    for blk in blocks:
        if isinstance(blk, dict) and blk.get("type") == "text":
            text = (blk.get("text") or "").strip()
            if text:
                parts.append(text)
    body_md = "\n\n".join(parts).strip()
    if not body_md:
        return None, "empty_response"
    return body_md, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def extract_pdf_url(
    url: str,
    *,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    api_timeout:      float = DEFAULT_API_TIMEOUT,
) -> PdfExtractResult:
    """End-to-end PDF → structured markdown.

    Never raises. Soft-fails through `error` field; caller in
    extractors.py falls back to Jina on any failure.

    Cost note: a typical 10-page paper costs ~$0.10 on Sonnet 4.5 via
    direct Anthropic. The tier-2 escalation gate keeps this cheap by
    only firing on borderline-match URLs (see extractors.py).
    """
    started = time.time()

    if not url or not url.startswith(("http://", "https://")):
        return PdfExtractResult(url=url, error="invalid_url", ok=False)

    if not is_available():
        return PdfExtractResult(url=url, error="no_api_key", ok=False)

    pdf_bytes, dl_err = await _download_pdf(url, timeout_sec=download_timeout)
    if pdf_bytes is None:
        return PdfExtractResult(
            url=url, error=dl_err or "download_failed", ok=False,
            latency_ms=int((time.time() - started) * 1000),
        )

    body, api_err = await _extract_via_anthropic(
        pdf_bytes, timeout_sec=api_timeout,
    )
    if body is None:
        return PdfExtractResult(
            url=url, error=api_err or "api_failed", ok=False,
            latency_ms=int((time.time() - started) * 1000),
        )

    truncated = False
    if len(body) > MAX_OUTPUT_CHARS:
        body = body[:MAX_OUTPUT_CHARS] + "\n\n...[truncated by Constellax]"
        truncated = True

    if truncated:
        log.debug("pdf extract truncated %s to %d chars", url, MAX_OUTPUT_CHARS)

    return PdfExtractResult(
        url        = url,
        body       = body,
        chars      = len(body),
        ok         = True,
        latency_ms = int((time.time() - started) * 1000),
    )


__all__ = [
    "ANTHROPIC_PDF_MODEL",
    "MAX_PDF_BYTES",
    "MAX_OUTPUT_CHARS",
    "PdfExtractResult",
    "is_pdf_url",
    "is_available",
    "extract_pdf_url",
]
