"""
EmbeddingService — generates vector embeddings for iteration text.

THE PURPOSE
===========
Without an embedding per iteration there is no semantic memory recall.
"Find threads about pricing tension" requires a vector representation of
each iteration's content. This service produces that vector.

The embedding is stored on IterationRecord.embedding and indexed by
ThreadStore.find_similar_iterations.

PROVIDER
========
Default: Google Gemini text-embedding-004 (768-dim, $0.000025/1k tokens).
Configurable via:
    GRAPHIFY_EMBEDDING_MODEL   default: text-embedding-004
    GEMINI_API_KEY / GOOGLE_API_KEY

The class swaps cleanly to a different provider — change `_call_provider`.

SAFETY
======
Every call returns None on failure (no key, network error, malformed
response, timeout). The dispatcher hook checks for None and stores the
iteration with embedding=None. The pipeline never crashes because of a
failed embedding.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

log = logging.getLogger("constellax.embedding")


DEFAULT_EMBED_MODEL = os.environ.get("GRAPHIFY_EMBEDDING_MODEL", "gemini-embedding-001")
DEFAULT_TIMEOUT_SEC = 10.0
MAX_INPUT_CHARS = 30_000     # safety bound for one call

# gemini-embedding-001 emits 3072-dim vectors by default (matryoshka model).
# Our Neo4j vector index is 1536-dim by default — see NEO4J_EMBEDDING_DIM in
# neo4j_backend.init_schema. The two MUST match or Neo4j silently drops the
# vector at index time (the property still lands on the node but doesn't
# participate in similarity search). We force 1536 here so the embed-and-
# query path is symmetric. Override via NEO4J_EMBEDDING_DIM if you want a
# different size, and update init_schema's default in tandem.
DEFAULT_OUTPUT_DIM = int(os.environ.get("NEO4J_EMBEDDING_DIM", "1536"))


@dataclass
class EmbeddingResult:
    """Outcome of one embedding call. Carries telemetry for ModelCall provenance."""
    vector:     list[float] | None
    model:      str
    tokens:     int = 0
    latency_ms: int = 0
    success:    bool = False
    error:      str | None = None


class GeminiEmbeddingService:
    """Async embedding via Gemini's `text-embedding-004` (or whichever
    GRAPHIFY_EMBEDDING_MODEL specifies). Single method: `embed(text)`."""

    def __init__(
        self,
        model: str | None = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        output_dim: int | None = None,
    ):
        self.model = model or DEFAULT_EMBED_MODEL
        self.timeout_sec = timeout_sec
        self.output_dim = output_dim or DEFAULT_OUTPUT_DIM

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string. Always returns an EmbeddingResult;
        on any failure, `vector` is None and `error` carries the reason."""
        if not text or not text.strip():
            return EmbeddingResult(vector=None, model=self.model, success=False, error="empty input")

        # Guard against accidentally embedding a 1 MB blob
        trimmed = text[:MAX_INPUT_CHARS]

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return EmbeddingResult(
                vector=None, model=self.model, success=False,
                error="no GEMINI_API_KEY or GOOGLE_API_KEY",
            )

        start = time.time()
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = await asyncio.wait_for(
                client.aio.models.embed_content(
                    model=self.model,
                    contents=trimmed,
                    # Force matching dim so vectors land in the Neo4j vector
                    # index. Without this, gemini-embedding-001 returns 3072
                    # and our 1536-dim index silently rejects them. The dict
                    # form keeps the call portable across google-genai SDK
                    # versions (newer use types.EmbedContentConfig).
                    config={"output_dimensionality": self.output_dim},
                ),
                timeout=self.timeout_sec,
            )
        except asyncio.TimeoutError:
            return EmbeddingResult(
                vector=None, model=self.model, success=False, error="timeout",
                latency_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            log.warning("embedding call failed: %s", e)
            return EmbeddingResult(
                vector=None, model=self.model, success=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.time() - start) * 1000),
            )

        latency_ms = int((time.time() - start) * 1000)
        vector = _extract_vector(response)
        if not vector:
            return EmbeddingResult(
                vector=None, model=self.model, success=False,
                error="response had no embedding vector", latency_ms=latency_ms,
            )
        # google-genai doesn't return token counts on embed; estimate.
        approx_tokens = max(1, len(trimmed) // 4)
        return EmbeddingResult(
            vector=vector, model=self.model, tokens=approx_tokens,
            latency_ms=latency_ms, success=True,
        )


def _extract_vector(response) -> list[float] | None:
    """Pull the embedding vector out of whatever shape google-genai returned.
    The SDK has changed this format across versions — be defensive."""
    # New SDK shape: response.embeddings is a list of objects with .values
    try:
        embeds = getattr(response, "embeddings", None)
        if embeds and len(embeds) > 0:
            values = getattr(embeds[0], "values", None)
            if values:
                return list(values)
    except Exception:
        pass
    # Older shape: response.embedding is a list of floats
    try:
        embedding = getattr(response, "embedding", None)
        if embedding:
            return list(embedding)
    except Exception:
        pass
    # Dict-like response (some versions)
    try:
        if isinstance(response, dict):
            if "embeddings" in response and response["embeddings"]:
                return list(response["embeddings"][0].get("values") or [])
            if "embedding" in response:
                return list(response["embedding"])
    except Exception:
        pass
    return None
