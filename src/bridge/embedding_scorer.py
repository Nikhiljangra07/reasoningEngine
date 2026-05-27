"""
EmbeddingScorer — semantic similarity via vector embeddings.

The production-grade alternative to KeywordJaccardScorer. Uses
LLMClient.embed() to convert text to vectors and cosine similarity to
compare them. Catches synonyms ("login" ≈ "auth"), paraphrasing, and
typos that the keyword scorer misses entirely.

Plugs into MemoryAdapter via the SimilarityScorer Protocol:

    from src.llm.client import LLMClient, ClientMode
    from src.bridge.embedding_scorer import EmbeddingScorer
    from src.bridge.memory_adapter import MemoryAdapter

    client = LLMClient(mode=ClientMode.LIVE)   # OpenRouter
    scorer = EmbeddingScorer(client=client)    # default: text-embedding-3-small
    adapter = MemoryAdapter(repo_root=".", project_id="abc", scorer=scorer)

Cost shape (default model openai/text-embedding-3-small via OpenRouter):
    ~$0.000002 per embed call (~100 input tokens / typical decision text)
    ~$0.02 per 1M tokens

Cache: text → vector in-memory dict. Repeated queries for the same text
do not re-call the API. Cache survives the lifetime of the scorer
instance — pass a shared dict across adapters in the same process if
you want longer-lived caching.

ISOLATION: imports only from src.llm.client and stdlib (math). No
direct engine, bridge type, or HTTP wiring — embeddings flow through
LLMClient which already handles OpenRouter routing + retries + logging.
"""

from __future__ import annotations

import math

from src.llm.client import LLMClient
from src.llm.provider_map import DEFAULT_EMBEDDING_MODEL


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two equal-length vectors.

    Range: [-1, 1]. Returns 0.0 for empty, length-mismatched, or
    zero-magnitude inputs (so callers never get NaN/Inf).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingScorer:
    """
    Vector-embedding similarity scorer (SimilarityScorer Protocol).

    Flow per score() call:
        1. embed(query) — cache lookup, else API call
        2. embed(document) — cache lookup, else API call
        3. cosine_similarity(q_vec, d_vec)
        4. clamp to [0, 1] (Protocol contract — negative cosines treated as zero)

    The cache is keyed by the EXACT text string. Whitespace and case
    matter for cache hits but not for embedding semantics. Use
    `normalize_text()` upstream if you want to canonicalize before
    caching (not done automatically to keep the scorer minimal).
    """

    def __init__(
        self,
        client: LLMClient,
        model: str = DEFAULT_EMBEDDING_MODEL,
        cache: dict[str, list[float]] | None = None,
    ):
        self._client = client
        self._model = model
        self._cache: dict[str, list[float]] = cache if cache is not None else {}

    async def _embed(self, text: str) -> list[float]:
        """Fetch embedding for text, with cache. Falls through to API on miss."""
        if text in self._cache:
            return self._cache[text]
        vec = await self._client.embed(text, model=self._model)
        self._cache[text] = vec
        return vec

    async def score(self, query: str, document: str) -> float:
        """
        SimilarityScorer Protocol implementation.

        Returns 0.0 immediately for empty inputs (no API call). Otherwise
        fetches both vectors (cached or fresh) and returns clamped cosine
        similarity in [0.0, 1.0].

        Raises on persistent embedding failures — callers can catch
        and degrade to a fallback scorer if desired.
        """
        if not query or not document:
            return 0.0
        q_vec = await self._embed(query)
        d_vec = await self._embed(document)
        sim = cosine_similarity(q_vec, d_vec)
        # Cosine range [-1, 1]. Negative = orthogonal/opposite → treat as zero
        # for the [0, 1] Protocol contract. Clamping rather than abs() because
        # an opposite-direction vector isn't "similar," it's the opposite.
        return max(0.0, sim)

    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
