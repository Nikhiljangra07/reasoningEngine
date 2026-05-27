"""
Similarity Scorer — pluggable backend for decision-text similarity.

The Memory V2 adapter calls into this layer to rank decisions against a
query. Multiple backends can coexist; the adapter picks one at
construction time via `MemoryAdapter(scorer=...)`.

Implementations in this drop:
    KeywordJaccardScorer  — default. Token-set Jaccard. No deps, deterministic.

Implementations DEFERRED to a follow-up turn (so the user can pick the
model without being locked in here):
    EmbeddingScorer       — vector embeddings via OpenRouter (e.g.
                            openai/text-embedding-3-small). Requires an
                            LLMClient with embedding support + a cache.
    LocalEmbeddingScorer  — local sentence-transformers / BGE. Heavy dep.

Anyone building a future scorer implements the SimilarityScorer Protocol
below — that's the only contract. Then pass `scorer=MyScorer()` to
MemoryAdapter and nothing else changes.

ISOLATION: stdlib only. No engine, no LLM client.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class SimilarityScorer(Protocol):
    """
    The single contract every scorer must satisfy.

    Returns a similarity score in [0.0, 1.0] where 1.0 = identical and
    0.0 = no signal. Score symmetry (score(a, b) == score(b, a)) is
    recommended but not required.

    Async signature so embedding-based scorers (which need to make API
    calls to fetch vectors) can implement the same Protocol as
    keyword-based scorers (which compute synchronously and just return
    immediately from their async wrapper).
    """

    async def score(self, query: str, document: str) -> float: ...


# ---------------------------------------------------------------------------
# Token utilities — shared across scorers
# ---------------------------------------------------------------------------

# Lowercase alphanumeric tokens of length > 2. Filters short noise
# ("a", "is", "to") that hurt similarity signal.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Tokenize text → set of lowercase alphanumeric tokens, length > 2."""
    if not text:
        return set()
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


# ---------------------------------------------------------------------------
# Default scorer — keyword Jaccard
# ---------------------------------------------------------------------------

class KeywordJaccardScorer:
    """
    Naive token-set Jaccard similarity.

    Score = |query_tokens ∩ doc_tokens| / |query_tokens ∪ doc_tokens|.

    Strengths:
        - Deterministic, fast, zero dependencies
        - Strong signal on exact-keyword overlap ("auth", "idempotency")
        - Works well for code-decision titles that share domain vocabulary

    Weaknesses:
        - Cannot match across vocabularies ("login" vs "auth", "PR" vs "pull request")
        - No semantic generalization — only exact tokens
        - Filtering tokens of length ≤ 2 drops "v2", "ai", "io" etc.

    Use this as the default until an embedding scorer is wired up; this
    class is the baseline an embedding scorer must outperform to justify
    its added cost/latency.
    """

    async def score(self, query: str, document: str) -> float:
        q = tokenize(query)
        d = tokenize(document)
        if not q or not d:
            return 0.0
        union = len(q | d)
        if union == 0:
            return 0.0
        return len(q & d) / union
