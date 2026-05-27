"""
Similarity Scorer tests.

Verifies:
    1. tokenize helper
    2. KeywordJaccardScorer behavior + edge cases
    3. SimilarityScorer Protocol — custom scorer drops in

No LLM calls, no API. Pure logic.

Run: PYTHONPATH=. python3 tests/test_similarity.py
"""

from __future__ import annotations

import asyncio

from src.bridge.embedding_scorer import EmbeddingScorer, cosine_similarity
from src.bridge.similarity import (
    KeywordJaccardScorer,
    SimilarityScorer,
    tokenize,
)
from src.llm.client import ClientMode, LLMClient


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
# 1. tokenize helper
# ---------------------------------------------------------------------------

@test("1.1 tokenize lowercases + extracts alphanumeric")
def test_tokenize_basic():
    out = tokenize("Hello WORLD foo")
    assert out == {"hello", "world", "foo"}


@test("1.2 tokenize drops tokens of length <= 2")
def test_tokenize_drops_short():
    out = tokenize("the io v2 abc")
    # "the" → kept (length 3), "io" → dropped (2), "v2" → dropped (2), "abc" → kept
    assert "the" in out and "abc" in out
    assert "io" not in out and "v2" not in out


@test("1.3 tokenize returns empty set for empty / None-equivalent input")
def test_tokenize_empty():
    assert tokenize("") == set()
    assert tokenize("   ") == set()
    assert tokenize("!@#$%") == set()


@test("1.4 tokenize treats punctuation as separator")
def test_tokenize_punctuation():
    out = tokenize("user-name.foo,bar")
    assert out == {"user", "name", "foo", "bar"}


# ---------------------------------------------------------------------------
# 2. KeywordJaccardScorer
# ---------------------------------------------------------------------------

@test("2.1 identical text → score = 1.0")
async def test_jaccard_identical():
    s = KeywordJaccardScorer()
    assert await s.score("idempotency strategy", "idempotency strategy") == 1.0


@test("2.2 disjoint vocab → score = 0.0")
async def test_jaccard_disjoint():
    s = KeywordJaccardScorer()
    assert await s.score("idempotency", "quantum chromodynamics") == 0.0


@test("2.3 partial overlap → score in (0, 1)")
async def test_jaccard_partial():
    s = KeywordJaccardScorer()
    score = await s.score("idempotency keys", "idempotency strategy payments")
    assert 0.0 < score < 1.0
    assert abs(score - 0.25) < 1e-9


@test("2.4 empty query → score = 0.0")
async def test_jaccard_empty_query():
    s = KeywordJaccardScorer()
    assert await s.score("", "any doc") == 0.0


@test("2.5 empty document → score = 0.0")
async def test_jaccard_empty_doc():
    s = KeywordJaccardScorer()
    assert await s.score("any query", "") == 0.0


@test("2.6 case-insensitive matching")
async def test_jaccard_case():
    s = KeywordJaccardScorer()
    assert await s.score("Refund Handler", "refund handler") == 1.0


@test("2.7 punctuation does not affect score")
async def test_jaccard_punctuation():
    s = KeywordJaccardScorer()
    a = await s.score("auth, login!", "login auth")
    b = await s.score("auth login", "login auth")
    assert a == b == 1.0


# ---------------------------------------------------------------------------
# 3. SimilarityScorer Protocol — pluggability
# ---------------------------------------------------------------------------

@test("3.1 KeywordJaccardScorer satisfies the SimilarityScorer Protocol (runtime check)")
def test_jaccard_satisfies_protocol():
    s = KeywordJaccardScorer()
    assert isinstance(s, SimilarityScorer)


@test("3.2 custom async scorer with just .score() satisfies the Protocol")
async def test_custom_scorer_satisfies_protocol():
    class FixedScorer:
        async def score(self, query: str, document: str) -> float:
            return 0.42
    s = FixedScorer()
    assert isinstance(s, SimilarityScorer)
    assert await s.score("anything", "anything else") == 0.42


@test("3.3 custom scorer can be plugged into MemoryAdapter")
async def test_custom_scorer_in_adapter():
    from src.bridge.memory_adapter import MemoryAdapter
    from src.bridge.types import DecisionAnchor

    class AlwaysHalfScorer:
        async def score(self, q: str, d: str) -> float:
            return 0.5

    adapter = MemoryAdapter(repo_root=".", scorer=AlwaysHalfScorer())
    decision = DecisionAnchor(
        id="D-test",
        title="anything",
        rationale="anything",
        evidence=[],
        status="OPEN",
        created_at=0.0,
        code_refs=[],
        tags=[],
    )
    await adapter.store_decision(decision)
    # Even with vastly different texts, the AlwaysHalfScorer returns 0.5,
    # so the decision will be found (score > 0).
    hits = await adapter.find_similar_decisions("totally unrelated quantum", k=3)
    assert len(hits) == 1


# ---------------------------------------------------------------------------
# 4. cosine_similarity primitive
# ---------------------------------------------------------------------------

@test("4.1 cosine of identical non-zero vectors = 1.0")
def test_cosine_identical():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


@test("4.2 cosine of orthogonal vectors = 0.0")
def test_cosine_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


@test("4.3 cosine of opposite vectors = -1.0 (raw, before scorer clamps)")
def test_cosine_opposite():
    assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-9


@test("4.4 cosine: empty / mismatched / zero-norm inputs → 0.0 (no NaN)")
def test_cosine_edge_cases():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# 5. EmbeddingScorer (via MOCK LLMClient — deterministic, no API calls)
# ---------------------------------------------------------------------------

def _mk_embed_client():
    return LLMClient(mode=ClientMode.MOCK)


@test("5.1 identical text → cosine 1.0 → clamped score 1.0")
async def test_embed_identical():
    s = EmbeddingScorer(client=_mk_embed_client())
    score = await s.score("idempotency strategy", "idempotency strategy")
    assert abs(score - 1.0) < 1e-9


@test("5.2 totally disjoint tokens → cosine 0 → clamped score 0.0")
async def test_embed_disjoint():
    s = EmbeddingScorer(client=_mk_embed_client())
    # Hashing-trick mock embedding: distinct tokens → distinct buckets
    score = await s.score("idempotency strategy auth", "quantum chromodynamics nucleosynthesis")
    assert score < 0.05  # essentially zero


@test("5.3 overlapping tokens → score in (0, 1)")
async def test_embed_partial_overlap():
    s = EmbeddingScorer(client=_mk_embed_client())
    score = await s.score("auth login session", "auth jwt token")
    # Shared token: "auth" → some signal, not full
    assert 0.0 < score < 1.0


@test("5.4 empty inputs return 0.0 WITHOUT calling the API")
async def test_embed_empty_short_circuits():
    client = _mk_embed_client()
    s = EmbeddingScorer(client=client)
    assert await s.score("", "anything") == 0.0
    assert await s.score("anything", "") == 0.0
    # No embed calls should have been logged
    assert len([log for log in client.call_log if log.domain == "embedding"]) == 0


@test("5.5 cache hit avoids the second API call")
async def test_embed_cache_hit():
    client = _mk_embed_client()
    s = EmbeddingScorer(client=client)
    await s.score("auth login", "auth jwt")
    calls_after_first = len([log for log in client.call_log if log.domain == "embedding"])
    assert calls_after_first == 2  # both texts embedded once
    # Same query, same doc — cache should serve both
    await s.score("auth login", "auth jwt")
    calls_after_second = len([log for log in client.call_log if log.domain == "embedding"])
    assert calls_after_second == 2  # no new API calls
    assert s.cache_size() == 2


@test("5.6 clear_cache forces re-embedding")
async def test_embed_clear_cache():
    client = _mk_embed_client()
    s = EmbeddingScorer(client=client)
    await s.score("auth login", "auth jwt")
    s.clear_cache()
    assert s.cache_size() == 0
    await s.score("auth login", "auth jwt")
    # Two original calls + two more after clear
    assert len([log for log in client.call_log if log.domain == "embedding"]) == 4


@test("5.7 EmbeddingScorer satisfies the SimilarityScorer Protocol")
def test_embed_satisfies_protocol():
    s = EmbeddingScorer(client=_mk_embed_client())
    assert isinstance(s, SimilarityScorer)


@test("5.8 EmbeddingScorer plugs into MemoryAdapter end-to-end")
async def test_embed_in_adapter():
    from src.bridge.memory_adapter import MemoryAdapter
    from src.bridge.types import CodeRef, DecisionAnchor
    client = _mk_embed_client()
    scorer = EmbeddingScorer(client=client)
    adapter = MemoryAdapter(repo_root=".", project_id="proj-1", scorer=scorer)
    decision = DecisionAnchor(
        id="D-001",
        title="idempotency keys for refunds",
        rationale="prevent duplicate refunds via request-shape hash",
        evidence=["incident INC-0042"],
        status="SETTLED",
        created_at=0.0,
        code_refs=[CodeRef(file_path="src/refund.ts", line_start=1, line_end=10)],
        tags=["payments", "idempotency"],
    )
    await adapter.store_decision(decision)
    # Strong overlap on "idempotency"
    hits = await adapter.find_similar_decisions("idempotency strategy", k=3)
    assert len(hits) == 1
    assert hits[0].id == "D-001"


@test("5.9 EmbeddingScorer cost is logged in client.call_log with domain='embedding'")
async def test_embed_cost_logged():
    client = _mk_embed_client()
    s = EmbeddingScorer(client=client)
    await s.score("hello world", "hello world")
    embedding_logs = [log for log in client.call_log if log.domain == "embedding"]
    # 1 embed call (identical text serves both query + doc via cache)
    assert len(embedding_logs) == 1
    log = embedding_logs[0]
    assert log.model == "openai/text-embedding-3-small"   # default
    assert log.output_tokens == 0
    assert log.input_tokens > 0
    assert log.success is True


@test("5.10 custom model param flows through to the call log")
async def test_embed_custom_model():
    client = _mk_embed_client()
    s = EmbeddingScorer(client=client, model="openai/text-embedding-3-large")
    await s.score("hi", "hi")
    log = [l for l in client.call_log if l.domain == "embedding"][0]
    assert log.model == "openai/text-embedding-3-large"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_tokenize_basic,
    test_tokenize_drops_short,
    test_tokenize_empty,
    test_tokenize_punctuation,
    test_jaccard_identical,
    test_jaccard_disjoint,
    test_jaccard_partial,
    test_jaccard_empty_query,
    test_jaccard_empty_doc,
    test_jaccard_case,
    test_jaccard_punctuation,
    test_jaccard_satisfies_protocol,
    test_custom_scorer_satisfies_protocol,
    test_custom_scorer_in_adapter,
    test_cosine_identical,
    test_cosine_orthogonal,
    test_cosine_opposite,
    test_cosine_edge_cases,
    test_embed_identical,
    test_embed_disjoint,
    test_embed_partial_overlap,
    test_embed_empty_short_circuits,
    test_embed_cache_hit,
    test_embed_clear_cache,
    test_embed_satisfies_protocol,
    test_embed_in_adapter,
    test_embed_cost_logged,
    test_embed_custom_model,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} similarity tests...")
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
