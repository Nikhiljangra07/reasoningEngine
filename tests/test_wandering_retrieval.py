"""
Tests for the retrieval-mesh expansion (D4-D9 in WANDERING_ROOM_DECISIONS.md):

  - session_state.py — per-wander URL dedup + follow-on queue
  - extractors.py — tier-2 escalation gate + link extraction
  - exa_provider.py — graceful no-key behavior (network calls mocked out)
  - trust.py — domain weight + confidence promotion rules
  - fetcher.py — multi-provider chain with dedup-aware hit selection

Pure-logic tests (no real HTTP). Network-dependent paths are exercised
via monkey-patched providers. The minimum guarantee: every wiring point
behaves correctly when keys are absent, when results come back empty,
and when the provider chain falls through.

Run:
  PYTHONPATH=. .venv/bin/python tests/test_wandering_retrieval.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from src.bridge.web_search import SearchHit, SearchResult
from src.wandering.exa_provider import (
    ExaHit,
    ExaResult,
    find_similar,
    is_available,
    search as exa_search,
)
from src.wandering.extractors import (
    ExtractResult,
    extract_links,
    extract_url,
    should_escalate_to_tier2,
    TIER2_ESCALATION_CEILING,
    TIER2_MIN_MATCHED_NODES,
)
from src.wandering.fetcher import web_search_fetcher
from src.wandering.report import Confidence
from src.wandering.session_state import (
    FollowonItem,
    SessionState,
    normalize_url,
)
from src.wandering.trust import (
    DEFAULT_WEIGHT,
    PROMOTION_THRESHOLD,
    adjust_confidence,
    domain_weight,
)


# Mini test harness — same pattern as the other tests/ files. Lets the
# file double as a pytest-collected module AND a standalone script.

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
    except AssertionError as e:
        FAILED += 1
        ERRORS.append((name, f"FAIL: {e}"))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, f"ERROR: {type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


# ============================================================================
# session_state — URL dedup + follow-on queue
# ============================================================================


@test("1.1 normalize_url lowercases scheme/host, drops fragment, keeps query")
def test_normalize_url_basics():
    assert normalize_url("HTTPS://Example.com/PATH?q=1#frag") == \
        "https://example.com/PATH?q=1"
    assert normalize_url("") == ""
    assert normalize_url("https://example.com/") == "https://example.com"


@test("1.2 SessionState.has_visited returns False before first mark")
def test_has_visited_empty():
    state = SessionState(session_id="s1")
    assert state.has_visited("https://a.com") is False
    assert state.has_visited("") is False


@test("1.3 mark_visited returns True first time, False on dup")
async def test_mark_visited_dedup():
    state = SessionState(session_id="s1")
    first = await state.mark_visited("https://a.com/page")
    assert first is True
    second = await state.mark_visited("https://a.com/page")
    assert second is False
    # Same URL with different casing / fragment still dedups.
    third = await state.mark_visited("HTTPS://A.com/page#x")
    assert third is False


@test("1.4 follow-on enqueue rejects already-visited URLs")
async def test_followon_skips_visited():
    state = SessionState(session_id="s1")
    await state.mark_visited("https://a.com/page")
    added = await state.enqueue_followon(FollowonItem(
        url="https://a.com/page", score=0.9,
    ))
    assert added is False
    assert state.peek_followon_count() == 0


@test("1.5 follow-on enqueue dedups within queue")
async def test_followon_dedup_within_queue():
    state = SessionState(session_id="s1")
    await state.enqueue_followon(FollowonItem(url="https://x.com/a", score=0.5))
    second = await state.enqueue_followon(FollowonItem(url="https://x.com/a", score=0.9))
    assert second is False
    assert state.peek_followon_count() == 1


@test("1.6 follow-on queue stays sorted highest-priority-first")
async def test_followon_sort_order():
    state = SessionState(session_id="s1")
    await state.enqueue_followon(FollowonItem(url="https://a.com/1", score=0.2))
    await state.enqueue_followon(FollowonItem(url="https://b.com/2", score=0.8))
    await state.enqueue_followon(FollowonItem(url="https://c.com/3", score=0.5))
    top = await state.pop_followon()
    assert top is not None
    assert top.url == "https://b.com/2"
    next_ = await state.pop_followon()
    assert next_ is not None
    assert next_.url == "https://c.com/3"


@test("1.7 follow-on queue respects MAX_FOLLOWON_QUEUE")
async def test_followon_max_capacity():
    state = SessionState(session_id="s1")
    state.MAX_FOLLOWON_QUEUE = 3
    await state.enqueue_followon(FollowonItem(url="https://a.com/1", score=0.9))
    await state.enqueue_followon(FollowonItem(url="https://a.com/2", score=0.8))
    await state.enqueue_followon(FollowonItem(url="https://a.com/3", score=0.7))
    # This one should oust the lowest (https://a.com/3 at 0.7)
    await state.enqueue_followon(FollowonItem(url="https://a.com/4", score=0.95))
    assert state.peek_followon_count() == 3
    urls = [fi.url for fi in state.followon_queue]
    assert "https://a.com/4" in urls
    assert "https://a.com/3" not in urls


@test("1.8 FollowonItem clamps score to [0,1]")
def test_followon_score_clamp():
    high = FollowonItem(url="https://x.com", score=1.7)
    low = FollowonItem(url="https://y.com", score=-0.4)
    assert high.score == 1.0
    assert low.score == 0.0


# ============================================================================
# extractors — tier-2 escalation gate + link extraction
# ============================================================================


@test("2.1 should_escalate_to_tier2 fires on borderline match")
def test_escalate_borderline():
    # 1 node matched, both essence and mechanism below 0.7 → escalate.
    assert should_escalate_to_tier2(
        total_matched_nodes=1, essence_ratio=0.3, mechanism_ratio=0.4,
    ) is True


@test("2.2 should_escalate_to_tier2 skips when no signal")
def test_escalate_no_signal():
    assert should_escalate_to_tier2(
        total_matched_nodes=0, essence_ratio=0.0, mechanism_ratio=0.0,
    ) is False


@test("2.3 should_escalate_to_tier2 skips when already strong")
def test_escalate_already_strong():
    assert should_escalate_to_tier2(
        total_matched_nodes=3, essence_ratio=0.8, mechanism_ratio=0.4,
    ) is False
    assert should_escalate_to_tier2(
        total_matched_nodes=3, essence_ratio=0.4, mechanism_ratio=0.8,
    ) is False


@test("2.4 should_escalate_to_tier2 ceiling matches advertised constant")
def test_escalate_ceiling():
    # Right at the ceiling is treated as strong (no escalate).
    assert should_escalate_to_tier2(
        total_matched_nodes=1,
        essence_ratio=TIER2_ESCALATION_CEILING,
        mechanism_ratio=0.0,
    ) is False
    # Just below the ceiling triggers.
    assert should_escalate_to_tier2(
        total_matched_nodes=1,
        essence_ratio=TIER2_ESCALATION_CEILING - 0.01,
        mechanism_ratio=0.0,
    ) is True


@test("2.4b should_escalate_to_tier2 zero-match without URL stays skipped")
def test_escalate_zero_match_no_url():
    # No URL → sampling escape hatch can't fire → False.
    assert should_escalate_to_tier2(
        total_matched_nodes=0, essence_ratio=0.0, mechanism_ratio=0.0,
        url="",
    ) is False


@test("2.4c should_escalate_to_tier2 zero-match from untrusted URL stays skipped")
def test_escalate_zero_match_untrusted():
    # Random blog: domain_weight = 1.00 < PROMOTION_THRESHOLD → False.
    assert should_escalate_to_tier2(
        total_matched_nodes=0, essence_ratio=0.0, mechanism_ratio=0.0,
        url="https://randomblog.example/post",
    ) is False


@test("2.4d should_escalate_to_tier2 sampling fires on some trusted zero-match URLs")
def test_escalate_zero_match_trusted_sampling():
    # The sampling rate is ~15% — we don't know which specific URLs land
    # in the sample, but across many trusted-domain URLs, SOME must.
    # Generate enough URLs that the binomial probability of zero hits
    # is negligible (P(0 hits | 100 trials, p=0.15) ≈ 1e-7).
    urls = [f"https://arxiv.org/abs/{i:04d}.{i*7:05d}" for i in range(100)]
    escalated = sum(
        1
        for u in urls
        if should_escalate_to_tier2(
            total_matched_nodes=0, essence_ratio=0.0, mechanism_ratio=0.0,
            url=u,
        )
    )
    # Expect ~15 (12-18 within reasonable variance for n=100, p=0.15).
    # Looser bound (5-25) tolerates hash-distribution skew without
    # making the test flaky.
    assert 5 <= escalated <= 25, (
        f"trusted zero-match sampling rate looked off: {escalated}/100 escalated "
        f"(expected ~15)"
    )


@test("2.4e should_escalate_to_tier2 sampling is deterministic per URL")
def test_escalate_sampling_deterministic():
    # Same URL must yield the same decision across calls — debugging
    # depends on this. Run the same URL 5 times; results must agree.
    url = "https://arxiv.org/abs/9999.99999"
    decisions = [
        should_escalate_to_tier2(
            total_matched_nodes=0, essence_ratio=0.0, mechanism_ratio=0.0,
            url=url,
        )
        for _ in range(5)
    ]
    assert all(d == decisions[0] for d in decisions), "sampling must be deterministic per URL"


@test("2.4f should_escalate_to_tier2 main path is unaffected by url kwarg")
def test_escalate_main_path_with_url():
    # When there's signal, the URL kwarg is ignored — main path fires.
    assert should_escalate_to_tier2(
        total_matched_nodes=1, essence_ratio=0.3, mechanism_ratio=0.4,
        url="https://anything.example/x",
    ) is True


@test("2.5 extract_links pulls (text, url) pairs from markdown")
def test_extract_links_basic():
    md = """
    See [a paper](https://arxiv.org/abs/1234) and [a blog](http://blog.com/x).
    Also [a paper](https://arxiv.org/abs/1234) (duplicate).
    """
    links = extract_links(md)
    assert len(links) == 2
    assert ("a paper", "https://arxiv.org/abs/1234") in links
    assert ("a blog", "http://blog.com/x") in links


@test("2.6 extract_links drops obvious nav anchors")
def test_extract_links_drops_nav():
    md = "[Home](https://x.com) [Subscribe](https://x.com/sub) [Real](https://x.com/post)"
    links = extract_links(md)
    urls = [u for _, u in links]
    assert "https://x.com" not in urls
    assert "https://x.com/sub" not in urls
    assert "https://x.com/post" in urls


@test("2.7 extract_links respects max_links cap")
def test_extract_links_cap():
    md = " ".join(f"[link{i}](https://x.com/{i})" for i in range(30))
    links = extract_links(md, max_links=5)
    assert len(links) == 5


@test("2.8 extract_url returns ok=False on invalid scheme")
async def test_extract_url_rejects_bad_scheme():
    result = await extract_url("ftp://x.com")
    assert result.ok is False
    assert result.error == "invalid_url"


# ============================================================================
# exa_provider — graceful no-key behavior
# ============================================================================


@test("3.1 is_available reflects EXA_API_KEY presence")
def test_exa_is_available_no_key():
    saved = os.environ.pop("EXA_API_KEY", None)
    try:
        assert is_available() is False
    finally:
        if saved is not None:
            os.environ["EXA_API_KEY"] = saved


@test("3.2 exa_search returns no_api_key error when key absent")
async def test_exa_search_no_key():
    saved = os.environ.pop("EXA_API_KEY", None)
    try:
        result = await exa_search("test query")
        assert result.ok is False
        assert result.error == "no_api_key"
        assert result.hits == []
    finally:
        if saved is not None:
            os.environ["EXA_API_KEY"] = saved


@test("3.3 find_similar returns no_api_key error when key absent")
async def test_find_similar_no_key():
    saved = os.environ.pop("EXA_API_KEY", None)
    try:
        result = await find_similar("https://example.com/x")
        assert result.ok is False
        assert result.error == "no_api_key"
    finally:
        if saved is not None:
            os.environ["EXA_API_KEY"] = saved


@test("3.4 find_similar rejects non-http URLs")
async def test_find_similar_bad_url():
    saved = os.environ.get("EXA_API_KEY")
    os.environ["EXA_API_KEY"] = "test-key"
    try:
        result = await find_similar("not-a-url")
        assert result.ok is False
        assert result.error == "invalid_url"
    finally:
        if saved is None:
            os.environ.pop("EXA_API_KEY", None)
        else:
            os.environ["EXA_API_KEY"] = saved


# ============================================================================
# trust — domain weight + confidence promotion
# ============================================================================


@test("4.1 domain_weight returns default for unknown domain")
def test_domain_weight_default():
    assert domain_weight("https://random-blog.example.io/post") == DEFAULT_WEIGHT


@test("4.2 domain_weight elevates arxiv")
def test_domain_weight_arxiv():
    assert domain_weight("https://arxiv.org/abs/1234") == 1.20


@test("4.3 domain_weight handles www prefix")
def test_domain_weight_www():
    assert domain_weight("https://www.wikipedia.org/wiki/Topic") == 1.10


@test("4.4 domain_weight uses suffix match for .edu")
def test_domain_weight_edu_suffix():
    assert domain_weight("https://cs.stanford.edu/papers/x") == 1.10
    assert domain_weight("https://lab.mit.edu/x") == 1.10


@test("4.5 domain_weight uses longest-suffix on .ox.ac.uk vs .ac.uk")
def test_domain_weight_longest_suffix():
    assert domain_weight("https://www.cs.ox.ac.uk/research") == 1.15
    assert domain_weight("https://imperial.ac.uk/x") == 1.10


@test("4.6 adjust_confidence never demotes")
def test_adjust_confidence_no_demote():
    # HIGH stays HIGH even with default weight.
    result = adjust_confidence(
        Confidence.HIGH, url="https://untrusted.example.com/", total_matched_nodes=5,
    )
    assert result == Confidence.HIGH


@test("4.7 adjust_confidence promotes LOW to MEDIUM on trusted source")
def test_adjust_confidence_low_to_med():
    result = adjust_confidence(
        Confidence.LOW, url="https://arxiv.org/abs/1234", total_matched_nodes=1,
    )
    assert result == Confidence.MEDIUM


@test("4.8 adjust_confidence does NOT promote MEDIUM (domain alone insufficient)")
def test_adjust_confidence_med_no_promote():
    # MEDIUM → HIGH used to fire on any trusted-domain hit with at least
    # one matched node. Removed: absolute `total_matched_nodes` count
    # doesn't generalize across variable constellation sizes. Trust now
    # only adjusts LOW (one-step nudge for weak-but-cited matches).
    # See DESIGN NOTE in trust.adjust_confidence.
    result = adjust_confidence(
        Confidence.MEDIUM, url="https://wikipedia.org/wiki/X", total_matched_nodes=2,
    )
    assert result == Confidence.MEDIUM
    # Even with a strong trusted domain + many matches, MEDIUM stays.
    result_arxiv = adjust_confidence(
        Confidence.MEDIUM, url="https://arxiv.org/abs/2501.00001",
        total_matched_nodes=10,
    )
    assert result_arxiv == Confidence.MEDIUM


@test("4.9 adjust_confidence does NOT promote when match count is 0")
def test_adjust_confidence_no_signal():
    result = adjust_confidence(
        Confidence.LOW, url="https://arxiv.org/x", total_matched_nodes=0,
    )
    assert result == Confidence.LOW


@test("4.10 adjust_confidence neutral weight leaves base unchanged")
def test_adjust_confidence_neutral():
    result = adjust_confidence(
        Confidence.MEDIUM, url="https://random.com/x", total_matched_nodes=2,
    )
    assert result == Confidence.MEDIUM


# ============================================================================
# fetcher — dedup-aware hit selection (using stubbed search results)
# ============================================================================


@test("5.1 fetcher picks first unvisited hit when state has dupes")
async def test_fetcher_skips_visited():
    """Validate that web_search_fetcher consults session_state and picks
    the first unvisited hit. We patch _search_with_chain to return a
    fixed result with two URLs (one visited)."""
    from src.wandering import fetcher as fmod

    state = SessionState(session_id="s")
    await state.mark_visited("https://a.com/visited")

    fixed = SearchResult(
        query="anchor physics",
        provider="test",
        hits=[
            SearchHit(title="V", url="https://a.com/visited", snippet="visited"),
            SearchHit(title="N", url="https://a.com/novel", snippet="novel snippet"),
        ],
    )

    async def fake_chain(_q):
        return fixed

    orig = fmod._search_with_chain
    fmod._search_with_chain = fake_chain  # type: ignore[assignment]
    try:
        result = await web_search_fetcher(
            "physics", "anchor", session_state=state,
        )
    finally:
        fmod._search_with_chain = orig  # type: ignore[assignment]

    assert result.url == "https://a.com/novel"
    assert state.has_visited("https://a.com/novel") is True


@test("5.2 fetcher honors follow-on queue before running search")
async def test_fetcher_drains_followon_first():
    from src.wandering import fetcher as fmod
    from src.wandering import extractors

    state = SessionState(session_id="s")
    await state.enqueue_followon(FollowonItem(
        url="https://followon.com/page", score=0.9, origin="findsimilar",
    ))

    # Make sure search would NOT be called (it isn't needed). Patch the
    # chain to raise if reached.
    async def fake_chain(_q):
        raise AssertionError("fetcher should not run a search when queue has items")

    async def fake_extract(url, **kwargs):
        return ExtractResult(
            url=url, body=f"extracted body for {url}", chars=20, ok=True, latency_ms=5,
        )

    orig_chain = fmod._search_with_chain
    orig_extract = extractors.extract_url
    fmod._search_with_chain = fake_chain  # type: ignore[assignment]
    extractors.extract_url = fake_extract  # type: ignore[assignment]
    try:
        result = await web_search_fetcher("physics", "anchor", session_state=state)
    finally:
        fmod._search_with_chain = orig_chain  # type: ignore[assignment]
        extractors.extract_url = orig_extract  # type: ignore[assignment]

    assert result.url == "https://followon.com/page"
    assert "extracted body" in result.body
    # The follow-on URL should now be marked visited.
    assert state.has_visited("https://followon.com/page") is True
    # Queue is drained.
    assert state.peek_followon_count() == 0


@test("5.3 fetcher uses index 0 when session_state is None (legacy compat)")
async def test_fetcher_no_state_legacy():
    from src.wandering import fetcher as fmod

    fixed = SearchResult(
        query="x",
        provider="test",
        hits=[
            SearchHit(title="A", url="https://a.com/1", snippet="a"),
            SearchHit(title="B", url="https://b.com/2", snippet="b"),
        ],
    )

    async def fake_chain(_q):
        return fixed

    orig = fmod._search_with_chain
    fmod._search_with_chain = fake_chain  # type: ignore[assignment]
    try:
        result = await web_search_fetcher("physics", "anchor", session_state=None)
    finally:
        fmod._search_with_chain = orig  # type: ignore[assignment]

    # No state → always top hit.
    assert result.url == "https://a.com/1"


# ============================================================================
# Test runner
# ============================================================================


def main():
    tests = [
        # session_state
        test_normalize_url_basics,
        test_has_visited_empty,
        test_mark_visited_dedup,
        test_followon_skips_visited,
        test_followon_dedup_within_queue,
        test_followon_sort_order,
        test_followon_max_capacity,
        test_followon_score_clamp,
        # extractors
        test_escalate_borderline,
        test_escalate_no_signal,
        test_escalate_already_strong,
        test_escalate_ceiling,
        test_escalate_zero_match_no_url,
        test_escalate_zero_match_untrusted,
        test_escalate_zero_match_trusted_sampling,
        test_escalate_sampling_deterministic,
        test_escalate_main_path_with_url,
        test_extract_links_basic,
        test_extract_links_drops_nav,
        test_extract_links_cap,
        test_extract_url_rejects_bad_scheme,
        # exa
        test_exa_is_available_no_key,
        test_exa_search_no_key,
        test_find_similar_no_key,
        test_find_similar_bad_url,
        # trust
        test_domain_weight_default,
        test_domain_weight_arxiv,
        test_domain_weight_www,
        test_domain_weight_edu_suffix,
        test_domain_weight_longest_suffix,
        test_adjust_confidence_no_demote,
        test_adjust_confidence_low_to_med,
        test_adjust_confidence_med_no_promote,
        test_adjust_confidence_no_signal,
        test_adjust_confidence_neutral,
        # fetcher
        test_fetcher_skips_visited,
        test_fetcher_drains_followon_first,
        test_fetcher_no_state_legacy,
    ]
    for fn in tests:
        run_test(fn)
    print()
    print("=" * 60)
    print(f"  {PASSED} passed, {FAILED} failed")
    print("=" * 60)
    if FAILED:
        for name, err in ERRORS:
            print(f"  {name}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
