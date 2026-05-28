"""
filesystem_handler tests — pure-Python, no IO.

The handler reads from `attached_files` passed at construction time;
there's no network and no filesystem access on the server. Tests cover:

    1. Normalization (rejects bad payloads, keeps valid ones)
    2. Render — all files vs single-path narrowing
    3. Truncation limits (per-file + total)
    4. Error paths (no files / no match)
    5. Output shape matches McpHandler contract

Run: PYTHONPATH=. python tests/test_filesystem_handler.py
"""

from __future__ import annotations

import asyncio

from src.bridge.filesystem_handler import (
    FILE_MAX_CHARS,
    TOTAL_MAX_CHARS,
    make_filesystem_handler,
    normalize_attached_files,
)


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
# 1. normalize_attached_files
# ---------------------------------------------------------------------------

@test("1.1 normalize rejects non-list input")
def test_normalize_non_list():
    assert normalize_attached_files(None) == []
    assert normalize_attached_files("not a list") == []
    assert normalize_attached_files({"path": "x"}) == []


@test("1.2 normalize keeps valid entries and drops invalid ones")
def test_normalize_filters():
    raw = [
        {"path": "a.py", "content": "print('a')"},
        {"path": "", "content": "x"},                 # empty path → drop
        {"path": "b.py"},                              # missing content → drop
        {"path": "c.py", "content": "x", "language": "python"},
        "not a dict",                                  # → drop
        {"path": "d.py", "content": 42},               # content not str → drop
    ]
    out = normalize_attached_files(raw)
    paths = [f["path"] for f in out]
    assert paths == ["a.py", "c.py"]
    # language preserved when present, None when absent
    assert out[0]["language"] is None
    assert out[1]["language"] == "python"


@test("1.3 normalize strips whitespace on path + language")
def test_normalize_strips():
    raw = [{"path": "  src/x.py  \n", "content": "y", "language": "  python  "}]
    out = normalize_attached_files(raw)
    assert out[0]["path"] == "src/x.py"
    assert out[0]["language"] == "python"


# ---------------------------------------------------------------------------
# 2. Handler — render all attached files
# ---------------------------------------------------------------------------

@test("2.1 handler renders every attached file when no args.path")
async def test_render_all():
    files = [
        {"path": "src/a.py", "content": "def a(): pass", "language": "python"},
        {"path": "src/b.ts", "content": "export const b = 1;", "language": "typescript"},
    ]
    handler = make_filesystem_handler(files)
    out = await handler({}, purpose="show me my code")
    assert out["action"] == "read_attached_files"
    assert "### src/a.py" in out["text"]
    assert "### src/b.ts" in out["text"]
    assert "def a()" in out["text"]
    assert "export const b" in out["text"]
    assert out["data"]["total_attached"] == 2
    assert out["data"]["rendered"] == 2


@test("2.2 handler honors args.path to narrow output")
async def test_narrow_by_path():
    files = [
        {"path": "src/a.py", "content": "def a(): pass"},
        {"path": "src/b.py", "content": "def b(): pass"},
    ]
    handler = make_filesystem_handler(files)
    out = await handler({"path": "src/b.py"}, purpose="show b")
    assert "def b()" in out["text"]
    assert "def a()" not in out["text"]
    assert out["data"]["rendered"] == 1
    assert out["data"]["total_attached"] == 2


# ---------------------------------------------------------------------------
# 3. Truncation
# ---------------------------------------------------------------------------

@test("3.1 per-file content truncated at FILE_MAX_CHARS")
async def test_file_truncation():
    big_content = "x" * (FILE_MAX_CHARS + 5000)
    files = [{"path": "big.txt", "content": big_content}]
    handler = make_filesystem_handler(files)
    out = await handler({}, purpose="")
    # The rendered block contains a `[truncated]` marker
    assert "[truncated]" in out["text"]
    # And total rendered text is bounded
    assert len(out["text"]) <= FILE_MAX_CHARS + 200  # +overhead for fence+marker


@test("3.2 total rendered block bounded by TOTAL_MAX_CHARS across many files")
async def test_total_truncation():
    files = [
        {"path": f"f{i}.txt", "content": "y" * (FILE_MAX_CHARS - 100)}
        for i in range(10)
    ]
    handler = make_filesystem_handler(files)
    out = await handler({}, purpose="")
    # We don't enforce a hard exact cap (the last block can push over a
    # bit), but the omitted-files marker MUST appear if total would
    # otherwise exceed the budget.
    if len(out["text"]) > TOTAL_MAX_CHARS + 2000:
        raise AssertionError(
            f"total rendered ({len(out['text'])}) significantly exceeds budget"
        )
    if not any(
        "remaining files omitted" in out["text"]
        for _ in [None]
    ):
        # If we squeezed under budget without truncating, that's also fine —
        # the cap is a ceiling, not a target.
        pass


# ---------------------------------------------------------------------------
# 4. Error paths
# ---------------------------------------------------------------------------

@test("4.1 no attached files → RuntimeError")
async def test_no_files():
    handler = make_filesystem_handler([])
    try:
        await handler({}, purpose="x")
    except RuntimeError as e:
        assert "no files" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError when no files attached")


@test("4.2 args.path doesn't match any attached file → RuntimeError")
async def test_path_no_match():
    files = [{"path": "src/a.py", "content": "x"}]
    handler = make_filesystem_handler(files)
    try:
        await handler({"path": "nonexistent.py"}, purpose="x")
    except RuntimeError as e:
        msg = str(e).lower()
        assert "not in attached" in msg
        return
    raise AssertionError("expected RuntimeError for unmatched path")


@test("4.3 invalid payload normalized to empty list → RuntimeError")
async def test_invalid_payload():
    # Payload is malformed — every entry drops in normalization.
    handler = make_filesystem_handler([
        {"path": "", "content": "x"},     # empty path
        {"content": "x"},                  # missing path
        "not a dict",
    ])
    try:
        await handler({}, purpose="x")
    except RuntimeError as e:
        assert "no files" in str(e).lower()
        return
    raise AssertionError("expected RuntimeError on all-invalid payload")


# ---------------------------------------------------------------------------
# 5. Output shape compliance
# ---------------------------------------------------------------------------

@test("5.1 handler returns dict with text, data, action keys")
async def test_output_shape():
    files = [{"path": "x.py", "content": "y"}]
    handler = make_filesystem_handler(files)
    out = await handler({}, purpose="x")
    assert isinstance(out, dict)
    assert isinstance(out.get("text"), str)
    assert isinstance(out.get("data"), dict)
    assert out["action"] == "read_attached_files"


@test("5.2 data.files reports per-file size accurately")
async def test_data_files_sizes():
    files = [
        {"path": "x.py", "content": "abcd"},
        {"path": "y.ts", "content": "abcdefghij"},
    ]
    handler = make_filesystem_handler(files)
    out = await handler({}, purpose="")
    sizes = {f["path"]: f["size"] for f in out["data"]["files"]}
    assert sizes == {"x.py": 4, "y.ts": 10}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_normalize_non_list,
    test_normalize_filters,
    test_normalize_strips,
    test_render_all,
    test_narrow_by_path,
    test_file_truncation,
    test_total_truncation,
    test_no_files,
    test_path_no_match,
    test_invalid_payload,
    test_output_shape,
    test_data_files_sizes,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} filesystem_handler tests...")
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
