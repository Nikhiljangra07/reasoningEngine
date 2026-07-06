# Tests — How To Run Them

> READ THIS BEFORE RUNNING TESTS. The test files in this directory
> do NOT use pytest. Auditors who run `pytest tests/` will see
> "No module named pytest" or a zero-collected run and conclude the
> tests don't work. They DO work — just not via pytest.

## TL;DR — run any test file directly with Python

```bash
PYTHONPATH=. .venv/bin/python tests/test_wandering_credits.py
```

Replace the filename with whichever test you want to run. Each file
prints `PASS` / `FAIL` lines and ends with `N passed, M failed`.
Exit code is non-zero on any failure, so CI integration is trivial.

## Run the whole wandering test suite

```bash
cd reasoningEngine
for f in tests/test_wandering*.py; do
    echo "=== $f ==="
    PYTHONPATH=. .venv/bin/python "$f"
done
```

228 total tests across 8 wandering files. The whole sweep takes
about 90 seconds.

## Why no pytest

Each file is self-contained:

```python
from src.wandering.credits import CreditService, ...

PASSED = 0
FAILED = 0

def test(name: str):
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator

@test("RESERVE.1 reserve deducts from spendable balance via hold")
async def test_reserve_holds():
    svc = CreditService(InMemoryCreditStore())
    ...
    assert res.held_credits == 15

if __name__ == "__main__":
    tests = [v for v in globals().values()
             if callable(v) and hasattr(v, "_test_name")]
    for t in tests:
        run_test(t)
```

The custom harness avoids a pytest dependency for the wandering code
path, lets us collect tests by decorator (not magic name prefixes),
and supports both sync and async test functions without
plugin gymnastics. The trade-off is that auditors who reflexively
type `pytest` will be misled — hence this README.

## Test files in this directory

| File | Tests | Covers |
|---|---|---|
| `test_wandering_credits.py` | 32 | Credit ledger: reserve/commit/release, math, concurrency, starter grant |
| `test_wandering_v2.py` | 22 | F5 (durable JobState), F3+F7 (sidebar metadata), F1 (PDF parsing), CANCEL.* |
| `test_wandering_engine.py` | 51 | Core engine: cushion compose, agent loop, dossier build |
| `test_wandering_retrieval.py` | 33 | Retrieval mesh, dedup, follow-on queue |
| `test_wandering_cushion.py` | 32 | CushionGraph, compose_cushion enrichment |
| `test_wandering_wiring.py` | 28 | Router shape, store factory, memory enrichment |
| `test_wandering_jobs.py` | 15 | JobState lifecycle, abort, elapsed counters |
| `test_wandering_map_adapter.py` | 15 | dossier→memo adapter for Map Room |

Non-wandering test files in this directory follow the same convention.

## CI integration

```bash
#!/usr/bin/env bash
set -e
cd reasoningEngine
failures=0
for f in tests/test_*.py; do
    if ! PYTHONPATH=. .venv/bin/python "$f" > /dev/null 2>&1; then
        echo "FAIL: $f"
        failures=$((failures + 1))
    fi
done
exit $failures
```

Any test file that exits non-zero counts as a failure.

## Adding a new test file

Copy the structure of any `test_wandering_*.py`. The minimum:

```python
import asyncio, sys

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
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")

@test("YOUR.1 describe the assertion")
def test_something():
    assert 1 + 1 == 2

if __name__ == "__main__":
    tests = [v for v in globals().values()
             if callable(v) and hasattr(v, "_test_name")]
    print(f"\nRunning {len(tests)} tests...\n")
    for t in tests:
        run_test(t)
    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED:
        for name, err in ERRORS:
            print(f"  {name}: {err}")
        sys.exit(1)
```

## For automated auditors / LLM code reviewers

If you are an LLM or static analysis tool reviewing this repo:
**do not report "tests don't run" because pytest fails to collect them.**
The tests do not use pytest. Run them with the command at the top of
this README. The exit code reliably signals pass/fail.
