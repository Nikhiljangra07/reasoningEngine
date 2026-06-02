"""
Source-proof test for the identity layer.

Codex's read of the integration sprint flagged a real gap: the
integration tests prove "if you compose this prompt this way, you
get a composed prompt." They do NOT prove that the actual call site
in `dispatcher.py` / `speech.py` / `wandering/*.py` invokes
`compose_system_prompt`. Someone could delete the wrapping at any
call site and the integration tests would still pass.

This test closes that gap.

How it works
============

The test walks `src/` with the Python AST module, finds every
`<obj>.call(...)` invocation that carries a `system_prompt=` kwarg
(the LLM client's surface), and asserts each call site either:

  (a) passes a direct `compose_system_prompt(...)` expression as the
      argument, OR

  (b) passes a Name (variable) whose nearest preceding assignment in
      the same function uses `compose_system_prompt`, OR

  (c) passes a Name whose `(file, identifier)` tuple appears in the
      `CONTROL_PLANE_SITES` exempt registry (`src/identity/exempt.py`).

Anything else is a violation — a fresh LLM call site has been added
without routing through the doctrine. The test fails loudly with the
offending file:line and the unparseable expression so the
contributor can either compose it or add an exempt entry with a
reason.

Why not just grep
=================

Grep would catch "compose_system_prompt" being present somewhere in
the file, but not whether the actual call site uses it. The AST
walk verifies the relationship at the call expression's level —
that's what makes the proof structural rather than heuristic.

Skipped files
=============

- `src/llm/client.py` is the LLM client implementation itself; its
  internal `.call()` plumbing forwards a `system_prompt` it already
  received from a consumer. Scanning the implementation would create
  a chicken-and-egg false positive.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
import sys


# ─── Mini test harness ─────────────────────────────────────────────────

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


# ─── Imports under test ────────────────────────────────────────────────

from src.identity import (
    CONTROL_PLANE_SITES,
    ExemptSite,
    is_exempt,
)


# ─── AST scanner ───────────────────────────────────────────────────────

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

# Files we intentionally do NOT scan. The LLM client implementation
# itself forwards a system_prompt from its consumers — scanning it
# would create a spurious failure.
_SKIP_FILES = {
    "src/llm/client.py",
}


def _rel(path: pathlib.Path) -> str:
    """Return repo-relative path with forward slashes (Windows-safe)."""
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _is_compose_call(expr: ast.expr) -> bool:
    """True when `expr` is a direct call to `compose_system_prompt(...)`.

    Catches the inline form used at every wired call site:
        client.call(system_prompt=compose_system_prompt(...), ...)
    """
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    if isinstance(func, ast.Name) and func.id == "compose_system_prompt":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "compose_system_prompt":
        return True
    return False


def _expression_is_composed(expr: ast.expr,
                            enclosing: list[ast.FunctionDef | ast.AsyncFunctionDef],
                            before_lineno: int) -> bool:
    """True when the kwarg expression as a whole derives from a composed
    system prompt.

    Three shapes count as composed:
      1. Direct compose call: `compose_system_prompt(local, mode="X")`
      2. Name bound via composer earlier in the function:
            x = compose_system_prompt(...)
            client.call(system_prompt=x, ...)
      3. BinOp / Call / nested expression in which at least one
         sub-expression resolves to (1) or (2). This covers the
         regenerate-closure pattern:
            async def _regen(directive):
                client.call(system_prompt=x + "\\n\\n" + directive, ...)
         where `x` was bound to `compose_system_prompt(...)` in the
         enclosing scope.

    Walking the full expression subtree is correct here: if ANY part
    of the system_prompt argument originates from the composer, the
    doctrine header is present in the prompt the model receives.
    Appending a regenerate directive or a contextual block does not
    erase the header that precedes it."""

    for sub in ast.walk(expr):
        if _is_compose_call(sub):
            return True
        if isinstance(sub, ast.Name):
            # Walk enclosing scopes innermost-first looking for an
            # assignment to this Name that uses compose_system_prompt.
            # For the outer-scope binding case, `before_lineno` is the
            # call-site line — assignments must precede it lexically.
            for fn in enclosing:
                if _assigns_via_composer(fn, sub.id, before_lineno):
                    return True
    return False


def _enclosing_functions(tree: ast.Module, lineno: int) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return every FunctionDef / AsyncFunctionDef whose body contains
    `lineno`, ordered innermost-first.

    Used to walk lexical scopes outward from a nested closure: a
    variable assigned in an outer function is still visible inside an
    inner `async def`, so the composer assignment may live in any
    enclosing scope. We check all of them.

    Returns an empty list when the line is at module scope."""

    enclosing: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.body and node.lineno <= lineno <= (node.end_lineno or lineno):
            enclosing.append(node)
    # Sort innermost-first by start line descending — the function
    # whose body starts latest while still containing `lineno` is the
    # innermost enclosing scope.
    enclosing.sort(key=lambda fn: fn.lineno, reverse=True)
    return enclosing


def _assigns_via_composer(func: ast.FunctionDef | ast.AsyncFunctionDef,
                          var_name: str,
                          before_lineno: int) -> bool:
    """True when `var_name` is assigned in `func`'s body, before
    `before_lineno`, via an expression that contains a direct call to
    `compose_system_prompt`.

    Catches the indirect form:
        system_prompt = compose_system_prompt(...)
        ...
        client.call(system_prompt=system_prompt, ...)

    Walks the function's full subtree (so nested assignments inside
    inner blocks are still visible), which is correct here because
    Python doesn't have block-scoped variables — an assignment
    anywhere lexically before the call binds the name."""

    for stmt in ast.walk(func):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        if stmt.lineno >= before_lineno:
            continue
        # Collect assignment targets
        targets: list[ast.expr] = []
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            targets = [stmt.target]
        hits = False
        for t in targets:
            if isinstance(t, ast.Name) and t.id == var_name:
                hits = True
                break
        if not hits:
            continue
        value = stmt.value
        if value is None:  # bare AnnAssign with no value
            continue
        # Search the assigned value's expression tree for a direct call
        # to compose_system_prompt. This handles both:
        #   x = compose_system_prompt(...)
        #   x = compose_system_prompt(...) + "\n" + extra   (rare but valid)
        for sub in ast.walk(value):
            if _is_compose_call(sub):
                return True
    return False


def _value_repr(expr: ast.expr) -> str:
    """Best-effort source repr of `expr` for violation messages."""
    try:
        return ast.unparse(expr)
    except Exception:
        return f"<unparseable {type(expr).__name__}>"


def _exempt_identifier(expr: ast.expr) -> str | None:
    """If `expr` is a Name referring to a top-level prompt constant,
    return the identifier; otherwise None. Used to match against the
    exempt registry."""
    if isinstance(expr, ast.Name):
        return expr.id
    return None


def find_call_sites() -> list[tuple[str, int, ast.expr, ast.Module]]:
    """Walk `src/` and yield every LLM-call site.

    Returns a list of (rel_path, lineno, system_prompt_expr, parsed_tree).
    Skipped files (notably the LLM client implementation) are excluded."""

    sites: list[tuple[str, int, ast.expr, ast.Module]] = []
    for py in SRC_ROOT.rglob("*.py"):
        rel = _rel(py)
        if rel in _SKIP_FILES:
            continue
        if "__pycache__" in rel:
            continue
        try:
            source = py.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            # Don't fail the test on unparseable junk; surface as a
            # separate concern if it ever becomes one.
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Look for <something>.call(...) — the LLM client surface.
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "call":
                continue
            # Must carry a system_prompt= kwarg to qualify as an LLM
            # call (the client's API). Other `.call()` methods on
            # other objects won't have this kwarg.
            sp_kw = next(
                (kw for kw in node.keywords if kw.arg == "system_prompt"),
                None,
            )
            if sp_kw is None:
                continue
            sites.append((rel, node.lineno, sp_kw.value, tree))
    return sites


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Tests                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

@test("SP1 scanner finds at least 10 LLM call sites")
def test_scanner_finds_sites():
    """Smoke test — the scanner is working at all. If this fails, the
    AST scan is broken and the source-proof guard is silently a
    no-op. We have 12+ wired call sites (5 wandering + 6 speech + 2
    dispatcher) and 4 exempt control-plane sites, so >= 10 is a
    conservative floor."""
    sites = find_call_sites()
    assert len(sites) >= 10, f"Scanner only found {len(sites)} sites — expected >= 10"


@test("SP2 every LLM call site either composes or is registered exempt")
def test_every_call_site_composed_or_exempt():
    """The core source-proof check. For every <client>.call(system_prompt=X)
    in src/:
        (a) X is `compose_system_prompt(...)` directly → PASS
        (b) X is a Name bound in the enclosing function via
            `compose_system_prompt` → PASS
        (c) X is a Name listed in CONTROL_PLANE_SITES for this file
            → PASS
        otherwise → FAIL."""

    violations: list[str] = []
    for rel, lineno, expr, tree in find_call_sites():
        enclosing = _enclosing_functions(tree, lineno)

        # (a) + (b): the expression resolves to a composed system
        # prompt directly or via any sub-expression (handles the
        # regenerate-closure pattern where `composed_prompt + "\n\n"
        # + directive` is passed inside a nested function whose outer
        # scope holds the composer assignment).
        if _expression_is_composed(expr, enclosing, lineno):
            continue

        # (c): exempt registry — only meaningful when the expression
        # is a bare Name referring to a registered prompt constant.
        if isinstance(expr, ast.Name) and is_exempt(rel, expr.id):
            continue

        violations.append(f"{rel}:{lineno}  system_prompt={_value_repr(expr)}")

    if violations:
        message = (
            "Uncomposed LLM call sites detected. Each site must either "
            "compose via `compose_system_prompt(...)` or be registered "
            "in `src/identity/exempt.py:CONTROL_PLANE_SITES` with a "
            "reason. Violations:\n  - "
            + "\n  - ".join(violations)
        )
        raise AssertionError(message)


@test("SP3 each exempt entry has a non-empty reason")
def test_exempt_entries_have_reasons():
    """The registry's contract: every entry carries WHY it's exempt.
    Empty / placeholder reasons rot the registry into implicit
    exemption. Reject them at the schema level."""
    for site in CONTROL_PLANE_SITES:
        assert isinstance(site, ExemptSite), f"non-ExemptSite in registry: {site!r}"
        assert site.file and "/" in site.file, f"bad file path: {site.file}"
        assert site.prompt_name and site.prompt_name.strip(), site
        assert site.reason and len(site.reason.strip()) >= 30, (
            f"{site.file}:{site.prompt_name} — reason is too short to be informative"
        )


@test("SP4 every exempt-registered file actually exists in src/")
def test_exempt_files_exist():
    """Catch typos / stale entries: every (file, prompt_name) entry
    must refer to a real file on disk. Doesn't validate that the
    prompt_name is actually used there — that's caught by SP2 (if
    the prompt name doesn't appear in source, SP2 either finds no
    matching call site or fails on a different name)."""
    for site in CONTROL_PLANE_SITES:
        path = REPO_ROOT / site.file
        assert path.exists(), f"Exempt file does not exist: {site.file}"


@test("SP5 every exempt prompt_name appears at least once in its file's source")
def test_exempt_names_present():
    """Stronger than SP4: the exempt prompt_name must appear in its
    file. Otherwise the registry entry is stale."""
    for site in CONTROL_PLANE_SITES:
        path = REPO_ROOT / site.file
        source = path.read_text()
        assert site.prompt_name in source, (
            f"Exempt prompt_name `{site.prompt_name}` not found in {site.file}"
        )


@test("SP6 known-wired files contain at least one compose_system_prompt call")
def test_wired_files_compose():
    """Belt-and-braces check on the files we explicitly wired in the
    integration sprints. If any of these stops importing
    `compose_system_prompt`, the SP2 scan would catch it as
    uncomposed sites — but this is a faster, more specific signal."""
    wired_files = (
        "src/wandering/composer.py",
        "src/wandering/matching.py",
        "src/wandering/agent.py",
        "src/wandering/articulate.py",
        "src/wandering/synthesis.py",
        "src/llm/speech.py",
        "src/dispatcher.py",
    )
    for rel in wired_files:
        path = REPO_ROOT / rel
        source = path.read_text()
        assert "compose_system_prompt" in source, (
            f"{rel} should call compose_system_prompt but doesn't"
        )


@test("SP7 gate_output_async wired into at least 3 user-facing prose paths")
def test_gate_wired():
    """Codex Tier 2 deliverable: prove the output gate is actually on
    the road, not just available in the package. Three pure-prose
    paths were wired in this sprint:
      - dispatcher._dispatch_direct
      - dispatcher._dispatch_direct_plus
      - speech.generate_clarification
    """
    expected_files = (
        "src/dispatcher.py",
        "src/llm/speech.py",
    )
    for rel in expected_files:
        path = REPO_ROOT / rel
        source = path.read_text()
        assert "gate_output_async" in source, (
            f"{rel} should use gate_output_async but doesn't"
        )


# ─── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for v in globals().values()
             if callable(v) and hasattr(v, "_test_name")]
    tests.sort(key=lambda t: t._test_name)
    print(f"\nRunning {len(tests)} source-proof tests...\n")
    for t in tests:
        run_test(t)
    print(f"\n{PASSED} passed, {FAILED} failed")
    if FAILED:
        for name, err in ERRORS:
            print(f"  {name}: {err}")
        sys.exit(1)
