"""
Project identity + registry tests.

Verifies:
    1. Fingerprint of a git repo uses git-remote signal
    2. Fingerprint of a non-git tempdir falls back to path
    3. Determinism — same input always yields same project_id
    4. Different inputs yield different project_ids
    5. Registry exact-id match
    6. Registry git-remote match (cross-machine simulation)
    7. Registry root-commit match
    8. Registry repo-name match → needs_user_confirmation
    9. Registry no-match
    10. disambiguate_project produces correct action for each scenario

No LLM calls, no API. Uses real subprocess for git probing but degrades
silently when git is missing.

Run: PYTHONPATH=. python3 tests/test_project_identity.py
"""

from __future__ import annotations

import json
import os
import tempfile

from src.project import (
    ProjectFingerprint,
    ProjectRegistry,
    compute_fingerprint,
    disambiguate_project,
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
        fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ---------------------------------------------------------------------------
# 1. Fingerprint of a real git repo (use this very repo)
# ---------------------------------------------------------------------------

@test("1.1 fingerprint of reasoningEngine repo uses git-remote or root-commit")
def test_fingerprint_git_repo():
    # This repo is /Users/nikhil/Desktop/reasoningEngine — has git
    fp = compute_fingerprint("/Users/nikhil/Desktop/reasoningEngine")
    assert fp.project_id, "project_id should be non-empty"
    assert fp.repo_name, "repo_name should be non-empty"
    # Either remote URL or root commit should be populated (this repo has git)
    assert fp.git_remote_url or fp.git_root_commit, (
        "expected at least one git signal on this repo"
    )
    # Primary signal must be one of the git options, NOT the path fallback
    assert fp.signals["primary"] in ("git_remote", "git_root_commit")


# ---------------------------------------------------------------------------
# 2. Fingerprint of a non-git tempdir falls back to path
# ---------------------------------------------------------------------------

@test("2.1 fingerprint of non-git tempdir falls back to abs_path")
def test_fingerprint_no_git():
    with tempfile.TemporaryDirectory() as tmp:
        fp = compute_fingerprint(tmp)
        assert fp.project_id
        assert fp.git_remote_url is None
        assert fp.git_root_commit is None
        assert fp.signals["primary"] == "abs_path_fallback"


# ---------------------------------------------------------------------------
# 3. Determinism — same input always yields the same project_id
# ---------------------------------------------------------------------------

@test("3.1 fingerprinting same git repo twice yields same project_id")
def test_determinism_git():
    fp1 = compute_fingerprint("/Users/nikhil/Desktop/reasoningEngine")
    fp2 = compute_fingerprint("/Users/nikhil/Desktop/reasoningEngine")
    assert fp1.project_id == fp2.project_id


@test("3.2 fingerprinting same tempdir twice yields same project_id")
def test_determinism_nogit():
    with tempfile.TemporaryDirectory() as tmp:
        fp1 = compute_fingerprint(tmp)
        fp2 = compute_fingerprint(tmp)
        assert fp1.project_id == fp2.project_id


# ---------------------------------------------------------------------------
# 4. Different inputs yield different project_ids
# ---------------------------------------------------------------------------

@test("4.1 two unrelated tempdirs have different project_ids")
def test_different_dirs_differ():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        fp_a = compute_fingerprint(a)
        fp_b = compute_fingerprint(b)
        assert fp_a.project_id != fp_b.project_id


# ---------------------------------------------------------------------------
# 5. Registry exact-id match
# ---------------------------------------------------------------------------

@test("5.1 lookup after register returns confidence=1.0")
def test_registry_exact_match():
    reg = ProjectRegistry()
    fp = compute_fingerprint("/Users/nikhil/Desktop/reasoningEngine")
    reg.register(fp)
    match = reg.lookup(fp)
    assert match.matched is True
    assert match.confidence == 1.0
    assert match.matched_project_id == fp.project_id


# ---------------------------------------------------------------------------
# 6. Registry git-remote match (different path → same remote → same project)
# ---------------------------------------------------------------------------

@test("6.1 different project_id but same git_remote → 0.9 match")
def test_registry_remote_match():
    reg = ProjectRegistry()
    # Simulate the "same project, different machine" scenario:
    # two fingerprints with different project_ids but the same git remote.
    fp_machine_a = ProjectFingerprint(
        project_id="aaaaaaaaaaaaaaaa",
        repo_root="/Users/alice/projects/foo",
        repo_name="foo",
        git_remote_url="git@github.com:org/foo.git",
        git_root_commit="abc123",
    )
    fp_machine_b = ProjectFingerprint(
        project_id="bbbbbbbbbbbbbbbb",  # different id (hypothetically)
        repo_root="/Users/bob/code/foo",
        repo_name="foo",
        git_remote_url="git@github.com:org/foo.git",   # SAME remote
        git_root_commit="xyz789",
    )
    reg.register(fp_machine_a)
    match = reg.lookup(fp_machine_b)
    assert match.matched is True
    assert 0.85 < match.confidence < 1.0
    assert match.matched_project_id == "aaaaaaaaaaaaaaaa"
    assert "git remote" in match.reason


# ---------------------------------------------------------------------------
# 7. Registry git-root-commit match (fallback when remote differs)
# ---------------------------------------------------------------------------

@test("7.1 same root commit but different remotes → 0.85 match")
def test_registry_root_commit_match():
    reg = ProjectRegistry()
    fp_a = ProjectFingerprint(
        project_id="aaaa", repo_root="/p1", repo_name="proj",
        git_remote_url="git@github.com:old/repo.git",
        git_root_commit="commit_xyz",
    )
    fp_b = ProjectFingerprint(
        project_id="bbbb", repo_root="/p2", repo_name="proj",
        git_remote_url="git@github.com:new/repo.git",   # different remote (renamed/forked)
        git_root_commit="commit_xyz",                   # SAME root commit
    )
    reg.register(fp_a)
    match = reg.lookup(fp_b)
    assert match.matched is True
    assert match.confidence == 0.85
    assert "root commit" in match.reason


# ---------------------------------------------------------------------------
# 8. Registry ambiguous-name match → needs user confirmation
# ---------------------------------------------------------------------------

@test("8.1 same repo_name without git on either side → 0.5 + needs_user_confirmation")
def test_registry_ambiguous():
    reg = ProjectRegistry()
    fp_a = ProjectFingerprint(
        project_id="aaaa", repo_root="/u1/notes", repo_name="notes",
    )
    fp_b = ProjectFingerprint(
        project_id="bbbb", repo_root="/u2/notes", repo_name="notes",
    )
    reg.register(fp_a)
    match = reg.lookup(fp_b)
    assert match.matched is True
    assert match.confidence == 0.5
    assert match.needs_user_confirmation is True


@test("8.2 same repo_name but one side HAS git signals → no ambiguous match")
def test_registry_name_match_blocked_by_git():
    reg = ProjectRegistry()
    fp_a = ProjectFingerprint(
        project_id="aaaa", repo_root="/u1/notes", repo_name="notes",
        git_remote_url="git@github.com:org/notes.git",
    )
    fp_b = ProjectFingerprint(
        project_id="bbbb", repo_root="/u2/notes", repo_name="notes",
        # no git
    )
    reg.register(fp_a)
    match = reg.lookup(fp_b)
    # Should NOT be the ambiguous match — fp_a has git signals that fp_b can't match
    assert match.matched is False


# ---------------------------------------------------------------------------
# 9. Registry no-match
# ---------------------------------------------------------------------------

@test("9.1 lookup on empty registry returns no match")
def test_registry_empty():
    reg = ProjectRegistry()
    fp = ProjectFingerprint(
        project_id="aaaa", repo_root="/x", repo_name="x",
    )
    match = reg.lookup(fp)
    assert match.matched is False
    assert match.confidence == 0.0


@test("9.2 lookup of unrelated project returns no match")
def test_registry_unrelated():
    reg = ProjectRegistry()
    reg.register(ProjectFingerprint(
        project_id="aaaa", repo_root="/p1", repo_name="alpha",
        git_remote_url="git@github.com:org/alpha.git",
    ))
    fp = ProjectFingerprint(
        project_id="bbbb", repo_root="/p2", repo_name="beta",
        git_remote_url="git@github.com:org/beta.git",
    )
    match = reg.lookup(fp)
    assert match.matched is False


# ---------------------------------------------------------------------------
# 10. disambiguate_project produces correct action for each scenario
# ---------------------------------------------------------------------------

@test("10.1 disambiguate: empty registry → action='register'")
def test_disambiguate_new():
    reg = ProjectRegistry()
    fp = ProjectFingerprint(
        project_id="aaaa", repo_root="/x", repo_name="x",
    )
    out = disambiguate_project(fp, reg)
    assert out["action"] == "register"


@test("10.2 disambiguate: exact match → action='matched'")
def test_disambiguate_matched():
    reg = ProjectRegistry()
    fp = ProjectFingerprint(
        project_id="aaaa", repo_root="/x", repo_name="x",
    )
    reg.register(fp)
    out = disambiguate_project(fp, reg)
    assert out["action"] == "matched"
    assert out["confidence"] == 1.0


@test("10.3 disambiguate: ambiguous name match → action='ask_user' with options")
def test_disambiguate_ask():
    reg = ProjectRegistry()
    fp_a = ProjectFingerprint(
        project_id="aaaa", repo_root="/u1/notes", repo_name="notes",
    )
    fp_b = ProjectFingerprint(
        project_id="bbbb", repo_root="/u2/notes", repo_name="notes",
    )
    reg.register(fp_a)
    out = disambiguate_project(fp_b, reg)
    assert out["action"] == "ask_user"
    assert len(out["user_options"]) == 2
    option_ids = [o["id"] for o in out["user_options"]]
    assert "same" in option_ids
    assert "new" in option_ids


@test("10.4 disambiguate: high-confidence remote match → action='matched' (no prompt)")
def test_disambiguate_silent_high_conf():
    reg = ProjectRegistry()
    fp_a = ProjectFingerprint(
        project_id="aaaa", repo_root="/u1", repo_name="foo",
        git_remote_url="git@github.com:org/foo.git",
    )
    fp_b = ProjectFingerprint(
        project_id="bbbb", repo_root="/u2", repo_name="foo",
        git_remote_url="git@github.com:org/foo.git",
    )
    reg.register(fp_a)
    out = disambiguate_project(fp_b, reg)
    # confidence=0.9, no needs_user_confirmation → matched silently
    assert out["action"] == "matched"
    assert out["confidence"] == 0.9


# ---------------------------------------------------------------------------
# 11. Registry housekeeping
# ---------------------------------------------------------------------------

@test("11.1 forget() removes a registered project")
def test_forget():
    reg = ProjectRegistry()
    fp = ProjectFingerprint(project_id="aaaa", repo_root="/x", repo_name="x")
    reg.register(fp)
    assert reg.forget("aaaa") is True
    assert reg.forget("aaaa") is False  # already gone
    assert reg.get("aaaa") is None


@test("11.2 list_projects returns all registered fingerprints")
def test_list():
    reg = ProjectRegistry()
    for i in range(3):
        reg.register(ProjectFingerprint(
            project_id=f"id{i}", repo_root=f"/p{i}", repo_name=f"name{i}",
        ))
    assert len(reg.list_projects()) == 3
    assert set(reg.all_ids()) == {"id0", "id1", "id2"}


# ---------------------------------------------------------------------------
# 12. Catastrophic-prevention smoke: two unrelated repos NEVER match
# ---------------------------------------------------------------------------

@test("12.1 two unrelated projects with different remotes NEVER match")
def test_catastrophic_prevention():
    reg = ProjectRegistry()
    project_a = ProjectFingerprint(
        project_id="aaaa", repo_root="/u/projA", repo_name="projA",
        git_remote_url="git@github.com:org/projA.git",
        git_root_commit="aaaaa",
    )
    project_b = ProjectFingerprint(
        project_id="bbbb", repo_root="/u/projB", repo_name="projB",
        git_remote_url="git@github.com:org/projB.git",
        git_root_commit="bbbbb",
    )
    reg.register(project_a)
    match = reg.lookup(project_b)
    assert match.matched is False
    # And disambiguate must mark B as a NEW project, not match A
    out = disambiguate_project(project_b, reg)
    assert out["action"] == "register"


# ---------------------------------------------------------------------------
# 13. Disk persistence (opt-in via storage_path)
# ---------------------------------------------------------------------------

@test("13.1 no storage_path → in-memory only (back-compat)")
def test_no_persistence():
    reg = ProjectRegistry()
    reg.register(ProjectFingerprint(project_id="x", repo_root="/x", repo_name="x"))
    assert len(reg.list_projects()) == 1
    # No file should have been created anywhere
    # (just verify call didn't raise; absence of file is the guarantee)


@test("13.2 register() writes to disk when storage_path is set")
def test_persistence_register_writes():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "projects.json")
        reg = ProjectRegistry(storage_path=path)
        reg.register(ProjectFingerprint(
            project_id="abc", repo_root="/r", repo_name="r",
            git_remote_url="git@github.com:org/r.git",
            git_root_commit="xyz",
        ))
        assert os.path.exists(path), "expected file to be written on register"
        with open(path) as f:
            data = json.load(f)
        assert "projects" in data
        ids = [p["project_id"] for p in data["projects"]]
        assert "abc" in ids


@test("13.3 second registry loads prior state from disk")
def test_persistence_load_on_construct():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "projects.json")
        reg1 = ProjectRegistry(storage_path=path)
        reg1.register(ProjectFingerprint(
            project_id="abc", repo_root="/r", repo_name="myrepo",
            git_remote_url="git@github.com:org/r.git",
        ))
        # Fresh registry pointing at same file should load the same data
        reg2 = ProjectRegistry(storage_path=path)
        loaded = reg2.get("abc")
        assert loaded is not None
        assert loaded.repo_name == "myrepo"
        assert loaded.git_remote_url == "git@github.com:org/r.git"


@test("13.4 forget() updates disk")
def test_persistence_forget():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "projects.json")
        reg = ProjectRegistry(storage_path=path)
        reg.register(ProjectFingerprint(
            project_id="abc", repo_root="/r", repo_name="r",
        ))
        reg.forget("abc")
        # Reload — should be gone
        reg2 = ProjectRegistry(storage_path=path)
        assert reg2.get("abc") is None


@test("13.5 corrupted JSON → registry starts fresh, no crash")
def test_persistence_corrupted():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "projects.json")
        with open(path, "w") as f:
            f.write("this is not valid json {{{ ]]]")
        # Should NOT raise — degrades silently to empty
        reg = ProjectRegistry(storage_path=path)
        assert reg.list_projects() == []


@test("13.6 missing file → empty registry, no error")
def test_persistence_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "does_not_exist.json")
        reg = ProjectRegistry(storage_path=path)
        assert reg.list_projects() == []
        # File still doesn't exist (no auto-create on construct without mutation)
        assert not os.path.exists(path)


@test("13.7 nested storage path creates parent directories")
def test_persistence_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "subdir", "projects.json")
        reg = ProjectRegistry(storage_path=path)
        reg.register(ProjectFingerprint(
            project_id="x", repo_root="/x", repo_name="x",
        ))
        assert os.path.exists(path)


@test("13.8 schema drift — extra fields in JSON don't crash load")
def test_persistence_schema_drift():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "projects.json")
        # Write a JSON with an unknown extra field on a project entry
        bad = {
            "projects": [
                {
                    "project_id": "abc",
                    "repo_root": "/r",
                    "repo_name": "r",
                    "git_remote_url": None,
                    "git_root_commit": None,
                    "git_branch": None,
                    "created_at": 0.0,
                    "signals": {},
                    "unknown_future_field": "ignore me",  # not in dataclass
                },
            ],
        }
        with open(path, "w") as f:
            json.dump(bad, f)
        # The unknown field will cause ProjectFingerprint(**raw) to raise
        # TypeError → load() should skip this entry, not crash.
        reg = ProjectRegistry(storage_path=path)
        assert reg.list_projects() == []  # entry with extra field was skipped


@test("13.9 atomic write: .tmp file does not survive after save")
def test_persistence_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "projects.json")
        reg = ProjectRegistry(storage_path=path)
        reg.register(ProjectFingerprint(project_id="x", repo_root="/x", repo_name="x"))
        # Atomic write means we use a .tmp + rename; .tmp should not remain
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_fingerprint_git_repo,
    test_fingerprint_no_git,
    test_determinism_git,
    test_determinism_nogit,
    test_different_dirs_differ,
    test_registry_exact_match,
    test_registry_remote_match,
    test_registry_root_commit_match,
    test_registry_ambiguous,
    test_registry_name_match_blocked_by_git,
    test_registry_empty,
    test_registry_unrelated,
    test_disambiguate_new,
    test_disambiguate_matched,
    test_disambiguate_ask,
    test_disambiguate_silent_high_conf,
    test_forget,
    test_list,
    test_catastrophic_prevention,
    test_no_persistence,
    test_persistence_register_writes,
    test_persistence_load_on_construct,
    test_persistence_forget,
    test_persistence_corrupted,
    test_persistence_missing_file,
    test_persistence_creates_parent_dirs,
    test_persistence_schema_drift,
    test_persistence_atomic,
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} project identity tests...")
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
