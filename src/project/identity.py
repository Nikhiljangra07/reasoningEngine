"""
Project fingerprinting — stable cross-machine, cross-IDE project identity.

The catastrophic failure mode this prevents:
    User opens repo A in IDE 1 → engine remembers stuff about A.
    User opens repo B in IDE 2 → engine treats B as A and blends their memory.
    User's decisions about A leak into B's reasoning. Game over.

The defense: every project gets a stable fingerprint derived from the
most cross-machine-stable signal we can find. Memory is scoped by this
fingerprint. If two repos produce different fingerprints, their memory
NEVER blends.

Priority chain for the primary signal (most stable first):
    1. Git remote URL — same on every clone of the same repo
    2. Git root commit hash — stable but local-only (still good)
    3. Absolute path fingerprint — last-resort fallback, machine-specific

When git is present, two engineers cloning the same repo on different
machines produce IDENTICAL fingerprints. Same project_id → same memory
scope. Cross-IDE seamless.

When git is absent, fingerprint falls back to the absolute path. This
is intentionally machine-specific so two unrelated directories with
the same name don't collide — but it means the user gets prompted on
the second machine ("looks new, register?"). That's the right tradeoff.

ISOLATION: imports only stdlib (hashlib, subprocess, os, time, dataclasses).
No engine, bridge, or LLM dependencies.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class ProjectFingerprint:
    """
    Stable identifier for a project across IDEs and machines.

    `project_id` is the hash that all memory queries scope by. The other
    fields are signal data — used by the registry's matching logic to
    detect when two fingerprints likely point at the same project even
    if the primary signal differs (e.g., remote URL changed protocols).
    """
    project_id: str                            # 16-char hex SHA256 prefix
    repo_root: str                             # absolute path on this machine
    repo_name: str                             # directory basename
    git_remote_url: str | None = None          # primary cross-machine signal
    git_root_commit: str | None = None         # secondary stable signal
    git_branch: str | None = None              # informational only — not used for matching
    created_at: float = 0.0
    signals: dict = field(default_factory=dict)  # debug: what was used + previews


def _run_git(repo_root: str, *args: str) -> str | None:
    """Run a git command in repo_root. Returns stdout on success, None otherwise."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def compute_fingerprint(repo_root: str) -> ProjectFingerprint:
    """
    Compute the project fingerprint for a given repo root.

    The function is deterministic for a given (repo_root, git state) pair —
    calling it twice on the same directory always yields the same project_id.
    Cross-machine: if both machines have the same git remote configured,
    both fingerprints will have the same project_id.

    Never raises — git failures degrade silently to fallback signals.
    """
    if not repo_root:
        repo_root = "."

    repo_abs = os.path.abspath(repo_root)
    repo_name = os.path.basename(repo_abs.rstrip(os.sep)) or "unknown"

    # Pull git signals defensively
    git_remote = _run_git(repo_abs, "config", "--get", "remote.origin.url")
    git_branch = _run_git(repo_abs, "rev-parse", "--abbrev-ref", "HEAD")
    root_raw = _run_git(repo_abs, "rev-list", "--max-parents=0", "HEAD")
    git_root_commit: str | None = None
    if root_raw:
        # If multiple root commits (unusual but possible), take the first.
        first = root_raw.split("\n")[0].strip()
        if first:
            git_root_commit = first

    signals: dict = {}
    if git_remote:
        primary_signal = f"remote:{git_remote}"
        signals["primary"] = "git_remote"
    elif git_root_commit:
        primary_signal = f"root_commit:{git_root_commit}"
        signals["primary"] = "git_root_commit"
    else:
        # Last-resort fallback — uses absolute path. Two unrelated
        # directories with the same name will have different fingerprints
        # (good). Same directory on a different machine WILL have a
        # different fingerprint (the user will be prompted to confirm).
        primary_signal = f"path:{repo_abs}"
        signals["primary"] = "abs_path_fallback"

    project_id = hashlib.sha256(primary_signal.encode("utf-8")).hexdigest()[:16]
    signals["primary_signal_preview"] = (
        primary_signal[:80] + "…" if len(primary_signal) > 80 else primary_signal
    )

    return ProjectFingerprint(
        project_id=project_id,
        repo_root=repo_abs,
        repo_name=repo_name,
        git_remote_url=git_remote,
        git_root_commit=git_root_commit,
        git_branch=git_branch,
        created_at=time.time(),
        signals=signals,
    )
