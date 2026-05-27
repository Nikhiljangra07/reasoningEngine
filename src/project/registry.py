"""
Project Registry — tracks which projects the user has authorized.

The registry is the gatekeeper that prevents memory blending. Every
project fingerprint must be registered before any memory query is
scoped to it. When a new fingerprint comes in:

    Exact project_id match           → high-confidence reuse, no prompt
    Git remote match (different id)  → high-confidence reuse, no prompt
                                       (handles remote URL protocol drift,
                                        e.g. https → ssh)
    Git root commit match            → high-confidence reuse, no prompt
    Repo name match, no git signals  → AMBIGUOUS, ask user
    Nothing matches                  → register as new

The disambiguate_project() function emits a structured response for
each case so the conversational layer can surface prompts where needed
and silently proceed where matches are high-confidence.

STORAGE: in-memory by default; optional JSON-file persistence via the
`storage_path` constructor argument. When `storage_path` is supplied:
    - Construction loads any prior state from disk (silent no-op if file
      doesn't exist or is corrupted — registry starts fresh).
    - Every `register()` / `forget()` auto-saves to disk.
    - Recommended path for installs: `~/.constellax/projects.json`.

ISOLATION: imports only src.project.identity and stdlib. No engine,
bridge, or LLM dependencies.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from src.project.identity import ProjectFingerprint


@dataclass
class ProjectIdentityMatch:
    """Result of looking up a fingerprint in the registry."""
    matched: bool
    confidence: float                  # 0.0 - 1.0
    matched_project_id: str | None
    reason: str
    needs_user_confirmation: bool = False


class ProjectRegistry:
    """
    Tracks authorized projects. Single source of truth for "have we seen
    this fingerprint before, and if so under what project_id?"
    """

    def __init__(self, storage_path: str | None = None):
        self._projects: dict[str, ProjectFingerprint] = {}
        self._storage_path = storage_path
        if storage_path:
            self.load()

    # -----------------------------------------------------------------------
    # Mutations
    # -----------------------------------------------------------------------

    def register(self, fp: ProjectFingerprint) -> None:
        """Save a fingerprint. Idempotent — re-registering the same id is a no-op upsert."""
        self._projects[fp.project_id] = fp
        self._autosave()

    def forget(self, project_id: str) -> bool:
        """Remove a registered project. Returns True if it existed."""
        if project_id in self._projects:
            del self._projects[project_id]
            self._autosave()
            return True
        return False

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def lookup(self, fp: ProjectFingerprint) -> ProjectIdentityMatch:
        """
        Try to identify this fingerprint against known projects.

        Match precedence (first hit wins):
            1.00 — exact project_id match
            0.90 — git remote URL matches another project's remote
            0.85 — git root commit matches
            0.50 — repo_name matches but no git signals to confirm
                   (sets needs_user_confirmation=True)
            0.00 — no match
        """
        if not self._projects:
            return ProjectIdentityMatch(
                matched=False, confidence=0.0,
                matched_project_id=None,
                reason="no projects registered yet",
            )

        # 1. Exact project_id match
        if fp.project_id in self._projects:
            return ProjectIdentityMatch(
                matched=True, confidence=1.0,
                matched_project_id=fp.project_id,
                reason="exact project_id match",
            )

        # 2. Git remote URL match (handles same project across machines)
        if fp.git_remote_url:
            for existing in self._projects.values():
                if existing.git_remote_url == fp.git_remote_url:
                    return ProjectIdentityMatch(
                        matched=True, confidence=0.9,
                        matched_project_id=existing.project_id,
                        reason=f"git remote matches: {fp.git_remote_url}",
                    )

        # 3. Git root commit match (stable backup signal)
        if fp.git_root_commit:
            for existing in self._projects.values():
                if existing.git_root_commit == fp.git_root_commit:
                    return ProjectIdentityMatch(
                        matched=True, confidence=0.85,
                        matched_project_id=existing.project_id,
                        reason="git root commit matches",
                    )

        # 4. Repo name match with no git on EITHER side — ambiguous
        for existing in self._projects.values():
            if (existing.repo_name == fp.repo_name
                    and not fp.git_remote_url
                    and not existing.git_remote_url
                    and not fp.git_root_commit
                    and not existing.git_root_commit):
                return ProjectIdentityMatch(
                    matched=True, confidence=0.5,
                    matched_project_id=existing.project_id,
                    reason=(
                        f"repo name matches ({fp.repo_name}) but no git "
                        "signals to confirm — could be an unrelated directory"
                    ),
                    needs_user_confirmation=True,
                )

        return ProjectIdentityMatch(
            matched=False, confidence=0.0,
            matched_project_id=None,
            reason="no known project matches this fingerprint",
        )

    def get(self, project_id: str) -> ProjectFingerprint | None:
        return self._projects.get(project_id)

    def list_projects(self) -> list[ProjectFingerprint]:
        return list(self._projects.values())

    def all_ids(self) -> list[str]:
        return list(self._projects.keys())

    # -----------------------------------------------------------------------
    # Persistence (optional, opt-in via storage_path on construction)
    # -----------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """
        Write current state to disk as JSON. Uses `storage_path` from the
        constructor if `path` is not supplied. No-op if neither is set.
        """
        target = path or self._storage_path
        if not target:
            return
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        data = {
            "projects": [asdict(fp) for fp in self._projects.values()],
        }
        # Write atomically: write to temp file in same dir, then rename.
        # Prevents half-written files on crash.
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, target)

    def load(self, path: str | None = None) -> None:
        """
        Replace in-memory state from disk JSON. Uses `storage_path` if `path`
        not supplied. Silent no-op on:
            - missing file (first install)
            - corrupted JSON (don't crash; user can rebuild via re-registration)
            - schema drift (extra/missing fields handled by ProjectFingerprint
              dataclass defaults)
        """
        target = path or self._storage_path
        if not target or not os.path.exists(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Corrupted file — start fresh rather than crash on startup.
            return

        if not isinstance(data, dict):
            return

        loaded: dict[str, ProjectFingerprint] = {}
        for raw in data.get("projects", []) or []:
            if not isinstance(raw, dict):
                continue
            try:
                fp = ProjectFingerprint(**raw)
            except TypeError:
                # Field mismatch (schema drift) — skip this entry, keep going.
                continue
            loaded[fp.project_id] = fp
        self._projects = loaded

    def _autosave(self) -> None:
        """Called after every mutation when persistence is enabled."""
        if self._storage_path:
            self.save()


# ---------------------------------------------------------------------------
# Conversational disambiguation — what to surface to the user
# ---------------------------------------------------------------------------

def disambiguate_project(
    fingerprint: ProjectFingerprint,
    registry: ProjectRegistry,
) -> dict:
    """
    Decide what to do with a fingerprint and produce a structured response.

    Three outcomes:
        action == "register"  — new project, register and proceed
        action == "matched"   — known project (high confidence), use existing scope
        action == "ask_user"  — ambiguous, surface to user before proceeding

    This is the catastrophic-blending defense. When in doubt, never
    silently reuse memory — ask. The shape mirrors the conversational
    missing-capability response so the frontend reuses parsers.
    """
    match = registry.lookup(fingerprint)

    if not match.matched:
        return {
            "action": "register",
            "project_id": fingerprint.project_id,
            "repo_name": fingerprint.repo_name,
            "fingerprint": fingerprint,
            "message": (
                f"This looks like a new project: {fingerprint.repo_name}. "
                f"I'll scope memory and graphify to fingerprint "
                f"{fingerprint.project_id}."
            ),
        }

    if match.confidence >= 0.85 and not match.needs_user_confirmation:
        return {
            "action": "matched",
            "project_id": match.matched_project_id,
            "confidence": match.confidence,
            "reason": match.reason,
            "message": f"Recognized as known project ({match.reason}).",
        }

    # Ambiguous — needs user input
    return {
        "action": "ask_user",
        "matched_project_id": match.matched_project_id,
        "confidence": match.confidence,
        "reason": match.reason,
        "current_fingerprint": fingerprint,
        "message": (
            f"This might be a project I've seen before "
            f"(id={match.matched_project_id}) — {match.reason}. "
            "Is this the same project, or a new one? If same, I'll reuse "
            "the prior memory; if new, I'll register a fresh scope so "
            "nothing blends."
        ),
        "user_options": [
            {"id": "same", "label": "Same project — use existing memory"},
            {"id": "new",  "label": "New project — register fresh, don't blend"},
        ],
    }
