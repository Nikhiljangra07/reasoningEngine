"""
Project identity package — prevents memory blending across projects.

Public symbols re-exported here so callers can write:
    from src.project import ProjectFingerprint, ProjectRegistry, ...
"""

from __future__ import annotations

from src.project.identity import (
    ProjectFingerprint,
    compute_fingerprint,
)
from src.project.registry import (
    ProjectIdentityMatch,
    ProjectRegistry,
    disambiguate_project,
)

__all__ = [
    "ProjectFingerprint",
    "ProjectIdentityMatch",
    "ProjectRegistry",
    "compute_fingerprint",
    "disambiguate_project",
]
