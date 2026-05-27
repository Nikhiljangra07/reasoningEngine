"""
Capabilities package — registry of what the engine can and can't access.

Single public symbol set is re-exported here so callers can write:
    from src.capabilities import CapabilityRegistry, PermissionState, ...
"""

from __future__ import annotations

from src.capabilities.registry import (
    Capability,
    CapabilityRegistry,
    CapabilityStatus,
    CapabilityType,
    CapabilityUsage,
    PermissionState,
)

__all__ = [
    "Capability",
    "CapabilityRegistry",
    "CapabilityStatus",
    "CapabilityType",
    "CapabilityUsage",
    "PermissionState",
]
