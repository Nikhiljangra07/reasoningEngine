"""
Bridge — graphify (code structure) ↔ Memory V2 (decision memory).

This package connects two complementary memory systems without blending
them. They remain distinct stores. The bridge offers a unified READ
interface and a drift-detection comparison layer.

- graphify  = lossless code structure (functions, calls, schemas).
- Memory V2 = lossy decision memory (anchors in Falkor + fingerprints in
              Chroma). Currently implemented for life-decision context in
              lora-v1-frontend; will be ported to reasoningEngine in a
              separate task. This package defines the interface that port
              will fill.

Public surface:
    BridgeClient         — the facade wuxing calls
    DecisionAnchor       — a single decision with code back-pointers
    CodeRef              — a file:line reference into the codebase
    ContextFingerprint   — vector embedding of the conditions a decision was made under
    DriftReport          — output of detect_drift()
    BridgeQuery / BridgeResult — placeholders for future composite queries
"""

from __future__ import annotations

from src.bridge.client import BridgeClient
from src.bridge.types import (
    BridgeQuery,
    BridgeResult,
    CodeRef,
    ContextFingerprint,
    DecisionAnchor,
    DriftReport,
)

__all__ = [
    "BridgeClient",
    "BridgeQuery",
    "BridgeResult",
    "CodeRef",
    "ContextFingerprint",
    "DecisionAnchor",
    "DriftReport",
]
