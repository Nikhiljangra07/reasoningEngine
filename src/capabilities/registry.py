"""
Capability Registry — single source of truth for what the engine can access.

Tracks every MCP / external capability Constellax knows about:
    - status         (AVAILABLE / MISSING / ABSENT_BY_DESIGN)
    - permission     (ALWAYS_ALLOWED / ASK_ONCE_PENDING / ASK_ONCE_GRANTED / DENIED / ABSENT_BY_DESIGN)
    - description    (one-sentence human-readable: what this lets Constellax do)
    - install_hint   (CLI command or URL to connect it, if applicable)
    - usage_log      (history of firings for "checking GitHub for #482..." notifications)

Two-tier permission model (locked design):
    ALWAYS_ALLOWED      — local reads (memory, graph, bridge). No prompt. Safe.
    ASK_ONCE_PENDING    — external reads (web_search, github, docs, browser).
                          First call prompts the user, subsequent calls notify
                          transparently. Graduates to ASK_ONCE_GRANTED on consent.
    DENIED              — user revoked.
    ABSENT_BY_DESIGN    — writes (file edits, shell exec, send_message). Not a
                          permission state — the engine has no hands. Cannot
                          be flipped to AVAILABLE.

This is the brain-extension safety property: writes aren't denied, they
don't exist. The registry enforces that constraint structurally — every
mutation method refuses to alter an ABSENT_BY_DESIGN capability.

Stubbed in this drop: _DEFAULT_CAPABILITIES is hardcoded. The real MCP
router (Step 8) will update statuses based on actual connection state at
startup.

ISOLATION: no imports from the engine, bridge, or LLM client. Pure data
+ small state machine.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CapabilityType(str, Enum):
    """What kind of operation this capability represents."""
    LOCAL_READ = "local_read"          # memory_v2, graphify, bridge — local data
    EXTERNAL_READ = "external_read"    # web_search, github, docs, browser — network
    WRITE = "write"                    # edits, commands, sends — absent in our product


class CapabilityStatus(str, Enum):
    """Whether this capability is wired up RIGHT NOW."""
    AVAILABLE = "available"
    MISSING = "missing"
    ABSENT_BY_DESIGN = "absent_by_design"


class PermissionState(str, Enum):
    """The user's stance on this capability for the current scope."""
    ALWAYS_ALLOWED = "always_allowed"        # local reads, or external after consent
    ASK_ONCE_PENDING = "ask_once_pending"    # external read awaiting first consent
    ASK_ONCE_GRANTED = "ask_once_granted"    # consented; subsequent calls notify only
    DENIED = "denied"
    ABSENT_BY_DESIGN = "absent_by_design"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class CapabilityUsage:
    """A single firing of a capability — feeds 'checking X for Y...' notifications."""
    capability: str
    purpose: str
    timestamp: float
    success: bool = True
    error: str | None = None


@dataclass
class Capability:
    """A registered capability — identity, current state, human description."""
    name: str
    type: CapabilityType
    status: CapabilityStatus
    permission: PermissionState
    description: str
    install_hint: str = ""


# ---------------------------------------------------------------------------
# Default seed — what we declare we know about
# ---------------------------------------------------------------------------

_DEFAULT_CAPABILITIES: list[Capability] = [
    # Local reads — always allowed in principle. Status reflects whether
    # the underlying data/file is present.
    Capability(
        name="memory_v2",
        type=CapabilityType.LOCAL_READ,
        status=CapabilityStatus.AVAILABLE,
        permission=PermissionState.ALWAYS_ALLOWED,
        description="Reads prior decisions and cross-thread iterations from the Memory V2 store (Neo4j-backed Decision Trace).",
        install_hint="Set NEO4J_URI + NEO4J_PASSWORD; the server wires Neo4jAnchorBackend + MemoryRetriever automatically.",
    ),
    Capability(
        name="graphify",
        type=CapabilityType.LOCAL_READ,
        status=CapabilityStatus.AVAILABLE,  # bridge adapter works against graph.json
        permission=PermissionState.ALWAYS_ALLOWED,
        description="Reads code structure (calls, imports, dependencies) of this repo.",
        install_hint="Run `graphify extract .` at the repo root to generate graphify-out/graph.json.",
    ),
    Capability(
        name="bridge",
        type=CapabilityType.LOCAL_READ,
        status=CapabilityStatus.AVAILABLE,
        permission=PermissionState.ALWAYS_ALLOWED,
        description="Bridge layer connecting graphify (code structure) and Memory V2 (decisions).",
    ),
    Capability(
        # Local file reads from the IDE/extension host. Mirrors the frontend
        # MCP_OPTIONS picker key. UI opt-in is the consent gate; the registry
        # mirrors graphify's LOCAL_READ + ALWAYS_ALLOWED posture. Status
        # stays MISSING until the file-read client is wired through
        # mcp_router (Phase B).
        name="filesystem",
        type=CapabilityType.LOCAL_READ,
        status=CapabilityStatus.MISSING,
        permission=PermissionState.ALWAYS_ALLOWED,
        description="Reads files from the directory the user is working in (extension/IDE host only — no network).",
        install_hint="Wire @modelcontextprotocol/server-filesystem in src/mcp_router.py with the project root scoped path.",
    ),

    # External reads — ask-once, status reflects whether the host has
    # connected the implementation.
    Capability(
        name="web_search",
        type=CapabilityType.EXTERNAL_READ,
        # Tavily client is real (src/bridge/web_search.py) and TAVILY_API_KEY
        # drives provider selection. Falls through to DuckDuckGo HTML if no
        # key is set, so the capability is functionally AVAILABLE either way.
        status=CapabilityStatus.AVAILABLE,
        permission=PermissionState.ASK_ONCE_PENDING,
        description="Search the public web for current info beyond the 2025 training cutoff (Tavily primary, DuckDuckGo fallback).",
        install_hint="Set TAVILY_API_KEY for Tavily; without it DuckDuckGo HTML is used automatically.",
    ),
    Capability(
        name="github",
        type=CapabilityType.EXTERNAL_READ,
        status=CapabilityStatus.MISSING,
        permission=PermissionState.ASK_ONCE_PENDING,
        description="Read GitHub repo state — PRs, issues, recent commits, branch state.",
        install_hint="claude mcp add github",
    ),
    Capability(
        name="docs",
        type=CapabilityType.EXTERNAL_READ,
        status=CapabilityStatus.MISSING,
        permission=PermissionState.ASK_ONCE_PENDING,
        description="Look up specific library or API documentation.",
        install_hint="claude mcp add context7",
    ),
    Capability(
        name="browser",
        type=CapabilityType.EXTERNAL_READ,
        status=CapabilityStatus.MISSING,
        permission=PermissionState.ASK_ONCE_PENDING,
        description="Drive a real browser to verify UI behavior.",
        install_hint="claude mcp add playwright",
    ),

    # Writes — absent by design. No permission concept applies.
    Capability(
        name="file_write",
        type=CapabilityType.WRITE,
        status=CapabilityStatus.ABSENT_BY_DESIGN,
        permission=PermissionState.ABSENT_BY_DESIGN,
        description="Editing files. Constellax is a thinking partner — by design, the engine cannot write.",
    ),
    Capability(
        name="shell_exec",
        type=CapabilityType.WRITE,
        status=CapabilityStatus.ABSENT_BY_DESIGN,
        permission=PermissionState.ABSENT_BY_DESIGN,
        description="Running shell commands with side effects. Absent by design.",
    ),
    Capability(
        name="send_message",
        type=CapabilityType.WRITE,
        status=CapabilityStatus.ABSENT_BY_DESIGN,
        permission=PermissionState.ABSENT_BY_DESIGN,
        description="Sending emails, messages, or notifications. Absent by design.",
    ),
]


# ---------------------------------------------------------------------------
# The Registry
# ---------------------------------------------------------------------------

class CapabilityRegistry:
    """
    Central registry of what Constellax can and can't do right now.

    Consulted by the triage gate (populates mcps_needed correctly) and the
    route dispatcher (decides whether to fire an MCP or surface a
    conversational missing-capability response).
    """

    def __init__(self, capabilities: list[Capability] | None = None):
        self._capabilities: dict[str, Capability] = {}
        seed = capabilities if capabilities is not None else _DEFAULT_CAPABILITIES
        # Copy each Capability so mutations on this registry never leak into
        # the module-level _DEFAULT_CAPABILITIES (or any other registry that
        # was seeded from the same source list).
        for c in seed:
            self._capabilities[c.name] = copy.copy(c)
        self._usage_log: list[CapabilityUsage] = []

    # -----------------------------------------------------------------------
    # Read-side queries
    # -----------------------------------------------------------------------

    def get(self, name: str) -> Capability | None:
        return self._capabilities.get(name)

    def all(self) -> list[Capability]:
        return list(self._capabilities.values())

    def available(self) -> list[Capability]:
        return [c for c in self._capabilities.values()
                if c.status == CapabilityStatus.AVAILABLE]

    def missing(self) -> list[Capability]:
        return [c for c in self._capabilities.values()
                if c.status == CapabilityStatus.MISSING]

    def absent_by_design(self) -> list[Capability]:
        return [c for c in self._capabilities.values()
                if c.status == CapabilityStatus.ABSENT_BY_DESIGN]

    def permission(self, name: str) -> PermissionState | None:
        cap = self._capabilities.get(name)
        return cap.permission if cap else None

    def describe(self, name: str) -> str:
        cap = self._capabilities.get(name)
        return cap.description if cap else ""

    def is_available(self, name: str) -> bool:
        cap = self._capabilities.get(name)
        return cap is not None and cap.status == CapabilityStatus.AVAILABLE

    def is_absent_by_design(self, name: str) -> bool:
        cap = self._capabilities.get(name)
        return cap is not None and cap.status == CapabilityStatus.ABSENT_BY_DESIGN

    def needs_consent(self, name: str) -> bool:
        """True if the next firing requires asking the user first."""
        cap = self._capabilities.get(name)
        return cap is not None and cap.permission == PermissionState.ASK_ONCE_PENDING

    def can_fire(self, name: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason_if_not_allowed).

        Allowed only if status == AVAILABLE and permission allows it.
        Caller uses this before attempting any MCP call.
        """
        cap = self._capabilities.get(name)
        if cap is None:
            return False, f"capability '{name}' is not registered"
        if cap.status == CapabilityStatus.ABSENT_BY_DESIGN:
            return False, "absent by design — Constellax is a thinking partner, not an agent"
        if cap.status == CapabilityStatus.MISSING:
            return False, f"capability '{name}' is not connected"
        if cap.permission == PermissionState.DENIED:
            return False, f"user has denied '{name}'"
        if cap.permission == PermissionState.ASK_ONCE_PENDING:
            return False, f"first use of '{name}' — needs user consent"
        return True, ""

    # -----------------------------------------------------------------------
    # State mutations (called by the MCP router, Step 8)
    # All mutations refuse to alter ABSENT_BY_DESIGN capabilities — that's
    # the structural safety property.
    # -----------------------------------------------------------------------

    def mark_available(self, name: str) -> None:
        cap = self._capabilities.get(name)
        if cap is None or cap.status == CapabilityStatus.ABSENT_BY_DESIGN:
            return
        cap.status = CapabilityStatus.AVAILABLE

    def mark_missing(self, name: str) -> None:
        cap = self._capabilities.get(name)
        if cap is None or cap.status == CapabilityStatus.ABSENT_BY_DESIGN:
            return
        cap.status = CapabilityStatus.MISSING

    def grant_permission(self, name: str) -> None:
        """User said yes to an ask_once prompt — graduate to granted state."""
        cap = self._capabilities.get(name)
        if cap is None or cap.permission == PermissionState.ABSENT_BY_DESIGN:
            return
        cap.permission = PermissionState.ASK_ONCE_GRANTED

    def deny_permission(self, name: str) -> None:
        cap = self._capabilities.get(name)
        if cap is None or cap.permission == PermissionState.ABSENT_BY_DESIGN:
            return
        cap.permission = PermissionState.DENIED

    def reset_permission(self, name: str) -> None:
        """
        Roll back permission to its default for the capability's type.
        Local reads → ALWAYS_ALLOWED. External reads → ASK_ONCE_PENDING.
        """
        cap = self._capabilities.get(name)
        if cap is None or cap.permission == PermissionState.ABSENT_BY_DESIGN:
            return
        if cap.type == CapabilityType.LOCAL_READ:
            cap.permission = PermissionState.ALWAYS_ALLOWED
        else:
            cap.permission = PermissionState.ASK_ONCE_PENDING

    # -----------------------------------------------------------------------
    # Usage log — powers "checking X for Y..." notifications
    # -----------------------------------------------------------------------

    def log_usage(
        self,
        name: str,
        purpose: str,
        success: bool = True,
        error: str | None = None,
    ) -> CapabilityUsage:
        usage = CapabilityUsage(
            capability=name,
            purpose=purpose,
            timestamp=time.time(),
            success=success,
            error=error,
        )
        self._usage_log.append(usage)
        return usage

    def usage_history(self, name: str | None = None) -> list[CapabilityUsage]:
        if name is None:
            return list(self._usage_log)
        return [u for u in self._usage_log if u.capability == name]

    def clear_usage_log(self) -> None:
        """Useful for per-request isolation if the registry is shared across requests."""
        self._usage_log.clear()

    # -----------------------------------------------------------------------
    # Conversational response builder
    # Used when the triage gate wants an MCP that isn't available.
    # -----------------------------------------------------------------------

    def build_missing_capability_response(
        self,
        name: str,
        why_needed: str,
        current_confidence: float = 0.5,
        confidence_if_connected: float = 0.85,
        fallback_quality: str = "medium",
        fallback_caveat: str = "",
    ) -> dict:
        """
        Build the structured user-facing JSON for a missing capability.

        Combines registry knowledge (description, install hint, status) with
        reasoning-engine context (why needed, confidence numbers) into the
        locked conversational shape.

        Three shapes are emitted:
        - unknown    — capability isn't registered at all (programming error)
        - absent     — capability is ABSENT_BY_DESIGN; no install option offered
        - missing    — capability could be connected; user_options includes install hint
        """
        cap = self._capabilities.get(name)
        if cap is None:
            return {
                "capability": name,
                "error": "unknown_capability",
                "why_needed": why_needed,
            }

        if cap.status == CapabilityStatus.ABSENT_BY_DESIGN:
            return {
                "capability": name,
                "absent_by_design": True,
                "why_needed": why_needed,
                "explanation": cap.description,
                "user_options": [],  # structural, nothing to offer
            }

        user_options: list[dict] = []
        if cap.install_hint:
            user_options.append({
                "id": "connect",
                "label": f"Connect {name}",
                "instruction": cap.install_hint,
            })
        user_options.append({"id": "fallback", "label": "Continue without it"})
        user_options.append({"id": "skip", "label": "Skip this part"})

        return {
            "capability": name,
            "description": cap.description,
            "why_needed": why_needed,
            "current_confidence": current_confidence,
            "confidence_if_connected": confidence_if_connected,
            "fallback_quality": fallback_quality,
            "fallback_caveat": fallback_caveat,
            "user_options": user_options,
        }
