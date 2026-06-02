"""
Sovereignty — the three anti-laws and the Map-Not-March counter.

The user is sovereign over the move. The reasoning core surfaces
paths, names consequences, projects failures — and stops there. This
module encodes the constraints that prevent the engine from drifting
into execution, argument, or emotional padding.

Two pieces.

ANTI_LAWS / AntiLaw
    Three rules with human-readable text, machine-checkable
    predicates, and remediation hints used by voice.lint to decide
    when to regenerate.

MapNotMarchCounter
    Session-scoped state, keyed by (session_id, position_hash). After
    two strikes on the same essential position, the next response
    must shift from argument to cartography (Path A consequence /
    Path B consequence / Path C if it exists).
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Anti-laws
# ---------------------------------------------------------------------------

class AntiLawKind(str, Enum):
    NO_EXECUTION = "no_execution"
    NO_ARGUMENT  = "no_argument"
    NO_PADDING   = "no_padding"


@dataclass(frozen=True)
class AntiLaw:
    """One of the three anti-laws.

    `kind` is the enum tag. `text` is the doctrine clause used in the
    runtime header. `forbidden` is a tuple of regex patterns that
    indicate the rule has been violated in an output. `remediation`
    is a short clause voice.lint uses when asking the model to
    regenerate."""

    kind:        AntiLawKind
    text:        str
    forbidden:   tuple[str, ...]
    remediation: str


# NO EXECUTION — the engine surfaces paths, the user moves. Forbidden
# patterns are first-person commitments to act on the user's behalf.
_NO_EXECUTION = AntiLaw(
    kind=AntiLawKind.NO_EXECUTION,
    text=(
        "You do not edit files, run commands, patch code, or act on "
        "the user's systems. You read, reason, articulate. The user "
        "makes the move."
    ),
    forbidden=(
        r"\bi['’]ll (?:edit|update|modify|patch|commit|push|deploy|run)\b",
        r"\bi (?:will|am going to) (?:edit|update|modify|patch|commit|push|deploy|run)\b",
        r"\blet me (?:edit|update|modify|patch|commit|push|deploy|run)\b",
        r"\bi['’]ve (?:edited|updated|modified|patched|committed|pushed|deployed)\b",
    ),
    remediation=(
        "Rewrite as a surfaced path the user can take, not an action "
        "you will execute for them. Name the move, name the cost, "
        "leave the trigger to the user."
    ),
)

# NO ARGUMENT — when the user pushes back, state the position once,
# then shift to cartography on a second pushback. Forbidden patterns
# are the verbal tics that signal the model is digging in.
_NO_ARGUMENT = AntiLaw(
    kind=AntiLawKind.NO_ARGUMENT,
    text=(
        "When the user pushes back, state the position once. If they "
        "push back again, switch to cartography: Path A consequence, "
        "Path B consequence, Path C if one exists. The user decides."
    ),
    forbidden=(
        # These fire only when the Map-Not-March counter says we're on
        # strike-2 or later — voice.lint composes the rule with the
        # counter before flagging.
        r"\bas i (?:said|mentioned|noted) (?:earlier|before|already)\b",
        r"\bto reiterate\b",
        r"\bi (?:still|must|have to) (?:disagree|insist|push back)\b",
        r"\byou['’]re missing the point\b",
    ),
    remediation=(
        "The user has already restated their position. Stop arguing. "
        "Lay out the paths: each option, its concrete consequence, "
        "its failure mode. End on the user's choice."
    ),
)

# NO PADDING — no openers, no therapy, no emotional commentary unless
# directly relevant. The 'unless directly relevant' clause matters —
# this is NOT 'no emotional intelligence', it is no padding.
_NO_PADDING = AntiLaw(
    kind=AntiLawKind.NO_PADDING,
    text=(
        "No openers. No therapy. No emotional commentary unless the "
        "user's emotional state is directly relevant to the decision. "
        "When it is relevant, name it once and move."
    ),
    forbidden=(
        # Opener fluff — handled primarily by voice.strip; lint catches
        # the cases strip misses.
        r"^\s*(?:great|that['’]s a great|excellent|wonderful|fantastic) (?:question|point)\b",
        r"^\s*(?:i['’]m happy|i['’]d be happy|happy) to (?:help|assist)\b",
        r"^\s*(?:certainly|absolutely|of course)[,!.]",
        r"^\s*let me (?:think about this|unpack this|break this down) for you\b",
        # Therapy openers — the cushioning prefix that softens the
        # actual analysis.
        r"\bi hear you[,.]",
        r"\bthat sounds (?:really )?(?:hard|tough|difficult)\b",
        r"\bi['’]m sorry you['’]re (?:going through|dealing with|feeling)\b",
        # Permission-asking that pads without serving.
        r"\bwould it be helpful if i\b",
        r"\bdoes that make sense\?\s*$",
    ),
    remediation=(
        "Strip the opener. State the analysis from sentence one. If "
        "you named an emotion, retain the naming ONLY when the user's "
        "emotional state is directly relevant to the decision; "
        "otherwise drop it."
    ),
)


ANTI_LAWS: tuple[AntiLaw, ...] = (_NO_EXECUTION, _NO_ARGUMENT, _NO_PADDING)


def get_anti_law(kind: AntiLawKind) -> AntiLaw:
    """Lookup helper. Raises KeyError for unknown kinds."""
    for law in ANTI_LAWS:
        if law.kind == kind:
            return law
    raise KeyError(f"Unknown AntiLawKind: {kind}")


# ---------------------------------------------------------------------------
# Map-Not-March counter
# ---------------------------------------------------------------------------

# After this many same-position restatements, the engine forces the
# response into cartography mode. Two means: the user has stated the
# position once, pushed back once, and is about to push back again —
# we don't get a third round of argument.
MAP_NOT_MARCH_THRESHOLD: int = 2


def position_hash(text: str) -> str:
    """Normalize a position statement and return a stable hash.

    Lowercase, strip punctuation, collapse whitespace, drop common
    filler words. Two pushbacks that are essentially the same
    position phrased differently hash to the same key, so the counter
    catches them. Returns a 12-char hex prefix — plenty unique per
    session, cheap to compute, easy to log."""

    if not text:
        return "0" * 12

    cleaned = text.lower()
    # Drop punctuation that doesn't change meaning.
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    # Drop common filler so 'i really think x' and 'i think x' match.
    fillers = (
        " really ", " just ", " actually ", " honestly ", " literally ",
        " kinda ", " sort of ", " kind of ", " maybe ", " probably ",
        " i mean ", " you know ", " like ",
    )
    padded = f" {cleaned} "
    for f in fillers:
        padded = padded.replace(f, " ")
    cleaned = re.sub(r"\s+", " ", padded).strip()

    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
    return digest[:12]


@dataclass
class _CounterEntry:
    count:    int
    last_seen_text: str


class MapNotMarchCounter:
    """Session-scoped repeat-position counter.

    The counter is keyed by (session_id, position_hash). The
    synthesizer calls `note(session_id, text)` each time the user
    restates a position; calls `should_force_map(session_id, text)`
    before composing the next response to decide whether to switch to
    cartography mode.

    Thread-safe; the counter may be queried from worker threads while
    a separate writer updates it.

    State is in-process. The engine recreates the counter on restart;
    a fresh server start resets all sessions to strike-zero. That's
    acceptable: by the time a session crosses a restart boundary, the
    user has had time to think, and a fresh argument round is fine."""

    def __init__(self) -> None:
        self._state: dict[tuple[str, str], _CounterEntry] = {}
        self._lock = threading.Lock()

    def note(self, session_id: str, text: str) -> int:
        """Record a position restatement. Returns the new count for
        that (session, position) key."""
        h = position_hash(text)
        with self._lock:
            key = (session_id, h)
            entry = self._state.get(key)
            if entry is None:
                entry = _CounterEntry(count=1, last_seen_text=text)
            else:
                entry.count += 1
                entry.last_seen_text = text
            self._state[key] = entry
            return entry.count

    def current(self, session_id: str, text: str) -> int:
        """Read the current count without incrementing. Returns 0
        when the position has not been seen yet."""
        h = position_hash(text)
        with self._lock:
            entry = self._state.get((session_id, h))
            return entry.count if entry is not None else 0

    def should_force_map(self, session_id: str, text: str) -> bool:
        """True when this position has been restated enough times
        that the next response must switch to cartography mode."""
        return self.current(session_id, text) >= MAP_NOT_MARCH_THRESHOLD

    def reset(self, session_id: str) -> None:
        """Drop all counters for a session. Called on session end."""
        with self._lock:
            keys_to_drop = [k for k in self._state if k[0] == session_id]
            for k in keys_to_drop:
                del self._state[k]

    def clear(self) -> None:
        """Drop all counters across all sessions. Test-only helper."""
        with self._lock:
            self._state.clear()
