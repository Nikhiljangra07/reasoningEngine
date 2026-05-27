"""
MarkdownDecisionLogExtractor — parse a CLAUDE.md-style decision log into
typed DecisionAnchor records.

WHY THIS EXISTS
===============
Users write decisions in a markdown file (their own CLAUDE.md for their
project). The system reads that file, extracts structured records, and
persists them as DecisionAnchor nodes in Neo4j. This is the
"capture layer" of the foundational memory pipeline — the user's own
decisions become first-class queryable graph nodes without any LLM call.

DETERMINISTIC, NO LLM
=====================
This parser is regex-only. No LLM, no network, no API key. The format
is strict enough that a Python parser handles it cleanly; that gives us:
  - Zero per-parse cost
  - Reproducible: same markdown → same DecisionAnchors every time
  - Fast: a typical CLAUDE.md parses in milliseconds
  - Stable IDs: the ID is a SHA256 hash of the lowercased title, so
    re-parsing the same file produces the same IDs → MERGE in Neo4j
    is idempotent (no duplicates).

MARKDOWN CONTRACT
=================
The expected format under any `## DECISIONS` (or `## Decision Log`) section:

    ## DECISIONS

    ### Migrate to Neo4j Aura
    - status: settled
    - rationale: GDS + native vector indexes consolidate two services
    - date: 2026-05-27
    - tags: db, migration, infra
    - evidence:
        - 18 parity checks pass
        - Aura Free fits beta workload

    ### Build Map Room visualizer
    - status: settled
    - rationale: User shouldn't see raw nodes/searches
    - date: 2026-05-25
    - tags: ui, visualization

ALLOWED TOP-LEVEL FIELDS
========================
The parser recognizes these `- key: value` fields under a `### <title>`:
  - status:    OPEN | SETTLED | DRIFTED | SUPERSEDED | REJECTED (default OPEN)
  - rationale: free-form text (single line)
  - tags:      comma-separated tag list
  - date:      free-form date string (stored as-is; no parsing)
  - evidence:  followed by indented sub-bullets (one bullet per evidence item)
  - id:        explicit ID override; otherwise auto-generated from title

Unknown fields land in `DecisionAnchor.meta` via a future extension point;
for v1 we drop them silently with a warning log.

CODE BLOCKS ARE IGNORED
=======================
Fenced code blocks (```...```) are stripped before parsing so that a
literal `### whatever` inside a code example isn't picked up as a heading.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from src.bridge.types import DecisionAnchor

log = logging.getLogger("constellax.decision_log_parser")


# Allowed values for DecisionAnchor.status. Anything else lowers to "OPEN".
_ALLOWED_STATUSES = {"OPEN", "SETTLED", "DRIFTED", "SUPERSEDED", "REJECTED"}


# ─── Regex toolbox ───────────────────────────────────────────────────

# Matches a top-level `## ...` heading. Used to detect section boundaries.
_TOP_SECTION_RE = re.compile(r"^##\s+(?!#)(.+?)\s*$", re.MULTILINE)

# Matches `## DECISIONS` or `## Decision Log` (case-insensitive, optional
# trailing punctuation/text). Used to detect the start of decision sections.
_DECISIONS_SECTION_RE = re.compile(
    r"^##\s+(decisions?|decision\s+log)\b", re.IGNORECASE | re.MULTILINE,
)

# Matches `### Title` — the boundary between individual decisions.
_DECISION_HEADING_RE = re.compile(r"^###\s+(?!#)(.+?)\s*$")

# Matches `- key: value` at any indent. We disambiguate top-level bullets
# (no leading spaces) from sub-bullets via the captured indent group.
_BULLET_RE = re.compile(r"^(\s*)-\s*([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$")

# Matches a plain sub-bullet (`-  text`) inside a key like evidence.
_SUB_BULLET_RE = re.compile(r"^(\s+)-\s+(.+?)\s*$")

# Fenced code blocks (`` ``` `` open and close on their own lines).
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


# ─── Public API ──────────────────────────────────────────────────────

class MarkdownDecisionLogExtractor:
    """Parses CLAUDE.md-style decision logs into DecisionAnchor lists.

    Stateless — safe to share one instance across the process and across
    threads. Each parse() call is independent."""

    def parse(self, markdown: str, *, project_id: str | None = None) -> list[DecisionAnchor]:
        """Parse a markdown string. Returns DecisionAnchors in order of
        appearance. Empty list if no DECISIONS section is present."""
        if not markdown:
            return []
        # Strip fenced code blocks first so literal `### foo` inside code
        # examples isn't mistaken for a real heading.
        cleaned = _FENCE_RE.sub("", markdown)
        # Iterate the decision sections and accumulate.
        out: list[DecisionAnchor] = []
        for section in self._iter_decision_sections(cleaned):
            out.extend(self._parse_section(section, project_id=project_id))
        return out

    def parse_file(self, path: str | Path, *, project_id: str | None = None) -> list[DecisionAnchor]:
        """Read a file from disk and parse its content. Raises FileNotFoundError
        if the file doesn't exist; UnicodeDecodeError on non-UTF8 content."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        return self.parse(text, project_id=project_id)

    # ─── Internal: section + decision extraction ─────────────────────

    def _iter_decision_sections(self, text: str):
        """Yield the body of each `## DECISIONS` section (everything from
        the heading line through the next `## <something>` or EOF). The
        heading itself is included so headings inside (like `### Title`)
        keep their line offsets — simplifies downstream regex matching."""
        # Find positions of all top-level `##` headings.
        section_starts = list(_TOP_SECTION_RE.finditer(text))
        for i, m in enumerate(section_starts):
            heading = m.group(0)
            if not _DECISIONS_SECTION_RE.match(heading):
                continue
            # Body runs from after this heading line to before the next
            # top-level heading (or to EOF).
            body_start = m.end()
            body_end = (
                section_starts[i + 1].start()
                if i + 1 < len(section_starts)
                else len(text)
            )
            yield text[body_start:body_end]

    def _parse_section(self, section_body: str, *, project_id: str | None) -> list[DecisionAnchor]:
        """Walk the section body line-by-line, building up DecisionAnchors
        as we see `### <title>` boundaries."""
        decisions: list[DecisionAnchor] = []
        current: dict | None = None     # accumulator dict; converted to DecisionAnchor on flush
        in_evidence = False              # True while we're inside an `evidence:` block

        for line in section_body.splitlines():
            heading = _DECISION_HEADING_RE.match(line)
            if heading:
                # Boundary: flush previous decision (if any), start a new one.
                if current is not None:
                    decisions.append(self._to_anchor(current, project_id))
                current = {
                    "title": heading.group(1).strip(),
                    "status": "OPEN",
                    "rationale": "",
                    "tags": [],
                    "date": "",
                    "evidence": [],
                    "explicit_id": None,
                }
                in_evidence = False
                continue

            if current is None:
                # Lines outside any ### block — skip.
                continue

            bullet = _BULLET_RE.match(line)
            if bullet and bullet.group(1) == "":
                # Top-level field bullet (no indent before the `-`).
                key = bullet.group(2).lower()
                value = bullet.group(3).strip()
                if key == "evidence":
                    # Evidence is multi-line; the value on this line is an
                    # inline first item (rare but valid), and subsequent
                    # indented `- ...` lines extend the list.
                    if value:
                        current["evidence"].append(value)
                    in_evidence = True
                elif key == "tags":
                    current["tags"] = [
                        t.strip() for t in value.split(",") if t.strip()
                    ]
                    in_evidence = False
                elif key == "id":
                    current["explicit_id"] = value
                    in_evidence = False
                elif key in {"status", "rationale", "date"}:
                    current[key] = value
                    in_evidence = False
                else:
                    log.debug("decision-log: unrecognized field %r — dropped", key)
                    in_evidence = False
                continue

            # Indented sub-bullet, only relevant inside an evidence block.
            sub = _SUB_BULLET_RE.match(line)
            if sub and in_evidence:
                current["evidence"].append(sub.group(2).strip())
                continue

            # Anything else (blank line, prose) closes the current evidence list.
            if line.strip() == "":
                in_evidence = False

        # Flush the trailing decision if any.
        if current is not None:
            decisions.append(self._to_anchor(current, project_id))
        return decisions

    def _to_anchor(self, acc: dict, project_id: str | None) -> DecisionAnchor:
        """Convert the accumulator dict into a typed DecisionAnchor.

        ID handling: if the markdown set an explicit `id:` field we use
        that; otherwise we derive a stable hash from the lowercased title
        so re-parses produce the same ID (MERGE-friendly)."""
        title = acc.get("title", "").strip() or "(untitled)"
        explicit_id = acc.get("explicit_id")
        anchor_id = explicit_id or _stable_id_from_title(title, project_id)
        status = (acc.get("status") or "OPEN").strip().upper()
        if status not in _ALLOWED_STATUSES:
            log.debug("decision-log: invalid status %r → defaulting to OPEN", status)
            status = "OPEN"
        # DecisionAnchor expects float created_at. We parse the date later
        # if needed; for now we store it in title-prefix or just default to
        # 0.0 to avoid lying about a timestamp we can't verify.
        # Date string is kept on the dict but DecisionAnchor doesn't have a
        # `date_str` field — we drop it. If users want it preserved, we'd
        # add a `meta` dict to DecisionAnchor and stash it there.
        return DecisionAnchor(
            id=anchor_id,
            title=title,
            rationale=acc.get("rationale", "").strip(),
            evidence=list(acc.get("evidence") or []),
            status=status,
            created_at=0.0,                      # unknown — not in the markdown
            code_refs=[],                        # this parser doesn't extract code refs in v1
            tags=list(acc.get("tags") or []),
        )


# ─── Stable ID derivation ────────────────────────────────────────────

def _stable_id_from_title(title: str, project_id: str | None) -> str:
    """SHA-256 of lowercased title (scoped by project_id when present),
    truncated to 12 hex chars and prefixed `D-`. Deterministic across
    re-parses; two different titles can collide only at 2^48 odds."""
    key = (project_id or "") + "::" + title.strip().lower()
    return "D-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
