"""
Combination Cache — Institutional Memory.

Filtered-out variables are NOT garbage. They are stored with:
- The combination itself
- The problem context it came from
- The result it produced

On future problems, Chemistry's Self-Assembly queries this cache
by semantic similarity before activating concepts. Cache hits
become pre-computed priors for the Bayesian backbone, saving
compute on similar problems.

The engine gets smarter over time without recomputing.

ISOLATION: Imports ONLY from src.core.types.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.core.types import Variable, Direction, FrameworkID


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached combination."""
    variable_name: str
    variable_description: str
    variable_magnitude: float
    variable_direction: str
    variable_confidence: float
    source_framework: str
    problem_context: str
    result_produced: str
    iteration_cached: int
    connection_score: int
    timestamp: str = ""


@dataclass
class CacheStore:
    """The full cache of filtered combinations."""
    entries: list[CacheEntry] = field(default_factory=list)
    _file_path: str = ""


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------

def create_cache(file_path: str = "") -> CacheStore:
    """Create or load a cache store."""
    store = CacheStore(_file_path=file_path)

    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                for entry_data in data.get("entries", []):
                    store.entries.append(CacheEntry(**entry_data))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # corrupted cache — start fresh

    return store


def cache_variable(
    store: CacheStore,
    variable: Variable,
    problem_context: str,
    result_produced: str,
    iteration: int,
    connection_score: int,
) -> None:
    """Add a filtered-out variable to the cache."""
    entry = CacheEntry(
        variable_name=variable.name,
        variable_description=variable.description,
        variable_magnitude=variable.magnitude,
        variable_direction=variable.direction.value,
        variable_confidence=variable.confidence,
        source_framework=variable.source_framework.value,
        problem_context=problem_context,
        result_produced=result_produced,
        iteration_cached=iteration,
        connection_score=connection_score,
    )
    store.entries.append(entry)


def query_cache(
    store: CacheStore,
    problem_statement: str,
    top_n: int = 10,
) -> list[CacheEntry]:
    """
    Query the cache by semantic similarity to a new problem.

    Returns the top N most relevant cached combinations.
    Uses keyword overlap as a lightweight similarity measure.
    (Future: replace with embedding-based cosine similarity.)
    """
    if not store.entries:
        return []

    problem_words = set(problem_statement.lower().split())

    scored = []
    for entry in store.entries:
        # Score by keyword overlap between problem contexts
        context_words = set(entry.problem_context.lower().split())
        desc_words = set(entry.variable_description.lower().split())
        all_entry_words = context_words | desc_words

        overlap = len(problem_words & all_entry_words)
        total = len(problem_words | all_entry_words) or 1
        similarity = overlap / total

        # Boost by magnitude and confidence (higher = more useful prior)
        boost = entry.variable_magnitude * 0.3 + entry.variable_confidence * 0.2
        score = similarity + boost

        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_n]]


def cache_to_priors(entries: list[CacheEntry]) -> list[Variable]:
    """
    Convert cache entries into Variables that can serve as
    pre-computed priors for the Bayesian backbone.
    """
    priors = []
    for entry in entries:
        # Discount confidence — these are cached, not fresh
        discounted_confidence = entry.variable_confidence * 0.6

        try:
            direction = Direction(entry.variable_direction)
        except ValueError:
            direction = Direction.NEUTRAL

        try:
            framework = FrameworkID(entry.source_framework)
        except ValueError:
            framework = FrameworkID.BAYESIAN

        priors.append(Variable(
            name=f"cached_{entry.variable_name}",
            description=(
                f"[CACHED PRIOR] {entry.variable_description} "
                f"(from similar problem: {entry.problem_context[:60]})"
            ),
            magnitude=entry.variable_magnitude * 0.7,  # discount magnitude too
            direction=direction,
            confidence=discounted_confidence,
            source_framework=framework,
            is_hidden=False,
            is_user_stated=False,
            evidence=[
                f"Cached from: {entry.problem_context[:80]}",
                f"Original result: {entry.result_produced[:80]}",
                f"Original confidence: {entry.variable_confidence:.2f}",
                f"Discounted to: {discounted_confidence:.2f}",
            ],
        ))

    return priors


def save_cache(store: CacheStore) -> None:
    """Persist cache to disk."""
    if not store._file_path:
        return

    data = {
        "entries": [
            {
                "variable_name": e.variable_name,
                "variable_description": e.variable_description,
                "variable_magnitude": e.variable_magnitude,
                "variable_direction": e.variable_direction,
                "variable_confidence": e.variable_confidence,
                "source_framework": e.source_framework,
                "problem_context": e.problem_context,
                "result_produced": e.result_produced,
                "iteration_cached": e.iteration_cached,
                "connection_score": e.connection_score,
                "timestamp": e.timestamp,
            }
            for e in store.entries
        ]
    }

    os.makedirs(os.path.dirname(store._file_path) or ".", exist_ok=True)
    with open(store._file_path, "w") as f:
        json.dump(data, f, indent=2)
