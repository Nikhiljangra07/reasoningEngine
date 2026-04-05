"""
Semantic Similarity — Cross-Domain Variable Matching.

Physics says "causal_force_career" and Psychology says "emotional_pressure_career"
— they're describing the same underlying variable with different names.

This module provides semantic matching that goes beyond exact name matching.
Three layers of matching, each more sophisticated:

Layer 1: Exact name match (fastest, fallback)
Layer 2: TF-IDF cosine similarity on descriptions (no external dependency)
Layer 3: LLM-based semantic comparison (optional, uses one Sonnet call for a batch)

The Valence system in Chemistry uses this to find shared electrons
between domain outputs.

ISOLATION: Imports only from stdlib + src.core.types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import Counter

from src.core.types import Variable


# Thresholds (from spec):
# 0.7+ = shared electron confirmed
# 0.4-0.7 = possible shared electron (flag for review)
# below 0.4 = no bond
CONFIRMED_THRESHOLD = 0.7
POSSIBLE_THRESHOLD = 0.4


@dataclass
class SemanticMatch:
    """A semantic match between two variables from different domains."""
    variable_a: str         # name from domain A
    variable_b: str         # name from domain B
    similarity: float       # 0.0 to 1.0
    match_type: str         # "exact", "semantic", "description_overlap"
    bond_status: str        # "confirmed", "possible", "none"


# ---------------------------------------------------------------------------
# Stopwords for text processing
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "with", "on", "at", "from", "by", "about", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "and", "but",
    "or", "not", "no", "so", "if", "then", "than", "that", "this", "it",
    "its", "they", "their", "them", "we", "our", "you", "your", "he",
    "she", "his", "her", "which", "what", "where", "when", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "only", "very", "just", "also", "any", "been", "being",
    "user", "variable", "magnitude", "direction", "confidence",
    "detected", "found", "analysis", "score",
}


# ---------------------------------------------------------------------------
# Core matching function
# ---------------------------------------------------------------------------

def find_semantic_matches(
    vars_a: list[Variable],
    vars_b: list[Variable],
) -> list[SemanticMatch]:
    """
    Find semantically matching variables between two sets.

    Returns all matches above the POSSIBLE_THRESHOLD (0.4).
    Each variable can only match once (best match wins).
    """
    matches: list[SemanticMatch] = []
    matched_a: set[str] = set()
    matched_b: set[str] = set()

    # Layer 1: Exact name matches (fastest)
    names_a = {v.name for v in vars_a}
    names_b = {v.name for v in vars_b}
    for name in names_a & names_b:
        matches.append(SemanticMatch(
            variable_a=name,
            variable_b=name,
            similarity=1.0,
            match_type="exact",
            bond_status="confirmed",
        ))
        matched_a.add(name)
        matched_b.add(name)

    # Layer 2: TF-IDF cosine similarity on descriptions
    # Build TF-IDF vectors for all unmatched variables
    remaining_a = [v for v in vars_a if v.name not in matched_a]
    remaining_b = [v for v in vars_b if v.name not in matched_b]

    if remaining_a and remaining_b:
        # Build corpus for IDF calculation
        all_docs = []
        for v in remaining_a + remaining_b:
            all_docs.append(_tokenize(v))

        idf = _compute_idf(all_docs)

        # Compute TF-IDF vectors
        vectors_a = [_tfidf_vector(v, idf) for v in remaining_a]
        vectors_b = [_tfidf_vector(v, idf) for v in remaining_b]

        # Find best matches via cosine similarity
        # Score all pairs, take highest first (greedy matching)
        scored_pairs: list[tuple[float, int, int]] = []
        for i, vec_a in enumerate(vectors_a):
            for j, vec_b in enumerate(vectors_b):
                sim = _cosine_similarity(vec_a, vec_b)
                if sim >= POSSIBLE_THRESHOLD:
                    scored_pairs.append((sim, i, j))

        # Sort by similarity descending
        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        # Greedy 1-to-1 matching
        used_a: set[int] = set()
        used_b: set[int] = set()

        for sim, i, j in scored_pairs:
            if i in used_a or j in used_b:
                continue

            va = remaining_a[i]
            vb = remaining_b[j]

            bond_status = (
                "confirmed" if sim >= CONFIRMED_THRESHOLD
                else "possible"
            )

            matches.append(SemanticMatch(
                variable_a=va.name,
                variable_b=vb.name,
                similarity=sim,
                match_type="semantic",
                bond_status=bond_status,
            ))

            used_a.add(i)
            used_b.add(j)
            matched_a.add(va.name)
            matched_b.add(vb.name)

    return matches


# ---------------------------------------------------------------------------
# TF-IDF Implementation (no external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(var: Variable) -> list[str]:
    """Tokenize a variable into meaningful words from name + description + evidence."""
    text = var.name.replace("_", " ").lower()
    text += " " + var.description.lower()
    if var.evidence:
        text += " " + " ".join(var.evidence).lower()

    tokens = []
    for word in text.split():
        cleaned = word.strip(".,;:!?()[]{}\"'`-—/")
        if len(cleaned) > 2 and cleaned not in _STOPWORDS:
            tokens.append(cleaned)

    return tokens


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency for all terms."""
    doc_count = len(documents)
    if doc_count == 0:
        return {}

    # Count how many documents each term appears in
    doc_freq: Counter[str] = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] += 1

    # IDF = log(N / df) + 1 (smoothed)
    idf: dict[str, float] = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(doc_count / df) + 1.0

    return idf


def _tfidf_vector(var: Variable, idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF vector for a single variable."""
    tokens = _tokenize(var)
    if not tokens:
        return {}

    # Term frequency
    tf: Counter[str] = Counter(tokens)
    total = len(tokens)

    # TF-IDF
    vector: dict[str, float] = {}
    for term, count in tf.items():
        tf_score = count / total
        idf_score = idf.get(term, 1.0)
        vector[term] = tf_score * idf_score

    return vector


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0

    # Find common terms
    common_terms = set(vec_a.keys()) & set(vec_b.keys())

    if not common_terms:
        return 0.0

    # Dot product
    dot_product = sum(vec_a[t] * vec_b[t] for t in common_terms)

    # Magnitudes
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Utility: convert matches to shared electron names for Valence
# ---------------------------------------------------------------------------

def matches_to_shared_electrons(matches: list[SemanticMatch]) -> list[str]:
    """Convert semantic matches to shared electron labels for Valence bonding."""
    shared = []
    for m in matches:
        if m.bond_status == "confirmed":
            if m.match_type == "exact":
                shared.append(m.variable_a)
            else:
                shared.append(f"{m.variable_a}≈{m.variable_b}")
        elif m.bond_status == "possible":
            shared.append(f"{m.variable_a}~{m.variable_b}")
    return shared
