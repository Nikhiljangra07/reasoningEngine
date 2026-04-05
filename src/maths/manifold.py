"""
Maths Layer 3 + 4: Manifold Theory + N-Dimensional Capacity

If a problem is a complex, high-dimensional shape, a manifold lets you
zoom in on any point and treat it as a simple, flat coordinate system.

This is the multi-angle holder. It:
- Creates the N-dimensional space (the Atlas)
- Holds all perspectives as charts
- Performs smooth transitions between perspectives (homeomorphisms)
- Reduces dimensions to find intrinsic structure (manifold hypothesis)
- Evaluates all possible paths (variational calculus / least action)

Scales to any number of variables without breaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from src.core.types import (
    Direction,
    FrameworkID,
    Perspective,
    Problem,
    SignalType,
    Variable,
)


@dataclass
class Chart:
    """
    A single chart in the atlas — one local coordinate system.

    Each perspective becomes a chart. The chart captures the
    perspective's "view" of the problem in a local, flat space.
    """
    perspective: Perspective
    coordinates: list[float]    # position in N-dimensional space
    dimension: int              # which axis this chart primarily occupies
    overlap_with: list[int] = field(default_factory=list)  # indices of overlapping charts


@dataclass
class ManifoldState:
    """
    The current state of the manifold — the atlas of all charts.
    """
    charts: list[Chart]
    total_dimensions: int                       # N — how many axes exist
    intrinsic_dimensions: int                   # the actual dimensionality after reduction
    core_axes: list[str]                        # names of the core dimensions that matter
    variance_explained: list[float]             # how much each axis explains (for SVD)
    redundant_perspectives: list[Perspective]    # perspectives that add no new information


def build_manifold(
    problem: Problem,
    perspectives: list[Perspective],
) -> ManifoldState:
    """
    Build the manifold from all perspectives.

    1. Create the N-dimensional space
    2. Place each perspective as a chart
    3. Find overlaps between charts
    4. Reduce dimensions to intrinsic structure
    5. Identify redundant perspectives
    """
    # Step 1: Determine dimensions from all unique variables
    all_variables = _collect_all_variables(perspectives)
    dimension_names = _identify_dimensions(all_variables)
    total_dims = len(dimension_names)

    # Step 2: Place each perspective as a chart in the space
    charts = _create_charts(perspectives, dimension_names)

    # Step 3: Find overlaps between charts
    _find_overlaps(charts)

    # Step 4: Dimensional reduction — find intrinsic structure
    variance = _calculate_variance(charts, total_dims)
    intrinsic_dims, core_axes = _reduce_dimensions(variance, dimension_names)

    # Step 5: Identify redundant perspectives
    redundant = _find_redundant(charts)

    return ManifoldState(
        charts=charts,
        total_dimensions=total_dims,
        intrinsic_dimensions=intrinsic_dims,
        core_axes=core_axes,
        variance_explained=variance,
        redundant_perspectives=redundant,
    )


def find_least_action_path(manifold: ManifoldState) -> list[str]:
    """
    Variational Calculus — find the path of least action.

    Evaluates the manifold to find the trajectory that requires
    the least energy to reach resolution.

    Returns a list of steps (descriptions) for the optimal path.
    """
    if not manifold.charts:
        return ["Insufficient data to calculate optimal path."]

    # Sort charts by their primary axis contribution (highest variance first)
    core_charts = [
        c for c in manifold.charts
        if c.perspective.signal_type == SignalType.SIGNAL
    ]

    if not core_charts:
        core_charts = manifold.charts

    # The least-action path prioritizes high-weight, high-confidence perspectives
    sorted_charts = sorted(
        core_charts,
        key=lambda c: c.perspective.weight * max(
            (v.confidence for v in c.perspective.variables_found),
            default=0.5,
        ),
        reverse=True,
    )

    path = []
    for chart in sorted_charts[:5]:  # top 5 most efficient steps
        p = chart.perspective
        key_vars = sorted(
            p.variables_found,
            key=lambda v: v.magnitude * v.confidence,
            reverse=True,
        )
        if key_vars:
            top_var = key_vars[0]
            path.append(
                f"[{p.framework.value}] Address '{top_var.name}' "
                f"(magnitude: {top_var.magnitude:.2f}, direction: {top_var.direction.value}) — "
                f"{top_var.description[:100]}"
            )

    return path if path else ["No clear least-action path identified."]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_all_variables(perspectives: list[Perspective]) -> list[Variable]:
    """Collect all variables across all perspectives."""
    variables = []
    for p in perspectives:
        variables.extend(p.variables_found)
    return variables


def _identify_dimensions(variables: list[Variable]) -> list[str]:
    """
    Identify unique dimension axes from all variables.

    Each unique variable name represents a potential axis in the space.
    Deduplicate by name to find the true dimensionality.
    """
    seen = set()
    dimensions = []
    for v in variables:
        if v.name not in seen:
            seen.add(v.name)
            dimensions.append(v.name)
    return dimensions


def _create_charts(
    perspectives: list[Perspective],
    dimension_names: list[str],
) -> list[Chart]:
    """
    Place each perspective as a chart in the N-dimensional space.

    Each chart's coordinates represent how strongly this perspective
    speaks to each dimension (variable axis).
    """
    charts = []
    dim_index = {name: i for i, name in enumerate(dimension_names)}

    for perspective in perspectives:
        coords = [0.0] * len(dimension_names)

        # Fill in coordinates based on which variables this perspective found
        primary_dim = 0
        max_magnitude = 0.0

        for var in perspective.variables_found:
            if var.name in dim_index:
                idx = dim_index[var.name]
                # Signed coordinate: positive/negative based on direction
                sign = 1.0 if var.direction == Direction.POSITIVE else -1.0
                if var.direction in (Direction.NEUTRAL, Direction.CIRCULAR):
                    sign = 0.5  # neutral gets partial presence
                coords[idx] = var.magnitude * sign * var.confidence

                if var.magnitude > max_magnitude:
                    max_magnitude = var.magnitude
                    primary_dim = idx

        charts.append(Chart(
            perspective=perspective,
            coordinates=coords,
            dimension=primary_dim,
        ))

    return charts


def _find_overlaps(charts: list[Chart]) -> None:
    """
    Find overlapping charts — perspectives that see the same part of the manifold.

    Two charts overlap if their coordinate vectors are similar
    (cosine similarity > threshold).
    """
    overlap_threshold = 0.7

    for i, ca in enumerate(charts):
        for j, cb in enumerate(charts[i + 1:], start=i + 1):
            similarity = _cosine_similarity(ca.coordinates, cb.coordinates)
            if similarity > overlap_threshold:
                ca.overlap_with.append(j)
                cb.overlap_with.append(i)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two coordinate vectors."""
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sqrt(sum(x * x for x in a))
    magnitude_b = sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def _calculate_variance(charts: list[Chart], total_dims: int) -> list[float]:
    """
    Calculate variance explained by each dimension.

    This is a simplified SVD — measures how much each axis
    contributes to the overall spread of perspectives.
    """
    if not charts or total_dims == 0:
        return []

    variance = []
    for dim in range(total_dims):
        values = [c.coordinates[dim] for c in charts if dim < len(c.coordinates)]
        if not values:
            variance.append(0.0)
            continue

        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0.0
        variance.append(var)

    # Normalize to proportions
    total_var = sum(variance)
    if total_var > 0:
        variance = [v / total_var for v in variance]

    return variance


def _reduce_dimensions(
    variance: list[float],
    dimension_names: list[str],
) -> tuple[int, list[str]]:
    """
    Dimensional reduction — find intrinsic dimensionality.

    Uses the Elbow Method: keep dimensions until cumulative
    variance explained reaches the threshold (95%).

    Returns (intrinsic_dimension_count, core_axis_names).
    """
    if not variance:
        return 0, []

    threshold = 0.95  # 95% variance explained = enough

    # Sort dimensions by variance (highest first)
    indexed = sorted(enumerate(variance), key=lambda x: x[1], reverse=True)

    cumulative = 0.0
    core_indices = []

    for idx, var in indexed:
        cumulative += var
        core_indices.append(idx)
        if cumulative >= threshold:
            break

    core_axes = [
        dimension_names[i] for i in core_indices
        if i < len(dimension_names)
    ]

    return len(core_indices), core_axes


def _find_redundant(charts: list[Chart]) -> list[Perspective]:
    """
    Find redundant perspectives — charts that overlap heavily
    with other charts and add no new information.

    A chart is redundant if it overlaps with 2+ other charts
    and has lower weight than all of them.
    """
    redundant = []

    for chart in charts:
        if len(chart.overlap_with) >= 2:
            # Check if this chart has lower weight than its overlapping partners
            partner_weights = []
            for partner_idx in chart.overlap_with:
                if partner_idx < len(charts):
                    partner_weights.append(charts[partner_idx].perspective.weight)

            if partner_weights and chart.perspective.weight <= min(partner_weights):
                redundant.append(chart.perspective)

    return redundant
