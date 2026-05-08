"""Shared visualization utilities."""

from __future__ import annotations


def convex_hull_2d(points):
    """Compute convex hull of 2D points.

    Returns hull vertices or None if scipy is unavailable or the
    computation fails.
    """
    try:
        from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

        hull = ConvexHull(points)
        return points[hull.vertices]
    except (ImportError, Exception):
        return None
