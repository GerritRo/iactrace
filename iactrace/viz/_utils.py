from __future__ import annotations


def convex_hull_2d(points):
    """Convex hull of 2-D points, or None if it cannot be computed."""
    try:
        from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return None
