from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face_preprocess.types import Mesh


@dataclass(frozen=True)
class SurfaceResidualResult:
    forward: dict[str, float]
    reverse: dict[str, float]
    chamfer: float
    hausdorff: float


def _summarize(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def _point_triangle_distances(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Exact closest surface distance with bounded-memory chunking."""
    try:
        import trimesh
    except ImportError:
        # The dependency is part of the runtime lock. This fallback keeps the
        # diagnostic usable in minimal environments for small meshes.
        result = np.full(len(points), np.inf, dtype=float)
        for point_index, point in enumerate(np.asarray(points, dtype=float)):
            best = np.inf
            for triangle in triangles:
                best = min(best, _point_triangle_distance(point, triangle))
            result[point_index] = best
        return result
    surface = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(len(triangles) * 3, dtype=np.int64).reshape(-1, 3),
        process=False,
    )
    try:
        _, distances, _ = trimesh.proximity.closest_point(surface, points)
    except Exception:
        if len(points) * len(triangles) > 2_000_000:
            raise RuntimeError(
                "accelerated surface proximity is unavailable; install the locked rtree dependency"
            )
        _, distances, _ = trimesh.proximity.closest_point_naive(surface, points)
    return np.asarray(distances, dtype=float)


def _point_triangle_distance(point: np.ndarray, triangle: np.ndarray) -> float:
    # Real-Time Collision Detection, Christer Ericson, section 5.1.5.
    a, b, c = triangle
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(point - (a + v * ab)))
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(point - (a + w * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(point - (b + w * (c - b))))
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return float(np.linalg.norm(point - (a + ab * v + ac * w)))


def surface_residuals(source: Mesh, target: Mesh) -> SurfaceResidualResult:
    forward_values = _point_triangle_distances(source.vertices, target.vertices[target.faces])
    reverse_values = _point_triangle_distances(target.vertices, source.vertices[source.faces])
    forward = _summarize(forward_values)
    reverse = _summarize(reverse_values)
    return SurfaceResidualResult(
        forward=forward,
        reverse=reverse,
        chamfer=float(0.5 * (forward["mean"] + reverse["mean"])),
        hausdorff=float(max(forward["max"], reverse["max"])),
    )


def _segment_triangle_intersects(
    start: np.ndarray, end: np.ndarray, triangle: np.ndarray, epsilon: float = 1.0e-10
) -> bool:
    direction = end - start
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    h = np.cross(direction, edge2)
    determinant = float(edge1 @ h)
    if abs(determinant) <= epsilon:
        return False
    inverse = 1.0 / determinant
    s = start - triangle[0]
    u = inverse * float(s @ h)
    if u < -epsilon or u > 1.0 + epsilon:
        return False
    q = np.cross(s, edge1)
    v = inverse * float(direction @ q)
    if v < -epsilon or u + v > 1.0 + epsilon:
        return False
    t = inverse * float(edge2 @ q)
    return -epsilon <= t <= 1.0 + epsilon


def _orientation_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment_2d(
    point: np.ndarray, start: np.ndarray, end: np.ndarray, epsilon: float
) -> bool:
    return (
        min(start[0], end[0]) - epsilon
        <= point[0]
        <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon
        <= point[1]
        <= max(start[1], end[1]) + epsilon
        and abs(_orientation_2d(start, end, point)) <= epsilon
    )


def _segments_intersect_2d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, epsilon: float
) -> bool:
    o1, o2 = _orientation_2d(a, b, c), _orientation_2d(a, b, d)
    o3, o4 = _orientation_2d(c, d, a), _orientation_2d(c, d, b)
    if ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon)) and (
        (o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon)
    ):
        return True
    return (
        _on_segment_2d(c, a, b, epsilon)
        or _on_segment_2d(d, a, b, epsilon)
        or _on_segment_2d(a, c, d, epsilon)
        or _on_segment_2d(b, c, d, epsilon)
    )


def _point_in_triangle_2d(
    point: np.ndarray, triangle: np.ndarray, epsilon: float
) -> bool:
    values = [
        _orientation_2d(triangle[index], triangle[(index + 1) % 3], point)
        for index in range(3)
    ]
    return not (any(value > epsilon for value in values) and any(value < -epsilon for value in values))


def _coplanar_triangles_intersect(
    first: np.ndarray, second: np.ndarray, normal: np.ndarray, epsilon: float
) -> bool:
    drop_axis = int(np.argmax(np.abs(normal)))
    first_2d = np.delete(first, drop_axis, axis=1)
    second_2d = np.delete(second, drop_axis, axis=1)
    for first_index in range(3):
        for second_index in range(3):
            if _segments_intersect_2d(
                first_2d[first_index],
                first_2d[(first_index + 1) % 3],
                second_2d[second_index],
                second_2d[(second_index + 1) % 3],
                epsilon,
            ):
                return True
    return _point_in_triangle_2d(
        first_2d[0], second_2d, epsilon
    ) or _point_in_triangle_2d(second_2d[0], first_2d, epsilon)


def _triangles_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    epsilon = 1.0e-10
    first_normal = np.cross(first[1] - first[0], first[2] - first[0])
    second_normal = np.cross(second[1] - second[0], second[2] - second[0])
    if np.linalg.norm(first_normal) <= epsilon or np.linalg.norm(second_normal) <= epsilon:
        return False
    first_plane = (second - first[0]) @ first_normal
    second_plane = (first - second[0]) @ second_normal
    if np.all(np.abs(first_plane) <= epsilon) and np.all(np.abs(second_plane) <= epsilon):
        return _coplanar_triangles_intersect(
            first, second, first_normal, epsilon
        )
    if (np.all(first_plane > epsilon) or np.all(first_plane < -epsilon)) or (
        np.all(second_plane > epsilon) or np.all(second_plane < -epsilon)
    ):
        return False
    for triangle, other in ((first, second), (second, first)):
        for index in range(3):
            if _segment_triangle_intersects(
                triangle[index], triangle[(index + 1) % 3], other
            ):
                return True
    return False


def self_intersection_summary(mesh: Mesh, *, max_pairs: int = 1000) -> dict[str, object]:
    """Return intersecting non-adjacent face pairs using AABB broad-phase."""
    triangles = mesh.vertices[mesh.faces]
    minimum = triangles.min(axis=1)
    maximum = triangles.max(axis=1)
    order = np.argsort(minimum[:, 0], kind="stable")
    pairs: list[list[int]] = []
    for order_position, raw_first in enumerate(order):
        first = int(raw_first)
        for raw_second in order[order_position + 1 :]:
            second = int(raw_second)
            if minimum[second, 0] > maximum[first, 0]:
                break
            if np.intersect1d(mesh.faces[first], mesh.faces[second]).size:
                continue
            if np.any(maximum[first, 1:] < minimum[second, 1:]) or np.any(
                maximum[second, 1:] < minimum[first, 1:]
            ):
                continue
            if _triangles_intersect(triangles[first], triangles[second]):
                pairs.append([min(first, second), max(first, second)])
                if len(pairs) >= max_pairs:
                    return {"count": len(pairs), "pairs": pairs, "truncated": True}
    pairs.sort()
    return {"count": len(pairs), "pairs": pairs, "truncated": False}
