from __future__ import annotations

import numpy as np
import random


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("points must be a non-empty Nx3 array")
    center = vertices.mean(axis=0)
    scale = float(np.linalg.norm(vertices - center, axis=1).max())
    if scale <= 0.0:
        raise ValueError("cannot normalize a degenerate point cloud")
    return ((vertices - center) / scale).astype(np.float32), center, scale


def sample_indices(point_count: int, num_points: int, seed: int) -> np.ndarray:
    if point_count <= 0 or num_points <= 0:
        raise ValueError("point_count and num_points must be positive")
    rng = random.Random(int(seed))
    if point_count >= num_points:
        indices = rng.sample(range(point_count), int(num_points))
    else:
        indices = [rng.randrange(point_count) for _ in range(int(num_points))]
    return np.asarray(indices, dtype=np.int64)
