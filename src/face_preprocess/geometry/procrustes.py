from __future__ import annotations

import numpy as np


def transformation_from_points(points1: np.ndarray, points2: np.ndarray, allow_scaling: bool) -> np.ndarray:
    p1 = np.asarray(points1, dtype=float)
    p2 = np.asarray(points2, dtype=float)
    if p1.shape != p2.shape or p1.ndim != 2 or p1.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {p1.shape} and {p2.shape}")

    c1 = p1.mean(axis=0)
    c2 = p2.mean(axis=0)
    x1 = p1 - c1
    x2 = p2 - c2
    norm1 = np.sqrt((x1 * x1).sum())
    norm2 = np.sqrt((x2 * x2).sum())
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Cannot align degenerate point sets")
    x1 /= norm1
    x2 /= norm2

    u, _, vt = np.linalg.svd(x1.T @ x2)
    rotation = (u @ vt).T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = (u @ vt).T

    scale = norm2 / norm1 if allow_scaling else 1.0
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = c2 - c1 @ transform[:3, :3].T
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    rot = np.asarray(transform, dtype=float)[:3, :3]
    trans = np.asarray(transform, dtype=float)[:3, 3]
    return arr @ rot.T + trans
