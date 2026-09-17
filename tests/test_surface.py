from __future__ import annotations

import numpy as np

from face_preprocess.geometry.surface import self_intersection_summary, surface_residuals
from face_preprocess.types import Mesh


def test_surface_residuals_are_zero_for_identical_triangle():
    mesh = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]], dtype=np.int64),
    )

    result = surface_residuals(mesh, mesh)

    assert result.forward["max"] == 0.0
    assert result.reverse["max"] == 0.0
    assert result.chamfer == 0.0
    assert result.hausdorff == 0.0


def test_self_intersection_counts_nonadjacent_crossing_triangles():
    mesh = Mesh(
        np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -0.5, -1.0],
                [0.0, -0.5, 1.0],
                [0.0, 0.5, 0.0],
            ]
        ),
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
    )

    summary = self_intersection_summary(mesh)

    assert summary["count"] == 1
    assert summary["pairs"] == [[0, 1]]


def test_self_intersection_excludes_triangles_sharing_vertices():
    mesh = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
    )

    assert self_intersection_summary(mesh)["count"] == 0


def test_self_intersection_detects_coplanar_overlap():
    mesh = Mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.5, 0.5, 0.0],
                [1.5, 0.5, 0.0],
                [0.5, 1.5, 0.0],
            ]
        ),
        np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
    )

    assert self_intersection_summary(mesh)["count"] == 1
