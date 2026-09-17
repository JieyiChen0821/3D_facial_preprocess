from __future__ import annotations

import numpy as np

from face_preprocess.models.common import normalize_points, sample_indices
from face_preprocess.models.crop import crop_mesh, probabilities_to_mask, vote_indices
from face_preprocess.models.landmark import LANDMARK_NAMES, snap_landmarks
from face_preprocess.types import Mesh


def test_sampling_and_voting_are_deterministic_and_cover_all_points():
    first = sample_indices(10, 6, 123)
    second = sample_indices(10, 6, 123)
    np.testing.assert_array_equal(first, second)

    votes = vote_indices(10, num_points=6, extra_passes=1, seed=123)
    covered = np.unique(np.concatenate(votes))
    np.testing.assert_array_equal(covered, np.arange(10))


def test_normalize_points_returns_reversible_center_and_scale():
    points = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    normalized, center, scale = normalize_points(points)
    np.testing.assert_allclose(normalized * scale + center, points)
    assert scale == 1.0


def test_crop_mask_postprocessing_preserves_complete_faces_and_remaps():
    mesh = Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 0]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    probabilities = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)

    smoothed, initial, final = probabilities_to_mask(
        mesh, probabilities, threshold=0.5, smooth_iterations=0
    )
    cropped = crop_mesh(mesh, final)

    np.testing.assert_array_equal(initial, [True, True, True, False])
    np.testing.assert_array_equal(final, initial)
    assert len(cropped.vertices) == 3
    np.testing.assert_array_equal(cropped.faces, [[0, 1, 2]])
    assert smoothed.dtype == np.float32


def test_crop_mesh_removes_kept_but_unreferenced_vertices():
    mesh = Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.int64),
    )

    cropped = crop_mesh(mesh, np.ones(4, dtype=bool))

    assert len(cropped.vertices) == 3
    np.testing.assert_array_equal(cropped.faces, [[0, 1, 2]])


def test_landmark_names_and_snapping_are_stable():
    assert LANDMARK_NAMES[0] == "bijian"
    assert len(LANDMARK_NAMES) == 9
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    continuous = np.array([[0.2, 0.0, 0.0], [1.6, 0.0, 0.0]])

    snapped, indices, distances = snap_landmarks(continuous, vertices)

    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_allclose(snapped, vertices)
    np.testing.assert_allclose(distances, [0.2, 0.4])
