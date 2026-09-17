from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_preprocess.geometry.landmark_alignment import (
    align_mesh_with_landmarks,
    load_template_landmarks,
)
from face_preprocess.models.landmark import LANDMARK_NAMES
from face_preprocess.types import Mesh


def _nine_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 3.0, 1.0],
            [0.0, 2.0, -1.0],
            [0.0, 0.5, 0.0],
            [3.0, 2.0, -1.0],
            [1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [-3.0, 2.0, -1.0],
            [1.5, -2.0, -0.5],
            [-1.5, -2.0, -0.5],
        ],
        dtype=float,
    )


def _mesh() -> Mesh:
    vertices = np.vstack([_nine_points(), [[0.0, -3.0, 2.0]]])
    return Mesh(vertices, np.asarray([[0, 1, 3], [0, 5, 6], [2, 7, 8]], dtype=np.int64))


def test_nine_point_similarity_alignment_recovers_known_transform_for_full_mesh():
    mesh = _mesh()
    source = _nine_points()
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )
    target = source @ (1.75 * rotation).T + np.asarray([12.0, -4.0, 3.5])

    result = align_mesh_with_landmarks(mesh, source, target)

    np.testing.assert_allclose(result.aligned_landmarks, target, atol=1e-10)
    np.testing.assert_allclose(
        result.mesh.vertices,
        mesh.vertices @ (1.75 * rotation).T + np.asarray([12.0, -4.0, 3.5]),
        atol=1e-10,
    )
    assert result.similarity_scale == pytest.approx(1.75)
    assert result.determinant > 0.0
    assert result.landmark_rms == pytest.approx(0.0, abs=1e-10)


def test_alignment_passes_all_nine_points_with_equal_weight(monkeypatch):
    source = _nine_points()
    target = source + np.arange(9, dtype=float)[:, None] * np.asarray([0.1, -0.2, 0.3])
    captured: dict[str, object] = {}

    def fake_transform(points1, points2, allow_scaling):
        captured["source"] = np.asarray(points1).copy()
        captured["target"] = np.asarray(points2).copy()
        captured["allow_scaling"] = allow_scaling
        return np.eye(4)

    monkeypatch.setattr(
        "face_preprocess.geometry.landmark_alignment.transformation_from_points",
        fake_transform,
    )

    align_mesh_with_landmarks(_mesh(), source, target)

    np.testing.assert_array_equal(captured["source"], source)
    np.testing.assert_array_equal(captured["target"], target)
    assert np.asarray(captured["source"]).shape == (9, 3)
    assert captured["allow_scaling"] is True


@pytest.mark.parametrize(
    "source",
    (
        np.zeros((9, 3)),
        np.column_stack([np.arange(9), np.zeros(9), np.zeros(9)]),
        np.full((9, 3), np.nan),
    ),
)
def test_alignment_rejects_invalid_or_degenerate_landmark_sets(source):
    with pytest.raises(ValueError, match="landmark"):
        align_mesh_with_landmarks(_mesh(), source, _nine_points())


def test_template_landmark_csv_requires_exact_canonical_order(tmp_path: Path):
    path = tmp_path / "landmarks.csv"
    rows = ["name,x,y,z"] + [
        f"{name},{point[0]},{point[1]},{point[2]}"
        for name, point in zip(LANDMARK_NAMES, _nine_points(), strict=True)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    values = load_template_landmarks(path, LANDMARK_NAMES)

    assert values.shape == (9, 3)
    path.write_text("\n".join([rows[0], rows[2], rows[1], *rows[3:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names/order"):
        load_template_landmarks(path, LANDMARK_NAMES)


def test_reflected_landmarks_never_produce_a_reflection_transform():
    source = _nine_points()
    reflected = source.copy()
    reflected[:, 0] *= -1.0

    result = align_mesh_with_landmarks(_mesh(), source, reflected)

    assert result.determinant > 0.0
