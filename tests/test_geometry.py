from __future__ import annotations

import numpy as np
import pytest

from face_preprocess.geometry.gpa import gpa_align
from face_preprocess.geometry.procrustes import apply_transform, transformation_from_points
from face_preprocess.geometry.symmetry import symmetrize
from face_preprocess.types import Mesh


def _tetrahedron() -> Mesh:
    return Mesh(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        ),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
    )


def test_similarity_transform_recovers_scale_rotation_and_translation():
    source = _tetrahedron().vertices
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = source @ (2.5 * rotation).T + np.array([4.0, -2.0, 3.0])

    transform = transformation_from_points(source, target, allow_scaling=True)

    np.testing.assert_allclose(apply_transform(source, transform), target, atol=1e-10)
    assert np.linalg.det(transform[:3, :3]) > 0


def test_gpa_returns_diagnostic_transforms_and_valid_unitless_mesh():
    template = _tetrahedron()
    source = Mesh(template.vertices * 3.0 + np.array([10.0, -4.0, 2.0]), template.faces)

    result = gpa_align(source, template.vertices)

    np.testing.assert_allclose(result.mesh.vertices, template.vertices, atol=1e-10)
    assert result.similarity_scale > 0
    assert result.combined_determinant > 0
    assert result.centroid_norm >= 0
    assert result.centroid_size > 0


def test_gpa_rejects_reflection(monkeypatch):
    mesh = _tetrahedron()
    reflected = np.eye(4)
    reflected[0, 0] = -1

    monkeypatch.setattr("face_preprocess.geometry.gpa.transformation_from_points", lambda *args, **kwargs: reflected)

    with pytest.raises(ValueError, match="reflection"):
        gpa_align(mesh, mesh.vertices)


def test_symmetrize_reports_displacement_and_enforces_refindex():
    mesh = Mesh(
        np.array([[-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.int64),
    )

    result = symmetrize(mesh, np.array([1, 0, 2], dtype=np.int64))

    np.testing.assert_allclose(result.mesh.vertices[:, 0], [-1.5, 1.5, 0.0])
    assert result.displacement_max > 0
    with pytest.raises(ValueError, match="length"):
        symmetrize(mesh, np.array([0, 1], dtype=np.int64))

