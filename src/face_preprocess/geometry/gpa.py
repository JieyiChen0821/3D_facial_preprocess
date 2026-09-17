from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face_preprocess.geometry.procrustes import apply_transform, transformation_from_points
from face_preprocess.types import Mesh


@dataclass(frozen=True)
class GPAResult:
    mesh: Mesh
    transform: np.ndarray
    similarity_scale: float
    combined_determinant: float
    centroid_norm: float
    centroid_size: float


def gpa_align(mesh: Mesh, template_vertices: np.ndarray) -> GPAResult:
    """Similarity-align a registered mesh to the unitless reference template."""
    target = np.asarray(template_vertices, dtype=float)
    if target.shape != mesh.vertices.shape:
        raise ValueError(
            f"GPA requires matching vertex arrays, got {mesh.vertices.shape} and {target.shape}"
        )

    centroid = mesh.vertices.mean(axis=0)
    centered = mesh.vertices - centroid
    centroid_size = float(np.linalg.norm(centered))
    if centroid_size <= 0.0:
        raise ValueError("Cannot GPA-align a degenerate mesh")

    transform = transformation_from_points(mesh.vertices, target, allow_scaling=True)
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0.0:
        raise ValueError("GPA transform contains a reflection or singular transform")

    scale = float(np.cbrt(determinant))
    aligned = apply_transform(mesh.vertices, transform)
    return GPAResult(
        mesh=Mesh(aligned, mesh.faces.copy()),
        transform=transform,
        similarity_scale=scale,
        combined_determinant=determinant,
        centroid_norm=float(np.linalg.norm(centroid)),
        centroid_size=centroid_size,
    )

