from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from face_preprocess.geometry.procrustes import apply_transform, transformation_from_points
from face_preprocess.types import Mesh


@dataclass(frozen=True)
class LandmarkAlignmentResult:
    mesh: Mesh
    transform: np.ndarray
    aligned_landmarks: np.ndarray
    similarity_scale: float
    determinant: float
    landmark_rms: float


def _validated_landmarks(value: np.ndarray, label: str) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.shape != (9, 3):
        raise ValueError(f"{label} landmark coordinates must have shape (9, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} landmark coordinates contain non-finite values")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError(f"{label} landmark coordinates are degenerate")
    return points


def align_mesh_with_landmarks(
    mesh: Mesh,
    source_landmarks: np.ndarray,
    template_landmarks: np.ndarray,
) -> LandmarkAlignmentResult:
    source = _validated_landmarks(source_landmarks, "source")
    target = _validated_landmarks(template_landmarks, "template")
    transform = np.asarray(
        transformation_from_points(source, target, allow_scaling=True), dtype=float
    )
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("landmark alignment produced an invalid transform")
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular_values))
    if determinant <= 0.0 or scale <= 0.0:
        raise ValueError("landmark alignment produced a reflection or non-positive scale")
    if not np.allclose(singular_values, scale, rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("landmark alignment transform is not a uniform-scale similarity")
    aligned_landmarks = apply_transform(source, transform)
    aligned_vertices = apply_transform(mesh.vertices, transform)
    if not np.isfinite(aligned_landmarks).all() or not np.isfinite(aligned_vertices).all():
        raise ValueError("landmark alignment produced non-finite coordinates")
    residual = aligned_landmarks - target
    landmark_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return LandmarkAlignmentResult(
        mesh=Mesh(aligned_vertices, mesh.faces.copy()),
        transform=transform,
        aligned_landmarks=aligned_landmarks,
        similarity_scale=scale,
        determinant=determinant,
        landmark_rms=landmark_rms,
    )


def load_template_landmarks(
    path: str | Path, expected_names: Sequence[str]
) -> np.ndarray:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ("name", "x", "y", "z"):
                raise ValueError(
                    "template landmark CSV header must be exactly: name,x,y,z"
                )
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read template landmark CSV {source}: {exc}") from exc
    names = tuple(str(row["name"]) for row in rows)
    expected = tuple(str(item) for item in expected_names)
    if len(expected) != 9 or names != expected:
        raise ValueError(
            f"template landmark names/order mismatch: expected {expected}, got {names}"
        )
    try:
        points = np.asarray(
            [[float(row[axis]) for axis in ("x", "y", "z")] for row in rows],
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("template landmark CSV contains an invalid coordinate") from exc
    return _validated_landmarks(points, "template")
