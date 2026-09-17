from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from face_preprocess.geometry.mesh_features import (
    badly_sized_triangle_vertex_mask,
    boundary_vertex_mask,
    vertex_normals,
)
from face_preprocess.types import Mesh

_MIN_AFFINITY = 1.0e-4
_MIN_DISTANCE_SQUARED = 1.0e-6
_MIN_SMOOTHING_WEIGHT = 1.0e-5


def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    safe = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0.0)
    return sparse.diags(safe).dot(matrix).tocsr()


def _features_from_mesh(mesh: Mesh, reference_normals: np.ndarray | None = None) -> np.ndarray:
    normals = vertex_normals(mesh.vertices, mesh.faces, reference_normals=reference_normals)
    return np.column_stack([np.asarray(mesh.vertices, dtype=float), normals])


def _knn_affinity(
    floating_features: np.ndarray,
    target_features: np.ndarray,
    num_neighbours: int,
    normalize: bool,
) -> sparse.csr_matrix:
    floating = np.asarray(floating_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    k = max(1, min(int(num_neighbours), len(target)))
    distances, indices = cKDTree(target).query(floating, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    squared = np.maximum(distances * distances, _MIN_DISTANCE_SQUARED)
    weights = 1.0 / squared
    floating_normals = floating[:, None, 3:6]
    target_normals = target[indices, 3:6]
    orientation = np.sum(floating_normals * target_normals, axis=2) / 2.0 + 0.5
    weights = np.maximum(weights * orientation, _MIN_AFFINITY)

    rows = np.repeat(np.arange(len(floating)), k)
    matrix = sparse.csr_matrix(
        (weights.ravel(), (rows, indices.ravel())),
        shape=(len(floating), len(target)),
    )
    if normalize:
        matrix = _row_normalize(matrix)
    return matrix


def _affinity_to_correspondences(
    affinity: sparse.csr_matrix,
    target_features: np.ndarray,
    target_flags: np.ndarray,
    floating_flags: np.ndarray,
    flag_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _row_normalize(affinity)
    corresponding = normalized.dot(np.asarray(target_features, dtype=float))
    weighted_flags = np.asarray(normalized.dot(np.asarray(target_flags, dtype=float))).ravel()
    flags = (weighted_flags > float(flag_threshold)).astype(float)
    flags *= np.asarray(floating_flags, dtype=float)
    return np.asarray(corresponding, dtype=float), flags


def compute_correspondences(
    floating_features: np.ndarray,
    target_features: np.ndarray,
    floating_flags: np.ndarray,
    target_flags: np.ndarray,
    symmetric: bool,
    num_neighbours: int,
    flag_threshold: float,
    equalize_push_pull: bool,
) -> tuple[np.ndarray, np.ndarray]:
    floating = np.asarray(floating_features, dtype=float)
    target = np.asarray(target_features, dtype=float)
    f_flags = np.asarray(floating_flags, dtype=float)
    t_flags = np.asarray(target_flags, dtype=float)

    if not symmetric:
        affinity = _knn_affinity(floating, target, num_neighbours, normalize=True)
        return _affinity_to_correspondences(affinity, target, t_flags, f_flags, flag_threshold)

    push = _knn_affinity(floating, target, num_neighbours, normalize=False)
    pull = _knn_affinity(target, floating, num_neighbours, normalize=False)
    pull_features, pull_flags = _affinity_to_correspondences(pull, floating, f_flags, t_flags, flag_threshold)
    del pull_features
    if equalize_push_pull:
        push = _row_normalize(push)
        pull = _row_normalize(pull)

    affinity = push + pull.transpose().tocsr()
    corresponding, flags = _affinity_to_correspondences(affinity, target, t_flags, f_flags, flag_threshold)
    corresponding_pull_flags = np.asarray(_row_normalize(affinity).dot(pull_flags)).ravel()
    flags *= (corresponding_pull_flags > float(flag_threshold)).astype(float)
    return corresponding, flags


def compute_inlier_weights(
    floating_features: np.ndarray,
    corresponding_features: np.ndarray,
    corresponding_flags: np.ndarray,
    kappa: float,
    use_orientation: bool,
) -> np.ndarray:
    floating = np.asarray(floating_features, dtype=float)
    corresponding = np.asarray(corresponding_features, dtype=float)
    weights = np.asarray(corresponding_flags, dtype=float).copy()
    if len(weights) == 0:
        return weights

    for _ in range(10):
        diff = corresponding - floating
        distance_squared = np.sum(diff * diff, axis=1)
        denominator = float(np.sum(weights))
        if denominator <= 0.0:
            return np.zeros_like(weights)
        sigma = float(np.sqrt(np.sum(weights * distance_squared) / denominator))
        sigma = min(max(sigma, 0.1), 10.0)
        density = 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * distance_squared / (sigma * sigma))
        lambda_value = 1.0 / (np.sqrt(2.0 * np.pi) * sigma) * np.exp(-0.5 * float(kappa) * float(kappa))
        probability = density / (density + lambda_value)
        weights *= probability

    if use_orientation:
        orientation = np.sum(floating[:, 3:6] * corresponding[:, 3:6], axis=1) / 2.0 + 0.5
        weights *= np.clip(orientation, 0.0, 1.0)
    return weights


def _nearest_indices_and_weights(points: np.ndarray, k: int, sigma: float, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_points = len(points)
    neighbours = max(1, min(int(k), n_points))
    distances, indices = cKDTree(points).query(points, k=neighbours)
    if neighbours == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    squared = distances * distances
    weights = np.exp(-0.5 * squared / (float(sigma) * float(sigma)))
    weights *= np.asarray(flags, dtype=float)[indices]
    weights = (1.0 - _MIN_SMOOTHING_WEIGHT) * weights + _MIN_SMOOTHING_WEIGHT
    weights /= np.sum(weights, axis=1, keepdims=True)
    return indices, weights


def _weights_to_sparse_matrix(indices: np.ndarray, weights: np.ndarray) -> sparse.csr_matrix:
    n_rows, n_neighbours = indices.shape
    rows = np.repeat(np.arange(n_rows), n_neighbours)
    return sparse.csr_matrix(
        (weights.ravel(), (rows, indices.ravel())),
        shape=(n_rows, n_rows),
    )


def _smooth_field(field: np.ndarray, indices: np.ndarray, base_weights: np.ndarray, inlier_weights: np.ndarray, passes: int) -> np.ndarray:
    current = np.asarray(field, dtype=float).copy()
    neighbour_inliers = np.asarray(inlier_weights, dtype=float)[indices]
    weights = neighbour_inliers * base_weights
    weights = (1.0 - _MIN_SMOOTHING_WEIGHT) * weights + _MIN_SMOOTHING_WEIGHT
    weights /= np.sum(weights, axis=1, keepdims=True)
    smoother = _weights_to_sparse_matrix(indices, weights)
    for _ in range(max(0, int(passes))):
        current = smoother.dot(current)
    return current


def _diffuse_outliers(
    displacement: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    inlier_weights: np.ndarray,
    passes: int,
) -> np.ndarray:
    current = np.asarray(displacement, dtype=float).copy()
    outlier = np.asarray(inlier_weights, dtype=float) <= 0.8
    if not np.any(outlier):
        return current
    smoother = _weights_to_sparse_matrix(indices, weights)
    for _ in range(max(0, int(passes))):
        previous = current.copy()
        averages = smoother.dot(previous)
        current[outlier] = previous[outlier] * inlier_weights[outlier, None] + (
            1.0 - inlier_weights[outlier, None]
        ) * averages[outlier]
    return current


def viscoelastic_update(
    floating_features: np.ndarray,
    corresponding_features: np.ndarray,
    faces: np.ndarray,
    floating_flags: np.ndarray,
    inlier_weights: np.ndarray,
    displacement_field: np.ndarray,
    num_smoothing_neighbours: int,
    sigma: float,
    viscous_iterations: int,
    elastic_iterations: int,
    outlier_diffusion_iterations: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(floating_features, dtype=float).copy()
    corresponding = np.asarray(corresponding_features, dtype=float)
    displacement = np.asarray(displacement_field, dtype=float).copy()
    old_displacement = displacement.copy()

    indices, smoothing_weights = _nearest_indices_and_weights(
        features[:, :3],
        num_smoothing_neighbours,
        sigma,
        floating_flags,
    )
    force = (corresponding[:, :3] - features[:, :3]) * np.asarray(inlier_weights, dtype=float)[:, None]
    regularized_force = _smooth_field(force, indices, smoothing_weights, inlier_weights, viscous_iterations)
    displacement += regularized_force
    displacement = _smooth_field(displacement, indices, smoothing_weights, inlier_weights, elastic_iterations)
    displacement = _diffuse_outliers(displacement, indices, smoothing_weights, inlier_weights, outlier_diffusion_iterations)

    features[:, :3] += displacement - old_displacement
    features[:, 3:6] = vertex_normals(features[:, :3], faces, reference_normals=features[:, 3:6])
    return features, displacement


def _weighted_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    allow_scaling: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = w > 1.0e-8
    if np.count_nonzero(keep) < 3:
        keep = np.ones_like(w, dtype=bool)
        w = np.ones_like(w, dtype=float)
    src = src[keep]
    dst = dst[keep]
    w = w[keep]
    w = w / np.sum(w)

    src_centroid = np.sum(src * w[:, None], axis=0)
    dst_centroid = np.sum(dst * w[:, None], axis=0)
    x = src - src_centroid
    y = dst - dst_centroid
    covariance = (x * w[:, None]).T @ y
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = (u @ vt).T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = (u @ vt).T

    scale = 1.0
    if allow_scaling:
        denominator = float(np.sum(w * np.sum(x * x, axis=1)))
        if denominator > 0.0:
            scale = float(np.sum(singular_values) / denominator)
    translation = dst_centroid - src_centroid @ (scale * rotation).T
    return rotation, translation, scale


def _apply_rigid_update(features: np.ndarray, corresponding: np.ndarray, weights: np.ndarray, use_scaling: bool) -> np.ndarray:
    updated = np.asarray(features, dtype=float).copy()
    rotation, translation, scale = _weighted_similarity_transform(
        updated[:, :3],
        corresponding[:, :3],
        weights,
        use_scaling,
    )
    updated[:, :3] = updated[:, :3] @ (scale * rotation).T + translation
    updated[:, 3:6] = updated[:, 3:6] @ rotation.T
    norms = np.linalg.norm(updated[:, 3:6], axis=1)
    keep = norms > 0.0
    updated[keep, 3:6] /= norms[keep, None]
    return updated


def shape_mapper_annealed_iterations(start: int, end: int, iteration: int, num_iterations: int) -> int:
    if num_iterations <= 1 or start <= end:
        return int(end)
    rate = np.exp(np.log(float(end) / float(start)) / float(num_iterations - 1))
    return max(int(end), int(round(float(start) * rate ** (int(iteration) - 1))))


def _mesh_flags(
    mesh: Mesh,
    flag_boundary: bool,
    flag_badly_sized_triangles: bool,
    triangle_size_zscore: float,
) -> np.ndarray:
    flags = np.ones(len(mesh.vertices), dtype=float)
    if flag_boundary:
        flags[boundary_vertex_mask(len(mesh.vertices), mesh.faces)] = 0.0
    if flag_badly_sized_triangles:
        flags[badly_sized_triangle_vertex_mask(mesh.vertices, mesh.faces, triangle_size_zscore)] = 0.0
    return flags


@dataclass(frozen=True)
class PythonMeshMonkBackend:
    rigid_iterations: int = 150
    rigid_use_scaling: bool = True
    rigid_inlier_kappa: float = 3.0
    nonrigid_iterations: int = 200
    nonrigid_inlier_kappa: float = 12.0
    correspondences_symmetric: bool = True
    correspondences_num_neighbours: int = 3
    correspondences_flag_threshold: float = 0.9
    correspondences_equalize_push_pull: bool = False
    transform_sigma: float = 3.0
    transform_num_viscous_iterations_start: int = 200
    transform_num_viscous_iterations_end: int = 1
    transform_num_elastic_iterations_start: int = 200
    transform_num_elastic_iterations_end: int = 1
    transform_num_neighbours: int = 80
    flag_floating_boundary: bool = True
    flag_target_boundary: bool = True
    flag_target_badly_sized_triangles: bool = True
    triangle_size_zscore: float = 6.0
    outlier_diffusion_iterations: int = 15

    def map_mesh(self, target: Mesh, template: Mesh) -> Mesh:
        return self.map_mesh_diagnostic(target=target, template=template).mesh

    def map_mesh_diagnostic(self, target: Mesh, template: Mesh) -> "MeshMonkResult":
        floating_features = _features_from_mesh(template)
        target_features = _features_from_mesh(target)
        floating_flags = _mesh_flags(
            template,
            flag_boundary=self.flag_floating_boundary,
            flag_badly_sized_triangles=False,
            triangle_size_zscore=self.triangle_size_zscore,
        )
        target_flags = _mesh_flags(
            target,
            flag_boundary=self.flag_target_boundary,
            flag_badly_sized_triangles=self.flag_target_badly_sized_triangles,
            triangle_size_zscore=self.triangle_size_zscore,
        )

        trajectory: list[dict[str, float | int | str]] = []
        rigid_scale = 1.0
        for iteration in range(max(0, int(self.rigid_iterations)) + 1):
            corresponding, flags = compute_correspondences(
                floating_features,
                target_features,
                floating_flags,
                target_flags,
                symmetric=self.correspondences_symmetric,
                num_neighbours=self.correspondences_num_neighbours,
                flag_threshold=self.correspondences_flag_threshold,
                equalize_push_pull=self.correspondences_equalize_push_pull,
            )
            weights = compute_inlier_weights(
                floating_features,
                corresponding,
                flags,
                kappa=self.rigid_inlier_kappa,
                use_orientation=True,
            )
            rotation, translation, scale = _weighted_similarity_transform(
                floating_features[:, :3],
                corresponding[:, :3],
                weights,
                self.rigid_use_scaling,
            )
            floating_features[:, :3] = (
                floating_features[:, :3] @ (scale * rotation).T + translation
            )
            floating_features[:, 3:6] = floating_features[:, 3:6] @ rotation.T
            normal_norms = np.linalg.norm(floating_features[:, 3:6], axis=1)
            keep_normals = normal_norms > 0.0
            floating_features[keep_normals, 3:6] /= normal_norms[keep_normals, None]
            rigid_scale *= scale
            if iteration in {0, int(self.rigid_iterations)} or iteration % 10 == 0:
                residual = np.linalg.norm(
                    corresponding[:, :3] - floating_features[:, :3], axis=1
                )
                trajectory.append(
                    {
                        "phase": "rigid",
                        "iteration": iteration,
                        "residual_mean": float(np.mean(residual)),
                        "inlier_weight_mean": float(np.mean(weights)),
                    }
                )

        rigid_end_vertices = floating_features[:, :3].copy()
        for iteration in range(max(0, int(self.nonrigid_iterations)) + 1):
            corresponding, flags = compute_correspondences(
                floating_features,
                target_features,
                floating_flags,
                target_flags,
                symmetric=self.correspondences_symmetric,
                num_neighbours=self.correspondences_num_neighbours,
                flag_threshold=self.correspondences_flag_threshold,
                equalize_push_pull=self.correspondences_equalize_push_pull,
            )
            weights = compute_inlier_weights(
                floating_features,
                corresponding,
                flags,
                kappa=self.nonrigid_inlier_kappa,
                use_orientation=True,
            )
            viscous = shape_mapper_annealed_iterations(
                self.transform_num_viscous_iterations_start,
                self.transform_num_viscous_iterations_end,
                iteration,
                max(1, int(self.nonrigid_iterations)),
            )
            elastic = shape_mapper_annealed_iterations(
                self.transform_num_elastic_iterations_start,
                self.transform_num_elastic_iterations_end,
                iteration,
                max(1, int(self.nonrigid_iterations)),
            )
            displacement = np.zeros((len(floating_features), 3), dtype=float)
            floating_features, _ = viscoelastic_update(
                floating_features,
                corresponding,
                template.faces,
                floating_flags,
                weights,
                displacement,
                num_smoothing_neighbours=self.transform_num_neighbours,
                sigma=self.transform_sigma,
                viscous_iterations=viscous,
                elastic_iterations=elastic,
                outlier_diffusion_iterations=self.outlier_diffusion_iterations,
            )
            if iteration in {0, int(self.nonrigid_iterations)} or iteration % 10 == 0:
                residual = np.linalg.norm(
                    corresponding[:, :3] - floating_features[:, :3], axis=1
                )
                trajectory.append(
                    {
                        "phase": "nonrigid",
                        "iteration": iteration,
                        "residual_mean": float(np.mean(residual)),
                        "inlier_weight_mean": float(np.mean(weights)),
                    }
                )

        return MeshMonkResult(
            mesh=Mesh(vertices=floating_features[:, :3].copy(), faces=template.faces.copy()),
            final_correspondences=corresponding[:, :3].copy(),
            final_weights=weights.copy(),
            rigid_end_vertices=rigid_end_vertices,
            rigid_scale=float(rigid_scale),
            trajectory=tuple(trajectory),
        )


@dataclass(frozen=True)
class MeshMonkResult:
    mesh: Mesh
    final_correspondences: np.ndarray
    final_weights: np.ndarray
    rigid_end_vertices: np.ndarray
    rigid_scale: float
    trajectory: tuple[dict[str, float | int | str], ...]
