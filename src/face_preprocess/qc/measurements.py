from __future__ import annotations

from collections import Counter, deque

import numpy as np

from face_preprocess.config import DeviceProfile
from face_preprocess.geometry.mesh_features import vertex_normals
from face_preprocess.types import Measurement, Mesh


def _edge_table(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    return np.sort(edges, axis=1)


def _component_sizes(mesh: Mesh) -> list[int]:
    adjacency: list[set[int]] = [set() for _ in mesh.vertices]
    for a, b, c in mesh.faces:
        adjacency[int(a)].update((int(b), int(c)))
        adjacency[int(b)].update((int(a), int(c)))
        adjacency[int(c)].update((int(a), int(b)))
    visited = np.zeros(len(mesh.vertices), dtype=bool)
    sizes: list[int] = []
    for start in range(len(mesh.vertices)):
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[int] = deque([start])
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def measure_input_mesh(mesh: Mesh, profile: DeviceProfile) -> dict[str, Measurement]:
    edges = _edge_table(mesh.faces)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    edge_lengths = np.linalg.norm(
        mesh.vertices[unique_edges[:, 0]] - mesh.vertices[unique_edges[:, 1]], axis=1
    )
    triangles = mesh.vertices[mesh.faces]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    component_sizes = _component_sizes(mesh)
    extents = np.ptp(mesh.vertices, axis=0)
    rounded_vertices = np.ascontiguousarray(mesh.vertices).view(
        np.dtype((np.void, mesh.vertices.dtype.itemsize * 3))
    )
    duplicate_vertices = len(mesh.vertices) - len(np.unique(rounded_vertices))
    sorted_faces = np.sort(mesh.faces, axis=1)
    face_rows = np.ascontiguousarray(sorted_faces).view(
        np.dtype((np.void, sorted_faces.dtype.itemsize * 3))
    )
    duplicate_faces = len(mesh.faces) - len(np.unique(face_rows))
    normals = vertex_normals(mesh.vertices, mesh.faces)
    normal_lengths = np.linalg.norm(normals, axis=1)
    volume = float(np.prod(extents))
    surface_area = float(0.5 * np.sum(double_areas))
    edge_median = float(np.median(edge_lengths))
    edge_mad = float(np.median(np.abs(edge_lengths - edge_median)))
    metrics = {
        "input.edge_length.expected_mm": Measurement(
            profile.expected_edge_length_mm, "mm"
        ),
        "input.edge_length.mean_mm": Measurement(float(np.mean(edge_lengths)), "mm"),
        "input.edge_length.median_mm": Measurement(
            edge_median, "mm"
        ),
        "input.edge_length.p05_mm": Measurement(
            float(np.percentile(edge_lengths, 5.0)), "mm"
        ),
        "input.edge_length.p95_mm": Measurement(
            float(np.percentile(edge_lengths, 95.0)), "mm"
        ),
        "input.edge_length.mad_mm": Measurement(edge_mad, "mm"),
        "input.edge_length.median_to_expected_ratio": Measurement(
            edge_median / profile.expected_edge_length_mm, "ratio"
        ),
        "input.degenerate_faces.count": Measurement(
            int(np.count_nonzero(double_areas <= 1.0e-12)), "count"
        ),
        "input.duplicate_geometry.vertex_count": Measurement(
            int(duplicate_vertices), "count"
        ),
        "input.duplicate_geometry.face_count": Measurement(
            int(duplicate_faces), "count"
        ),
        "input.connected_components.count": Measurement(
            len(component_sizes), "count"
        ),
        "input.connected_components.largest_fraction": Measurement(
            float(component_sizes[0] / len(mesh.vertices)), "ratio"
        ),
        "input.connected_components.vertex_counts": Measurement(
            component_sizes,
            "count",
            shape=(len(component_sizes),),
            storage="inline",
        ),
        "input.topology.boundary_edge_count": Measurement(
            int(np.count_nonzero(edge_counts == 1)), "count"
        ),
        "input.topology.boundary_edge_fraction": Measurement(
            float(np.mean(edge_counts == 1)), "ratio"
        ),
        "input.topology.nonmanifold_edge_count": Measurement(
            int(np.count_nonzero(edge_counts > 2)), "count"
        ),
        "input.extent_density.extent_x_mm": Measurement(float(extents[0]), "mm"),
        "input.extent_density.extent_y_mm": Measurement(float(extents[1]), "mm"),
        "input.extent_density.extent_z_mm": Measurement(float(extents[2]), "mm"),
        "input.extent_density.vertices_per_mm3": Measurement(
            float(len(mesh.vertices) / volume) if volume > 0.0 else None,
            "count/mm3",
            compute_status="computed" if volume > 0.0 else "not_applicable",
        ),
        "input.extent_density.surface_area_mm2": Measurement(surface_area, "mm2"),
        "input.extent_density.vertices_per_mm2": Measurement(
            float(len(mesh.vertices) / surface_area) if surface_area > 0.0 else None,
            "count/mm2",
            compute_status="computed" if surface_area > 0.0 else "not_applicable",
        ),
        "input.normal_consistency.zero_normal_fraction": Measurement(
            float(np.mean(normal_lengths == 0.0)), "ratio"
        ),
        "input.normal_consistency.zero_face_normal_fraction": Measurement(
            float(np.mean(double_areas <= 1.0e-12)), "ratio"
        ),
    }
    return metrics


def measurements_to_dict(
    measurements: dict[str, Measurement],
) -> dict[str, dict[str, object]]:
    return {name: value.as_dict() for name, value in sorted(measurements.items())}
