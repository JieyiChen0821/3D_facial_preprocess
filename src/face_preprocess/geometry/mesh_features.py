from __future__ import annotations

import numpy as np


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1)
    out = np.zeros_like(arr)
    keep = norms > 0.0
    out[keep] = arr[keep] / norms[keep, None]
    return out


def vertex_normals(vertices: np.ndarray, faces: np.ndarray, reference_normals: np.ndarray | None = None) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(verts)
    if len(tris) == 0:
        return normals

    tri_vertices = verts[tris]
    face_normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    # Match OpenMesh/MeshMonk: vertex normals average unit face normals, not area-weighted crosses.
    face_normals = _normalize_vectors(face_normals)
    for corner in range(3):
        np.add.at(normals, tris[:, corner], face_normals)

    normals = _normalize_vectors(normals)
    if reference_normals is not None and len(reference_normals) == len(normals):
        ref = np.asarray(reference_normals, dtype=float)
        if np.sum(normals * ref) < 0.0:
            normals = -normals
    return normals


def boundary_vertex_mask(n_vertices: int, faces: np.ndarray) -> np.ndarray:
    edge_counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = [int(item) for item in face]
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    mask = np.zeros(int(n_vertices), dtype=bool)
    for (u, v), count in edge_counts.items():
        if count == 1:
            mask[u] = True
            mask[v] = True
    return mask


def badly_sized_triangle_vertex_mask(vertices: np.ndarray, faces: np.ndarray, zscore_threshold: float) -> np.ndarray:
    verts = np.asarray(vertices, dtype=float)
    tris = np.asarray(faces, dtype=np.int64)
    mask = np.zeros(len(verts), dtype=bool)
    if len(tris) == 0:
        return mask

    tri_vertices = verts[tris]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]),
        axis=1,
    )
    std = float(np.std(areas))
    if std <= 0.0:
        return mask
    zscores = (areas - float(np.mean(areas))) / std
    bad_faces = tris[np.abs(zscores) > float(zscore_threshold)]
    if len(bad_faces):
        mask[np.unique(bad_faces)] = True
    return mask
