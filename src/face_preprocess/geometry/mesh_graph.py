from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def adjacency_from_faces(n_vertices: int, faces: np.ndarray) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = [int(x) for x in face]
        rows.extend([a, b, b, c, c, a])
        cols.extend([b, a, c, b, a, c])
    data = np.ones(len(rows), dtype=float)
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    adj.data[:] = 1.0
    return adj


def smooth_displacements(displacements: np.ndarray, faces: np.ndarray, smoothing_lambda: float) -> np.ndarray:
    disp = np.asarray(displacements, dtype=float)
    if smoothing_lambda <= 0:
        return disp
    n_vertices = disp.shape[0]
    adj = adjacency_from_faces(n_vertices, faces)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    laplacian = sparse.diags(degree) - adj
    system = sparse.eye(n_vertices, format="csr") + float(smoothing_lambda) * laplacian
    smoothed = np.zeros_like(disp)
    for dim in range(3):
        smoothed[:, dim] = spsolve(system, disp[:, dim])
    return smoothed
