from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from face_preprocess.types import Mesh


@dataclass(frozen=True)
class SymmetryResult:
    mesh: Mesh
    displacement_mean: float
    displacement_p95: float
    displacement_max: float


def symmetrize(mesh: Mesh, refindex: np.ndarray) -> SymmetryResult:
    """Average each vertex with its reflected left/right counterpart."""
    index = np.asarray(refindex, dtype=np.int64).reshape(-1)
    if len(index) != len(mesh.vertices):
        raise ValueError(
            f"refindex length {len(index)} does not match vertex count {len(mesh.vertices)}"
        )
    if len(index) and (int(index.min()) < 0 or int(index.max()) >= len(index)):
        raise ValueError("refindex contains an out-of-range vertex index")
    if not np.array_equal(index[index], np.arange(len(index), dtype=np.int64)):
        raise ValueError("refindex must be an involution")

    reflected = mesh.vertices[index].copy()
    reflected[:, 0] *= -1.0
    vertices = 0.5 * (mesh.vertices + reflected)
    displacement = np.linalg.norm(vertices - mesh.vertices, axis=1)
    return SymmetryResult(
        mesh=Mesh(vertices, mesh.faces.copy()),
        displacement_mean=float(np.mean(displacement)),
        displacement_p95=float(np.percentile(displacement, 95.0)),
        displacement_max=float(np.max(displacement)),
    )

