from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError(f"vertices must have shape [N, 3], got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError(f"faces must have shape [M, 3], got {faces.shape}")
        object.__setattr__(self, "vertices", np.ascontiguousarray(vertices))
        object.__setattr__(self, "faces", np.ascontiguousarray(faces))


@dataclass(frozen=True)
class ObjInventory:
    texture_vertices: int = 0
    vertex_normals: int = 0
    material_directives: int = 0
    object_group_directives: int = 0
    smoothing_directives: int = 0
    unknown_directives: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class InputRecord:
    path: Path
    sample_id: str
    source_file_sha256: str
    geometry_sha256: str
    relative_path: str
    subject_group: str | None = None
    batch_id: str | None = None
    user_metadata: dict[str, str] = field(default_factory=dict)
    output_key: str | None = None
    obj_inventory: ObjInventory | None = None
    input_error: str | None = None
    duplicate_geometry_group_size: int = 1


@dataclass(frozen=True)
class Measurement:
    value: Any
    unit: str
    compute_status: str = "computed"
    error: str | None = None
    shape: tuple[int, ...] | None = None
    storage: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": self.value,
            "unit": self.unit,
            "compute_status": self.compute_status,
            "error": self.error,
        }
        if self.shape is not None:
            payload["shape"] = list(self.shape)
        if self.storage is not None:
            payload["storage"] = self.storage
        return payload
