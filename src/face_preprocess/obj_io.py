from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import os
from pathlib import Path

import numpy as np

from face_preprocess.errors import ObjFormatError
from face_preprocess.types import Mesh, ObjInventory


def _face_index(token: str, vertex_count: int, path: Path, line_number: int) -> int:
    raw_token = token.split("/", 1)[0]
    if not raw_token:
        raise ObjFormatError(f"missing vertex index at line {line_number} in {path}")
    try:
        raw_index = int(raw_token)
    except ValueError as exc:
        raise ObjFormatError(f"invalid face index {raw_token!r} at line {line_number} in {path}") from exc
    if raw_index == 0:
        raise ObjFormatError(f"OBJ index zero is invalid at line {line_number} in {path}")
    index = raw_index - 1 if raw_index > 0 else vertex_count + raw_index
    if index < 0 or index >= vertex_count:
        raise ObjFormatError(f"face index out of range at line {line_number} in {path}: {raw_index}")
    return index


def read_obj(path: str | Path) -> tuple[Mesh, ObjInventory]:
    source = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    texture_vertices = 0
    vertex_normals = 0
    material_directives = 0
    object_group_directives = 0
    smoothing_directives = 0
    unknown: Counter[str] = Counter()

    try:
        handle = source.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ObjFormatError(f"cannot read OBJ {source}: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.split("#", 1)[0].strip()
            if not stripped:
                continue
            parts = stripped.split()
            directive = parts[0]
            if directive == "v":
                if len(parts) < 4:
                    raise ObjFormatError(f"vertex line {line_number} in {source} needs three coordinates")
                try:
                    point = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError as exc:
                    raise ObjFormatError(f"invalid vertex at line {line_number} in {source}") from exc
                if not all(math.isfinite(value) for value in point):
                    raise ObjFormatError(f"non-finite vertex at line {line_number} in {source}")
                vertices.append(point)
            elif directive == "f":
                tokens = parts[1:]
                if len(tokens) != 3:
                    raise ObjFormatError(
                        f"face at line {line_number} in {source} must contain exactly three vertices"
                    )
                faces.append([_face_index(token, len(vertices), source, line_number) for token in tokens])
            elif directive == "vt":
                texture_vertices += 1
            elif directive == "vn":
                vertex_normals += 1
            elif directive in {"mtllib", "usemtl"}:
                material_directives += 1
            elif directive in {"o", "g"}:
                object_group_directives += 1
            elif directive == "s":
                smoothing_directives += 1
            else:
                unknown[directive] += 1

    if not vertices:
        raise ObjFormatError(f"OBJ must contain at least one vertex: {source}")
    if not faces:
        raise ObjFormatError(f"OBJ must contain at least one triangular face: {source}")

    mesh = Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    inventory = ObjInventory(
        texture_vertices=texture_vertices,
        vertex_normals=vertex_normals,
        material_directives=material_directives,
        object_group_directives=object_group_directives,
        smoothing_directives=smoothing_directives,
        unknown_directives=dict(sorted(unknown.items())),
    )
    return mesh, inventory


def write_obj(path: str | Path, mesh: Mesh) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not np.all(np.isfinite(mesh.vertices)):
        raise ObjFormatError("cannot write non-finite vertices")
    if len(mesh.faces) and (mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices)):
        raise ObjFormatError("cannot write faces with out-of-range indices")
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for x, y, z in mesh.vertices:
            handle.write(f"v {x:.10f} {y:.10f} {z:.10f}\n")
        for a, b, c in mesh.faces:
            handle.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")
        handle.flush()
        os.fsync(handle.fileno())
