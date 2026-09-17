from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np

from face_preprocess.types import Mesh


def source_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_sha256(mesh: Mesh) -> str:
    vertices = np.array(mesh.vertices, dtype="<f8", order="C", copy=True)
    vertices[vertices == 0.0] = 0.0
    faces = np.array(mesh.faces, dtype="<i8", order="C", copy=True)
    digest = hashlib.sha256()
    digest.update(b"face-preprocess-geometry-v1\0")
    digest.update(struct.pack("<QQ", *vertices.shape))
    digest.update(vertices.tobytes(order="C"))
    digest.update(struct.pack("<QQ", *faces.shape))
    digest.update(faces.tobytes(order="C"))
    return digest.hexdigest()


def derive_stage_seed(geometry_hash: str, stage: str, base_seed: int) -> int:
    payload = f"{geometry_hash}:{stage}:{int(base_seed)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)

