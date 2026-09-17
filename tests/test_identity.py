from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from face_preprocess.hashing import derive_stage_seed, geometry_sha256, source_file_sha256
from face_preprocess.identity import assign_output_keys
from face_preprocess.types import InputRecord, Mesh


def _mesh(vertices: list[list[float]], faces: list[list[int]]) -> Mesh:
    return Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))


def test_geometry_hash_normalizes_negative_zero_but_preserves_order():
    first = _mesh([[0.0, -0.0, 0.0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    same = _mesh([[-0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    reordered = _mesh([[1, 0, 0], [0, 0, 0], [0, 1, 0]], [[1, 0, 2]])

    assert geometry_sha256(first) == geometry_sha256(same)
    assert geometry_sha256(first) != geometry_sha256(reordered)


def test_source_hash_uses_original_bytes(tmp_path: Path):
    left = tmp_path / "left.obj"
    right = tmp_path / "right.obj"
    left.write_bytes(b"v 0 0 0\n")
    right.write_bytes(b"v 0.0 0.0 0.0\n")

    assert source_file_sha256(left) != source_file_sha256(right)


def test_stage_seed_is_stable_and_stage_specific():
    geometry_hash = "a" * 64

    assert derive_stage_seed(geometry_hash, "crop.formal", 42) == derive_stage_seed(
        geometry_hash, "crop.formal", 42
    )
    assert derive_stage_seed(geometry_hash, "crop.formal", 42) != derive_stage_seed(
        geometry_hash, "landmark.formal", 42
    )


def test_collision_group_uses_full_deterministic_uuidv5():
    records = [
        InputRecord(Path("A.obj"), "A", "1" * 64, "a" * 64, "A.obj"),
        InputRecord(Path("a.obj"), "a", "2" * 64, "b" * 64, "a.obj"),
    ]

    first = assign_output_keys(records)
    second = assign_output_keys(records)

    assert [item.output_key for item in first] == [item.output_key for item in second]
    assert all(re.fullmatch(r"[Aa]__[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", item.output_key or "") for item in first)

