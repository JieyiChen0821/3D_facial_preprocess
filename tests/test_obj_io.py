from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face_preprocess.errors import ObjFormatError
from face_preprocess.obj_io import read_obj, write_obj
from face_preprocess.types import Mesh


def test_read_obj_supports_slash_tokens_negative_indices_and_inventory(tmp_path: Path):
    path = tmp_path / "face.obj"
    path.write_text(
        "\n".join(
            (
                "mtllib material.mtl",
                "o face",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 0",
                "vn 0 0 1",
                "usemtl skin",
                "f 1/1/1 2/1/1 -1/1/1",
            )
        ),
        encoding="utf-8",
    )

    mesh, inventory = read_obj(path)

    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2]])
    assert inventory.texture_vertices == 1
    assert inventory.vertex_normals == 1
    assert inventory.material_directives == 2
    assert inventory.object_group_directives == 1


def test_read_obj_rejects_ngon_instead_of_truncating(tmp_path: Path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n", encoding="utf-8")

    with pytest.raises(ObjFormatError, match="exactly three"):
        read_obj(path)


def test_read_obj_accepts_inline_comments(tmp_path: Path):
    path = tmp_path / "commented.obj"
    path.write_text(
        "v 0 0 0 # origin\nv 1 0 0\nv 0 1 0\nf 1 2 3 # triangle\n",
        encoding="utf-8",
    )

    mesh, _ = read_obj(path)

    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2]])


@pytest.mark.parametrize(
    "content, expected",
    (
        ("v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", "non-finite"),
        ("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n", "out of range"),
        ("v 0 0 0\n", "at least one triangular face"),
    ),
)
def test_read_obj_rejects_hard_geometry_errors(tmp_path: Path, content: str, expected: str):
    path = tmp_path / "bad.obj"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ObjFormatError, match=expected):
        read_obj(path)


def test_write_obj_is_canonical_utf8_lf_and_ten_decimals(tmp_path: Path):
    mesh = Mesh(
        vertices=np.array([[0.0, -0.0, 1.25], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    path = tmp_path / "out.obj"

    write_obj(path, mesh)

    payload = path.read_bytes()
    assert b"\r\n" not in payload
    assert payload == (
        b"v 0.0000000000 -0.0000000000 1.2500000000\n"
        b"v 1.0000000000 0.0000000000 0.0000000000\n"
        b"v 0.0000000000 1.0000000000 0.0000000000\n"
        b"f 1 2 3\n"
    )
