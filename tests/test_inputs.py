from __future__ import annotations

import csv
from pathlib import Path

import pytest

from face_preprocess.errors import ManifestError
from face_preprocess.inputs import discover_inputs


TRIANGLE_OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def test_manifest_paths_are_relative_and_extra_columns_are_user_metadata(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    obj = data_dir / "face.obj"
    obj.write_text(TRIANGLE_OBJ, encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_obj", "sample_id", "subject_group", "batch_id", "note"])
        writer.writeheader()
        writer.writerow(
            {
                "input_obj": "data/face.obj",
                "sample_id": "display-only",
                "subject_group": "family-1",
                "batch_id": "batch-a",
                "note": "kept verbatim",
            }
        )

    records = discover_inputs(input_manifest=manifest)

    assert records[0].path == obj.resolve()
    assert records[0].sample_id == "display-only"
    assert records[0].user_metadata == {"note": "kept verbatim"}


def test_duplicate_manifest_path_is_a_startup_error(tmp_path: Path):
    obj = tmp_path / "face.obj"
    obj.write_text(TRIANGLE_OBJ, encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("input_obj,sample_id\nface.obj,a\nface.obj,b\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="duplicate input path"):
        discover_inputs(input_manifest=manifest)


def test_flat_directory_discovery_is_case_insensitive_and_non_recursive(tmp_path: Path):
    (tmp_path / "a.OBJ").write_text(TRIANGLE_OBJ, encoding="utf-8")
    (tmp_path / "b.obj").write_text(TRIANGLE_OBJ, encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.obj").write_text(TRIANGLE_OBJ, encoding="utf-8")

    records = discover_inputs(input_path=tmp_path)

    assert [item.path.name for item in records] == ["a.OBJ", "b.obj"]


def test_invalid_obj_is_retained_as_a_sample_failure_candidate(tmp_path: Path):
    (tmp_path / "good.obj").write_text(TRIANGLE_OBJ, encoding="utf-8")
    (tmp_path / "bad.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
        encoding="utf-8",
    )

    records = discover_inputs(input_path=tmp_path)

    assert len(records) == 2
    bad = next(item for item in records if item.path.name == "bad.obj")
    assert bad.input_error and "ObjFormatError" in bad.input_error


def test_duplicate_geometry_is_recorded_without_deduplication(tmp_path: Path):
    (tmp_path / "a.obj").write_text(TRIANGLE_OBJ, encoding="utf-8")
    (tmp_path / "b.obj").write_text(TRIANGLE_OBJ, encoding="utf-8")

    records = discover_inputs(input_path=tmp_path)

    assert len(records) == 2
    assert {item.duplicate_geometry_group_size for item in records} == {2}
