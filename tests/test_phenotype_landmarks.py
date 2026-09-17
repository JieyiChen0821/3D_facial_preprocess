from __future__ import annotations

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from face_preprocess.obj_io import write_obj
from face_preprocess.phenotype_landmarks import (
    PhenotypeAssets,
    PhenotypeLandmarks,
    load_phenotype_assets,
    project_phenotype_landmarks,
    write_landmark_csv,
    write_picked_points,
)
from face_preprocess.types import Mesh


def _mesh() -> Mesh:
    return Mesh(
        vertices=np.array(
            [
                [1.0, 1.0, 1.0],
                [11.0, 1.0, 1.0],
                [1.0, 11.0, 1.0],
                [1.0, 1.0, 11.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
    )


def _assets() -> PhenotypeAssets:
    return PhenotypeAssets(
        names=("A", "B"),
        face_indices=np.array([0, 1]),
        lambdas=np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]),
        template_faces=np.array([[0, 1, 2], [0, 2, 3]]),
    )


def test_project_phenotype_landmarks_uses_historical_lambda_order() -> None:
    result = project_phenotype_landmarks(_mesh(), _assets())

    assert result.names == ("A", "B")
    np.testing.assert_allclose(result.coordinates, [[4.0, 3.0, 1.0], [1.0, 8.0, 2.0]])


def test_project_phenotype_landmarks_rejects_topology_mismatch() -> None:
    mesh = Mesh(_mesh().vertices, np.array([[0, 2, 1], [0, 2, 3]]))

    with pytest.raises(ValueError, match="topology"):
        project_phenotype_landmarks(mesh, _assets())


def test_project_phenotype_landmarks_rejects_non_finite_vertices() -> None:
    vertices = _mesh().vertices.copy()
    vertices[0, 0] = np.nan
    mesh = Mesh(vertices, _mesh().faces)

    with pytest.raises(ValueError, match="finite"):
        project_phenotype_landmarks(mesh, _assets())


def test_project_phenotype_landmarks_rejects_invalid_face_index() -> None:
    assets = PhenotypeAssets(
        names=("A",),
        face_indices=np.array([2]),
        lambdas=np.array([[0.2, 0.3, 0.5]]),
        template_faces=_mesh().faces,
    )

    with pytest.raises(ValueError, match="face index"):
        project_phenotype_landmarks(_mesh(), assets)


def test_project_phenotype_landmarks_rejects_non_barycentric_weights() -> None:
    assets = PhenotypeAssets(
        names=("A",),
        face_indices=np.array([0]),
        lambdas=np.array([[0.2, 0.3, 0.6]]),
        template_faces=_mesh().faces,
    )

    with pytest.raises(ValueError, match="sum to one"):
        project_phenotype_landmarks(_mesh(), assets)


def test_load_phenotype_assets_rejects_name_count_mismatch(tmp_path: Path) -> None:
    template_obj = tmp_path / "template.obj"
    write_obj(template_obj, _mesh())
    lamda_csv = tmp_path / "lamda.csv"
    lamda_csv.write_text("0,0.2,0.3,0.5\n1,0.1,0.7,0.2\n", encoding="utf-8")
    names_pp = tmp_path / "names.pp"
    names_pp.write_text(
        "<PickedPoints><point name=\"A\" x=\"0\" y=\"0\" z=\"0\" /></PickedPoints>\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Name count"):
        load_phenotype_assets(template_obj, lamda_csv, names_pp)


def test_writers_create_parseable_named_csv_and_pp(tmp_path: Path) -> None:
    result = PhenotypeLandmarks(
        names=("A&B", "B"),
        coordinates=np.array([[4.0, 3.0, 1.0], [1.0, 8.0, 2.0]]),
    )
    csv_path = tmp_path / "landmarks.csv"
    pp_path = tmp_path / "landmarks.pp"

    write_landmark_csv(csv_path, result)
    write_picked_points(pp_path, result, 'sample & "scan".obj')

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["name", "x", "y", "z"],
        ["A&B", "4", "3", "1"],
        ["B", "1", "8", "2"],
    ]

    root = ET.parse(pp_path).getroot()
    assert root.tag == "PickedPoints"
    assert root.find("./DocumentData/DataFileName").attrib["name"] == 'sample & "scan".obj'
    points = root.findall("point")
    assert [point.attrib["name"] for point in points] == ["A&B", "B"]
    assert [float(points[0].attrib[axis]) for axis in ("x", "y", "z")] == [4.0, 3.0, 1.0]
