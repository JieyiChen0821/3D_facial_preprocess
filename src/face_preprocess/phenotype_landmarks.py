from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from face_preprocess.obj_io import read_obj
from face_preprocess.types import Mesh


@dataclass(frozen=True)
class PhenotypeAssets:
    names: tuple[str, ...]
    face_indices: np.ndarray
    lambdas: np.ndarray
    template_faces: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        object.__setattr__(
            self,
            "face_indices",
            np.ascontiguousarray(np.asarray(self.face_indices, dtype=np.int64)),
        )
        object.__setattr__(
            self,
            "lambdas",
            np.ascontiguousarray(np.asarray(self.lambdas, dtype=np.float64)),
        )
        object.__setattr__(
            self,
            "template_faces",
            np.ascontiguousarray(np.asarray(self.template_faces, dtype=np.int64)),
        )


@dataclass(frozen=True)
class PhenotypeLandmarks:
    names: tuple[str, ...]
    coordinates: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        object.__setattr__(
            self,
            "coordinates",
            np.ascontiguousarray(np.asarray(self.coordinates, dtype=np.float64)),
        )


def _read_lamda_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    face_indices: list[int] = []
    lambdas: list[tuple[float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(
                    f"Expected 4 columns in {path}:{line_number}, got {len(row)}"
                )
            face_indices.append(int(float(row[0])))
            lambdas.append((float(row[1]), float(row[2]), float(row[3])))
    if not face_indices:
        raise ValueError(f"No lamda rows found in {path}")
    return np.asarray(face_indices, dtype=np.int64), np.asarray(lambdas, dtype=np.float64)


def _read_landmark_names(path: Path) -> tuple[str, ...]:
    root = ET.parse(path).getroot()
    names = tuple(
        str(point.attrib["name"]).strip()
        for point in root.iter("point")
        if str(point.attrib.get("name", "")).strip()
    )
    if not names:
        raise ValueError(f"No point names found in {path}")
    return names


def load_phenotype_assets(
    template_obj: Path,
    lamda_csv: Path,
    names_pp: Path,
) -> PhenotypeAssets:
    template, _ = read_obj(template_obj)
    face_indices, lambdas = _read_lamda_rows(lamda_csv)
    names = _read_landmark_names(names_pp)
    if len(names) != len(face_indices):
        raise ValueError(
            f"Name count ({len(names)}) does not match lamda row count ({len(face_indices)})"
        )
    assets = PhenotypeAssets(
        names=names,
        face_indices=face_indices,
        lambdas=lambdas,
        template_faces=template.faces,
    )
    _validate_assets(assets)
    return assets


def _validate_assets(assets: PhenotypeAssets) -> None:
    count = len(assets.names)
    if not count or any(not name for name in assets.names):
        raise ValueError("Phenotype landmark names must be non-empty")
    if len(set(assets.names)) != count:
        raise ValueError("Phenotype landmark names must be unique")
    if assets.face_indices.shape != (count,):
        raise ValueError("Phenotype face index count must match landmark names")
    if assets.lambdas.shape != (count, 3):
        raise ValueError("Phenotype lambda rows must have shape [landmarks, 3]")
    if assets.template_faces.ndim != 2 or assets.template_faces.shape[1:] != (3,):
        raise ValueError("Phenotype template faces must have shape [faces, 3]")
    if not np.all(np.isfinite(assets.lambdas)):
        raise ValueError("Phenotype lambda values must be finite")
    if not np.allclose(assets.lambdas.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Phenotype lambda rows must sum to one")
    if np.any(assets.face_indices < 0) or np.any(
        assets.face_indices >= len(assets.template_faces)
    ):
        raise ValueError("Phenotype face index is out of range")


def project_phenotype_landmarks(
    mesh: Mesh,
    assets: PhenotypeAssets,
) -> PhenotypeLandmarks:
    _validate_assets(assets)
    if mesh.faces.shape != assets.template_faces.shape or not np.array_equal(
        mesh.faces, assets.template_faces
    ):
        raise ValueError("Mesh topology does not match the phenotype landmark template")
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError("Mesh vertices must be finite")

    triangles = mesh.vertices[mesh.faces[assets.face_indices]]
    lambda1 = assets.lambdas[:, 0, None]
    lambda2 = assets.lambdas[:, 1, None]
    lambda3 = assets.lambdas[:, 2, None]
    coordinates = (
        lambda3 * triangles[:, 0]
        + lambda2 * triangles[:, 1]
        + lambda1 * triangles[:, 2]
    )
    return PhenotypeLandmarks(assets.names, coordinates)


def _validate_result(result: PhenotypeLandmarks) -> None:
    if result.coordinates.shape != (len(result.names), 3):
        raise ValueError("Phenotype landmark coordinates must have shape [landmarks, 3]")
    if not np.all(np.isfinite(result.coordinates)):
        raise ValueError("Phenotype landmark coordinates must be finite")


def _format_coordinate(value: float) -> str:
    return f"{float(value):.12g}"


def write_landmark_csv(path: Path, result: PhenotypeLandmarks) -> None:
    _validate_result(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "x", "y", "z"])
        for name, coordinate in zip(result.names, result.coordinates, strict=True):
            writer.writerow([name, *(_format_coordinate(value) for value in coordinate)])


def write_picked_points(
    path: Path,
    result: PhenotypeLandmarks,
    source_name: str,
) -> None:
    _validate_result(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("PickedPoints")
    document = ET.SubElement(root, "DocumentData")
    ET.SubElement(document, "DataFileName", {"name": str(source_name)})
    for name, coordinate in zip(result.names, result.coordinates, strict=True):
        ET.SubElement(
            root,
            "point",
            {
                "name": name,
                "x": _format_coordinate(coordinate[0]),
                "y": _format_coordinate(coordinate[1]),
                "z": _format_coordinate(coordinate[2]),
                "active": "1",
            },
        )
    ET.indent(root, space=" ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
