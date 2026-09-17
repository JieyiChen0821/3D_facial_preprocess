from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from face_preprocess.errors import ConfigError
from face_preprocess.geometry.meshmonk import MeshMonkResult
from face_preprocess.models.crop import CropResult
from face_preprocess.models.landmark import LANDMARK_NAMES, LandmarkResult
from face_preprocess.obj_io import write_obj
from face_preprocess.pipeline import PipelineDependencies, RunOptions, run_pipeline
from face_preprocess.types import Mesh


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh() -> Mesh:
    return Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
    )


class Crop:
    def crop(self, mesh, **kwargs):
        mask = np.ones(len(mesh.vertices), dtype=bool)
        return CropResult(mesh, np.ones(len(mesh.vertices)), mask, mask, {"retained_vertex_fraction": 1.0})


class Landmark:
    def predict(self, mesh, **kwargs):
        points = mesh.vertices[np.asarray([0, 1, 2, 3, 1, 2, 3, 0, 1])]
        return LandmarkResult(LANDMARK_NAMES, points, points, np.zeros(9, dtype=int), np.zeros(9), np.ones(9), np.zeros(9))


class Registration:
    def map_mesh_diagnostic(self, target, template):
        return MeshMonkResult(template, template.vertices, np.ones(4), template.vertices, 1.0, ({"residual_mean": 0.0},))


def _files(root: Path) -> tuple[Path, Path, Path, Path]:
    input_obj = root / "input.obj"
    template_obj = root / "template.obj"
    write_obj(input_obj, _mesh())
    write_obj(template_obj, _mesh())
    points = root / "Template.txt"
    np.savetxt(points, _mesh().vertices)
    template_landmarks = root / "template_landmarks.csv"
    landmark_points = _mesh().vertices[np.asarray([0, 1, 2, 3, 1, 2, 3, 0, 1])]
    template_landmarks.write_text(
        "name,x,y,z\n"
        + "\n".join(
            f"{name},{point[0]},{point[1]},{point[2]}"
            for name, point in zip(LANDMARK_NAMES, landmark_points, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    refindex = root / "refindex.txt"
    np.savetxt(refindex, np.arange(1, 5), fmt="%d")
    crop = root / "crop.pt"
    landmark = root / "landmark.pt"
    phenotype_lamda = root / "phenotype_lamda.csv"
    phenotype_lamda.write_text("0,0.2,0.3,0.5\n", encoding="utf-8")
    phenotype_names = root / "phenotype_names.pp"
    phenotype_names.write_text(
        '<PickedPoints><point name="A" x="0" y="0" z="0" /></PickedPoints>\n',
        encoding="utf-8",
    )
    crop.write_bytes(b"crop")
    landmark.write_bytes(b"landmark")
    crop_profile = root / "crop.yaml"
    crop_profile.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile_id": "crop",
                "profile_version": "1",
                "stage": "crop",
                "adapter": "pointnext_crop_v1",
                "assets": {
                    "checkpoint": {"path": str(crop), "sha256": _sha(crop)}
                },
                "common": {"base_seed": 42, "precision": "fp32"},
                "parameters": {
                    "input": {
                        "features": ["xyz"],
                        "normalization": "centroid_max_radius",
                        "sampling": {
                            "strategy": "shuffled_cover_plus_random_votes",
                            "num_points": 4,
                        },
                    },
                    "inference": {"extra_passes": 2},
                    "output_mapping": {"face_class_index": 1},
                    "postprocess": {
                        "probability_threshold": 0.5,
                        "smooth_iterations": 0,
                        "smooth_alpha": 0.5,
                        "keep_largest_component": True,
                        "min_component_vertices": 0,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    landmark_profile = root / "landmark.yaml"
    landmark_profile.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile_id": "landmark",
                "profile_version": "1",
                "stage": "landmark",
                "adapter": "heatmap_offset_landmark_v1",
                "assets": {
                    "checkpoint": {"path": str(landmark), "sha256": _sha(landmark)}
                },
                "common": {"base_seed": 20260610, "precision": "fp32"},
                "parameters": {
                    "input": {
                        "features": ["xyz"],
                        "normalization": "centroid_max_radius",
                        "sampling": {
                            "strategy": "deterministic_random_fixed_count",
                            "num_points": 4,
                        },
                    },
                    "inference": {"top_k": 2},
                    "postprocess": {"snap_method": "nearest_mesh_vertex"},
                    "output_mapping": {
                        "native_order": list(LANDMARK_NAMES),
                        "canonical_order": list(LANDMARK_NAMES),
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pipeline = root / "pipeline.yaml"
    pipeline.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile_id": "test",
                "profile_version": "1",
                "models": {
                    "crop_profile": str(crop_profile),
                    "landmark_profile": str(landmark_profile),
                },
                "geometry": {
                    "template_obj": str(template_obj),
                    "template_obj_sha256": _sha(template_obj),
                    "template_landmarks": str(template_landmarks),
                    "template_landmarks_sha256": _sha(template_landmarks),
                    "template_points": str(points),
                    "template_points_sha256": _sha(points),
                    "refindex": str(refindex),
                    "refindex_sha256": _sha(refindex),
                    "expected_template_vertices": 4,
                    "expected_template_faces": 4,
                },
                "processing": {"symmetry_enabled": False},
                "phenotype_landmarks": {
                    "template_obj": str(template_obj),
                    "template_obj_sha256": _sha(template_obj),
                    "lamda_csv": str(phenotype_lamda),
                    "lamda_csv_sha256": _sha(phenotype_lamda),
                    "names_pp": str(phenotype_names),
                    "names_pp_sha256": _sha(phenotype_names),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    device = root / "device.yaml"
    device.write_text(
        "schema_version: 1\nprofile_id: d\nprofile_version: '1'\ncoordinate_unit: mm\nexpected_edge_length_mm: 1.2\nacquisition_device: x\nacquisition_protocol: x\n",
        encoding="utf-8",
    )
    qc = root / "qc.yaml"
    qc.write_text(
        "schema_version: 1\nprofile_id: q\nprofile_version: '1'\nmetric_groups: {}\nrules: []\n",
        encoding="utf-8",
    )
    return input_obj, pipeline, device, qc


def test_runner_compact_full_and_strict_resume(tmp_path: Path):
    input_obj, pipeline, device, qc = _files(tmp_path)
    deps = PipelineDependencies(Crop(), Landmark(), Registration())
    compact_dir = tmp_path / "compact"
    base = RunOptions(input_obj, None, compact_dir, pipeline, device, qc)

    first = run_pipeline(base, deps)
    resumed = run_pipeline(RunOptions(**{**base.__dict__, "resume": True}), deps)
    full = run_pipeline(
        RunOptions(input_obj, None, tmp_path / "full", pipeline, device, qc, output_mode="full"),
        deps,
    )

    assert first.passed == 1
    assert resumed.skipped == 1 and resumed.processed == 0
    assert len(list((compact_dir / "final_obj").glob("*.obj"))) == 1
    assert not list((compact_dir / "artifacts").rglob("*.obj"))
    assert len(list((full.run_dir / "artifacts").rglob("*.obj"))) == 4
    assert len(list((full.run_dir / "artifacts").rglob("*.npz"))) == 1
    assert (compact_dir / "metadata" / "samples.csv").exists()
    assert (compact_dir / "logs" / "events.jsonl").exists()
    metadata = json.loads(
        (compact_dir / "metadata" / "samples" / "input.json").read_text(encoding="utf-8")
    )
    assert metadata["profiles"]["crop_model"] == ["crop", "1"]
    assert metadata["adapters"]["crop"]["name"] == "pointnext_crop_v1"
    assert metadata["adapters"]["landmark"]["name"] == "heatmap_offset_landmark_v1"
    assert "phenotype_landmarks" not in metadata
    assert not (compact_dir / "phenotype_landmarks").exists()
    resolved = json.loads(
        (compact_dir / "configs" / "resolved_config.json").read_text(encoding="utf-8")
    )
    assert resolved["crop"]["parameters"]["postprocess"]["smooth_alpha"] == 0.5


def test_runner_writes_optional_phenotype_outputs_and_records_hashes(tmp_path: Path):
    input_obj, pipeline, device, qc = _files(tmp_path)
    run_dir = tmp_path / "phenotype_run"

    result = run_pipeline(
        RunOptions(
            input_obj,
            None,
            run_dir,
            pipeline,
            device,
            qc,
            phenotype_landmarks=True,
        ),
        PipelineDependencies(Crop(), Landmark(), Registration()),
    )

    assert result.passed == 1
    csv_path = run_dir / "phenotype_landmarks" / "input.csv"
    pp_path = run_dir / "phenotype_landmarks" / "input.pp"
    assert csv_path.is_file()
    assert pp_path.is_file()
    metadata = json.loads(
        (run_dir / "metadata" / "samples" / "input.json").read_text(encoding="utf-8")
    )
    phenotype = metadata["phenotype_landmarks"]
    assert phenotype["count"] == 1
    assert phenotype["csv"] == "phenotype_landmarks/input.csv"
    assert phenotype["picked_points"] == "phenotype_landmarks/input.pp"
    assert phenotype["lamda_sha256"] == _sha(tmp_path / "phenotype_lamda.csv")
    assert phenotype["names_sha256"] == _sha(tmp_path / "phenotype_names.pp")


def test_requested_phenotype_topology_failure_commits_no_final_obj(tmp_path: Path):
    input_obj, pipeline, device, qc = _files(tmp_path)
    wrong_template = tmp_path / "wrong_template.obj"
    mesh = _mesh()
    write_obj(wrong_template, Mesh(mesh.vertices, mesh.faces[[1, 0, 2, 3]]))
    payload = yaml.safe_load(pipeline.read_text(encoding="utf-8"))
    payload["phenotype_landmarks"]["template_obj"] = str(wrong_template)
    payload["phenotype_landmarks"]["template_obj_sha256"] = _sha(wrong_template)
    pipeline.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_pipeline(
        RunOptions(
            input_obj,
            None,
            tmp_path / "bad_phenotype_run",
            pipeline,
            device,
            qc,
            phenotype_landmarks=True,
        ),
        PipelineDependencies(Crop(), Landmark(), Registration()),
    )

    assert result.failed == 1
    assert not list((result.run_dir / "final_obj").glob("*.obj"))
    assert not (result.run_dir / "phenotype_landmarks").exists()


def test_phenotype_flag_changes_resume_fingerprint(tmp_path: Path):
    input_obj, pipeline, device, qc = _files(tmp_path)
    run_dir = tmp_path / "resume_run"
    deps = PipelineDependencies(Crop(), Landmark(), Registration())
    base = RunOptions(input_obj, None, run_dir, pipeline, device, qc)
    run_pipeline(base, deps)

    with pytest.raises(ConfigError, match="fingerprint mismatch"):
        run_pipeline(
            RunOptions(
                **{
                    **base.__dict__,
                    "resume": True,
                    "phenotype_landmarks": True,
                }
            ),
            deps,
        )


def test_invalid_obj_fails_only_that_sample_and_batch_continues(tmp_path: Path):
    _, pipeline, device, qc = _files(tmp_path)
    (tmp_path / "bad.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
        encoding="utf-8",
    )
    result = run_pipeline(
        RunOptions(tmp_path, None, tmp_path / "run", pipeline, device, qc),
        PipelineDependencies(Crop(), Landmark(), Registration()),
    )

    assert result.total == 3
    assert result.failed == 1
    assert result.passed == 2
    assert len(list((result.run_dir / "final_obj").glob("*.obj"))) == 2
