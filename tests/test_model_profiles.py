from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from face_preprocess.config import load_run_config
from face_preprocess.errors import ConfigError


LANDMARK_ORDER = [
    "bijian",
    "bigen",
    "bixia",
    "wyjzuo",
    "nyjzuo",
    "nyjyou",
    "wyjyou",
    "kouzuo",
    "kouyou",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_profile_config(root: Path) -> tuple[Path, Path, Path]:
    assets = root / "assets"
    assets.mkdir()
    crop_checkpoint = assets / "crop.pt"
    landmark_checkpoint = assets / "landmark.pt"
    labels = assets / "labels.txt"
    template_obj = assets / "template.obj"
    template_landmarks = assets / "template_landmarks.csv"
    template_points = assets / "template.txt"
    refindex = assets / "refindex.txt"
    for path, value in (
        (crop_checkpoint, b"crop"),
        (landmark_checkpoint, b"landmark"),
        (labels, b"face\nbackground\n"),
        (template_obj, b"obj"),
        (template_landmarks, b"name,x,y,z\n"),
        (template_points, b"points"),
        (refindex, b"refindex"),
    ):
        path.write_bytes(value)

    crop_profile = root / "models" / "crop.yaml"
    _write_yaml(
        crop_profile,
        {
            "schema_version": 1,
            "profile_id": "crop_current",
            "profile_version": "1.0.0",
            "stage": "crop",
            "adapter": "pointnext_crop_v1",
            "assets": {
                "checkpoint": {
                    "path": "../assets/crop.pt",
                    "sha256": _sha(crop_checkpoint),
                },
                "labels": {
                    "path": "../assets/labels.txt",
                    "sha256": _sha(labels),
                },
            },
            "common": {"base_seed": 42, "precision": "fp32"},
            "parameters": {
                "input": {
                    "features": ["xyz"],
                    "normalization": "centroid_max_radius",
                    "sampling": {
                        "strategy": "shuffled_cover_plus_random_votes",
                        "num_points": 32768,
                    },
                },
                "inference": {"extra_passes": 2},
                "output_mapping": {"face_class_index": 1},
                "postprocess": {
                    "probability_threshold": 0.5,
                    "smooth_iterations": 0,
                    "keep_largest_component": True,
                    "min_component_vertices": 100,
                },
            },
        },
    )
    landmark_profile = root / "models" / "landmark.yaml"
    _write_yaml(
        landmark_profile,
        {
            "schema_version": 1,
            "profile_id": "landmark_current",
            "profile_version": "1.0.0",
            "stage": "landmark",
            "adapter": "heatmap_offset_landmark_v1",
            "assets": {
                "checkpoint": {
                    "path": "../assets/landmark.pt",
                    "sha256": _sha(landmark_checkpoint),
                }
            },
            "common": {"base_seed": 20260610, "precision": "fp32"},
            "parameters": {
                "input": {
                    "features": ["xyz"],
                    "normalization": "centroid_max_radius",
                    "sampling": {
                        "strategy": "deterministic_random_fixed_count",
                        "num_points": 32768,
                    },
                },
                "inference": {"top_k": 16},
                "postprocess": {"snap_method": "nearest_mesh_vertex"},
                "output_mapping": {
                    "native_order": LANDMARK_ORDER,
                    "canonical_order": LANDMARK_ORDER,
                },
            },
        },
    )

    pipeline = root / "pipeline.yaml"
    _write_yaml(
        pipeline,
        {
            "schema_version": 1,
            "profile_id": "server_landmarks_v1",
            "profile_version": "1.1.0",
            "models": {
                "crop_profile": "models/crop.yaml",
                "landmark_profile": "models/landmark.yaml",
            },
            "geometry": {
                "template_obj": "assets/template.obj",
                "template_obj_sha256": _sha(template_obj),
                "template_landmarks": "assets/template_landmarks.csv",
                "template_landmarks_sha256": _sha(template_landmarks),
                "template_points": "assets/template.txt",
                "template_points_sha256": _sha(template_points),
                "refindex": "assets/refindex.txt",
                "refindex_sha256": _sha(refindex),
                "expected_template_vertices": 7906,
                "expected_template_faces": 15598,
            },
            "processing": {"symmetry_enabled": False},
        },
    )
    device = root / "device.yaml"
    _write_yaml(
        device,
        {
            "schema_version": 1,
            "profile_id": "scanner",
            "profile_version": "1.0.0",
            "coordinate_unit": "mm",
            "expected_edge_length_mm": 1.2,
            "acquisition_device": "scanner-a",
            "acquisition_protocol": "protocol-a",
        },
    )
    qc = root / "qc.yaml"
    _write_yaml(
        qc,
        {
            "schema_version": 1,
            "profile_id": "explore",
            "profile_version": "1.0.0",
            "metric_groups": {"input.edge_length": "record_only"},
            "rules": [],
        },
    )
    return pipeline, device, qc


def test_pipeline_loads_independent_model_profiles_and_all_named_assets(tmp_path: Path):
    pipeline, device, qc = _write_profile_config(tmp_path)

    config = load_run_config(pipeline, device, qc, "standard", False)

    assert config.profile_id == "server_landmarks_v1"
    assert config.crop.adapter == "pointnext_crop_v1"
    assert config.crop.profile_id == "crop_current"
    assert config.crop.checkpoint == (tmp_path / "assets/crop.pt").resolve()
    assert config.crop.assets["labels"].path == (tmp_path / "assets/labels.txt").resolve()
    assert config.crop.precision == "fp32"
    assert config.crop.parameters["postprocess"]["smooth_alpha"] == 0.5
    assert config.landmark.parameters["inference"]["top_k"] == 16
    assert config.geometry.template_landmarks == (tmp_path / "assets/template_landmarks.csv").resolve()
    assert set(config.config_hashes) == {
        "pipeline",
        "device",
        "qc",
        "crop_model",
        "landmark_model",
    }
    assert config.source_paths["crop_model"].name == "crop.yaml"


def test_unknown_adapter_is_rejected_while_loading_shared_configuration(tmp_path: Path):
    pipeline, device, qc = _write_profile_config(tmp_path)
    crop_profile = tmp_path / "models/crop.yaml"
    payload = yaml.safe_load(crop_profile.read_text(encoding="utf-8"))
    payload["adapter"] = "missing_crop_v9"
    _write_yaml(crop_profile, payload)

    with pytest.raises(ConfigError, match="unknown crop adapter"):
        load_run_config(pipeline, device, qc, "standard", False)


def test_model_profile_stage_and_unknown_fields_are_strict(tmp_path: Path):
    pipeline, device, qc = _write_profile_config(tmp_path)
    landmark_profile = tmp_path / "models/landmark.yaml"
    payload = yaml.safe_load(landmark_profile.read_text(encoding="utf-8"))
    payload["stage"] = "crop"
    _write_yaml(landmark_profile, payload)

    with pytest.raises(ConfigError, match="stage must be landmark"):
        load_run_config(pipeline, device, qc, "standard", False)
