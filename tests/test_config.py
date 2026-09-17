from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from face_preprocess.config import load_run_config, validate_artifact
from face_preprocess.errors import ArtifactError, ConfigError


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_configs(root: Path) -> tuple[Path, Path, Path]:
    crop = root / "crop.pt"
    landmark = root / "landmark.pt"
    template_obj = root / "template.obj"
    template_landmarks = root / "template_landmarks.csv"
    template_txt = root / "template.txt"
    refindex = root / "refindex.txt"
    phenotype_lamda = root / "phenotype_lamda.csv"
    phenotype_names = root / "phenotype_names.pp"
    for path, payload in (
        (crop, b"crop"),
        (landmark, b"landmark"),
        (template_obj, b"obj"),
        (template_landmarks, b"name,x,y,z\n"),
        (template_txt, b"txt"),
        (refindex, b"ref"),
        (phenotype_lamda, b"0,0.2,0.3,0.5\n"),
        (
            phenotype_names,
            b'<PickedPoints><point name="A" x="0" y="0" z="0" /></PickedPoints>\n',
        ),
    ):
        path.write_bytes(payload)
    crop_profile = {
        "schema_version": 1,
        "profile_id": "crop",
        "profile_version": "1.0.0",
        "stage": "crop",
        "adapter": "pointnext_crop_v1",
        "assets": {
            "checkpoint": {"path": "crop.pt", "sha256": _sha(crop)}
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
                "smooth_alpha": 0.5,
                "keep_largest_component": True,
                "min_component_vertices": 100,
            },
        },
    }
    landmark_order = [
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
    landmark_profile = {
        "schema_version": 1,
        "profile_id": "landmark",
        "profile_version": "1.0.0",
        "stage": "landmark",
        "adapter": "heatmap_offset_landmark_v1",
        "assets": {
            "checkpoint": {"path": "landmark.pt", "sha256": _sha(landmark)}
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
                "native_order": landmark_order,
                "canonical_order": landmark_order,
            },
        },
    }
    (root / "crop.yaml").write_text(
        yaml.safe_dump(crop_profile, sort_keys=False), encoding="utf-8"
    )
    (root / "landmark.yaml").write_text(
        yaml.safe_dump(landmark_profile, sort_keys=False), encoding="utf-8"
    )
    pipeline = {
        "schema_version": 1,
        "profile_id": "server",
        "profile_version": "1.0.0",
        "models": {
            "crop_profile": "crop.yaml",
            "landmark_profile": "landmark.yaml",
        },
        "geometry": {
            "template_obj": "template.obj",
            "template_obj_sha256": _sha(template_obj),
            "template_landmarks": "template_landmarks.csv",
            "template_landmarks_sha256": _sha(template_landmarks),
            "template_points": "template.txt",
            "template_points_sha256": _sha(template_txt),
            "refindex": "refindex.txt",
            "refindex_sha256": _sha(refindex),
            "expected_template_vertices": 7906,
            "expected_template_faces": 15598,
        },
        "processing": {"symmetry_enabled": False},
        "phenotype_landmarks": {
            "template_obj": "template.obj",
            "template_obj_sha256": _sha(template_obj),
            "lamda_csv": "phenotype_lamda.csv",
            "lamda_csv_sha256": _sha(phenotype_lamda),
            "names_pp": "phenotype_names.pp",
            "names_pp_sha256": _sha(phenotype_names),
        },
    }
    device = {
        "schema_version": 1,
        "profile_id": "scanner",
        "profile_version": "1.0.0",
        "coordinate_unit": "mm",
        "expected_edge_length_mm": 1.2,
        "acquisition_device": "scanner-a",
        "acquisition_protocol": "protocol-a",
    }
    qc = {
        "schema_version": 1,
        "profile_id": "explore",
        "profile_version": "1.0.0",
        "metric_groups": {"input.edge_length": "record_only"},
        "rules": [
            {
                "rule_id": "edge_large",
                "metric": "input.edge_length.lcc.median_mm",
                "operator": "gt",
                "threshold": 2.0,
                "unit": "mm",
                "severity": "warning",
                "minimum_qc_level": "standard",
            }
        ],
    }
    pipeline_path = root / "pipeline.yaml"
    device_path = root / "device.yaml"
    qc_path = root / "qc.yaml"
    pipeline_path.write_text(yaml.safe_dump(pipeline, sort_keys=False), encoding="utf-8")
    device_path.write_text(yaml.safe_dump(device, sort_keys=False), encoding="utf-8")
    qc_path.write_text(yaml.safe_dump(qc, sort_keys=False), encoding="utf-8")
    return pipeline_path, device_path, qc_path


def test_config_paths_resolve_relative_to_owning_yaml(tmp_path: Path):
    pipeline, device, qc = _write_configs(tmp_path)

    resolved = load_run_config(pipeline, device, qc, qc_level="standard", symmetry_override=False)

    assert resolved.crop.checkpoint == (tmp_path / "crop.pt").resolve()
    assert resolved.geometry.template_obj == (tmp_path / "template.obj").resolve()
    assert resolved.geometry.template_landmarks == (tmp_path / "template_landmarks.csv").resolve()
    assert resolved.phenotype.template_obj == (tmp_path / "template.obj").resolve()
    assert resolved.phenotype.lamda_csv == (tmp_path / "phenotype_lamda.csv").resolve()
    assert resolved.phenotype.names_pp == (tmp_path / "phenotype_names.pp").resolve()
    assert len(resolved.phenotype.lamda_csv_sha256) == 64
    assert resolved.device.expected_edge_length_mm == 1.2
    assert len(resolved.config_hashes["pipeline"]) == 64


def test_unknown_config_field_is_rejected(tmp_path: Path):
    pipeline, device, qc = _write_configs(tmp_path)
    payload = yaml.safe_load(pipeline.read_text(encoding="utf-8"))
    payload["mystery"] = True
    pipeline.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown fields"):
        load_run_config(pipeline, device, qc, qc_level="standard", symmetry_override=False)


def test_artifact_hash_mismatch_is_fatal(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")

    with pytest.raises(ArtifactError, match="SHA256 mismatch"):
        validate_artifact(path, "0" * 64)


@pytest.mark.parametrize("mutation", ("unknown", "missing"))
def test_phenotype_config_is_strict(tmp_path: Path, mutation: str):
    pipeline, device, qc = _write_configs(tmp_path)
    payload = yaml.safe_load(pipeline.read_text(encoding="utf-8"))
    if mutation == "unknown":
        payload["phenotype_landmarks"]["mystery"] = True
    else:
        del payload["phenotype_landmarks"]["names_pp_sha256"]
    pipeline.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="phenotype"):
        load_run_config(pipeline, device, qc, qc_level="standard", symmetry_override=False)


@pytest.mark.parametrize(
    "mutate, expected",
    (
        (
            lambda qc: qc["rules"].append(dict(qc["rules"][0])),
            "duplicate rule_id",
        ),
        (
            lambda qc: qc["rules"][0].update(operator="eval"),
            "unsupported operator",
        ),
        (
            lambda qc: qc["rules"][0].update(minimum_qc_level="enhanced"),
            "enhanced-only",
        ),
    ),
)
def test_qc_contract_validation(tmp_path: Path, mutate, expected: str):
    pipeline, device, qc_path = _write_configs(tmp_path)
    qc = yaml.safe_load(qc_path.read_text(encoding="utf-8"))
    mutate(qc)
    qc_path.write_text(yaml.safe_dump(qc, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match=expected):
        load_run_config(pipeline, device, qc_path, qc_level="standard", symmetry_override=False)
