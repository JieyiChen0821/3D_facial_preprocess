from __future__ import annotations

from pathlib import Path
import hashlib

import yaml

from face_preprocess import __version__

ROOT = Path(__file__).parents[1]


def test_distribution_files_exist_and_slurm_has_no_hardcoded_module_load():
    required = [
        "environments/environment.yml",
        "environments/requirements-linux-cuda121.lock.txt",
        "environments/requirements-linux-cpu.lock.txt",
        "environments/requirements-windows-cuda121.lock.txt",
        "environments/requirements-windows-cpu.lock.txt",
        "scripts/bootstrap_server.sh",
        "scripts/bootstrap_windows.ps1",
        "slurm/doctor.sbatch",
        "slurm/run_face_preprocess.sbatch",
        "README_zh.md",
        "docs/metadata_schema_zh.md",
        "docs/qc_calibration_zh.md",
        "docs/server_runbook_zh.md",
        "configs/models/crop_pointnext_current.yaml",
        "configs/models/landmark_heatmap_offset_current.yaml",
        "assets/template/landmarks_9_reference.csv",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative
    for path in (ROOT / "slurm").glob("*.sbatch"):
        content = path.read_text(encoding="utf-8")
        assert "module load" not in content
        assert "face-preprocess" in content


def test_packaged_configs_and_hashes_are_valid():
    pipeline = yaml.safe_load((ROOT / "configs" / "pipeline_server.yaml").read_text(encoding="utf-8"))
    assert pipeline["geometry"]["expected_template_vertices"] == 7906
    crop = yaml.safe_load(
        (ROOT / "configs" / pipeline["models"]["crop_profile"]).read_text(encoding="utf-8")
    )
    landmark = yaml.safe_load(
        (ROOT / "configs" / pipeline["models"]["landmark_profile"]).read_text(encoding="utf-8")
    )
    hashes = [
        crop["assets"]["checkpoint"]["sha256"],
        landmark["assets"]["checkpoint"]["sha256"],
        pipeline["geometry"]["template_obj_sha256"],
        pipeline["geometry"]["template_landmarks_sha256"],
        pipeline["geometry"]["template_points_sha256"],
        pipeline["geometry"]["refindex_sha256"],
    ]
    assert all(len(value) == 64 and value == value.lower() for value in hashes)
    exploratory = yaml.safe_load((ROOT / "configs" / "qc_exploratory.yaml").read_text(encoding="utf-8"))
    assert len(exploratory["metric_groups"]) == 24


def test_landmark_pipeline_has_independent_version_identity():
    pipeline = yaml.safe_load(
        (ROOT / "configs" / "pipeline_server.yaml").read_text(encoding="utf-8")
    )

    assert __version__ == "1.1.0"
    assert pipeline["profile_id"] == "server_landmarks_v1"
    assert pipeline["profile_version"] == "1.1.0"
    assert 'version = "1.1.0"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_packaged_asset_manifest_matches_every_file():
    for line in (ROOT / "assets" / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / "assets" / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
