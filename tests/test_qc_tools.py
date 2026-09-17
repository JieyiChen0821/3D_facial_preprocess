from __future__ import annotations

import json
from pathlib import Path

import yaml

from face_preprocess.qc.tools import calibrate_qc, evaluate_run_qc, summarize_runs


def _run(root: Path) -> Path:
    sample_dir = root / "metadata" / "samples"
    sample_dir.mkdir(parents=True)
    (sample_dir / "a.json").write_text(
        json.dumps(
            {
                "output_key": "a",
                "sample_id": "display",
                "status": "passed",
                "measurements": {
                    "crop.retention.vertex_fraction": {
                        "value": 0.7,
                        "unit": "ratio",
                        "compute_status": "computed",
                        "error": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _qc(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profile_id": "q",
                "profile_version": "1",
                "metric_groups": {"crop.retention": "warning"},
                "rules": [
                    {
                        "rule_id": "r",
                        "metric": "crop.retention.vertex_fraction",
                        "operator": "lt",
                        "threshold": 0.8,
                        "unit": "ratio",
                        "severity": "warning",
                        "minimum_qc_level": "standard",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_summarize_and_evaluate_do_not_mutate_sample_metadata(tmp_path: Path):
    run = _run(tmp_path / "run")
    original = (run / "metadata" / "samples" / "a.json").read_bytes()
    summary = summarize_runs([run], tmp_path / "summary")
    evaluation = evaluate_run_qc(run, _qc(tmp_path / "qc.yaml"), tmp_path / "evaluation")

    assert summary["sample_count"] == 1
    assert evaluation["counts"]["warning"] == 1
    assert (run / "metadata" / "samples" / "a.json").read_bytes() == original


def test_calibration_writes_candidate_without_overwriting_input(tmp_path: Path):
    run = _run(tmp_path / "run")
    source = _qc(tmp_path / "qc.yaml")
    original = source.read_bytes()
    plan = tmp_path / "plan.yaml"
    plan.write_text("schema_version: 1\nplan_id: x\nsample_count_target: 5\nsampling_seed: 1\nstratify_by: []\nnotes: x\n")

    result = calibrate_qc([run], source, plan, tmp_path / "calibration")

    assert Path(result["candidate_config"]).exists()
    assert source.read_bytes() == original
    assert result["available_sample_count"] == 1


def test_calibration_can_report_labeled_detection_performance(tmp_path: Path):
    run = _run(tmp_path / "run")
    source = _qc(tmp_path / "qc.yaml")
    plan = tmp_path / "plan.yaml"
    plan.write_text("schema_version: 1\nplan_id: x\nsample_count_target: 5\nsampling_seed: 1\nstratify_by: []\nnotes: x\n")
    labels = tmp_path / "labels.csv"
    labels.write_text("output_key,consensus_label\na,unacceptable\n", encoding="utf-8")

    result = calibrate_qc(
        [run], source, plan, tmp_path / "calibration", labels=labels
    )

    assert result["labeled_performance"]["true_positive"] == 1
    assert (tmp_path / "calibration" / "labeled_predictions.csv").exists()
