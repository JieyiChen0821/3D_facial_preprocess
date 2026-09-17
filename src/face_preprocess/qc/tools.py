from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import yaml

from face_preprocess.config import load_qc_config
from face_preprocess.errors import ConfigError
from face_preprocess.qc.rules import evaluate_qc


def _fresh_output(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise ConfigError(f"QC output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _sample_records(run_dirs: Iterable[str | Path]) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for raw_root in run_dirs:
        root = Path(raw_root).resolve()
        sample_dir = root / "metadata" / "samples"
        if not sample_dir.is_dir():
            raise ConfigError(f"run has no sample metadata directory: {root}")
        for path in sorted(sample_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(f"invalid sample metadata: {path}") from exc
            records.append((root, payload))
    return records


def _scalar_rows(records: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root, record in records:
        for metric, raw in sorted(record.get("measurements", {}).items()):
            if not isinstance(raw, dict):
                continue
            value = raw.get("value")
            if isinstance(value, (list, dict)) or value is None:
                continue
            rows.append(
                {
                    "run_dir": str(root),
                    "output_key": record.get("output_key"),
                    "sample_id": record.get("sample_id"),
                    "status": record.get("status"),
                    "metric": metric,
                    "value": value,
                    "unit": raw.get("unit"),
                    "compute_status": raw.get("compute_status"),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def summarize_runs(run_dirs: list[str | Path], output: str | Path) -> dict[str, Any]:
    destination = _fresh_output(output)
    records = _sample_records(run_dirs)
    rows = _scalar_rows(records)
    _write_csv(destination / "measurements_long.csv", rows)
    status_counts: dict[str, int] = {}
    for _, record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema": "face-preprocess-qc-summary-v1",
        "run_count": len({str(root) for root, _ in records}),
        "sample_count": len(records),
        "measurement_row_count": len(rows),
        "status_counts": status_counts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def evaluate_run_qc(
    run_dir: str | Path,
    qc_config: str | Path,
    output: str | Path,
    *,
    qc_level: str = "standard",
) -> dict[str, Any]:
    destination = _fresh_output(output)
    config = load_qc_config(qc_config, qc_level)
    records = _sample_records([run_dir])
    counts = {"passed": 0, "warning": 0, "failed": 0}
    rows: list[dict[str, Any]] = []
    for _, record in records:
        hard = tuple(record.get("hard_failures", ()))
        assessment = evaluate_qc(
            record.get("measurements", {}),
            config,
            level=qc_level,
            hard_failures=hard,
        )
        counts[assessment.status] += 1
        rows.append(
            {
                "output_key": record.get("output_key"),
                "sample_id": record.get("sample_id"),
                "status": assessment.status,
                "qc_incomplete": assessment.qc_incomplete,
                "triggered_rules": [
                    item.rule_id for item in assessment.results if item.triggered
                ],
            }
        )
    payload = {
        "schema": "face-preprocess-qc-evaluation-v1",
        "source_run": str(Path(run_dir).resolve()),
        "qc_config": str(Path(qc_config).resolve()),
        "qc_level": qc_level,
        "sample_count": len(rows),
        "counts": counts,
        "results": rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def calibrate_qc(
    run_dirs: list[str | Path],
    qc_config: str | Path,
    validation_plan: str | Path,
    output: str | Path,
    labels: str | Path | None = None,
) -> dict[str, Any]:
    destination = _fresh_output(output)
    plan_path = Path(validation_plan).resolve()
    try:
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"invalid validation plan: {plan_path}") from exc
    required = {
        "schema_version",
        "plan_id",
        "sample_count_target",
        "sampling_seed",
        "stratify_by",
        "notes",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise ConfigError("validation plan fields do not match schema v1")
    target = int(plan["sample_count_target"])
    if target <= 0:
        raise ConfigError("sample_count_target must be positive")
    records = _sample_records(run_dirs)
    rng = random.Random(int(plan["sampling_seed"]))
    selected = records.copy()
    rng.shuffle(selected)
    selected = selected[:target]
    metric_values: dict[tuple[str, str], list[float]] = {}
    for _, record in selected:
        for metric, raw in record.get("measurements", {}).items():
            if not isinstance(raw, dict) or raw.get("compute_status") != "computed":
                continue
            value = raw.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metric_values.setdefault((metric, str(raw.get("unit", ""))), []).append(
                float(value)
            )
    distributions: list[dict[str, Any]] = []
    for (metric, unit), values in sorted(metric_values.items()):
        array = np.asarray(values, dtype=float)
        distributions.append(
            {
                "metric": metric,
                "unit": unit,
                "n": len(array),
                "min": float(np.min(array)),
                "p01": float(np.percentile(array, 1)),
                "p05": float(np.percentile(array, 5)),
                "median": float(np.median(array)),
                "p95": float(np.percentile(array, 95)),
                "p99": float(np.percentile(array, 99)),
                "max": float(np.max(array)),
            }
        )
    _write_csv(destination / "metric_distributions.csv", distributions)
    source_config = Path(qc_config).resolve()
    candidate = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    candidate["profile_id"] = f"{candidate['profile_id']}_candidate"
    candidate["profile_version"] = "candidate"
    candidate_path = destination / "candidate_qc.yaml"
    candidate_path.write_text(
        yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    labeled_performance: dict[str, Any] | None = None
    if labels is not None:
        label_path = Path(labels).resolve()
        with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
            label_rows = list(csv.DictReader(handle))
        if not label_rows:
            raise ConfigError("labels CSV is empty")
        label_field = (
            "consensus_label"
            if "consensus_label" in label_rows[0]
            else "label"
            if "label" in label_rows[0]
            else None
        )
        key_field = (
            "output_key"
            if "output_key" in label_rows[0]
            else "sample_id"
            if "sample_id" in label_rows[0]
            else None
        )
        if label_field is None or key_field is None:
            raise ConfigError(
                "labels CSV needs output_key or sample_id plus label or consensus_label"
            )
        labels_by_key: dict[str, str] = {}
        for row in label_rows:
            key = str(row.get(key_field, "")).strip()
            if not key:
                continue
            if key in labels_by_key:
                raise ConfigError(f"duplicate label key: {key}")
            labels_by_key[key] = str(row.get(label_field, "")).strip().lower()
        evaluation_config = load_qc_config(qc_config, "enhanced")
        tp = fp = tn = fn = 0
        performance_rows: list[dict[str, Any]] = []
        for _, record in selected:
            key = str(record.get(key_field, ""))
            label = labels_by_key.get(key)
            if label not in {"acceptable", "unacceptable"}:
                continue
            assessment = evaluate_qc(
                record.get("measurements", {}),
                evaluation_config,
                level="enhanced",
                hard_failures=tuple(record.get("hard_failures", ())),
            )
            predicted_abnormal = assessment.status in {"warning", "failed"}
            actual_abnormal = label == "unacceptable"
            if predicted_abnormal and actual_abnormal:
                tp += 1
            elif predicted_abnormal:
                fp += 1
            elif actual_abnormal:
                fn += 1
            else:
                tn += 1
            performance_rows.append(
                {
                    key_field: key,
                    "label": label,
                    "qc_status": assessment.status,
                    "predicted_abnormal": predicted_abnormal,
                }
            )
        _write_csv(destination / "labeled_predictions.csv", performance_rows)
        labeled_performance = {
            "evaluable_count": tp + fp + tn + fn,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "unacceptable_detection_rate": tp / (tp + fn) if tp + fn else None,
            "acceptable_warning_rate": fp / (fp + tn) if fp + tn else None,
        }
        (destination / "labeled_performance.json").write_text(
            json.dumps(
                labeled_performance,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    report = {
        "schema": "face-preprocess-qc-calibration-v1",
        "available_sample_count": len(records),
        "requested_sample_count": target,
        "selected_sample_count": len(selected),
        "candidate_config": str(candidate_path),
        "metric_count": len(distributions),
        "labeled_performance": labeled_performance,
        "note": "Candidate thresholds are not frozen automatically; edit and review the YAML manually.",
    }
    (destination / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
