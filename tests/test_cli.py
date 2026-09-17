from __future__ import annotations

import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET


EXPECTED_COMMANDS = (
    "doctor",
    "adapter-info",
    "validate-run",
    "hash-assets",
    "phenotype-landmarks",
    "adapter-info",
    "run",
    "qc-summarize",
    "qc-render",
    "qc-calibrate",
    "qc-evaluate",
    "rebuild-summary",
)


def test_help_lists_stable_commands(run_cli):
    result = run_cli(["--help"])

    assert result.returncode == 0, result.stderr
    for command in EXPECTED_COMMANDS:
        assert command in result.stdout


def test_version_does_not_require_torch(run_cli):
    result = run_cli(["--version"])

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith("face-preprocess ")


def test_json_doctor_reports_missing_torch_without_crashing(run_cli):
    result = run_cli(["--json", "doctor"])

    assert result.returncode == 0, result.stderr
    assert '"command": "doctor"' in result.stdout
    assert '"torch"' in result.stdout


def test_retry_failed_requires_resume(run_cli):
    result = run_cli(["run", "--retry-failed"])

    assert result.returncode == 2
    assert "--retry-failed requires --resume" in result.stderr


def test_run_help_lists_optional_phenotype_flag(run_cli):
    result = run_cli(["run", "--help"])

    assert result.returncode == 0, result.stderr
    assert "--phenotype-landmarks" in result.stdout


def test_phenotype_landmarks_command_writes_csv_and_pp(run_cli, tmp_path: Path):
    root = Path(__file__).parents[1]
    output = tmp_path / "phenotype"

    result = run_cli(
        [
            "--json",
            "phenotype-landmarks",
            "--input",
            str(root / "assets" / "template" / "0000.obj"),
            "--output",
            str(output),
            "--config",
            str(root / "configs" / "pipeline_server.yaml"),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "phenotype-landmarks"
    assert payload["status"] == "ok"
    assert payload["processed"] == 1
    csv_path = output / "0000_landmarks.csv"
    pp_path = output / "0000_landmarks.pp"
    assert csv_path.is_file()
    assert pp_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 33
    assert len(ET.parse(pp_path).getroot().findall("point")) == 32


def test_phenotype_landmarks_command_can_omit_pp(run_cli, tmp_path: Path):
    root = Path(__file__).parents[1]
    output = tmp_path / "phenotype"

    result = run_cli(
        [
            "phenotype-landmarks",
            "--input",
            str(root / "assets" / "template" / "0000.obj"),
            "--output",
            str(output),
            "--config",
            str(root / "configs" / "pipeline_server.yaml"),
            "--no-pp",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert (output / "0000_landmarks.csv").is_file()
    assert not (output / "0000_landmarks.pp").exists()


def test_phenotype_landmarks_command_rejects_wrong_topology(run_cli, tmp_path: Path):
    root = Path(__file__).parents[1]
    input_obj = tmp_path / "wrong.obj"
    input_obj.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )

    result = run_cli(
        [
            "phenotype-landmarks",
            "--input",
            str(input_obj),
            "--output",
            str(tmp_path / "output"),
            "--config",
            str(root / "configs" / "pipeline_server.yaml"),
        ]
    )

    assert result.returncode != 0
    assert "topology" in result.stderr.lower()
