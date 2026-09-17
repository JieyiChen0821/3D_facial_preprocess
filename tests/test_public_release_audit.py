from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import torch


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_public_release", ROOT / "scripts" / "audit_public_release.py"
)
assert SPEC is not None and SPEC.loader is not None
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)
audit_repository = audit_module.audit_repository


def _tracked_repo(tmp_path: Path, files: dict[str, bytes | str]) -> Path:
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def _safe_checkpoint(path: Path, **metadata: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {"weight": torch.tensor([1.0])},
            "model": "pointnext",
            **metadata,
        },
        path,
    )


def test_audit_allows_selected_assets_and_generic_documentation_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "assets" / "models" / "crop_best.pt"
    _safe_checkpoint(checkpoint)
    repo = _tracked_repo(
        tmp_path,
        {
            "README.md": (
                "Use /path/to/input.obj or C:\\path\\to\\input.obj.\n"
                "Git Bash may be installed under C:\\Program Files\\Git.\n"
            ),
            "assets/template/0000.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            "assets/phenotype/lamda_ADNP.csv": "0,0.2,0.3,0.5\n",
        },
    )
    subprocess.run(["git", "add", "assets/models/crop_best.pt"], cwd=repo, check=True)

    report = audit_repository(repo)

    assert report["status"] == "ok"
    assert report["findings"] == []
    assert report["checkpoints_scanned"] == 1


def test_audit_reports_prohibited_paths_private_text_and_extra_checkpoint(
    tmp_path: Path,
) -> None:
    _safe_checkpoint(tmp_path / "models" / "historical.pt")
    repo = _tracked_repo(
        tmp_path,
        {
            "sever_files/predictions.csv": "sample_id,name\n001,Person Name\n",
            "notes.txt": (
                "source=D:\\private\\subject01.obj\n"
                "email=person@hospital.example\n"
                "phone=13812345678\n"
                "api_key=real-secret-value\n"
                "project=F:\\FDCH\\co-xuqiong\\study\n"
            ),
        },
    )
    subprocess.run(["git", "add", "models/historical.pt"], cwd=repo, check=True)

    report = audit_repository(repo)
    rules = {finding["rule"] for finding in report["findings"]}

    assert report["status"] == "blocked"
    assert "prohibited_path" in rules
    assert "unexpected_checkpoint" in rules
    assert "windows_absolute_path" in rules
    assert "email_address" in rules
    assert "phone_number" in rules
    assert "credential_value" in rules
    assert "private_project_term" in rules


def test_audit_inspects_checkpoint_string_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "assets" / "models" / "crop_best.pt"
    _safe_checkpoint(checkpoint, source_path=r"D:\training\subject_manifest.csv")
    repo = _tracked_repo(tmp_path, {})
    subprocess.run(["git", "add", "assets/models/crop_best.pt"], cwd=repo, check=True)

    report = audit_repository(repo)

    checkpoint_findings = [
        finding
        for finding in report["findings"]
        if finding["rule"] == "checkpoint_private_metadata"
    ]
    assert report["status"] == "blocked"
    assert checkpoint_findings
    assert checkpoint_findings[0]["path"] == "assets/models/crop_best.pt"
