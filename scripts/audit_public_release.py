#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable


ALLOWED_CHECKPOINTS = {
    "assets/models/crop_best.pt",
    "assets/models/landmark_best.pt",
}
ALLOWED_DATA_ASSETS = {
    "assets/template/0000.obj",
    "assets/template/landmarks_9_reference.csv",
    "assets/phenotype/lamda_ADNP.csv",
}
PROHIBITED_SEGMENTS = {
    "build",
    "dist",
    "outputs",
    "output",
    "verification",
    "sever_files",
    "pilot_manual_001",
    "__pycache__",
    ".pytest_cache",
    "training",
    "predictions",
}
PROHIBITED_SUFFIXES = {".npy", ".npz", ".xlsx", ".xls"}
CONTENT_SCAN_EXEMPT = {
    "scripts/audit_public_release.py",
    "tests/test_public_release_audit.py",
}

WINDOWS_PATH = re.compile(
    r"(?i)\b[A-Z]:\\(?!path\\to(?:\\|\b)|Program Files(?:\\|\b))[^\s`\"']+"
)
UNIX_HOME = re.compile(r"/home/(?!user(?:/|\b)|path(?:/|\b))[^/\s]+")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<![0-9A-Fa-f])1[3-9][0-9]{9}(?![0-9A-Fa-f])")
CREDENTIAL = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|password|passwd|secret)\s*[:=]\s*[\"']?([^\s\"']+)"
)
PRIVATE_PROJECT_TERM = re.compile(r"(?i)\bFDCH\b|co-xuqiong")
SUSPICIOUS_CHECKPOINT_KEYS = re.compile(
    r"(?i)(?:source|data|dataset|manifest|subject|patient|individual).*(?:path|file|id)|"
    r"(?:path|file).*(?:source|data|dataset|manifest|subject|patient|individual)"
)


def _tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {message}")
    return sorted(
        item.decode("utf-8", errors="strict").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    )


def _finding(path: str, rule: str, detail: str) -> dict[str, str]:
    return {"path": path, "rule": rule, "detail": detail}


def _path_findings(relative: str) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    path = Path(relative)
    lower_parts = {part.casefold() for part in path.parts}
    prohibited = sorted(lower_parts & PROHIBITED_SEGMENTS)
    if prohibited or any(part.casefold().endswith(".egg-info") for part in path.parts):
        detail = prohibited[0] if prohibited else "generated egg-info"
        findings.append(_finding(relative, "prohibited_path", detail))
    suffix = path.suffix.casefold()
    if suffix == ".pt" and relative not in ALLOWED_CHECKPOINTS:
        findings.append(
            _finding(relative, "unexpected_checkpoint", "only selected inference checkpoints are allowed")
        )
    if suffix in PROHIBITED_SUFFIXES:
        findings.append(_finding(relative, "prohibited_data_file", suffix))
    if suffix in {".obj", ".csv"} and relative not in ALLOWED_DATA_ASSETS:
        findings.append(
            _finding(relative, "prohibited_data_file", "OBJ/CSV is not an allowlisted scientific asset")
        )
    return findings


def _text_findings(relative: str, text: str) -> list[dict[str, str]]:
    if relative in CONTENT_SCAN_EXEMPT:
        return []
    checks = (
        ("windows_absolute_path", WINDOWS_PATH),
        ("unix_home_path", UNIX_HOME),
        ("email_address", EMAIL),
        ("phone_number", PHONE),
        ("credential_value", CREDENTIAL),
        ("private_project_term", PRIVATE_PROJECT_TERM),
    )
    findings: list[dict[str, str]] = []
    for rule, pattern in checks:
        match = pattern.search(text)
        if match:
            findings.append(_finding(relative, rule, match.group(0)[:160]))
    return findings


def _read_text(path: Path) -> str | None:
    if path.stat().st_size > 5 * 1024 * 1024:
        return None
    payload = path.read_bytes()
    if b"\0" in payload:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _walk_checkpoint_metadata(value: Any, trail: tuple[str, ...] = ()) -> Iterable[str]:
    try:
        import torch
    except ImportError:  # pragma: no cover - checkpoint scan reports import failure earlier
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child = (*trail, key_text)
            if SUSPICIOUS_CHECKPOINT_KEYS.search(key_text):
                yield ".".join(child)
            yield from _walk_checkpoint_metadata(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_checkpoint_metadata(item, (*trail, str(index)))
        return
    if isinstance(value, str):
        if (
            WINDOWS_PATH.search(value)
            or UNIX_HOME.search(value)
            or EMAIL.search(value)
            or PHONE.search(value)
            or PRIVATE_PROJECT_TERM.search(value)
        ):
            yield ".".join(trail) or "<root>"


def _checkpoint_findings(root: Path, relative: str) -> list[dict[str, str]]:
    try:
        import torch
    except ImportError as exc:
        return [_finding(relative, "checkpoint_load_error", str(exc))]
    try:
        payload = torch.load(root / relative, map_location="cpu", weights_only=True)
    except Exception as exc:
        return [
            _finding(
                relative,
                "checkpoint_load_error",
                f"{type(exc).__name__}: {exc}",
            )
        ]
    trails = sorted(set(_walk_checkpoint_metadata(payload)))
    return [
        _finding(relative, "checkpoint_private_metadata", trail)
        for trail in trails
    ]


def audit_repository(root: str | Path) -> dict[str, Any]:
    resolved = Path(root).resolve()
    tracked = _tracked_files(resolved)
    findings: list[dict[str, str]] = []
    checkpoint_count = 0
    for relative in tracked:
        findings.extend(_path_findings(relative))
        path = resolved / Path(relative)
        if relative in ALLOWED_CHECKPOINTS:
            checkpoint_count += 1
            findings.extend(_checkpoint_findings(resolved, relative))
            continue
        text = _read_text(path)
        if text is not None:
            findings.extend(_text_findings(relative, text))
    findings.sort(key=lambda item: (item["path"], item["rule"], item["detail"]))
    return {
        "status": "ok" if not findings else "blocked",
        "files_scanned": len(tracked),
        "checkpoints_scanned": checkpoint_count,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a Git index for public-release privacy risks.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    report = audit_repository(args.root)
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(f"status: {report['status']}")
        print(f"files_scanned: {report['files_scanned']}")
        print(f"checkpoints_scanned: {report['checkpoints_scanned']}")
        for finding in report["findings"]:
            print(f"{finding['rule']}: {finding['path']}: {finding['detail']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
