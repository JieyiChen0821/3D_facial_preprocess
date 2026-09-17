from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "bootstrap_windows.ps1"


def _writable_drive_with_space() -> Path:
    drive = Path(tempfile.gettempdir()).resolve()
    try:
        if shutil.disk_usage(drive).free >= 3 * 1024**3:
            probe = Path(tempfile.mkdtemp(prefix="face-preprocess-probe-", dir=drive))
            probe.rmdir()
            return drive
    except OSError:
        pass
    pytest.skip("The system temporary drive is not writable with at least 3 GB free")


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell bootstrap only")
def test_bootstrap_stops_when_conda_returns_nonzero(tmp_path: Path) -> None:
    drive = _writable_drive_with_space()
    runtime_root = Path(tempfile.mkdtemp(prefix="face-preprocess-bootstrap-", dir=drive))
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_conda = fake_bin / "mamba.cmd"
    fake_conda.write_text("@echo off\r\nexit /b 7\r\n", encoding="ascii")

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-Backend",
                "cpu",
                "-EnvironmentPath",
                str(runtime_root / "env"),
                "-CacheRoot",
                str(runtime_root / "cache"),
                "-CondaExecutable",
                str(fake_conda),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Conda command failed with exit code 7" in output
    assert "is ready" not in output


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell bootstrap only")
def test_bootstrap_installs_project_without_editable_path_file(tmp_path: Path) -> None:
    drive = _writable_drive_with_space()
    runtime_root = Path(tempfile.mkdtemp(prefix="face-preprocess-bootstrap-", dir=drive))
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    command_log = tmp_path / "mamba-commands.txt"
    fake_conda = fake_bin / "mamba.cmd"
    fake_conda.write_text(
        '@echo off\r\necho %*>>"%FAKE_CONDA_LOG%"\r\nexit /b 0\r\n',
        encoding="ascii",
    )

    env = os.environ.copy()
    env["FAKE_CONDA_LOG"] = str(command_log)
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-Backend",
                "cpu",
                "-EnvironmentPath",
                str(runtime_root / "env"),
                "-CacheRoot",
                str(runtime_root / "cache"),
                "-CondaExecutable",
                str(fake_conda),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)

    assert result.returncode == 0, result.stdout + result.stderr
    commands = command_log.read_text(encoding="utf-8")
    project_install = next(
        line
        for line in commands.splitlines()
        if "pip install --no-deps" in line
    )
    assert " -e " not in f" {project_install} "
