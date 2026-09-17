from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
BOOTSTRAP = ROOT / "scripts" / "bootstrap_server.sh"
VERIFY_SERVER = ROOT / "scripts" / "verify_server.sh"
SLURM_FILES = (
    ROOT / "slurm" / "doctor.sbatch",
    ROOT / "slurm" / "run_face_preprocess.sbatch",
)


def _bash_executable() -> str:
    discovered = shutil.which("bash")
    if discovered:
        return discovered
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
        if git_bash.is_file():
            return str(git_bash)
    pytest.skip("Bash is unavailable")


def _run_bootstrap(tmp_path: Path, exit_code: int) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    command_log = tmp_path / "mamba-commands.txt"
    fake_mamba = fake_bin / "mamba"
    fake_mamba.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "${FAKE_CONDA_LOG}"\n'
        'exit "${FAKE_CONDA_EXIT}"\n',
        encoding="utf-8",
        newline="\n",
    )
    fake_mamba.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["FAKE_CONDA_LOG"] = str(command_log)
    env["FAKE_CONDA_EXIT"] = str(exit_code)
    return subprocess.run(
        [_bash_executable(), str(BOOTSTRAP), "cpu", "face-preprocess-test"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_server_shell_files_parse_with_bash() -> None:
    bash = _bash_executable()
    for path in (BOOTSTRAP, *SLURM_FILES):
        result = subprocess.run(
            [bash, "-n", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_server_bootstrap_stops_when_conda_fails(tmp_path: Path) -> None:
    result = _run_bootstrap(tmp_path, exit_code=7)
    assert result.returncode == 7
    assert "is ready" not in result.stdout


def test_server_bootstrap_runs_all_installation_checks(tmp_path: Path) -> None:
    result = _run_bootstrap(tmp_path, exit_code=0)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Environment face-preprocess-test is ready." in result.stdout

    commands = (tmp_path / "mamba-commands.txt").read_text(encoding="utf-8").splitlines()
    assert len(commands) == 4
    assert commands[0] == "create -y -n face-preprocess-test python=3.10 pip=24.3"
    assert "requirements-linux-cpu.lock.txt" in commands[1]
    assert "python -m pip install --no-deps -e" in commands[2]
    assert commands[3] == "run -n face-preprocess-test face-preprocess --json doctor"


def test_server_acceptance_runs_strict_gates_and_full_sample(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    command_log = tmp_path / "mamba-commands.txt"
    fake_mamba = fake_bin / "mamba"
    fake_mamba.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "${FAKE_CONDA_LOG}"\n'
        "exit 0\n",
        encoding="utf-8",
        newline="\n",
    )
    fake_mamba.chmod(0o755)
    input_obj = tmp_path / "raw.obj"
    input_obj.write_text("v 0 0 0\n", encoding="utf-8", newline="\n")
    output = tmp_path / "acceptance-output"

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["FAKE_CONDA_LOG"] = str(command_log)
    result = subprocess.run(
        [
            _bash_executable(),
            str(VERIFY_SERVER),
            str(input_obj),
            str(output),
            "face-preprocess-test",
            "cpu",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Server acceptance passed." in result.stdout
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert len(commands) == 4
    assert "face-preprocess --json doctor" in commands[0]
    assert "--device cpu" in commands[0]
    assert commands[1] == "run -n face-preprocess-test python -m pytest -m model_smoke -q"
    assert "face-preprocess --json validate-run" in commands[2]
    assert "face-preprocess --json run" in commands[3]
    assert "--qc-level standard" in commands[3]
    assert "--output-mode full" in commands[3]
