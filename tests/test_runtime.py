from __future__ import annotations

import json
from pathlib import Path

import pytest

from face_preprocess.errors import ConfigError
from face_preprocess.runtime import RunLayout, classify_resume_record


FINGERPRINTS = {
    "input_set": "1" * 64,
    "scientific": "2" * 64,
    "execution": "3" * 64,
}


def test_sample_transaction_commits_metadata_last(tmp_path: Path):
    layout = RunLayout.prepare(tmp_path / "run", FINGERPRINTS)
    transaction = layout.begin_sample("face", "attempt-1")
    transaction.stage_bytes("final_obj/face.obj", b"v 0.0000000000 0.0000000000 0.0000000000\n")

    transaction.commit({"status": "pass", "sample_id": "display"})

    assert (layout.root / "final_obj" / "face.obj").exists()
    metadata = json.loads((layout.root / "metadata" / "samples" / "face.json").read_text(encoding="utf-8"))
    assert metadata["commit_state"] == "complete"
    assert metadata["attempt_id"] == "attempt-1"


def test_resume_rejects_any_fingerprint_mismatch(tmp_path: Path):
    output = tmp_path / "run"
    RunLayout.prepare(output, FINGERPRINTS)
    changed = {**FINGERPRINTS, "execution": "4" * 64}

    with pytest.raises(ConfigError, match="resume fingerprint mismatch"):
        RunLayout.prepare(output, changed, resume=True)


def test_overwrite_archives_only_valid_pipeline_run(tmp_path: Path):
    output = tmp_path / "run"
    RunLayout.prepare(output, FINGERPRINTS)

    fresh = RunLayout.prepare(output, FINGERPRINTS, overwrite=True)

    assert fresh.root == output
    archived = list(tmp_path.glob("run.superseded.*"))
    assert len(archived) == 1
    assert (archived[0] / "run_manifest.json").exists()


def test_overwrite_refuses_arbitrary_nonempty_directory(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ConfigError, match="not a valid face-preprocess run"):
        RunLayout.prepare(output, FINGERPRINTS, overwrite=True)


@pytest.mark.parametrize(
    "status, retry_failed, expected",
    (
        ("pass", False, "skip"),
        ("warning", False, "skip"),
        ("failed", False, "skip"),
        ("failed", True, "retry"),
    ),
)
def test_resume_terminal_status_policy(status: str, retry_failed: bool, expected: str):
    metadata = {
        "commit_state": "complete",
        "status": status,
        "fingerprints": FINGERPRINTS,
    }

    assert classify_resume_record(metadata, FINGERPRINTS, retry_failed=retry_failed) == expected


def test_incomplete_record_is_recomputed():
    metadata = {
        "commit_state": "incomplete",
        "status": "failed",
        "fingerprints": FINGERPRINTS,
    }

    assert classify_resume_record(metadata, FINGERPRINTS, retry_failed=False) == "retry"


def test_failed_transaction_never_promotes_staged_final(tmp_path: Path):
    layout = RunLayout.prepare(tmp_path / "run", FINGERPRINTS)
    transaction = layout.begin_sample("face", "attempt-failed")
    transaction.stage_bytes("final_obj/face.obj", b"partial")

    transaction.fail({"status": "failed"})

    assert not (layout.root / "final_obj" / "face.obj").exists()
    assert (
        layout.root / "incomplete" / "face" / "attempt-failed" / "final_obj" / "face.obj"
    ).exists()
