from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import numpy as np

from face_preprocess.errors import ConfigError


RUN_MANIFEST_SCHEMA = "face-preprocess-run-v1"
RUN_DIRECTORIES = (
    "configs",
    "metadata/samples",
    "metadata/attempts",
    "final_obj",
    "artifacts",
    "qc/evaluations",
    "logs",
    "incomplete",
    ".staging",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "run_manifest.json"
    if not path.is_file():
        raise ConfigError(f"output directory is not a valid face-preprocess run: {root}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"output directory is not a valid face-preprocess run: {root}") from exc
    if payload.get("schema") != RUN_MANIFEST_SCHEMA or payload.get("tool") != "face-preprocess":
        raise ConfigError(f"output directory is not a valid face-preprocess run: {root}")
    return payload


def _same_fingerprints(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    return dict(left) == dict(right)


def classify_resume_record(
    metadata: Mapping[str, Any],
    fingerprints: Mapping[str, str],
    retry_failed: bool,
) -> str:
    existing = metadata.get("fingerprints")
    if not isinstance(existing, dict) or not _same_fingerprints(existing, fingerprints):
        raise ConfigError("resume fingerprint mismatch for existing sample")
    if metadata.get("commit_state") != "complete":
        return "retry"
    if metadata.get("status") == "failed" and retry_failed:
        return "retry"
    return "skip"


@dataclass(frozen=True)
class RunLayout:
    root: Path
    run_id: str
    fingerprints: dict[str, str]

    @classmethod
    def prepare(
        cls,
        output: str | Path,
        fingerprints: Mapping[str, str],
        resume: bool = False,
        overwrite: bool = False,
    ) -> "RunLayout":
        if resume and overwrite:
            raise ConfigError("--resume and --overwrite are mutually exclusive")
        root = Path(output).resolve()
        if root.exists() and any(root.iterdir()):
            if resume:
                manifest = _read_manifest(root)
                existing = manifest.get("fingerprints")
                if not isinstance(existing, dict) or not _same_fingerprints(existing, fingerprints):
                    raise ConfigError("resume fingerprint mismatch for output directory")
                return cls(root=root, run_id=str(manifest["run_id"]), fingerprints=dict(fingerprints))
            if overwrite:
                _read_manifest(root)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archived = root.with_name(f"{root.name}.superseded.{stamp}")
                root.rename(archived)
            else:
                raise ConfigError("non-empty output directory requires --resume or --overwrite")
        root.mkdir(parents=True, exist_ok=True)
        for relative in RUN_DIRECTORIES:
            (root / relative).mkdir(parents=True, exist_ok=True)
        run_id = str(uuid.uuid4())
        manifest = {
            "schema": RUN_MANIFEST_SCHEMA,
            "tool": "face-preprocess",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fingerprints": dict(fingerprints),
        }
        _atomic_json(root / "run_manifest.json", manifest)
        return cls(root=root, run_id=run_id, fingerprints=dict(fingerprints))

    def begin_sample(self, output_key: str, attempt_id: str | None = None) -> "SampleTransaction":
        return SampleTransaction(
            layout=self,
            output_key=output_key,
            attempt_id=attempt_id or str(uuid.uuid4()),
        )


@dataclass
class SampleTransaction:
    layout: RunLayout
    output_key: str
    attempt_id: str

    @property
    def staging_root(self) -> Path:
        return self.layout.root / ".staging" / self.output_key / self.attempt_id

    def _safe_relative(self, relative: str | Path) -> Path:
        value = Path(relative)
        if value.is_absolute() or ".." in value.parts:
            raise ConfigError(f"unsafe staged path: {relative}")
        return value

    def stage_bytes(self, relative: str | Path, payload: bytes) -> Path:
        safe = self._safe_relative(relative)
        target = self.staging_root / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return target

    def stage_path(self, relative: str | Path) -> Path:
        safe = self._safe_relative(relative)
        target = self.staging_root / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _commit_metadata(self, metadata: Mapping[str, Any]) -> None:
        committed = dict(metadata)
        committed.update(
            {
                "commit_state": "complete",
                "output_key": self.output_key,
                "attempt_id": self.attempt_id,
                "run_id": self.layout.run_id,
                "fingerprints": dict(self.layout.fingerprints),
                "committed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        attempt_path = (
            self.layout.root
            / "metadata"
            / "attempts"
            / self.output_key
            / f"{self.attempt_id}.json"
        )
        _atomic_json(attempt_path, committed)
        _atomic_json(
            self.layout.root / "metadata" / "samples" / f"{self.output_key}.json",
            committed,
        )

    def preserve_incomplete(self) -> Path | None:
        if not self.staging_root.exists():
            return None
        destination = (
            self.layout.root / "incomplete" / self.output_key / self.attempt_id
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ConfigError(f"incomplete attempt already exists: {destination}")
        self.staging_root.rename(destination)
        return destination

    def fail(self, metadata: Mapping[str, Any]) -> None:
        incomplete = self.preserve_incomplete()
        payload = dict(metadata)
        if incomplete is not None:
            payload["incomplete_artifacts"] = str(incomplete.relative_to(self.layout.root))
        self._commit_metadata(payload)

    def commit(self, metadata: Mapping[str, Any]) -> None:
        if self.staging_root.exists():
            staged_files = sorted(path for path in self.staging_root.rglob("*") if path.is_file())
            for staged in staged_files:
                relative = staged.relative_to(self.staging_root)
                target = self.layout.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
        self._commit_metadata(metadata)
        if self.staging_root.exists():
            shutil.rmtree(self.staging_root)
