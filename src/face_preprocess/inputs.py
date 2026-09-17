from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path
import unicodedata

from face_preprocess.errors import ManifestError, ObjFormatError
from face_preprocess.hashing import geometry_sha256, source_file_sha256
from face_preprocess.identity import assign_output_keys
from face_preprocess.obj_io import read_obj
from face_preprocess.types import InputRecord


KNOWN_MANIFEST_COLUMNS = {"input_obj", "sample_id", "subject_group", "batch_id"}


def _sort_key(record: InputRecord) -> tuple[str, str, str]:
    name = unicodedata.normalize("NFC", record.path.name)
    return name.casefold(), name, record.geometry_sha256


def _build_record(
    path: Path,
    relative_path: str,
    sample_id: str | None = None,
    subject_group: str | None = None,
    batch_id: str | None = None,
    user_metadata: dict[str, str] | None = None,
) -> InputRecord:
    try:
        file_hash = source_file_sha256(path)
    except OSError as exc:
        normalized = unicodedata.normalize("NFC", str(path.resolve()))
        file_hash = hashlib.sha256(f"unreadable:{normalized}".encode("utf-8")).hexdigest()
        return InputRecord(
            path=path.resolve(),
            sample_id=(sample_id or path.stem).strip() or path.stem,
            source_file_sha256=file_hash,
            geometry_sha256=hashlib.sha256(
                f"invalid-geometry:{file_hash}".encode("ascii")
            ).hexdigest(),
            relative_path=unicodedata.normalize("NFC", relative_path),
            subject_group=subject_group or None,
            batch_id=batch_id or None,
            user_metadata=user_metadata or {},
            input_error=f"{type(exc).__name__}: {exc}",
        )
    try:
        mesh, inventory = read_obj(path)
    except (ObjFormatError, OSError) as exc:
        return InputRecord(
            path=path.resolve(),
            sample_id=(sample_id or path.stem).strip() or path.stem,
            source_file_sha256=file_hash,
            geometry_sha256=hashlib.sha256(
                f"invalid-geometry:{file_hash}".encode("ascii")
            ).hexdigest(),
            relative_path=unicodedata.normalize("NFC", relative_path),
            subject_group=subject_group or None,
            batch_id=batch_id or None,
            user_metadata=user_metadata or {},
            input_error=f"{type(exc).__name__}: {exc}",
        )
    return InputRecord(
        path=path.resolve(),
        sample_id=(sample_id or path.stem).strip() or path.stem,
        source_file_sha256=file_hash,
        geometry_sha256=geometry_sha256(mesh),
        relative_path=unicodedata.normalize("NFC", relative_path),
        subject_group=subject_group or None,
        batch_id=batch_id or None,
        user_metadata=user_metadata or {},
        obj_inventory=inventory,
    )


def _discover_path(input_path: Path) -> list[InputRecord]:
    resolved = input_path.resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".obj":
            raise ManifestError(f"input file is not an OBJ: {resolved}")
        return [_build_record(resolved, resolved.name)]
    if not resolved.is_dir():
        raise ManifestError(f"input path does not exist: {resolved}")
    files = [item for item in resolved.iterdir() if item.is_file() and item.suffix.lower() == ".obj"]
    return [_build_record(item, item.name) for item in files]


def _discover_manifest(manifest: Path) -> list[InputRecord]:
    source = manifest.resolve()
    if not source.is_file():
        raise ManifestError(f"input manifest does not exist: {source}")
    records: list[InputRecord] = []
    seen_paths: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "input_obj" not in reader.fieldnames:
            raise ManifestError("input manifest must contain input_obj")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("input_obj") or "").strip()
            if not raw_path:
                raise ManifestError(f"empty input_obj at manifest row {row_number}")
            path = (source.parent / raw_path).resolve()
            duplicate_key = unicodedata.normalize("NFC", str(path)).casefold()
            if duplicate_key in seen_paths:
                raise ManifestError(f"duplicate input path in manifest: {path}")
            seen_paths.add(duplicate_key)
            extras = {
                key: value
                for key, value in row.items()
                if key not in KNOWN_MANIFEST_COLUMNS and key is not None and value is not None
            }
            records.append(
                _build_record(
                    path=path,
                    relative_path=raw_path,
                    sample_id=row.get("sample_id"),
                    subject_group=(row.get("subject_group") or "").strip() or None,
                    batch_id=(row.get("batch_id") or "").strip() or None,
                    user_metadata=extras,
                )
            )
    return records


def discover_inputs(
    input_path: str | Path | None = None,
    input_manifest: str | Path | None = None,
) -> list[InputRecord]:
    if (input_path is None) == (input_manifest is None):
        raise ManifestError("exactly one of input_path or input_manifest is required")
    records = (
        _discover_path(Path(input_path))
        if input_path is not None
        else _discover_manifest(Path(input_manifest))
    )
    if not records:
        raise ManifestError("no OBJ files found")
    ordered = sorted(records, key=_sort_key)
    geometry_counts = Counter(record.geometry_sha256 for record in ordered)
    annotated = [
        replace(
            record,
            duplicate_geometry_group_size=geometry_counts[record.geometry_sha256],
        )
        for record in ordered
    ]
    return assign_output_keys(annotated)
