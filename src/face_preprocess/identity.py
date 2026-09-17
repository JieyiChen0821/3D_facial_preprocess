from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import re
import unicodedata
import uuid
from typing import Sequence

from face_preprocess.types import InputRecord


OUTPUT_KEY_NAMESPACE = uuid.UUID("5c32e665-9f41-5f9f-8424-aa33ad97f171")
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_REPEATED_UNDERSCORES = re.compile(r"_+")


def sanitize_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    cleaned = _UNSAFE_FILENAME.sub("_", normalized)
    cleaned = _REPEATED_UNDERSCORES.sub("_", cleaned).strip(" ._")
    return cleaned or "sample"


def assign_output_keys(records: Sequence[InputRecord]) -> list[InputRecord]:
    stems = [sanitize_stem(record.path.stem) for record in records]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, stem in enumerate(stems):
        groups[stem.casefold()].append(index)

    resolved: list[InputRecord] = []
    for index, record in enumerate(records):
        stem = stems[index]
        if len(groups[stem.casefold()]) == 1:
            key = stem
        else:
            identity = f"{unicodedata.normalize('NFC', record.relative_path)}|{record.geometry_sha256}"
            key = f"{stem}__{uuid.uuid5(OUTPUT_KEY_NAMESPACE, identity)}"
        resolved.append(replace(record, output_key=key))
    return resolved

