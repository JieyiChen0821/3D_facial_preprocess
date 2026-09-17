from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from face_preprocess.errors import ArtifactError, ConfigError


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
QC_OPERATORS = frozenset({"gt", "gte", "lt", "lte", "inside", "outside", "equals"})
QC_GROUP_MODES = frozenset({"disabled", "record_only", "warning"})


@dataclass(frozen=True)
class ArtifactIdentity:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int
    profile_id: str
    profile_version: str
    stage: str
    adapter: str
    adapter_version: str
    adapter_source_sha256: str
    assets: dict[str, ArtifactIdentity]
    base_seed: int
    precision: str
    parameters: dict[str, Any]

    @property
    def checkpoint(self) -> Path:
        return self.assets["checkpoint"].path

    @property
    def sha256(self) -> str:
        return self.assets["checkpoint"].sha256

    @property
    def num_points(self) -> int:
        return int(self.parameters["input"]["sampling"]["num_points"])

    @property
    def top_k(self) -> int | None:
        value = self.parameters.get("inference", {}).get("top_k")
        return int(value) if value is not None else None


@dataclass(frozen=True)
class GeometryConfig:
    template_obj: Path
    template_obj_sha256: str
    template_landmarks: Path
    template_landmarks_sha256: str
    template_points: Path
    template_points_sha256: str
    refindex: Path
    refindex_sha256: str
    expected_template_vertices: int
    expected_template_faces: int


@dataclass(frozen=True)
class PhenotypeConfig:
    template_obj: Path
    template_obj_sha256: str
    lamda_csv: Path
    lamda_csv_sha256: str
    names_pp: Path
    names_pp_sha256: str


@dataclass(frozen=True)
class ProcessingConfig:
    symmetry_enabled: bool
    texture_mode: str = "geometry_only"
    meshmonk: dict[str, Any] | None = None


@dataclass(frozen=True)
class DeviceProfile:
    schema_version: int
    profile_id: str
    profile_version: str
    coordinate_unit: str
    expected_edge_length_mm: float
    acquisition_device: str
    acquisition_protocol: str


@dataclass(frozen=True)
class QCRule:
    rule_id: str
    metric: str
    operator: str
    threshold: Any
    unit: str
    severity: str
    minimum_qc_level: str


@dataclass(frozen=True)
class QCConfig:
    schema_version: int
    profile_id: str
    profile_version: str
    metric_groups: dict[str, str]
    rules: tuple[QCRule, ...]


@dataclass(frozen=True)
class ResolvedRunConfig:
    schema_version: int
    profile_id: str
    profile_version: str
    crop: ModelConfig
    landmark: ModelConfig
    geometry: GeometryConfig
    processing: ProcessingConfig
    device: DeviceProfile
    qc: QCConfig
    symmetry_enabled: bool
    config_hashes: dict[str, str]
    source_paths: dict[str, Path]
    phenotype: PhenotypeConfig | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(path: str | Path, expected_sha256: str) -> ArtifactIdentity:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ArtifactError(f"artifact file does not exist: {resolved}")
    if not SHA256_PATTERN.fullmatch(str(expected_sha256)):
        raise ArtifactError(f"invalid expected SHA256 for {resolved}")
    actual = _sha256(resolved)
    if actual != expected_sha256:
        raise ArtifactError(f"SHA256 mismatch for {resolved}: expected {expected_sha256}, got {actual}")
    return ArtifactIdentity(path=resolved, sha256=actual, size_bytes=resolved.stat().st_size)


def _load_yaml(path: str | Path, label: str) -> tuple[Path, dict[str, Any], str]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigError(f"{label} configuration does not exist: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{label} configuration must be a mapping: {source}")
    return source, payload, _sha256(source)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _strict_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    required: set[str],
    label: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(required - set(payload))
    if unknown:
        raise ConfigError(f"{label} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"{label} is missing fields: {', '.join(missing)}")


def _identity(payload: Mapping[str, Any], label: str) -> tuple[int, str, str]:
    schema_version = int(payload["schema_version"])
    if schema_version != 1:
        raise ConfigError(f"{label} schema_version must be 1")
    profile_id = str(payload["profile_id"]).strip()
    profile_version = str(payload["profile_version"]).strip()
    if not profile_id or not profile_version:
        raise ConfigError(f"{label} profile_id and profile_version must be non-empty")
    return schema_version, profile_id, profile_version


def _resolve_asset(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _model_profile_config(path: Path, expected_stage: str) -> tuple[ModelConfig, Path, str]:
    source, payload, profile_hash = _load_yaml(path, f"{expected_stage} model profile")
    fields = {
        "schema_version",
        "profile_id",
        "profile_version",
        "stage",
        "adapter",
        "assets",
        "common",
        "parameters",
    }
    _strict_keys(payload, fields, fields, f"{expected_stage} model profile")
    schema, profile_id, profile_version = _identity(payload, f"{expected_stage} model profile")
    stage = str(payload["stage"])
    if stage != expected_stage:
        raise ConfigError(f"{expected_stage} model profile stage must be {expected_stage}, got {stage}")
    adapter = str(payload["adapter"]).strip()
    if not adapter:
        raise ConfigError(f"{expected_stage} model profile adapter must be non-empty")

    raw_assets = _mapping(payload["assets"], f"{expected_stage} model profile assets")
    if not raw_assets:
        raise ConfigError(f"{expected_stage} model profile assets must not be empty")
    assets: dict[str, ArtifactIdentity] = {}
    for raw_name, raw_value in raw_assets.items():
        name = str(raw_name)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ConfigError(f"invalid {expected_stage} asset name: {name!r}")
        value = _mapping(raw_value, f"{expected_stage} model asset {name}")
        _strict_keys(value, {"path", "sha256"}, {"path", "sha256"}, f"{expected_stage} model asset {name}")
        assets[name] = validate_artifact(
            _resolve_asset(source.parent, value["path"]), str(value["sha256"])
        )

    common = _mapping(payload["common"], f"{expected_stage} model profile common")
    _strict_keys(
        common,
        {"base_seed", "precision"},
        {"base_seed", "precision"},
        f"{expected_stage} model profile common",
    )
    base_seed = int(common["base_seed"])
    precision = str(common["precision"])
    parameters = _mapping(payload["parameters"], f"{expected_stage} model profile parameters")

    from face_preprocess.adapters import (
        adapter_source_sha256,
        get_adapter_spec,
        resolve_adapter_configuration,
    )

    spec = get_adapter_spec(expected_stage, adapter)
    resolved_parameters = resolve_adapter_configuration(
        stage=expected_stage,
        adapter=adapter,
        asset_names=set(assets),
        precision=precision,
        parameters=parameters,
    )
    return (
        ModelConfig(
            schema_version=schema,
            profile_id=profile_id,
            profile_version=profile_version,
            stage=stage,
            adapter=adapter,
            adapter_version=spec.version,
            adapter_source_sha256=adapter_source_sha256(spec),
            assets=assets,
            base_seed=base_seed,
            precision=precision,
            parameters=resolved_parameters,
        ),
        source,
        profile_hash,
    )


def _geometry_config(base: Path, payload: Mapping[str, Any]) -> GeometryConfig:
    required = {
        "template_obj",
        "template_obj_sha256",
        "template_landmarks",
        "template_landmarks_sha256",
        "template_points",
        "template_points_sha256",
        "refindex",
        "refindex_sha256",
        "expected_template_vertices",
        "expected_template_faces",
    }
    _strict_keys(payload, required, required, "pipeline.geometry")
    template_obj = validate_artifact(
        _resolve_asset(base, payload["template_obj"]), str(payload["template_obj_sha256"])
    )
    template_landmarks = validate_artifact(
        _resolve_asset(base, payload["template_landmarks"]),
        str(payload["template_landmarks_sha256"]),
    )
    template_points = validate_artifact(
        _resolve_asset(base, payload["template_points"]), str(payload["template_points_sha256"])
    )
    refindex = validate_artifact(_resolve_asset(base, payload["refindex"]), str(payload["refindex_sha256"]))
    return GeometryConfig(
        template_obj=template_obj.path,
        template_obj_sha256=template_obj.sha256,
        template_landmarks=template_landmarks.path,
        template_landmarks_sha256=template_landmarks.sha256,
        template_points=template_points.path,
        template_points_sha256=template_points.sha256,
        refindex=refindex.path,
        refindex_sha256=refindex.sha256,
        expected_template_vertices=int(payload["expected_template_vertices"]),
        expected_template_faces=int(payload["expected_template_faces"]),
    )


def _phenotype_config(base: Path, payload: Mapping[str, Any]) -> PhenotypeConfig:
    required = {
        "template_obj",
        "template_obj_sha256",
        "lamda_csv",
        "lamda_csv_sha256",
        "names_pp",
        "names_pp_sha256",
    }
    _strict_keys(payload, required, required, "pipeline.phenotype_landmarks")
    template = validate_artifact(
        _resolve_asset(base, payload["template_obj"]),
        str(payload["template_obj_sha256"]),
    )
    lamda = validate_artifact(
        _resolve_asset(base, payload["lamda_csv"]),
        str(payload["lamda_csv_sha256"]),
    )
    names = validate_artifact(
        _resolve_asset(base, payload["names_pp"]),
        str(payload["names_pp_sha256"]),
    )
    return PhenotypeConfig(
        template_obj=template.path,
        template_obj_sha256=template.sha256,
        lamda_csv=lamda.path,
        lamda_csv_sha256=lamda.sha256,
        names_pp=names.path,
        names_pp_sha256=names.sha256,
    )


def _device_config(payload: Mapping[str, Any]) -> DeviceProfile:
    allowed = {
        "schema_version",
        "profile_id",
        "profile_version",
        "coordinate_unit",
        "expected_edge_length_mm",
        "acquisition_device",
        "acquisition_protocol",
    }
    _strict_keys(payload, allowed, allowed, "device profile")
    schema, profile_id, profile_version = _identity(payload, "device profile")
    coordinate_unit = str(payload["coordinate_unit"])
    if coordinate_unit != "mm":
        raise ConfigError("device profile coordinate_unit must be mm")
    expected = float(payload["expected_edge_length_mm"])
    if expected <= 0:
        raise ConfigError("expected_edge_length_mm must be positive")
    return DeviceProfile(
        schema_version=schema,
        profile_id=profile_id,
        profile_version=profile_version,
        coordinate_unit=coordinate_unit,
        expected_edge_length_mm=expected,
        acquisition_device=str(payload["acquisition_device"]),
        acquisition_protocol=str(payload["acquisition_protocol"]),
    )


def _qc_config(payload: Mapping[str, Any], qc_level: str) -> QCConfig:
    allowed = {"schema_version", "profile_id", "profile_version", "metric_groups", "rules"}
    _strict_keys(payload, allowed, allowed, "QC configuration")
    schema, profile_id, profile_version = _identity(payload, "QC configuration")
    groups_raw = _mapping(payload["metric_groups"], "QC metric_groups")
    groups = {str(key): str(value) for key, value in groups_raw.items()}
    from face_preprocess.qc.registry import EMPIRICAL_GROUPS, empirical_group_for_metric

    unknown_groups = sorted(set(groups) - set(EMPIRICAL_GROUPS))
    if unknown_groups:
        raise ConfigError(f"unknown empirical QC group: {', '.join(unknown_groups)}")
    invalid_modes = sorted({value for value in groups.values() if value not in QC_GROUP_MODES})
    if invalid_modes:
        raise ConfigError(f"unsupported QC group mode: {', '.join(invalid_modes)}")
    rules_raw = payload["rules"]
    if not isinstance(rules_raw, list):
        raise ConfigError("QC rules must be a list")
    rules: list[QCRule] = []
    seen: set[str] = set()
    rule_fields = {
        "rule_id",
        "metric",
        "operator",
        "threshold",
        "unit",
        "severity",
        "minimum_qc_level",
    }
    for index, raw in enumerate(rules_raw):
        rule = _mapping(raw, f"QC rule {index}")
        _strict_keys(rule, rule_fields, rule_fields, f"QC rule {index}")
        rule_id = str(rule["rule_id"])
        if rule_id in seen:
            raise ConfigError(f"duplicate rule_id: {rule_id}")
        seen.add(rule_id)
        operator = str(rule["operator"])
        if operator not in QC_OPERATORS:
            raise ConfigError(f"unsupported operator: {operator}")
        threshold = rule["threshold"]
        if operator in {"gt", "gte", "lt", "lte"} and (
            isinstance(threshold, bool) or not isinstance(threshold, (int, float))
        ):
            raise ConfigError(f"{operator} threshold must be numeric: {rule_id}")
        if operator in {"inside", "outside"} and (
            not isinstance(threshold, (list, tuple))
            or len(threshold) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in threshold
            )
        ):
            raise ConfigError(f"{operator} threshold must be two numeric values: {rule_id}")
        severity = str(rule["severity"])
        if severity not in {"warning", "record_only"}:
            raise ConfigError(f"unsupported empirical severity: {severity}")
        minimum = str(rule["minimum_qc_level"])
        if minimum not in {"standard", "enhanced"}:
            raise ConfigError(f"invalid minimum_qc_level: {minimum}")
        if qc_level == "standard" and minimum == "enhanced" and severity == "warning":
            raise ConfigError(f"enhanced-only warning rule is incompatible with standard QC: {rule_id}")
        metric = str(rule["metric"])
        metric_group = empirical_group_for_metric(metric)
        if metric_group is None:
            raise ConfigError(f"QC rule references an unknown empirical metric: {metric}")
        if metric_group not in groups:
            raise ConfigError(
                f"QC rule group is missing from metric_groups: {metric_group}"
            )
        if metric_group in {"crop.stability", "landmark.stability"} and minimum != "enhanced":
            raise ConfigError(
                f"stability rule must require enhanced QC: {rule_id}"
            )
        unit = str(rule["unit"]).strip()
        if not unit:
            raise ConfigError(f"QC rule unit must be non-empty: {rule_id}")
        rules.append(
            QCRule(
                rule_id=rule_id,
                metric=metric,
                operator=operator,
                threshold=threshold,
                unit=unit,
                severity=severity,
                minimum_qc_level=minimum,
            )
        )
    return QCConfig(
        schema_version=schema,
        profile_id=profile_id,
        profile_version=profile_version,
        metric_groups=groups,
        rules=tuple(rules),
    )


def load_run_config(
    pipeline_config: str | Path,
    device_profile: str | Path,
    qc_config: str | Path,
    qc_level: str,
    symmetry_override: bool,
) -> ResolvedRunConfig:
    if qc_level not in {"standard", "enhanced"}:
        raise ConfigError(f"invalid qc_level: {qc_level}")
    pipeline_path, pipeline, pipeline_hash = _load_yaml(pipeline_config, "pipeline")
    device_path, device, device_hash = _load_yaml(device_profile, "device")
    qc_path, qc, qc_hash = _load_yaml(qc_config, "QC")

    required_top_fields = {
        "schema_version",
        "profile_id",
        "profile_version",
        "models",
        "geometry",
        "processing",
    }
    top_fields = required_top_fields | {"phenotype_landmarks"}
    _strict_keys(pipeline, top_fields, required_top_fields, "pipeline")
    schema, profile_id, profile_version = _identity(pipeline, "pipeline")
    models = _mapping(pipeline["models"], "pipeline.models")
    model_fields = {"crop_profile", "landmark_profile"}
    _strict_keys(models, model_fields, model_fields, "pipeline.models")
    processing = _mapping(pipeline["processing"], "pipeline.processing")
    processing_fields = {"symmetry_enabled", "texture_mode", "meshmonk"}
    _strict_keys(processing, processing_fields, {"symmetry_enabled"}, "pipeline.processing")
    meshmonk_raw = processing.get("meshmonk", {})
    if not isinstance(meshmonk_raw, dict):
        raise ConfigError("pipeline.processing.meshmonk must be a mapping")
    texture_mode = str(processing.get("texture_mode", "geometry_only"))
    if texture_mode != "geometry_only":
        raise ConfigError(
            "only texture_mode=geometry_only is implemented in v1; texture inventory is retained for a future adapter"
        )
    processing_config = ProcessingConfig(
        symmetry_enabled=bool(processing["symmetry_enabled"]) or bool(symmetry_override),
        texture_mode=texture_mode,
        meshmonk={str(key): value for key, value in meshmonk_raw.items()},
    )

    crop, crop_path, crop_hash = _model_profile_config(
        _resolve_asset(pipeline_path.parent, models["crop_profile"]), "crop"
    )
    landmark, landmark_path, landmark_hash = _model_profile_config(
        _resolve_asset(pipeline_path.parent, models["landmark_profile"]), "landmark"
    )
    geometry = _geometry_config(pipeline_path.parent, _mapping(pipeline["geometry"], "pipeline.geometry"))
    phenotype = (
        _phenotype_config(
            pipeline_path.parent,
            _mapping(pipeline["phenotype_landmarks"], "pipeline.phenotype_landmarks"),
        )
        if "phenotype_landmarks" in pipeline
        else None
    )
    return ResolvedRunConfig(
        schema_version=schema,
        profile_id=profile_id,
        profile_version=profile_version,
        crop=crop,
        landmark=landmark,
        geometry=geometry,
        processing=processing_config,
        device=_device_config(device),
        qc=_qc_config(qc, qc_level=qc_level),
        symmetry_enabled=processing_config.symmetry_enabled,
        config_hashes={
            "pipeline": pipeline_hash,
            "device": device_hash,
            "qc": qc_hash,
            "crop_model": crop_hash,
            "landmark_model": landmark_hash,
        },
        source_paths={
            "pipeline": pipeline_path,
            "device": device_path,
            "qc": qc_path,
            "crop_model": crop_path,
            "landmark_model": landmark_path,
        },
        phenotype=phenotype,
    )


def load_phenotype_config(path: str | Path) -> PhenotypeConfig:
    source, payload, _ = _load_yaml(path, "pipeline")
    if "phenotype_landmarks" not in payload:
        raise ConfigError("pipeline is missing fields: phenotype_landmarks")
    return _phenotype_config(
        source.parent,
        _mapping(payload["phenotype_landmarks"], "pipeline.phenotype_landmarks"),
    )


def load_qc_config(path: str | Path, qc_level: str) -> QCConfig:
    _, payload, _ = _load_yaml(path, "QC")
    return _qc_config(payload, qc_level=qc_level)
