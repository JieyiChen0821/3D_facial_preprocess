from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from face_preprocess.errors import ConfigError


AdapterFactory = Callable[[Any, str], Any]


@dataclass(frozen=True)
class ParameterSpec:
    path: str
    value_type: str
    default: Any
    description: str
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class AdapterSpec:
    stage: str
    name: str
    version: str
    factory: AdapterFactory
    required_assets: tuple[str, ...]
    supported_precisions: tuple[str, ...]
    parameters: tuple[ParameterSpec, ...]
    source_files: tuple[str, ...] = ()


_REGISTRY: dict[tuple[str, str], AdapterSpec] = {}


def register_adapter(spec: AdapterSpec, *, replace: bool = False) -> None:
    if spec.stage not in {"crop", "landmark"}:
        raise ConfigError(f"unsupported adapter stage: {spec.stage}")
    if not spec.name or not spec.version:
        raise ConfigError("adapter name and version must be non-empty")
    if not spec.source_files:
        raise ConfigError(f"adapter {spec.name!r} must declare package-local source_files")
    package_root = Path(__file__).resolve().parents[1]
    for relative in spec.source_files:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ConfigError(
                f"adapter {spec.name!r} source_files must be package-local: {relative}"
            )
        if not (package_root / candidate).is_file():
            raise ConfigError(f"adapter source file does not exist: {package_root / candidate}")
    key = (spec.stage, spec.name)
    if key in _REGISTRY and not replace:
        raise ConfigError(f"adapter is already registered: {spec.stage}/{spec.name}")
    paths = [item.path for item in spec.parameters]
    if len(paths) != len(set(paths)):
        raise ConfigError(f"adapter has duplicate parameter paths: {spec.name}")
    _REGISTRY[key] = spec


def get_adapter_spec(stage: str, name: str) -> AdapterSpec:
    try:
        return _REGISTRY[(stage, name)]
    except KeyError as exc:
        available = ", ".join(
            sorted(item.name for item in _REGISTRY.values() if item.stage == stage)
        ) or "none"
        raise ConfigError(
            f"unknown {stage} adapter {name!r}; registered adapters: {available}"
        ) from exc


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            output.update(_flatten(item, path))
        else:
            output[path] = item
    return output


def _set_nested(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = deepcopy(value)


def _validate_value(spec: ParameterSpec, value: Any, adapter: str) -> Any:
    label = f"{adapter}.parameters.{spec.path}"
    valid = False
    if spec.value_type == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif spec.value_type == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif spec.value_type == "boolean":
        valid = isinstance(value, bool)
    elif spec.value_type == "string":
        valid = isinstance(value, str)
    elif spec.value_type == "string_list":
        valid = isinstance(value, list) and all(isinstance(item, str) for item in value)
    else:  # adapter authors get a classified configuration error
        raise ConfigError(f"unsupported parameter type {spec.value_type!r} in {adapter}")
    if not valid:
        raise ConfigError(f"{label} must be {spec.value_type}")
    comparable = tuple(value) if isinstance(value, list) else value
    choices = tuple(tuple(item) if isinstance(item, list) else item for item in spec.choices)
    if choices and comparable not in choices:
        raise ConfigError(f"{label} must be one of {list(spec.choices)!r}")
    if spec.minimum is not None and float(value) < spec.minimum:
        raise ConfigError(f"{label} must be >= {spec.minimum}")
    if spec.maximum is not None and float(value) > spec.maximum:
        raise ConfigError(f"{label} must be <= {spec.maximum}")
    return deepcopy(value)


def resolve_adapter_configuration(
    *,
    stage: str,
    adapter: str,
    asset_names: set[str],
    precision: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    spec = get_adapter_spec(stage, adapter)
    missing_assets = sorted(set(spec.required_assets) - set(asset_names))
    if missing_assets:
        raise ConfigError(
            f"{stage} adapter {adapter!r} is missing assets: {', '.join(missing_assets)}"
        )
    if precision not in spec.supported_precisions:
        raise ConfigError(
            f"{stage} adapter {adapter!r} does not support precision {precision!r}; "
            f"supported: {', '.join(spec.supported_precisions)}"
        )
    supplied = _flatten(parameters)
    definitions = {item.path: item for item in spec.parameters}
    unknown = sorted(set(supplied) - set(definitions))
    if unknown:
        raise ConfigError(
            f"{stage} adapter {adapter!r} has unknown parameter(s): "
            + ", ".join(f"parameters.{item}" for item in unknown)
        )
    resolved: dict[str, Any] = {}
    for path, definition in definitions.items():
        raw = supplied[path] if path in supplied else definition.default
        _set_nested(resolved, path, _validate_value(definition, raw, adapter))
    return resolved


def _parameter_payload(item: ParameterSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": f"parameters.{item.path}",
        "type": item.value_type,
        "default": deepcopy(item.default),
        "description": item.description,
    }
    if item.choices:
        payload["choices"] = deepcopy(list(item.choices))
    if item.minimum is not None:
        payload["minimum"] = item.minimum
    if item.maximum is not None:
        payload["maximum"] = item.maximum
    return payload


def adapter_source_sha256(spec: AdapterSpec) -> str:
    package_root = Path(__file__).resolve().parents[1]
    rows: list[tuple[str, str]] = []
    for relative in sorted(spec.source_files):
        path = package_root / relative
        if not path.is_file():
            raise ConfigError(f"adapter source file does not exist: {path}")
        rows.append((relative.replace("\\", "/"), hashlib.sha256(path.read_bytes()).hexdigest()))
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def list_adapter_info(stage: str | None = None, name: str | None = None) -> list[dict[str, Any]]:
    if name is not None and stage is None:
        raise ConfigError("--adapter requires --stage")
    if name is not None:
        specs = [get_adapter_spec(str(stage), name)]
    else:
        specs = sorted(
            (item for item in _REGISTRY.values() if stage is None or item.stage == stage),
            key=lambda item: (item.stage, item.name),
        )
    return [
        {
            "stage": spec.stage,
            "name": spec.name,
            "version": spec.version,
            "required_assets": list(spec.required_assets),
            "supported_precisions": list(spec.supported_precisions),
            "source_sha256": adapter_source_sha256(spec),
            "parameters": [_parameter_payload(item) for item in spec.parameters],
        }
        for spec in specs
    ]


def build_adapter(stage: str, config: Any, device: str) -> Any:
    spec = get_adapter_spec(stage, str(config.adapter))
    return spec.factory(config, device)


def _crop_factory(config: Any, device: str) -> Any:
    from face_preprocess.adapters.builtin import PointNextCropAdapter

    return PointNextCropAdapter.from_config(config, device)


def _landmark_factory(config: Any, device: str) -> Any:
    from face_preprocess.adapters.builtin import HeatmapOffsetLandmarkAdapter

    return HeatmapOffsetLandmarkAdapter.from_config(config, device)


_COMMON_SOURCE = ("adapters/registry.py", "adapters/builtin.py", "adapters/contracts.py")

register_adapter(
    AdapterSpec(
        stage="crop",
        name="pointnext_crop_v1",
        version="1.0.0",
        factory=_crop_factory,
        required_assets=("checkpoint",),
        supported_precisions=("fp32",),
        parameters=(
            ParameterSpec("input.features", "string_list", ["xyz"], "Input feature channels.", choices=(["xyz"],)),
            ParameterSpec("input.normalization", "string", "centroid_max_radius", "Point normalization.", choices=("centroid_max_radius",)),
            ParameterSpec("input.sampling.strategy", "string", "shuffled_cover_plus_random_votes", "Voting sampler.", choices=("shuffled_cover_plus_random_votes",)),
            ParameterSpec("input.sampling.num_points", "integer", 32768, "Points per inference pass.", minimum=1),
            ParameterSpec("inference.extra_passes", "integer", 2, "Additional random voting passes.", minimum=0),
            ParameterSpec("output_mapping.face_class_index", "integer", 1, "Logit class interpreted as face.", minimum=0),
            ParameterSpec("postprocess.probability_threshold", "number", 0.5, "Face probability threshold.", minimum=0.0, maximum=1.0),
            ParameterSpec("postprocess.smooth_iterations", "integer", 0, "Graph smoothing iterations.", minimum=0),
            ParameterSpec("postprocess.smooth_alpha", "number", 0.5, "Graph smoothing blend.", minimum=0.0, maximum=1.0),
            ParameterSpec("postprocess.keep_largest_component", "boolean", True, "Keep only the largest selected component."),
            ParameterSpec("postprocess.min_component_vertices", "integer", 100, "Minimum selected component size.", minimum=0),
        ),
        source_files=_COMMON_SOURCE + (
            "models/crop.py",
            "models/common.py",
            "models/checkpoint.py",
            "models/_crop_network.py",
        ),
    )
)

_LANDMARK_ORDER = [
    "bijian",
    "bigen",
    "bixia",
    "wyjzuo",
    "nyjzuo",
    "nyjyou",
    "wyjyou",
    "kouzuo",
    "kouyou",
]

register_adapter(
    AdapterSpec(
        stage="landmark",
        name="heatmap_offset_landmark_v1",
        version="1.0.0",
        factory=_landmark_factory,
        required_assets=("checkpoint",),
        supported_precisions=("fp32",),
        parameters=(
            ParameterSpec("input.features", "string_list", ["xyz"], "Input feature channels.", choices=(["xyz"],)),
            ParameterSpec("input.normalization", "string", "centroid_max_radius", "Point normalization.", choices=("centroid_max_radius",)),
            ParameterSpec("input.sampling.strategy", "string", "deterministic_random_fixed_count", "Landmark point sampler.", choices=("deterministic_random_fixed_count",)),
            ParameterSpec("input.sampling.num_points", "integer", 32768, "Points per inference.", minimum=1),
            ParameterSpec("inference.top_k", "integer", 16, "Heatmap candidates averaged per landmark.", minimum=1),
            ParameterSpec("postprocess.snap_method", "string", "nearest_mesh_vertex", "Continuous-to-mesh snapping method.", choices=("nearest_mesh_vertex",)),
            ParameterSpec("output_mapping.native_order", "string_list", _LANDMARK_ORDER, "Checkpoint output order.", choices=(_LANDMARK_ORDER,)),
            ParameterSpec("output_mapping.canonical_order", "string_list", _LANDMARK_ORDER, "Pipeline landmark order.", choices=(_LANDMARK_ORDER,)),
        ),
        source_files=_COMMON_SOURCE + (
            "models/landmark.py",
            "models/common.py",
            "models/checkpoint.py",
            "models/_landmark_network.py",
        ),
    )
)
