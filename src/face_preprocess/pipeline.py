from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
from typing import Any
import uuid

import numpy as np
from scipy.spatial import cKDTree

from face_preprocess.adapters import build_adapter
from face_preprocess.adapters.contracts import CropAdapterResult, LandmarkAdapterResult
from face_preprocess.config import ResolvedRunConfig
from face_preprocess.config import load_run_config
from face_preprocess.errors import ConfigError, RunFatalError, SampleFailure
from face_preprocess.geometry.gpa import gpa_align
from face_preprocess.geometry.landmark_alignment import (
    align_mesh_with_landmarks,
    load_template_landmarks,
)
from face_preprocess.geometry.meshmonk import PythonMeshMonkBackend
from face_preprocess.geometry.surface import self_intersection_summary, surface_residuals
from face_preprocess.geometry.symmetry import symmetrize
from face_preprocess.hashing import derive_stage_seed
from face_preprocess.inputs import discover_inputs
from face_preprocess.models.landmark import LANDMARK_NAMES
from face_preprocess.qc.measurements import measure_input_mesh, measurements_to_dict
from face_preprocess.qc.rules import QCAssessment, evaluate_qc
from face_preprocess.qc.registry import HARD_CONTRACT_GROUPS
from face_preprocess.obj_io import read_obj, write_obj
from face_preprocess.phenotype_landmarks import (
    PhenotypeAssets,
    load_phenotype_assets,
    project_phenotype_landmarks,
    write_landmark_csv,
    write_picked_points,
)
from face_preprocess.runtime import RunLayout, classify_resume_record
from face_preprocess.types import Measurement, Mesh
from face_preprocess import __version__


_EVENT_LOCK = threading.Lock()


@dataclass(frozen=True)
class DiagnosticRun:
    label: str
    seed_slot: str
    angles_deg: tuple[float, float, float]


@dataclass(frozen=True)
class PipelineDependencies:
    crop: Any
    landmark: Any
    registration: Any


@dataclass(frozen=True)
class SamplePipelineResult:
    status: str
    final_mesh: Mesh | None
    metadata: dict[str, Any]
    artifacts: dict[str, Mesh]
    array_artifacts: dict[str, np.ndarray]


@dataclass(frozen=True)
class RunOptions:
    input_path: Path | None
    input_manifest: Path | None
    output: Path
    pipeline_config: Path
    device_profile: Path
    qc_config: Path
    device: str = "auto"
    workers: int = 1
    qc_level: str = "standard"
    output_mode: str = "compact"
    symmetry: bool = False
    resume: bool = False
    overwrite: bool = False
    retry_failed: bool = False
    phenotype_landmarks: bool = False


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_dir: Path
    total: int
    processed: int
    skipped: int
    passed: int
    warning: int
    failed: int
    device: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        return payload


def enhanced_schedule() -> tuple[DiagnosticRun, ...]:
    return (
        DiagnosticRun("D1", "S1", (0.0, 0.0, 0.0)),
        DiagnosticRun("D2", "S2", (0.0, 0.0, 0.0)),
        DiagnosticRun("D3", "S3", (5.0, -5.0, 5.0)),
        DiagnosticRun("D4", "S4", (5.0, -5.0, 5.0)),
    )


def rotation_matrix_xyz(angles_deg: tuple[float, float, float]) -> np.ndarray:
    x, y, z = np.deg2rad(np.asarray(angles_deg, dtype=float))
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rotated_mesh(mesh: Mesh, rotation: np.ndarray) -> tuple[Mesh, np.ndarray]:
    center = mesh.vertices.mean(axis=0)
    vertices = (mesh.vertices - center) @ rotation.T + center
    return Mesh(vertices, mesh.faces.copy()), center


def _inverse_points(points: np.ndarray, rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float) - center) @ rotation + center


def _measurement(
    measurements: dict[str, Measurement], name: str, value: Any, unit: str
) -> None:
    measurements[name] = Measurement(value, unit)


def _crop_metrics(measurements: dict[str, Measurement], crop: CropAdapterResult) -> None:
    metric_map = {
        "retained_vertex_fraction": ("crop.retention.vertex_fraction", "ratio"),
        "retained_face_fraction": ("crop.retention.face_fraction", "ratio"),
        "probability_mean": ("crop.probability.mean", "probability"),
        "probability_p05": ("crop.probability.p05", "probability"),
        "probability_p95": ("crop.probability.p95", "probability"),
        "postprocess_changed_vertices": ("crop.postprocess_delta.vertex_count", "count"),
        "postprocess_changed_fraction": ("crop.postprocess_delta.vertex_fraction", "ratio"),
        "kept_component_count": ("crop.spatial_structure.kept_component_count", "count"),
        "largest_component_fraction": (
            "crop.spatial_structure.largest_component_fraction",
            "ratio",
        ),
    }
    metrics = crop.metrics or {}
    for source, (target, unit) in metric_map.items():
        if source in metrics:
            _measurement(measurements, target, metrics[source], unit)
        else:
            measurements[target] = Measurement(None, unit, compute_status="not_available")
    if "kept_component_count" not in metrics and crop.final_mask is not None:
        _measurement(
            measurements,
            "crop.spatial_structure.kept_component_count",
            1 if np.any(crop.final_mask) else 0,
            "count",
        )


def _landmark_metrics(
    measurements: dict[str, Measurement], result: LandmarkAdapterResult, mesh: Mesh
) -> None:
    _measurement(
        measurements,
        "landmark.prediction.coordinates",
        result.snapped.tolist(),
        "mm",
    )
    if result.heatmap_peaks is None:
        measurements["landmark.heatmap.peak_min"] = Measurement(
            None, "probability", compute_status="not_available"
        )
        measurements["landmark.heatmap.peak_mean"] = Measurement(
            None, "probability", compute_status="not_available"
        )
    else:
        _measurement(
            measurements,
            "landmark.heatmap.peak_min",
            float(np.min(result.heatmap_peaks)),
            "probability",
        )
        _measurement(
            measurements,
            "landmark.heatmap.peak_mean",
            float(np.mean(result.heatmap_peaks)),
            "probability",
        )
    if result.snap_distances is None:
        measurements["landmark.snap.distance_mean_mm"] = Measurement(
            None, "mm", compute_status="not_available"
        )
        measurements["landmark.snap.distance_max_mm"] = Measurement(
            None, "mm", compute_status="not_available"
        )
    else:
        _measurement(
            measurements,
            "landmark.snap.distance_mean_mm",
            float(np.mean(result.snap_distances)),
            "mm",
        )
        _measurement(
            measurements,
            "landmark.snap.distance_max_mm",
            float(np.max(result.snap_distances)),
            "mm",
        )
    by_name = {name: result.snapped[index] for index, name in enumerate(result.names)}
    pairs = {
        "outer_eye_width_mm": ("wyjzuo", "wyjyou"),
        "inner_eye_width_mm": ("nyjzuo", "nyjyou"),
        "mouth_width_mm": ("kouzuo", "kouyou"),
        "nose_height_mm": ("bigen", "bixia"),
    }
    for label, (left, right) in pairs.items():
        _measurement(
            measurements,
            f"landmark.anatomy.{label}",
            float(np.linalg.norm(by_name[left] - by_name[right])),
            "mm",
        )
    midline = np.vstack([by_name["bijian"], by_name["bigen"], by_name["bixia"]])
    _measurement(
        measurements,
        "landmark.midline_symmetry.x_range_mm",
        float(np.ptp(midline[:, 0])),
        "mm",
    )
    # Euclidean distance to any boundary vertex is a stable, interpretable
    # diagnostic and remains raw for later threshold calibration.
    edges = np.sort(
        np.concatenate(
            [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]],
            axis=0,
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_indices = np.unique(unique_edges[counts == 1])
    if len(boundary_indices):
        boundary = mesh.vertices[boundary_indices]
        distances = np.min(
            np.linalg.norm(
                result.snapped[:, None, :] - boundary[None, :, :], axis=2
            ),
            axis=1,
        )
        _measurement(
            measurements,
            "landmark.crop_boundary.minimum_distance_mm",
            float(np.min(distances)),
            "mm",
        )
    else:
        measurements["landmark.crop_boundary.minimum_distance_mm"] = Measurement(
            None, "mm", compute_status="not_applicable"
        )


def _enhanced_diagnostics(
    raw_mesh: Mesh,
    formal_crop: CropAdapterResult,
    formal_landmarks: LandmarkAdapterResult,
    geometry_hash: str,
    config: ResolvedRunConfig,
    deps: PipelineDependencies,
    measurements: dict[str, Measurement],
) -> dict[str, Any]:
    crop_jaccard: list[float] = []
    landmark_rms: list[float] = []
    records: list[dict[str, Any]] = []
    for diagnostic in enhanced_schedule():
        rotation = rotation_matrix_xyz(diagnostic.angles_deg)
        rotated_raw, raw_center = _rotated_mesh(raw_mesh, rotation)
        crop_seed = derive_stage_seed(
            geometry_hash, f"crop:{diagnostic.seed_slot}", config.crop.base_seed
        )
        jaccard: float | None = None
        if formal_crop.final_mask is not None:
            crop = deps.crop.crop(rotated_raw, seed=crop_seed)
            if crop.final_mask is not None:
                intersection = np.count_nonzero(formal_crop.final_mask & crop.final_mask)
                union = np.count_nonzero(formal_crop.final_mask | crop.final_mask)
                jaccard = float(intersection / union) if union else 1.0
                crop_jaccard.append(jaccard)

        rotated_cropped, cropped_center = _rotated_mesh(formal_crop.mesh, rotation)
        landmark_seed = derive_stage_seed(
            geometry_hash, f"landmark:{diagnostic.seed_slot}", config.landmark.base_seed
        )
        landmark = deps.landmark.predict(rotated_cropped, seed=landmark_seed)
        restored = _inverse_points(landmark.snapped, rotation, cropped_center)
        rms = float(
            np.sqrt(np.mean(np.sum((restored - formal_landmarks.snapped) ** 2, axis=1)))
        )
        landmark_rms.append(rms)
        records.append(
            {
                "label": diagnostic.label,
                "seed_slot": diagnostic.seed_slot,
                "angles_deg": list(diagnostic.angles_deg),
                "crop_seed": crop_seed,
                "landmark_seed": landmark_seed,
                "crop_jaccard": jaccard,
                "landmark_rms_mm": rms,
                "rotation_center_raw": raw_center.tolist(),
            }
        )
    if crop_jaccard:
        _measurement(
            measurements, "crop.stability.jaccard_min", float(min(crop_jaccard)), "ratio"
        )
    else:
        measurements["crop.stability.jaccard_min"] = Measurement(
            None, "ratio", compute_status="not_available"
        )
    _measurement(
        measurements,
        "landmark.stability.rms_max_mm",
        float(max(landmark_rms)),
        "mm",
    )
    return {"runs": records}


def process_sample(
    raw_mesh: Mesh,
    geometry_hash: str,
    config: ResolvedRunConfig,
    template_mesh: Mesh,
    template_landmarks: np.ndarray,
    template_points: np.ndarray,
    refindex: np.ndarray | None,
    deps: PipelineDependencies,
    qc_level: str,
) -> SamplePipelineResult:
    total_started = time.perf_counter()
    measurements = measure_input_mesh(raw_mesh, config.device)
    hard_failures: list[str] = []
    if (
        len(template_mesh.vertices) != config.geometry.expected_template_vertices
        or len(template_mesh.faces) != config.geometry.expected_template_faces
    ):
        hard_failures.append("contract.template_topology")
        assessment = evaluate_qc(
            measurements, config.qc, level=qc_level, hard_failures=hard_failures
        )
        return SamplePipelineResult(
            status=assessment.status,
            final_mesh=None,
            metadata={
                "hard_failures": list(assessment.hard_failures),
                "hard_contracts": _hard_contract_dict(assessment.hard_failures),
                "measurements": measurements_to_dict(measurements),
                "qc": _assessment_dict(assessment),
            },
            artifacts={},
            array_artifacts={},
        )

    crop_seed = derive_stage_seed(geometry_hash, "crop:S0", config.crop.base_seed)
    stage_started = time.perf_counter()
    crop = deps.crop.crop(raw_mesh, seed=crop_seed)
    _measurement(
        measurements,
        "telemetry.stage_seconds.crop",
        time.perf_counter() - stage_started,
        "s",
    )
    _crop_metrics(measurements, crop)

    landmark_seed = derive_stage_seed(
        geometry_hash, "landmark:S0", config.landmark.base_seed
    )
    stage_started = time.perf_counter()
    landmarks = deps.landmark.predict(crop.mesh, seed=landmark_seed)
    _measurement(
        measurements,
        "telemetry.stage_seconds.landmark",
        time.perf_counter() - stage_started,
        "s",
    )
    if tuple(landmarks.names) != LANDMARK_NAMES or np.asarray(landmarks.snapped).shape != (9, 3):
        raise SampleFailure(
            f"landmark adapter must return canonical names/order and snapped shape (9, 3): "
            f"got {tuple(landmarks.names)} and {np.asarray(landmarks.snapped).shape}",
            contract="contract.landmarks_complete",
        )
    if not np.isfinite(landmarks.snapped).all():
        raise SampleFailure(
            "landmark adapter returned non-finite snapped coordinates",
            contract="contract.landmarks_finite",
        )
    snap_distances, _ = cKDTree(crop.mesh.vertices).query(landmarks.snapped, k=1)
    coordinate_scale = max(1.0, float(np.max(np.abs(crop.mesh.vertices))))
    snap_tolerance = max(1.0e-8, coordinate_scale * 1.0e-12)
    if np.any(np.asarray(snap_distances) > snap_tolerance):
        raise SampleFailure(
            "landmark adapter returned snapped coordinates that are not crop-mesh vertices",
            contract="contract.landmarks_snapped_on_mesh",
        )
    _landmark_metrics(measurements, landmarks, crop.mesh)
    bijian = landmarks.snapped[landmarks.names.index("bijian")]
    stage_started = time.perf_counter()
    try:
        coarse_alignment = align_mesh_with_landmarks(
            crop.mesh, landmarks.snapped, template_landmarks
        )
    except ValueError as exc:
        raise SampleFailure(
            f"9-point landmark coarse alignment failed: {exc}",
            contract="contract.landmark_alignment",
        ) from exc
    landmark_aligned = coarse_alignment.mesh
    _measurement(
        measurements,
        "telemetry.stage_seconds.coarse_alignment",
        time.perf_counter() - stage_started,
        "s",
    )

    stage_started = time.perf_counter()
    registration = deps.registration.map_mesh_diagnostic(
        target=landmark_aligned, template=template_mesh
    )
    _measurement(
        measurements,
        "telemetry.stage_seconds.registration",
        time.perf_counter() - stage_started,
        "s",
    )
    try:
        residual = surface_residuals(registration.mesh, landmark_aligned)
        _measurement(
            measurements,
            "registration.surface_residual.chamfer_mm",
            residual.chamfer,
            "mm",
        )
        _measurement(
            measurements,
            "registration.surface_residual.hausdorff_mm",
            residual.hausdorff,
            "mm",
        )
    except Exception as exc:
        measurements["registration.surface_residual.chamfer_mm"] = Measurement(
            None, "mm", compute_status="computation_error", error=str(exc)
        )
        measurements["registration.surface_residual.hausdorff_mm"] = Measurement(
            None, "mm", compute_status="computation_error", error=str(exc)
        )
    trajectory = list(registration.trajectory)
    _measurement(
        measurements,
        "registration.convergence.final_residual_mean_mm",
        float(trajectory[-1].get("residual_mean", np.nan)),
        "mm",
    )
    deformation = np.linalg.norm(
        registration.mesh.vertices - registration.rigid_end_vertices, axis=1
    )
    _measurement(
        measurements,
        "registration.deformation.mean_mm",
        float(np.mean(deformation)),
        "mm",
    )
    _measurement(
        measurements,
        "registration.deformation.p95_mm",
        float(np.percentile(deformation, 95.0)),
        "mm",
    )

    stage_started = time.perf_counter()
    gpa = gpa_align(registration.mesh, template_points)
    _measurement(
        measurements,
        "telemetry.stage_seconds.gpa",
        time.perf_counter() - stage_started,
        "s",
    )
    _measurement(
        measurements, "gpa.transform.scale", gpa.similarity_scale, "ratio"
    )
    _measurement(
        measurements,
        "gpa.transform.determinant",
        gpa.combined_determinant,
        "ratio",
    )
    final = gpa.mesh
    symmetry_metadata: dict[str, Any] | None = None
    if config.symmetry_enabled:
        if refindex is None:
            hard_failures.append("contract.final_geometry")
        else:
            symmetric = symmetrize(final, refindex)
            final = symmetric.mesh
            symmetry_metadata = {
                "displacement_mean": symmetric.displacement_mean,
                "displacement_p95": symmetric.displacement_p95,
                "displacement_max": symmetric.displacement_max,
            }
    if (
        len(final.vertices) != config.geometry.expected_template_vertices
        or len(final.faces) != config.geometry.expected_template_faces
        or not np.all(np.isfinite(final.vertices))
    ):
        hard_failures.append("contract.final_geometry")
    try:
        intersections = self_intersection_summary(final)
        _measurement(
            measurements,
            "final.geometry.self_intersection_count",
            int(intersections["count"]),
            "count",
        )
    except Exception as exc:
        measurements["final.geometry.self_intersection_count"] = Measurement(
            None, "count", compute_status="computation_error", error=str(exc)
        )
    _measurement(
        measurements,
        "final.geometry.vertex_count",
        len(final.vertices),
        "count",
    )
    enhanced: dict[str, Any] | None = None
    if qc_level == "enhanced":
        enhanced = _enhanced_diagnostics(
            raw_mesh,
            crop,
            landmarks,
            geometry_hash,
            config,
            deps,
            measurements,
        )
    _measurement(
        measurements,
        "telemetry.stage_seconds.total",
        time.perf_counter() - total_started,
        "s",
    )
    assessment = evaluate_qc(
        measurements, config.qc, level=qc_level, hard_failures=hard_failures
    )
    metadata = {
        "stage_seeds": {"crop": crop_seed, "landmark": landmark_seed},
        "landmarks": landmarks.coordinates(),
        "bijian": [float(value) for value in bijian],
        "coarse_alignment": {
            "method": "similarity_gpa_9_snapped_landmarks",
            "landmark_names": list(LANDMARK_NAMES),
            "weights": "equal",
            "allow_scaling": True,
            "allow_reflection": False,
            "template_landmarks_sha256": config.geometry.template_landmarks_sha256,
            "transform": coarse_alignment.transform.tolist(),
            "similarity_scale": coarse_alignment.similarity_scale,
            "determinant": coarse_alignment.determinant,
            "landmark_rms_mm": coarse_alignment.landmark_rms,
        },
        "registration": {
            "rigid_scale": registration.rigid_scale,
            "trajectory": trajectory,
        },
        "gpa": {
            "transform": gpa.transform.tolist(),
            "similarity_scale": gpa.similarity_scale,
            "combined_determinant": gpa.combined_determinant,
            "centroid_norm": gpa.centroid_norm,
            "centroid_size": gpa.centroid_size,
        },
        "symmetry": symmetry_metadata,
        "enhanced_diagnostics": enhanced,
        "hard_failures": list(assessment.hard_failures),
        "hard_contracts": _hard_contract_dict(assessment.hard_failures),
        "coordinate_space": "template_normalized",
        "coordinate_unit": "unitless",
        "measurements": measurements_to_dict(measurements),
        "qc": _assessment_dict(assessment),
    }
    return SamplePipelineResult(
        status=assessment.status,
        final_mesh=None if assessment.status == "failed" else final,
        metadata=metadata,
        artifacts={
            "cropped": crop.mesh,
            "landmark_aligned": landmark_aligned,
            "mapped": registration.mesh,
            "gpa": gpa.mesh,
        },
        array_artifacts={
            **(
                {"crop_probabilities": crop.probabilities}
                if crop.probabilities is not None
                else {}
            ),
            **(
                {"crop_initial_mask": crop.initial_mask.astype(np.uint8)}
                if crop.initial_mask is not None
                else {}
            ),
            **(
                {"crop_final_mask": crop.final_mask.astype(np.uint8)}
                if crop.final_mask is not None
                else {}
            ),
            **(
                {"landmark_continuous": landmarks.continuous}
                if landmarks.continuous is not None
                else {}
            ),
            "landmark_snapped": landmarks.snapped,
            **(
                {"landmark_snapped_vertex_indices": landmarks.snapped_vertex_indices}
                if landmarks.snapped_vertex_indices is not None
                else {}
            ),
            **(
                {"landmark_snap_distances": landmarks.snap_distances}
                if landmarks.snap_distances is not None
                else {}
            ),
            **(
                {"landmark_heatmap_peaks": landmarks.heatmap_peaks}
                if landmarks.heatmap_peaks is not None
                else {}
            ),
            **(
                {"landmark_topk_spreads": landmarks.topk_spreads}
                if landmarks.topk_spreads is not None
                else {}
            ),
            "registration_final_correspondences": registration.final_correspondences,
            "registration_final_weights": registration.final_weights,
            "registration_rigid_end_vertices": registration.rigid_end_vertices,
        },
    )


def _assessment_dict(assessment: QCAssessment) -> dict[str, Any]:
    return {
        "status": assessment.status,
        "qc_incomplete": assessment.qc_incomplete,
        "hard_failures": list(assessment.hard_failures),
        "rules": [asdict(item) for item in assessment.results],
    }


def _hard_contract_dict(failures: list[str] | tuple[str, ...]) -> dict[str, str]:
    failed = set(failures)
    return {
        name: ("failed" if name in failed else "passed")
        for name in HARD_CONTRACT_GROUPS
    }


def build_dependencies(config: ResolvedRunConfig, device: str) -> PipelineDependencies:
    crop = build_adapter("crop", config.crop, device)
    landmark = build_adapter("landmark", config.landmark, device)
    options = config.processing.meshmonk or {}
    allowed = set(PythonMeshMonkBackend.__dataclass_fields__)
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"unknown MeshMonk options: {', '.join(unknown)}")
    registration = PythonMeshMonkBackend(**options)
    return PipelineDependencies(crop, landmark, registration)


def resolve_device(requested: str) -> str:
    if requested not in {"auto", "cuda", "cpu"}:
        raise ConfigError(f"unsupported device: {requested}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        from face_preprocess.models.checkpoint import require_torch

        torch = require_torch()
    except RuntimeError:
        if requested == "cuda":
            raise RunFatalError("CUDA was requested but PyTorch is not installed")
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RunFatalError("CUDA was requested but torch.cuda.is_available() is false")
    return "cuda" if requested == "auto" and available else ("cpu" if requested == "auto" else requested)


def configure_torch_determinism() -> None:
    try:
        from face_preprocess.models.checkpoint import require_torch

        torch = require_torch()
    except RuntimeError:
        return
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _source_identity() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(b"face-preprocess-source-v1\0")
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_identity(device: str) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    try:
        from importlib import metadata

        for package in ("numpy", "scipy", "pandas", "PyYAML", "trimesh", "matplotlib", "rtree"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
    except ImportError:  # pragma: no cover - Python 3.10 always provides it
        pass
    torch_info: dict[str, Any] = {"installed": False}
    try:
        from face_preprocess.models.checkpoint import require_torch

        torch = require_torch()
        torch_info = {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else None
            ),
        }
    except RuntimeError:
        pass
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "device": device,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
        ),
        "packages": versions,
        "torch": torch_info,
    }


def _run_fingerprints(
    records: list[Any],
    config: ResolvedRunConfig,
    options: RunOptions,
    device: str,
) -> dict[str, str]:
    runtime_identity = _runtime_identity(device)
    return {
        "input_set": _stable_hash(
            [
                {
                    "relative_path": record.relative_path,
                    "source_file_sha256": record.source_file_sha256,
                    "geometry_sha256": record.geometry_sha256,
                }
                for record in records
            ]
        ),
        "scientific": _stable_hash(
            {
                "config_hashes": config.config_hashes,
                "adapters": {
                    "crop": {
                        "name": config.crop.adapter,
                        "version": config.crop.adapter_version,
                        "source_sha256": config.crop.adapter_source_sha256,
                    },
                    "landmark": {
                        "name": config.landmark.adapter,
                        "version": config.landmark.adapter_version,
                        "source_sha256": config.landmark.adapter_source_sha256,
                    },
                },
                "symmetry": config.symmetry_enabled,
                "phenotype_landmarks": options.phenotype_landmarks,
                "phenotype_assets": (
                    {
                        "template_obj_sha256": config.phenotype.template_obj_sha256,
                        "lamda_csv_sha256": config.phenotype.lamda_csv_sha256,
                        "names_pp_sha256": config.phenotype.names_pp_sha256,
                    }
                    if options.phenotype_landmarks and config.phenotype is not None
                    else None
                ),
                "qc_level": options.qc_level,
                "source_identity": _source_identity(),
            }
        ),
        "execution": _stable_hash(
            {
                "tool_version": __version__,
                "device": device,
                "workers": options.workers,
                "output_mode": options.output_mode,
                "runtime": runtime_identity,
            }
        ),
    }


def _load_geometry_assets(
    config: ResolvedRunConfig,
) -> tuple[Mesh, np.ndarray, np.ndarray, np.ndarray]:
    template_mesh, _ = read_obj(config.geometry.template_obj)
    try:
        template_landmarks = load_template_landmarks(
            config.geometry.template_landmarks, LANDMARK_NAMES
        )
    except ValueError as exc:
        raise RunFatalError(f"invalid template landmark asset: {exc}") from exc
    template_points = np.loadtxt(config.geometry.template_points, dtype=float)
    if template_points.shape != (config.geometry.expected_template_vertices, 3):
        raise RunFatalError(
            f"template point shape mismatch: {template_points.shape}"
        )
    if (
        len(template_mesh.vertices) != config.geometry.expected_template_vertices
        or len(template_mesh.faces) != config.geometry.expected_template_faces
    ):
        raise RunFatalError(
            "template OBJ topology mismatch: "
            f"{len(template_mesh.vertices)}/{len(template_mesh.faces)}"
        )
    refindex = np.loadtxt(config.geometry.refindex, dtype=np.int64).reshape(-1)
    if len(refindex) != config.geometry.expected_template_vertices:
        raise RunFatalError(f"refindex length mismatch: {len(refindex)}")
    if len(refindex) and refindex.min() == 1 and refindex.max() == len(refindex):
        refindex = refindex - 1
    if len(refindex) and (refindex.min() < 0 or refindex.max() >= len(refindex)):
        raise RunFatalError("refindex contains out-of-range values")
    return template_mesh, template_landmarks, template_points, refindex


def _sample_metadata_base(record: Any, config: ResolvedRunConfig) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "relative_path": record.relative_path,
        "source_path": str(record.path),
        "source_file_sha256": record.source_file_sha256,
        "geometry_sha256": record.geometry_sha256,
        "subject_group": record.subject_group,
        "batch_id": record.batch_id,
        "user_metadata": record.user_metadata,
        "obj_inventory": asdict(record.obj_inventory) if record.obj_inventory is not None else None,
        "input_error": record.input_error,
        "duplicate_geometry_group_size": record.duplicate_geometry_group_size,
        "profiles": {
            "pipeline": [config.profile_id, config.profile_version],
            "crop_model": [config.crop.profile_id, config.crop.profile_version],
            "landmark_model": [
                config.landmark.profile_id,
                config.landmark.profile_version,
            ],
            "device": [config.device.profile_id, config.device.profile_version],
            "qc": [config.qc.profile_id, config.qc.profile_version],
        },
        "adapters": {
            "crop": {
                "name": config.crop.adapter,
                "version": config.crop.adapter_version,
                "source_sha256": config.crop.adapter_source_sha256,
                "precision": config.crop.precision,
                "assets": {
                    name: {
                        "sha256": identity.sha256,
                        "size_bytes": identity.size_bytes,
                    }
                    for name, identity in sorted(config.crop.assets.items())
                },
            },
            "landmark": {
                "name": config.landmark.adapter,
                "version": config.landmark.adapter_version,
                "source_sha256": config.landmark.adapter_source_sha256,
                "precision": config.landmark.precision,
                "assets": {
                    name: {
                        "sha256": identity.sha256,
                        "size_bytes": identity.size_bytes,
                    }
                    for name, identity in sorted(config.landmark.assets.items())
                },
            },
        },
    }


def _is_run_fatal_exception(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, (MemoryError, RunFatalError)) or (
        "cuda" in message and ("out of memory" in message or "device-side assert" in message)
    )


def _atomic_summary(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _append_event(layout: RunLayout, event: dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": layout.run_id,
        **event,
    }
    encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True)
    with _EVENT_LOCK:
        with (layout.root / "logs" / "events.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(encoded + "\n")


def _peak_rss_measurement() -> dict[str, Any]:
    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        megabytes = raw / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)
        return Measurement(megabytes, "MB").as_dict()
    except (ImportError, AttributeError, OSError):
        return Measurement(
            None,
            "MB",
            compute_status="not_applicable",
            error="peak RSS is unavailable on this platform",
        ).as_dict()


def rebuild_run_summary(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    records: list[dict[str, Any]] = []
    for path in sorted((root / "metadata" / "samples").glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    counts = {"passed": 0, "warning": 0, "failed": 0}
    for record in records:
        status = str(record.get("status", "failed"))
        if status in counts:
            counts[status] += 1
    csv_rows = [
        {
            "output_key": record.get("output_key"),
            "sample_id": record.get("sample_id"),
            "status": record.get("status"),
            "source_file_sha256": record.get("source_file_sha256"),
            "geometry_sha256": record.get("geometry_sha256"),
            "final_obj": record.get("final_obj"),
            "qc_incomplete": (record.get("qc") or {}).get("qc_incomplete"),
            "committed_at": record.get("committed_at"),
        }
        for record in records
    ]
    csv_path = root / "metadata" / "samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "output_key",
            "sample_id",
            "status",
            "source_file_sha256",
            "geometry_sha256",
            "final_obj",
            "qc_incomplete",
            "committed_at",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    payload = {
        "schema": "face-preprocess-summary-v1",
        "sample_count": len(records),
        "counts": counts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_summary(root / "summary.json", payload)
    return payload


def _process_record(
    record: Any,
    layout: RunLayout,
    config: ResolvedRunConfig,
    template_mesh: Mesh,
    template_landmarks: np.ndarray,
    template_points: np.ndarray,
    refindex: np.ndarray,
    deps: PipelineDependencies,
    options: RunOptions,
    device: str,
    phenotype_assets: PhenotypeAssets | None,
) -> str:
    transaction = layout.begin_sample(str(record.output_key))
    base = _sample_metadata_base(record, config)
    try:
        if record.input_error:
            raise SampleFailure(record.input_error)
        raw_mesh, _ = read_obj(record.path)
        result = process_sample(
            raw_mesh,
            record.geometry_sha256,
            config,
            template_mesh,
            template_landmarks,
            template_points,
            refindex if config.symmetry_enabled else None,
            deps,
            options.qc_level,
        )
        if options.phenotype_landmarks and result.final_mesh is not None:
            if config.phenotype is None or phenotype_assets is None:
                raise SampleFailure(
                    "phenotype landmark assets are not configured",
                    contract="contract.phenotype_landmarks",
                )
            try:
                phenotype_result = project_phenotype_landmarks(
                    result.final_mesh, phenotype_assets
                )
            except ValueError as exc:
                raise SampleFailure(
                    f"phenotype landmark projection failed: {exc}",
                    contract="contract.phenotype_landmarks",
                ) from exc
            phenotype_csv = f"phenotype_landmarks/{record.output_key}.csv"
            phenotype_pp = f"phenotype_landmarks/{record.output_key}.pp"
            write_landmark_csv(
                transaction.stage_path(phenotype_csv), phenotype_result
            )
            write_picked_points(
                transaction.stage_path(phenotype_pp),
                phenotype_result,
                record.path.name,
            )
            result.metadata["phenotype_landmarks"] = {
                "count": len(phenotype_result.names),
                "names": list(phenotype_result.names),
                "csv": phenotype_csv,
                "picked_points": phenotype_pp,
                "template_sha256": config.phenotype.template_obj_sha256,
                "lamda_sha256": config.phenotype.lamda_csv_sha256,
                "names_sha256": config.phenotype.names_pp_sha256,
            }
        final_size = 0
        if result.final_mesh is not None:
            staged_final = transaction.stage_path(f"final_obj/{record.output_key}.obj")
            write_obj(staged_final, result.final_mesh)
            final_size = staged_final.stat().st_size
        if options.output_mode == "full":
            for stage, mesh in result.artifacts.items():
                write_obj(
                    transaction.stage_path(
                        f"artifacts/{record.output_key}/{stage}.obj"
                    ),
                    mesh,
                )
            np.savez_compressed(
                transaction.stage_path(
                    f"artifacts/{record.output_key}/diagnostic_arrays.npz"
                ),
                **result.array_artifacts,
            )
        result.metadata["measurements"].update(
            {
                "telemetry.device.kind": Measurement(device, "category").as_dict(),
                "telemetry.io.source_bytes": Measurement(
                    record.path.stat().st_size, "bytes"
                ).as_dict(),
                "telemetry.io.final_obj_bytes": Measurement(
                    final_size, "bytes"
                ).as_dict(),
                "telemetry.peak_rss.mb": _peak_rss_measurement(),
            }
        )
        metadata = {
            **base,
            **result.metadata,
            "status": result.status,
            "output_mode": options.output_mode,
            "qc_level": options.qc_level,
            "final_obj": (
                f"final_obj/{record.output_key}.obj"
                if result.final_mesh is not None
                else None
            ),
            "diagnostic_arrays": (
                f"artifacts/{record.output_key}/diagnostic_arrays.npz"
                if options.output_mode == "full"
                else None
            ),
        }
        transaction.commit(metadata)
        _append_event(
            layout,
            {
                "event": "sample_complete",
                "output_key": record.output_key,
                "sample_id": record.sample_id,
                "status": result.status,
            },
        )
        return result.status
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)) or _is_run_fatal_exception(exc):
            transaction.preserve_incomplete()
            raise
        failure_contracts: tuple[str, ...]
        if record.input_error:
            failure_contracts = ("contract.obj_readable",)
        elif isinstance(exc, SampleFailure) and exc.contract:
            failure_contracts = (exc.contract,)
        else:
            failure_contracts = ()
        transaction.fail(
            {
                **base,
                "status": "failed",
                "final_obj": None,
                "hard_failures": list(failure_contracts),
                "hard_contracts": (
                    _hard_contract_dict(failure_contracts)
                    if failure_contracts
                    else {name: "not_evaluated" for name in HARD_CONTRACT_GROUPS}
                ),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _append_event(
            layout,
            {
                "event": "sample_complete",
                "output_key": record.output_key,
                "sample_id": record.sample_id,
                "status": "failed",
                "error_type": type(exc).__name__,
            },
        )
        return "failed"


def run_pipeline(
    options: RunOptions, deps: PipelineDependencies | None = None
) -> RunSummary:
    if options.output_mode not in {"compact", "full"}:
        raise ConfigError("output_mode must be compact or full")
    if options.qc_level not in {"standard", "enhanced"}:
        raise ConfigError("qc_level must be standard or enhanced")
    if options.workers <= 0:
        raise ConfigError("workers must be positive")
    records = discover_inputs(options.input_path, options.input_manifest)
    config = load_run_config(
        options.pipeline_config,
        options.device_profile,
        options.qc_config,
        options.qc_level,
        options.symmetry,
    )
    phenotype_assets: PhenotypeAssets | None = None
    if options.phenotype_landmarks:
        if config.phenotype is None:
            raise ConfigError(
                "--phenotype-landmarks requires pipeline.phenotype_landmarks configuration"
            )
        try:
            phenotype_assets = load_phenotype_assets(
                config.phenotype.template_obj,
                config.phenotype.lamda_csv,
                config.phenotype.names_pp,
            )
        except ValueError as exc:
            raise RunFatalError(f"invalid phenotype landmark assets: {exc}") from exc
    device = resolve_device(options.device)
    configure_torch_determinism()
    if device == "cuda" and options.workers != 1:
        raise ConfigError("CUDA runs require --workers 1")
    fingerprints = _run_fingerprints(records, config, options, device)
    layout = RunLayout.prepare(
        options.output,
        fingerprints,
        resume=options.resume,
        overwrite=options.overwrite,
    )
    _append_event(
        layout,
        {
            "event": "run_start" if not options.resume else "run_resume",
            "sample_count": len(records),
            "device": device,
            "workers": options.workers,
        },
    )
    for name, source in config.source_paths.items():
        destination = layout.root / "configs" / f"{name}{source.suffix.lower()}"
        if not destination.exists():
            shutil.copy2(source, destination)
    resolved_path = layout.root / "configs" / "resolved_config.json"
    if not resolved_path.exists():
        _atomic_summary(resolved_path, _jsonable(asdict(config)))
    environment_path = layout.root / "configs" / "runtime_environment.json"
    if not environment_path.exists():
        _atomic_summary(
            environment_path,
            {
                "source_identity": _source_identity(),
                "runtime": _runtime_identity(device),
            },
        )
    template_mesh, template_landmarks, template_points, refindex = _load_geometry_assets(config)
    actual_deps = deps or build_dependencies(config, device)

    pending: list[Any] = []
    skipped = 0
    if options.resume:
        for record in records:
            metadata_path = (
                layout.root / "metadata" / "samples" / f"{record.output_key}.json"
            )
            if not metadata_path.exists():
                pending.append(record)
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            decision = classify_resume_record(
                metadata, fingerprints, retry_failed=options.retry_failed
            )
            if decision == "skip":
                skipped += 1
            else:
                pending.append(record)
    else:
        pending = records

    statuses: list[str] = []
    if device == "cpu" and options.workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=options.workers) as executor:
            futures: dict[Future[str], Any] = {
                executor.submit(
                    _process_record,
                    record,
                    layout,
                    config,
                    template_mesh,
                    template_landmarks,
                    template_points,
                    refindex,
                    actual_deps,
                    options,
                    device,
                    phenotype_assets,
                ): record
                for record in pending
            }
            for future in as_completed(futures):
                statuses.append(future.result())
    else:
        for record in pending:
            statuses.append(
                _process_record(
                    record,
                    layout,
                    config,
                    template_mesh,
                    template_landmarks,
                    template_points,
                    refindex,
                    actual_deps,
                    options,
                    device,
                    phenotype_assets,
                )
            )
    final_counts = rebuild_run_summary(layout.root)["counts"]
    summary = RunSummary(
        run_id=layout.run_id,
        run_dir=layout.root,
        total=len(records),
        processed=len(pending),
        skipped=skipped,
        passed=int(final_counts["passed"]),
        warning=int(final_counts["warning"]),
        failed=int(final_counts["failed"]),
        device=device,
    )
    _append_event(layout, {"event": "run_complete", **summary.as_dict()})
    return summary
