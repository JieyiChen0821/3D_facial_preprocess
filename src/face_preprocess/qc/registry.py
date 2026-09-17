from __future__ import annotations


EMPIRICAL_GROUPS = (
    "input.edge_length",
    "input.degenerate_faces",
    "input.duplicate_geometry",
    "input.connected_components",
    "input.topology",
    "input.extent_density",
    "input.normal_consistency",
    "crop.retention",
    "crop.probability",
    "crop.spatial_structure",
    "crop.postprocess_delta",
    "landmark.prediction",
    "landmark.heatmap",
    "landmark.snap",
    "landmark.anatomy",
    "landmark.midline_symmetry",
    "landmark.crop_boundary",
    "registration.surface_residual",
    "registration.convergence",
    "registration.deformation",
    "gpa.transform",
    "final.geometry",
    "crop.stability",
    "landmark.stability",
)

HARD_CONTRACT_GROUPS = (
    "contract.obj_readable",
    "contract.triangular_faces",
    "contract.finite_geometry",
    "contract.nonempty_geometry",
    "contract.artifact_integrity",
    "contract.crop_nonempty",
    "contract.landmarks_complete",
    "contract.landmarks_finite",
    "contract.landmarks_snapped_on_mesh",
    "contract.landmark_alignment",
    "contract.template_topology",
    "contract.final_geometry",
)

TELEMETRY_GROUPS = (
    "telemetry.stage_seconds",
    "telemetry.peak_rss",
    "telemetry.device",
    "telemetry.io",
)


def empirical_group_for_metric(metric: str) -> str | None:
    candidates = [group for group in EMPIRICAL_GROUPS if metric == group or metric.startswith(group + ".")]
    return max(candidates, key=len) if candidates else None
