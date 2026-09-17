from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from face_preprocess.adapters.contracts import CropAdapterResult, LandmarkAdapterResult
from face_preprocess.config import (
    ArtifactIdentity,
    DeviceProfile,
    GeometryConfig,
    ModelConfig,
    ProcessingConfig,
    QCConfig,
    QCRule,
    ResolvedRunConfig,
)
from face_preprocess.geometry.meshmonk import MeshMonkResult
from face_preprocess.errors import SampleFailure
from face_preprocess.models.crop import CropResult
from face_preprocess.models.landmark import LANDMARK_NAMES, LandmarkResult
from face_preprocess.pipeline import (
    PipelineDependencies,
    enhanced_schedule,
    process_sample,
    rotation_matrix_xyz,
)
from face_preprocess.types import Mesh


def _mesh() -> Mesh:
    return Mesh(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
    )


def _config(tmp_path: Path, qc: QCConfig | None = None) -> ResolvedRunConfig:
    artifact = tmp_path / "x"
    artifact.write_text("x", encoding="utf-8")
    identity = ArtifactIdentity(artifact, "0" * 64, 1)
    crop_model = ModelConfig(
        1,
        "crop",
        "1",
        "crop",
        "pointnext_crop_v1",
        "1.0.0",
        "1" * 64,
        {"checkpoint": identity},
        42,
        "fp32",
        {"input": {"sampling": {"num_points": 4}}},
    )
    landmark_model = ModelConfig(
        1,
        "landmark",
        "1",
        "landmark",
        "heatmap_offset_landmark_v1",
        "1.0.0",
        "2" * 64,
        {"checkpoint": identity},
        20260610,
        "fp32",
        {"input": {"sampling": {"num_points": 4}}, "inference": {"top_k": 2}},
    )
    geometry = GeometryConfig(
        artifact,
        "0" * 64,
        artifact,
        "0" * 64,
        artifact,
        "0" * 64,
        artifact,
        "0" * 64,
        4,
        4,
    )
    return ResolvedRunConfig(
        1,
        "test",
        "1",
        crop_model,
        landmark_model,
        geometry,
        ProcessingConfig(False),
        DeviceProfile(1, "scanner", "1", "mm", 1.2, "x", "x"),
        qc or QCConfig(1, "qc", "1", {}, ()),
        False,
        {"pipeline": "1" * 64, "device": "2" * 64, "qc": "3" * 64},
        {"pipeline": artifact, "device": artifact, "qc": artifact},
    )


class FakeCrop:
    def __init__(self, events: list[str]):
        self.events = events

    def crop(self, mesh, **kwargs):
        self.events.append("crop")
        mask = np.ones(len(mesh.vertices), dtype=bool)
        return CropResult(mesh, np.ones(len(mesh.vertices)), mask, mask, {"retained_vertex_fraction": 1.0})


class FakeLandmark:
    def __init__(self, events: list[str]):
        self.events = events

    def predict(self, mesh, **kwargs):
        self.events.append("landmark")
        points = mesh.vertices[np.asarray([0, 1, 2, 3, 1, 2, 3, 0, 1])]
        return LandmarkResult(
            LANDMARK_NAMES,
            points,
            points,
            np.zeros(9, dtype=np.int64),
            np.zeros(9),
            np.ones(9),
            np.zeros(9),
        )


class FakeRegistration:
    def __init__(self, events: list[str]):
        self.events = events
        self.target = None

    def map_mesh_diagnostic(self, target, template):
        self.events.append("registration")
        self.target = target
        return MeshMonkResult(
            template,
            template.vertices,
            np.ones(len(template.vertices)),
            template.vertices,
            1.0,
            ({"phase": "rigid", "iteration": 0, "residual_mean": 0.0},),
        )


def _template_landmarks() -> np.ndarray:
    source = _mesh().vertices[np.asarray([0, 1, 2, 3, 1, 2, 3, 0, 1])]
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return source @ (2.0 * rotation).T + np.asarray([5.0, -3.0, 1.0])


def test_process_sample_runs_locked_stage_order_and_uses_all_nine_landmarks(tmp_path: Path):
    events: list[str] = []
    registration = FakeRegistration(events)
    result = process_sample(
        raw_mesh=_mesh(),
        geometry_hash="a" * 64,
        config=_config(tmp_path),
        template_mesh=_mesh(),
        template_landmarks=_template_landmarks(),
        template_points=_mesh().vertices,
        refindex=None,
        deps=PipelineDependencies(FakeCrop(events), FakeLandmark(events), registration),
        qc_level="standard",
    )

    assert events == ["crop", "landmark", "registration"]
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        registration.target.vertices,
        _mesh().vertices @ (2.0 * rotation).T + np.asarray([5.0, -3.0, 1.0]),
    )
    assert result.status == "passed"
    assert result.final_mesh is not None
    assert result.metadata["landmarks"]["bijian"] == [0.0, 0.0, 0.0]
    assert result.metadata["coarse_alignment"]["method"] == "similarity_gpa_9_snapped_landmarks"
    assert result.metadata["coarse_alignment"]["landmark_names"] == list(LANDMARK_NAMES)
    assert "landmark_aligned" in result.artifacts
    assert "nose_aligned" not in result.artifacts


def test_warning_keeps_final_and_hard_topology_failure_suppresses_it(tmp_path: Path):
    qc = QCConfig(
        1,
        "qc",
        "1",
        {"crop.retention": "warning"},
        (
            QCRule(
                "retention",
                "crop.retention.vertex_fraction",
                "lt",
                1.1,
                "ratio",
                "warning",
                "standard",
            ),
        ),
    )
    config = _config(tmp_path, qc)
    deps = PipelineDependencies(FakeCrop([]), FakeLandmark([]), FakeRegistration([]))

    warning = process_sample(
        _mesh(),
        "a" * 64,
        config,
        _mesh(),
        _template_landmarks(),
        _mesh().vertices,
        None,
        deps,
        "standard",
    )
    failed = process_sample(
        _mesh(),
        "a" * 64,
        replace(config, geometry=replace(config.geometry, expected_template_vertices=5)),
        _mesh(),
        _template_landmarks(),
        _mesh().vertices,
        None,
        deps,
        "standard",
    )

    assert warning.status == "warning" and warning.final_mesh is not None
    assert failed.status == "failed" and failed.final_mesh is None


def test_enhanced_schedule_and_rotation_are_exact():
    schedule = enhanced_schedule()
    assert [item.label for item in schedule] == ["D1", "D2", "D3", "D4"]
    assert schedule[0].angles_deg == (0.0, 0.0, 0.0)
    assert schedule[2].angles_deg == (5.0, -5.0, 5.0)
    np.testing.assert_allclose(
        rotation_matrix_xyz((5.0, -5.0, 5.0)).T
        @ rotation_matrix_xyz((5.0, -5.0, 5.0)),
        np.eye(3),
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ("misordered", "nonfinite", "off_mesh", "degenerate"))
def test_invalid_landmark_contract_fails_without_registration_or_nose_fallback(
    tmp_path: Path, mode: str
):
    events: list[str] = []

    class InvalidLandmark(FakeLandmark):
        def predict(self, mesh, **kwargs):
            result = super().predict(mesh, **kwargs)
            names = result.names
            snapped = result.snapped.copy()
            if mode == "misordered":
                names = (names[1], names[0], *names[2:])
            elif mode == "nonfinite":
                snapped[2, 1] = np.nan
            elif mode == "off_mesh":
                snapped[8] = [99.0, 98.0, 97.0]
            else:
                snapped[:] = mesh.vertices[0]
            return replace(result, names=names, snapped=snapped)

    with pytest.raises(SampleFailure):
        process_sample(
            _mesh(),
            "a" * 64,
            _config(tmp_path),
            _mesh(),
            _template_landmarks(),
            _mesh().vertices,
            None,
            PipelineDependencies(
                FakeCrop(events), InvalidLandmark(events), FakeRegistration(events)
            ),
            "standard",
        )

    assert "registration" not in events


def test_adapter_specific_diagnostics_are_optional_and_recorded_as_unavailable(
    tmp_path: Path,
):
    points = _mesh().vertices[np.asarray([0, 1, 2, 3, 1, 2, 3, 0, 1])]

    class CoreCrop:
        def crop(self, mesh, *, seed):
            return CropAdapterResult(mesh=mesh)

    class CoreLandmark:
        def predict(self, mesh, *, seed):
            return LandmarkAdapterResult(names=LANDMARK_NAMES, snapped=points)

    result = process_sample(
        _mesh(),
        "a" * 64,
        _config(tmp_path),
        _mesh(),
        _template_landmarks(),
        _mesh().vertices,
        None,
        PipelineDependencies(CoreCrop(), CoreLandmark(), FakeRegistration([])),
        "standard",
    )

    assert result.status == "passed"
    assert (
        result.metadata["measurements"]["crop.probability.mean"]["compute_status"]
        == "not_available"
    )
    assert (
        result.metadata["measurements"]["landmark.heatmap.peak_min"]["compute_status"]
        == "not_available"
    )
    assert "landmark_continuous" not in result.array_artifacts
    assert "landmark_snapped" in result.array_artifacts
